"""Search API 密钥统一管理模块。

所有 Web Search API 的密钥统一存放在配置文件中（默认
``~/.auraderma/search_config.json``），按提供方名称分字段存储。
同时也支持通过同名环境变量覆盖（优先级更高）。

配置示例 (``~/.auraderma/search_config.json``):

.. code:: json

    {
      "metasoApiKey": "sk-xxx",
      "baiduApiKey": "bce-xxx",
      "tavilyApiKey": "tvly-xxx",
      "perplexityApiKey": "pplx-xxx",
      "exaApiKey": "exa-xxx",
      "braveApiKey": "brave-xxx",
      "ollamaApiKey": "ollama-xxx",
      "searxngEndpoint": "http://localhost:8080"
    }

Env 变量命名规则：
  - ``AURADERMA_METASO_API_KEY`` → metasoApiKey
  - ``AURADERMA_BAIDU_API_KEY``  → baiduApiKey
  -  依此类推
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 配置文件路径
# ---------------------------------------------------------------------------

CONFIG_DIR_NAME = ".auraderma"
CONFIG_FILE_NAME = "search_config.json"


def _default_config_path() -> Path:
    return Path.home() / CONFIG_DIR_NAME / CONFIG_FILE_NAME


# ---------------------------------------------------------------------------
# 密钥读取器（env > config）
# ---------------------------------------------------------------------------

_ENV_PREFIX = "AURADERMA_"

# 字段名 → (env 变量名(不含前缀), 默认值)
_API_KEY_FIELDS: dict[str, tuple[str, str | None]] = {
    "metasoApiKey": ("METASO_API_KEY", None),
    "baiduApiKey": ("BAIDU_API_KEY", None),
    "tavilyApiKey": ("TAVILY_API_KEY", None),
    "perplexityApiKey": ("PERPLEXITY_API_KEY", None),
    "exaApiKey": ("EXA_API_KEY", None),
    "braveApiKey": ("BRAVE_API_KEY", None),
    "ollamaApiKey": ("OLLAMA_API_KEY", None),
    "searxngEndpoint": ("SEARXNG_ENDPOINT", "http://localhost:8080"),
    "webSearchEngine": ("WEB_SEARCH_PROVIDER", "bing"),
}


def _load_search_config() -> dict:
    """加载 search_config.json，失败返回空 dict。"""
    path = _default_config_path()
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, PermissionError):
        return {}


def ensure_search_config() -> Path:
    """确保配置文件存在，不存在则自动创建空的 JSON 文件。

    Returns:
        配置文件的 ``Path``。
    """
    config_path = _default_config_path()
    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("{}\n", encoding="utf-8")
    return config_path


# ---------------------------------------------------------------------------
# 提供方名称 → 配置字段名映射
# ---------------------------------------------------------------------------

# 需要 API Key 的提供方映射
PROVIDER_TO_FIELD: dict[str, str] = {
    "metaso": "metasoApiKey",
    "baidu": "baiduApiKey",
    "tavily": "tavilyApiKey",
    "perplexity": "perplexityApiKey",
    "exa": "exaApiKey",
    "brave": "braveApiKey",
    "ollama": "ollamaApiKey",
}

# 需要 endpoint URL 的提供方映射
PROVIDER_TO_ENDPOINT_FIELD: dict[str, str] = {
    "searxng": "searxngEndpoint",
}

# 无需任何配置的提供方
PROVIDERS_NO_CONFIG: set[str] = {"bing", "bing-intl"}

# 所有支持的提供方
ALL_SUPPORTED_PROVIDERS: set[str] = (
    set(PROVIDER_TO_FIELD)
    | set(PROVIDER_TO_ENDPOINT_FIELD)
    | PROVIDERS_NO_CONFIG
)


def _get_field(field_name: str) -> str | None:
    """env > config > default，返回字段值或 None。"""
    if field_name not in _API_KEY_FIELDS:
        return None
    env_name, default = _API_KEY_FIELDS[field_name]
    # 1. 环境变量
    env_val = os.getenv(f"{_ENV_PREFIX}{env_name}")
    if env_val is not None and env_val.strip():
        return env_val.strip()
    # 2. 配置文件
    config = _load_search_config()
    cfg_val = config.get(field_name)
    if cfg_val is not None and isinstance(cfg_val, str) and cfg_val.strip():
        return cfg_val.strip()
    # 3. 内置默认值
    return default


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------

class SearchApiKeys:
    """所有 Search API 密钥的统一访问入口。

    每次访问都重新读取 config 文件（不做内存缓存），
    以便运行时修改文件后立刻生效。
    """

    @staticmethod
    def metaso() -> str | None:
        return _get_field("metasoApiKey")

    @staticmethod
    def baidu() -> str | None:
        return _get_field("baiduApiKey")

    @staticmethod
    def tavily() -> str | None:
        return _get_field("tavilyApiKey")

    @staticmethod
    def perplexity() -> str | None:
        return _get_field("perplexityApiKey")

    @staticmethod
    def exa() -> str | None:
        return _get_field("exaApiKey")

    @staticmethod
    def brave() -> str | None:
        return _get_field("braveApiKey")

    @staticmethod
    def ollama() -> str | None:
        return _get_field("ollamaApiKey")

    @staticmethod
    def searxng_endpoint() -> str | None:
        return _get_field("searxngEndpoint")

    @staticmethod
    def default_engine() -> str:
        val = _get_field("webSearchEngine")
        return val if val else "bing"


def provider_config_field(provider: str) -> str | None:
    """根据提供方名称返回对应的配置字段名。

    Args:
        provider: 提供方名称，如 ``"tavily"``、``"searxng"``。

    Returns:
        配置字段名。若该提供方不需要配置则返回 ``"<no-config>"``，
        若提供方不被支持则返回 ``None``。
    """
    provider = provider.lower().strip()
    if provider in PROVIDERS_NO_CONFIG:
        return "<no-config>"
    if provider in PROVIDER_TO_FIELD:
        return PROVIDER_TO_FIELD[provider]
    if provider in PROVIDER_TO_ENDPOINT_FIELD:
        return PROVIDER_TO_ENDPOINT_FIELD[provider]
    return None


def save_api_key_for_provider(provider: str, value: str) -> str | None:
    """为指定提供方保存 API Key 到配置文件。

    Args:
        provider: 提供方名称。
        value: API Key 值。

    Returns:
        成功返回 ``None``，失败返回错误消息字符串。
    """
    field = provider_config_field(provider)
    if field is None:
        return f"不支持的搜索提供方：'{provider}'"
    if field == "<no-config>":
        return f"'{provider}' 不需要 API Key，可直接使用"
    save_search_config(**{field: value})
    return None


def save_search_config(**overrides: str) -> None:
    """保存/更新 search_config.json 中的指定字段。

    Args:
        **overrides: 字段名 → 值。值为空字符串或 None 表示删除该字段。
    """
    config_dir = Path.home() / CONFIG_DIR_NAME
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / CONFIG_FILE_NAME

    config = _load_search_config()
    changed = False
    for key, value in overrides.items():
        if key not in _API_KEY_FIELDS:
            continue
        if value is None or (isinstance(value, str) and not value.strip()):
            if key in config:
                del config[key]
                changed = True
        else:
            stripped = value.strip()
            if config.get(key) != stripped:
                config[key] = stripped
                changed = True

    if changed:
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
