# 项目：护肤Agent

AuraDerma 是一个以 ReAct 思考→行动→观察循环编排、基于 Qdrant 向量检索（RAG）与插件化技能协同、由 DeepSeek 等 OpenAI 兼容模型驱动的终端护肤顾问 Agent。

## 目录结构

```
AuraDerma/
├── src/
│   ├── cli.py / ad_entry.py        # 双入口：auraderma（Click）/ ad（自动拉起 Qdrant）
│   ├── agent_react.py              # ReAct 思考→行动→观察循环
│   ├── prompts_react.py            # ReAct 提示词模板
│   ├── prompts.py                  # 领域提示词模板
│   ├── llm.py                      # LLM 客户端 + 本地 Embedding
│   ├── retrieval.py                # Qdrant 检索器
│   ├── memory.py                   # 记忆策略与文件存储
│   ├── skill_manager.py            # 技能加载与分发
│   ├── skills_registry.py          # 技能注册表
│   ├── config.py                   # .env 配置加载
│   ├── ingest.py / ingest_products.py  # 数据导入
│   ├── reporter.py / schema.py     # 输出渲染 / 数据模型
│   ├── core/                       # logger.py、json_parser.py
│   ├── services/                   # intent / workflow / regimen / retrieval / weather / answer
│   └── console/                    # commands.py、readline.py
├── skills/                         # 插件技能：web_search / weather / file_read
├── data/                           # 运行时产物（不入库）：models/ memory/ logs/ raw/ qdrant/
├── docker-compose.yml              # Qdrant 服务定义
├── pyproject.toml                  # 依赖、入口、ruff 配置
└── REASONIX.md                     # 本文件
```

## 技术栈

### 语言与工程
- Python ≥ 3.11，`pyproject.toml` 声明依赖与入口（setuptools，`src/` 布局）
- 双 CLI 入口：`auraderma`（Click）与 `ad`（一键启动，自动拉起 Qdrant 容器）
- 代码规范：ruff（line-length=100）

### LLM 与 Embedding
- `openai` SDK 走 OpenAI 兼容协议访问 DeepSeek 端点（`.env` 配置 base/key/默认模型，可切换 qwen、kimi、glm 等兼容模型）
- 本地中文 Embedding：`sentence-transformers` + `BAAI/bge-small-zh-v1.5`，首次从 HuggingFace 下载，缓存于 `data/models/`

### 向量检索（RAG 底座）
- Qdrant 向量库，Docker 部署（`qdrant/qdrant:v1.11.5`，映射宿主机 `localhost:6388`）
- `qdrant-client` 三集合：产品库 / 用户记忆 / 文档
- 检索链路：向量检索 + 记忆路由 + 技能路由 + 网页兜底

### Agent 编排
- 核心：ReAct 思考→行动→观察循环（`agent_react.py` + `prompts_react.py`）
- 领域服务：意图分类、工作流规划、检索、护肤体系规划、天气、回答生成（`src/services/`）
- 健壮 JSON 解析（`core/json_parser.py` 自动修复 LLM 输出）

### 技能系统（插件化）
- `skills/` 目录每技能含 `skill.py` + `skill.md` + `summary.md`，`skill_manager.py` 加载分发
- `web_search`：10 种搜索引擎后端（`requests` + BeautifulSoup，默认 Bing HTML 爬取免 Key）
- `weather`：wttr.in 查询，气候自适应质地推荐
- `file_read`：PyMuPDF(PDF)、python-docx + olefile + pywin32(DOC/DOCX)、TXT

### 数据、记忆与配置
- 长期记忆：自动抽取用户肤质/过敏原/偏好，文件持久化 `data/memory` + Qdrant 记忆集合
- 产品数据：`data/raw/*.jsonl` 经 ingest 命令导入 Qdrant
- 配置：`python-dotenv` 加载 `.env`，支持 `.env.local` 覆盖
- 结构化日志：JSON 文件轮转（`data/logs/`，按模块分文件）

## 代码风格

- 遵循 ruff（`line-length=100`），提交前运行 `ruff check .`
- 文件首行 `from __future__ import annotations`，函数标注完整类型
- 注释与文档字符串统一使用中文，命名用英文；避免无意义的缩写与魔数

```python
# ✅ 推荐：类型注解 + 中文文档字符串 + 语义化命名
from __future__ import annotations


def build_retrieval_query(skin_type: str, concerns: list[str], limit: int = 5) -> str:
    """根据用户肤质与诉求拼接产品检索查询语句。"""
    keywords = [skin_type, *concerns]
    return " ".join(keywords[:limit])
```

```python
# ❌ 避免：无类型注解、无文档、命名晦涩、魔数散落
def q(p, l=5):
    return p["skin"] + "护肤"
```

## 安全边界

### 🚫 禁止（Never）
- 在代码或文档中硬编码任何 API Key、token 或其他密钥
- 提交 `.env`、`.env.local`、`*.pem`、`*.key`、`credentials*.json`（均已在 `.gitignore` 排除）
- 在日志中输出用户个人信息（PII）
- 绕过 PreToolUse / git-push-check hook 的拦截

### ⚠️ 先确认（Ask）
- 更换默认模型或 LLM 端点（影响成本与行为）
- 修改 Qdrant collection 名称或端口映射（6388）
- 改动 ReAct 或领域提示词模板
- 新增需要外部网络的搜索引擎后端

### ✅ 始终（Always）
- 提交前运行 `ruff check .`（line-length=100）
- push 前执行 `git diff --cached --name-only` 复核暂存内容
- 代码注释与文档使用中文
- 新增技能遵循 `skill.py` + `skill.md` + `summary.md` 三件套

