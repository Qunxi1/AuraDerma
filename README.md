# AuraDerma

<h3 align="center">AI 护肤助手 — 基于 RAG 与多技能协同的终端 Agent</h3>
<p align="center">DeepSeek 驱动 · 向量检索引擎 · 10 种 Web 搜索后端 · 跨平台 CLI</p>

<br/>

## 特性

- **意图理解 + 工作流规划**。LLM 自动分析用户问题意图（单品推荐 / 多品类 / 护肤体系 / 纯聊天），规划需要执行的流程，再给出回答。
- **向量检索引擎**。基于 Qdrant 的本地向量数据库，使用 `BAAI/bge-small-zh-v1.5` 中文 embedding 模型，对产品库 / 用户记忆 / 文档做语义检索。
- **多技能协同**。三个内置技能：`web_search`（10 种搜索引擎后端，默认 Bing HTML 爬取免 Key）、`weather`（wttr.in 气候自适应建议）、`file_read`（PDF / DOCX / DOC / TXT 解析）。
- **气候自适应推荐**。根据用户所在城市的温度 + 湿度，自动调整推荐产品的质地（轻薄水乳 vs 滋润面霜）。
- **长期记忆**。自动从对话中提取用户肤质、过敏原、偏好等 profile 信息，跨会话持久化。
- **跨平台 CLI**。Windows 使用 msvcrt 增强终端（下拉补全、历史搜索），Linux / macOS 使用 readline 基础输入。

<br/>

## 安装

### 前置依赖

- Python **3.11+**
- Docker Desktop（运行 Qdrant 向量数据库）
- Git

```bash
# 1. 克隆项目
git clone https://github.com/Qunxi1/AuraDerma.git
cd AuraDerma

# 2. 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux / macOS

# 3. 安装项目
pip install -e .
```

首次启动时会自动从 HuggingFace 下载 embedding 模型（约 100MB），保存在 `data/models/` 目录下，后续启动直接加载本地缓存。

<br/>

## 快速开始

### 1. 配置环境变量

```powershell
# Windows PowerShell
Copy-Item .env.example .env
```

编辑 `.env` 文件，填入你的 DeepSeek API Key：

```ini
AURADERMA_MODEL_API_KEY=sk-your-key-here
```

> 其他搜索引擎的 API Key 在运行时通过 `/search-config` 命令配置，不需要写在 `.env` 里。

### 2. 启动 Qdrant

```bash
docker compose up -d qdrant
```

Qdrant 默认绑定在 `localhost:6388`（避开了 Windows 端口保留范围 6283-6382）。

### 3. 导入产品数据

```bash
# 从结构化文本文件导入护肤品到向量库
auraderma ingest-products path/to/products.txt
```

### 4. 启动对话

```bash
ad                  # 一键启动（自动检查并启动 Qdrant）
# 或
auraderma chat      # 手动启动 Qdrant 后使用
```

```
╭────────────────────────────────────────────╮
│  AuraDerma  -  model: deepseek-v4-flash    │
│  /help for commands - Tab for suggestions  │
╰────────────────────────────────────────────╯
AuraDerma [deepseek-v4-flash] > 推荐一款适合油皮的爽肤水
```

<br/>

## 配置

AuraDerma 使用 `.env` 文件管理所有配置，运行时会自动载入 `.env.local` 覆盖（适合机器特定的配置，不提交到 Git）。

```ini
# ── 模型
AURADERMA_MODEL_BASE=https://api.deepseek.com
AURADERMA_MODEL_API_KEY=sk-...
AURADERMA_DEFAULT_MODEL=deepseek-v4-flash

# ── 向量数据库 (Qdrant)
AURADERMA_QDRANT_URL=http://localhost:6388
AURADERMA_QDRANT_API_KEY=
AURADERMA_QDRANT_PRODUCTS=AuraDerma_products
AURADERMA_QDRANT_MEMORY=AuraDerma_memory
AURADERMA_QDRANT_DOCS=AuraDerma_docs

# ── 技能目录
AURADERMA_SKILLS_DIR=./skills

# ── 数据目录（记忆、日志、模型缓存）
AURADERMA_DATA_DIR=./data

# ── Web 搜索
AURADERMA_WEB_SEARCH_ENABLED=true
AURADERMA_WEB_SEARCH_PROVIDER=bing        # bing(默认) | bing-intl | searxng | metaso
                                           # baidu | tavily | perplexity | exa | brave | ollama
```

搜索提供方运行时切换：

```
AuraDerma > /search-engine tavily
AuraDerma > /search-config tavily tvly-xxxxx
```

<br/>

## 交互命令

在对话中使用 `/` 前缀的斜杠命令：

| 命令 | 说明 |
|------|------|
| `/help` | 显示所有命令 |
| `/model [name]` | 查看或切换模型 |
| `/models` | 列出可用模型 |
| `/search-engine [name]` | 查看或切换搜索引擎 |
| `/search-config <name> <key>` | 配置 API Key |
| `/memory` | 查看用户记忆摘要 |
| `/skills` | 列出可用技能 |
| `/save <note>` | 手动保存一条记忆 |
| `/quit` | 退出对话 |

Tab 键自动补全命令，↑↓ 键浏览历史。

<br/>

## 可用模型

```
deepseek-v4-flash
deepseek-v4-pro
deepseek-reasoner
qwen3.6-plus
kimi-k2.5
glm-5
custom-openai-compatible
```

<br/>

## 技能系统

技能代码全部放在 `skills/` 目录下，与 `src/` 核心代码解耦。`src/skill_manager.py` 只负责加载和分发。

```
skills/
├── web_search/
│   ├── client.py    ← 10 种搜索引擎的核心实现（769 行）
│   ├── skill.py     ← 适配器，包装成工具接口（24 行）
│   ├── skill.md     ← LLM 路由时使用的技能描述
│   └── summary.md   ← 技能摘要
├── weather/
│   ├── skill.py     ← wttr.in 天气查询（完全自包含）
│   ├── skill.md
│   └── summary.md
├── file_read/
│   ├── skill.py     ← PDF / DOCX / DOC / TXT 解析（完全自包含）
│   ├── skill.md
│   └── summary.md
└── __init__.py
```

添加新技能：在 `skills/` 下创建子目录，放入 `skill.py` + `skill.md` + `summary.md` 即可。

<br/>

## 项目结构

```
src/
├── core/                        # 核心基础设施
│   ├── logger.py                # 结构化日志（控制台 + 文件轮转）
│   └── json_parser.py           # 健壮 JSON 解析（自动修复 + 错误追溯）
├── services/                    # 领域服务
│   ├── intent_service.py        # 意图分类
│   ├── workflow_service.py      # 工作流规划
│   ├── regimen_service.py       # 护肤体系规划
│   ├── retrieval_service.py     # 产品 / 记忆 / 文档检索
│   ├── weather_service.py       # 天气查询
│   └── answer_service.py        # 回答生成
├── console/                     # CLI 组件
│   ├── commands.py              # 命令定义与处理器
│   └── readline.py              # 跨平台交互终端
├── agent.py                     # 轻量编排器（~210 行）
├── cli.py                       # Click 入口
├── skill_manager.py             # 技能加载与分发
├── memory.py                    # 记忆策略与文件存储
├── retrieval.py                 # Qdrant 检索器
├── llm.py                       # LLM 客户端 + 本地 Embedding
├── prompts.py                   # 所有 Prompt 模板
└── config.py                    # 环境变量配置
```

<br/>

## 处理流程

```
用户输入
  ↓
① 意图分类   — IntentService
    单品类 / 多品类 / 护肤体系 / 纯聊天
  ↓
② 工作流规划 — WorkflowService
    决定执行哪些流程（产品搜索、记忆查询、天气、Web 搜索...）
  ↓
③ 检索阶段
    向量检索（产品 / 记忆 / 文档）+ 记忆路由 + 技能路由
  ↓
④ 增强阶段
    网页搜索（内部无结果时） + 天气数据（气候自适应）
  ↓
⑤ 回答生成  — AnswerService
    根据意图模式（单品 / 多品类 / 护肤体系 / 纯聊天）选择模板生成回答
```

<br/>

## 日志与调试

设置环境变量即可开启 DEBUG 日志：

```powershell
$env:AURADERMA_DEBUG = "1"
ad
```

日志文件位置：`data/logs/auraderma.json.log`（自动轮转，最多 5×10MB）。

JSON 解析失败时会将完整的原始 LLM 返回文本记录到日志，方便排查问题。

<br/>

## License

MIT
