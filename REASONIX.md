# 项目：护肤Agent

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
