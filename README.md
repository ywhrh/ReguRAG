# ReguRAG

> 面向金融监管合规场景的法规问答助手——检索增强生成（RAG），每条答案均可追溯至原文，并内置防幻觉机制。

**English summary:** ReguRAG is a Retrieval-Augmented Generation (RAG) system for financial regulatory Q&A. It retrieves relevant clauses from a local regulatory document library, cites the exact source passages in every answer, and refuses to generate answers when retrieved content falls below a relevance threshold — preventing hallucination in a domain where errors carry real legal risk.

---

## 目录

- [项目简介](#项目简介)
- [功能特性](#功能特性)
- [技术栈](#技术栈)
- [架构与工作流程](#架构与工作流程)
- [快速开始](#快速开始)
- [网页界面使用方法](#网页界面使用方法)
- [项目结构](#项目结构)
- [数据来源](#数据来源)
- [配置说明](#配置说明)
- [设计思路](#设计思路)
- [Roadmap](#roadmap)

---

## 项目简介

ReguRAG 是一个专为**金融监管/合规场景**设计的法规问答助手。它将本地法规 PDF/TXT 文档向量化存储，用户提问后自动检索最相关的法规片段，拼入结构化 Prompt 后交由 Claude 生成答案，并在回答中附带**可核对的法规原文出处**。

**核心出发点**：在合规场景下，一个答错的条款可能带来监管处罚甚至法律责任。因此本项目的优先级不是"尽量回答"，而是"**答对比答快更重要，不确定时明确说不知道**"。

---

## 功能特性

- **答案完全可溯源**：每条回答都附带所依据的法规片段、来源文件名及相关度分数，用户可直接核对原文。
- **防幻觉兜底机制**：检索到的内容相关度低于阈值时，系统拒绝调用 Claude，直接返回固定提示语，彻底避免模型"瞎编"。
- **数据/指令分离**：Prompt 中用 XML 标签严格区分法规原文区（`<regulations>`）和行为约束区（`<instructions>`），防止数据内容干扰模型行为。
- **零成本 Embedding**：使用本地开源多语言模型（sentence-transformers），向量化阶段完全不消耗 API，支持中英文法规混合场景。
- **参数集中配置**：切分参数、相关度阈值、模型名称等均在 `config.py` 一处管理，便于调优实验。
- **支持 PDF 和 TXT**：直接加载监管机构发布的 PDF 格式法规文件，无需手动转换。

---

## 技术栈

| 层级 | 技术选型 | 说明 |
|------|----------|------|
| 文档加载 | LangChain (`TextLoader`, `PyPDFLoader`) | 支持 .txt / .pdf |
| 文本切分 | `RecursiveCharacterTextSplitter` | 优先在段落/句子边界切分 |
| Embedding | `paraphrase-multilingual-MiniLM-L12-v2` | 本地运行，支持中英文，无 API 费用 |
| 向量库 | Chroma（本地持久化） | 轻量，无需外部服务，适合本地开发 |
| 答案生成 | Llama 3.3 70B via Groq API（免费） | 仅用于最终回答生成 |
| 编排框架 | LangChain | 统一接口，便于后续替换组件 |
| 网页界面 | Streamlit | 本地浏览器调试界面，单页极简 |
| 运行环境 | Python 3.12 | |

---

## 架构与工作流程

```
┌─────────────────────────────────────────────────────────────────┐
│                         建库阶段（一次性）                         │
│                                                                   │
│  data/*.pdf / *.txt                                               │
│        │                                                          │
│        ▼                                                          │
│  [文档加载]  TextLoader / PyPDFLoader                             │
│        │                                                          │
│        ▼                                                          │
│  [文本切分]  RecursiveCharacterTextSplitter                       │
│             chunk_size=800, overlap=100                           │
│        │                                                          │
│        ▼                                                          │
│  [向量化]   paraphrase-multilingual-MiniLM-L12-v2（本地）         │
│        │                                                          │
│        ▼                                                          │
│  [写入 Chroma]  持久化到 chroma_db/                               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       问答阶段（每次提问）                          │
│                                                                   │
│  用户输入问题                                                       │
│        │                                                          │
│        ▼                                                          │
│  [向量检索]  从 Chroma 检索 Top-K 相关片段（返回片段 + 相关度分数）  │
│        │                                                          │
│        ▼                                                          │
│  [阈值判断]  最高分 < 0.3 ？                                        │
│        │         │                                                │
│        │      YES ▼                                               │
│        │    返回兜底消息（不调用 Claude，彻底防幻觉）               │
│        │                                                          │
│      NO ▼                                                         │
│  [Prompt 拼接]  法规原文用 <regulations> 包裹，                    │
│                行为约束用 <instructions> 包裹                      │
│        │                                                          │
│        ▼                                                          │
│  [Claude API]  生成答案（含片段引用编号）                           │
│        │                                                          │
│        ▼                                                          │
│  输出答案 + 参考来源（文件名、相关度、内容预览）                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/your-username/ReguRAG.git
cd ReguRAG

# 创建并激活虚拟环境（Python 3.10+ 均可）
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

> **注意**：`sentence-transformers` 首次运行会从 HuggingFace 下载约 400MB 的模型文件并缓存到本地，之后无需重复下载。

### 3. 配置 API Key

```bash
# 复制模板
cp .env.example .env

# 编辑 .env，填入你的 Groq API Key
# 免费申请地址：https://console.groq.com/keys
```

`.env` 文件内容如下：

```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx
```

### 4. 准备法规文档

项目 `data/` 目录已附带两个示例文件供快速验证：

```
data/
├── china_banking_regulation_sample.txt   # 中文示例：商业银行资本与流动性监管要点
└── basel_framework_sample.txt            # 英文示例：Basel III 框架核心规定
```

如需加载真实法规文档，将 `.pdf` 或 `.txt` 文件放入 `data/` 目录后重新运行建库命令即可。

### 5. 建立向量库

```bash
python main.py build
```

预期输出：

```
[1/3] 加载文档（来源目录：./data/）
  [OK] 已加载 txt：basel_framework_sample.txt（1 段）
  [OK] 已加载 txt：china_banking_regulation_sample.txt（1 段）

[2/3] 切分文本块
文本切分完成：2 段 → 23 个 chunk

[3/3] 向量化并写入 Chroma
...
向量库构建完成：共 23 条记录，已持久化到 ./chroma_db/

建库完成！运行以下命令开始提问：
  python main.py ask
```

### 6. 开始提问

```bash
python main.py ask
```

**示例问答（中文）**：

```
请输入您的问题：商业银行的核心一级资本充足率最低要求是多少？

【答案】
根据【片段1】，商业银行核心一级资本充足率不得低于 5%。在此基础上，
还须计提 2.5% 的储备资本，系统重要性银行须额外计提附加资本。

【参考来源】
  ▸ 片段 1（相关度 0.821）
    文件：data/china_banking_regulation_sample.txt
    内容：根据《商业银行资本管理办法》相关规定，商业银行应当持续满足以下资本充足率监管要求：
    1. 核心一级资本充足率不得低于 5%...
```

**示例问答（英文）**：

```
请输入您的问题：What is the minimum LCR requirement under Basel III?

【答案】
According to [Fragment 1], the minimum Liquidity Coverage Ratio (LCR) under Basel III
is 100%. Banks must hold sufficient high-quality liquid assets (HQLA) to cover net
cash outflows over a 30-day stress period...
```

**示例兜底（无关问题）**：

```
请输入您的问题：今天股市行情怎么样？

最高相关度：0.041（阈值：0.3）
相关度低于阈值，触发兜底逻辑，不调用 Claude

【答案】
未在现有法规库中找到与您问题相关的依据。
建议您咨询专业合规人员，或直接查阅以下官方来源：...
```

---

## 网页界面使用方法

除命令行外，项目提供了一个基于 Streamlit 的浏览器界面，方便本地调试时不用每次敲命令。

### 前提

先完成"快速开始"中的**建库步骤**（`python main.py build`），向量库建好后才能启动界面。

### 启动命令

```bash
streamlit run app.py
```

### 访问地址

启动成功后，终端会显示：

```
Local URL: http://localhost:8501
```

在浏览器打开 **http://localhost:8501** 即可使用。

### 界面功能

- **输入框**：输入问题（中英文均可）
- **提问按钮**：触发检索和 LLM 调用
- **答案区**：显示 LLM 生成的回答
- **引用来源区**：展示每条答案所依据的法规原文片段、来源文件、相关度分数，可点击展开核对
- **兜底提示**：相关度不足时，显示橙色警告，说明 LLM 未被调用

---

## 项目结构

```
ReguRAG/
├── main.py                    # CLI 入口：build（建库）/ ask（问答）
├── app.py                     # Streamlit 网页界面入口（streamlit run app.py）
├── config.py                  # 集中配置：模型、阈值、路径、切分参数等
├── requirements.txt           # Python 依赖
├── .env.example               # API Key 配置模板
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── document_loader.py     # 文档加载（txt/pdf）+ 文本切分
│   ├── vector_store.py        # Embedding 模型 + Chroma 建库/加载/检索
│   └── qa_chain.py            # Prompt 构建 + 阈值判断 + Claude 调用
│
├── data/                      # 放置法规文档（.txt / .pdf）
│   ├── china_banking_regulation_sample.txt
│   └── basel_framework_sample.txt
│
└── chroma_db/                 # 自动生成，Chroma 向量库持久化目录（不提交 git）
```

---

## 数据来源

示例文件中的内容为简化示意性文本，真实使用时请从以下官方渠道获取正式法规文档：

| 机构 | 说明 | 获取地址 |
|------|------|----------|
| 国家金融监督管理总局（NFRA） | 银行、保险监管规则 | https://www.nfra.gov.cn |
| 中国人民银行（PBC） | 货币政策、支付清算相关法规 | https://www.pbc.gov.cn |
| 中国证券监督管理委员会（CSRC） | 证券、基金监管规则 | https://www.csrc.gov.cn |
| 巴塞尔委员会（BCBS） | Basel I/II/III 国际监管框架 | https://www.bis.org/bcbs/ |
| SEC EDGAR（美国） | 上市公司信息披露、联邦证券法 | https://www.sec.gov/edgar |
| EUR-Lex（欧盟） | CRR/CRD、MiFID II 等欧盟法规 | https://eur-lex.europa.eu |

**使用建议**：下载 PDF 后直接放入 `data/` 目录，运行 `python main.py build` 重建向量库即可，无需修改代码。

---

## 配置说明

所有关键参数集中在 `config.py`，调优时只需修改此文件：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | 用于答案生成的 Groq 模型 |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | 本地 embedding 模型 |
| `CHUNK_SIZE` | `800` | 每个文本块的最大字符数，越小检索越精准但上下文越少 |
| `CHUNK_OVERLAP` | `100` | 相邻块重叠字符数，减少关键信息被切断的概率 |
| `TOP_K` | `4` | 每次检索返回的候选片段数量 |
| `RELEVANCE_THRESHOLD` | `0.3` | 防幻觉阈值，低于此值拒绝生成（调高更严格，调低更宽松） |

---

## 设计思路

**为什么在金融合规场景必须防幻觉？**

普通问答场景里，LLM 给出一个不太准确的答案顶多带来困惑。但在合规场景下：
- 误引一条不存在的监管条款，可能导致企业按错误标准设计产品
- 错误解读资本充足率要求，可能使银行内部政策偏离监管红线
- 幻觉内容一旦被引用进合规报告，责任难以追究

因此本项目的防幻觉策略是**结构性的**，而非依赖模型自我约束：

1. **阈值拦截**：在调用 LLM 之前判断相关度，不相关的问题根本不进入生成环节。
2. **Prompt 约束**：即使进入生成环节，Prompt 中用 XML 标签将法规原文与指令严格隔离，并明确禁止模型使用原文以外的知识。
3. **来源展示**：在 UI 层将参考来源和相关度分数一起展示给用户，让用户自行判断，而不是只接受一个结论。

**为什么 Embedding 不用 Claude / OpenAI，答案生成才用 Claude？**

- 建库阶段可能处理数千个文本块，每块都调用付费 Embedding API 成本极高。
- `paraphrase-multilingual-MiniLM-L12-v2` 完全离线运行，中英文效果均可，首次下载约 400MB 后永久缓存，边际成本为零。
- 答案生成是最需要语言理解与推理能力的环节，Claude 的优势在此处体现。

---

## Roadmap

- [ ] **多轮对话支持**：记录对话历史，支持追问（"那它的计算方式呢？"）
- [ ] **评估体系**：引入 RAGAS 或自定义指标，量化检索召回率和答案忠实度
- [ ] **混合检索**：结合关键词检索（BM25）与向量检索，提升在专有名词（如"NSFR"）上的召回率
- [ ] **文档来源管理**：支持按监管机构、生效日期、法规类型筛选检索范围
- [ ] **增量建库**：新增文档时只更新新文档对应的向量，无需全量重建
- [x] **Web UI**：基于 Streamlit 提供浏览器交互界面（`app.py`）
- [ ] **更多格式**：支持 .docx（Word 格式监管指引）
- [ ] **答案置信度**：在答案旁展示结构化的置信度评分和不确定性说明

---

## 许可证

MIT License

---

*本项目为个人学习与作品集项目，示例数据仅供演示，不构成任何法律或合规建议。*
