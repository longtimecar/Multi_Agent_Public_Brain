# 跨 Agent 个人记忆：开源方案调研

调研日期：2026-10-31。范围为各项目当时公开的官方文档、README 和源码说明；除本项目既有测试记录外，没有安装竞品、读取个人历史、跑性能测试，也没有对第三方项目的质量作独立验证。页面通常描述当前 `main`，安装包版本可能不同。`LLM Wiki` 是一类思路，也有同名具体项目；`OpenMemory` 则至少有两个不同来源，不能混为一谈。

## 结论

这个目标有现实的开源参考，尤其 OpenViking 已将本地 Agent 插件、历史日志导入、按预算组装上下文和可视化控制台组合起来，值得优先做独立、同题对照评估；但「有功能」不等于「在本机真实可用、召回准确、符合只采用用户原话作为证据」。目前没有证据证明直接切换任一项目即可解决所有问题。

对 membrain 的主要启发是：独立管理原始证据和可重建索引；先较宽召回，再二次排序并控制实际注入；让概览查询按类型处理；把历史回填与持续捕获分开；展示每次从哪里召回、为什么选择、实际注入了什么。证据和冲突规则不能直接照搬其他项目。下文的建议是调研后的设计建议，并非已实现。

## 快速比较

| 方案 | 侧重点与存储 | 收录和共享 | 检索与上下文 | 对个人使用的判断 |
| --- | --- | --- | --- | --- |
| [OpenViking](https://docs.openviking.ai/zh/concepts/01-architecture) | 上下文数据库；文件式 URI + 内容后端 + 向量索引，目录摘要、概览与正文分层 | 插件覆盖多个本地 Agent，支持本地历史日志导入；也提供 MCP 和网关 | 向量检索、可选意图扩展和重排；不同客户端有自己的自动注入和预算 | 最接近用户想要的「现成完整链路」，但需验证角色证据、跨端行为、版本和实际效果 |
| [Mem0 自托管](https://docs.mem0.ai/open-source/quickstart) | 记忆管理库/服务，可配置模型及向量后端 | 应用传入消息并标注用户等标识，需自行完成各 Agent 接入和历史来源 | 按用户等过滤搜索；实际注入取决于调用应用 | 可作为后台组件，默认不是统一抓取全部原生聊天的系统 |
| [CaviraOSS LongMemory](https://github.com/CaviraOSS/LongMemory)（旧名 OpenMemory） | 本地优先记忆引擎；声明时序、证据、关系、检索与预算 | 提供服务、CLI、MCP、界面和编辑器工作流；具体历史适配范围需实验 | 声明多种召回、时间和来源解释；以版本文档为准 | 值得用相同数据测试，但不应把旧版五种扇区宣传视为现行全部实现 |
| [Microsoft LLM Wiki](https://github.com/microsoft/llmwiki) | VS Code 扩展 + 核心 MCP；原材料、生成的 Markdown wiki、规则三层 | 导入文件/目录及文件监听；MCP 供 Agent 使用 | 索引、标题、摘要与正文加权全文查询，逐页读取 | 适合作为可读的知识整理层；本身不等于自动捕获各 Agent 的个人记忆 |
| [AlexanderMakarov/llm-wiki](https://github.com/AlexanderMakarov/llm-wiki) | 会话转 Wiki 的本地工具和 MCP，可构建给人浏览的网站 | 核心适配器同步本地 Claude Code 和 Codex 会话；其他需单独启用适配器 | 分批综合、候选审核后写 Wiki；Agent 可通过 MCP 访问 | 历史资料导入有参考价值；审核是其产品流程，须评估与本项目自动写入目标的取舍 |
| [Basic Memory](https://github.com/basicmachines-co/basic-memory) | 以 Markdown 为真源，SQLite/其他后端为索引；有关系、语义检索与 MCP | 多客户端 MCP，Claude Code 另有插件和会话简报 | 可配置混合检索和可选重排，是否自动获取仍依赖宿主集成 | 文件可迁移性很接近本项目，但接入与写入规则要单独验证 |
| [Graphiti](https://github.com/getzep/graphiti) | 开源时序知识图谱框架；实体、关系、事件和来源 | 开发者自建数据库、写入服务、接入和管理功能 | 语义、关键词、图遍历；关系带有效时间 | 擅长复杂关系/时序；维护开销较大，自动失效策略也需适配本项目规则 |

## OpenViking：后端、分层、召回和注入

### 后端负责什么

其架构由 HTTP/SDK/CLI 访问服务层，服务层分别提供文件、检索、会话、资源导入、导出备份等；会话提交触发抽取和去重，文档导入触发解析和摘要；内容与向量索引分别存储。[架构](https://docs.openviking.ai/en/concepts/01-architecture)、[存储](https://docs.openviking.ai/zh/concepts/05-storage)。

`viking://...` 是统一访问标识和虚拟文件接口，不意味着所有内容只有虚拟结构。官方文档说明内容存储支持本地文件系统，也可使用其他后端；向量库保存 URI、向量、元数据及部分可读文本。因此「实际文件」和「索引数据库」是组合，不应简单说只有其一；备份和恢复需覆盖两侧。[存储](https://docs.openviking.ai/zh/concepts/05-storage)。

### 三个读取层次

| 层次 | 保存/返回什么 | 用途 |
| --- | --- | --- |
| L0 | 目录级 `.abstract.md` 摘要；默认正文目标 256 字符 | 快速定位和向量检索 |
| L1 | 目录级 `.overview.md` 概览；默认正文目标 4000 字符 | 理解目录、导航，可供重排 |
| L2 | 原始文件或解析后的详细内容 | 按需读取完整细节 |

这是目录层级附属摘要，不是每个普通文件必有三份文件；其字符目标也不是已保存文件的绝对硬上限。[上下文层级](https://docs.openviking.ai/zh/concepts/03-context-layers)。分层目的在于不必把全文一股脑放入模型，但摘要可能过期或遗漏细节，涉及事实真实性时应追溯原文。

### 什么时候收录、注入和读取

官方将「支持 MCP 查询」与「自动捕获和召回」分别描述：Claude Code、Codex、OpenCode 等有对应插件；Claude Code 插件在会话开始时注入用户画像/索引，在用户提交输入后、模型回答前自动召回，在回复后后台捕获，并在压缩或结束时提交。具体触发、预算和来源卡片受宿主与版本影响。[接入矩阵](https://docs.openviking.ai/zh/agent-integrations/01-overview)、[Claude Code 插件](https://docs.openviking.ai/zh/agent-integrations/02-claude-code)、[Codex 插件](https://docs.openviking.ai/zh/agent-integrations/04-codex)。

当前插件文档还说明会捕获工具调用及结果；这本身就与 membrain「只能使用用户亲自输入作为事实证据」的约束不同，不能直接将其捕获结果当作本项目合格事实。OpenViking 的 Working Memory 当前默认关闭，不能把它当成默认开启的能力。[Claude Code 插件](https://docs.openviking.ai/zh/agent-integrations/02-claude-code)、[Working Memory 配置](https://docs.openviking.ai/zh/guides/working-memory-default-off)。

OpenViking Gateway（Beta）提供另一种接入：能配置模型 Base URL 的客户端把请求发到网关，网关按消息召回、转发、保存对话并可在后续提交后提取记忆；也能代表模型执行记忆工具。这需要客户端支持相应的模型 API 和网关配置，并不证明它能够自动接管 ChatGPT 等厂商原生网页对话或读取其既有历史。[网关](https://docs.openviking.ai/zh/guides/15-gateway)。

Claude 插件文档中会话开始总限额默认是 9500 字节，最终 raw-find 兜底的内联召回预算默认 2000 tokens；不同阶段预算不应混为同一个硬上限。[Claude Code 插件](https://docs.openviking.ai/zh/agent-integrations/02-claude-code)。这些数字是其现行文档的默认值，不是本项目最佳值的实验证明。

检索文档对当前实现的说明是全局向量搜索，允许筛选路径、类型及层级；`find` 不做重排，`search` 可在会话上下文和配置条件下进行意图扩展和重排；`search(mode="context")` 默认不对最终合并候选进行重排，需显式启用。其通用介绍仍有「递归下钻逐层检索」表述，和当前专门检索文档存在版本/表述差异；以实际安装版本源码或行为为准。[检索原理](https://docs.openviking.ai/en/concepts/07-retrieval)、[检索 API](https://docs.openviking.ai/en/api/06-retrieval)。

### 历史回填与可视化

本地日志导入支持一次性导入历史和持续监听，默认关闭，需要显式配置；提供仅统计的 dry-run 与时间筛选。官方列出 Claude Code、Codex 等本地格式，具体支持名单以其页面为准；这不是对 ChatGPT、Claude 或 Gemini 网页全部历史的自动读取能力。[日志导入](https://docs.openviking.ai/zh/agent-integrations/09-log-ingestion)。

官方还提供 Windows x64 / macOS 的 OpenViking Helper（Beta），管理本地 Agent 接入、会话轨迹和记忆，提供中文界面；它依赖 OpenViking 服务，不能自行保证相应插件和每个 Agent 均已接入。底层 OpenViking 开源，但官方说明 Helper 本身是闭源软件。[Helper](https://docs.openviking.ai/zh/agent-integrations/14-openviking-helper)、[能力列表](https://docs.openviking.ai/zh/agent-integrations/16-capability-reference)。

需要特别验收：导入时输入的角色范围、提取模型是否使用助手内容作事实依据、污染过滤、冲突如何处理、历史回填去重、现有格式和非 ASCII Windows 路径，以及调用成本。现行文档描述的功能不能直接证明这些符合 membrain 的要求。

## OpenMemory：必须分清的三种东西

1. **Mem0 旧版 OpenMemory**：原先的自托管 MCP + 仪表盘项目已归档，不再积极维护；官方建议寻找自托管服务的用户使用 Mem0 自托管服务。[归档说明](https://github.com/mem0ai/openmemory/blob/main/openmemory-archive/README.md)。
2. **现在的 mem0ai/openmemory**：这是面向 Claude Code、Codex、OpenCode 的本地会话迁移工具，提供终端选择和预览；实时自动同步仍在路线图中，不能把它描述为已经完成的实时记忆服务器。[README](https://github.com/mem0ai/openmemory/blob/main/README.md)。项目说明指出当前迁移保留文字轮次，跳过工具调用、工具结果和思考块；它跨工具搬会话，不等于把其中的用户事实自动整理进 membrain。[项目说明](https://github.com/mem0ai/openmemory/blob/main/CLAUDE.md)。
3. **CaviraOSS 项目**：曾用 OpenMemory 名称，现在的项目是 LongMemory；其官方描述包括不可变内容、记录时间与有效时间、来源、关系、矛盾、预算内的证据选择、Wiki 和 MCP/HTTP/管理界面。[LongMemory 说明](https://github.com/CaviraOSS/LongMemory/blob/main/Why.md)、[项目主页](https://github.com/CaviraOSS/LongMemory)。具体哪些 Agent 可自动抓取、数据如何导入、产品功能成熟度和我们的规则是否兼容，仍需对当前版本进行独立验证。其 [1.2.1 发布说明](https://github.com/CaviraOSS/LongMemory/releases/tag/1.2.1)提到 episodic、semantic、procedural、emotional、reflective 五种 sector；这是版本相关设计，不应直接理解为所有同名产品的实现，其中情感/反思分类也不能自动引入本项目的性格推断。

## LLM Wiki：把知识整理成可读页面，而非无限注入

以 Microsoft 的实现为例：原材料层保存导入文档；LLM 生成的 Wiki 层形成实体、主题、摘要、交叉链接；规则层约束组织方式。它的核心包提供页面和索引读写、增量/批量导入、搜索与 MCP；VS Code 扩展提供界面、文件导入和 Copilot 问答。查询使用索引标题、摘要与页面正文的加权全文搜索，而不应默认称其使用向量搜索。可手工或批量导入，放入 `raw/` 的文件也会由 watcher 处理；文档没有证明它自动采集所有 Agent 的对话。[README](https://github.com/microsoft/llmwiki)、[架构](https://github.com/microsoft/llmwiki/blob/main/ARCHITECTURE.md)。

另一个独立实现 [AlexanderMakarov/llm-wiki](https://github.com/AlexanderMakarov/llm-wiki) 则更侧重 Agent 会话知识化：`sync` 读取本机 Claude Code 与 Codex CLI 会话到原始目录，`synth` 生成来源页面及候选，候选经审核后写入 Wiki，`build` 生成浏览网站；MCP 给 Agent 查询。同名 ChatGPT 及其他适配器属于需单独启用的 contrib 类别，并不能证明自动读取 ChatGPT 浏览器全部历史；本机可用性、角色识别、中文和 Windows 路径要另测。它依据 [Karpathy 提出的 LLM Wiki 方法](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)与其他项目构建，方法本身、该项目和 Microsoft 实现是不同对象。

适合借鉴为「人物/项目/主题可读汇总」，让模型先看目录和相关摘要，有必要时再打开证据；不适合把自动生成的综合页面当作原始证据。membrain 中 Wiki/摘要如果引入，应作为可重建的派生视图，保持原始用户证据和事实账本为真源。OpenViking 也提供 [LLM Wiki 编译示例](https://docs.openviking.ai/zh/context-compilation/02-llm-wiki)，但需另行验证它对本项目证据和冲突规则的兼容性。

## 其他可借鉴方案

- **Basic Memory**：Markdown 是真源，SQLite 是索引；支持语义/关键词混合搜索和可选重排，后者增加耗时且默认关闭。MCP 解决可调用接口，但不意味着每个客户端都会无提示地自动搜索或自动采集。[架构说明](https://github.com/basicmachines-co/basic-memory/blob/main/docs/ai-assistant-guide-extended.md)、[项目主页](https://github.com/basicmachines-co/basic-memory)、[语义检索](https://github.com/basicmachines-co/basic-memory/blob/main/docs/semantic-search.md)。
- **Graphiti**：事件作为证据，实体和关系带时间信息，检索混合关键词、语义和图遍历；其框架需要自建服务和图数据库，自动使旧关系失效的行为不能直接替代 membrain 的「矛盾先标记，用户决定」。该项目提供参考 MCP/REST 服务，但官方说明它们不是开箱即用的生产部署。[Graphiti](https://github.com/getzep/graphiti)。
- **Mem0**：自托管 SDK/服务接收应用提交的消息、抽取和检索与用户等标识关联的记忆，具体持久化和模型可配置。应用仍需把消息送入并把返回的记忆用于后续请求；托管平台能力与开源自托管能力、版本之间不能默认等同。[开源快速开始](https://docs.mem0.ai/open-source/quickstart)、[项目主页](https://github.com/mem0ai/mem0)。

## 对 membrain 的差距和可执行设计

当前项目的真实方案已写在 [DESIGN.md](DESIGN.md) 和 [PLAN.md](PLAN.md)：Markdown 事实、JSONL 原始对话和生命周期账本是持久内容；SQLite FTS5 + 向量是可重建索引；用户消息触发召回，后台模型提取新事实，MCP 和项目级 hooks 接入。语义阈值和基于有限短语识别的概览模式仍有盲区；跨 Agent 实际验收、全局接入和历史回填仍有边界。合成样例不等于真实场景召回率，也不能证明模型回答会正确使用记忆。

建议分环节改进，均须经过后续实现和对照验证：

1. **采集与回填**：先为本地 Codex/Claude 历史会话做预览、按时间/会话选择、仅用户输入的解析、来源和稳定 ID 去重、后台批量提取；其他网页产品从官方导出入手。保留原始引用、时间和来源；自动失败不影响对话。不同 Agent 的授权和可用接口分别核验。
2. **写入与更新**：分类识别「事实、偏好、目标、项目」并校验逐字引用；先找相似旧记录，再判定新增/补证/显式变更。矛盾保持 contested，不自动判胜负；约束只由用户明确维护。助手答复、工具内容和注入块不可作为证据。信心仍由代码计算。
3. **派生层**：在不可变事实和账本外，增加可重建的主题摘要及目录，保留到事实 ID 和原话的链接；摘要更新和删除后必须同步失效或重建。摘要供导航，不是新的证据或自动修改 `memory/profile.md` 的依据。
4. **召回**：根据最近对话和当前提问识别特定查询、广泛概览、项目查询及更正；关键词、语义、类别/项目过滤并行产生较宽候选，再做中英文适配的重排、去重、时间及冲突状态检查。概览走分类浏览/分页，避免只靠单一向量阈值。重排的模型失败时保留清晰的降级与观测，避免把阈值分数当作事实置信度。粗召回可以偏重不漏，最终注入必须有预算和相关性筛选。
5. **注入时机与量**：会话开始仅注入经用户确认的稳定约束/必要画像；每轮注入当前任务相关的少量事实及必要来源，按会话去重，在上下文压缩或切换话题时重新评估。Agent 需要细节时通过 MCP 再读取来源。具体预算先作为待测参数，不保证某个固定数值最好。对已撤回、已变更或有争议的信息，后续重新注入更新标记；已经进入模型上下文的旧文本不能靠删除数据库自动消失，应考虑压缩/新会话提示。
6. **可视化后端**：在 1966 页面展示接入状态、最近一次捕获时间、待处理/失败的历史回填、每次候选及最终注入的来源和理由、预算与耗时、冲突及可纠错入口；不要展示私密原文给第三方，具体展示和数据传输以用户选择及配置为准。

## 验收：如何比较，何时才算改进

在用户允许的样本或隔离合成数据上构建同一批有标准答案的中英文测试，覆盖直接问题、宽泛概览、跨轮指代、项目切换、纠错、互相矛盾、较老事实、无关输入和多 Agent 场景。分别记录：候选召回率 Recall@K、最终注入的相关率 Precision@K、正确记忆未进入上下文的比例、错误/重复注入、每轮 token、延迟及模型费用。进一步比较真实回答是否使用正确记忆、是否恰当地说明争议、是否正确引用来源。标明数据规模、分母、模型/版本和测试日期，避免只引用供应商宣传数字。

建议先用相同数据、同等输入与预算分别评估现有 membrain 和 OpenViking；如另试 Basic Memory、LongMemory，也须将历史来源、接入范围、模型与重复处理说明清楚。先预览和隔离测试，确认规则兼容、效果和成本，再决定继续改造、替换后端还是直接采用成品；这次调研没有进行迁移或安装。
