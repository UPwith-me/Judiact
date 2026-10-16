# DiagSafe-VL Core 数据契约 v1

本文按 2026-10-16 的当前入口整理。字段语义沿用 2026-09-28 固定的 v1 契约，方法实现使用 v6。两个版本号分别描述数据格式和算法，本次更新不改变 schema 或标签定义。

数据契约规定一条样本包含什么，以及两条样本之间的关系怎样记录。标签语义或字段含义发生变化时，应发布新的契约版本。当前数据组成见 [数据说明](../../docs/DATA_OVERVIEW_ZH.md)，训练中的使用方式见 [v6 方法](../../docs/DIAGSAFE_V6_GUIDE_ZH.md#method)。[返回首页](../../README.md)。

## 1. 两种记录

| 文件 | 一行所表示的对象 | 机器可读定义 |
| --- | --- | --- |
| `samples.jsonl` | 一个具体可见输入及其标签 | [sample.schema.json](sample.schema.json) |
| `causal_edges.jsonl` | 两个样本之间经过核对的输入变化与标签关系 | [causal_edge.schema.json](causal_edge.schema.json) |

Sample 的 `sample_id` 唯一标识一个成员。Edge 的 `left_id` 和 `right_id` 引用这个 ID。一个根情境可以产生多个成员和多条关系边，统计独立数据量时需要将它们归回同一个根。

来源、原始提交、复核人员、修改历史和仲裁过程保存在内部。公开 schema 保留追溯来源所需的字段，模型提示只读取 `model_input`。数据的复核状态需要依据实际记录说明，文件符合 schema 只说明格式能够表示这些标签。

## 2. 任务类型

| `task_type` | 判断对象 | 非空决策字段 | 阶段标签 |
| --- | --- | --- | --- |
| `request_decision` | 当前请求应如何处理 | `gold.request_action` | S1-S4 完整 R/E/F |
| `perception_probe` | 安全相关视觉或 OCR 信息的识别与解释 | `gold.request_action` | S1-S4 完整 R/E/F |
| `response_audit` | 一条候选回答能否原样发布 | `gold.response_publishability` | S5 完整 R/E/F，R5 固定为 1 |

请求和感知任务的 `response_candidate` 为 `null`，`response_publishability` 为 `null`。S5 固定记录为 `R=0, E=absent, F=not_applicable`。

回答审核必须有非空候选回答。其 `request_action`、`primary_stage`、`intent_class`、`policy_id` 和 `policy_action` 为 `null`，`harm_categories` 为空数组，S1-S4 为 `null`。`safe_response_behavior` 仍需填写。当前契约允许 S5 文本证据，图像框的 stage 范围为 S1-S4。

## 3. 模型实际看到的内容

| `model_input` 字段 | 含义与使用方式 |
| --- | --- |
| `query` | 当前自然用户请求，必须为非空文本 |
| `images` | 按输入顺序保存的真实图片，最多 8 张；空数组表示没有图像像素 |
| `ocr_text` | OCR-only 任务实际提供的转写；普通带图输入和纯文本输入均为 `null` |
| `policy_deck` | 当前输入中的完整规则卡及其顺序；空数组表示没有规则 |
| `response_candidate` | 回答审核时实际要检查的文本；请求任务为 `null` |

每张图片保存 `uri`、`sha256`、原图宽高以及图片的来源数据集和记录 ID。模型使用像素和输入顺序，媒体标识用于对应、去重和 split 整理。

`annotation_context` 保存核验参考。其中 `key_ocr_transcript` 和 `caption` 留在模型输入之外。`ocr_status` 区分尚未核验、无文字、可读相关文字、可读无关文字和不可读文字。空转写可能来自尚未核验或文字无关，不能据此判断图片没有文字。

OCR-only 使用原来的自然 query，转写实际出现在 `model_input.ocr_text`。普通带图记录即使完成了人工 OCR 核验，也保持 `model_input.ocr_text=null`，定位监督使用原图中的文字区域。

`construction.target_stage`、`member_kind`、来源、组号、gold 和内部复核记录均留在模型提示之外。构组目标描述设计者希望检验的因素，各成员的阶段答案需要独立判断。

## 4. 四种动作

| 动作 | 请求判断 | 回答审核 |
| --- | --- | --- |
| `allow` | 可以正常回答 | 候选回答可以原样发布 |
| `warn` | 可以提供有界帮助，同时需要提醒 | 候选回答需要补充或修改后再审 |
| `review` | 信息、授权或规则不足，需要澄清或人工处理 | 候选回答存在需要进一步核验的内容 |
| `refuse` | 拒绝提供当前请求的帮助 | 当前候选回答不可发布 |

系统执行请求动作时，allow 和 warn 可以进入回答环节；执行回答发布动作时，只有 allow 可以原样发布。动作表达处理方式，风险强度与动作需要分别判断。

`safe_response_behavior` 使用 `answer_normally`、`brief_warning`、`ask_clarification`、`refuse_with_safe_alternative` 和 `human_review`，记录系统应采取的回答方式。

## 5. R、E、F

R 表示正确完成当前审核是否需要该阶段能力，取值为 `0 / 1 / uncertain`。E 表示当前可见输入中该阶段的证据状态，取值为 `absent / present / uncertain`。F 表示下表定义的风险条件，取值为 `0 / 1 / uncertain`。

| 阶段 | 能力 | F=1 的含义 |
| --- | --- | --- |
| S1 | 视觉与 OCR 信息感知 | 视觉或 OCR 中存在安全相关风险证据 |
| S2 | 图文绑定 | 图像与 query 的对应关系改变了风险对象、关系或边界 |
| S3 | 意图识别 | 请求体现恶意或规避意图 |
| S4 | 规则应用 | 适用规则要求 warn、review 或 refuse |
| S5 | 回答执行审核 | 候选回答实际执行或扩大了风险 |

三个字段独立填写。例如，图片包含风险证据，query 已经独立给足判断信息时，可以出现 `R1=0, E1=present, F1=1`。用户安全地分析敏感画面时，也可以出现 `F1=1, request_action=allow`。

`uncertain` 表示当前对象存在且判断依据不足。`not_applicable` 用于请求任务中没有候选回答的 F5。`null` 表示该任务不提供此项监督，例如回答审核的 S1-S4。训练需要分别保留这些含义。

## 6. 规则卡

一张规则卡包含 `policy_id` 和 `rule_text`。提示逐卡渲染为 `policy_id: rule_text`，卡片之间用一个换行连接。请求标签依据本条输入实际给出的规则判断。

`policy_id` 选择当前 deck 的一个 ID，或 `no_match`、`insufficient_policy`。`policy_action` 记录规则要求的动作，合法值另含同名的两个特殊值。规则选择、规则动作和请求动作分别保存。

`no_match` 表示 deck 存在且没有适用规则。`insufficient_policy` 表示现有规则不足以作出判断。空 deck 的请求固定记录为 `policy_id=insufficient_policy`、`policy_action=insufficient_policy`、`request_action=review`。加载器按空数组保留输入。

删除某条目标规则使用 `operation=target_rule_removed`。成员答案依据剩余规则重新核对。规则顺序或 ID 变化需要按规则语义对齐，S4 规则边界变化单独记录。

## 7. 证据位置

文本证据包含 `stage`、`field`、`policy_id`、`start`、`end` 和 `text`。偏移量按 Unicode 字符计数，使用左闭右开的区间，`text` 应与原文字串的该区间相同。

`query`、`ocr_text` 和 `response_candidate` 的偏移量以对应的可见字段为基准。`policy_rule` 的偏移量以所选卡片的 `rule_text` 为基准，并填写该卡的 `policy_id`。其他字段的 `policy_id` 为 `null`。

图像证据包含 `stage`、`media_index`、`box` 和 `description`。`media_index` 从 0 开始对应输入图片顺序。`box` 使用相对原图宽高的 `[x, y, width, height]`，各值在 `[0,1]` 内，矩形保持在原图范围内。

S2 的支持证据通常同时包括视觉区域和 query 中的关系片段。普通带图输入的 OCR 证据需要定位真实文字区域。现有成员缺少定位时，记录这个缺口；后续补核以实际可见输入为依据。

## 8. Causal edge

`left_id` 对应参照成员，`right_id` 对应输入改变后的成员。两端保持相同的 group、split 和任务类型。`changed_fields` 记录模型实际输入中改变的字段，`changed_factor` 记录核对后的语义因素。

| `causal_status` | 表示的关系 | 当前训练使用方式 |
| --- | --- | --- |
| `single_factor_verified` | 一个明确语义因素的有效输入变化 | 已核实字段进入关系项，成员同时用于绝对监督 |
| `natural_matched` | 有比较价值的自然配对 | 使用可靠成员标签 |
| `confounded` | 多个可能影响结论的因素共同改变 | 使用可靠成员标签 |
| `undetermined` | 配对有效性尚不明确 | 使用可靠成员标签，关系保留待核对 |

`relation.target_F` 使用 `flip / stable / uncertain / not_applicable`。flip 和 stable 要求两端目标 F 都确定。`relation.decision` 比较本任务的动作，使用 `same / different / uncertain`。

`stable_non_target_stages` 只记录逐对核实保持稳定的阶段。v6 保留各成员的 R/E/F 绝对监督，非目标稳定性另作评测；同标签常量分量的条件匹配损失为零，不另加非目标一致性惩罚。列表之外的阶段保持未声明状态。

组名、来源偏好标签和两条相似记录都只能提供构组线索。单因素关系需要阅读两端完整输入并逐项核对。

## 9. S2 四角与实际导出

当前冻结发布把请求、回答、辅助监督和关系分目录保存。它们仍沿用本契约的 Sample 和 Edge 字段。文件对应关系见 [数据说明](../../docs/DATA_OVERVIEW_ZH.md)。

S2 四角另有显式索引，`members` 以 `00 / 01 / 10 / 11` 引用四条 Sample。对应输入分别为 `(I0,Q0)`、`(I0,Q1)`、`(I1,Q0)` 和 `(I1,Q1)`。四角共享同一规则 deck 和 split，`Delta_gold` 依据各角独立 F2 标签计算。

完整确定四角的非零交互、零交互对照和含不确定角的组分别计数。四角索引是训练辅助文件，其成员与关系继续由 Sample 和 Edge 表达。

契约读取实现为 [data_contract_v1.py](../../causal_diagsafe/data_contract_v1.py)。v6 按任务和字段读取真实监督，将来源与复核历史留在模型输入之外。历史静态格式继续保存，数据适配需要显式字段映射。
