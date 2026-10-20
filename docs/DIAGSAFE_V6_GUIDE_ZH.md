# DiagSafe-VL v6 方法与运行说明

本文保留 v6 的历史信息流与运行入口。当前 v7 的两阶段潜变量训练及独立推理命令见 [v7 方法与运行说明](DIAGSAFE_V7_GUIDE_ZH.md)，下文参数不作为 v7 配置。

本页对应 2026-10-16 同步的正式主代码。方法协议为 `grounded_audit_v6_20261016`，证据解码版本为 `xgrammar_0_2_8_audit_v3_compact`，数据契约保持 v1。本文只说明当前实现，不收录内部机制试验和运行日志，也不将方法代码视为已验证的效果结论。

[返回首页](../README.md) · [当前数据](DATA_OVERVIEW_ZH.md) · [字段定义](../data_contract/v1/README_ZH.md)

<a id="method"></a>
## 1. 方法与信息流

系统审核请求和真实候选回答，实际回答由独立模型生成。请求动作 allow/warn 可以进入回答环节；回答发布动作只有 allow 允许原样发布。S1-S5、意图、规则与证据保留各自的判断，不用最终动作反推阶段标签。

主干为完整 Qwen3-VL-4B-Instruct，保留原 LM head，只更新语言 attention q/k/v/o 的共享 LoRA。没有新增分类头、路由器或第二个审核模型。

输入只取 `model_input` 中的真实图片、自然 query、实际 deck 和候选回答。普通带图输入不额外提供人工 OCR 或 caption；OCR-only 保留原自然 query 和实际转写。来源、组号、构组目标、gold 和内部复核记录不进入提示。

一次请求审核按下列顺序执行。

1. 生成 S1、S2、S3 证据块。
2. 读取 S1-S3 的 R/E/F、意图和当前 deck 的 policy ID。
3. 生成 S4 规则证据。
4. 读取 policy action、S4 的 R/E/F、请求动作和安全回答行为。

回答审核生成 S5 证据，再读取 E5、F5、发布动作和回答行为。R5 由存在候选回答的任务契约固定为 1。原生辅助只读取来源实际支持的安全问题，不补造完整阶段标签。

分类使用原 LM head 在合法候选 token 上归一化，并取 argmax。选出的类别保存到对外审核记录；后续模型上下文中的该位置统一写入固定的 ` ?` token，不写入选出的类别。占位符不代表 uncertain。原图、原文、已生成证据及共享隐藏状态仍可被后续字段利用，因此这也不意味着各字段统计独立。

证据包括可见文本的精确引文和原图 0-1000 坐标框，由 XGrammar 约束为紧凑 JSON。每个证据块最多生成 512 token。解析错误、错误引用和截断仍如实记录，语法约束不保证语义正确。

## 2. 训练对象与目标

每个完整 epoch 中，一个成员首次被使用时，由当时的模型仅根据可见输入生成证据；该轮后续使用复用这些精确证据 token，下一轮再刷新。生成停止梯度，生成时读出的类别不进入后续分类上下文。

分类和关系损失读取原输入与自主证据，包括错误或截断证据。人工证据另用一条前向监督引文和定位，该前向上的分类损失不参与训练。这样保留定位教学，同时避免 gold 证据替分类完成推理。

训练目标为

\[
L=L_{record}+L_{matching}+0.1L_{native}.
\]

`L_record` 包括动作、诊断字段和证据监督。请求与回答各占一半记录预算，任务内部先平均根或可见请求上下文，再平均同根成员。动作保留独立预算，其他实际有监督字段共享诊断预算。证据 token 损失按完整 train、根加权的字段平均目标长度归一化，不按每条证据自身长度归一化。

默认 `determinate_risk_action` 从冻结 train 的实际损失系数计算类别倍率。动作的实际观测类别等预算，F1-F5 仅对确定的 0/1 等预算，uncertain 保留原有系数质量；R、E、意图、policy、行为和原生监督不做同样的类别均衡。输出由此成为参考测度下的条件分数，不能当作已校准的自然分布风险概率；推理不再另做先验补偿。

`L_matching` 使用已核实关系，在组内固定各标签计数后，比较正确分配与其他可能分配。对一个字段的成员预测 \(q_i\)，目标为

\[
L_{g,f}=\log\sum_{\pi\in\Omega(n)}
\exp\!\left(\sum_i\log q_{i,f}(\pi_i)\right)
-\sum_i\log q_{i,f}(y_i).
\]

分母采用精确计数动态规划。各成员读取自己的原输入与自主证据，不交叉喂入别人的 gold 证据。只使用实际成立的字段关系；同标签常量分量的匹配损失为零。S4 顺序/ID 不变性单独保留成员监督和评测，不冒充规则边界变化。组内相对匹配无法确定绝对边界，因此单条监督仍然必要。

原生辅助只使用真实的 prompt、response 或 image 安全监督。来源 chosen/rejected 不直接当作回答安全 gold。已知定位冲突只在训练投影中屏蔽对应证据块，保留其他字段及原始数据，详见 [数据说明](DATA_OVERVIEW_ZH.md)。

<a id="training"></a>
## 3. 环境与完整训练

正式训练使用 Linux、Python 3.11、Transformers Trainer、Accelerate DDP 和 PEFT。共用环境版本记录在 [train-v6.txt](../requirements/train-v6.txt)，其中包含基础训练依赖与 XGrammar 0.2.8。CUDA 对应的 PyTorch/torchvision wheel 需要按训练机器选择；公开 benchmark 的 lmms-eval 环境单独准备，此清单不包含其全部依赖。

在新建环境中安装依赖，不需要为阅读文档修改正在运行的环境。

```bash
python -m pip install -r requirements/train-v6.txt
```

| 项目 | 正式设置 |
| --- | --- |
| 初始化 | 公开 Qwen3-VL-4B-Instruct，新建零增量 LoRA，不继承旧实验权重 |
| 可训练部分 | 语言 attention q/k/v/o LoRA，r=16、alpha=32、dropout=0.05 |
| 冻结部分 | 基础模型权重、视觉编码器及原 LM head |
| 精度 | BF16 主干与激活，FP32 损失，不量化；启用 TF32 |
| 图像面积参数 | processor shortest_edge=65536、longest_edge=1003520 |
| 优化 | AdamW，lr=2e-5，betas=(0.9,0.999)，eps=1e-8，weight decay=0.01 |
| 学习率与裁剪 | 完整运行 cosine，warmup 3%，梯度范数上限 1 |
| 轮次与种子 | 连续 3 个完整 epoch，seed=42，不按 A/B/C 接续 |
| 更新预算 | 全局约 64 个任务根单位；当前清单每轮 658 更新，共 1974 更新 |
| 四卡启动微批 | 审核 8、原生 16、取证 8；padding token 预算 16384 |
| 激活重计算 | 审核全语言层，原生每隔一层 |
| 关系 / 原生系数 | 1 / 0.1 |

在仓库根目录运行下列命令，三个路径变量指向机器上已经准备好的模型、发布和新输出目录。发布目录包含媒体可访问路径与 `reviews/` 根归并记录，详见 [文件布局](DATA_OVERVIEW_ZH.md#files)。

```bash
export MODEL=/path/to/Qwen3-VL-4B-Instruct
export RELEASE=/path/to/visual_refusal_fill_20261015
export RUN=/path/to/new_v6_run
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4

python -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m causal_diagsafe.cli.train_v6 \
  --model "$MODEL" --release "$RELEASE" --output "$RUN" \
  --epochs 3 --seed 42 --effective-roots 64 \
  --record-batch-size 8 --native-batch-size 16 --rollout-batch-size 8 \
  --max-batch-tokens 16384 --native-checkpoint-layer-stride 2 \
  --fact-dispatch balanced --recovery-save-steps 100
```

[正式启动脚本](../scripts/ds_disk_grounded_v6.sh) 将四卡训练与训练后的完整 dev 串联，默认目录对应项目服务器布局，可用 `DIAGSAFE_ROOT`、`DIAGSAFE_CODE`、`DIAGSAFE_V6_RUN` 指定根目录、代码目录和输出。通用机器可直接使用上面的 CLI，不需要复制服务器目录名。

四卡只在同一次更新内分担自主取证，按证据块数与输入长度分配工作，再将结果返回原请求卡。标签损失的 packet、更新预算与根权重保持不变，不提前取下一次更新的证据。

每个完整 epoch 保存并评测自然 dev。checkpoint 选择量先平均各任务中实际存在动作类别的 NLL，再平均请求与回答两个任务。只在完整 epoch checkpoint 中选择，optimizer 和 scheduler 全程连续。首步及每 100 步的快照只用于恢复，不参与中途模型选择；恢复入口是 `--resume-from-checkpoint /path/to/checkpoint-N`。

训练输出包括 `selected/` adapter、`run_settings.json`、`sampling.json`、`risk_action_balance.json`、`normalization.json`、逐轮 `dev_selection/` 和恢复 checkpoint。自主证据缓存位于 `training_facts/`，恢复时只复用保存步之前生成的记录。

<a id="evaluation"></a>
## 4. 推理与评测

当前共用 CLI 保留 v4 文件名，**必须显式使用 `--protocol v6`**。输入文件可以是完整 Sample，也可以是纯 `model_input` 对象；完整 Sample 中的 gold 不会传给模型。

```bash
python -m causal_diagsafe.cli.infer_v4 --protocol v6 \
  --model "$MODEL" --adapter "$RUN/selected" \
  --inputs /path/to/inputs.jsonl --output /path/to/audits.jsonl

python -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m causal_diagsafe.cli.evaluate_v4 --protocol v6 \
  --model "$MODEL" --adapter "$RUN/selected" \
  --release "$RELEASE" --output "$RUN/dev_eval" \
  --initial-batch-size 8 --max-batch-size 8
```

动作使用确定性 argmax，证据不采样。自然 dev 请求与回答分别保留分母，输出动作 macro-F1、NLL、各类召回、危险放行、过拒绝及 review 比例；阶段、policy 和证据另报。机制评测区分严格组正确、S2 风险四角全对、最终动作四角全对及交互方向，不能只报其中较高的一项。

自然 dev 请求 299 条、回答 200 条。回答 review 只有 3 条，分层指标必须连同分母解释。隐藏测试和公开 benchmark 不参与选模型、调阈值或选择数据量。

公开评测使用 [evaluate_public_v4.py](../causal_diagsafe/cli/evaluate_public_v4.py) 的 v6 协议，沿用既有 lmms-eval 任务接口、[外部评测配置](../configs/external_baselines_20261009.json) 和预处理数据，不启动外部模型训练。配置中的 `method_models` 仍保存历史 v3 任务名；v6 adapter 由该 CLI 的 `--adapter` 与 `--protocol v6` 显式指定。ORBench 与 JailBreakV-28K 继续排除。

这个公开入口将四动作映射为 policy-hold 代理指标，请求 review/refuse 视为暂不放行，回答只有 allow 视为原样发布。它与原生 harmfulness、生成系统 ASR 含义不同，不能混成同一结果表。完整生成系统比较仍复用独立回答生成、回答审核与 judge 流程。

<a id="implementation"></a>
## 5. 实现索引与结论边界

| 职责 | 代码 |
| --- | --- |
| Qwen、processor 与 LoRA | [diagsafe_v3.py](../causal_diagsafe/models/diagsafe_v3.py) |
| 正式协议 / JSON 语法 | [audit_v6.py](../causal_diagsafe/audit_v6.py)、[audit_constraints.py](../causal_diagsafe/audit_constraints.py) |
| 自主证据与固定类别占位符 | [v6_sequence.py](../causal_diagsafe/training/v6_sequence.py) |
| 数据读取 / 证据投影 | [v4_data.py](../causal_diagsafe/training/v4_data.py)、[v5_data.py](../causal_diagsafe/training/v5_data.py) |
| 根权重 / 单条损失 | [v4_losses.py](../causal_diagsafe/training/v4_losses.py)、[v6_supervision.py](../causal_diagsafe/training/v6_supervision.py) |
| 条件匹配 | [v5_matching.py](../causal_diagsafe/training/v5_matching.py) |
| 自主证据上的统一目标 | [v6_trainer.py](../causal_diagsafe/training/v6_trainer.py) |
| 逐轮取证、四卡分配与选择 | [v6_formal.py](../causal_diagsafe/training/v6_formal.py) |
| 训练入口 | [train_v6.py](../causal_diagsafe/cli/train_v6.py) |
| 正式推理 | [inference_v6.py](../causal_diagsafe/inference_v6.py) |
| 评测聚合 | [evaluation_v4.py](../causal_diagsafe/evaluation_v4.py) |

共享模块保留旧版本文件名，正式路径由上述 v6 入口组合。模块内部为历史研究保留的可选分支不等于正式默认设置，应以 `FormalAuditTrainer` 和运行设置为准。

自主证据存在错误与轮内陈旧性，停止梯度的生成也不是精确边际似然。类别占位符只切断类别文本回填，不证明证据是必要因果中介。组内匹配不保证非目标字段稳定或绝对动作正确。当前仍需完整训练与独立评测检验这些能力，本页不作收敛、最优或泛化保证。
