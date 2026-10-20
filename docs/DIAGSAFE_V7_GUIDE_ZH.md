# DiagSafe-VL v7 方法与运行说明

当前入口为 `train_v7_latent`。数据契约沿用 v1，冻结数据沿用 `visual_refusal_fill_20261015`。本文记录主方法与可运行入口，不包含内部诊断脚本、训练结果或新模型的性能承诺。

[返回首页](../README.md) · [数据说明](DATA_OVERVIEW_ZH.md) · [字段契约](../data_contract/v1/README_ZH.md)

<a id="method"></a>
## 1. 输入、过程与输出

模型审核请求是否可以协助，或实际候选回答能否原样发布。它不会生成被审核请求的实际答案。

```text
实际可见输入 X -> 来源事实 W
W、实际规则 P -> 规则链接 L
W -> 请求 S1-S3 与意图；W、P -> 回答 E5/F5
W、P、初始诊断、L -> 请求 policy 与 S4
事实、来源归属、规则与诊断 -> D -> 最终动作 A -> 安全回答行为
```

所有节点共享同一 Qwen3-VL-4B-Instruct、一份语言侧 LoRA 和原 LM head。各节点重新编码自己的明确输入，不共享跨节点 KV 或隐藏状态。没有额外的分类主干、奖励模型或最终动作 router。

### 1.1 来源事实 W

W 记录行为主体、操作与对象、否定、授权、条件作用域和观察状态。事实引用当前输入的来源 ID：`query`、实际提供的 `ocr_text`、`response_candidate` 或 `image_i`。程序物化被引用的真实文本字段或原始图片，事实 ID 按顺序分配。

W 不读取规则、gold、来源数据集名称、构组目标或组号。普通带图输入不泄漏人工 OCR 和 caption。文本声称、图片观察与尚未建立的前提分开，不能把未观察到改成不存在。

这些来源单元是整段字段和整张原图；细粒度证据定位是另一个输出，不能把整图引用算成精准框选。模型生成的事实内容仍可能错误，来源引用存在不证明其解释真实。

### 1.2 规则、诊断与动作

L 将当前规则的前提、例外和未明确条件关联到当前事实。规则 ID 只用于引用实际条款，原文从当前 deck 物化。提议 L 时不提供阶段判断或最终答案。

R/E/F、意图与 policy 继续按契约监督。请求审核预测 S1-S4；回答审核预测 E5/F5 和回答可发布性，R5 由候选存在固定为 1，不补造回答任务的 S1-S4 gold。

最终动作只读 D 中的事实、来源类型、规则和诊断，不读取 source spans、原图、原始 query/candidate、前一节点 logits 或 KV。`policy_action` 正常预测与评测，但从最终动作的父字段中移除，避免直接复制这个近似答案。

自由事实文本仍有携带统计捷径的可能。结构上的无旁路不等于语义忠实，也不保证改变任意 F 都必须改变动作。诊断需要结合真实事实和实际规则解释。

细粒度证据在动作之后独立生成，使用实际可见来源定位；它是解释输出和监督项，不被宣称为已经驱动动作的前置证据。

### 1.3 与 schema 的关系

推理入口保存完整 `trace`，包含 W、L、诊断、动作、行为及证据。请求与回答的动作分别解释为 `request_action` 和 `response_publishability`，不能把回答中的安全拒绝自动判为不可发布。

该 trace 不是完整 Sample 的自动重写。原 Sample 中的 `primary_stage`、`harm_categories` 等数据字段仍保留在数据侧，本版没有为它们补造预测。来源、复核历史和机器生成标记不进入模型提示，机器参考也不获得人工终审身份。

## 2. 两阶段训练

### 2.1 弱监督 SFT

32B 教师仅对 train 的真实输入生成 W/L 弱参考，不接收最终动作、阶段 gold 或 gold 定位。相同来源可以复用 W，实际 deck 不同时重新生成 L。教师不改写已有标签，也不常驻部署。

SFT 使用机器过程参考、已有类别和定位、有效关系及原生监督。文本按节点长度归一化，作为初始化目标。结构无效或无法支持已有定位的参考保留失败原因，不用默认答案填补；该成员仍可以在第二阶段使用学生候选。

### 2.2 潜变量学习

已有类别、动作、行为及可用定位构成观测 Y，事实表达与规则关联构成潜变量 z。观测目标为

\[
\mathcal L_{\rm obs}=-\log\sum_z p_\theta(z\mid X,P)p_\theta(Y\mid X,P,z).
\]

公式中的输入仍遵守前述节点权限，动作因子始终为 `p(A|D)`。观测父字段只在计算联合似然时 teacher-force；生成候选与独立推理不看这些 gold。

每轮开始冻结一份旧 LoRA。每成员从该策略采样 M=2 份候选，另外保留一份可用固定参考 b。固定参考作为离散原子单独求和，学生采样估计其补集：

\[
\widehat Z_Y=p_{\rm old}(b)p_{\rm old}(Y|b)
+\frac1M\sum_j\mathbf1[z_j\ne b]
\frac{p_{\rm old}(z_j)p_{\rm old}(Y|z_j)}{q(z_j)}.
\]

没有可用 b 时，仅使用学生候选。相等按实际 token 路径判定，不做语义去重。固定参考采用声明的序列化；学生轨迹保留实际采样 token，不重写后再沿用旧概率。

q 在生成时、语法约束之后逐 token 记录。生成与重算使用同一 XGrammar 支持集合，包括 EOS；不以 top-k 截断或 token 平均冒充整条概率。数值重算可能因 BF16 批形状产生差异，不能用重算值替代实际采样 q。

对同一积分测度分别得到归一化先验权重 w_prior 与后验权重 w_post，停止这些权重的梯度，使用

\[
\widetilde{\mathcal L}=
-\sum_k(w_{\rm post,k}-w_{\rm prior,k})\log p_\theta(z_k)
-\sum_k w_{\rm post,k}\log p_\theta(Y|z_k).
\]

先验 score 在总体期望下为零，用作控制变量；候选对观测的解释力完全相同时，不额外强化随机采到的措辞。有限候选自归一化仍有偏差，没有精确 EM 或单调提升保证。代理 loss 不能直接作为边缘 NLL 曲线解释。

被截断或无法支持已有证据的候选没有完整观测似然；无效候选不拿半份观测参加关系训练。整个候选集合没有观测支持时记录失败，不伪造后验或正确 review。可用参考也可能语义错误，正确标签不能唯一识别所有真实解释。

### 2.3 根权重与已有关系

核心监督按请求、回答各半分配，再按根/可见上下文和成员计量。动作类平衡作用于整条观测权重，不在联合似然内部给某个字段随意加幂。原生监督系数为 0.1，仅使用来源真正支持的 prompt/response/image harmfulness。

已有核实关系继续做固定标签计数下的条件匹配，使用各成员可用候选上下文的字段概率。完整 S2 四角替代重复配对事件；非零、零交互和不确定组分开；常量分量不产生方向性损失。关系项是额外复合监督，不是假定归一化联合模型中的一个概率因子。

<a id="training"></a>
## 3. 环境与正式设置

使用 Python 3.11、Transformers Trainer、Accelerate 和 PEFT。学生环境沿用 torch 2.7.1、transformers 4.57.1、peft 0.17.1、accelerate 1.10.1 和 XGrammar 0.2.8。

```bash
python3.11 -m venv .venv-train
.venv-train/bin/python -m pip install -r requirements/train-v7.txt
python3.11 -m venv .venv-teacher
.venv-teacher/bin/python -m pip install -r requirements/teacher-v7.txt
```

两个环境分开。vLLM 0.11.0 解析它所需的 torch/CUDA 依赖，不与学生环境的 torch pin 混装。CUDA wheel 和驱动需要与实际机器匹配；原生扩展编译需要对应 Python 开发头文件，非系统路径可通过 `CPATH` 提供。本仓库不附带模型权重与 CUDA 工具链。

| 项目 | 本轮设置 |
| --- | --- |
| 学生起点 | 公开 Qwen3-VL-4B-Instruct，新语言 LoRA |
| LoRA | q/k/v/o，r16，alpha32；视觉、主干和原 LM head 冻结 |
| 轮次 | 2 个完整 SFT epoch，随后 2 个完整潜变量 epoch |
| 学习率 | SFT 2e-5，潜变量 1e-5；阶段内 cosine、3% warmup |
| 优化器 | fused AdamW，betas 0.9/0.999，eps 1e-8，wd 0.01，clip 1 |
| dropout | SFT 0.05，潜变量 0 |
| 学生采样 | 每成员每轮 2 份；temperature 1、top_p 1、top_k 0 |
| 学生生成预算 | 每个自由文本节点 2048 tokens |
| 数值精度 | BF16 主干/激活，FP32 LoRA/概率损失，FP64 后验归一化；TF32 关闭 |
| 学生并行 | 四卡 DDP，名义全局 64 根/更新 |
| 批次上限 | 类别 16、文本 2、原生 32、rollout 16；padded tokens 24576 |
| 重计算 | 每层激活重计算；冻结视觉结果缓存；不缩图或量化 |
| 保存 | 每个完整 epoch 的 checkpoint 与自主自然 dev |

教师默认使用两个 TP2 副本、BF16 权重与 KV、每副本 32 个并发序列、显存比例 0.86。仅在结构生成失败时尝试一次修复，修复预算为原预算的两倍；超出学生生成空间的参考不进入第二阶段的固定原子。教师 token 概率不被虚构为学生采样分布。

### 3.1 准备数据与路径

完整数据布局见 [训练读取的文件](DATA_OVERVIEW_ZH.md#files)。还需发布内的根归并复核文件、可访问的真实媒体、学生和教师模型。代码仓库不包含这些私有运行输入。

四卡完整顺序任务可从仓库目录启动：

```bash
export DIAGSAFE_ROOT=/data/diagsafe
export DIAGSAFE_PYTHON="$PWD/.venv-train/bin/python"
export DIAGSAFE_VLLM_PYTHON="$PWD/.venv-teacher/bin/python"
export DIAGSAFE_TEACHER="$DIAGSAFE_ROOT/raw/models/Qwen3-VL-32B-Instruct"
bash scripts/ds_disk_v7_latent_20261020.sh
```

默认数据为 `$DIAGSAFE_ROOT/releases/visual_refusal_fill_20261015`，学生为 `$DIAGSAFE_ROOT/raw/models/Qwen3-VL-4B-Instruct`。`DIAGSAFE_RELEASE/MODEL/TEACHER/BANK/RUN/CODE` 均可覆盖默认路径。启动脚本保留与仓库既有服务器脚本相同的目录约定，但不写死个人账号。

脚本先生成 `bank.rank0.jsonl` 和 `bank.rank1.jsonl`，教师结束后释放四卡，再训练学生。教师阶段关闭 `expandable_segments`，避免与当前 vLLM custom all-reduce 的 CUDA IPC 冲突；学生阶段开启，不改变浮点精度。

### 3.2 分别启动学生阶段

已经有过程参考时，也可单独调用主 CLI：

```bash
SPY=.venv-train/bin/python
MODEL=/data/models/Qwen3-VL-4B-Instruct
RELEASE=/data/diagsafe/releases/visual_refusal_fill_20261015
BANK=/data/diagsafe/work/v7_latent_bank_20261020
RUN=/data/diagsafe/runs/v7/seed42

"$SPY" -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m causal_diagsafe.cli.train_v7_latent \
  --model "$MODEL" --release "$RELEASE" --bank "$BANK" \
  --output "$RUN/sft" --phase sft --epochs 2

"$SPY" -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m causal_diagsafe.cli.train_v7_latent \
  --model "$MODEL" --release "$RELEASE" --bank "$BANK" \
  --output "$RUN/latent" --phase latent --epochs 2 --adapter "$RUN/sft/final"
```

`--adapter` 是阶段初始化，阶段切换时新建 optimizer/scheduler，不是恢复中断位置的 optimizer/RNG。代码没有声称任意断点都可无损恢复。每轮开始刷新冻结旧 LoRA，完成提议与无梯度重算后才建立学生反向图。

`run_settings.json` 保存实际参数、根计量和数据范围；`training_result.json` 记录完成的步数；`final/` 是阶段末 adapter。完整 epoch 的 checkpoint 全部保留，不能把 `final` 文件名理解成已完成 dev 选择。

<a id="evaluation"></a>
## 4. 自主推理与评测

### 4.1 输入与推理入口

输入 JSONL 的每行可以是 Sample，或仅包含完整 `model_input` 可见字段的对象。模型只接收 `query/images/ocr_text/policy_deck/response_candidate`；如有 Sample 的其他字段，不会将 gold、来源与构组信息传给模型。图片 URI 必须对应实际可读取的媒体。

```bash
.venv-train/bin/python -m causal_diagsafe.cli.infer_v7 \
  --model /data/models/Qwen3-VL-4B-Instruct \
  --adapter /data/diagsafe/runs/v7/seed42/latent/final \
  --inputs /data/inputs.jsonl --output /data/predictions.jsonl
```

输出逐条保存 `sample_id` 与 trace。无效 W/L 保留错误；不会把生成失败替换成成功的 review。证据错误另外保留，不把动作可解析当成证据正确。

### 4.2 固定自然 dev

```bash
.venv-train/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m causal_diagsafe.cli.evaluate_v7 \
  --model /data/models/Qwen3-VL-4B-Instruct \
  --adapter /data/diagsafe/runs/v7/seed42/latent/final \
  --release /data/diagsafe/releases/visual_refusal_fill_20261015 \
  --output /data/diagsafe/runs/v7/dev
```

该入口只读取 `natural_request/dev.jsonl` 和 `natural_response/dev.jsonl`，与训练内的逐轮评测相同，默认不读取隐藏测试。当前固定分母为请求 299、回答 200。

输出包括逐条预测、各任务四类混淆矩阵、accuracy、macro-F1、NLL、均衡 NLL、适用字段准确率和过程计数。无效结果留在分母，动作概率为零，计算对数沿用 1e-300 下限；不作为 review 正确样本计数。

选择量是先按审核任务、再按观测动作类别平均的 NLL。它与普通 accuracy 或两任务混合总体 NLL 不同。空事实、空规则链接、非空证据和证据错误均报告，不据此临时修改选择规则。

本版没有将自然 dev 入口包装成完整的机制/公开 benchmark 评分器，也没有将旧版本结果标为 v7。公开系统评测需接入当前自主审核接口并沿用相应 benchmark 的输入与评分协议；这些结果与隐藏测试不能用于调节本轮数据量或方法。

<a id="implementation"></a>
## 5. 代码范围

| 文件 | 作用 |
| --- | --- |
| `audit_v7.py` | 来源目录、事实与规则物化、节点读取权限和动作输入 |
| `training/v7_latent_data.py` | 已有监督与候选视图、根权重、证据可见性 |
| `training/v7_latent_objective.py` | 固定原子、随机补集、先验/后验权重与梯度代理 |
| `training/v7_sequence.py` | 官方 processor、真实采样概率和语法归一化序列概率 |
| `training/v7_objective.py` | 冻结轮首 LoRA；保留共享序列读出的 CE 辅助函数 |
| `training/v7_latent_trainer.py` | HF Trainer、DDP、SFT、潜变量更新和有效关系 |
| `inference_v7.py` | 不接收标注对象的自主分批推理 |
| `evaluation_v7.py` | 固定自然 dev、失败分母与选择量 |
| `cli/build_v7_latent_bank.py` | train-only vLLM 弱参考生成 |
| `cli/train_v7_latent.py` | 两阶段主训练入口 |
| `cli/infer_v7.py`、`cli/evaluate_v7.py` | 推理和独立自然 dev 命令 |

共享的 `audit_v4/v5`、`training/v4_data/v4_events/v4_losses/v4_sequence/v4_visual_cache/v5_data/v5_matching` 沿用数据契约、根归并、分批、定位与条件匹配实现。它们不是另一套要接续执行的训练流程。V7 不依赖旧 `v7_trainer/v7_data` 的人工逐节点纠错，也不需要旧 v6 训练器。

本次仓库发布仅保留方法必需代码、启动入口、依赖及文档；内部小实验、诊断报告、自动参考、完整数据、模型权重与账号凭证不上传。
