# CausalDiagSafe-VL v0 算法实现说明

## 1. 算法对象

本工程把论文算法对象固定为：

```text
CausalDiagSafe-VL = CSSB + IICT + EGCR
```

`CSSB` 是阶段状态瓶颈。模型不再自由生成 `S1-S5 pass/fail` 作为主要监督，而是从 sentinel hidden states 预测：

```text
R_k: 第 k 阶段能力是否 required
E_k: 第 k 阶段证据是否 absent / present / uncertain
F_k: 第 k 阶段风险条件是否 activated
```

`IICT` 是输入侧干预一致性训练。它要求：

```text
目标阶段随干预翻转。
非目标阶段在 mutable mask 允许范围内保持稳定。
最终决策只在跨越 policy boundary 时翻转。
```

`EGCR` 是证据门控因果 router。它用 `R/E/F`、intent、policy、benign、execution risk 生成最终决策，并对 refuse logit 施加单调非负约束。

## 2. 为什么这不是 benchmark 主线

300-500 条 causal groups 在这里被视为算法监督与校准信号：

```text
causal-train: 训练 CSSB heads 和 IICT
causal-calibration: 校准 EGCR 阈值和温度
causal-hidden: 只用于最终验证，不包装成新 benchmark
```

主贡献是算法层，而不是数据集本身。

## 3. 当前必须保留的失败审计

每次候选模型评测至少报告：

```text
final decision accuracy / macro-F1
harmful allow
under-refusal
over-refusal
strict group boundary success
target-stage flip
non-target invariance
stage all-clear / all-risk / multi-risk collapse
policy leakage under wrong/missing/decoy policy
cross-modal / OCR / meme / text-only / policy-boundary 分桶
```

## 4. 旧路线的定位

E20/E22/E24 保留为 baseline、negative finding 和消融参照：

```text
E20: 当前最稳 final-decision-calibrated adapter
E22: targeted safety clip，unsafe refusal 增强但过拒上升
E24: group contrast，boundary 有提升但 stage collapse 未解决
```

新代码不再把旧生成式 `stage_labels` 当作因果诊断目标。

