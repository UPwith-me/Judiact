# CausalDiagSafe-VL v0 历史说明

本文保留 2026-10-12 更新首页前的本机 README 内容，其中的架构、数据路径和运行安排属于早期版本。当前方法与数据入口见 [项目首页](../../README.md)。

本目录是和历史 `diagsafe/` 隔离的新算法工程骨架，目标服务于已冻结的新方向：

```text
CausalDiagSafe-VL = CSSB + IICT + EGCR
```

- `CSSB`: Causal Stage-State Bottleneck，用 `R/E/F` 显式阶段状态替代旧的生成式 `S1-S5 pass/fail` trace。
- `IICT`: Input-side Intervention Consistency Training，用输入侧干预训练目标阶段翻转、非目标阶段稳定和决策边界一致性。
- `EGCR`: Evidence-Gated Causal Router，一个可训练、可校准、可消融、带单调约束的因果决策层，输出 `allow/warn/review/refuse`。

## 当前交付范围

本版重点完成可审计、可单测、可接入服务器训练的代码基础：

- 新 schema 与 parser：支持 `R/E/F`、intervention group、policy perturbation、legacy JSON 输出兼容解析。
- Stage 0 / benchmark evaluator：计算 final decision、over-refusal、harmful allow、bucket、stage collapse、R/E/F 指标。
- Strict group audit：计算 boundary success、target-stage flip、non-target invariance、policy leakage。
- CSSB heads：从 sentinel hidden states 训练 `R/E/F`、intent、policy、benign、decision heads。
- EGCR router：可学习单调 evidence-gated router，不是手写 if-else。
- IICT losses：target/evidence flip、mutable non-target invariance、decision flip/invariance、benign guard、calibration。
- CLI：preflight、prediction evaluation、group audit、cached hidden-state probe 训练。
- 正式数据适配：支持 `causal_diagsafe_unified_v1` 的 `model_input / targets / metadata`
  嵌套结构，并可核对同步的 `*_full_audit.jsonl` 与全局 group registry。

## 不做的事

- 不在本地重复下载大数据或模型。
- 不把 300-500 causal groups 定位为 benchmark 主贡献。
- 不继续扩大旧 OPD / E24 生成式 JSON margin 路线。
- 不把旧 `stage_labels: pass/fail/uncertain` 当作因果诊断证据。
- 不再以旧 E20 作为初始化；旧 E20 使用旧数据且其训练路径未读取图像像素。

## 快速自检

在 `F:\digsafe` 下执行：

```powershell
python -m pytest causal_diagsafe_vl_20260630\tests
```

## CLI 示例

### 0. ebcloud 正式发布核验

服务器上的联合训练输入已经固定在：

```text
/data02/jipei/datasets/causal_diagsafe_v0/combined_manual_gold_v1/splits
```

模型输入始终读取 `model_input.query` 和 `model_input.images`；S4/S5 干预还必须分别读取
`model_input.policy_text` 和 `model_input.response_candidate`。`*_full_audit.jsonl` 中的人工证据、
role、target stage、来源和审核备注仅用于监督、审计与 provenance，不得拼进 prompt。运行以下
CPU 检查不会读取图像像素或启动训练：

```bash
python -m causal_diagsafe.cli.inspect_release \
  --config configs/ebcloud_causal_gold_v1.json \
  --out-dir reports/release_inspection
```

联合集为 `2524/350/350`。当前 `dev/test` 均来自 cross-modal 冻结集；独立 text-only
发布只有 `1704` 条 train，dev/test 为空。因此 text-only 泛化评测仍应使用冻结的 XSTest、
OR-Bench 等外部集合，不能拿联合集 dev/test 代替。

### 1. 数据 preflight

```powershell
python -m causal_diagsafe.cli.preflight ^
  --input path\to\records.jsonl ^
  --out-dir path\to\preflight_report
```

### 2. 评测预测结果

```powershell
python -m causal_diagsafe.cli.evaluate_predictions ^
  --gold path\to\records.jsonl ^
  --pred path\to\predictions.jsonl ^
  --out-dir path\to\eval_report
```

### 3. 成组干预审计

```powershell
python -m causal_diagsafe.cli.audit_groups ^
  --gold path\to\causal_groups.jsonl ^
  --pred path\to\predictions.jsonl ^
  --out-dir path\to\group_audit
```

### 4. H1 cached hidden-state probe

`train_probe` 不直接依赖 Qwen 推理。服务器端先把 sentinel hidden states 导出为 `.pt`，再训练 heads/router：

```powershell
python -m causal_diagsafe.cli.train_probe ^
  --train-cache path\to\train_hidden.pt ^
  --dev-cache path\to\dev_hidden.pt ^
  --out-dir path\to\h1_probe ^
  --hidden-size 2048 ^
  --head-kind mlp ^
  --epochs 5
```

缓存字段见 [docs/CACHED_HIDDEN_FORMAT.md](../CACHED_HIDDEN_FORMAT.md)。

### 5. H1c gold-state oracle router

```powershell
python -m causal_diagsafe.cli.gold_router_eval ^
  --train path\to\causal_train.jsonl ^
  --eval path\to\causal_dev.jsonl ^
  --out-dir path\to\h1c_gold_router
```

如果 gold-state router 都无法优于 E24，说明 schema、R/E/F 标签或 EGCR 逻辑存在问题，不应进入 H2 LoRA。

### 6. 新人工数据上的 E20-Clean 重建

先导出 B1-B5 共用的干净训练清单：

```bash
python -m causal_diagsafe.cli.prepare_clean_reproduction \
  --config configs/ebcloud_causal_gold_v1.json \
  --out-dir /data04/datasets/jipei/datasets/causal_diagsafe_v0/algorithm_cache_v1/e20_clean_reproduction
```

导出器会保留原始 `source_role`，按最终人工决策生成训练边界角色，并禁止把 role、stage、
source 或人工 evidence 拼入 prompt。E20 风格的 512 个 triplet 明确标为 calibration triplet，
不能计入 causal group 指标。

B1 在服务器上的短命令：

```bash
scripts/ebcloud_b1_clean_joint_sft.sh preflight
scripts/ebcloud_b1_clean_joint_sft.sh train
```

`preflight` 不占 GPU，会检查全部 2524 条输入、820 张训练图像和每条 tokenized target；
`train` 只申请一张空闲卡，并持续输出 step、显存和 ETA。当前进度已停在 `train` 执行前。
