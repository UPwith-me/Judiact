# CausalDiagSafe-VL v0

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

## 不做的事

- 不在本地重复下载大数据或模型。
- 不把 300-500 causal groups 定位为 benchmark 主贡献。
- 不继续扩大旧 OPD / E24 生成式 JSON margin 路线。
- 不把旧 `stage_labels: pass/fail/uncertain` 当作因果诊断证据。

## 快速自检

在 `F:\digsafe` 下执行：

```powershell
python -m pytest causal_diagsafe_vl_20260630\tests
```

## CLI 示例

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

缓存字段见 [docs/CACHED_HIDDEN_FORMAT.md](docs/CACHED_HIDDEN_FORMAT.md)。

### 5. H1c gold-state oracle router

```powershell
python -m causal_diagsafe.cli.gold_router_eval ^
  --train path\to\causal_train.jsonl ^
  --eval path\to\causal_dev.jsonl ^
  --out-dir path\to\h1c_gold_router
```

如果 gold-state router 都无法优于 E24，说明 schema、R/E/F 标签或 EGCR 逻辑存在问题，不应进入 H2 LoRA。
