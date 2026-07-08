# CausalDiagSafe-VL v0 运行清单

## 1. 本地确认

```powershell
cd F:\digsafe\causal_diagsafe_vl_20260630
python -m pytest
python -m compileall -q causal_diagsafe
```

当前本地结果：

```text
13 passed
compileall passed
```

## 2. 服务器部署建议

建议上传整个目录到服务器系统盘，例如：

```text
/root/DiagSafe/causal_diagsafe_vl_20260630
```

数据、模型和 hidden cache 放到服务器数据盘或项目目录下的独立路径，不放在本地 Windows 工作区。

## 3. 下一步最小执行顺序

### Step A: preflight

先对当前训练/评测 JSONL 做 preflight：

```bash
python -m causal_diagsafe.cli.preflight \
  --input /path/to/records.jsonl \
  --out-dir /path/to/reports/preflight_v0
```

必须检查：

```text
validation_error_count
target_stage 分布
source-modality confound
R/E/F coverage
refuse 标签占比
```

### Step B: E20 历史预测重新审计

对已有 benchmark 预测重新跑新 evaluator：

```bash
python -m causal_diagsafe.cli.evaluate_predictions \
  --gold /path/to/gold.jsonl \
  --pred /path/to/e20_predictions.jsonl \
  --out-dir /path/to/reports/e20_new_audit
```

重点看：

```text
harmful_allow
over_refusal
stage all_clear_rate
legacy_stage_output_rate
bucket_metrics
```

### Step C: strict group audit

```bash
python -m causal_diagsafe.cli.audit_groups \
  --gold /path/to/causal_groups.jsonl \
  --pred /path/to/predictions.jsonl \
  --out-dir /path/to/reports/group_audit_v0
```

重点看：

```text
boundary_success_rate
target_stage_flip_rate
non_target_invariance_rate
policy_leakage_rate
by_target_stage
```

### Step D: source-only baseline

```bash
python -m causal_diagsafe.cli.source_only_baseline \
  --train /path/to/train.jsonl \
  --eval /path/to/dev.jsonl \
  --out-dir /path/to/reports/source_only_baseline
```

如果 source-only 结果接近 CSSB/EGCR，说明新算法仍可能被 source、modality、policy shortcut 污染。

### Step E: H1 hidden-state probe

先跑 H1c gold-state oracle router：

```bash
python -m causal_diagsafe.cli.gold_router_eval \
  --train /path/to/causal_train.jsonl \
  --eval /path/to/causal_dev.jsonl \
  --out-dir /path/to/runs/h1c_gold_router
```

如果 H1c 失败，说明 schema、R/E/F 标签或 EGCR 逻辑有问题，不应继续 H2。

然后服务器端导出 Qwen sentinel hidden states，再跑 H1a/H1b：

```bash
python -m causal_diagsafe.cli.train_probe \
  --train-cache /path/to/train_hidden.pt \
  --dev-cache /path/to/dev_hidden.pt \
  --out-dir /path/to/runs/h1_probe_linear \
  --hidden-size 2048 \
  --head-kind linear \
  --epochs 5

python -m causal_diagsafe.cli.train_probe \
  --train-cache /path/to/train_hidden.pt \
  --dev-cache /path/to/dev_hidden.pt \
  --out-dir /path/to/runs/h1_probe_mlp \
  --hidden-size 2048 \
  --head-kind mlp \
  --epochs 5
```

## 4. Gate

进入 H2 LoRA 前，至少满足：

```text
H1c gold-state oracle router 可明显超过 E24 boundary_success。
source-only baseline 明显弱于 H1a/H1b。
H1a 或 H1b 在 target-stage F/R/E 上有非随机信号。
H1 不显著恶化 XSTest-style over-refusal。
```

否则不要直接开 H2，先修 schema、R/E/F 标签和 causal groups。

## 5. 记录要求

每次运行记录：

```text
实验名
代码目录 commit 或文件快照
数据路径
模型初始化
GPU id
PID
输出目录
核心指标
失败样本路径
下一步决策
```
