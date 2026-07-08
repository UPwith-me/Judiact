# Cached Hidden-State 格式

`train_probe` 读取 `.pt` 文件。建议服务器端导出一个字典：

```python
{
    "stage_hidden": FloatTensor[N, 5, H],
    "decision_hidden": FloatTensor[N, H],
    "labels": {
        "decision": LongTensor[N],          # allow=0, warn=1, review=2, refuse=3
        "R": FloatTensor[N, 5],             # 0/1
        "E": LongTensor[N, 5],              # absent=0, present=1, uncertain=2
        "F": FloatTensor[N, 5],             # 0/1
        "intent": LongTensor[N],            # 见 constants.INTENT_CLASSES
        "policy_action": LongTensor[N],     # allow=0, warn=1, review=2, refuse=3
        "benign": FloatTensor[N]            # 0/1
    },
    "sample_ids": ["..."],
    "meta": {...}
}
```

可选字段：

```python
{
    "group": {
        "group_id": [...],
        "target_stage": LongTensor[N],
        "role": [...]
    },
    "weights": {
        "decision": FloatTensor[N],
        "R": FloatTensor[N, 5],
        "E": FloatTensor[N, 5],
        "F": FloatTensor[N, 5]
    }
}
```

H1a 对应 `head_kind=linear`。H1b 对应 `head_kind=mlp`。H1c 不需要模型 hidden states，直接使用 gold `R/E/F` 喂 EGCR，可通过 `evaluate_predictions` 或单独脚本构造。

