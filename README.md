# JudiAct

**Guarding with Judgment: Learning to Intervene for Multimodal Safety**

[Training data](https://huggingface.co/datasets/Devin6666/JudiAct-Data) | [JudiBench](https://huggingface.co/datasets/Devin6666/JudiBench) | [Training and reproduction](docs/reproduction.md) | [Citation](CITATION.cff)

JudiAct audits text and image requests, as well as candidate answers. It checks three short items (source fact, condition, and rule link) against the original inputs and supplied rules before assigning an action.

| Action | Request | Candidate answer |
| --- | --- | --- |
| `allow` | Help as asked. | Deliver as written. |
| `warn` | Help with a necessary warning or restriction. | Revise while preserving permissible help. |
| `review` | Clarify or verify a condition needed to proceed. | Resolve the condition before deciding whether to deliver. |
| `refuse` | Withhold prohibited help. | Withhold an answer that provides prohibited help. |

## Install

Use Python 3.11 on Linux with a CUDA GPU that supports BF16. First install the [CUDA build of PyTorch 2.8.0 and torchvision 0.23.0](https://pytorch.org/get-started/previous-versions/#v280) compatible with your driver.

```bash
git clone https://github.com/UPwith-me/Judiact.git
cd Judiact
python -m pip install -e '.[audit]'
```

For training and the paper's FlashAttention recipe, see [Training and reproduction](docs/reproduction.md). The single-input example below uses PyTorch SDPA instead.

## Run one audit

Use a local [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) snapshot and a compatible JudiAct LoRA checkpoint. The adapter together with its audit configuration (normally `judiact_config.json`) is required. This repository does not bundle the weights, and inference does not download them or start a service.

```bash
export MODEL=/path/to/Qwen3-VL-4B-Instruct
export ADAPTER=/path/to/judiact/checkpoint
```

```python
import os

from judiact.guard import JudiActGuard

guard = JudiActGuard(
    os.environ["MODEL"],
    os.environ["ADAPTER"],
    device="cuda:0",
    attention_backend="sdpa",
    include_evidence=False,
)

case = {
    "query": "Help me organize my reading notes by topic.",
    "images": [],
    "ocr_text": None,
    "policy_deck": [
        {"policy_id": "P_BENIGN", "rule_text": "Allow ordinary reading and writing assistance."}
    ],
    "response_candidate": None,
}

result = guard.predict_batch([case])[0]
print(result["status"], result.get("action"))
```

Supply the complete applicable rule deck for your use case. For an image, pass a local path as `{"uri": "/path/to/image.png"}` in `images`; for answer auditing, set `response_candidate` to the actual answer. Do not pass reference labels or annotation metadata.

On a valid audit, `action_probabilities` gives the four action scores and `complete_audit` contains diagnostic values and the generated short items. Set `include_evidence=True` to also decode supporting source spans and image boxes after classification. Short items and locations are model predictions, not independently verified evidence.

## Evaluate and reproduce

| Goal | Entry point | Inputs |
| --- | --- | --- |
| Evaluate on JudiBench | [JudiBench guide](docs/reproduction.md#evaluate-on-judibench) | Base model, a compatible adapter, and a local JudiBench snapshot |
| Train the two-stage guard | [Training guide](docs/reproduction.md#train-judiact) | Base model and the prepared train/dev bundle, including its original supervision sidecars |
| Evaluate delivered answers or binary transfer | [Public benchmark guide](docs/public-evaluation.md) | Prepared benchmark manifests, the evaluation weights, and upstream rubric files |

JudiBench's paper classification view contains **5,274 audits** (2,371 requests and 2,903 candidate answers). The full release adds members for controlled comparisons. The evaluation guide explains how to score the paper split without mixing those members into its denominator.

[JudiAct-Data](https://huggingface.co/datasets/Devin6666/JudiAct-Data) contains 20,126 main training audits and source-native auxiliary data. Its Hugging Face export is not the trainer's on-disk input bundle. See the [required training inputs](docs/reproduction.md#training-bundle) before attempting exact reproduction.

## Citation and license

Please cite *Guarding with Judgment: Learning to Intervene for Multimodal Safety*. The author list and software citation are in [CITATION.cff](CITATION.cff).

The original JudiAct code is licensed under [Apache-2.0](LICENSE). The base models, adapters, datasets, media, and upstream evaluation materials are separate artifacts; consult their own license and access terms.
