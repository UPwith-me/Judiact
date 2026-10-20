# DiagSafe-VL v3 方法与训练设置（历史）

本文保留 2026-10-12 的 v3 实验设置，文件名保留以便历史链接继续使用。**当前 v7 的方法、数据入口和运行命令见 [v7 方法与运行说明](DIAGSAFE_V7_GUIDE_ZH.md)**；下文参数与路径属于 v3，不能直接当作 v7 配置。历史方法定义见 [v3 方法](DIAGSAFE_V3_METHOD_EXECUTION_PLAN_20261008_ZH.md)。

## 1. 学习任务

主干为 `Qwen/Qwen3-VL-4B-Instruct`。模型通过同一预训练语言输出路径判断请求动作、S1-S4 R/E/F、意图、实际 policy ID/action、安全回答行为和证据位置。回答审核独立判断发布动作、S5 E/F、回答行为与可见文本证据。

输入只取 `model_input` 中的 query、图片、OCR-only 转写、当前 deck 和实际候选回答。来源、构组信息、人工参考 OCR、caption 和 gold 留在输入之外。

当前 `primary_stage` 和 `harm_categories` 保留为分层信息。模型的动作和其他分类字段各自读取原始输入，推理问题之间保持独立。

## 2. 训练数据与归一化

训练使用冻结发布 `v3_train_dev_20261008`，包括 2648 个请求根、1173 个回答上下文和 36192 个原生辅助根，总计 40013 个训练单位。

请求根内的多个成员平均，回答上下文内的多条候选平均，原生辅助按自身根平均。每个成员再平均有效分类与正定位字段，定位答案先平均 token loss。请求、回答和辅助的全轮绝对监督权重各为 1/3。

有效关系按根或上下文簇、关系组、字段依次平均。关系批的额外成员前向只计算关系目标，绝对监督保持原来的根权重。普通来源标签只监督其实际支持的字段。

完整 epoch 遍历固定清单。seed 确定根顺序和分层排布，三轮使用同一份清单，两个配对模型使用相同顺序。每轮 626 次 optimizer 更新，三轮 1878 次。

## 3. 模型参数

| 项目 | 当前设置 |
| --- | --- |
| 初始化 | 同一公开 Qwen3-VL-4B-Instruct 基础模型 |
| 输出层 | 原 lm_head，权重冻结，参与损失反传路径 |
| 可训练参数 | 语言 attention q/k/v/o 的 LoRA |
| LoRA | r=16，alpha=32，dropout=0.05，bias=none |
| 初始化方式 | PEFT 默认无增量初始化 |
| 基础冻结参数 | 视觉塔、merger、embedding、lm_head 及其他基础权重 |
| 精度 | BF16，TF32 开启，SDPA |
| 训练 cache | KV cache 关闭 |
| 梯度检查点 | 使用官方模型的 gradient checkpointing |
| 图像预算 | 官方 processor，像素预算下界 65536、上界 1003520 |
| 文本处理 | 保留完整可见文本，processor 使用 truncation=false |

视觉预算取自当前 processor 的实际 size 配置。长 query、多图、长规则或长候选的计算成本按真实 token 长度处理，静默删除这些字段会改变监督对象。

## 4. 优化与更新

| 项目 | 当前设置 |
| --- | --- |
| 框架 | Transformers Trainer、PEFT、Accelerate、PyTorch |
| 优化器 | AdamW |
| 学习率 | 2e-5 |
| beta | 0.9 / 0.999 |
| epsilon | 1e-8 |
| weight decay | 0.01 |
| grad clip | 1.0 |
| scheduler | 覆盖完整三轮的一次 cosine decay |
| warmup | 总 optimizer steps 的 3% |
| epochs | 3 |
| 当前配对 seed | 42 |
| GPU | 实际实验资源为 4 张 A800-SXM4-80GB |
| 名义有效批量 | 每次更新约 64 个根或请求上下文 |
| 每卡 Trainer batch | 一个预先排布的 packet |
| gradient accumulation steps | 1 |
| 视图 microbatch | 默认 4，定位任务按字段计算 |

一个 packet 内包含多个根、分类视图、定位视图和关系任务。各小批次顺序反传，最后一次真实反传统一同步 DDP 梯度，Trainer 完成一次优化更新。组成员保持在同一设备。

一个完整 epoch 包含 153603 个分类视图、23424 个正定位字段视图和 13797 个关系前向视图。这些视图共用原有根和成员。

## 5. 配对方法

`v3` 使用语义监督及 lambda=1 的关系项。`no_relation` 使用同一语义监督和 lambda=0，保留相同关系前向、成员曝光、dropout 计算路径及更新次数。

两个模型分别从同一基础模型开始，使用一个共享 LoRA 和一个连续三轮优化计划。数据中的请求、回答、辅助及关系同时参与训练。

lambda 和学习率为这轮固定研究设置，其效果需要实验支持。其他 seed 或数据规模的比较应沿用相同的输入、任务权重、主干及选择协议。

## 6. 轮末保存与选择

每个完整 epoch 保存 checkpoint 和恢复状态。正式选择比较 epoch 1、2、3，选择量为

\[
NLL_{\mathrm{select}}
=\tfrac12\overline{NLL}_{\mathrm{request}}
+\tfrac12\overline{NLL}_{\mathrm{response}}.
\]

自然 dev 的请求和回答分母分别为 299 与 200。四分类概率来自当前合法候选集合内的归一化，选择使用原始概率。机制、policy 和证据诊断单独保留。

中间恢复点用于中断恢复，恢复同一 optimizer、scheduler 和训练位置。新增轮数或修改参数需要记录新的接续设置。公开集合和隐藏测试保留评测用途。

## 7. 代码与数据路径

| 组件 | 路径 |
| --- | --- |
| schema | `data_contract/v1/` |
| 模型 | `causal_diagsafe/models/diagsafe_v3.py` |
| 清单和关系任务 | `causal_diagsafe/training/v3_data.py` |
| 固定审核问题 | `causal_diagsafe/training/v3_views.py` |
| loss | `causal_diagsafe/training/v3_losses.py` |
| Trainer | `causal_diagsafe/training/v3_trainer.py` |
| 训练入口 | `causal_diagsafe/cli/train_v3.py` |
| 推理 | `causal_diagsafe/inference_v3.py` |
| 诊断统计 | `causal_diagsafe/evaluation_v3.py` |

当前实验目录在 `/disk/u20110017/diagsafe_v1`。冻结数据位于 `releases/v3_train_dev_20261008`，基础模型位于 `raw/models/Qwen3-VL-4B-Instruct`。根归并使用的两个内部文件通过 `--reviews` 指定。

训练入口复制基础模型的 `download_source.json` 到运行设置，用于记录实际来源 revision。媒体 URI 需要能在训练机器上读取。数据和模型目录的内容由实验服务器单独提供。

## 8. 最小运行方式

现有训练环境为 Python 3.11.9、PyTorch 2.7.1+cu118、torchvision 0.22.1+cu118、Transformers 4.57.1、PEFT 0.17.1、Accelerate 1.10.1 和 datasets 4.2.0。直接依赖见 [train-v3.txt](../requirements/train-v3.txt)。下面的安装命令用于新建环境。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements/train-v3.txt
```

在仓库根目录运行以下入口，路径替换为机器上的实际目录。命令会创建新的训练输出，复核目录需包含 `confirmed_core_image_aliases.jsonl` 和 `confirmed_image_root_families.jsonl`。

```bash
export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m causal_diagsafe.cli.train_v3 \
    --model /path/to/Qwen3-VL-4B-Instruct \
    --release /path/to/v3_train_dev_20261008 \
    --reviews /path/to/root_reviews \
    --output /path/to/new_run/v3 \
    --variant v3 --epochs 3 --seed 42 --base-dev
```

配对 SFT 使用相同参数与数据，`--variant no_relation`，输出指向独立目录。当前服务器连续运行脚本为 [ds_disk_semantic_v3.sh](../scripts/ds_disk_semantic_v3.sh)，其中的目录是本轮服务器布局。

公开任务适配复用 lmms-eval，外部模型接口和下载配置也随代码提供。相关脚本使用独立评测环境，额外依赖及第三方官方实现需在该环境准备。主训练依赖文件只描述 v3 训练栈。
