# BERT4Rec 官方实现核查与重跑

对照版本：[FeiSun/BERT4Rec @615eaf2](https://github.com/FeiSun/BERT4Rec/tree/615eaf2004abecda487a38d5b0c72f3dcfcae5b3)。本项目采用 PyTorch 移植，模型在 `bert4rec.py`，正式训练入口在 `baselines.py`。

## 核查与修改

| 项目 | 修改前 | 官方实现及此次修改 |
| --- | --- | --- |
| 遮盖数量 | 每个位置独立 Bernoulli | 随机排列位置后，固定选 `min(上限, max(1, round(有效长度 × 比例)))` 个 |
| 遮盖内容 | 80% MASK / 10% 随机 / 10% 不变 | `run_beauty.sh` 设置 `mask_prob=1.0`，选中位置全部 MASK |
| 训练样本 | 每轮重新随机遮盖一份 | 10 份固定遮盖副本，加 1 份只遮最后物品的样本 |
| 输入与位置 | 左侧 padding，位置表只有历史长度 + 1 | 右侧 padding，50 个位置；最后真实位置后追加 MASK |
| Transformer | 通用 PyTorch EncoderLayer | 独立 Q/K/V；head 维度缩放；双向 key padding mask；post-LayerNorm；FFN 中间激活后没有额外 dropout |
| GELU / 初始化 | erf GELU、普通正态；MHA 部分权重继承默认初始化 | 官方 tanh GELU；所有投影、嵌入采用 std=0.02、截断至 ±0.04 的正态 |
| 嵌入 / 输出 | 无 token type；只在真实物品上做 MLM 分类 | 加 token type 0 的嵌入；MLM 输出包含官方词表对应的 4 个特殊/空闲 ID |
| 验证 / 测试 | 全目录排名 | 真实目标 + 100 个按物品频率、无放回抽取的负例；排除已有历史和目标 |

原来的全目录排名比 101 候选排名更难，不能用它解释 Sports 指标“虚高”。此次修复的依据是实现差异，并不预设修复后指标下降。

模型对应官方 [`modeling.py`](https://github.com/FeiSun/BERT4Rec/blob/615eaf2004abecda487a38d5b0c72f3dcfcae5b3/modeling.py) 的 `embedding_postprocessor`、`attention_layer`、`transformer_model`；预测头对应 [`run.py`](https://github.com/FeiSun/BERT4Rec/blob/615eaf2004abecda487a38d5b0c72f3dcfcae5b3/run.py) 的 `get_masked_lm_output`。遮盖与固定副本对应 [`gen_data_fin.py`](https://github.com/FeiSun/BERT4Rec/blob/615eaf2004abecda487a38d5b0c72f3dcfcae5b3/gen_data_fin.py)，具体参数以 [`run_beauty.sh`](https://github.com/FeiSun/BERT4Rec/blob/615eaf2004abecda487a38d5b0c72f3dcfcae5b3/run_beauty.sh) 和 Beauty64 JSON 为准。评估对应 `EvalHooks.after_run`，默认 `use_pop_random=True`、seed=12345。

已有的 BertAdam、无偏差修正的一二阶矩、epsilon=1e-6、排除 bias/LayerNorm 的 weight decay、100 步 warmup、400000 步线性衰减、梯度范数 5 保留。Beauty/Toys/Sports 使用 hidden64、2 层、2 heads、hidden dropout0.5、attention dropout0.2、mask比例0.6。Toys/Sports 没有作者公开的专属配置，继续标为 Beauty 配置移植。默认没有 epoch 上限，按验证 NDCG@10、patience50 个 epoch 早停，这是用户要求对原版固定步数训练的覆盖。

## 保留的数据协议差异

- 源文件提供历史最多 10 项与下一物品，没有完整用户 ID/历史。模型位置容量为 50，但不会虚构缺失的过去交互；每个输入实际最多 11 个有效 token。因此无法复现原始长历史的滑窗数据生成。
- 原作者脚本将验证交互合入训练，并以包含测试交互的 `user_test_data` 构造词频。工程保留现有训练/验证/测试划分，固定副本只来自训练行；负例频率只统计训练行的历史及目标出现次数，不读取验证/测试目标估计频率。窗口重复造成的词频也与原始逐用户交互次数不同。
- 负例只从训练出现过的物品抽取；正例即使未在训练出现也保留。过滤范围是源文件中提供的历史，无法恢复完整用户已见集合。负例是流行度采样，不是 SASRec 候选集。
- 本次没有安装 TensorFlow 1.x 或声称逐权重数值等价。检查脚本直接执行原仓库的纯 Python 遮盖函数与评估 Hook，对照遮盖 RNG、候选 ID 和 HR/NDCG。

## 仅重跑三个数据集的 BERT4Rec

环境与预处理完成后，指定原实验目录和一个新目录：

```bash
source .venv/bin/activate
python -u rerun_bert.py \
  --source runs/paper_newchoice_20261031 \
  --output runs/bert_official \
  --datasets Beauty Toys Sports --seeds 100 101 102 \
  --workers 3 --threads-per-worker 1 --tf32
```

共 9 次训练。实时日志显示在当前窗口，每次验证集选好模型后立即测试；权重只在内存保留，不保存 checkpoint。中断的单次训练从头开始，已完成的训练跳过。旧主表结果留在原目录。

- 每次结果：`runs/bert_official/experiments/<数据集>/bert4rec/seed_<种子>/report.json`
- 固定验证/测试候选：同级 `bert_val_candidates.npy`、`bert_test_candidates.npy`，第 0 列为真实目标。
- 候选配置：同级 `bert_<val/test>_evaluation.json`。
- 进度和单任务日志：`runs/bert_official/parallel/`。
- 均值与标准差：`runs/bert_official/summary/tables.md`。

报告标记 `evaluation_protocol=bert_popularity_100`，汇总按评估协议分组。这些新指标不能直接与其他模型已有的 `full_catalog` 指标比较或计算相对提升。

验证命令：

```bash
python -m unittest test_bert_official test_baselines test_training -v
git clone https://github.com/FeiSun/BERT4Rec.git /tmp/BERT4Rec
git -C /tmp/BERT4Rec checkout 615eaf2004abecda487a38d5b0c72f3dcfcae5b3
python check_bert_official.py --source /tmp/BERT4Rec
```
