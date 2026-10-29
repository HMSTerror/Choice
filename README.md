# Choice

以下在 Windows PowerShell 中操作，以 Toys 数据集为例。

## 0. 下载与配置环境

安装 Python 3.11、Git 和 Git LFS，然后执行：

```powershell
git lfs install
git clone https://github.com/HMSTerror/Choice.git
cd Choice
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

以上安装 CUDA 12.4 版 PyTorch。仅使用 CPU 时，跳过安装 PyTorch 的单独命令，直接安装 `requirements.txt`。

## 1. 数据预处理

```powershell
.\run.cmd --dataset Toys --output runs\toys --stage prepare
```

读取 `dataset/ATG/` 中的 Toys 源数据，生成 `runs/toys/prepared/`，包含 `train.npz`、`val.npz`、`test.npz`、`train_target_counts.npy` 和 `metadata.json`。

## 2. 训练 SASRec，生成候选集

依次执行：

```powershell
.\run.cmd --dataset Toys --output runs\toys --stage proposer
.\run.cmd --dataset Toys --output runs\toys --stage artifacts
```

第一条训练 SASRec，第二条加载最佳 SASRec，为训练、验证和测试样本生成 top-100 候选，并构建邻居和分区。产物均在 `runs/toys/`：

- `sasrec_best.pt`：最佳 SASRec 模型。
- `sasrec_log.jsonl`：SASRec 每轮训练记录。
- `candidates.pt`：三个划分的候选集。
- `structure.pt`、`artifacts.json`：邻居、分区等结构及配置信息。

## 3. 训练 Choice 主模型

```powershell
.\run.cmd --dataset Toys --output runs\toys --stage train
```

使用预处理数据和候选集训练，保留邻居、开启固定候选质量门，不使用辅助项。生成 `runs/toys/choice_best.pt`、`choice_log.jsonl` 和 `training.json`，分别保存最佳模型、每轮训练记录和训练汇总。

## 4. 测试最佳模型

```powershell
.\run.cmd --dataset Toys --output runs\toys --stage evaluate
```

加载最佳 Choice 模型，在测试集评估。结果显示在当前窗口，并保存到 `runs/toys/report.json`，包含 HR/NDCG@5、@10。

## 一条命令运行完整主流程

完成第 0 步后，也可以直接用下面的命令自动执行第 1—4 步：

```powershell
.\run.cmd --dataset Toys --output runs\toys_full
```

**阶段、训练轮次、损失、验证指标和最佳轮次会实时显示在运行命令的当前窗口中。** 训练记录同时保存到输出目录中的 `sasrec_log.jsonl` 和 `choice_log.jsonl`。

默认自动选择 GPU，无 GPU 时使用 CPU。SASRec 和 Choice 均不设总轮数上限，分别在连续 10 轮、50 轮无验证改善后早停；Choice 每 10 轮验证一次。

分步运行时保持相同的 `--output`；重新训练请换一个新的输出目录。所有生成产物位于 `runs/`，不会被 Git 上传。更换数据集时修改 `--dataset`，可选 `Toys`、`Sports`、`Beauty`、`Steam`、`ML1M`。
