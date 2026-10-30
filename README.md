# Choice

Linux 下按以下三个阶段运行。默认覆盖 Steam、Beauty、Toys、Sports，主表九种模型，以及正文和技术附录中的实验。

## 一阶段：下载、配置环境和数据预处理

先安装 Python 3.10—3.12、Git、Git LFS 和 C++ 编译器（Ubuntu：`sudo apt-get install python3-venv git git-lfs build-essential`），然后执行：

```bash
git lfs install
git clone https://github.com/HMSTerror/Choice.git
cd Choice
git lfs pull
bash setup.sh
bash run.sh --stage prepare --output runs/paper
```

`setup.sh` 创建 `.venv`，安装 CUDA 12.4 版 PyTorch、Infomap、Faiss 等依赖，服务器需有支持 CUDA 12.4 的 NVIDIA 驱动；仅用 CPU 时执行 `bash setup.sh --cpu`。预处理将源数据整理为历史序列、长度和预测目标，保存在 `runs/paper/data/<数据集>/prepared/`；同时生成 PreferDiff 的固定特征和 DDSR 的 OPQ 语义编码，保存在同级的 `semantic.pt`。原始数据保留在 `dataset/`。

## 二阶段：训练 SASRec，生成全部候选集

```bash
bash run.sh --stage candidates --output runs/paper
```

依次训练四个数据集、六个种子（100—105）的 SASRec，测试其验证集最佳模型，再生成训练、验证和测试集的 top-100 候选及对照候选。每组保存在 `runs/paper/candidates/<数据集>/seed_<种子>/`：

- `sasrec_best.pt`、`sasrec_log.jsonl`、`report.json`：最佳 SASRec、每轮记录和测试结果。
- `candidates.pt`：原候选集；`candidates_popularity.pt`、`candidates_swap.pt`：对照候选集。
- `structure.pt`、`artifacts.json`：邻居、分区和结构说明。

该命令也会自动补做缺少的数据预处理。

## 三阶段：一键运行全部实验，逐个训练后测试

```bash
bash run.sh --stage experiments --output runs/paper
```

A100 等大显存 GPU 可直接并行运行全流程（环境安装后执行）：

```bash
bash run.sh --stage all --workers 4 --tf32 --output runs/paper_a100
```

四个独立训练进程共享 GPU；默认使用固定模型配置，不进行超参数搜索，附录效率计时独占 GPU。总日志在当前终端，各任务日志与进度保存在输出目录的 `parallel/`；重新执行同一命令会跳过完成任务。`--tf32` 开启 Ampere Tensor Core 的 TF32 矩阵乘法，训练批量与早停规则保持不变。

主表包括 SASRec、Caser、GRU4Rec、BERT4Rec、PreferDiff、DiffRec、DDSR、PreferGrow、ChoiceGrow；每个单元运行 3 次（seed 100–102），SASRec 复用第二阶段结果。因子消融和前向对照仍按清单使用 6 个训练种子。脚本使用论文明确参数及注明来源的作者配置，运行 2×2 消融、候选信息对照、Toys 前向对照，以及附录的初始化交叉、退出率审计与匹配重训、五分区重复、算子效率和 Steam 全测试集单条推理计时。每个正式模型训练完，立即加载验证最佳模型测试，再进入下一任务；融合系数和辅助负例权重仍按对应实验定义在验证集选择。

**训练轮次、损失、验证指标和测试结果实时显示在当前窗口。** 模型、日志和报告保存在 `runs/paper/experiments/<数据集>/<实验>/seed_<种子>/`。`report.json` 是测试结果；`runs/paper/summary/tables.md` 是汇总表，旁边有逐次结果 `runs.csv` 和按表分组的 JSON。效率结果在 `runs/paper/efficiency/`。

两个训练阶段默认均无总轮数上限，使用验证集早停；扩散模型正式测试使用三个固定采样流的均值。ChoiceGrow 开启固定候选质量门、保留新增邻居，主模型无辅助项；`guided_negatives` 是单独的论文对照实验。

关闭窗口或任务中断后，重新执行相同命令会复用已完成的任务；中断的单个训练会备份后从头重跑。修改配置后请使用新的 `--output`。所有运行产物在 `runs/` 中，均不上传 Git。

重新跑全套实验时使用新的输出目录，例如将三个命令中的 `runs/paper` 均换为 `runs/paper_all`，以免复用旧结果。实验设置在 `config.json` 和 `paper_suite.json`；默认有 468 个正式任务（含 12 个复用的主表 SASRec），并行全流程共 528 个任务，另有附录评测。主表种子由 `main_seeds` 设置；其他主实验种子仍由 `seeds` 设置。来源及不能由当前数据恢复的原论文协议差异见 [配置核查](docs/BASELINE_PAPER_CONFIGS.md)，各表对应见 [EXPERIMENTS.md](EXPERIMENTS.md)。

只检查流程时，可运行小样本版本（单种子、每个模型一轮，不用于效果结论）：

```bash
bash run.sh --stage all --smoke --output runs/smoke
```

只跑单个数据集时，在第二、三阶段都加 `--datasets Toys`；只跑主表时，第三阶段加 `--groups main`。所有阶段均支持 `--device cuda:0` 或 `--device cpu`。
