# Choice

在 Linux 下进入项目目录，依次执行以下步骤。训练记录实时显示在当前终端。

## 0. 下载并配置环境

使用 Python 3.10—3.12：

```bash
sudo apt-get install python3-venv git git-lfs build-essential
git lfs install
git clone https://github.com/HMSTerror/Choice.git
cd Choice
git lfs pull
bash setup.sh
```

## 1. 数据预处理

```bash
bash run.sh --stage prepare --output runs/paper
```

读取 `dataset/`，生成训练输入和语义特征，保存在 `runs/paper/data/<数据集>/`。

## 2. 训练 SASRec 并生成候选集

```bash
bash run.sh --stage candidates --workers 4 --tf32 --output runs/paper
```

训练、测试 SASRec，生成候选集和结构文件，保存在 `runs/paper/candidates/<数据集>/seed_<种子>/`。

## 3. 运行全部实验

```bash
bash run.sh --stage experiments --workers 4 --tf32 --output runs/paper
```

每个模型训练完立即测试。测试结果保存在 `runs/paper/experiments/<数据集>/<实验>/seed_<种子>/report.json`，汇总在 `runs/paper/summary/tables.md`。

## 4. 一键运行完整流程

环境配置完成后，也可以直接执行：

```bash
bash run.sh --stage all --workers 4 --tf32 --output runs/paper_all
```

默认运行 Steam、Beauty、Toys、Sports。只跑三个 Amazon 数据集时，加 `--datasets Beauty Toys Sports`；只跑主表时，加 `--groups main`。各实验分组的命令见 [实验启动命令](EXPERIMENTS.md)。

## 5. 单独重跑 BERT4Rec

复用第 1 步的数据，三个数据集各跑 1 次：

```bash
.venv/bin/python -u rerun_bert.py \
  --source runs/paper --output runs/bert_seed100 \
  --datasets Beauty Toys Sports --seeds 100 \
  --workers 3 --threads-per-worker 1 --tf32
```

结果汇总在 `runs/bert_seed100/summary/tables.md`。每个数据集跑 3 次时，改为 `--seeds 100 101 102` 并使用新的输出目录。

## 6. 查看进度和结果

训练记录直接显示在启动命令的窗口。并行任务进度在 `<输出目录>/parallel/status.json`，单任务日志在同级 `.log` 文件中。

重新生成全实验汇总：

```bash
bash run.sh --stage summary --output runs/paper
```

中断后重新执行原命令，会跳过已完成任务。重新开始一轮实验或修改配置时，换一个新的 `--output`。
