# BERT4Rec 启动命令

在项目目录执行；`--source` 指向已完成数据预处理的实验目录。

三个数据集各跑 1 次：

```bash
.venv/bin/python -u rerun_bert.py \
  --source runs/paper --output runs/bert_seed100 \
  --datasets Beauty Toys Sports --seeds 100 \
  --workers 3 --threads-per-worker 1 --tf32
```

各跑 3 次时，改为 `--seeds 100 101 102`，并使用新的输出目录。

训练记录实时显示在当前终端，每次训练结束立即测试。输出位置：

- 测试结果：`runs/bert_seed100/experiments/<数据集>/bert4rec/seed_100/report.json`。
- 汇总表：`runs/bert_seed100/summary/tables.md`。
- 任务进度与日志：`runs/bert_seed100/parallel/`。

中断后重新执行同一命令，会跳过已完成任务。
