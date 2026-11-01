# 实验启动命令

完成 [README](README.md) 的环境配置和候选集生成后，在项目目录执行。所有命令使用同一个 `--output runs/paper`。

## 全部实验

```bash
bash run.sh --stage experiments --workers 4 --tf32 --output runs/paper
```

## 按分组运行

将 `<分组>` 替换为下表中的值：

```bash
bash run.sh --stage experiments --groups <分组> --workers 4 --tf32 --output runs/paper
```

| 实验 | 分组 |
| --- | --- |
| 主表 | `main` |
| 结构消融 | `factorial` |
| 辅助项与候选对照 | `auxiliary` |
| Toys 前向过程对照 | `forward` |
| 退出率匹配重训 | `matched_exit` |
| 初始化交叉 | `initialization` |
| 分区重复 | `partitions` |
| 效率测试（先完成相关模型训练） | `efficiency` |

多个分组可一起指定，例如 `--groups main factorial`。只跑三个 Amazon 数据集时，加 `--datasets Beauty Toys Sports`。

## 查看结果

```bash
bash run.sh --stage summary --output runs/paper
```

训练和测试记录实时显示在终端。单次结果在 `runs/paper/experiments/`，进度与日志在 `runs/paper/parallel/`，汇总表在 `runs/paper/summary/tables.md`。

单独重跑 BERT4Rec 的命令见 [BERT4Rec 启动命令](docs/BERT4REC_ALIGNMENT.md)。
