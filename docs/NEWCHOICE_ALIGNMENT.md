# NewChoice 核心实现对齐（2026-10-31）

来源为本机 `NewChoice/model/transformer.py`、`NewChoice/losses.py`、
`NewChoice/configs/model/small.yaml` 及 `pipeline_best_combo.json`。

## 模型

使用 `DDiTBlock_UCUR`、`DDitFinalLayer4Rec` 的核心计算和初始化顺序：
旋转位置编码、显式 padding/因果掩码、按 head 维度缩放的注意力、
Conv1d 前馈、单次前馈输出 dropout、三个 scale-only LayerNorm、
256 维正弦时间输入和 GELU 时间映射、条件拼接线性输出层。
默认一个历史块；条件维度等于 hidden_size。共享输入/输出物品 embedding。

原 Choice 的 HistoryEncoder + Linear 不是完全不同的模型家族；其
注意力、残差和线性读出大体对应原版。实际差异包含前馈 ReLU 后额外的
dropout、Conv1d 改 Linear 及模块创建顺序导致的初始化差异。
旧版有界锚定分数和“先引导、后有界化”的运算已经存在。
现在明确保留 `oricosChoice`：`cap*tanh((logit[y]-logit[x])/temperature/cap)`，
自环分数强制为零，默认 cap=4、temperature=1。

主模型不移植 best_combo 中 auxiliary_rank_weight=0.1 的辅助损失，
不引入 item bias、额外读出或 untied embeddings；用户要求的新增邻居仍保留。
因此这次修订不能保证重现含辅助项的 best_combo 指标。

## 训练协议

ChoiceGrow 及其同系列结构/候选消融默认 `patience=100`，按 epoch 计数，
每 10 轮验证，无总 epoch 上限，AdamW 恒定学习率。
离散时间网格仍是 linspace(1, .001, 21)，但按
`p_i = .1/21 + .9*dsigma_i/sum(dsigma)` 抽样，
CTMC score entropy 乘 `dsigma_i/(21*p_i)`。
10% 均匀分量保证全支撑，修正保持原均匀时间网格目标的期望及梯度。
独立辅助对照损失不乘时间重要性修正。

PreferGrow 及 guided_negatives 继续使用 uniform 时间抽样、patience=50；
共用的新去噪器也需要重新训练。其他六个 baseline 保留自身固定配置，
未单独指定 patience 的仍为 50，PreferDiff / DDSR 的 10 不变。
SASRec 的训练及候选生成不受此次改动影响。

## 验证与运行结果

```bash
python -m unittest discover -p 'test_*.py' -v
python check_newchoice_parity.py --source /path/to/NewChoice
```

第二条从原始源码提取核心 Python 定义，不需要安装 NewChoice 的训练框架，
验证同种子初始化、训练/评估前向和参数梯度、0/1/2 强度引导分数。
这不是训练效果测试；指标应重新测得。

旧模型权重与新版层名称及 Conv1d 参数形状不兼容。使用新的输出目录，
不得将旧 ChoiceGrow / PreferGrow / 消融报告混入新版均值。
可复用数据、SASRec 候选和其他六个 baseline 的已完成报告；复用需记录原始配置和来源。
