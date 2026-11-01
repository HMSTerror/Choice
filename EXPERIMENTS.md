# 实验清单

默认运行正文和所提供技术附录中的实验。主表恢复九种模型；所有指标由当前运行计算，不预置论文数值。论文的四个数据集为 Steam、Beauty、Toys、Sports；源文件中额外保留的 ML1M 不属于该论文实验清单。

| 分组（`--groups`） | 实验 | 数据集与重复 |
| --- | --- | --- |
| `main` | SASRec、Caser、GRU4Rec、BERT4Rec、PreferDiff、DiffRec、DDSR、PreferGrow、ChoiceGrow | 四个数据集 × 3 runs（seed 100–102）；SASRec 复用候选阶段模型及测试 |
| `factorial` | Shared/Root、User/Root、Shared/Three-Level、User/Three-Level | 四个数据集 × 六个种子；完整模型与主表复用 |
| `auxiliary` | PreferGrow 分数融合、候选引导负例；ChoiceGrow 流行度匹配候选 | 四个数据集 × 三个种子；基础模型与主表复用 |
| `forward` | Proposal-Root、Proposal-Hierarchy、Dynamic-Mass、History-Swap、Shuffle-Scope、Uniform-Rate、Real-Scope | Toys × 六个种子；Shuffle-Scope 与主模型复用 |
| `matched_exit` | 四个 factorial 臂的验证集平均退出率匹配重训；测试集退出率及 21 个时间点的保留率诊断 | 四个数据集 × 六个种子 |
| `initialization` | Shared/Three-Level 与 User/Three-Level 的训练生成器 × 反向初始化交叉 | 四个数据集；复用六种子的最佳模型，配对测试流 |
| `partitions` | Shuffle 与 Real，五个分区种子 × 三个训练种子；分区/训练方差分解与候选集中度、块熵诊断 | 四个数据集 |
| `efficiency` | 1K—5K 稠密与结构化前向、反向算子时间/误差，三次运行均值；100K 时间/存储/归一化误差与失败计数；Steam 全测试集、batch=1、在线候选生成的端到端耗时 | 当前硬件实测；端到端使用预先固定的 seed=100 |

无上限早停与邻居增强沿用本项目设置。论文正文的 300 轮上限不覆盖用户指定的默认值。扩散模型正式测试为三个采样流，种子按 `SHA256(run_seed:split:stream)` 派生；三个流先各自计算指标，再取均值。BERT4Rec 为固定候选集上的一次确定性评估，独立训练次数仍由主表种子设置决定。

## 选择规则

默认关闭 ChoiceGrow 和六种 baseline 的超参数搜索。ChoiceGrow 固定使用用户指定配置：AdamW 恒定学习率 `0.0006167980008189615`、候选质量 `m=0.01`、reset `alpha=0.035`、root share `0.7835`、fine share `0.887`、总预算 `-log(0.1)`。三层预算为 `[0.4421780796256581, 0.056331593007552826, 1.8040754203608345]`，运行时乘 `1-alpha`；reset 速率为 alpha。分区为随机嵌套 `64→1024→全目录`。候选质量在混入邻居前为 1%，旧值 `0.010228021396912973` 对应 1.0228%，本轮采用明确指定的新数值。

分数融合系数和引导负例权重也只按验证集选择。所有已选择的配置保存在 `runs/paper/selected/`。已有验证选定参数可填入 `paper_suite.json` 的 `dataset_overrides`，或用 `--skip-tuning` 使用 `config.json`，但三个阶段需保持设置一致。

其他模型从 `paper_suite.json/baselines/fixed_configs` 读取独立的维度、学习率、批量、目标和优化器；有论文明确参数则优先论文，缺项使用注明来源的官方配置。DDSR 未公开完整逐数据参数且未找到可核实官方仓库，学习率 `1e-4` 和 skip `100` 是论文范围首项的预先固定选择，并非官方默认或原论文最佳。完整来源、版本与冲突见 [配置核查](docs/BASELINE_PAPER_CONFIGS.md)。SASRec 和 PreferGrow 保留既有固定设置。主表模型在 3 个种子上训练，附录按原清单重复；验证集继续用于早停和 checkpoint 选择，正式训练完立即测试。

## 主表模型实现与数据适配

`baselines.py` 使用现有历史输入，各方法有独立结构、扰动对象和目标函数。BERT4Rec 默认调用本轮修正后的 `bert4rec.py`，使用真实目标加 100 个流行度负例评估；其他模型保留全目录评估。汇总带评估协议字段，不同协议的指标不能直接比较。它们不依赖本地其他工程或运行时下载的基线仓库。

| 模型 | 实现 |
| --- | --- |
| SASRec | 一层因果 Transformer，ID 嵌入；正例与 64 个采样负例的 logistic 目标。候选只使用历史，排除历史中的物品，不读取目标。 |
| GRU4Rec | 首发 RSC15 TOP1 示例：单层100维、批内负例、tanh 输出、hidden dropout .5，lr .01、batch50、Adagrad（epsilon在平方根内）。三个独立 one-hot 输入门矩阵；按当前独立历史样本训练，不能恢复 session-parallel 状态传递。 |
| Caser | 最后5项历史、50维，垂直4/水平16通道，独立输出物品矩阵和偏置；3负例 BCE、Adam lr .001、batch100、dropout .5、L2 1e-6。缺用户 ID 与多未来目标，保留历史表示、单目标适配。 |
| BERT4Rec | 修正后的官方结构移植：两层双向 Transformer，Beauty/Toys/Sports hidden64、2 heads、mask .6；Steam hidden256、8 heads、mask .4。hidden dropout分别 .5/.2，attention dropout .2；10 份全 MASK 固定遮盖副本加 mask-last。lr1e-4、batch256、uncorrected BERT Adam、decay .01、clip5、warmup100、40万/80万步线性衰减。右侧补齐、50 位置容量，实际历史仍为源数据提供的最多10项。验证/测试为目标 + 100 个流行度负例，见 [实现核查](docs/BERT4REC_ALIGNMENT.md)。 |
| PreferDiff | Amazon原生3072维固定白化特征和模型，Steam为现有768维；cosine重建偏好目标。论文Table12：Beauty lambda .8/w6、Toys .5/w4、Sports .4/w2；Steam缺原始最佳点，使用默认lambda .5/w4。AdamW lr1e-4、batch256、condition dropout .1、2000步训练、论文20步DDIM。 |
| DiffRec | **Wang 等的交互向量版本**；使用官方main.py默认AdamW lr1e-4、batch400、MLP[1000]、time10、x0、linear-var、T100、noise scale .1/min .0001/max .02、SNR加权、sampling_steps0和确定性逆向更新。原论文没有当前四数据集的最佳点。 |
| DDSR | arXiv:2410.23994 的 **OPQ + Uniform** 路径：32 个 256 项码本，余弦累计保留率、1000 步；扰动整个历史的语义码，逐位置预测下一物品的干净语义码并计算交叉熵。推理从历史码开始，用式 (7) 的联合边缘概率归一化后迭代跳步采样，最后按码本质心距离给全目录排序。 |
| PreferGrow | 官方 HybridWise 的全目录秩一转移与额外 nonpreference 状态；公共 score-entropy 去噪器和采样器。 |
| ChoiceGrow | 历史条件目的分布、嵌套作用域与全局 reset，精确结构化前向核、score-entropy、incoming-edge 逆向率与自适应 tau-leaping。主模型只增加本项目保留的邻居混合。 |

语义源为仓库已有的物品描述向量：Steam 用 `bert_emb.pickle`（768 维），三个 Amazon 类别用 `3large_emb.pickle`（3072 维）。PreferDiff 用**只在训练出现物品上拟合**的完整原生维度白化，不再降到256维；DDSR 在同一训练物品集合上拟合 OPQ，然后给整个目录编码。目录物品描述可用，但验证/测试交互和目标不参与拟合。`semantic.json` 记录来源、维度、训练物品数和量化方式。

本轮按各模型原论文核对参数，但数据和训练协议仍是统一适配：源数据只有固定历史—单目标样本、没有用户 ID；因此不能恢复 Caser 用户向量、GRU4Rec session-parallel 状态传递或 DiffRec 完整用户交互矩阵。DDSR 的 Amazon 文本来源、全目录解码，PreferDiff 的训练物品白化拟合与历史编码器适配，PreferGrow 的共享去噪器/采样器也仍有区别。本文配置来源与剩余适配已单独记录，不将参数对齐写作原仓库端到端等价复现。

参考实现与算法来源：

- PreferDiff：[lswhim/PreferDiff，fdf45b0](https://github.com/lswhim/PreferDiff/tree/fdf45b0325f2aa19eee4e92196f5b87d5848a1e3)。
- DiffRec：[YiyanXu/DiffRec，e42532f](https://github.com/YiyanXu/DiffRec/tree/e42532f5f329a5f1c5230f05ad7e2930d77b9f75)。
- DDSR：[Breaking Determinism，v2 第 3—4 节与附录 B/C](https://arxiv.org/html/2410.23994v2)。
- PreferGrow：[Hugo-Chinn/PreferGrow，7451996](https://github.com/Hugo-Chinn/PreferGrow/tree/7451996bbd6d289cf96457870cdc45a79fab8311)。

## 各对照的实现

- Shared 目的分布使用训练频次基础分布，关闭邻居；User 目的分布保留本项目的邻居混合。Root 把相同的作用域预算全部分配给根，Uniform-Rate 将三层预算等分，均保持 reset 不变。
- Proposal 两臂在候选区与非候选区内部采用均匀权重。Dynamic-Mass 用历史编码学习 `[.001,.05]` 内的候选质量，包含离散前向抽样的 score-function 梯度项。
- 流行度匹配候选通过训练频次附近的物品双射替换；History-Swap 在训练频次相近的历史间交换候选。两者均不读取验证或测试目标。
- Real-Scope 从训练序列构建有向转移，执行 directed Infomap，多次固定重启取最佳码长；用 Markov times `{2,4,8,16}` 的模块签名作共识排序，再按 Shuffle 相同的 64/1024 容量打包。细块到粗块严格嵌套。论文未给出的打包细节在此明确固定。
- 退出率匹配仅以验证集计算 `Shared/Root 平均退出率 / 当前臂平均退出率`，统一缩放当前臂所有速率，再重新训练。
- PreferGrow 使用官方 `HybridWise` 的秩一全局转移分布（含额外 nonpreference 状态），共享当前工程的去噪网络、score-entropy、早停和评估器。默认 Hybrid 与各数据集 gamma 在运行前固定，未按测试集选择；这是统一评估器下的适配实现。原方法参考：<https://github.com/Hugo-Chinn/PreferGrow/tree/7451996bbd6d289cf96457870cdc45a79fab8311>。

## 查看清单与检查

```bash
bash run.sh --stage plan
bash run.sh --stage summary --output runs/paper
.venv/bin/python -B -m unittest -v test_core test_training test_experiments test_baselines
```

`summary/table1_main.json`、`table2_factorial.json`、`table3_auxiliary.json`、`table3_forward_toys.json` 按正文表格整理；Table 3A 的基础模型也只取对应的三个对照种子。`report.json` 保留正式测试的每个采样流，标准差按训练运行计算。

| 附录表 | 输出（`summary/` 下） |
| --- | --- |
| 4 初始化交叉 | `appendix_table4_initialization.json`，两种训练生成器 × 两种初始化；对角线复用正式测试 |
| 5 作用域对照 | `appendix_table5_scope.json`，User-Root 六种子与 Real/Shuffle 跨分区均值、分区 SD |
| 6 退出率和保留/替换率 | `appendix_table6_corruption.json`，每个训练种子的均值、p10、p90；每个测试查询与 21 个时间点等权 |
| 7 匹配退出率重训 | `appendix_table7_matched_exit.json`，验证集统一倍率和六种子指标 |
| 8 方差分解 | `appendix_table8_variance.json` / `partition_reml.json`，平衡随机截距 REML 的分区、训练残差和总 SD |
| 9 算子基准 | `appendix_table9_operators.json`，每次运行及三次均值、稠密误差、100K 存储和检查结果 |
| 10 流程成本组成 | 算子外的候选生成、终端分布、去噪和最终排序均进入端到端计时；层级构建离线执行 |
| 11 训练和推理成本 | `appendix_table11_cost.json`，Steam 单条查询 median/P95、queries/s、峰值已分配显存、反向子步和训练 GPU-hours |

端到端使用 Steam 全部测试查询，计时覆盖 CPU 查询读取、传输、ChoiceGrow 在线 SASRec top-100、反向生成、全目录 top-10 返回 CPU；PreferGrow 无 proposer 调用。预热与测试随机流分离，CUDA 在每条查询前后同步。记录每条延迟，训练时间截至验证选定检查点，并同时保留早停总耗时；SASRec 预训练成本单列并给出每查询摊销。硬件型号写入效率报告，不能将算子加速比当作整条推荐链路加速比。

效率数字只代表实际运行硬件。默认的 100K 测试只测算子，不创建 100K 物品的完整推荐模型；稠密比较排除去噪器开销。`--smoke` 会缩小样本、种子、分区重复和效率规模。

## 单 GPU 并行执行

`--workers 4` 以独立进程调度不同训练种子、数据集及验证试验。每个任务仍使用原有训练批量、目标和验证选择规则；搜索阶段不测试，正式模型在同一进程内完成训练后测试。候选就绪后可启动不依赖语义编码的任务；PreferDiff、DDSR 等待语义文件。每个数据集的全部 ChoiceGrow 搜索试验完成后才进行前三名重训和配置选择。相同分区文件用进程锁构建，JSON 用原子替换发布，汇总只由协调器写入。

`parallel/status.json` 保存 PID、完成数、失败和待运行数；每个任务独立记录 `.log` 和完成标记。多个任务的训练输出也实时写到协调器终端。检测到显存不足时，会降低并发数并重新运行该任务；训练内部仍沿用备份后重启的规则。所有正式任务结束后才进行效率测量，避免并发污染计时。运行中可写 `parallel/worker_limit.json` 为 `{"workers": 2}` 等值，以限制后续启动的数量。

候选文件无损保存为 int32，进入模型前转换为 int64；向量化批量读取保留原 DataLoader 的打乱顺序和跨轮次 RNG 行为。`--tf32` 的开关记录在 `parallel/spec.json`。A100 服务器可额外启动 CUDA MPS 以重叠不同进程的 GPU 工作；科学配置中 CTMC 的系数指数继续使用 float64，未启用半精度概率运算。Python 3.10 使用 Infomap 2.9.2，Python 3.11 及以上使用 2.15.0。
