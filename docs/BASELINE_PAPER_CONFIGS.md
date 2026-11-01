# 六种 baseline 的原论文与作者配置核查

核查日期：2026-10-31。研究范围为 Caser、GRU4Rec、BERT4Rec、PreferDiff、DiffRec、DDSR；目标数据集为本工程的 Beauty、Toys、Sports、Steam。本文件记录来源与差异，不代表六模型在这四个数据集上都有公开的唯一最优配置。

## 本轮实际采用的固定设置

模型搜索已默认关闭；每轮验证、早停、最佳checkpoint选择及训练后即时测试继续执行。训练轮数沿用用户指定的无上限早停；这覆盖原仓库固定epoch上限。主表仍为seed100–102三次训练。附录定义的融合系数和引导负例权重验证选择保留，不能把它们误说成全部不再选择参数。

| 方法 | 学习率 / 批量 | 本轮采用值与来源说明 |
| --- | --- | --- |
| Caser | .001 / 100 | d50，L5、T1，horizontal16/vertical4，dropout.5，3负例BCE+Adam，L2=1e-6。batch取论文，lr等取作者例子；T1和缺用户项是数据适配。 |
| GRU4Rec | .01 / 50 | RSC15例子单层100维，TOP1+批内负例、tanh、hidden dropout.5、Adagrad；one-hot三门输入矩阵。当前仍逐条历史重建状态。 |
| BERT4Rec | 1e-4 / 256 | 2层2头；Beauty/Toys/Sports d64、mask.6、hidden dropout.5，Steam d256、mask.4、dropout.2，attention dropout.2。Toys/Sports借用Beauty配置；heads按论文文字。原版无bias correction Adam、weight decay.01、clip5、warmup100、40万/80万步decay。原历史50受当前固定数据10限制。 |
| PreferDiff | 1e-4 / 256 | Amazon d3072、Steam原生d768；Table12 Beauty λ.8/w6、Toys λ.5/w4、Sports λ.4/w2；Steam默认λ.5/w4。cosine偏好重建，AdamW恒定lr，diffusion2000/DDIM20，patience10、clip1。保留训练物品拟合白化和本地历史编码器。 |
| DiffRec | 1e-4 / 400 | **按用户“缺项采用官方默认”取main.py argparse默认**，不是下文Amazon-Book发布checkpoint例子：T100、scale.1、min.0001、max.02、MLP[1000]、time10、x0、AdamW、sampling0、reweight=True。 |
| DDSR | 1e-4 / 2048 | Adam、patience10、T1000、32×256码本；skip100。lr/skip为公开范围首项的预先固定选择，官方默认未公开，不能声称已经找到。 |

ChoiceGrow使用用户提供的明确数值：lr=`0.0006167980008189615`、m=`.01`、alpha=`.035`、root_share=`.7835`、fine_share=`.887`、Lambda=`-log(.1)`；三层预算为`[.4421780796256581,.056331593007552826,1.8040754203608345]`，层级速率再乘`.965`，reset速率`.035`。m=.01表示混入邻居前1%候选概率，文字旧值1.0228%并非本轮采用值。嵌套随机分区为64→1024→全目录，保留原邻居mix=.02。

固定参数不等于各作者原仓库完整复现：本轮只在当前提供的数据和统一全目录评估器内应用有据可查的参数与目标；下文逐模型记录剩余协议差异。

## 采用规则与结论

1. 有原论文明确数值时，优先论文数值；仅公布搜索范围时，不能把范围里的某一项称为“论文最优”。
2. 作者发布的可执行例子/默认配置可以作为不再搜索时的固定配置来源，但需标注为移植配置；原论文未做过目标数据集时，无法得到该数据集的原论文最优参数。
3. 本工程当前使用固定历史—目标数据和统一评测协议，缺用户 ID。Caser 的用户向量、GRU4Rec 的 session-parallel 状态更新、DiffRec 的完整用户交互矩阵不能仅通过改学习率恢复。论文配置与本工程协议覆盖项须分别记录。
4. 用户最新指定所有 baseline 采用原始论文配置；缺项使用有来源的作者默认、关闭搜索，取代此前 GRU4Rec 64 负例 BCE + Adam 指定。GRU4Rec 固定采用首发 RSC15 示例 TOP1 + Adagrad。

| 方法 | 可直接核实的原论文设置 | 目标数据集上是否公开完整固定最优配置 |
| --- | --- | --- |
| Caser | 3 负例 BCE，Adam，batch 100，FC dropout 0.5 | 否；论文数据为 MovieLens、Gowalla、Foursquare、Tmall；lr、维度、卷积通道等给搜索范围 |
| GRU4Rec | 单层 GRU，Adagrad；TOP1/BPR/CE；Table 2 给 RSC15/VIDEO 的不同配置 | 否；四个目标数据集均不在首篇原论文中 |
| BERT4Rec | lr 1e-4，Adam/解耦 decay，batch 256，2 层；Beauty mask 0.6、Steam mask 0.4 | Beauty/Steam 有作者运行脚本；Toys/Sports 没有；论文与代码存在 heads/部分优化器细节差异 |
| PreferDiff | Table 12 给 Sports/Beauty/Toys 的 lr、lambda、guidance、维度；diffusion 2000、DDIM 20 | 三个 Amazon 数据集有；Steam 没有同等单域训练最佳配置；Table 12 与 README 不一致 |
| DiffRec | 时间嵌入 10，低噪声扩散；作者发布可执行 checkpoint 配置 | 否；原论文是 Amazon-Book/Yelp/ML-1M，非这里四个数据集 |
| DDSR | Adam，batch 2048，patience 10，PQ 32×256，T=1000 | 否；lr 和 skip 只给搜索集合；未找到论文链接的官方发布仓库 |

## Caser

来源：原论文 [Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding](https://arxiv.org/abs/1809.07426)，§3.5、§4.1；论文脚注指向作者的 [MatConvNet 仓库](https://github.com/graytowne/caser/tree/a981663a608bc3f393fee3bf9f7d8098676dd0f2)。另有同一作者的 [PyTorch 版本](https://github.com/graytowne/caser_pytorch/tree/c4e10e4c12f0fce8a0cba12008b3aff5d07a26d1)，README 明确提示可能优于论文报告值。

- 原论文：每正例随机采样 **3 个负例**，binary cross-entropy；Adam；batch **100**；所有参数 L2；FC dropout **0.5**。
- 原论文仅公布 lr 搜索 `{1, 1e-1, ..., 1e-4}`、维度 `{5,10,20,30,50,100}`、历史长度 `{1,...,9}`、目标数 `{1,2,3}`、horizontal filters `{4,8,16,32,64}`、vertical filters `{1,2,4,8,16}`，没有四个目标数据集的最优点。
- 作者 [main_caser.m](https://github.com/graytowne/caser/blob/a981663a608bc3f393fee3bf9f7d8098676dd0f2/main_caser.m) 的可执行例子：`L=5, T=3, lr=1e-3, l2=1e-6, negatives=3, d=50, nv=4, nh=16, dropout=0.5`，激活均 ReLU。
- **paper/code 冲突**：[caser_train.m](https://github.com/graytowne/caser/blob/a981663a608bc3f393fee3bf9f7d8098676dd0f2/caser_train.m) batch 固定 **512**；论文写 **100**。按用户“论文内配置”的优先级，应采用 paper 100，并把 lr/d/nv/nh 等注明为作者例子移植。
- 当前工程不含真实用户 ID，只做历史表示；原模型 [caser.py](https://github.com/graytowne/caser_pytorch/blob/c4e10e4c12f0fce8a0cba12008b3aff5d07a26d1/caser.py) 使用独立用户向量、输入 item embeddings、输出 `W2/b2`。仅把负例数、Adam 和 dropout 改对，仍不能称为完全等价的原始 Caser。
- `T=3` 需要连续多个未来目标；当前单目标协议只能 `T=1`。`L=5` 可以取现有历史的最后 5 项；保留统一历史 10 则也是协议覆盖项。

## GRU4Rec

来源：首篇 [Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/abs/1511.06939)，Table 2、§4.2；作者的首发代码版本 [461c329](https://github.com/hidasib/GRU4Rec/tree/461c329c17ee096adbf55a00f0f182d52c4b729f)。不能把当前已升级到新损失版本的 README 默认倒推成 2016 论文配置。

| 数据/目标 | Batch | Hidden dropout | lr | Momentum | Optimizer |
| --- | ---: | ---: | ---: | ---: | --- |
| RSC15 TOP1 | 50 | 0.5 | 0.01 | 0 | Adagrad |
| RSC15 BPR | 50 | 0.2 | 0.05 | 0.2 | Adagrad |
| RSC15 CE | 500 | 0 | 0.01 | 0 | Adagrad |
| VIDEO TOP1 | 50 | 0.4 | 0.05 | 0 | Adagrad |
| VIDEO BPR | 50 | 0.3 | 0.1 | 0 | Adagrad |
| VIDEO CE | 200 | 0.1 | 0.05 | 0.3 | Adagrad |

论文先在 100 hidden 下调参，再报告 100/1000 hidden 两档结果；单 GRU 层效果最佳；使用 1-of-N 输入，不使用普通 item embedding 输入。负例默认来自同一 minibatch 的其他 session 下一物品，并不是固定 64 个均匀负例。

作者 [RSC15 例子](https://github.com/hidasib/GRU4Rec/blob/461c329c17ee096adbf55a00f0f182d52c4b729f/examples/rsc15/run_rsc15.py) 可直接核实固定点：`layers=[100], loss=top1, batch=50, dropout_hidden=0.5, lr=0.01, momentum=0`，继承默认 Adagrad。[首发 gru4rec.py](https://github.com/hidasib/GRU4Rec/blob/461c329c17ee096adbf55a00f0f182d52c4b729f/gru4rec.py) constructor 自身默认 TOP1、batch50、dropout0.4、lr0.05、Adagrad、L2=0、grad_cap=0，与 VIDEO TOP1 对应；它与 RSC15 示例不是同一固定点。

按用户最新指示，工程应从 64 负例 BCE + Adam 改为上述 RSC15 示例 TOP1 + Adagrad。仍需处理 session-parallel 状态传递、output activation（TOP1 默认为 tanh）和负例构造，不能只换 loss 名称与 lr。

## BERT4Rec

**当前实现已按作者脚本修正**：固定遮盖数、全 MASK、10 份固定副本加 mask-last、右侧 padding、50 位置容量，以及独立 Q/K/V、初始化、GELU、token type 和 MLM 词表。验证/测试采用 100 个流行度负例，训练 MLM 仍使用全词表交叉熵。具体对照、训练词频与原版差异、仅重跑命令见 [BERT4Rec 核查与重跑](BERT4REC_ALIGNMENT.md)。

来源：[BERT4Rec](https://arxiv.org/abs/1904.06690)，§4.3；作者 [FeiSun/BERT4Rec @615eaf2](https://github.com/FeiSun/BERT4Rec/tree/615eaf2004abecda487a38d5b0c72f3dcfcae5b3)。

原论文明确：Adam，lr **1e-4**，beta **(0.9,0.999)**，weight decay **0.01**，学习率线性衰减，梯度范数阈值 **5**，2 层；文字称 heads=2，并说明每 head 维度32；batch **256**；Beauty/Steam 历史长 **50**；Beauty mask **0.6**，Steam mask **0.4**。目标为 masked-item 全目录交叉熵，不是 64 负例 BCE。

| 作者运行配置 | Beauty | Steam |
| --- | ---: | ---: |
| hidden | 64 | 256 |
| layers | 2 | 2 |
| heads（JSON） | 2 | 8 |
| hidden dropout | 0.5 | 0.2 |
| attention dropout | 0.2 | 0.2 |
| FFN intermediate | 256 | 1024 |
| max history | 50 | 50 |
| mask probability | 0.6 | 0.4 |
| batch | 256 | 256 |
| train step horizon | 400000 | 800000 |
| warmup steps | 100 | 100 |

来源：[run_beauty.sh](https://github.com/FeiSun/BERT4Rec/blob/615eaf2004abecda487a38d5b0c72f3dcfcae5b3/run_beauty.sh)、[run_steam.sh](https://github.com/FeiSun/BERT4Rec/blob/615eaf2004abecda487a38d5b0c72f3dcfcae5b3/run_steam.sh)、[Beauty64 JSON](https://github.com/FeiSun/BERT4Rec/blob/615eaf2004abecda487a38d5b0c72f3dcfcae5b3/bert_train/bert_config_beauty_64.json)、[Steam256 JSON](https://github.com/FeiSun/BERT4Rec/blob/615eaf2004abecda487a38d5b0c72f3dcfcae5b3/bert_train/bert_config_steam_256.json)。没有 Toys/Sports 配置；移植 Beauty 设置时必须明确注明。

需要记录的差异：

- 论文文字 heads=2，而 Steam256 JSON heads=8；后者符合每 head 32 维，不能把二者都说成完全相同的原论文固定值。
- [optimization.py](https://github.com/FeiSun/BERT4Rec/blob/615eaf2004abecda487a38d5b0c72f3dcfcae5b3/optimization.py) 使用 BERT 自定义 `AdamWeightDecayOptimizer`：epsilon **1e-6**，decay 排除 LayerNorm/bias；更新没有标准 PyTorch AdamW 的 bias correction。普通 AdamW 只是近似适配。
- 调度是前100步线性 warmup，之后按指定 global-step horizon 线性减到0；代码总步数不是 epoch。用户要求“无上限早停”时可保留衰减 horizon 而不设 epoch 上限，但到 horizon 后 lr 为0，无限继续不会再更新；改恒定 lr 则应注明用户/工程覆盖。
- 原作者数据生成包含多份遮盖副本、滑窗和原始历史；工程只有历史10时不能恢复50长度数据。不能通过 `history_length=50` 将缺失的过去40项补回来。

## PreferDiff

来源：[Preference Diffusion for Recommendation, arXiv:2410.13117v2](https://arxiv.org/html/2410.13117v2)，§4.1、Appendix D.2、Table 12；官方 [lswhim/PreferDiff @fdf45b0](https://github.com/lswhim/PreferDiff/tree/fdf45b0325f2aa19eee4e92196f5b87d5848a1e3)。

**按用户“论文内配置”应优先下表的论文 Table 12，不能用 README 表覆盖。**

| 数据集 | lr | weight decay | lambda（论文） | guidance w（论文） | embedding/hidden |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sports | 1e-4 | 0 | 0.4 | 2 | 3072 |
| Beauty | 1e-4 | 0 | 0.8 | 6 | 3072 |
| Toys | 1e-4 | 0 | 0.5 | 4 | 3072 |

Steam 在论文跨平台泛化实验出现，但未给与上表等同的 Steam 单域训练最佳配置。

论文 D.2：AdamW；diffusion **2000**；DDIM **20**；history condition dropout **0.1**；linear beta **1e-4→0.02**；负例为 minibatch 其他目标物品。论文正文强调低维会明显损失性能。

### 论文与官方代码冲突

| 项目 | 论文 v2 | 官方 README/模型 YAML |
| --- | --- | --- |
| Beauty lambda / w | 0.8 / 6 | 0.4 / 8 |
| Toys lambda / w | 0.5 / 4 | 0.6 / 6 |
| DDIM steps | 20 | 100 |

来源：[README](https://github.com/lswhim/PreferDiff/blob/fdf45b0325f2aa19eee4e92196f5b87d5848a1e3/README.md)、[模型 config.yaml](https://github.com/lswhim/PreferDiff/blob/fdf45b0325f2aa19eee4e92196f5b87d5848a1e3/models/PreferDiff/config.yaml)。模型 YAML 还给 gamma5、lambda0.5、w4、1层、2head、dropout0.1；它是默认值，不能覆盖逐数据最佳表。

[default.yaml](https://github.com/lswhim/PreferDiff/blob/fdf45b0325f2aa19eee4e92196f5b87d5848a1e3/default.yaml) 另给 history10、batch256、eval batch32、epoch150、patience10、max_grad_norm1、warmup_steps10000；但 [trainer.py](https://github.com/lswhim/PreferDiff/blob/fdf45b0325f2aa19eee4e92196f5b87d5848a1e3/trainer.py) 没有实例化 scheduler，`scheduler.step()` 为注释，实际是 **AdamW 恒定 lr，没有生效的 warmup**。固定150 epoch 可被用户的无上限早停设置覆盖，但要记录。

### 3072 不只是源特征宽度

[PreferDiff 模型](https://github.com/lswhim/PreferDiff/blob/fdf45b0325f2aa19eee4e92196f5b87d5848a1e3/models/PreferDiff/_model.py) 继承 [SASRec 模型](https://github.com/lswhim/PreferDiff/blob/fdf45b0325f2aa19eee4e92196f5b87d5848a1e3/models/SASRec/_model.py)：输入、位置嵌入、Transformer hidden、生成目标全是 hidden_size **3072**。time MLP 为 `3072→6144→3072`，最终去噪线性层 `9216→3072`。

[modules.py 的 diagonalize_and_scale](https://github.com/lswhim/PreferDiff/blob/fdf45b0325f2aa19eee4e92196f5b87d5848a1e3/modules.py) 计算完整协方差 eigendecomposition 后白化 `(e-mean)@O@D`，**没有截断降维**；Transformer FFN 也是3072到3072。工程当前投影到256并用hidden256，是实质模型改动；单改 lambda/w/lr 不足以符合上表维度。

论文主要 cosine 偏好目标与官方 CLI `--loss_type=cosine` 也应一起检查，不能直接把 YAML 默认 `loss_type:l2` 当论文最佳。

## DiffRec

来源：[Diffusion Recommender Model](https://arxiv.org/abs/2304.04971)，§4 与实验设置；官方 [YiyanXu/DiffRec @e42532f](https://github.com/YiyanXu/DiffRec/tree/e42532f5f329a5f1c5230f05ad7e2930d77b9f75)。指 Wang 等交互向量 DiffRec，不是其他同名序列模型。

原论文仅实验 Amazon-Book、Yelp、MovieLens-1M；没有 Beauty/Toys/Sports/Steam 的“论文最佳配置”。原论文公布搜索范围：time embedding固定10，MLP隐藏 `{[300],[200,600],[1000]}`；T `{2,5,10,40,50,100}`，inference forward-noise step `{0,T/4,T/2}`；lr、噪声scale/min/max也分别搜索。

作者 [README 的 Amazon-Book clean 训练命令](https://github.com/YiyanXu/DiffRec/blob/e42532f5f329a5f1c5230f05ad7e2930d77b9f75/README.md) 和 [inference.py](https://github.com/YiyanXu/DiffRec/blob/e42532f5f329a5f1c5230f05ad7e2930d77b9f75/DiffRec/inference.py) 明确发布以下完整固定例子：

```text
dataset=amazon-book_clean
lr=5e-5, weight_decay=0, batch=400
dims=[1000], timestep_embedding=10, mean_type=x0
T=5, noise_schedule=linear-var
noise_scale=1e-4, noise_min=5e-4, noise_max=0.005
sampling_steps=0, sampling_noise=False, reweight=True
```

`inference.py` 的 Amazon/Yelp clean checkpoint 文件名误写 reweight0，作者注释明确实际训练是 **reweight1**，不要照文件名设False。[main.py](https://github.com/YiyanXu/DiffRec/blob/e42532f5f329a5f1c5230f05ad7e2930d77b9f75/DiffRec/main.py) 使用 AdamW；其 argparse 默认lr1e-4、T100、scale0.1，不等于上述已发布 checkpoint 配置。

Amazon-Book clean 参数可作为四数据集不搜索时的固定移植方案，不能标注四数据集原论文最优。原模型按每个用户完整交互向量训练，工程按单条历史—目标向量适配，数据语义仍有差别。

## DDSR

来源：[Breaking Determinism: Fuzzy Modeling of Sequential Recommendation Using Discrete State Space Diffusion Model, arXiv:2410.23994v2](https://arxiv.org/html/2410.23994v2)，§4、Appendix B、C.1。

原文可确认：

- PQ semantic IDs：**32 个 codebooks，每个256**；另有RQ-VAE 6×256路线，两者不应混淆。
- Transformer 预测被扰动历史对应的干净下一物品 semantic codes；训练目标为离散代码交叉熵；time embedding为sinusoidal/cosine。
- 优化器 **Adam**；batch **2048**；验证 NDCG@10 选模型，early-stopping patience **10**。
- T **1000**；uniform transition 使用 cosine schedule；importance transition 使用另一条线性 schedule，不应混为同一设置。
- lr 仅给集合 `{0.0001,0.0002,0.0003,0.0005,0.001}`。
- sampling skip 仅给集合 `{100,50,35,28,23,20,17,15}`。
- 没有公布每数据集最终 lr/skip，也没有在论文中给出可据此确定 hidden、head数、layer数、dropout 的固定唯一值。本次未找到与论文作者身份及论文链接可核实的官方代码发布，不能把 RecBole 通用默认称为 DDSR 官方最佳。

因此关闭搜索时的任何DDSR lr（例如1e-4）、skip（例如100）、Transformer深度/宽度都是**明确固定的本地选择**；即使其数值在论文范围内也不能说是论文验证最优。

本轮预先固定 lr=1e-4、skip=100，均取论文所列范围的首项；其来源类型必须记作“paper range first entry / predeclared local choice”，不能记作“official default”。

原论文基于BERT文本语义；工程Amazon源3072维OpenAI向量和Steam768维BERT向量并不统一。PQ/OPQ实现、特征来源、按全目录距离排名等适配应保留披露。

## 给实现者的最小落地检查

- 禁用的是 validation hyperparameter search；保留每轮/每间隔验证、early stopping、validation checkpoint选择以及正式训练后的即时测试。
- 各模型独立解析 lr、batch、hidden、dropout、loss、optimizer，避免仍继承 ChoiceGrow 通用hidden256/batch256/AdamW后只在报告里改名。
- PreferDiff 论文维度3072需同时更改固定特征、history transformer、time MLP和去噪器；Steam只有768维源特征时不能凭padding宣称native3072原论文设定。
- BERT需要分开 hidden dropout与attention dropout，以及paper/official schedule差异；固定历史10无法恢复长历史50。
- 原论文没提供逐数据固定点的模型使用“原论文已知数值 + 作者固定例子或显式本地选择”组成配置，并在resolved config中记录来源类型。
- 旧搜索和新固定配置不可在同一冻结输出目录混用；保留之前结果，不让旧done/checkpoint标记跳过新配置。

