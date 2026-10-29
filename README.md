# Choice 论文代码实现初版

从 NewChoice 提取的独立实验：**保留训练集历史邻居，只训练 score-entropy，不包含辅助排序损失或辅助读出。** 默认数据集为 Toys。

## 1. 目录与环境

```text
Choice/
  dataset/       从 NewChoice 保留的数据源文件（缓存不纳入仓库）
  config.json    唯一的实验配置
  data.py        读取源划分、转换编号、检查数据、生成训练张量
  neighbors.py   从训练序列构建最近三步的后继邻居
  models.py      SASRec 候选模型与扩散去噪网络
  diffusion.py   目的分布、嵌套分区、精确前向核、损失和反向采样
  train.py       训练、候选冻结、验证选择、最终测试
  run.py         串起全部阶段
  run.cmd        Windows 入口
  test_core.py   数学与数据契约检查
  test_training.py 无上限训练的早停检查
  runs/          运行后生成的中间文件、模型和报告
```

运行代码不依赖 `E:\NewChoice`，只需要 Python、PyTorch、NumPy、Pandas。原数据中的嵌入文件也保留，但本实验不读取它们。

从 GitHub 获取代码与数据（需先安装 Git LFS）：

```powershell
git lfs install
git clone https://github.com/HMSTerror/Choice.git
cd Choice
```

原始 `.pickle` 文件通过 Git LFS 管理。仓库不包含本次训练的模型、中间产物和实验结果；运行后生成的文件均保存在被 Git 忽略的 `runs/` 目录中。

当前电脑可直接运行 `run.cmd`，它依次寻找 `CHOICE_PYTHON`、本目录虚拟环境、`E:\anaco\python.exe`、PATH 中的 Python。

换电脑时，在 PowerShell 中安装环境：

```powershell
cd E:\Choice
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如需 CUDA 12.4 版 PyTorch，先安装下列包，再安装上面的依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

## 2. 先跑通，再做完整实验

完整链路的小样本检查：

```powershell
cd E:\Choice
.\run.cmd --smoke --dataset Toys --output runs\smoke_toys
```

它使用 32 条训练、8 条验证、8 条测试记录；SASRec 和扩散模型各训练 1 轮。保持完整物品目录、256 维模型、100 个候选、邻居机制和 20 个反向外层区间。测试只使用一个随机流。该运行检查链路，不能用来判断推荐质量。

完整 Toys 实验：

```powershell
.\run.cmd --dataset Toys --output runs\toys_s100
```

默认自动使用 GPU；强制 CPU 时添加 `--device cpu`。默认读取完整划分，SASRec 和扩散模型的 `epochs` 均为 `0`，表示不设置训练轮数上限，只按验证集早停。SASRec 每轮验证，连续 10 轮没有改善时停止；扩散模型每 10 轮验证，连续 50 个训练轮次没有改善时停止。两者均以 NDCG@10 为主指标、HR@10 为同分时的比较指标。已有模型的输出目录不会被重新训练覆盖；新实验使用新的 `--output`。

固定候选质量门在 `destination.candidate_gate="fixed_mass"` 中显式启用，`candidate_mass=0.010228021396912973`；保留 2% 邻居混合。小样本模式仍将训练轮数设为 1。若需要有限轮数，可传入 `--epochs 300 --sasrec-epochs 100`。

其他数据集的命令仅需修改 `--dataset`，可选 `Toys`、`Sports`、`Beauty`、`Steam`、`ML1M`，也接受 `ATG`、`ASO`。默认门值与速率继承 Toys 配置，迁移到其他数据集不代表已经按其验证集重新调参。

## 3. 全部实现过程

### 第一步：数据预处理 — `data.py`

Toys/Beauty/Sports 优先读取 `train_data.txt`、`valid_data.txt`、`test_data.txt`。每行最后一个编号是目标，前面的编号是历史；文本中的物品编号从 1 开始，转换为从 0 开始。保留最近 10 个历史物品，在左侧用 `item_count` 补齐。

例如，目录有 5 个真实物品，文本行 `1 3 5` 转换为历史 `[0, 2]`、目标 `4`；长度为 4 时历史张量为 `[5, 5, 0, 2]`，真实长度为 2。实际实验长度为 10。

Steam/ML1M 没有相应文本划分，读取 `train_data.df`、`val_data.df`、`test_data.df` 中的 `seq`、`len_seq`、`next`。`.df` 是 Pandas pickle。物品数和序列长度读取 `data_statis.df`。

程序检查编号、目标、历史长度，保留原划分与样本顺序，输出 `prepared/{train,val,test}.npz` 和 `metadata.json`。基础质量使用训练目标的计数：`b(i) ∝ sqrt(count_train_next(i) + 1)`。

这些源文件已经是划分后的序列；本步骤不重新划分，也不声称从原始时间戳日志重建了论文数据集。

### 第二步：训练 SASRec — `models.py`、`train.py`

只用训练集更新 SASRec。模型宽度 256、2 个注意力头、1 层，训练采用正例与 64 个均匀负例的二元逻辑损失。SASRec 内部将左侧 padding 转为右侧 padding。

每轮用验证集全目录 NDCG@10 选择最佳检查点；相同时用 HR@10。测试集不参与选择。

### 第三步：冻结候选、构建邻居和分区 — `train.py`、`neighbors.py`

加载最佳 SASRec，冻结参数，对三个划分分别读取历史并生成 top-100 候选。候选生成屏蔽历史中已出现的物品；不读取目标，不使用候选分数大小，只保存成员编号。

邻居表只读取训练集：对每条训练记录，最近第 1/2/3 个历史物品向目标分别贡献 `1`、`1/2`、`1/3` 的计数；每个源物品增加 `1` 的自身先验，再选 top-16 并归一化。没有训练后继的源物品回退到自身。

分区使用种子 100 的固定随机排列，按 64/1024 个物品划分细块/粗块，再连接目录根块。**分区不读取交互边、不做行为聚类。** 同一细块只有一个粗块父节点。

保存 `candidates.pt`、`structure.pt`、`artifacts.json`。

### 第四步：扩散训练 — `models.py`、`diffusion.py`

先按基础质量 `b` 在候选与非候选区域分别归一化，候选区域总质量为 `m`，得到 `rho_fixed`。再混入历史邻居：

```text
rho = 0.98 * rho_fixed + 0.02 * rho_neighbors
```

历史邻居按 `exp(-0.2 * 距末尾的位置)` 聚合。混合后的分布仍全目录正值；`m` 是混合前的候选质量，混合后不保证候选总质量仍等于 `m`。

每个分区层把 `rho` 限制到当前物品所在块并归一化，构造：

```text
Q = sum_l (1-alpha)*rate_l*(P_l-I) + alpha*(G-I)
G 的每一行都是共享基础分布 b。
```

`exp(sigma*Q)` 用 7 个算子的闭合代数精确计算，只对 7×7 坐标矩阵求指数，不建立目录大小的稠密矩阵。

每条训练记录从 21 个时间网格点均匀抽样，用精确前向核把真实目标扰动为 `x_t`。去噪器读取历史、`x_t` 与噪声尺度，输出全目录对数比值，自身位置固定为 0。**唯一损失是乘噪声导数后的 denoising score-entropy。**

保留 NewChoice 的线性去噪头、padding 掩码、按头维度缩放注意力、EMA，以及用于 classifier-free guidance 的历史条件丢弃。对数比值使用 `4*tanh(raw/4)`；这些是明确保留的实现细节，不是辅助排序项。

每 10 轮验证，不设置总轮数上限；验证指标连续 50 个训练轮次没有改善时停止，保存最优的 EMA 模型到 `choice_best.pt`。从头训练，不提供断点续训或超参数搜索。

### 第五步：加载最佳模型并测试 — `diffusion.py`、`train.py`

每个历史只建立一次目的分布和生成器。反向初始状态从同一个生成器的 `b*exp(10*Q)` 抽样，随后沿 20 个外层区间从噪声尺度 10 走到 0.001。

反向速率使用入边 `Q(y,x)*exp(score(y))`，自适应选择子步长。保留 8.1 的对数退出率上限：触发时同时限制实际跳跃率，跳转目的地的相对概率保持不变；报告记录触发次数。

终点对数比值直接对全目录排序，最终排序不屏蔽历史物品。每次完整测试使用三个固定随机流，分别计算 HR/NDCG@5、@10 后平均；不是先平均分数再排序。测试结果不用于选检查点。

## 4. 分阶段运行与查看结果

以下命令与一次性运行经过相同代码，使用相同输出目录：

```powershell
.\run.cmd --dataset Toys --output runs\toys_staged --stage prepare
.\run.cmd --dataset Toys --output runs\toys_staged --stage proposer
.\run.cmd --dataset Toys --output runs\toys_staged --stage artifacts
.\run.cmd --dataset Toys --output runs\toys_staged --stage train
.\run.cmd --dataset Toys --output runs\toys_staged --stage evaluate
```

重评估已有最佳模型时只执行 `--stage evaluate`，保持相同配置、种子和 `--smoke` 设置。相对路径始终相对于本项目目录解析。

主要输出：

| 文件 | 内容 |
| --- | --- |
| `resolved_config.json` | 本次生效配置 |
| `prepared/metadata.json` | 来源、原始样本数、实际使用样本数 |
| `sasrec_log.jsonl` / `choice_log.jsonl` | 每轮损失、验证指标、最佳轮数 |
| `sasrec_best.pt` / `choice_best.pt` | 验证选择的检查点 |
| `candidates.pt` / `structure.pt` | 冻结候选、基础质量、邻居与分区 |
| `training.json` | 扩散训练轮数、验证选择和耗时 |
| `report.json` | 最佳轮数、各测试随机流及平均指标 |

检查数学与数据处理：

```powershell
E:\anaco\python.exe -B -m unittest -v test_core test_training
```

使用虚拟环境时将解释器换为 `.\.venv\Scripts\python.exe`。数值检查包含精确核与稠密指数的一致性、终端参考、反向方向、损失梯度、邻居权重和预处理契约。

## 5. 已完成的验证

- 9 项检查通过（7 项核心检查和 2 项无上限训练早停检查）；Toys 的 GPU 小样本全流程已跑通，报告位于 `runs/smoke_toys_check/report.json`。
- 五个数据集的 15 个划分读取结果，逐行与原 `.df` 的历史、长度和目标一致；原始数据源已保留（缓存文件不纳入仓库）。
- 映射相同权重后，保留的去噪网络与 NewChoice 的历史编码、引导分数最大差异均小于 `1e-6`；邻居编号和权重完全一致。
- 小样本结果不用于推荐效果结论；完整实验的进度、停止原因和指标以对应 `runs/` 目录内的日志与报告为准。
