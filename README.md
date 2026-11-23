# 统一数据与命令行分析

## 最简单的运行方式

先进入项目文件夹：

```bash
cd "/Users/wangxujing/Documents/courses/经济物理学/task/project3"
```

本机已有分析环境，可直接运行：

```bash
/opt/anaconda3/envs/ml/bin/python run.py hs300
```

这条命令会使用全部288股，输出特征值谱、样本外净值与回撤、压力期比较三张PNG，以及CSV结果表。结果在 `outputs/hs300/full/basic/`。

也可以直接输入价格文件夹，包含空格的路径请加引号：

```bash
/opt/anaconda3/envs/ml/bin/python run.py "data/cac40/prices"
```

查看可用数据：

```bash
/opt/anaconda3/envs/ml/bin/python run.py --list
```

## 每组数据的命令

以下命令默认执行 `basic` 分析。需要其他模块时，在命令末尾加 `--analysis 模块名`。

| 简称 | 中文名 | 文件数 | 命令 |
|---|---|---:|---|
| hs300 | 沪深300固定股票池 | 288 | `/opt/anaconda3/envs/ml/bin/python run.py hs300` |
| hs300_long100 | 沪深固定股票池长历史样本（2002–2026共同区间） | 100 | `/opt/anaconda3/envs/ml/bin/python run.py hs300_long100` |
| asx200 | 澳大利亚标普／澳证200指数 | 157 | `/opt/anaconda3/envs/ml/bin/python run.py asx200` |
| cac40 | 法国CAC40指数 | 38 | `/opt/anaconda3/envs/ml/bin/python run.py cac40` |
| dax40 | 德国DAX40指数 | 37 | `/opt/anaconda3/envs/ml/bin/python run.py dax40` |
| ftse100 | 英国富时100指数 | 93 | `/opt/anaconda3/envs/ml/bin/python run.py ftse100` |
| kospi200 | 韩国KOSPI200指数 | 154 | `/opt/anaconda3/envs/ml/bin/python run.py kospi200` |
| nasdaq100 | 美国纳斯达克100指数 | 84 | `/opt/anaconda3/envs/ml/bin/python run.py nasdaq100` |
| nifty50 | 印度NIFTY50指数 | 47 | `/opt/anaconda3/envs/ml/bin/python run.py nifty50` |
| nikkei225 | 日本日经225指数 | 219 | `/opt/anaconda3/envs/ml/bin/python run.py nikkei225` |
| sp500 | 美国标普500指数 | 473 | `/opt/anaconda3/envs/ml/bin/python run.py sp500` |

一次运行全部已登记数据的基础分析（当前11组）：

```bash
/opt/anaconda3/envs/ml/bin/python run.py all
```

文件数是现有可用股票数，并非指数名称中的名义成分数。每组股票在共有日期上对齐，原始数据及纳入清单有哈希记录；输入错误会明确报错，不会静默丢弃股票。

## 选择分析模块

| `--analysis` | 计算内容 | 适用数据 |
|---|---|---|
| basic | 完整谱、原始/RMT截尾两策略最小方差回测、压力日期比较 | 全部已登记数据 |
| audit | 逐股数据检查、异常收益、市场代理、相关性演化、年度/月度统计；沪深另核对三组档案 | 全部已登记数据 |
| rmt | 市场/组/剩余分解、参与率、股票数/窗口/市场阶段敏感性、四策略回测、历史事件比较、配对块bootstrap | 全部已登记数据 |
| structure | 厚尾、绝对收益ACF、前后期模式匹配、独立日期打乱；有标签时再算全部模式行业关联及BH校正 | 全部已登记数据；行业和真实指数部分仅沪深 |
| industry | 行业/二级行业聚类、富集与置换检验、行业关联树 | 仅沪深有标签；其他市场明确跳过 |
| scales | 日收益及5/21/63日重叠、非重叠、局部去均值、轮廓DMA，共13个矩阵及同处理随机参照 | 全部已登记数据 |
| modes | τ=1–250、KL时间分量和功率谱、前十模态、去市场协方差、500/750日滚动稳定性、八组留出验证和六种协方差风险预测 | 全部已登记数据；股票数不足44/144时省略对应固定排名模态 |
| all | 顺次执行以上全部模块 | 全部已登记数据；缺失行业数据的步骤明确记录跳过 |

例如，运行沪深全部模块：

```bash
/opt/anaconda3/envs/ml/bin/python run.py hs300 --analysis all
```

只运行日本数据的尺度实验：

```bash
/opt/anaconda3/envs/ml/bin/python run.py nikkei225 --analysis scales
```

只运行标普500的KL分析：

```bash
/opt/anaconda3/envs/ml/bin/python run.py sp500 --analysis modes
```

`all`、`modes`和`scales`包含较多特征值分解和随机实验，比基础分析耗时更长。先检查整条流程时，可以使用：

```bash
/opt/anaconda3/envs/ml/bin/python run.py cac40 --analysis all --quick
```

`--quick`减少置换和重采样次数，并将默认最大τ改为20。它用于流程预览，结果放进独立的 `full_quick/`，不能当成正式实验结论。正式默认：随机对照100次、标签置换999次、组合差异bootstrap2000次、模态bootstrap200次。可用 `--null-reps`、`--permutations`、`--bootstrap-reps` 和 `--tau-max` 显式覆盖。

## 输出放在哪里

默认结构：

```text
outputs/hs300/full/
  input_manifest.csv      输入文件哈希与纳入状态
  selected_stocks.csv     实际参与的股票顺序
  log_returns.csv         共同日期的日对数收益
  run_manifest.json       参数、版本、代码哈希及每个模块状态
  basic/                  三张PNG和两策略结果表
  audit/                  数据审计表和市场概况图
  rmt/                    谱分解、敏感性和四策略回测
  structure/              统计特征、模式和指数对照
  industry/               行业检验与树图
  scales/                 13个尺度矩阵及摘要
  modes/                  τ扫描、KL、滚动结构与风险预测
```

每个模块都有 `summary.json` 和 `run.log`。指定其他输出根目录：

```bash
/opt/anaconda3/envs/ml/bin/python run.py hs300 --output "/你的结果目录"
```

后面仍自动加 `hs300/full/<模块>/`。重复同一命令会更新该模块结果；原有历史结果集中放在 `reference_results/`。

## 代码架构与目录说明

项目由命令行入口、流程调度、研究计算、数据与配置、结果输出五部分组成。以下列出主要文件和目录；`<dataset>` 表示数据集简称，`<profile>` 表示 `full`、`sample100` 或相应的 `_quick` 运行配置。

```text
project3/
├── run.py                         # 结构与风险分析的命令行入口
├── pipeline.py                    # 数据加载、日历核验及七个分析模块的调度
├── run_ml.py                      # 机器学习组合实验入口
├── report_long_history.py         # 长历史补充实验与报告生成
├── download_long_history.py       # 长历史数据下载及来源记录
├── verify_data.py                 # 按 SHA-256 清单核验仓库数据
├── datasets.json                  # 数据集名称、路径及股票数量配置
├── data_manifest.json             # 原始数据文件的 SHA-256 清单
├── requirements.txt               # 基础分析依赖
├── requirements_ml.txt            # 机器学习依赖范围
├── requirements_ml_lock.txt       # 机器学习环境的锁定版本
├── engines/                       # 谱分解、RMT、组合回测、模态与风险分析
├── ml_experiment/                 # 机器学习实验实现
│   ├── __init__.py
│   ├── data.py                    # 特征、预测标签、时间切分及缓存
│   ├── models.py                  # 传统机器学习与神经网络模型
│   ├── run.py                     # 模型训练、目标权重生成与交易模拟
│   ├── report.py                  # 评价指标、稳健性比较、绘图及报告输出
│   └── checks.py                  # 信息边界、模型复现、持仓漂移与费用校验
├── data/                          # 随仓库提供的研究数据
│   ├── hs300/
│   │   ├── prices/                # 股票价格快照
│   │   ├── archived_returns/      # 历史收益率档案
│   │   └── archived_zscores/      # 历史标准化收益档案
│   ├── hs300_long100/
│   │   ├── prices/                # 长历史复权价格
│   │   ├── daily_details/         # 日线明细，含成交量日历核验证据
│   │   └── provenance/            # 原始接口响应及上市日期来源记录
│   ├── metadata/                  # 行业标签及市场指数
│   └── <其他数据集>/prices/       # 其他市场的股票价格快照
├── reference_results/             # 整合前结果，供历史对照
├── docs/                          # 项目说明文档
├── outputs/                       # 运行生成的结果，不纳入 Git
│   ├── <dataset>/<profile>/       # 分析结果、输入清单、日志与运行参数
│   └── hs300_long100/ml_portfolios/
│                                  # 模型检查点、组合结果与机器学习报告
└── .cache/                        # 本地环境和绘图库缓存，不纳入 Git
```

### 模块职责与调用关系

| 层次 | 入口或模块 | 主要职责 |
|---|---|---|
| 命令行入口 | `run.py`、`run_ml.py` | 读取命令行参数，启动对应分析或实验。 |
| 分析调度 | `pipeline.py` | 按配置读取数据、核验交易日历、选择股票池，调用研究引擎并记录模块状态。 |
| 研究计算 | `engines/` | 实现谱分解、相关结构、多尺度分析、协方差估计及基础组合回测。 |
| 机器学习实验 | `ml_experiment/` | 构造当时可得特征，按时间训练和验证模型，生成持仓并评价样本外表现。 |
| 数据与结果核验 | `verify_data.py`、`ml_experiment/checks.py` | 分别检查输入数据完整性，以及机器学习实验的信息边界和交易模拟实现。 |
| 报告与导出 | `report_long_history.py`、`ml_experiment/report.py` | 生成图表、结果表、中文报告与离线浏览页面。 |

主要流程如下：

```text
结构与风险分析：
  datasets.json + data/<dataset>/
    → run.py → pipeline.py → engines/
    → outputs/<dataset>/<profile>/

长历史补充报告：
  outputs/hs300_long100/full/（七个正式分析模块完成后）
    → report_long_history.py
    → supplement/、中文报告、离线页面

机器学习组合实验：
  长历史 full/ 中的收益矩阵、RMT 基线及风险评价组合清单
    → run_ml.py → ml_experiment/（特征 → 训练 → 配置与回测 → 评价与校验）
    → outputs/hs300_long100/ml_portfolios/
```

结构与风险分析使用 `run.py`；长历史补充报告使用 `report_long_history.py`；机器学习实验使用 `run_ml.py`。机器学习实验依赖长历史分析已生成的输入，运行顺序见下文。

`data/` 保存输入，`outputs/` 保存本次计算结果，`reference_results/` 保存历史对照。分析入口读取原始价格并将清洗记录另存到结果目录；`download_long_history.py` 是手动触发的独立下载入口，不会随常规分析自动执行。

## 运行环境与包依赖

需要 Python 3.11 或以上。价格文件采用 pandas 3.x 的 pickle 格式读取，因此要求 pandas 3.x。基础分析与机器学习实验分别使用独立虚拟环境。

### 依赖包

下表与仓库中的 `requirements.txt`、`requirements_ml.txt` 对应。`—` 表示该环境未直接声明此依赖。

| 包 | 基础分析版本要求 | 机器学习版本要求 | 用途 |
|---|---|---|---|
| NumPy | `>=2.0,<2.5` | `>=2.1,<2.5` | 数组运算、收益矩阵与特征值分解。 |
| pandas | `>=3.0,<3.1` | `>=3.0,<3.1` | 价格读取、日期对齐、滚动统计与结果表。 |
| SciPy | `>=1.15,<2` | `>=1.15,<1.17` | 约束优化、统计检验与层次聚类。 |
| Matplotlib | `>=3.9,<4` | `>=3.9,<4` | 特征值谱、净值、回撤等图表。 |
| seaborn | `>=0.13,<1` | — | 相关矩阵热图等统计图形。 |
| Pillow | `>=10,<13` | `>=10` | 图片读取、图像导出与结果校验。 |
| scikit-learn | — | `>=1.6,<1.9` | Ridge、Elastic Net 等传统模型。 |
| PyTorch（`torch`） | — | `>=2.6,<3` | 循环网络、TCN、Transformer 和图模型。 |
| LightGBM | — | `>=4.5,<5` | 梯度提升树收益预测。 |
| XGBoost | — | `>=2,<4` | 梯度提升树收益预测。 |
| joblib | — | `>=1.3` | 传统模型与预处理参数的保存和加载。 |

`requirements_ml_lock.txt` 另行记录机器学习实验环境的精确版本，包含直接和间接依赖。复现实验时使用该锁定文件；需要重新选择兼容版本时，使用 `requirements_ml.txt` 中的版本范围。两个文件是同一机器学习环境的两种安装方式。

### 基础分析环境

在项目根目录执行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run.py hs300
```

### 机器学习实验环境

安装已保存实验所用的锁定版本：

```bash
python3 -m venv .cache/ml_venv
.cache/ml_venv/bin/python -m pip install -r requirements_ml_lock.txt
```

如需按版本范围安装，将最后一条命令中的文件名替换为 `requirements_ml.txt`。机器学习实验使用长历史分析生成的输入，具体准备步骤见后文“长历史100股的完整结果与报告”和“机器学习投资组合与RMT比较”。

### 数据完整性检查

复制或克隆仓库后，可直接核对数据文件：

```bash
python3 verify_data.py
```

该检查脚本只使用 Python 标准库。`argparse`、`pathlib`、`hashlib`、`json`、`urllib`、`zipfile` 等也属于标准库，无需单独安装。

## 新增长历史100股：hs300_long100

从原有288股研究股票池挑选，非最新沪深300动态成分。每股保留经上市日期核对后的完整可用日频复权价格，最早单股记录为1991-01-29；100股共同区间为2002-02-05至2026-11-09，6083个价格日、6082个收益日，约24.8年。已排除未完成的当日日线与一只超过500日连续零收益的候选。

```bash
/opt/anaconda3/envs/ml/bin/python run.py hs300_long100
```

详细数据、100股名单和来源证据在`data/hs300_long100/`，下载代码为`download_long_history.py`。重新抓取并保存到另一个新目录：

```bash
/opt/anaconda3/envs/ml/bin/python download_long_history.py --end 2026-11-09 --dataset-id hs300_long100_refresh
```

原始响应和上市日期核对记录均已保存。报价采用Yahoo复权收盘，上市日期来自东方财富f189字段；没有跨数据源拼接价格。存在固定股票池、历史可得性和事后质量筛选偏差。
## 长历史100股的完整结果与报告

```bash
python run.py hs300_long100 --analysis all
python report_long_history.py hs300_long100
```

第一条运行七个正式分析模块，τ=1–250；第二条生成完整中文报告、离线PNG图表目录、全历史逐年比较、较大τ的前十个时间分量、全部100个日频时间分量/载荷、完整矩阵、每τ100次独立循环平移对照、数值校验。

- 主结果：`outputs/hs300_long100/full/完整分析报告.md`
- 图表和数据目录：`outputs/hs300_long100/full/index.html`（浏览器打开）
- 文件哈希清单：`outputs/hs300_long100/full/artifact_manifest.csv`

**交易日历核验：**原始下载价格中有81个共同日期，100股成交量全部为0且调整收盘价全部不变。分析入口利用同目录下的 `daily_details/*.csv.gz` 联合核验后排除这些日期，保存 `excluded_nontrading_dates.csv`，不改动原始价格。单只股票停牌不会因此被排除；缺少成交量证据时不猜测市场休市日。最终共同价格日6002、收益日6001，价格区间仍是2002-02-05至2026-11-09。原始文件统计的6083个价格记录包含上述供应商日期。

报告脚本要求七个正式分析模块运行完成，不能把 `--quick` 预览作为正式结果。生成的结果保存在 `outputs/`，不会自动上传到 Git。

## 机器学习投资组合与RMT比较

本次使用 `hs300_long100` 的100股、6001个日收益。统一评价区间为2010-01-29至2026-11-09，保留过去约六年用于训练与时间验证；所有特征和PCA使用当时可得信息。20种方法包括树模型、RNN/LSTM/GRU、TCN、分块Transformer、GNN、直接权重学习、HRP与KL风险预测。

```bash
python3 -m venv .cache/ml_venv
.cache/ml_venv/bin/python -m pip install -r requirements_ml_lock.txt
.cache/ml_venv/bin/python run_ml.py
```

- 中文报告：`outputs/hs300_long100/ml_portfolios/机器学习组合完整报告.md`
- 全部PNG与数据：`outputs/hs300_long100/ml_portfolios/index.html`
- 逐日净值：`outputs/hs300_long100/ml_portfolios/normalized_equity.csv`
- 检查点、每年度参数和成熟标签截止日保存在各模型目录。随机方法运行三个种子，主曲线使用平均目标权重。

已完成的相同版本模型会复用；只更新报告运行 `run_ml.py --report-only`。可用 `--models gru,lstm` 运行指定方法，`--prepare-only` 只准备特征。改动训练代码或参数时使用新 `--output`，保留旧结果。

主实验留出完整一个交易日再成交，并重新计算同口径RMT基线；原RMT日收益也从相同起点重绘。另保存额外执行延迟、交易费用、随机种子及区块重采样比较。股票池固定、历史区间已被查看，结果属于探索性研究。
