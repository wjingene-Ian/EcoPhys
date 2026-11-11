# Project 3：统一数据与命令行分析

两套 Project 和之前 Chen–Ma 扩展已整理到这个目录。实际运行只使用本目录的数据；移动整个 `project3` 文件夹后，仍可用相同的相对路径运行。统一入口是 `run.py`，纯命令行，图只输出 PNG。

默认使用选定数据文件夹内的**全部可用股票**，不会自动抽取100股。这里只包含已有价格快照，没有自动联网下载。

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

## 如何复现交接包的旧100股股票池

```bash
/opt/anaconda3/envs/ml/bin/python run.py hs300 --analysis rmt --assets 100
/opt/anaconda3/envs/ml/bin/python run.py hs300 --analysis structure --assets 100
/opt/anaconda3/envs/ml/bin/python run.py hs300 --analysis scales --assets 100
```

这三个命令显式使用交接包保存的固定100股名单和顺序。结果写入 `outputs/hs300/sample100/`，与默认288股的 `full/` 分开。数值环境和随机实验执行次序可能造成小差异；`reference_results/`保留旧结果供对照。

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

## 两套历史方法的口径

整合保留两套历史回测规则，以免把已有结果混为同一实验：

| 项目 | basic（原分析） | rmt（交接包方法） |
|---|---|---|
| 默认股票数 | 全部 | 全部；原交接包默认100，现须显式`--assets 100` |
| 协方差估计输入 | 简单日收益 | 对数日收益 |
| 实际持有收益 | 简单日收益 | 对数收益转换为简单收益 |
| 策略 | 原始、RMT截尾 | 等权、原始、硬过滤、RMT截尾 |
| 年化天数 | 252 | 242 |
| 训练/调仓/权重上限 | 504日/21日/5% | 504日/21日/5% |
| 初次建仓换手 | 1，扣10bps | 0，不扣首次成本 |
| 压力分析 | 波动最高10%、滚动收益最低10% | 预选2022、2024和2025事件窗口 |

`rmt`在100股及以下默认保留旧SLSQP求解器；超过100股时调用原项目已有的投影梯度求解器，使用同样的权重约束和交接包的岭项。可用`--solver slsqp`或`--solver projected`显式选择。求解器记录在结果摘要中，数值解不保证逐位相同。

其他结构分析默认使用全部入选股票。所有行业标签均是现有快照，不是历史时点行业。中心DMA只用于结构研究，未进入回测；没有HF1MA平滑。普通聚类树不构成Parisi超度量性证明。随机参照分别记录其生成方式，不能混用MP理论上界、独立打乱阈值和保留同期相关的共同日期打乱结果。

## 文件组织与来源

- `run.py`、`pipeline.py`：当前统一入口和适配代码。
- `engines/basic/`：之前基础复现的计算函数。
- `engines/handoff/`：交接包中的分析函数及历史辅助脚本。
- `engines/chenma/`：之前KL、滚动结构与风险预测代码。
- `data/<简称>/prices/`：真实复制的价格文件，没有指向旧目录的软链接。
- `data/hs300/archived_returns/`、`archived_zscores/`：沪深原始收益与标准化档案。
- `data/metadata/`：沪深行业、指数响应和旧100股名单。
- `reference_results/`：整合前的数值结果，仅作历史对照，未冒充本次新计算。
- `docs/handoff/`：封存交接说明，其中提及的旧目录和旧48页PPT属于历史版本。

只使用统一入口运行当前项目。`engines/`内保留的旧独立入口可能仍描述原项目目录；联网下载、PPT构建等旧辅助脚本不会被统一入口自动调用。

## 换一台机器运行

需要Python 3.11或以上。原始pickle由pandas 3.x写出，因此本项目要求pandas 3.x。在项目目录执行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run.py hs300
```

复制完成后可核对数据完整性：

```bash
python3 verify_data.py
```

本地已有环境若出现NumPy/SciPy版本兼容提示，建议使用上述独立环境与依赖范围。项目本身不会修改本机已有环境。

## 本次整合验证

- 十组数据的完整基础分析均已运行，主要回测指标与原项目保存结果一致。
- 沪深288股与法国38股的所有模块通过低重复次数的流程预览；法国行业步骤因无标签明确跳过。
- 标普500的473股KL/滚动/风险模块通过流程预览，绘图不再固定为288股。
- 旧100股组合入口与交接包主要指标最大差约0.000352，属于数值环境复跑差异。
- 检查了四策略权重约束、13个尺度矩阵的对称性/单位对角线/半正定性、前十时间分量单位方差，以及PNG输出格式。
- `validation_report.json`保存上述核对结果。`full_quick`与`sample100_quick`中的随机检验仅用于流程验证；尚未给十个市场都执行正式次数的全部高级分析。

测试生成的运行记录保留实际执行时的路径和代码哈希；复制或移动项目后重新运行会记录新路径，数据哈希可用于核对是否仍为同一批输入。

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

第一条运行七个正式分析模块，τ=1–250；第二条生成完整中文报告、离线PNG图表目录、全历史逐年比较、较大τ的前十个时间分量、全部100个日频时间分量/载荷、完整矩阵、每τ100次独立循环平移对照、数值校验和完整结果压缩包。

- 主结果：`outputs/hs300_long100/full/完整分析报告.md`
- 图表和数据目录：`outputs/hs300_long100/full/index.html`（浏览器打开）
- 全部结果：`outputs/hs300_long100/hs300_long100_all_results.zip`
- 文件哈希清单：`outputs/hs300_long100/full/artifact_manifest.csv`

**交易日历核验：**原始下载价格中有81个共同日期，100股成交量全部为0且调整收盘价全部不变。分析入口利用同目录下的 `daily_details/*.csv.gz` 联合核验后排除这些日期，保存 `excluded_nontrading_dates.csv`，不改动原始价格。单只股票停牌不会因此被排除；缺少成交量证据时不猜测市场休市日。最终共同价格日6002、收益日6001，价格区间仍是2002-02-05至2026-11-09。原始文件统计的6083个价格记录包含上述供应商日期。

报告脚本要求七模块正式运行完成，不能把 `--quick` 预览打包成正式结果。全部PNG、CSV、NPZ、日志、参数和实现快照都包含在结果压缩包；原始数据已在仓库 `data/`，不重复打包。`outputs/` 仍为生成目录，不自动上传Git。


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
- 完整压缩包：`outputs/hs300_long100/hs300_long100_ml_results.zip`
- 检查点、每年度参数和成熟标签截止日保存在各模型目录。随机方法运行三个种子，主曲线使用平均目标权重。

已完成的相同版本模型会复用；只更新报告运行 `run_ml.py --report-only`。可用 `--models gru,lstm` 运行指定方法，`--prepare-only` 只准备特征。改动训练代码或参数时使用新 `--output`，保留旧结果。

主实验留出完整一个交易日再成交，并重新计算同口径RMT基线；原RMT日收益也从相同起点重绘。另保存额外执行延迟、交易费用、随机种子及区块重采样比较。股票池固定、历史区间已被查看，结果属于探索性研究。
