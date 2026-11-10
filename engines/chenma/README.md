# Chen–Ma (2018) 日频原始价格研究

本管线读取 `../data300_price` 的股票日收盘价，只使用原始价格，以 τ=1–250 个交易日的全部整数跨度构造重叠对数收益。逐跨度计算股票相关矩阵、K–L 模态、功率谱和谱斜率。另有共同日期打乱对照及两个时间段的检查。

所有图均为 PNG。输入文件只读；派生结果集中写入 `outputs/`：

- `outputs/report.md`：中文研究报告；
- `outputs/figures/*.png`：图；
- `outputs/tables/*.csv`：每个跨度、模态、特征值及输入哈希；
- `outputs/run_metadata.json`：运行参数和环境信息。

新增 `outputs/figures/07_covariance_tau1.png`：τ=1 日原始对数收益的协方差热力图。左侧为股票代码顺序的完整矩阵；右侧用同一个股票排列重排行和列、遮蔽对角线，并按非对角元素第 99 百分位截断色标。排序先去除相关矩阵的最大共同模态，再用残差相似性做 8 组谱聚类，展示的仍是未标准化的原协方差。组及股票顺序记录在 `outputs/tables/covariance_tau1_order.json`。

`outputs/figures/08_market_removed_covariance_tau1.png`：在相同股票顺序和色标下比较原始协方差与去掉最大共同主成分后的残差协方差。两侧均遮蔽对角线。去除步骤在标准化日收益上进行，之后乘回原始标准差，因此图中仍为协方差单位。

`outputs/figures/09_market_mode_tau1_time.png`：用实际日期展示 τ=1 日最大特征值（市场）模态的时间分量，以及其 20 日滚动 RMS。图中把单位 L2 范数的时间分量乘以 √T，便于按单位时间标准差读取起伏；符号朝向与股票等权平均标准化收益一致。

`outputs/figures/10_tau1_top10_time_modes.png`：按特征值从大到小展示 τ=1 的前 10 个 K–L 时间模态。纵轴为各模态单位时间方差的无量纲得分，所有面板共用同一刻度。逐日数值见 `outputs/tables/tau1_top10_time_modes.csv`，对应特征值与方差份额见 `outputs/tables/tau1_top10_eigenmodes.csv`。

`outputs/mode_structure_report.md`：τ=1 前十模态的股票载荷、有效参与股票数、早晚时段及移动块 bootstrap 稳定性，以及 500/750 日滚动窗口的市场结构。图 `outputs/figures/11_tau1_top10_loadings.png` 至 `14_rolling_mode_stability.png` 全部为 PNG；逐股载荷、模态重合度、bootstrap 和逐窗口指标在 `outputs/tables/` 的同名 CSV 中。新分析由 `mode_structure_extension.py` 实现，参数保存在 `config.json` 的 `mode_structure` 字段；运行主 pipeline 会一并重算。

`outputs/residual_and_risk_report.md`：独立聚类检验去除最大共同模态后的股票群组能否跨早晚样本保留；再用滚动历史窗口比较原始、对角收缩及谱截断协方差对未来 20 日的风险估计。图 `outputs/figures/15_residual_correlation_holdout.png` 至 `18_equal_weight_risk_forecast.png` 全部为 PNG。分组、留出与置换结果在 `outputs/tables/residual_group_*`，逐测试块风险评分和方法摘要在 `outputs/tables/risk_*`。分析由 `residual_and_risk_extension.py` 实现；参数在 `config.json` 的 `residual_group` 与 `risk_estimation` 字段中。

## 运行

需要 Python 3.10+、NumPy、pandas 3.x、Pillow。数据 pickle 文件由 pandas 3.x 写出。

```bash
python run_pipeline.py --config config.json
```

论文使用 36 秒采样的美股盘中数据；本管线为日频方法迁移，不是原论文数值的精确复现。分析是描述性的，不构成交易策略。
