# K–L / PCA 模态分析的文献延伸与本数据的研究路线

本笔记基于 `data300_price` 中 288 只股票、2018-12-28 至 2026-09-28 的共同日收盘价，讨论 Chen–Ma (2018) 的特征值、特征向量、时间模态与频谱分析之外的用途。以下文献结论来自其他市场和时期，不能直接当作本数据的发现。

## 1. 市场模态、群组模态与股票载荷

- [Plerou et al., *Random matrix approach to cross correlations in financial data*, Physical Review E 65, 066126 (2002)](https://doi.org/10.1103/PhysRevE.65.066126)：用随机矩阵基准区分显著特征模态，发现最大模态对应市场共同作用，其余偏离随机基准的模态可与企业群组联系，并考察特征向量的时间稳定性。
- [Gopikrishnan et al., *Quantifying and interpreting collective behavior in financial markets*, Physical Review E 64, 035106 (2001)](https://doi.org/10.1103/PhysRevE.64.035106)：用大特征值对应的特征向量辨认股票子群，研究其时间稳定性与群组收益行为。

本数据可做：输出每个显著模态的股票载荷排序与热力图；用逆参与率 `IPR = sum_i a_i^4` 和有效参与数 `1/IPR` 衡量模态是广泛分布还是集中于少数股票。已有的 τ=1 前十模态时间曲线告诉我们“何时活跃”，载荷可回答“哪些股票参与”。没有行业分类时只能称为数据驱动群组，不应直接命名为行业。

## 2. 滚动市场状态与模态稳定性

- [Ren & Zhou, *Dynamic evolution of cross-correlations in the Chinese stock market*, PLOS ONE 9, e97711 (2014)](https://doi.org/10.1371/journal.pone.0097711)：在中国 A 股的滚动窗口上分析相关系数、特征值和特征向量的演化及显著参与股票。
- [Münnix et al., *Identifying States of a Financial Market*, Scientific Reports 2, 644 (2012)](https://doi.org/10.1038/srep00644)：比较不同时段的相关结构，并对市场状态进行分类。

本数据可做：以 τ=1 的日收益为主，先用 500–750 个交易日的窗口、每 20 日更新一次，绘制最大特征值份额、有效模态数、载荷集中度和相邻窗口模态重合度。窗口长度必须与 288 只股票的维度一起评估；较短窗口会使估计不稳定。比较第 2–10 模态时应用符号不变的载荷内积或子空间夹角，不能只按同一排名认定为同一经济模态。市场状态聚类属于探索性结果，需要检验对窗口长度和股票集合的敏感性。

## 3. 去市场模态后的群组与网络

- [Almog et al., *Mesoscopic Community Structure of Financial Markets Revealed by Price and Sign Fluctuations*, PLOS ONE 10, e0133679 (2015)](https://doi.org/10.1371/journal.pone.0133679)：研究从相关结构中识别中观股票群组及其稳健性。
- [Kenett et al., *Dominating Clasp of the Financial Sector Revealed by Partial Correlation Analysis of the Stock Market*, PLOS ONE 6, e15032 (2011)](https://doi.org/10.1371/journal.pone.0015032)：使用偏相关分析市场共同影响之外的联系；论文也明确说明偏相关不等于因果关系。

本数据已画出扣除最大共同模态的 τ=1 残差协方差。下一步可在残差**相关矩阵**上做稳健群组识别、比较组内与组间联系，再用时段切分或 bootstrap 检查群组是否稳定。若要验证是否对应真实行业，需要补充按日期匹配的行业标签；当前无标签聚类不能推出行业归属。

## 4. 相关矩阵去噪与风险估计

- [Laloux et al., *Noise Dressing of Financial Correlation Matrices*, Physical Review Letters 83, 1467 (1999)](https://doi.org/10.1103/PhysRevLett.83.1467)：显示样本相关矩阵中大量特征值与随机矩阵预测相近，提示直接使用高维样本相关矩阵的风险。
- [Bun, Bouchaud & Potters, *Cleaning large correlation matrices: Tools from Random Matrix Theory*, Physics Reports 666, 1–109 (2017)](https://doi.org/10.1016/j.physrep.2016.10.005)：综述并实证比较特征值去噪和相关矩阵估计方法。

本数据可做：在仅使用过去数据的滚动训练窗口中，比较原始协方差、收缩估计和模态去噪矩阵预测下一窗口风险的误差。必须做严格样本外验证；要研究真实可交易组合，还需确认收盘价是否复权、成分股是否存在存活者偏差，并纳入交易约束和成本。因此该方向应晚于结构分析。

## 5. 跨收益跨度的模态是否真正变化

对 τ=1、5、20、60、120、250 分别比较特征值份额、模态载荷、时间分量和参与率，可研究不同持有跨度的共同变化。但是 τ>1 的相邻收益窗口大量重叠，会自动使时间曲线平滑并压缩有效独立样本数。应同时使用不重叠收益、共同日期打乱对照，以及载荷/子空间重合度；不可把特征值排名相同当成模态身份相同。此前探索性计算中，τ=250 的前十模态方差份额为 89.5%，20 次日期打乱对照的中位数为 89.9%，说明该现象不能单独作为长期经济结构证据。

## 推荐先做的两项

1. **τ=1 前十模态的股票载荷与稳定性**：载荷表、载荷热图、有效参与数、早晚时段及 bootstrap 的重合度。这一步最直接补足现有时间模态图缺失的“股票含义”。
2. **滚动市场结构**：固定 τ=1，比较 500、750 日窗口与不同更新步长；用同一组图展示市场模态份额、模态参与数、残差群组稳定性，并对照市场总体收益和波动。先报告描述性变化，再讨论经济解释。

**完成状态（2026-10-15）**：两项均已实现，入口为 `run_pipeline.py` 调用的 `mode_structure_extension.py`，参数记录在 `config.json`。结果与方法限制见 `outputs/mode_structure_report.md`；新图为 `outputs/figures/11_tau1_top10_loadings.png` 至 `14_rolling_mode_stability.png`，逐股和逐窗口 CSV 在 `outputs/tables/`。

**第 3、4 项完成状态（2026-10-15）**：入口为 `run_pipeline.py` 调用的 `residual_and_risk_extension.py`，参数记录在 `config.json`。第 3 项以 2023 年前数据独立发现八个残差相关群组，再在 2023 年以后数据检验，并计算早晚独立分组及不重叠滚动窗口的调整兰德指数。第 4 项以过去 500/750 日估计协方差，评估接下来 20 日的组合波动误差与整体协方差伪评分。结果、限制和文件清单见 `outputs/residual_and_risk_report.md`；图 `15`–`18` 为 PNG，逐股票、逐窗口、逐测试块数据在 `outputs/tables/`。

派生图统一输出 PNG，数值输出 CSV，实验配置和随机种子记录在 pipeline 中。对任何“预测”或“危机预警”的说法，必须另做预先定义的样本外检验。
