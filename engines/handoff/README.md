# 沪深300数据、RMT 与投资组合分析

最终课程汇报见 `presentation_output/经济物理_Project3_RMT与投资组合.pptx`（34页，含讲稿备注）。配套文件为《逐页讲稿》《答辩与复现说明》和《AI_Audit_诊疗记录》。新增补充实验由 `project_completion.py` 生成，现场改参用 `defense_demo.py`。API下载100股已经实跑，结果单独存于 `api_refresh/`；相关核对记录保存在 `presentation_data/`。

当前已配置的 `.venv` 用于CSV阶段复现，依赖见 `requirements_csv.txt`。下文原始pickle读取需要pandas 3环境，依赖口径见原有 `requirements.txt`。

`analyze_hs300.py` 审计三组逐股 pickle，构造等权市场代理，计算收益、回撤、波动率、市场宽度、截面离散度、滚动平均相关性和最大相关矩阵特征值占比。

这批 pickle 需要 `pandas>=3.0`。在本目录运行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python analyze_hs300.py
```

默认输入为 `../Data/沪深300`，输出保存在当前目录的 `tables/`、`figures/` 及 `analysis_summary.json`。原始数据不会被修改。

第二阶段相关矩阵、随机矩阵理论和投资组合分析：

```bash
.venv/bin/python rmt_analysis.py
```

它使用 `tables/07_log_return_matrix.csv`，产生 `rmt_tables/`、`rmt_figures/`、`rmt_summary.json` 和初学者可读的 `rmt_report.md`。回测默认使用固定 100 只股票、504 日训练窗、21 日调仓、单股上限 5% 和单边换手 10 bps 成本。

第三阶段为股票补充当前名称和行业标签，并检验 RMT 组模式中的行业层级：

```bash
.venv/bin/python fetch_industry_metadata.py
.venv/bin/python industry_structure_analysis.py
```

主要结果见 `industry_structure_report.md`、`rmt_figures/12`--`15` 和 `rmt_tables/14`--`20`。聚类只使用价格收益数据，行业标签仅在聚类完成后用于解释和检验，避免把行业答案预先放进聚类。行业标签为 2026-10-08 查询到的当前分类，不是逐年历史分类。

指标口径：

- 可交易的等权代理：先把每只股票的对数收益转为简单收益，每日等权平均，再连乘；这等价于日频再平衡且未扣交易成本。
- 市场模式代理：288 只股票每日对数收益的截面平均。图中另列“成分股财富几何平均”，避免将两种口径混为一谈。
- 年化交易日：242 日。
- 日收益率原文件经核对为复权价格的对数收益率。
- 滚动平均相关窗口：60 交易日。
- 最大特征值占比窗口：120 交易日，每 5 交易日计算一次，中间日期前向填充仅用于显示。
- 日内成交量、行业分类、自由流通市值和沪深300官方指数值不在原始 pickle 中；当前行业分类由外部公开接口补充并另存，不会改动原始数据。

`tables/09_individual_return_anomalies.csv` 单列绝对单日收益超过 25% 的观测。它们可能来自长期停牌后复牌、公司行为或 Yahoo 复权处理，应在矩阵分析前逐条复核，不要直接默认为普通交易日波动。
