# Manuscript context

## Identity and scope
课程投资学报告《高贝塔股票是否获得更高收益？》，估计期 2015-2019、检验期 2020-2024，200 只沪深 A 股、五个固定历史贝塔组。正文载体 report/报告.md，生成模板 src/report_content.py，PDF 由 scripts/build_report_pdf.py 导出。

## Canonical terminology
历史贝塔用于事前分组；检验期贝塔描述实际风险暴露。收益差 H-L 的零均值检验与零 alpha 检验不同；时间序列组合 alpha 与横截面 SML 截距不同。MAE/RMSE 为 CAPM 指定基准线偏差，拟合残差相对自由估计经验线。收益数据为小数，文字中的月收益显示百分数。

## Current author state
- [working] 贝塔及平均收益均未保持严格递增；H-L 收益差和 alpha 均未显著偏离零。来源：当前稿及 hl_diff.csv。
- [working] 未拒绝 H-L 零 alpha 不代表模型得到验证，也不替代五组联合检验。来源：当前稿第四、七节。
- [working] 可得成员平均为统计口径，退出清偿/换股尚未核实。来源：当前稿数据处理和局限。
- 用户授权本轮学术审查和中文润色；这一授权不等于对新增科学解释的逐项接受。本轮没有添加机制或外部文献结论。

## Evidence roles
hl_diff.csv：月均差 -0.202%、HAC 区间 [-1.136%,0.732%]、alpha -0.248%/月、p=0.581。
portfolio_stats.csv：G4 alpha p=0.029（未经多重比较调整的单项结果）；并非所有组合均无显著 alpha。
sml_fit.csv、capm_deviations.csv：经验拟合和 CAPM 基准距离比较。s4_beta.py：三只示例来自名单查询时仍在市的样本子集。
s5_robustness.py：全损情景替换退出月收益为 -100%，不是在已观测损失之外额外扣减。

## Claim boundaries
仅解释本样本的已实现收益；不把相关、图形拟合或聚合效果改为因果。保留报价来源、代理期限和缺失处理限制。网站入口不是经核实的理论或方法文献支持。

## Open decisions
退出现金流、后复权分红算法、理论及计量文献的具体引文支持尚未核实。成员信息、人工复核和完整 AI 原始对话导出保持待办。多重比较与 alpha 联合检验仅建议，尚未执行。

## Superseded entries
旧版正收益差、收益单调上升及“组合更偏离 CAPM”的推断不再作为当前稿结论。润色前稿保存于 versions/，不覆盖原始证据。
