"""S7 的报告正文：直接引用计算结果，分页标记供 PDF 导出使用。"""
from pathlib import Path
import pandas as pd
from src.common import ROOT, load_csv, load_json, project_paths


def pct(x, n=3):
    return f"{x*100:.{n}f}%"


def table(frame):
    return frame.to_markdown(index=False)


def write_report(cfg):
    paths = project_paths(cfg)
    t = paths["tables"]
    sample = load_csv(paths["processed"] / "sample.csv")
    panel = load_csv(paths["processed"] / "panel_monthly.csv")
    universe = load_csv(paths["interim"] / "universe.csv")
    universe["industry_date"] = pd.to_datetime(universe["industry_date"], errors="coerce")
    cand = load_csv(paths["interim"] / "candidates.csv")
    beta = load_csv(t / "beta_est.csv")
    check = load_csv(t / "beta_check.csv")
    port = load_csv(t / "portfolio_stats.csv")
    stock = load_csv(t / "stock_test.csv")
    exclusions = load_csv(t / "stock_test_exclusions.csv")
    hl = load_csv(t / "hl_diff.csv").iloc[0]
    stab = load_csv(t / "beta_stability.csv").iloc[0]
    sml = load_csv(t / "sml_fit.csv")
    dev = load_csv(t / "capm_deviations.csv")
    robust = load_csv(t / "robustness.csv")
    exits = load_csv(paths["processed"] / "exit_audit.csv")
    s3 = load_json(paths["processed"] / "s3_manifest.json")["extra"]
    identity = load_csv(t / "beta_identity.csv").abs_diff.max()
    pr = load_csv(t / "portfolio_returns.csv")
    market = (pr.mret-pr.rf).mean()
    pi, pp = sml.iloc[0], sml.iloc[1]
    di, dp = dev.iloc[0], dev.iloc[1]
    historic_monotonic = port.mean_excess_ret.is_monotonic_increasing
    actual_monotonic = port.sort_values("beta_test").mean_excess_ret.is_monotonic_increasing
    hist_text = "严格递增" if historic_monotonic else "并非严格递增"
    test_text = "严格递增" if actual_monotonic else "并非严格递增"
    beta_text = "保持递增" if bool(stab.group_beta_monotonic) else "未保持严格递增"
    interval_text = "包含零" if hl.ci95_hac_low <= 0 <= hl.ci95_hac_high else "不包含零"
    pricing_text = "不能拒绝高低组 α 为零" if hl.p_alpha_diff >= .05 else "在 5% 水平拒绝高低组 α 为零"
    ex_text = "无" if exclusions.empty else "；".join(f"{r.code} {r.name}（{r.n_obs} 月）" for r in exclusions.itertuples())
    flow = pd.DataFrame({"步骤": ["冻结日沪深 A 股", "历史行业分层候选", "估计期合格", "最终样本", "进入个股检验回归"],
                         "数量": [len(universe), len(cand), s3["eligible_n"], len(sample), len(stock)]})
    short = port[["group", "avg_hist_beta", "beta_test", "se_beta", "mean_excess_ret", "alpha_monthly", "p_alpha"]].copy()
    for k in ["avg_hist_beta", "beta_test", "se_beta", "p_alpha"]:
        short[k] = short[k].map(lambda x:f"{x:.3f}")
    for k in ["mean_excess_ret", "alpha_monthly"]:
        short[k] = short[k].map(pct)
    short.columns = ["组合", "历史 β", "检验 β", "β标准误", "月均超额收益", "月度 α", "α的p值"]
    checks = check[["level", "code", "n_obs", "cov", "var", "beta_def", "beta_ols", "abs_diff"]].copy()
    for k in ["cov", "var", "beta_def", "beta_ols"]:
        checks[k] = checks[k].map(lambda x:f"{x:.6f}")
    checks.abs_diff = checks.abs_diff.map(lambda x:f"{x:.1e}")
    checks.columns = ["层次", "代码", "月数", "协方差", "市场方差", "定义β", "OLS β", "绝对差"]
    missing_rows = []
    for label, start, end in [("估计期", "2015-01", "2019-12"), ("检验期", "2020-01", "2024-12")]:
        d = panel[panel.period.between(start,end)]
        for reason, n in d[d.ret.isna()].return_status.value_counts().items():
            missing_rows.append({"阶段": label, "收益缺失原因": reason, "股月数": n})
    missing_table = pd.DataFrame(missing_rows)
    tails = exits[exits.delist_in_test].sort_values("unresolved_price_months",ascending=False).head(3)
    tails = tails[["code", "name", "last_quote_date", "delist_date", "unresolved_price_months"]].fillna("未退出/未知")
    tails.columns = ["代码", "简称", "末次有效月线", "退出日期", "未知价格月数"]
    fit_table = sml[["scope", "n", "intercept", "slope", "r2", "resid_sd"]].copy()
    for k in ["intercept", "slope", "resid_sd"]:
        fit_table[k] = fit_table[k].map(lambda x:f"{x:.5f}")
    fit_table.r2 = fit_table.r2.map(lambda x:f"{x:.3f}")
    fit_table.columns = ["口径", "观察点", "截距", "斜率", "R²", "拟合残差标准差"]
    dev_table = dev[["scope", "n", "mae", "rmse"]].copy()
    for k in ["mae", "rmse"]:
        dev_table[k] = dev_table[k].map(pct)
    dev_table.columns = ["口径", "观察点", "基准线偏差 MAE", "基准线偏差 RMSE"]
    robust_short = robust[["scenario", "hl_mean", "alpha_diff", "p_alpha_diff"]].copy()
    robust_short.scenario = ["严格缺失（主口径）", "内部缺月末价延续", "末价延续至退出/无额外损失", "退出月全损压力情景", "无风险利率简单折算"]
    for k in ["hl_mean", "alpha_diff"]:
        robust_short[k] = robust_short[k].map(pct)
    robust_short.p_alpha_diff = robust_short.p_alpha_diff.map(lambda x:f"{x:.3f}")
    robust_short.columns = ["情景", "月均 H-L", "月度 α", "α的p值"]
    ci_table = pd.DataFrame([
        ["月均收益差",pct(hl.mean_diff_monthly)],
        ["算术年化均值（12×月均差）",pct(hl.mean_diff_annual,2)],
        ["t 统计量",f"{hl.t_stat:.3f}"],
        ["95% t 区间",f"[{pct(hl.ci95_t_low)}, {pct(hl.ci95_t_high)}]"],
        ["95% Newey-West 区间",f"[{pct(hl.ci95_hac_low)}, {pct(hl.ci95_hac_high)}]"],
        ["95% 分块自助区间",f"[{pct(hl.ci95_boot_low)}, {pct(hl.ci95_boot_high)}]"],
    ],columns=["指标","结果"])
    page1 = f"""# 高贝塔股票是否获得更高收益？

投资学第二次团队作业 | 基于个股与投资组合的 CAPM 实证分析

组员与学号：待组员据实填写。修订日期：2026 年 10 月 28 日。

## 摘要

本文使用 2015-2019 年数据估计历史贝塔，按贝塔数值将股票分为五个等数量组合，检验 2020-2024 年的市场风险与平均超额收益关系。组合成员在检验期内保持固定。样本为 {len(sample)} 只沪深 A 股，覆盖 {sample.industry.nunique()} 个历史行业层；不以未来是否存续决定入样。名单来源的历史覆盖仍有限制，样本可能存在选择偏差。

按历史贝塔从低到高排列，五组月均超额收益{hist_text}。最高减最低组的收益差为 {pct(hl.mean_diff_monthly)}/月，HAC 95% 区间为 [{pct(hl.ci95_hac_low)}, {pct(hl.ci95_hac_high)}]，{interval_text}。高低组实际贝塔差为 {hl.beta_diff:.3f}，控制市场后的 α 为 {pct(hl.alpha_diff)}/月，p={hl.p_alpha_diff:.3f}；该检验与“平均收益差为零”的检验含义不同。

组合经验证券市场线（SML）的 R² 为 {pp.r2:.3f}，个股为 {pi.r2:.3f}。结合贝塔排序与价差检验，现有样本未显示出严格递增的风险—收益排序，高低组收益差及其 α 也未显著偏离零。个股与组合相对 CAPM 基准线的偏差仍需分别考察。上述结果以可得收益为基础，受未知缺月和退出现金流尚未核实的限制，不能直接解释为可交易策略的实际业绩。

## 样本概览

{table(flow)}

## 研究目标与判断标准

CAPM 描述预期超额收益与市场贝塔的关系。本文分别考察历史贝塔排序的稳定性、样本平均收益与基准预测的关系，以及定价限制的统计证据。分析以描述性比较、敏感性分析和经济解释为主；五个组合点的横截面回归仅用于概括图形关系。
"""
    page2 = rf"""# 一、数据来源与事前样本

## 数据口径

| 数据 | 来源与口径 |
|---|---|
| 上市/退出日期 | 交易所名单的 2026-10-23 保存快照，按日期重建 2019-12-31 股票池 |
| 行业 | baostock 按 2019-12-31 请求；来源更新日期为 {str(universe.industry_date.max())[:10]} |
| 个股价格 | baostock 后复权月线；退出末月以同源日线补齐，验证重叠月价格一致 |
| 市场 | 中证全指全收益 H00985，由月末指数计算月收益 |
| 无风险代理 | 中债国债曲线 1 年期月末年化到期收益率 |

股票池仅保留沪深 A 股，剔除退市名单中的 15 只 B 股，以统一个股收益与市场基准的币种口径。当前池 {len(universe)} 只，{int(universe.status.eq('delisted').sum())} 只在名单查询时已退出；该数量包含可能在 2024 年以后退出的股票，不等于检验期内退市数量。

## 历史行业分层抽样

行业快照保存原始 updateDate，并拒绝无日期或晚于冻结日的记录；未分类 {int(universe.industry.eq('未分类').sum())} 只。按行业数量比例以最大余数法分配候选名额，基础池 {int(cfg['sample']['target_n']*cfg['sample']['oversample_ratio'])} 只，追加池 {cfg['sample']['extra_candidates']} 只。种子 {cfg['project']['seed']}；合格池再以种子加 1 抽取最终样本。只依据估计期有效月数筛选，不使用检验期收益、贝塔或存续情况补样。

每只股票估计期至少 {cfg['sample']['min_est_months']} 个有效配对月。要求较长历史会排除部分较晚上市股票，按数量抽样与市值加权的市场基准也存在权重差异。因此，行业覆盖只能说明样本的分布范围，不能保证其完全代表市场。

## 收益与利率

月收益与超额收益分别定义为：

$$
r_{{i,t}}=\frac{{P^{{\mathrm{{adj}}}}_{{i,t}}}}{{P^{{\mathrm{{adj}}}}_{{i,t-1}}}}-1,\qquad r^e_{{i,t}}=r_{{i,t}}-r_{{f,t}}.
$$

其中，$P^{{\mathrm{{adj}}}}_{{i,t}}$ 为股票 $i$ 在月末 $t$ 的后复权价格。收益保存为小数，图表显示百分数。后复权价格反映分红送转调整；本项目尚未根据逐次分红现金流独立核对 baostock 复权算法，其与现金分红再投资路径的一致性仍待核实。

若年化到期收益率以百分数 $y_t$ 表示，则：

$$
r_{{f,t}}=\left(1+\frac{{y_t}}{{100}}\right)^{{1/12}}-1.
$$

它是 1 年期债券收益率的月度折算代理，不是实际持有 1 个月的国债总收益，也不严格等于月初可锁定的无风险收益。本文保留同月末代理，另做简单除以 12 的敏感性比较。

来源入口：[baostock](http://www.baostock.com/baostock/index.php/Python_API文档)、[腾讯行情](https://gu.qq.com/)、[中证指数](https://www.csindex.com.cn/)、[中债](https://www.chinabond.com.cn/)。具体接口及快照哈希见代码和阶段 manifest。
"""
    page3 = f"""# 二、缺失、停牌与退出审计

## 主口径

未知缺月保留为缺失，不凭前后报价推断整月停牌。只有相邻两个自然月都有有效价格时才计算月收益；缺月后首次出现报价的月份仍保留收益缺失，以免将多月累计涨跌计入单月。非有限价格或低于 {cfg['returns']['min_close']} 的价格视为无效。

停牌确认和终止收益覆盖均须有独立来源的事件记录。目前 verified_monthly_events.csv 尚无人工确认的事件，因此 suspended 为 False 仅表示缺少核实证据。strict 主口径同时排除来源成交量为零的月线：这类记录可能延续原有估值，不能据此认定当月存在可交易的零收益。

## 最终样本收益缺失统计

{table(missing_table)}

source_zero_volume 表示月线来源记录成交量为零；pre_listing、post_delist 是结构性空白；unknown_gap、unknown_tail 只说明来源缺少价格。previous_price_missing 也包括上市首个报价月和复牌后缺少上月价格的月份，不能一律称为停牌。

## 退出与尾部缺口示例

{table(tails)}

表中未知价格月数覆盖估计期和检验期，包含零成交量等缺少有效报价的月份，并非退市前连续缺口的长度。完整逐股记录见 data/processed/exit_audit.csv。正式退出日期之后不再使用行情；退出末月已用同源日线补齐可得实际成交报价。此前仍未知的尾部不会被标成自然退出。末次报价本身不能保证投资者能够按该价退出，退市也不必然对应总损失 100%。合并换股、现金清偿及转板后的实际经济结果仍需逐项核查。

## 组合收益的可解释范围

组合按每月可得成员收益求均值，并保存成员数。这是统计口径，不代表能在月初预知缺失并退出头寸；退出损失未核实的影响通过后文情景对照展示。
"""
    page4 = rf"""# 三、贝塔估计与固定分组

## 含截距时间序列回归

估计以下含截距回归，其中 $R^e_{{m,t}}$ 为市场超额收益：

$$
r^e_{{i,t}}=\alpha_i+\beta_i R^e_{{m,t}}+\varepsilon_{{i,t}}.
$$

2015-2019 年得到历史贝塔，按贝塔实际数值升序、代码作为并列排序规则，等分五组，各 {len(sample)//5} 只。检验期不重新分组，基础分析忽略交易成本及月度再平衡的流动性约束。

检验期回归刻画实际风险暴露，前后贝塔比较用于考察暴露的稳定性。定价检验则需要进一步比较收益与风险补偿。两阶段点估计均采用含截距 OLS；时间序列标准误采用 Newey-West，带宽 {cfg['regression']['hac_maxlags']}。存在缺月的个股回归使用有效观测序列，此时滞后指有效观测次序，并非严格的连续自然月。

个股检验回归至少要求 {cfg['regression']['min_test_months']} 个有效配对月；最终 {len(stock)} 只进入个股图。未能回归的股票：{ex_text}。无法进入个股回归的股票仍留在原组合，其收益可得的月份继续参与组合计算；完整原因见个股检验排除表。

## 定义法为什么与 OLS 一致

含截距 OLS 斜率等于中心化交叉乘积之和除以市场中心化平方和。样本协方差与样本方差均使用 $n-1$ 分母，二者相除后 $n-1$ 消去，得到同一公式：

$$
\widehat{{\beta}}_i=\frac{{\widehat{{\mathrm{{Cov}}}}(r^e_i,R^e_m)}}{{\widehat{{\mathrm{{Var}}}}(R^e_m)}}.
$$

必要条件是相同配对月份、相同收益口径且回归含截距。缺失必须成对剔除；不含截距回归使用未中心化乘积，因此一般不满足这一恒等式。

## 三只股票核对

{table(checks)}

三只示例从名单查询时仍在市的样本股票中选取，分别接近该子样本历史贝塔的 20%、50%、80% 分位。样本月数不同时，定义法必须使用该股票的实际配对月份内市场方差，不能机械地给三只股票使用同一个市场方差。

全体 {len(beta)} 只股票两种方法的最大绝对差为 {identity:.2e}，属于浮点误差；这与三只示例中最大误差是不同统计量。有效月份不足的具体原因应查上市日期和缺失审计，不能仅凭月数差异推断停牌。

## 贝塔分布

历史贝塔均值 {beta.beta.mean():.3f}，中位数 {beta.beta.median():.3f}，范围 [{beta.beta.min():.3f}, {beta.beta.max():.3f}]。这些统计量描述本文的等权样本，其分布不代表全部 A 股，也不构成模型有效性的证据。
"""
    page5 = rf"""# 四、五个组合与高低组收益差

## 五组合检验期结果

{table(short)}

平均可得成员数依次为 {', '.join(f'{v:.1f}' for v in port.avg_members)}；完整表还报告 alpha 标准误、每月成员数和时间序列 R²。这里的 α 是组合超额收益对市场超额收益回归的截距，与横截面 SML 截距的含义不同。五组 α 均为正；G4 的 p 值为 {float(port.loc[port.group.eq('G4'), 'p_alpha'].iloc[0]):.3f}，其余各组均高于 0.05。G4 的单项检验结果应单独报告，但本文未进行多重比较调整或五组 α 的联合检验。

## 最高减最低历史贝塔组（H-L）

{table(ci_table)}

算术年化仅为 12 倍月均差，不是某个可交易多空策略的实际年复合收益。H-L 的 t 区间依赖近似独立同分布等条件；HAC 带宽为 6；移动分块自助法块长 3、重抽 5000 次，依赖时间序列近似平稳和块长选择，因此也依赖相应的序列假设。

## 收益差检验与 CAPM 定价检验

平均收益差为零与 $\alpha$ 为零是两个不同假设。令 $d_t$ 为最高减最低组的月度收益差，估计：

$$
d_t=\alpha_{{H-L}}+\Delta\beta\,R^e_{{m,t}}+u_t.
$$

得到实际贝塔差 $\Delta\beta={hl.beta_diff:.3f}$。同期市场平均超额收益 {pct(market)}/月，对应样本 CAPM 基准收益差 {pct(hl.capm_predicted_diff)}/月。

高低组 alpha 为 {pct(hl.alpha_diff)}/月，HAC 标准误 {pct(hl.se_alpha_diff)}，p={hl.p_alpha_diff:.3f}；95% 区间 [{pct(hl.ci95_alpha_low)}, {pct(hl.ci95_alpha_high)}]。在当前设定下{pricing_text}。这一结果仅涉及高低组价差的定价限制，不能代替五个组合 alpha 的联合检验。

当贝塔差与市场平均溢价均非零时，样本 CAPM 基准给出的收益差也非零。因此，收益差均值检验与定价误差检验须分别解释；两种点估计的倍数关系本身不构成拒绝模型的依据。
"""
    page6 = f"""# 五、个股与组合的证券市场线

![个股和组合 SML 对比](../output/figures/fig3_sml_compare.png)

图示 2020-2024 年检验期结果，横轴为检验期贝塔，纵轴为平均月度超额收益。两图统一尺度；CAPM 基准线截距为零，斜率为市场平均超额收益 {market:.5f}。红色实线为横截面经验拟合线，虚线为 CAPM 基准线；左图为个股，右图为组合。图中收益以百分数显示，下表截距、斜率及拟合残差标准差按小数列示。

{table(fit_table)}

{table(dev_table)}

MAE、RMSE 按每个点相对图中统一基准线的纵向距离计算；个股完整 60 月子样本用于检查观察期不一致。其余不足 60 月的个股均值来自各自可得月份，和整段市场均值不完全匹配，应结合完整月子样本阅读。由于观察点数量和聚合程度不同，组合与个股 MAE 的比较仅作描述，不作为正式的模型比较检验。

拟合残差描述散点到自由估计经验线的距离，MAE/RMSE 描述散点到 CAPM 指定基准线的距离，二者回答不同问题。判断对基准线的偏离须同时考虑斜率、截距及散点位置；横截面负截距也不能替代某个组合的时间序列 alpha。

五个组合点只有三个横截面残差自由度。横截面 HC1 标准误不能解决共同冲击、贝塔估计误差和少量观察点问题，所以本文不把其显著性作为 CAPM 支持程度的最终判断。
"""
    page7 = f"""# 六、必答问题 1-4

## 问题 1：收益是否随检验期贝塔上升？

五组收益按历史贝塔排序{hist_text}，按检验期贝塔排序也{test_text}。个股经验 SML 斜率为 {pi.slope:.5f}，组合为 {pp.slope:.5f}，描述其总体方向，但不能把正斜率等同于每一对股票/组合都严格单调。

最高与最低历史贝塔组的月均差为 {pct(hl.mean_diff_monthly)}。HAC 区间{interval_text}，表明收益差的估计仍有较大不确定性。其与定价误差检验的区别见第四节。

## 问题 2：历史分组的贝塔排序是否保持？

组合检验期贝塔依次为 {', '.join(f'{x:.3f}' for x in port.beta_test)}，{beta_text}。个股历史/检验期贝塔 Spearman 相关系数 {stab.spearman_rho:.3f}，检验期贝塔对历史贝塔回归斜率 {stab.slope_test_on_hist:.3f}。

最高与最低组历史均值差为 {port.avg_hist_beta.iloc[-1]-port.avg_hist_beta.iloc[0]:.3f}，检验期实际差为 {hl.beta_diff:.3f}。解释收益差应使用后者作为实际市场风险差异。估计噪声、风险暴露变化等均可能影响排序；本研究没有识别各机制的贡献，不能把全部变化归因于某次市场事件或向均值回归。

![五组合历史和检验期贝塔](../output/figures/fig4_group_beta.png)

## 问题 3：分散化是否让关系更清晰？

组合 SML 拟合残差标准差为 {pct(pp.resid_sd)}，个股为 {pct(pi.resid_sd)}；组合贝塔标准误范围 [{port.se_beta.min():.3f}, {port.se_beta.max():.3f}]，个股中位数 {stock.se_beta.median():.3f}。个股与组合经验线的截距分别为 {pi.intercept:.5f}、{pp.intercept:.5f}，斜率分别为 {pi.slope:.5f}、{pp.slope:.5f}。较小的组合残差和贝塔标准误与分散部分公司特有波动的作用相容。不过，个股与组合 R² 的差异还受聚合和观察点数量影响，本文未单独识别这些因素各自的贡献。

## 问题 4：更接近基准线是否代表更强支持？

组合相对基准线的 MAE 为 {pct(dp.mae)}，小于个股的 {pct(di.mae)}。这一差异可能来自组内正负定价误差抵消，或平均后抽样噪声减小，不能单凭距离缩小判定 CAPM 获得更强支持。五个组合点也限制了对横截面关系的判断。
"""
    page8 = rf"""# 七、必答问题 5-6 与敏感性分析

## 问题 5：高贝塔没有更高收益能否否定 CAPM？

本样本 H-L 点估计为 {pct(hl.mean_diff_monthly)}/月。无论其符号如何，都必须同时考虑检验期实际贝塔差、市场平均超额收益的符号和均值估计误差。如果市场样本溢价为负，高贝塔组的已实现收益更低也可与样本 CAPM 基准方向一致；样本平均收益不是无噪声的预期收益。

本期市场样本溢价为 {pct(market)}/月，H-L 实际贝塔差为 {hl.beta_diff:.3f}。回归结果{pricing_text}，但未拒绝零 alpha 不等于证明其为零，也不能证明其他组合或个股全部满足模型。

## 问题 6：综合支持程度与两项主要局限

历史贝塔分组在检验期{beta_text}，个股前后贝塔的相关系数为 {stab.spearman_rho:.3f}；按检验期贝塔排列，五组样本收益{test_text}。这些结果说明，历史排序、样本收益表现和模型定价限制是不同层面的判断。现有证据未确立稳定的高贝塔收益优势，H-L 的零 alpha 假设也未被拒绝；结合第五节的点估计及其不确定性，尚不能据此验证 CAPM。

第一项关键局限是未知缺月、未核实的退出清偿和可得成员平均；它们会影响组合构成及收益测量。第二项是仅 60 个检验月、贝塔估计误差及市场代理不完整；增加股票数量不能消除共同市场冲击或替代更长时间序列。

一项优先可行的改进是逐股核实 exit_audit.csv 中的退出现金流，并用公告或独立行情确认未知月份，再在固定样本/分组下重算。扩展贝塔估计窗口时应只使用分组日前的数据。固定线性 Blume 调整 $\beta_i^{{\mathrm{{adj}}}}=0.67\widehat{{\beta}}_i+0.33$ 不改变排序，因此单独调整后再分组不会改善本作业的组别区分。

## 已执行的固定样本/分组敏感性对照

{table(robust_short)}

三种价格处理情景均依赖额外估值假设，不代表未知月份已被确认为真实停牌。全损情景将未核实的检验期退出月收益替换为 -100%；合并换股等可能有非零回收价值，故它不是实际业绩、不是统计置信界，也未涵盖所有可能路径。这些对照用于展示结果对终止收益处理假设的敏感程度。利率简单折算仍使用相同报价日期和期限，并未消除无风险代理的期限错配。
"""
    page9 = """# 八、必答问题 7、团队分工与复核

## AI 贡献及本次矫正

AI 协助检查代码、实现分析流水线、运行自动核对，并起草和编辑报告。项目修改已获用户授权；成员身份、真实分工及对 AI 结果的人工判断仍须由组员据实填写。现有记录尚不能确认完整对话归档和逐项人工复算已完成。

AI 参与的工作包括 A 股样本过滤、历史行业匹配、缺失和退出数据审计、收益差与 alpha 检验，以及图表和正文生成。报告另行区分了经验线拟合残差与 CAPM 基准线偏差，并说明固定线性 Blume 调整不改变排序。具体修改过程保留在 AI 使用记录中。

代码与数字的自动核对是 AI 执行的程序验证，不等同于组员人工复核。完整回归结果、固定分组、每月成员数、缺失/退出审计和情景分析均随项目提供。本次提示和修改摘要见 ai_log/2026-10-28_审查与修订.md；它不是完整原始对话导出。组员应保存真实对话记录并记录采纳/修改/否决的依据。

## 成员分工（待真实填写）

| 姓名/学号 | 负责内容 | 实际完成及复核证据 |
|---|---|---|
| 待填写 | 数据来源与退出事件核查 | 待填写 |
| 待填写 | 贝塔及组合独立复算 | 待填写 |
| 待填写 | 图表与统计解释复核 | 待填写 |
| 待填写 | 正文与提交材料整理 | 待填写 |
| 待填写 | 全流程交叉复核 | 待填写 |

## 提交前人工确认

| 内容 | 当前状态 |
|---|---|
| 姓名、学号、真实分工 | 待组员填写 |
| 退出公告、清偿/换股与停牌证据 | 待核实；未知项已列入审计 |
| 独立复算及组员签认 | 待人工执行，不能用 AI 自动检查替代 |
| 全部 AI 原始指令过程 | 本仓库有摘要；完整原始记录待导出归档 |
| 口头解释全部流程与结论 | 待组内确认 |

## 计算文件与附录索引

估计期逐股结果：output/tables/beta_est.csv。检验期逐股结果及不足月排除表：stock_test.csv、stock_test_exclusions.csv。组合分组和成员数：portfolio_membership.csv、portfolio_member_counts.csv。高低组结果：hl_diff.csv、hl_diff_monthly.csv。敏感性：robustness.csv。

数据审计：data/processed/missing_audit.csv、exit_audit.csv。历史来源快照：data/snapshots/。关键数字与源文件哈希：output/tables/numbers_lock.json。一键复现命令与验证入口见 README.md。
"""
    pages = [page1,page2,page3,page4,page5,page6,page7,page8,page9]
    path = ROOT / "report" / "报告.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n\n<!-- PAGEBREAK -->\n\n".join(pages), encoding="utf-8")
    return path
