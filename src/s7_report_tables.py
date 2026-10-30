"""S7 报告表格与数字锁定。

产出
  output/tables/report_tables.md  报告正文可直接引用的表格（Markdown）
  output/tables/numbers_lock.json 关键数字 + 来源文件哈希，保证正文与表格一致
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from src.common import (
    ensure_dirs,
    get_logger,
    load_config,
    load_csv,
    project_paths,
    run_manifest,
    save_json,
    sha256_file,
)


def fmt(x, nd: int = 4) -> str:
    try:
        if pd.isna(x):
            return "—"
    except Exception:  # noqa: BLE001
        pass
    return f"{x:.{nd}f}"


def pct(x, nd: int = 2) -> str:
    try:
        if pd.isna(x):
            return "—"
    except Exception:  # noqa: BLE001
        pass
    return f"{x * 100:.{nd}f}%"


def md_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.values]
    return "\n".join([header, sep] + rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="S7 报告表格")
    args = ap.parse_args(argv)

    cfg = load_config()
    ensure_dirs(cfg)
    log = get_logger(cfg)
    paths = project_paths(cfg)
    t, f = paths["tables"], paths["figures"]

    check = load_csv(t / "beta_check.csv")
    beta = load_csv(t / "beta_est.csv")
    port = load_csv(t / "portfolio_stats.csv")
    hl = load_csv(t / "hl_diff.csv").iloc[0]
    stab = load_csv(t / "beta_stability.csv").iloc[0]
    sml = load_csv(t / "sml_fit.csv")
    stock = load_csv(t / "stock_test.csv")
    sample = load_csv(paths["processed"] / "sample.csv")

    parts = ["# 报告用表（自动生成，勿手改）\n"]

    # 表 1 样本概览
    t1 = pd.DataFrame([
        ["股票池（2019-12-31 冻结，含此后退市）", f"{len(load_csv(paths['interim'] / 'universe.csv'))} 只"],
        ["最终样本", f"{len(sample)} 只"],
        ["覆盖行业（证监会大类）", f"{sample['industry'].nunique()} 个"],
        ["样本期", f"{cfg['sample']['est_start']} ~ {cfg['sample']['test_end']}"],
        ["估计期 / 检验期", f"2015–2019 / 2020–2024"],
        ["估计期最少有效月份", f"{cfg['sample']['min_est_months']} 个月"],
        ["市场基准", "中证全指全收益（H00985）"],
        ["无风险利率", "中债国债收益率曲线 1 年期，月度 = (1+y)^(1/12)−1"],
    ], columns=["项目", "说明"])
    parts.append("## 表 1 样本与数据口径\n\n" + md_table(t1) + "\n")

    # 表 2 三只股票核对
    t2 = check[["level", "code", "name", "industry", "n_obs", "cov", "var", "beta_def", "beta_ols", "abs_diff"]].copy()
    for c in ["cov", "var", "beta_def", "beta_ols"]:
        t2[c] = t2[c].map(lambda v: fmt(v, 6))
    t2["abs_diff"] = t2["abs_diff"].map(lambda v: f"{v:.1e}")
    t2.columns = ["分层", "代码", "简称", "行业", "月份数", "cov", "var", "定义法 β", "OLS β", "绝对差"]
    parts.append("## 表 2 三只股票的贝塔计算核对\n\n" + md_table(t2) + "\n")

    # 表 3 五组合对比
    t3 = port[["group", "avg_members", "avg_hist_beta", "beta_test", "se_beta",
               "mean_excess_ret", "alpha_monthly", "se_alpha", "p_alpha", "r2"]].copy()
    t3["avg_members"] = t3["avg_members"].map(lambda v: f"{v:.0f}")
    for c in ["avg_hist_beta", "beta_test", "se_beta"]:
        t3[c] = t3[c].map(fmt)
    for c in ["mean_excess_ret", "alpha_monthly", "se_alpha"]:
        t3[c] = t3[c].map(lambda v: fmt(v, 5))
    t3["p_alpha"] = t3["p_alpha"].map(lambda v: fmt(v, 3))
    t3["r2"] = t3["r2"].map(fmt)
    t3.columns = ["组合", "平均只数", "历史 β", "检验期 β", "β 标准误",
                  "平均月超额收益", "α(月度)", "α 标准误", "α 的 p 值", "R²"]
    parts.append("## 表 3 五个组合的检验期表现\n\n" + md_table(t3) + "\n")

    # 表 4 H−L
    t4 = pd.DataFrame([
        ["最高−最低组月度收益差均值", fmt(hl["mean_diff_monthly"], 5)],
        ["算术年化均值（12×月均差，非实际复合收益）", pct(hl["mean_diff_annual"])],
        ["t 统计量", fmt(hl["t_stat"], 2)],
        ["95% CI（t 分布）", f"[{fmt(hl['ci95_t_low'], 5)}, {fmt(hl['ci95_t_high'], 5)}]"],
        ["95% CI（Newey–West）", f"[{fmt(hl['ci95_hac_low'], 5)}, {fmt(hl['ci95_hac_high'], 5)}]"],
        ["95% CI（分块自助法）", f"[{fmt(hl['ci95_boot_low'], 5)}, {fmt(hl['ci95_boot_high'], 5)}]"],
        ["结论", hl["conclusion"]],
        ["高低组实际贝塔差", fmt(hl["beta_diff"])],
        ["CAPM 基准预测收益差（月度）", pct(hl["capm_predicted_diff"])],
        ["高低组 α（月度）", pct(hl["alpha_diff"])],
        ["α 的 HAC p 值", fmt(hl["p_alpha_diff"], 3)],
    ], columns=["指标", "数值"])
    parts.append("## 表 4 最高与最低历史贝塔组合的收益差\n\n" + md_table(t4) + "\n")

    # 表 5 贝塔稳定性
    t5 = pd.DataFrame([
        ["历史贝塔与检验期贝塔的 Spearman ρ", fmt(stab["spearman_rho"], 3)],
        ["检验期 β 对历史 β 的回归斜率", fmt(stab["slope_test_on_hist"], 3)],
        ["回归截距", fmt(stab["intercept"], 3)],
        ["回归 R²", fmt(stab["r2"], 3)],
        ["组均值贝塔是否严格递增", "是" if stab["group_beta_monotonic"] else "否"],
    ], columns=["指标", "数值"])
    parts.append("## 表 5 贝塔的稳定性\n\n" + md_table(t5) + "\n")

    # 表 6 SML 对比
    t6 = sml.copy()
    t6["intercept"] = t6["intercept"].map(lambda v: fmt(v, 5))
    t6["slope"] = t6["slope"].map(fmt)
    t6["se_slope"] = t6["se_slope"].map(lambda v: fmt(v, 5))
    t6["r2"] = t6["r2"].map(fmt)
    t6["resid_sd"] = t6["resid_sd"].map(lambda v: fmt(v, 5))
    t6 = t6[["scope", "n", "intercept", "slope", "se_slope", "r2", "resid_sd"]]
    t6.columns = ["口径", "观察点", "截距", "斜率", "斜率标准误", "R²", "残差标准差"]
    parts.append("## 表 6 个股与组合的经验 SML\n\n" + md_table(t6) + "\n")

    text = "\n".join(parts)
    (t / "report_tables.md").write_text(text, encoding="utf-8")
    log.info("报告表格已生成：%s", t / "report_tables.md")

    # 数字锁定
    lock = {
        "sample_n": int(len(sample)),
        "industries": int(sample["industry"].nunique()),
        "beta_est_mean": float(beta["beta"].mean()),
        "beta_est_median": float(beta["beta"].median()),
        "beta_check_max_diff": float(check["abs_diff"].max()),
        "beta_identity_max_diff": float(load_csv(t / "beta_identity.csv")["abs_diff"].max()),
        "group_beta_test": port["beta_test"].round(6).tolist(),
        "group_mean_excess": port["mean_excess_ret"].round(6).tolist(),
        "group_alpha": port["alpha_monthly"].round(6).tolist(),
        "hl_mean_monthly": float(hl["mean_diff_monthly"]),
        "hl_alpha_monthly": float(hl["alpha_diff"]),
        "hl_alpha_p": float(hl["p_alpha_diff"]),
        "hl_ci_t": [float(hl["ci95_t_low"]), float(hl["ci95_t_high"])],
        "hl_ci_boot": [float(hl["ci95_boot_low"]), float(hl["ci95_boot_high"])],
        "spearman": float(stab["spearman_rho"]),
        "beta_slope_test_on_hist": float(stab["slope_test_on_hist"]),
        "sml_individual": sml.loc[sml["scope"] == "个股"].iloc[0].drop("scope").to_dict(),
        "sml_portfolio": sml.loc[sml["scope"] == "组合"].iloc[0].drop("scope").to_dict(),
        "n_stocks_test": int(len(stock)),
        "sources": {
            name: sha256_file(t / name)
            for name in ["beta_est.csv", "portfolio_stats.csv", "hl_diff.csv",
                         "beta_stability.csv", "sml_fit.csv", "capm_deviations.csv",
                         "stock_test.csv", "stock_test_exclusions.csv", "portfolio_membership.csv",
                         "portfolio_member_counts.csv", "robustness.csv", "beta_identity.csv"]
        },
        "manifest": run_manifest(cfg, "s7_report"),
    }
    from src.report_content import write_report
    report_path = write_report(cfg)
    lock["report_sha256"] = sha256_file(report_path)
    from src.common import ROOT
    lock["inputs"] = {str(p.relative_to(ROOT)).replace("\\", "/"): sha256_file(p) for p in [
        paths["processed"] / "panel_monthly.csv", paths["processed"] / "sample.csv",
        paths["interim"] / "market_monthly.csv", paths["interim"] / "rf_monthly.csv",
        paths["interim"] / "price_cache_manifest.csv", paths["interim"] / "terminal_quote_audit.csv",
    ]}
    save_json(lock, t / "numbers_lock.json")
    log.info("数字锁定：%s", t / "numbers_lock.json")


if __name__ == "__main__":
    main()
