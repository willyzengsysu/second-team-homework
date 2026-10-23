"""S5 五组合构建与检验期分析。

规则
  - 按历史贝塔数值（非绝对值）从低到高排序，等分为 5 组；
  - 等权、分组固定、每月恢复等权 → 组合月收益 = 组内可得股票的算术平均；
  - 检验期不再按贝塔调整分组。

产出
  output/tables/portfolio_returns.csv  五组合月度收益与成员数
  output/tables/portfolio_stats.csv    五组合检验期统计（贝塔/阿尔法/标准误/平均超额收益）
  output/tables/hl_diff.csv            最高−最低组收益差及置信区间
  output/tables/beta_stability.csv     历史贝塔与检验期贝塔的稳定性
  output/tables/stock_test.csv         个股检验期统计（用于个股 SML）
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from src.common import (
    ensure_dirs,
    get_logger,
    load_config,
    load_csv,
    project_paths,
    run_manifest,
    save_csv,
    save_json,
    test_months,
)


def ols_alpha_beta(y: pd.Series, x: pd.Series, hac_lags: int | None) -> dict | None:
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(df) < 12:
        return None
    X = sm.add_constant(df["x"], has_constant="add")
    fit = (sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
           if hac_lags else sm.OLS(df["y"], X).fit())
    return {
        "n_obs": len(df),
        "alpha": float(fit.params["const"]),
        "beta": float(fit.params["x"]),
        "se_alpha": float(fit.bse["const"]),
        "se_beta": float(fit.bse["x"]),
        "t_alpha": float(fit.tvalues["const"]),
        "p_alpha": float(fit.pvalues["const"]),
        "r2": float(fit.rsquared),
        "resid_sd": float(np.std(fit.resid, ddof=1)),
    }


def ci_t(d: pd.Series) -> tuple[float, float]:
    """常规 t 区间：假设月度差近似独立同分布。"""
    n = d.notna().sum()
    se = d.std(ddof=1) / np.sqrt(n)
    crit = stats.t.ppf(0.975, n - 1)
    return float(d.mean() - crit * se), float(d.mean() + crit * se)


def ci_hac(d: pd.Series, lags: int) -> tuple[float, float]:
    """Newey-West 区间：允许月度差存在自相关与异方差。"""
    y = d.dropna()
    X = np.ones((len(y), 1))
    fit = sm.OLS(y.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    se = float(fit.bse[0])
    return float(y.mean() - 1.96 * se), float(y.mean() + 1.96 * se)


def ci_bootstrap(d: pd.Series, seed: int, n_boot: int = 5000, block: int = 3) -> tuple[float, float]:
    """分块自助法：保留月度差的短期序列相关。"""
    y = d.dropna().values
    n = len(y)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    starts = np.arange(0, n - block + 1)
    for b in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, n_blocks)])
        means[b] = y[idx[:n]].mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="S5 五组合分析")
    args = ap.parse_args(argv)

    cfg = load_config()
    ensure_dirs(cfg)
    log = get_logger(cfg)
    paths = project_paths(cfg)

    panel = load_csv(paths["processed"] / "panel_monthly.csv")
    beta = load_csv(paths["tables"] / "beta_est.csv")
    beta["code"] = beta["code"].astype(str).str.zfill(6)
    pf = cfg["portfolio"]
    hac = cfg["regression"]["hac_maxlags"] if cfg["regression"]["cov_type"] == "HAC" else None

    # ---- 1. 按历史贝塔排序分组（等数量）----
    beta = beta.sort_values("beta").reset_index(drop=True)
    labels = pf["group_labels"]
    beta["group"] = pd.cut(beta.index, bins=pf["n_groups"], labels=labels, include_lowest=True)
    log.info("分组完成：%s", beta.groupby("group", observed=False).size().to_dict())

    test = set(str(p) for p in test_months(cfg))
    tp = panel[panel["period"].isin(test)].copy()
    tp = tp.merge(beta[["code", "beta", "group", "industry", "name"]], on="code", how="inner",
                  suffixes=("", "_est"))
    mret = tp.groupby("period")["mret"].first()
    rf = tp.groupby("period")["rf"].first()

    # ---- 2. 组合月度收益（可得成员等权平均）----
    port = (
        tp.groupby(["period", "group"], observed=False)["ret"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "ret", "count": "n_members"})
    )
    wide = port.pivot(index="period", columns="group", values="ret")
    wide = wide.reindex(columns=labels)
    wide["mret"] = mret
    wide["rf"] = rf
    for g in labels:
        wide[f"{g}_ex"] = wide[g] - wide["rf"]
    wide = wide.reset_index().rename(columns={"index": "period"})
    save_csv(wide, paths["tables"] / "portfolio_returns.csv")
    log.info("组合月度收益：%s ~ %s，共 %d 个月",
             wide["period"].min(), wide["period"].max(), len(wide))

    # ---- 3. 组合检验期回归 ----
    rows = []
    for g in labels:
        y = wide[f"{g}_ex"]
        r = ols_alpha_beta(y, wide["mret"] - wide["rf"], hac)
        if r is None:
            continue
        rows.append({
            "group": g,
            "n_months": r["n_obs"],
            "avg_hist_beta": float(beta.loc[beta["group"] == g, "beta"].mean()),
            "beta_test": r["beta"],
            "se_beta": r["se_beta"],
            "alpha_monthly": r["alpha"],
            "se_alpha": r["se_alpha"],
            "t_alpha": r["t_alpha"],
            "p_alpha": r["p_alpha"],
            "r2": r["r2"],
            "mean_excess_ret": float(y.mean()),
            "std_excess_ret": float(y.std(ddof=1)),
            "avg_members": float(port.loc[port["group"] == g, "n_members"].mean()),
        })
    stats_df = pd.DataFrame(rows)
    save_csv(stats_df, paths["tables"] / "portfolio_stats.csv")
    log.info("五组合检验期贝塔：%s", stats_df["beta_test"].round(3).tolist())

    # ---- 4. 最高−最低组收益差 ----
    d = wide[f"{labels[-1]}_ex"] - wide[f"{labels[0]}_ex"]
    lo_t, hi_t = ci_t(d)
    lo_h, hi_h = ci_hac(d, hac or 6)
    lo_b, hi_b = ci_bootstrap(d, int(cfg["project"]["seed"]))
    verdict = ("区间不含 0" if lo_t > 0 or hi_t < 0 else "区间包含 0")

    hl = pd.DataFrame([{
        "high_group": labels[-1], "low_group": labels[0],
        "n_months": int(d.notna().sum()),
        "mean_diff_monthly": float(d.mean()),
        "mean_diff_annual": float((1 + d.mean()) ** 12 - 1),
        "std_diff": float(d.std(ddof=1)),
        "t_stat": float(d.mean() / (d.std(ddof=1) / np.sqrt(d.notna().sum()))),
        "ci95_t_low": lo_t, "ci95_t_high": hi_t,
        "ci95_hac_low": lo_h, "ci95_hac_high": hi_h,
        "ci95_boot_low": lo_b, "ci95_boot_high": hi_b,
        "conclusion": verdict,
    }])
    save_csv(hl, paths["tables"] / "hl_diff.csv")
    save_csv(d.rename("diff").reset_index().rename(columns={"index": "period"}),
             paths["tables"] / "hl_diff_monthly.csv")
    log.info("H−L 月度差 %.4f（年化 %.2f%%），t=%.2f，95%%CI [%.4f, %.4f]，%s",
             d.mean(), (1 + d.mean()) ** 12 - 1, hl["t_stat"].iloc[0], lo_t, hi_t, verdict)

    # ---- 5. 个股检验期统计（用于个股 SML）----
    rows = []
    for code, g in tp.groupby("code"):
        g = g.set_index("period").sort_index()
        r = ols_alpha_beta(g["ret_ex"], g["mret_ex"], hac)
        if r is None:
            continue
        rows.append({
            "code": code, "name": g["name"].iloc[0], "industry": g["industry"].iloc[0],
            "beta_hist": float(g["beta"].iloc[0]),
            "beta_test": r["beta"], "se_beta": r["se_beta"],
            "alpha": r["alpha"], "se_alpha": r["se_alpha"], "p_alpha": r["p_alpha"],
            "mean_excess_ret": float(g["ret_ex"].mean()),
            "n_obs": r["n_obs"],
            "group": str(g["group"].iloc[0]),
        })
    stock = pd.DataFrame(rows).sort_values("beta_test").reset_index(drop=True)
    save_csv(stock, paths["tables"] / "stock_test.csv")
    log.info("个股检验期：%d 只，贝塔均值 %.3f", len(stock), stock["beta_test"].mean())

    # ---- 6. 贝塔稳定性 ----
    rho = stats.spearmanr(stock["beta_hist"], stock["beta_test"])
    X = sm.add_constant(stock["beta_hist"])
    fit = sm.OLS(stock["beta_test"], X).fit()
    group_beta = stats_df[["group", "avg_hist_beta", "beta_test"]].copy()
    order_ok = group_beta["beta_test"].is_monotonic_increasing

    stab = pd.DataFrame([{
        "n_stocks": len(stock),
        "spearman_rho": float(rho.statistic),
        "spearman_p": float(rho.pvalue),
        "slope_test_on_hist": float(fit.params["beta_hist"]),
        "intercept": float(fit.params["const"]),
        "r2": float(fit.rsquared),
        "group_beta_monotonic": bool(order_ok),
        "corr_hist_test": float(stock["beta_hist"].corr(stock["beta_test"])),
        "corr_test_groupavg": float(
            pd.Series(group_beta["beta_test"].values).corr(
                pd.Series(group_beta["avg_hist_beta"].values), method="spearman")),
    }])
    save_csv(stab, paths["tables"] / "beta_stability.csv")
    save_csv(group_beta, paths["tables"] / "group_beta_compare.csv")
    log.info("贝塔稳定性：Spearman ρ=%.3f，检验期贝塔对历史贝塔斜率=%.3f，组间单调=%s",
             rho.statistic, fit.params["beta_hist"], order_ok)

    save_json(run_manifest(cfg, "s5_portfolio", {
        "groups": labels,
        "hl_mean_monthly": float(d.mean()),
        "hl_ci_t": [lo_t, hi_t],
        "spearman": float(rho.statistic),
        "group_beta_monotonic": bool(order_ok),
    }), paths["tables"] / "s5_manifest.json")
    log.info("S5 完成")


if __name__ == "__main__":
    main()
