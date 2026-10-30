"""固定样本、固定历史分组的缺失/终止收益敏感性对照；不替代已核实数据。"""
from __future__ import annotations

import pandas as pd

from src.common import load_config, load_csv, project_paths, save_csv, test_months
from src.s5_portfolio import ci_t, ci_hac, ols_alpha_beta


def scenario_returns(panel: pd.DataFrame, carry_tail=False, terminal_loss=False) -> pd.Series:
    values = panel.ret.copy()
    for _, g in panel.groupby("code"):
        g = g.sort_values("period")
        close = g.close.copy()
        observed = close.dropna()
        if observed.empty:
            continue
        last_period = g.loc[observed.index[-1], "period"]
        active = ~g.return_status.isin(["pre_listing", "post_delist"])
        if not carry_tail:
            active &= g.period.le(last_period)
        filled = close.ffill().where(active)
        ret = filled.pct_change(fill_method=None)
        # 有独立来源的终止收益始终优先于情景假设。
        verified = g.return_status.eq("verified_terminal_return")
        ret.loc[verified] = g.loc[verified, "ret"]
        gone = pd.to_datetime(g.delist_date.iloc[0], errors="coerce")
        if terminal_loss and pd.notna(gone):
            terminal = g.period.eq(str(gone.to_period("M"))) & ~verified
            ret.loc[terminal & filled.notna()] = -1.0
        values.loc[g.index] = ret
    return values


def main(argv=None):
    cfg = load_config()
    paths = project_paths(cfg)
    p = load_csv(paths["processed"] / "panel_monthly.csv")
    membership = load_csv(paths["tables"] / "portfolio_membership.csv")
    p = p.merge(membership[["code", "group"]], on="code", validate="many_to_one")
    dates = {str(m) for m in test_months(cfg)}
    labels, lags = cfg["portfolio"]["group_labels"], cfg["regression"]["hac_maxlags"]
    scenarios = [
        ("strict_available", p.ret, p.rf, "主口径；未知月及跨未知月收益保留缺失"),
        ("carry_internal_gaps", scenario_returns(p), p.rf, "假设内部缺月按末价估值；不代表停牌已核实"),
        ("carry_to_exit_no_loss", scenario_returns(p, True), p.rf, "假设尾部末价延续到退出，且无额外终止损失"),
        ("carry_to_exit_total_loss", scenario_returns(p, True, True), p.rf, "极端压力情景：未核实终止收益设为-100%；不是实际损失或统计置信界"),
    ]
    rf_raw = load_csv(paths["interim"] / "rf_monthly.csv").set_index("period").y_annual / 1200
    scenarios.append(("rf_simple", p.ret, p.period.map(rf_raw), "同样本/分组；无风险利率改为年化/12"))
    rows = []
    for name, ret, rf, assumption in scenarios:
        d = p.assign(ret=ret, rf=rf)
        d = d[d.period.isin(dates)]
        w = d.groupby(["period", "group"]).ret.mean().unstack().reindex(columns=labels)
        m = d.groupby("period").mret.first() - d.groupby("period").rf.first()
        diff = w[labels[-1]] - w[labels[0]]
        fit = ols_alpha_beta(diff, m, lags)
        low, high = ci_t(diff)
        hlo, hhi = ci_hac(diff, lags)
        rows.append({"scenario": name, "hl_mean": float(diff.mean()), "ci_t_low": low,
                     "ci_t_high": high, "ci_hac_low": hlo, "ci_hac_high": hhi,
                     "beta_diff": fit["beta"], "alpha_diff": fit["alpha"],
                     "p_alpha_diff": fit["p_alpha"], "n_months": fit["n_obs"],
                     "assumption": assumption})
    save_csv(pd.DataFrame(rows), paths["tables"] / "robustness.csv")


if __name__ == "__main__":
    main()
