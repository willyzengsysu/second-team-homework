"""S4 估计期贝塔：含截距 OLS + 定义法核对。

任务书要求
  - 用估计期数据对每只股票跑 r_i − rf = α + β(Rm − rf) + ε，取斜率为历史贝塔；
  - 任选低/中/高三只股票，用 cov/var 定义法核对；
  - 解释两种方法为何应一致（恒等式，不是近似）。

产出
  output/tables/beta_est.csv        全样本估计期贝塔
  output/tables/beta_check.csv      三只股票的两种算法核对表
  output/tables/beta_identity.csv   全样本一致性验证
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.common import (
    ensure_dirs,
    est_months,
    get_logger,
    load_config,
    load_csv,
    project_paths,
    run_manifest,
    save_csv,
    save_json,
)


def ols_beta(y: pd.Series, x: pd.Series, hac_lags: int | None = None) -> dict | None:
    """含截距 OLS：返回斜率（贝塔）、截距（阿尔法）及其标准误。"""
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    n = len(df)
    if n < 12 or df["x"].std(ddof=1) == 0:
        return None

    X = sm.add_constant(df["x"], has_constant="add")
    model = sm.OLS(df["y"], X)
    fit = model.fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags}) if hac_lags else model.fit()

    return {
        "n_obs": n,
        "alpha": float(fit.params["const"]),
        "beta": float(fit.params["x"]),
        "se_alpha": float(fit.bse["const"]),
        "se_beta": float(fit.bse["x"]),
        "t_alpha": float(fit.tvalues["const"]),
        "r2": float(fit.rsquared),
        "resid_sd": float(np.std(fit.resid, ddof=1)),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
    }


def beta_by_definition(y: pd.Series, x: pd.Series) -> dict | None:
    """定义法：样本协方差 / 样本方差（同样本、同为 n−1 分母）。"""
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(df) < 12:
        return None
    cov = np.cov(df["y"], df["x"], ddof=1)[0, 1]
    var = np.var(df["x"], ddof=1)
    if var == 0:
        return None
    return {"beta_def": float(cov / var), "cov": float(cov), "var": float(var), "n_obs": len(df)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="S4 估计期贝塔")
    args = ap.parse_args(argv)

    cfg = load_config()
    ensure_dirs(cfg)
    log = get_logger(cfg)
    paths = project_paths(cfg)

    panel = load_csv(paths["processed"] / "panel_monthly.csv")
    sample = load_csv(paths["processed"] / "sample.csv")
    sample["code"] = sample["code"].astype(str).str.zfill(6)

    est = set(str(p) for p in est_months(cfg))
    df = panel[panel["period"].isin(est)].copy()
    log.info("估计期 %s ~ %s：%d 行",
             min(est), max(est), len(df))

    hac = cfg["regression"]["hac_maxlags"] if cfg["regression"]["cov_type"] == "HAC" else None

    records, ident = [], []
    for code, g in df.groupby("code"):
        g = g.set_index("period").sort_index()
        res = ols_beta(g["ret_ex"], g["mret_ex"], hac_lags=hac)
        if res is None:
            continue
        res["code"] = code
        records.append(res)

        # 一致性验证：OLS 斜率 vs 定义法，逐只比对
        d = beta_by_definition(g["ret_ex"], g["mret_ex"])
        if d:
            ident.append({"code": code, "beta_ols": res["beta"], "beta_def": d["beta_def"],
                          "abs_diff": abs(res["beta"] - d["beta_def"]), "n_obs": d["n_obs"]})

    beta = pd.DataFrame(records)
    beta = beta.merge(sample[["code", "name", "industry", "sector", "status"]], on="code", how="left")
    beta = beta[["code", "name", "industry", "sector", "status", "n_obs", "alpha", "beta",
                 "se_alpha", "se_beta", "t_alpha", "r2", "resid_sd", "start", "end"]]
    beta = beta.sort_values("beta").reset_index(drop=True)
    save_csv(beta, paths["tables"] / "beta_est.csv")
    log.info("估计期贝塔：%d 只，均值 %.3f，中位数 %.3f，区间 [%.2f, %.2f]",
             len(beta), beta["beta"].mean(), beta["beta"].median(),
             beta["beta"].min(), beta["beta"].max())

    # 全样本一致性
    idf = pd.DataFrame(ident)
    save_csv(idf, paths["tables"] / "beta_identity.csv")
    log.info("定义法一致性：最大绝对差 %.2e（理论上应为 0，仅浮点误差）",
             idf["abs_diff"].max() if len(idf) else float("nan"))

    # 低/中/高三只核对表
    qs = beta["beta"].quantile([0.2, 0.5, 0.8]).values
    picked = []
    for label, q in zip(["低贝塔", "中贝塔", "高贝塔"], qs):
        idx = (beta["beta"] - q).abs().idxmin()
        picked.append({"level": label, **beta.loc[idx].to_dict()})

    rows = []
    for p in picked:
        g = df[df["code"] == p["code"]].set_index("period").sort_index()
        d = beta_by_definition(g["ret_ex"], g["mret_ex"])
        rows.append({
            "level": p["level"],
            "code": p["code"],
            "name": p["name"],
            "industry": p["industry"],
            "n_obs": d["n_obs"],
            "cov": d["cov"],
            "var": d["var"],
            "beta_def": d["beta_def"],
            "beta_ols": p["beta"],
            "abs_diff": abs(d["beta_def"] - p["beta"]),
            "alpha": p["alpha"],
            "se_beta": p["se_beta"],
            "r2": p["r2"],
        })
    check = pd.DataFrame(rows)
    save_csv(check, paths["tables"] / "beta_check.csv")
    log.info("三只股票核对完成：%s", ", ".join(f"{r['name']}={r['beta_ols']:.3f}" for _, r in check.iterrows()))

    save_json(run_manifest(cfg, "s4_beta", {
        "n_stocks": len(beta),
        "beta_mean": float(beta["beta"].mean()),
        "beta_median": float(beta["beta"].median()),
        "max_abs_diff_definition": float(idf["abs_diff"].max()) if len(idf) else None,
        "hac_lags": hac,
    }), paths["tables"] / "s4_manifest.json")
    log.info("S4 完成")


if __name__ == "__main__":
    main()
