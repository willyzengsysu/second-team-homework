"""S6 作图：个股与组合的证券市场线（SML）。

图
  fig1_sml_individual.png  个股散点 + 经验 SML + CAPM 基准线
  fig2_sml_portfolio.png   组合散点（5 点）+ 经验 SML + CAPM 基准线
  fig3_sml_compare.png     两图并排，坐标尺度一致
  fig4_group_beta.png      五组合历史贝塔 vs 检验期贝塔
  fig5_group_return.png    五组合平均月度超额收益与 H−L 差

CAPM 基准线：截距为 0，斜率为检验期市场平均月度超额收益。
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402

from src.common import (  # noqa: E402
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

C_GRID = dict(alpha=0.35, linestyle="--", linewidth=0.8, color="grey")


def setup_style(cfg: dict) -> None:
    plt.rcParams["font.sans-serif"] = cfg["output"]["chinese_font"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = cfg["output"]["figure_dpi"]
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["font.size"] = 10


def cross_section_fit(x: pd.Series, y: pd.Series) -> dict:
    df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    X = sm.add_constant(df["x"])
    fit = sm.OLS(df["y"], X).fit(cov_type="HC1")
    return {
        "n": len(df),
        "intercept": float(fit.params["const"]),
        "slope": float(fit.params["x"]),
        "se_intercept": float(fit.bse["const"]),
        "se_slope": float(fit.bse["x"]),
        "r2": float(fit.rsquared),
        "resid_sd": float(np.std(fit.resid, ddof=1)),
    }


def plot_sml(ax, x, y, fit, mkt_slope, title, xlim, ylim, point_labels=None, annotate=False):
    ax.scatter(x, y, s=26 if len(x) > 10 else 70, alpha=0.7,
               edgecolor="white", linewidth=0.5, zorder=3, label="样本点")

    xs = np.linspace(xlim[0], xlim[1], 50)
    ax.plot(xs, fit["intercept"] + fit["slope"] * xs, color="#c0392b", linewidth=1.6,
            label=f"经验 SML(斜率={fit['slope']:.3f})", zorder=4)
    ax.plot(xs, mkt_slope * xs, color="#2c3e50", linewidth=1.5, linestyle="--",
            label=f"CAPM 基准线(斜率={mkt_slope:.3f})", zorder=4)

    if annotate and point_labels is not None:
        for i, (xi, yi, lab) in enumerate(zip(x, y, point_labels)):
            dx, dy = (8, 8) if i % 2 == 0 else (8, -16)
            ax.annotate(lab, (xi, yi), textcoords="offset points", xytext=(dx, dy),
                        fontsize=8, color="#2c3e50")

    ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("检验期贝塔")
    ax.set_ylabel("平均月度超额收益")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc="upper left")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="S6 作图")
    args = ap.parse_args(argv)

    cfg = load_config()
    ensure_dirs(cfg)
    log = get_logger(cfg)
    paths = project_paths(cfg)
    setup_style(cfg)

    stock = load_csv(paths["tables"] / "stock_test.csv")
    port = load_csv(paths["tables"] / "portfolio_stats.csv")
    panel = load_csv(paths["processed"] / "panel_monthly.csv")

    test = set(str(p) for p in test_months(cfg))
    mkt_ex = panel[panel["period"].isin(test)].groupby("period")["mret_ex"].first().dropna()
    mkt_slope = float(mkt_ex.mean())
    log.info("检验期市场平均月度超额收益 %.4f（年化 %.2f%%）", mkt_slope, ((1 + mkt_slope) ** 12 - 1) * 100)

    fit_i = cross_section_fit(stock["beta_test"], stock["mean_excess_ret"])
    fit_p = cross_section_fit(port["beta_test"], port["mean_excess_ret"])
    save_csv(pd.DataFrame([{"scope": "个股", **fit_i}, {"scope": "组合", **fit_p}]),
             paths["tables"] / "sml_fit.csv")
    log.info("个股 SML：截距 %.4f，斜率 %.4f，R² %.3f", fit_i["intercept"], fit_i["slope"], fit_i["r2"])
    log.info("组合 SML：截距 %.4f，斜率 %.4f，R² %.3f", fit_p["intercept"], fit_p["slope"], fit_p["r2"])

    # 统一坐标尺度（两图可比）
    x_all = pd.concat([stock["beta_test"], port["beta_test"]])
    y_all = pd.concat([stock["mean_excess_ret"], port["mean_excess_ret"]])
    pad_x = (x_all.max() - x_all.min()) * 0.06
    pad_y = (y_all.max() - y_all.min()) * 0.08
    xlim = (max(0, x_all.min() - pad_x), x_all.max() + pad_x)
    ylim = (min(y_all.min(), 0) - pad_y, max(y_all.max(), 0) + pad_y)

    fig, ax = plt.subplots(figsize=(8, 5.6))
    plot_sml(ax, stock["beta_test"], stock["mean_excess_ret"], fit_i, mkt_slope,
             f"个股 SML（n={len(stock)}）", xlim, ylim)
    fig.tight_layout()
    fig.savefig(paths["figures"] / "fig1_sml_individual.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.6))
    plot_sml(ax, port["beta_test"], port["mean_excess_ret"], fit_p, mkt_slope,
             "组合 SML（n=5）", xlim, ylim,
             point_labels=port["group"].tolist(), annotate=True)
    fig.tight_layout()
    fig.savefig(paths["figures"] / "fig2_sml_portfolio.png", bbox_inches="tight")
    plt.close(fig)

    # 并排对比
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharex=True, sharey=True)
    plot_sml(axes[0], stock["beta_test"], stock["mean_excess_ret"], fit_i, mkt_slope,
             f"(a) 个股（n={len(stock)}）", xlim, ylim)
    plot_sml(axes[1], port["beta_test"], port["mean_excess_ret"], fit_p, mkt_slope,
             "(b) 组合（n=5）", xlim, ylim, point_labels=port["group"].tolist(), annotate=True)
    fig.tight_layout()
    fig.savefig(paths["figures"] / "fig3_sml_compare.png", bbox_inches="tight")
    plt.close(fig)

    # 组间贝塔对比
    fig, ax = plt.subplots(figsize=(8, 4.8))
    idx = np.arange(len(port))
    w = 0.38
    ax.bar(idx - w / 2, port["avg_hist_beta"], w, label="历史贝塔（2015–2019）", color="#7f8c8d")
    ax.bar(idx + w / 2, port["beta_test"], w, label="检验期贝塔（2020–2024）", color="#2980b9")
    ax.set_xticks(idx)
    ax.set_xticklabels(port["group"], fontsize=9)
    ax.set_ylabel("贝塔")
    ax.set_title("五个组合的贝塔：分组依据 vs 检验期实际暴露", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(paths["figures"] / "fig4_group_beta.png", bbox_inches="tight")
    plt.close(fig)

    # 组间收益
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = ["#27ae60" if v >= 0 else "#c0392b" for v in port["mean_excess_ret"]]
    ax.bar(idx, port["mean_excess_ret"], color=colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(idx)
    ax.set_xticklabels(port["group"], fontsize=9)
    ax.set_ylabel("平均月度超额收益")
    ax.set_title("五组合平均月度超额收益（检验期）", fontsize=11)
    fig.tight_layout()
    fig.savefig(paths["figures"] / "fig5_group_return.png", bbox_inches="tight")
    plt.close(fig)

    save_json(run_manifest(cfg, "s6_figures", {
        "market_excess_mean": mkt_slope,
        "sml_individual": fit_i,
        "sml_portfolio": fit_p,
    }), paths["figures"] / "s6_manifest.json")
    log.info("S6 完成：5 张图已输出到 %s", paths["figures"])


if __name__ == "__main__":
    main()
