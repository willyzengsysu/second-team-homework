"""S3 清洗与面板构建：月度收益率、超额收益、样本最终确定。

处理规则（写进报告的“数据处理”一节）
  1. 收益率用后复权收盘价计算，含分红再投资；
  2. 整月停牌：价格未变，收益记 0 并单独打标（不是用 0 填补缺失）；
     如需严格口径，把 config 的 missing.full_month_suspension 改为 strict；
  3. 退市：退市当月仍计入，之后自然退出组合，不做事后剔除；
  4. 组合内某股票当月无有效收益时，仅用可得成员求均值（等价于恢复等权），不填 0。

产出
  data/processed/panel_monthly.csv   长面板（code × period）
  data/processed/returns_wide.csv    月度收益宽表
  data/processed/sample.csv          最终样本（200 只）
  data/processed/exclusion_log.csv   逐只排除原因
  docs/数据字典.md                    变量说明
  docs/样本筛选记录.md                筛选漏斗
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.common import (
    allocate,
    ensure_dirs,
    est_months,
    get_logger,
    load_config,
    load_csv,
    month_range,
    project_paths,
    run_manifest,
    save_csv,
    save_json,
    test_months,
)


# --------------------------------------------------------------------------
# 单只股票 -> 月度收益序列
# --------------------------------------------------------------------------
def load_candidates(cfg: dict) -> pd.DataFrame:
    """读候选池，并把行业字段的空值/空串统一记为「未分类」。"""
    paths = project_paths(cfg)
    cand = load_csv(paths["interim"] / "candidates.csv")
    cand["code"] = cand["code"].astype(str).str.zfill(6)
    label = cfg["sample"]["industry_unknown_label"]
    for col in ("industry", "sector"):
        if col in cand.columns:
            cand[col] = (
                cand[col].fillna("").astype(str).str.strip()
                .replace({"": label, "nan": label})
            )
    return cand


def stock_monthly_returns(price: pd.DataFrame, grid: pd.PeriodIndex,
                          zero_marked: bool) -> tuple[pd.Series, pd.Series]:
    """返回 (收益率, 停牌标记)；窗口外为 NaN，窗口内缺失月按停牌处理。"""
    if price is None or price.empty:
        return pd.Series(index=grid, dtype=float), pd.Series(index=grid, dtype=bool)

    s = price.copy()
    s["period"] = pd.PeriodIndex(pd.to_datetime(s["date"]), freq="M")
    s = s.drop_duplicates(subset="period", keep="last").set_index("period")["close"]
    s = s.reindex(grid)

    if s.notna().sum() == 0:
        return pd.Series(index=grid, dtype=float), pd.Series(index=grid, dtype=bool)

    first, last = s.first_valid_index(), s.last_valid_index()
    window = s.loc[first:last]
    suspended = window.isna()

    # 停牌月价格不变：先前向填充再算收益，才能让下一个月收益正确
    filled = window.ffill()
    ret = filled.pct_change()
    if zero_marked:
        ret[suspended] = 0.0
    else:
        ret[suspended] = np.nan
    ret[filled.isna()] = np.nan

    susp_out = suspended.reindex(grid).astype("boolean").fillna(False).astype(bool)
    return ret.reindex(grid), susp_out


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def build_panel(cfg: dict, log) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = project_paths(cfg)
    s = cfg["sample"]

    cand = load_candidates(cfg)
    market = load_csv(paths["interim"] / "market_monthly.csv").rename(columns={"index": "period"})
    rf = load_csv(paths["interim"] / "rf_monthly.csv").rename(columns={"index": "period"})

    grid = month_range("2014-12", s["test_end"])
    grid_str = pd.Index([str(p) for p in grid])
    # 市场与无风险序列统一按 "YYYY-MM" 字符串对齐，与 panel["period"] 保持一致
    mret = market.set_index("period")["mret"].reindex(grid_str)
    rf_s = rf.set_index("period")["rf"].reindex(grid_str)

    rows, exclusions = [], []
    zero_marked = cfg["missing"]["full_month_suspension"] == "zero_marked"

    for _, r in cand.iterrows():
        code = r["code"]
        cache = paths["raw"] / "prices" / f"{code}.csv"
        if not cache.exists():
            exclusions.append({"code": code, "reason": "无行情缓存（抓取失败）", "valid_est_months": 0})
            continue

        price = load_csv(cache)
        ret, susp = stock_monthly_returns(price, grid, zero_marked)
        if ret.notna().sum() == 0:
            exclusions.append({"code": code, "reason": "区间内无有效价格", "valid_est_months": 0})
            continue

        for period in grid:
            v = ret.get(period, np.nan)
            rows.append({
                "code": code,
                "period": str(period),
                "ret": v,
                "suspended": bool(susp.get(period, False)),
            })

    panel = pd.DataFrame(rows)
    panel = panel.merge(cand[["code", "name", "industry", "sector", "status"]], on="code", how="left")
    panel["mret"] = panel["period"].map(mret)
    panel["rf"] = panel["period"].map(rf_s)
    panel["ret_ex"] = panel["ret"] - panel["rf"]
    panel["mret_ex"] = panel["mret"] - panel["rf"]

    # 有效配对月份（估计期）
    est = set(str(p) for p in est_months(cfg))
    valid = (
        panel[panel["period"].isin(est) & panel["ret_ex"].notna() & panel["mret_ex"].notna()]
        .groupby("code").size().rename("valid_est_months")
    )
    panel = panel.merge(valid, on="code", how="left")
    panel["valid_est_months"] = panel["valid_est_months"].fillna(0).astype(int)

    n_susp = int(panel["suspended"].sum())
    log.info("面板：%d 行，%d 只股票，%d 个停牌月（占 %.2f%%）",
             len(panel), panel["code"].nunique(), n_susp,
             100 * n_susp / max(len(panel), 1))
    return panel, pd.DataFrame(exclusions)


def finalize_sample(panel: pd.DataFrame, cand: pd.DataFrame, cfg: dict, log) -> pd.DataFrame:
    """按「估计期≥48个有效月」筛选，再按行业配额补足到 target_n。"""
    s = cfg["sample"]
    stats = (
        panel.groupby("code")["valid_est_months"].max().rename("valid_est_months").reset_index()
    )
    info = cand[["code", "name", "industry", "sector", "status", "stratum_size"]].merge(
        stats, on="code", how="left"
    )
    info["valid_est_months"] = info["valid_est_months"].fillna(0).astype(int)
    info["eligible"] = info["valid_est_months"] >= s["min_est_months"]

    excl = info[~info["eligible"]].copy()
    excl["reason"] = "估计期有效月份 < " + str(s["min_est_months"])
    log.info("有效月规则：合格 %d 只，剔除 %d 只", int(info["eligible"].sum()), len(excl))

    pool = info[info["eligible"]].copy()
    rng = np.random.default_rng(int(cfg["project"]["seed"]) + 1)
    quotas = allocate(pool["industry"].value_counts(), s["target_n"]).sort_index()

    picks = []
    for ind, q in quotas.items():
        codes = sorted(pool.loc[pool["industry"] == ind, "code"].tolist())
        take = min(int(q), len(codes))
        if take == 0:
            continue
        idx = rng.choice(len(codes), size=take, replace=False)
        picks.extend(codes[i] for i in idx)

    sample = pool[pool["code"].isin(picks)].copy()
    sample["group_quota"] = sample["industry"].map(quotas)
    log.info("最终样本 %d 只（合格 %d 只，未抽中 %d 只），覆盖 %d 个行业",
             len(sample), len(pool), len(pool) - len(sample), sample["industry"].nunique())
    return sample, excl, len(pool)


def write_docs(cfg: dict, panel: pd.DataFrame, sample: pd.DataFrame, excl: pd.DataFrame,
               log, n_eligible: int) -> None:
    paths = project_paths(cfg)
    docs = paths["processed"].parent.parent / "docs"
    docs.mkdir(parents=True, exist_ok=True)

    n_ind = sample["industry"].nunique()
    top = sample["industry"].value_counts().head(8)

    funnel = f"""# 样本筛选记录

## 漏斗

| 步骤 | 数量 | 说明 |
|---|---|---|
| 冻结日（{cfg['sample']['universe_asof']}）在市 A 股 | {len(pd.read_csv(paths['interim'] / 'universe.csv', encoding='utf-8-sig', dtype={'code': str}))} | 含此后退市股票 |
| 行业分层抽样候选池 | {len(pd.read_csv(paths['interim'] / 'candidates.csv', encoding='utf-8-sig', dtype={'code': str}))} | 基础 260 + 追加 40 |
| 估计期有效月份 ≥ {cfg['sample']['min_est_months']} | {n_eligible} | 剔除 {len(excl)} 只不足 48 个月的股票 |
| 随机抽取入样 | {len(sample)} | 合格但未抽中 {n_eligible - len(sample)} 只 |
| **最终样本** | **{len(sample)}** | 覆盖 {n_ind} 个行业 |

## 剔除原因

{excl['reason'].value_counts().to_string() if len(excl) else '无'}

## 行业分布（前 8）

{top.to_string()}

## 停牌与退市处理

- 整月停牌：价格未变，收益记 0 并打 `suspended` 标记（口径：`{cfg['missing']['full_month_suspension']}`）；
- 退市：退市当月仍计入收益，此后自然退出组合，未做事后剔除；
- 缺失：不填 0；组合均值只用当月可得成员（等价于恢复等权）。
"""
    (docs / "样本筛选记录.md").write_text(funnel, encoding="utf-8")

    dictionary = """# 数据字典

## panel_monthly.csv（长面板）

| 字段 | 含义 |
|---|---|
| code | 6 位股票代码 |
| period | 月份（YYYY-MM） |
| ret | 月度收益率（后复权，含分红再投资） |
| suspended | 整月停牌标记（该月收益记 0） |
| mret | 市场月度收益率（中证全指全收益） |
| rf | 月度无风险利率（1 年期国债年化按 (1+y)^(1/12)-1 折算） |
| ret_ex | 个股超额收益 = ret − rf |
| mret_ex | 市场超额收益 = mret − rf |
| name / industry / sector | 股票简称 / 证监会行业大类 / 门类 |
| status | listed：样本期内未退市；delisted：2020 年后退市 |
| valid_est_months | 估计期（2015–2019）有效配对月份数 |

## returns_wide.csv

行：月份；列：股票代码；值：月度收益率（缺失为空）。

## sample.csv

最终 200 只样本：代码、简称、行业、板块、状态、估计期有效月份、行业配额。
"""
    (docs / "数据字典.md").write_text(dictionary, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="S3 清洗与面板构建")
    args = ap.parse_args(argv)

    cfg = load_config()
    ensure_dirs(cfg)
    log = get_logger(cfg)
    paths = project_paths(cfg)
    log.info("S3 开始：构建面板")

    panel, exclusions = build_panel(cfg, log)
    cand = load_candidates(cfg)

    sample, excl2, n_eligible = finalize_sample(panel, cand, cfg, log)
    all_excl = pd.concat([exclusions, excl2[["code", "name", "industry", "valid_est_months", "reason"]]],
                         ignore_index=True) if len(exclusions) else excl2

    # 只保留样本
    panel_out = panel[panel["code"].isin(set(sample["code"]))].copy()
    wide = panel_out.pivot_table(index="period", columns="code", values="ret", dropna=False)
    wide = wide.sort_index()

    save_csv(panel_out, paths["processed"] / "panel_monthly.csv")
    save_csv(wide.reset_index(), paths["processed"] / "returns_wide.csv")
    save_csv(sample, paths["processed"] / "sample.csv")
    save_csv(all_excl, paths["processed"] / "exclusion_log.csv")
    write_docs(cfg, panel_out, sample, all_excl, log, n_eligible)

    # 概览
    est_n = panel_out[panel_out["period"].isin([str(p) for p in est_months(cfg)])]["ret_ex"].notna().sum()
    log.info("面板覆盖：样本 %d 只，估计期有效观测 %d，检验期月份 %d",
             sample["code"].nunique(), est_n, len(test_months(cfg)))

    save_json(run_manifest(cfg, "s3_clean", {
        "sample_n": len(sample),
        "industries": int(sample["industry"].nunique()),
        "panel_rows": len(panel_out),
        "suspended_rows": int(panel_out["suspended"].sum()),
        "excluded_n": len(all_excl),
    }), paths["processed"] / "s3_manifest.json")
    log.info("S3 完成")


if __name__ == "__main__":
    main()
