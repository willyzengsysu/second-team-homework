"""S3：不推断未知停牌、不跨未知缺月计算月收益，保留退出核查表。"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from src.common import (ROOT, allocate, ensure_dirs, est_months, get_logger,
                        load_config, load_csv, month_range, project_paths,
                        run_manifest, save_csv, save_json, test_months, price_cache_dir)


def load_candidates(cfg: dict) -> pd.DataFrame:
    cand = load_csv(project_paths(cfg)["interim"] / "candidates.csv")
    for col in ("industry", "sector"):
        cand[col] = cand[col].fillna("").str.strip().replace("", cfg["sample"]["industry_unknown_label"])
    if cand.code.duplicated().any():
        raise ValueError("候选池股票代码重复")
    return cand


def load_events(cfg: dict) -> pd.DataFrame:
    path = ROOT / cfg["missing"]["verified_events"]
    columns = ["code", "period", "event", "value", "source"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    d = load_csv(path)
    if not set(columns).issubset(d):
        raise ValueError("人工核实事件缺少必需字段")
    if d.duplicated(["code", "period"]).any() or d.source.fillna("").str.strip().eq("").any():
        raise ValueError("核实事件必须每股每月唯一且提供来源")
    if not d.event.isin(["suspension", "terminal_return"]).all():
        raise ValueError("未知事件类型")
    return d


def monthly_observations(price: pd.DataFrame, grid: pd.PeriodIndex,
                         list_date=None, delist_date=None, events=None,
                         zero_marked: bool = False, min_close: float = 0.01) -> pd.DataFrame:
    out = pd.DataFrame(index=grid)
    out.index.name = "period"
    out["close"] = np.nan
    out["quote_date"] = pd.NaT
    out["zero_volume_month"] = False
    if price is not None and not price.empty:
        q = price.copy()
        q["date"] = pd.to_datetime(q.date, errors="coerce")
        q["period"] = q.date.dt.to_period("M")
        q = q.dropna(subset=["date"]).sort_values("date").drop_duplicates("period", keep="last").set_index("period")
        out["close"] = pd.to_numeric(q.close, errors="coerce").reindex(grid)
        out["quote_date"] = q.date.reindex(grid)
        if "volume" in q:
            out["zero_volume_month"] = pd.to_numeric(q.volume, errors="coerce").reindex(grid).eq(0)
    invalid = out.close.notna() & (~np.isfinite(out.close) | (out.close < min_close))
    out.loc[invalid, "close"] = np.nan
    out.loc[out.zero_volume_month, "close"] = np.nan
    listed = pd.to_datetime(list_date, errors="coerce")
    gone = pd.to_datetime(delist_date, errors="coerce")
    before = pd.Series(False, index=grid) if pd.isna(listed) else pd.Series(grid < listed.to_period("M"), index=grid)
    after = pd.Series(False, index=grid) if pd.isna(gone) else pd.Series(grid > gone.to_period("M"), index=grid)
    out.loc[before | after, "close"] = np.nan
    observed = out.close.notna()
    out["return_status"] = "unknown_no_price"
    if observed.any():
        first, last = out[observed].index[[0, -1]]
        out.loc[(grid < first) & ~before, "return_status"] = "source_head_missing"
        out.loc[(grid > last) & ~after, "return_status"] = "unknown_tail"
        out.loc[(grid >= first) & (grid <= last) & ~observed, "return_status"] = "unknown_gap"
    out.loc[invalid, "return_status"] = "invalid_price"
    out.loc[out.zero_volume_month, "return_status"] = "source_zero_volume"
    out.loc[before, "return_status"] = "pre_listing"
    out.loc[after, "return_status"] = "post_delist"
    out["suspended"] = False
    out["event_source"] = ""
    values = out.close.copy()
    overrides = {}
    if events is not None:
        for r in events.itertuples():
            period = pd.Period(r.period, freq="M")
            if period not in grid or before.loc[period] or after.loc[period]:
                raise ValueError("核实事件超出股票的有效上市期间")
            out.loc[period, "event_source"] = r.source
            if r.event == "suspension":
                if observed.loc[period]:
                    raise ValueError("整月停牌事件与该月实际行情冲突")
                out.loc[period, "suspended"] = True
                out.loc[period, "return_status"] = "confirmed_suspension"
            else:
                value = float(r.value)
                if not np.isfinite(value) or value < -1:
                    raise ValueError("终止收益必须有限且不小于 -100%")
                overrides[period] = value
    # 只为有独立证据的整月停牌延续估值；未知缺月中断价格链。
    if zero_marked:
        for i in range(1, len(grid)):
            if out.suspended.iloc[i] and pd.notna(values.iloc[i-1]):
                values.iloc[i] = values.iloc[i-1]
    out["valuation_close"] = values
    out["ret"] = values.pct_change(fill_method=None)
    out.loc[observed & out.ret.isna(), "return_status"] = "previous_price_missing"
    out.loc[out.ret.notna() & ~out.suspended, "return_status"] = "observed"
    for period, value in overrides.items():
        out.loc[period, ["ret", "return_status"]] = [value, "verified_terminal_return"]
    out["quote_observed"] = observed
    return out


def stock_monthly_returns(price, grid, zero_marked=False):
    """兼容的两序列接口；没有独立事件证据，zero_marked 也不会填未知缺月。"""
    d = monthly_observations(price, grid, zero_marked=zero_marked)
    return d.ret, d.suspended


def build_panel(cfg: dict, log) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths, s = project_paths(cfg), cfg["sample"]
    cand, events = load_candidates(cfg), load_events(cfg)
    uncached = [code for code in cand.code if not (price_cache_dir(cfg) / f"{code}.csv").exists()]
    if uncached:
        raise ValueError(f"还有 {len(uncached)} 只候选缺少主来源缓存，请先完成 S2，不按抓取成败筛样")
    market = load_csv(paths["interim"] / "market_monthly.csv")
    rf = load_csv(paths["interim"] / "rf_monthly.csv")
    start = (pd.Period(s["est_start"], freq="M") - 1).strftime("%Y-%m")
    grid = month_range(start, s["test_end"])
    expected = set(str(m) for m in est_months(cfg).union(test_months(cfg)))
    for data, value in [(market, "mret"), (rf, "rf")]:
        if data.period.duplicated().any() or not expected.issubset(set(data.period)):
            raise ValueError(f"{value} 时间轴有重复或缺月")
        if data.set_index("period").loc[sorted(expected), value].isna().any():
            raise ValueError(f"{value} 研究期有缺失")
    rows, exclusions, exits = [], [], []
    for r in cand.itertuples():
        cache = price_cache_dir(cfg) / f"{r.code}.csv"
        price = load_csv(cache) if cache.exists() else pd.DataFrame(columns=["date", "close"])
        obs = monthly_observations(price, grid, r.list_date, r.delist_date,
                events[events.code.eq(r.code)],
                cfg["missing"]["full_month_suspension"] == "zero_marked",
                cfg["returns"]["min_close"])
        obs["code"] = r.code
        rows.append(obs.reset_index())
        if not cache.exists():
            exclusions.append({"code": r.code, "reason": "无行情缓存（抓取失败）"})
        valid = obs[obs.quote_observed]
        last = valid.index.max() if len(valid) else None
        gone = pd.to_datetime(r.delist_date, errors="coerce")
        unresolved = obs.loc[[m for m in grid if str(m) in expected], "return_status"].isin(
            ["unknown_tail", "unknown_gap", "source_head_missing", "unknown_no_price", "invalid_price", "source_zero_volume"])
        exits.append({"code": r.code, "name": r.name, "delist_date": r.delist_date,
                      "last_quote_date": str(valid.quote_date.max())[:10] if len(valid) else None,
                      "last_quote_month": str(last) if last else None,
                      "unresolved_price_months": int(unresolved.sum()),
                      "delist_in_test": bool(pd.notna(gone) and pd.Timestamp(s["test_start"]) <= gone <= pd.Timestamp(s["test_end"])),
                      "terminal_cashflow_verified": bool((events.code.eq(r.code) & events.event.eq("terminal_return")).any()),
                      "note": "行情末端不代表可成交退出；终止现金流未核实" if pd.notna(gone) and gone <= pd.Timestamp(s["test_end"]) else ""})
    panel = pd.concat(rows, ignore_index=True)
    panel["period"] = panel.period.astype(str)
    panel = panel.merge(cand.drop(columns=["stratum_size", "quota", "tranche"], errors="ignore"), on="code", validate="many_to_one")
    panel["mret"] = panel.period.map(market.set_index("period").mret)
    panel["rf"] = panel.period.map(rf.set_index("period").rf)
    panel["ret_ex"] = panel.ret - panel.rf
    panel["mret_ex"] = panel.mret - panel.rf
    valid = panel[panel.period.isin([str(m) for m in est_months(cfg)]) & panel.ret_ex.notna() & panel.mret_ex.notna()].groupby("code").size()
    panel["valid_est_months"] = panel.code.map(valid).fillna(0).astype(int)
    save_csv(pd.DataFrame(exits), paths["processed"] / "exit_audit_candidates.csv")
    return panel, pd.DataFrame(exclusions, columns=["code", "reason"])


def finalize_sample(panel, cand, cfg, log):
    counts = panel.groupby("code").valid_est_months.max()
    info = cand.copy()
    info["valid_est_months"] = info.code.map(counts).fillna(0).astype(int)
    info["eligible"] = info.valid_est_months >= cfg["sample"]["min_est_months"]
    pool = info[info.eligible].copy()
    if len(pool) < cfg["sample"]["target_n"]:
        raise ValueError(f"合格股票仅 {len(pool)} 只，请扩大候选池并补抓，不按未来存续补样")
    rng = np.random.default_rng(cfg["project"]["seed"] + 1)
    quotas = allocate(pool.industry.value_counts(), cfg["sample"]["target_n"]).sort_index()
    picked = []
    for ind, q in quotas.items():
        codes = sorted(pool.loc[pool.industry.eq(ind), "code"])
        picked.extend(codes[i] for i in rng.choice(len(codes), int(q), replace=False))
    sample = pool[pool.code.isin(picked)].copy()
    sample["group_quota"] = sample.industry.map(quotas)
    excl = info[~info.eligible].copy()
    excl["reason"] = f"估计期有效配对月份不足 {cfg['sample']['min_est_months']}"
    log.info("候选 %d，合格 %d，最终入样 %d，历史行业 %d 类", len(info), len(pool), len(sample), sample.industry.nunique())
    return sample, excl, len(pool)


def write_docs(cfg, panel, sample, excl, log, n_eligible):
    paths = project_paths(cfg)
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    candidates = load_candidates(cfg)
    text = f"""# 样本筛选记录

| 步骤 | 数量 |
|---|---:|
| 冻结日沪深 A 股（明确排除 B 股） | {len(load_csv(paths['interim'] / 'universe.csv'))} |
| 历史行业分层候选池 | {len(candidates)} |
| 估计期至少 {cfg['sample']['min_est_months']} 个有效配对月 | {n_eligible} |
| 合格但未抽中 | {n_eligible-len(sample)} |
| 最终样本 | {len(sample)} |
| 覆盖历史行业（含未分类层） | {sample.industry.nunique()} |

行业来源：{cfg['sample']['industry_snapshot']}；updateDate 不晚于冻结日。
上市/退出日期来自保存的名单快照；仍可能遗漏接口未覆盖的历史证券，不宣称完全消除选择偏差。

未知缺月不填零，也不跨缺月把多月涨跌作为单月收益。只有具有来源的独立停牌事件可标为 suspended。
退市日期以后的价格不入面板；退市前行情缺口和终止现金流不足见 exit_audit.csv。
组合可得成员均值属于统计口径，不能视为在未知缺失股票上实现了无成本退出。

逐月缺失原因见 missing_audit.csv；退出审计见 exit_audit.csv。
"""
    (docs / "样本筛选记录.md").write_text(text, encoding="utf-8")
    dictionary = """# 数据字典

所有收益均为小数，月份为 YYYY-MM，code 为六位字符串。

| 字段 | 含义 |
|---|---|
| ret / ret_ex | 月收益 / 扣同期无风险收益后的超额收益 |
| close / quote_date | baostock 后复权价格及来源月线日期；不一定是个股末次交易日，也不是退出成交证据 |
| mret / mret_ex | 中证全指全收益月收益 / 市场超额收益 |
| rf | 同月末 1 年期国债年化到期收益率的月复利折算代理 |
| quote_observed | 该月是否有有效实际行情；有行情不代表整月正常交易 |
| suspended | 独立事件证据确认的整月停牌，不从缺月推断 |
| return_status | 收益可用性及缺失原因，详见下表 |
| event_source | 核实停牌或终止收益的来源；未知时为空 |
| industry / industry_date | 冻结日行业及来源 updateDate |
| status | 名单抓取时 listed/delisted；不是在检验期末的存续状态 |
| list_date / delist_date | 名单提供的上市/退出日期，用于时间边界与审计 |
| valid_est_months | 估计期股票与市场超额收益均可用的月份数 |

| return_status | 含义 |
|---|---|
| observed | 当月与上月均有可用价格，正常计算月收益 |
| previous_price_missing | 当月有价格，上月缺失；不把跨月累计涨跌当作单月收益 |
| unknown_gap | 首末有效价格之间缺少行情，原因未核实 |
| unknown_tail | 最后行情后、正式退出前缺少行情，不认定自然退出 |
| source_head_missing / unknown_no_price | 数据头部缺失 / 无有效行情 |
| pre_listing / post_delist | 上市月之前 / 退出月之后的结构性空白 |
| source_zero_volume | 来源月线成交量为零，主口径不视作有效交易报价 |
| invalid_price | 非有限或低于配置阈值的价格 |
| confirmed_suspension | 独立事件确认的整月停牌；strict 时仍缺失 |
| verified_terminal_return | 有来源的终止收益覆盖值 |

sample.csv 为最终样本和历史行业信息。returns_wide.csv 为样本月收益宽表。
missing_audit.csv 逐行保留收益缺失及原因；exit_audit.csv 保留退出日期、末次行情及终止现金流核实状态。
portfolio_membership.csv 为固定分组；portfolio_member_counts.csv 为每组每月实际可得成员数。
stock_test_exclusions.csv 只记录因检验期不足而无法回归的个股，未从已形成的组合删除这些股票。
"""
    (docs / "数据字典.md").write_text(dictionary, encoding="utf-8")


def main(argv=None):
    argparse.ArgumentParser(description="S3 清洗与审计").parse_args(argv)
    cfg = load_config()
    ensure_dirs(cfg)
    log, paths = get_logger(cfg), project_paths(cfg)
    panel, fetch_excl = build_panel(cfg, log)
    sample, excl, n_eligible = finalize_sample(panel, load_candidates(cfg), cfg, log)
    if len(fetch_excl):
        excl = excl.merge(fetch_excl.rename(columns={"reason": "fetch_reason"}), on="code", how="left")
    panel = panel[panel.code.isin(sample.code)].copy()
    save_csv(panel, paths["processed"] / "panel_monthly.csv")
    save_csv(panel.pivot(index="period", columns="code", values="ret").reset_index(), paths["processed"] / "returns_wide.csv")
    save_csv(sample, paths["processed"] / "sample.csv")
    save_csv(excl, paths["processed"] / "exclusion_log.csv")
    audit = load_csv(paths["processed"] / "exit_audit_candidates.csv")
    save_csv(audit[audit.code.isin(sample.code)], paths["processed"] / "exit_audit.csv")
    missing = panel[panel.ret.isna() | panel.return_status.ne("observed")]
    save_csv(missing, paths["processed"] / "missing_audit.csv")
    write_docs(cfg, panel, sample, excl, log, n_eligible)
    save_json(run_manifest(cfg, "s3_clean", {
        "sample_n": len(sample), "eligible_n": n_eligible, "industries": int(sample.industry.nunique()),
        "panel_rows": len(panel), "confirmed_suspension_rows": int(panel.suspended.sum()),
        "missing_by_status": missing.return_status.value_counts().to_dict(),
        "excluded_n": len(excl),
    }), paths["processed"] / "s3_manifest.json")


if __name__ == "__main__":
    main()
