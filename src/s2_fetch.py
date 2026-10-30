"""S2 数据抓取：个股月线、市场全收益指数、月度无风险利率。

产出
  data/raw/prices/{code}.csv      个股月度行情缓存（不入库，可续抓）
  data/interim/market_monthly.csv 市场月度收益
  data/interim/rf_monthly.csv     月度无风险利率
  data/interim/fetch_log.csv      逐只抓取结果
  data/interim/s2_manifest.json   溯源
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from src.common import (
    ensure_dirs,
    get_logger,
    load_config,
    project_paths,
    price_cache_dir,
    sha256_file,
    run_manifest,
    save_csv,
    save_json,
    to_month_index,
)
from src.data_sources import (
    DataSourceError,
    fetch_market_index,
    fetch_riskfree,
    fetch_stock_prices,
    fetch_baostock_monthly,
    sleep_random,
    to_monthly,
)


def fetch_all_prices(codes: list[str], cfg: dict, log, force: bool = False) -> pd.DataFrame:
    """逐只抓取并缓存；已缓存的股票直接跳过，中断可续抓。"""
    s = cfg["sample"]
    # 多取一个月，保证 est_start 当月也能算出收益率
    start = (pd.Timestamp(s["est_start"]) - pd.DateOffset(months=1)).strftime("%Y%m%d")
    end = pd.Timestamp(s["test_end"]).strftime("%Y%m%d")
    raw_dir = price_cache_dir(cfg)
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows, n = [], len(codes)
    consecutive_failures = 0
    t0 = time.time()

    import baostock as bs
    use_bs = cfg["returns"]["provider"] == "baostock"
    if use_bs and cfg["returns"]["adjust"] != "hfq":
        raise ValueError("当前 baostock 主口径只支持后复权 hfq/adjustflag=1")
    login_needed = use_bs and any(not (raw_dir / f"{code}.csv").exists() or force for code in codes)
    if login_needed:
        login = bs.login()
        if login.error_code != "0":
            raise DataSourceError(f"baostock 登录失败：{login.error_msg}")
    for i, code in enumerate(codes, 1):
        cache = raw_dir / f"{code}.csv"
        if cache.exists() and not force:
            monthly = pd.read_csv(cache, encoding="utf-8-sig")
            status = "cache"
        else:
            try:
                quotes = (fetch_baostock_monthly(code, start, end) if use_bs else
                          fetch_stock_prices(code, start, end, cfg["returns"]["adjust"]))
                monthly = to_monthly(quotes)
                status = "ok" if len(monthly) else "empty"
            except DataSourceError as e:
                log.warning("  %s 失败：%s", code, e)
                monthly, status = pd.DataFrame(columns=["period", "date", "close"]), "failed"
            if len(monthly):
                save_csv(monthly, cache)
            sleep_random()          # 随机延迟，降低被封风险

        rows.append({
            "code": code,
            "status": status,
            "rows": len(monthly),
            "start": str(monthly["date"].iloc[0])[:10] if len(monthly) else None,
            "end": str(monthly["date"].iloc[-1])[:10] if len(monthly) else None,
        })
        consecutive_failures = consecutive_failures + 1 if status == "failed" else 0
        if consecutive_failures >= 3:
            save_csv(pd.DataFrame(rows), project_paths(cfg)["interim"] / "fetch_log.csv")
            raise DataSourceError("连续三只抓取失败，已保存进度并停止请求；稍后重跑可续抓")

        if i % 20 == 0 or i == n:
            ok = sum(1 for r in rows if r["status"] in ("ok", "cache"))
            rate = i / max(time.time() - t0, 1e-6)
            eta = (n - i) / rate / 60 if rate > 0 else 0
            log.info("  行情 %d/%d（有效 %d），已用 %.0f 分，预计剩余 %.0f 分",
                     i, n, ok, (time.time() - t0) / 60, eta)

    if login_needed:
        bs.logout()
    result = pd.DataFrame(rows)
    result["provider"] = cfg["returns"]["provider"]
    result["sha256"] = [sha256_file(raw_dir / f"{code}.csv") if (raw_dir / f"{code}.csv").exists() else None for code in result.code]
    save_csv(result, project_paths(cfg)["interim"] / "price_cache_manifest.csv")
    return result


def supplement_exit_quotes(cfg: dict, log, force=False) -> None:
    """同源日线补齐退市末月的月线缺口，不推断清偿或换股价值。"""
    import baostock as bs
    paths = project_paths(cfg)
    cand = pd.read_csv(paths["interim"] / "candidates.csv", dtype={"code": str})
    gone = pd.to_datetime(cand.delist_date, errors="coerce")
    cand = cand[gone.between(pd.Timestamp(cfg["sample"]["test_start"]),
                            pd.Timestamp(cfg["sample"]["test_end"]))]
    evidence = paths["raw"] / "daily_exit_baostock"
    evidence.mkdir(exist_ok=True)
    login_needed = force or any(not (evidence / f"{code}.csv").exists() for code in cand.code)
    if login_needed:
        login = bs.login()
        if login.error_code != "0":
            raise DataSourceError(login.error_msg)
    audit = []
    try:
        for r in cand.itertuples():
            path = evidence / f"{r.code}.csv"
            gone = pd.Timestamp(r.delist_date)
            start = (gone.to_period("M") - 1).start_time
            if force or not path.exists():
                symbol = ("sh." if r.code.startswith(("60", "68")) else "sz.") + r.code
                rs = bs.query_history_k_data_plus(symbol, "date,code,close,volume,tradestatus",
                        start_date=start.strftime("%Y-%m-%d"), end_date=gone.strftime("%Y-%m-%d"),
                        frequency="d", adjustflag="1")
                if rs.error_code != "0":
                    raise DataSourceError(f"退出日线 {r.code}: {rs.error_msg}")
                rows = []
                while rs.error_code == "0" and rs.next():
                    rows.append(rs.get_row_data())
                daily = pd.DataFrame(rows, columns=rs.fields)
                save_csv(daily, path)
                sleep_random()
            else:
                daily = pd.read_csv(path, dtype={"code": str})
            daily["date"] = pd.to_datetime(daily.date)
            for key in ["close", "volume", "tradestatus"]:
                daily[key] = pd.to_numeric(daily[key], errors="coerce")
            traded = daily[daily.tradestatus.eq(1) & daily.volume.gt(0) & daily.close.notna()].copy()
            cache = price_cache_dir(cfg) / f"{r.code}.csv"
            base = pd.read_csv(cache)
            base["date"] = pd.to_datetime(base.date)
            base["period"] = base.date.dt.to_period("M")
            traded["period"] = traded.date.dt.to_period("M")
            bars = traded.sort_values("date").groupby("period").agg(
                date=("date", "last"), close=("close", "last"), volume=("volume", "sum")).reset_index()
            merged = base.merge(bars, on="period", suffixes=("_month", "_daily"))
            # 相同复权口径在重叠月应一致；不把不同价格尺度静默拼接。
            if len(merged) and ((merged.close_month-merged.close_daily).abs()
                               > 1e-5 * merged.close_month.abs().clip(lower=1)).any():
                raise ValueError(f"{r.code} 同源日/月复权价格不一致，需人工核查")
            new = pd.concat([base[~base.period.isin(bars.period)], bars], ignore_index=True).sort_values("period")
            save_csv(new[["period", "date", "close", "volume"]], cache)
            audit.append({"code": r.code, "start": str(start.date()), "delist_date": r.delist_date,
                          "traded_daily_rows": len(traded), "daily_bar_months": len(bars),
                          "last_actual_trade_date": str(traded.date.max())[:10] if len(traded) else None,
                          "evidence_sha256": sha256_file(path),
                          "note": "仅补报价；终止清偿/换股价值仍待核实"})
    finally:
        if login_needed:
            bs.logout()
    save_csv(pd.DataFrame(audit), paths["interim"] / "terminal_quote_audit.csv")
    log.info("同源日线核查并补齐 %d 只候选的退出末月报价；不覆盖终止清偿收益", len(audit))


def fetch_market(cfg: dict, log) -> pd.DataFrame:
    m = cfg["market"]["primary"]
    s = cfg["sample"]
    start = (pd.Timestamp(s["est_start"]) - pd.DateOffset(months=1)).strftime("%Y%m%d")
    raw = fetch_market_index(m["code"], start, s["test_end"], m["url"])
    log.info("市场基准：%s（%s），%d 个交易日", m["name"], m["kind"], len(raw))

    raw["period"] = to_month_index(raw["date"])
    monthly = raw.sort_values("date").groupby("period", as_index=False).last()[["period", "date", "close"]]
    monthly["mret"] = monthly["close"].pct_change()
    monthly["index_name"] = m["name"]
    monthly["kind"] = m["kind"]
    return monthly.dropna(subset=["mret"]).reset_index(drop=True)


def build_riskfree(cfg: dict, log) -> pd.DataFrame:
    raw = fetch_riskfree(cfg)
    raw["period"] = to_month_index(raw["date"])
    monthly = raw.groupby("period", as_index=False).last()[["period", "date", "y_annual"]]

    months = pd.period_range(monthly["period"].min(), monthly["period"].max(), freq="M")
    before = len(monthly)
    monthly = monthly.set_index("period").reindex(months).ffill().reset_index()
    monthly = monthly.rename(columns={"index": "period"})
    filled = len(monthly) - before

    y = monthly["y_annual"] / 100.0
    if cfg["riskfree"]["annual_to_monthly"] == "compound":
        monthly["rf"] = (1.0 + y) ** (1.0 / 12.0) - 1.0
    else:
        monthly["rf"] = y / 12.0

    log.info("无风险利率：%d 个月（补齐 %d），年化均值 %.2f%%，月度口径 %s",
             len(monthly), filled, monthly["y_annual"].mean(), cfg["riskfree"]["annual_to_monthly"])
    return monthly


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="S2 数据抓取")
    ap.add_argument("--limit", type=int, default=None, help="只抓前N只（冒烟测试）")
    ap.add_argument("--force", action="store_true", help="忽略缓存重新抓取")
    ap.add_argument("--skip-prices", action="store_true", help="只抓指数与无风险利率")
    ap.add_argument("--refresh-benchmarks", action="store_true", help="更新已完整缓存的指数和利率")
    args = ap.parse_args(argv)

    cfg = load_config()
    ensure_dirs(cfg)
    log = get_logger(cfg)
    paths = project_paths(cfg)

    cand_path = paths["interim"] / "candidates.csv"
    codes: list[str] = []
    if cand_path.exists():
        codes = pd.read_csv(cand_path, encoding="utf-8-sig", dtype={"code": str})["code"].tolist()
    elif not args.skip_prices:
        raise FileNotFoundError("缺少 candidates.csv，请先运行 python -m src.s1_universe")
    if args.limit:
        codes = codes[: args.limit]

    if not args.skip_prices and codes:
        flog = fetch_all_prices(codes, cfg, log, force=args.force)
        if cfg["returns"]["provider"] == "baostock" and not args.limit:
            supplement_exit_quotes(cfg, log, args.force)
            for i, row in flog.iterrows():
                cache = price_cache_dir(cfg) / f"{row['code']}.csv"
                if cache.exists():
                    data = pd.read_csv(cache)
                    flog.loc[i, ["rows", "start", "end", "sha256"]] = [
                        len(data), str(data.date.iloc[0])[:10], str(data.date.iloc[-1])[:10], sha256_file(cache)]
            save_csv(flog, paths["interim"] / "price_cache_manifest.csv")
        save_csv(flog, paths["interim"] / "fetch_log.csv")
        log.info("行情抓取：成功 %d / 失败 %d / 空 %d",
                 int((flog["status"].isin(["ok", "cache"])).sum()),
                 int((flog["status"] == "failed").sum()),
                 int((flog["status"] == "empty").sum()))
        if flog["status"].isin(["failed", "empty"]).any():
            raise DataSourceError("候选行情不完整，请处理 fetch_log 后续抓，避免按抓取成败筛样")

    required = set(str(p) for p in pd.period_range(cfg["sample"]["est_start"],
                                                  cfg["sample"]["test_end"], freq="M"))
    def complete_cache(name, value):
        path = paths["interim"] / name
        if args.force or args.refresh_benchmarks or not path.exists():
            return None
        d = pd.read_csv(path, encoding="utf-8-sig")
        if not required.issubset(set(d.period)) or d.set_index("period").loc[sorted(required), value].isna().any():
            return None
        if name.startswith("market") and not (d["kind"].eq("total_return").all()
                and d["index_name"].eq(cfg["market"]["primary"]["name"]).all()):
            return None
        log.info("复用完整基准缓存：%s", name)
        return d
    market = complete_cache("market_monthly.csv", "mret")
    if market is None:
        market = fetch_market(cfg, log)
        save_csv(market, paths["interim"] / "market_monthly.csv")
    rf = complete_cache("rf_monthly.csv", "rf")
    if rf is None:
        rf = build_riskfree(cfg, log)
    else:
        y = rf["y_annual"] / 100
        rf["rf"] = ((1 + y) ** (1 / 12) - 1 if cfg["riskfree"]["annual_to_monthly"] == "compound" else y / 12)
    save_csv(rf, paths["interim"] / "rf_monthly.csv")

    save_json(
        run_manifest(cfg, "s2_fetch", {
            "stocks_requested": len(codes),
            "market_source": str(market["index_name"].iloc[0]),
            "market_kind": str(market["kind"].iloc[0]),
            "market_months": len(market),
            "rf_months": len(rf),
            "rf_mean_annual_pct": float(rf["y_annual"].mean()),
        }),
        paths["interim"] / "s2_manifest.json",
    )
    log.info("S2 完成：market %d 月 / rf %d 月", len(market), len(rf))


if __name__ == "__main__":
    main()
