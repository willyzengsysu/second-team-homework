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
    run_manifest,
    save_csv,
    save_json,
    to_month_index,
)
from src.data_sources import (
    DataSourceError,
    fetch_market_index,
    fetch_riskfree,
    fetch_stock_daily,
    sleep_random,
    to_monthly,
)


def fetch_all_prices(codes: list[str], cfg: dict, log, force: bool = False) -> pd.DataFrame:
    """逐只抓取并缓存；已缓存的股票直接跳过，中断可续抓。"""
    s = cfg["sample"]
    # 多取一个月，保证 est_start 当月也能算出收益率
    start = (pd.Timestamp(s["est_start"]) - pd.DateOffset(months=1)).strftime("%Y%m%d")
    end = pd.Timestamp(s["test_end"]).strftime("%Y%m%d")
    raw_dir = project_paths(cfg)["raw"] / "prices"
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows, n = [], len(codes)
    t0 = time.time()

    for i, code in enumerate(codes, 1):
        cache = raw_dir / f"{code}.csv"
        if cache.exists() and not force:
            monthly = pd.read_csv(cache, encoding="utf-8-sig")
            status = "cache"
        else:
            try:
                daily = fetch_stock_daily(code, start, end, cfg["returns"]["adjust"])
                monthly = to_monthly(daily)
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

        if i % 20 == 0 or i == n:
            ok = sum(1 for r in rows if r["status"] in ("ok", "cache"))
            rate = i / max(time.time() - t0, 1e-6)
            eta = (n - i) / rate / 60 if rate > 0 else 0
            log.info("  行情 %d/%d（有效 %d），已用 %.0f 分，预计剩余 %.0f 分",
                     i, n, ok, (time.time() - t0) / 60, eta)

    return pd.DataFrame(rows)


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
        save_csv(flog, paths["interim"] / "fetch_log.csv")
        log.info("行情抓取：成功 %d / 失败 %d / 空 %d",
                 int((flog["status"].isin(["ok", "cache"])).sum()),
                 int((flog["status"] == "failed").sum()),
                 int((flog["status"] == "empty").sum()))

    market = fetch_market(cfg, log)
    save_csv(market, paths["interim"] / "market_monthly.csv")

    rf = build_riskfree(cfg, log)
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
