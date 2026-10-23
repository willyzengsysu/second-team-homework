"""S1 股票池构建与行业分层抽样。

口径（关键：无幸存者偏差）
  冻结于 universe_asof（2019-12-31）：
    该日已在市并仍在交易的股票 ∪ 该日仍在交易、之后才退市的股票
  只用该日已公开的信息，不按未来是否存续筛选。

行业
  证监会行业大类（baostock 一次调用）；2020 年后退市的股票不在当前分类表中，
  归入「未分类」独立成层，按规模分配名额。

产出
  data/raw/industry.csv          行业分类缓存
  data/interim/universe.csv      股票池
  data/interim/candidates.csv    分层抽样候选池（200 × 1.3 = 260）
  data/interim/s1_manifest.json  溯源
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from src.common import (
    ensure_dirs,
    get_logger,
    load_config,
    project_paths,
    run_manifest,
    save_csv,
    save_json,
)
from src.data_sources import fetch_delisted, fetch_industry, fetch_listed


def allocate(counts: pd.Series, n: int) -> pd.Series:
    """按层规模比例分配名额（最大余数法 + 容量上限）。"""
    counts = counts[counts > 0]
    if counts.empty or n <= 0:
        return pd.Series(dtype=int)
    target = int(min(n, counts.sum()))
    raw = counts / counts.sum() * target
    alloc = np.minimum(np.floor(raw).astype(int), counts.astype(int))
    leftover = target - int(alloc.sum())

    guard = 0
    while leftover > 0 and guard < 100_000:
        deficit = (raw - alloc).where(alloc < counts)
        if deficit.dropna().empty:
            break
        alloc[deficit.idxmax()] += 1
        leftover -= 1
        guard += 1
    return alloc.astype(int)


def draw_candidates(universe: pd.DataFrame, cfg: dict, log) -> pd.DataFrame:
    s = cfg["sample"]
    n_cand = int(math.ceil(s["target_n"] * s["oversample_ratio"]))
    rng = np.random.default_rng(int(cfg["project"]["seed"]))

    quotas = allocate(universe["industry"].value_counts(), n_cand).sort_index()

    picks = []
    for ind, q in quotas.items():
        pool = sorted(universe.loc[universe["industry"] == ind, "code"].tolist())
        take = min(int(q), len(pool))
        if take == 0:
            continue
        idx = rng.choice(len(pool), size=take, replace=False)
        picks.extend(
            {"code": pool[i], "industry": ind, "stratum_size": len(pool), "quota": take} for i in idx
        )

    cand = pd.DataFrame(picks).merge(universe, on=["code", "industry"], how="left")
    log.info("候选池 %d 只，覆盖 %d 个行业层", len(cand), cand["industry"].nunique())
    return cand.sort_values(["industry", "code"]).reset_index(drop=True)


def build_universe(cfg: dict, log, force: bool = False) -> pd.DataFrame:
    paths = project_paths(cfg)
    out = paths["interim"] / "universe.csv"
    if out.exists() and not force:
        log.info("股票池已缓存：%s（--force 重建）", out.name)
        return pd.read_csv(out, encoding="utf-8-sig", dtype={"code": str})

    s = cfg["sample"]
    asof = pd.Timestamp(s["universe_asof"])

    listed = fetch_listed()
    delisted = fetch_delisted()
    log.info("交易所接口：在市 %d 只，退市名单 %d 只", len(listed), len(delisted))

    listed = listed[listed["list_date"].isna() | (listed["list_date"] <= asof)].copy()

    n_gone = int((delisted["delist_date"].notna() & (delisted["delist_date"] <= asof)).sum())
    keep = delisted[
        (delisted["list_date"].isna() | (delisted["list_date"] <= asof))
        & delisted["delist_date"].notna()
        & (delisted["delist_date"] > asof)
    ].copy()
    log.info("退市股：保留 %d 只（冻结日后退市），剔除 %d 只（冻结日前已退市）", len(keep), n_gone)

    uni = pd.concat([listed, keep], ignore_index=True).drop_duplicates(subset="code", keep="first")

    # 行业
    ind_path = paths["raw"] / "industry.csv"
    if ind_path.exists() and not force:
        imap = pd.read_csv(ind_path, encoding="utf-8-sig", dtype={"code": str})
    else:
        imap = fetch_industry()
        save_csv(imap, ind_path)
    uni = uni.merge(imap, on="code", how="left")
    uni["industry"] = uni["industry"].fillna(s["industry_unknown_label"])
    uni["sector"] = uni["sector"].fillna(s["industry_unknown_label"])

    uni = uni[uni["board"].isin(s["boards"] + ["退市股"])].copy()
    uni = uni.sort_values("code").reset_index(drop=True)
    save_csv(uni, out)

    n_del = int((uni["status"] == "delisted").sum())
    log.info(
        "股票池 %d 只：在市 %d / 之后退市 %d；行业 %d 类（未分类 %d 只）",
        len(uni), len(uni) - n_del, n_del,
        uni["industry"].nunique(),
        int((uni["industry"] == s["industry_unknown_label"]).sum()),
    )
    return uni


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="S1 股票池与行业分层抽样")
    ap.add_argument("--force", action="store_true", help="忽略缓存重建")
    args = ap.parse_args(argv)

    cfg = load_config()
    ensure_dirs(cfg)
    log = get_logger(cfg)
    log.info("S1 开始：股票池冻结于 %s", cfg["sample"]["universe_asof"])

    uni = build_universe(cfg, log, force=args.force)
    cand = draw_candidates(uni, cfg, log)

    paths = project_paths(cfg)
    save_csv(cand, paths["interim"] / "candidates.csv")
    save_json(
        run_manifest(cfg, "s1_universe", {
            "universe_n": len(uni),
            "listed_n": int((uni["status"] == "listed").sum()),
            "delisted_after_asof_n": int((uni["status"] == "delisted").sum()),
            "industry_layers": int(uni["industry"].nunique()),
            "candidates_n": len(cand),
        }),
        paths["interim"] / "s1_manifest.json",
    )
    log.info("S1 完成")


if __name__ == "__main__":
    main()
