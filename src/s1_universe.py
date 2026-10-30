"""S1 股票池构建与行业分层抽样。

口径（名单按历史日期重建；来源覆盖仍有限制）
  冻结于 universe_asof（2019-12-31）：
    该日已在市并仍在交易的股票 ∪ 该日仍在交易、之后才退市的股票
  只用该日已公开的信息，不按未来是否存续筛选。

行业
  证监会行业大类（baostock 按冻结日查询，验证 updateDate 不晚于冻结日）。

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
    ROOT,
    allocate,
    load_csv,
    sha256_file,
    ensure_dirs,
    get_logger,
    load_config,
    project_paths,
    run_manifest,
    save_csv,
    save_json,
)
from src.data_sources import fetch_delisted, fetch_industry, fetch_listed, _CSRC_SECTOR


def draw_tranche(universe: pd.DataFrame, n: int, seed: int, tag: str) -> pd.DataFrame:
    """按行业比例在给定股票池里抽一层。"""
    if n <= 0 or universe.empty:
        return pd.DataFrame(columns=["code", "industry", "stratum_size", "quota", "tranche"])

    rng = np.random.default_rng(seed)
    quotas = allocate(universe["industry"].value_counts(), n).sort_index()

    picks = []
    for ind, q in quotas.items():
        pool = sorted(universe.loc[universe["industry"] == ind, "code"].tolist())
        take = min(int(q), len(pool))
        if take == 0:
            continue
        idx = rng.choice(len(pool), size=take, replace=False)
        picks.extend(
            {"code": pool[i], "industry": ind, "stratum_size": len(pool), "quota": take, "tranche": tag}
            for i in idx
        )
    return pd.DataFrame(picks)


def draw_candidates(universe: pd.DataFrame, cfg: dict, log) -> pd.DataFrame:
    """两段式抽样：基础池 + 追加池。

    追加池的作用：基础池里总有一部分股票因估计期不足 48 个月被剔除，
    追加一批候选后，合格股票数才能补足到 target_n，同时不改动基础池的随机结果
    （避免重复抓取已有行情）。
    """
    s = cfg["sample"]
    seed = int(cfg["project"]["seed"])
    n_base = int(math.ceil(s["target_n"] * s["oversample_ratio"]))
    n_extra = int(s.get("extra_candidates", 0))

    base = draw_tranche(universe, n_base, seed, "base")
    cand = base

    if n_extra > 0:
        remaining = universe[~universe["code"].isin(base["code"])]
        extra = draw_tranche(remaining, n_extra, seed + 11, "extra")
        cand = pd.concat([base, extra], ignore_index=True)

    cand = cand.merge(universe.drop(columns=["industry"]), on="code", how="left")
    log.info(
        "候选池 %d 只（基础 %d + 追加 %d），覆盖 %d 个行业层",
        len(cand), len(base), len(cand) - len(base), cand["industry"].nunique(),
    )
    return cand.sort_values(["industry", "code"]).reset_index(drop=True)


def a_share_mask(codes: pd.Series) -> pd.Series:
    """接口名单只含股票；显式排除退市名单中的沪/深 B 股。"""
    return codes.astype(str).str.zfill(6).str.fullmatch(r"(?:60\d{4}|68\d{4}|00\d{4}|30\d{4})")


def historical_industry(frame: pd.DataFrame, asof: str) -> pd.DataFrame:
    required = {"code", "industry", "updateDate"}
    if not required.issubset(frame):
        raise ValueError("行业快照缺少 code/industry/updateDate，不能证明历史口径")
    d = frame.copy()
    d["industry_date"] = pd.to_datetime(d["updateDate"], errors="coerce")
    if d["industry_date"].isna().any() or (d["industry_date"] > pd.Timestamp(asof)).any():
        raise ValueError("行业快照包含无日期或冻结日之后的记录")
    d["code"] = d["code"].str.split(".").str[-1].str.zfill(6)
    d["industry"] = d["industry"].fillna("").str.strip().replace("", "未分类")
    d["sector"] = d["industry"].str[0].map(_CSRC_SECTOR).fillna("未分类")
    return d.sort_values("industry_date").drop_duplicates("code", keep="last")[[
        "code", "industry", "sector", "industry_date"]]


def build_universe(cfg: dict, log, force: bool = False) -> pd.DataFrame:
    s, paths = cfg["sample"], project_paths(cfg)
    snapshot = ROOT / s["listing_snapshot"]
    if force or not snapshot.exists():
        listed, delisted = fetch_listed(), fetch_delisted()
        meta = pd.concat([listed, delisted], ignore_index=True).drop_duplicates("code")
        save_csv(meta, snapshot)
    else:
        meta = load_csv(snapshot)
    meta = meta.drop(columns=["industry", "sector", "industry_date"], errors="ignore")
    meta["list_date"] = pd.to_datetime(meta.list_date, errors="coerce")
    meta["delist_date"] = pd.to_datetime(meta.get("delist_date"), errors="coerce")
    asof = pd.Timestamp(s["universe_asof"])
    valid = (a_share_mask(meta.code) & meta.list_date.notna() & (meta.list_date <= asof)
             & (meta.delist_date.isna() | (meta.delist_date > asof)))
    rejected = meta.loc[~valid].copy()
    rejected["reason"] = np.where(~a_share_mask(rejected.code), "非沪深A股",
                                  "上市/退出日期不符合冻结日或上市日期未知")
    save_csv(rejected, paths["interim"] / "universe_exclusions.csv")
    uni = meta.loc[valid].copy()
    industry_path = ROOT / s["industry_snapshot"]
    if force or not industry_path.exists():
        save_csv(fetch_industry(s["universe_asof"]), industry_path)
    imap = historical_industry(load_csv(industry_path), s["universe_asof"])
    uni = uni.merge(imap, on="code", how="left", validate="one_to_one")
    uni[["industry", "sector"]] = uni[["industry", "sector"]].fillna(s["industry_unknown_label"])
    uni["industry_asof"] = s["universe_asof"]
    uni = uni.sort_values("code").reset_index(drop=True)
    save_csv(uni, paths["interim"] / "universe.csv")
    log.info("冻结日 A 股 %d 只；剔除 %d 只；历史行业未分类 %d 只",
             len(uni), len(rejected), uni.industry.eq(s["industry_unknown_label"]).sum())
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
            "industry_max_date": str(uni.industry_date.max()),
            "snapshot_hashes": {k: sha256_file(ROOT / cfg["sample"][k])
                                for k in ["listing_snapshot", "industry_snapshot"]},
            "candidates_n": len(cand),
        }),
        paths["interim"] / "s1_manifest.json",
    )
    log.info("S1 完成")


if __name__ == "__main__":
    main()
