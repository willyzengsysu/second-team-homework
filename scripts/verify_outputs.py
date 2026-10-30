"""对真实产物作独立恒等式/边界核查，不调用流水线写输出。"""
import hashlib
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import load_config, load_csv, project_paths, price_cache_dir
from src.s1_universe import a_share_mask


def main():
    parser = argparse.ArgumentParser(description="核对已提交的分析产物及可选原始缓存")
    parser.add_argument("--offline", action="store_true", help="只核对仓库内数据/结果；不读取未入库的原始缓存")
    args = parser.parse_args()
    cfg = load_config()
    paths = project_paths(cfg)
    p = load_csv(paths["processed"] / "panel_monthly.csv")
    sample = load_csv(paths["processed"] / "sample.csv")
    universe = load_csv(paths["interim"] / "universe.csv")
    members = load_csv(paths["tables"] / "portfolio_membership.csv")
    stock = load_csv(paths["tables"] / "stock_test.csv")
    excluded = load_csv(paths["tables"] / "stock_test_exclusions.csv")
    port = load_csv(paths["tables"] / "portfolio_stats.csv")
    returns = load_csv(paths["tables"] / "portfolio_returns.csv")
    hl = load_csv(paths["tables"] / "hl_diff.csv").iloc[0]
    assert a_share_mask(universe.code).all() and a_share_mask(sample.code).all()
    assert len(sample) == cfg["sample"]["target_n"]
    assert sample.valid_est_months.ge(cfg["sample"]["min_est_months"]).all()
    assert pd.to_datetime(universe.industry_date.dropna()).le(pd.Timestamp(cfg["sample"]["universe_asof"])).all()
    assert not p.duplicated(["code", "period"]).any()
    assert set(sample.code) == set(members.code) == set(stock.code) | set(excluded.code)
    assert members.groupby("group").size().max() - members.groupby("group").size().min() <= 1
    assert not (p.suspended & p.event_source.fillna("").eq("")).any()
    assert p.loc[p.return_status.isin(["unknown_gap", "unknown_tail", "source_zero_volume", "post_delist"]), "ret"].isna().all()
    # 月度零成交量条目不得充当实际报价；有价格的月收益必须来自相邻自然月。
    assert not (p.zero_volume_month & p.quote_observed).any()
    for _, g in p.groupby("code"):
        g = g.sort_values("period")
        observed = g.return_status.eq("observed")
        expected = g.valuation_close / g.valuation_close.shift(1) - 1
        np.testing.assert_allclose(g.loc[observed, "ret"], expected.loc[observed], atol=1e-12)
    # 回归输出用 numpy 最小二乘独立复算，不复用生产回归函数。
    tp = p[p.period.between("2020-01", "2024-12")]
    for r in stock.itertuples():
        g = tp[tp.code.eq(r.code)][["ret_ex", "mret_ex"]].dropna()
        params = np.linalg.lstsq(np.column_stack([np.ones(len(g)), g.mret_ex]), g.ret_ex, rcond=None)[0]
        np.testing.assert_allclose(params, [r.alpha, r.beta_test], atol=1e-12)
        assert len(g) == r.n_obs
    # 每组收益实际等于当月可得成员均值；核对成员统计与固定分组。
    grouped = tp.merge(members[["code", "group"]], on="code").groupby(["period", "group"]).ret.mean().unstack()
    wr = returns.set_index("period")
    for g in cfg["portfolio"]["group_labels"]:
        np.testing.assert_allclose(grouped[g], wr[g], atol=1e-12)
    d = wr[cfg["portfolio"]["group_labels"][-1]] - wr[cfg["portfolio"]["group_labels"][0]]
    assert abs(d.mean()-hl.mean_diff_monthly) < 1e-12
    assert abs(hl.mean_diff_monthly-hl.alpha_diff-hl.capm_predicted_diff) < 1e-12
    assert abs(hl.mean_diff_annual-12*hl.mean_diff_monthly) < 1e-12
    lock = json.loads((paths["tables"] / "numbers_lock.json").read_text(encoding="utf-8"))
    for filename, expected in lock["sources"].items():
        assert hashlib.sha256((paths["tables"] / filename).read_bytes()).hexdigest() == expected, filename
    assert hashlib.sha256((ROOT / "report" / "报告.md").read_bytes()).hexdigest() == lock["report_sha256"]
    source_manifest = load_csv(paths["interim"] / "price_cache_manifest.csv")
    assert source_manifest.provider.eq(cfg["returns"]["provider"]).all()
    if not args.offline:
        for r in source_manifest.itertuples():
            path = price_cache_dir(cfg) / f"{r.code}.csv"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == r.sha256
    terminal = load_csv(paths["interim"] / "terminal_quote_audit.csv")
    for r in terminal.itertuples():
        if pd.notna(r.last_actual_trade_date):
            assert pd.Timestamp(r.last_actual_trade_date) <= pd.Timestamp(r.delist_date)
            if not args.offline:
                cache = load_csv(price_cache_dir(cfg) / f"{r.code}.csv")
                assert str(pd.Timestamp(r.last_actual_trade_date).to_period("M")) in set(cache.period)
    for filename, expected in lock["inputs"].items():
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == expected, filename
    print(f"真实产物核验通过：{len(sample)} 只样本、{len(stock)} 只个股回归、{len(port)} 组、{len(returns)} 个检验月；来源哈希与正文一致。")
    if args.offline:
        print("离线范围：已核对提交的数据、结果和正文；未验证原始行情缓存的内容或哈希。")


if __name__ == "__main__":
    main()
