"""一键复现：按顺序跑完 S1–S7。

用法
    python scripts/run_all.py                 # 全流程
    python scripts/run_all.py --from s3       # 从 S3 开始（数据已抓完时用）
    python scripts/run_all.py --only s4 s5    # 只跑指定阶段

每个阶段都把耗时和关键数字写进 logs/capm.log，并各自产出 *_manifest.json。
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import load_config, get_logger  # noqa: E402

STAGES = [
    ("s1", "src.s1_universe", "股票池与行业分层抽样"),
    ("s2", "src.s2_fetch", "个股行情 / 指数 / 无风险利率"),
    ("s3", "src.s3_clean", "清洗与面板构建"),
    ("s4", "src.s4_beta", "估计期贝塔与定义法核对"),
    ("s5", "src.s5_portfolio", "五组合与检验期分析"),
    ("s6", "src.s6_figures", "SML 作图"),
    ("s7", "src.s7_report_tables", "报告表格与数字锁定"),
]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="一键运行全流程")
    ap.add_argument("--from", dest="start", default="s1", choices=[s[0] for s in STAGES])
    ap.add_argument("--only", nargs="+", default=None, choices=[s[0] for s in STAGES])
    args = ap.parse_args(argv)

    cfg = load_config()
    log = get_logger(cfg)
    log.info("=" * 78)
    log.info("开始运行全流程（%s）", cfg["project"]["title"])

    if args.only:
        todo = [s for s in STAGES if s[0] in args.only]
    else:
        idx = [s[0] for s in STAGES].index(args.start)
        todo = STAGES[idx:]

    total = time.time()
    for key, module, desc in todo:
        log.info("── %s %s ──", key.upper(), desc)
        t0 = time.time()
        mod = importlib.import_module(module)
        mod.main([])
        log.info("── %s 用时 %.1f 秒 ──", key.upper(), time.time() - t0)

    log.info("全流程完成，总用时 %.1f 分钟", (time.time() - total) / 60)


if __name__ == "__main__":
    main()
