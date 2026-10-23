"""公共层：路径、配置、日志、IO、时间轴、溯源信息。

所有模块统一从这里取路径与口径，避免口径散落在各处。
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.yaml"

_LOGGER: logging.Logger | None = None


# --------------------------------------------------------------------------
# 配置与路径
# --------------------------------------------------------------------------
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """读取 YAML 配置。"""
    p = Path(path) if path else CONFIG_PATH
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def project_paths(cfg: dict) -> dict[str, Path]:
    """把配置里的相对路径解析成绝对路径。"""
    return {k: ROOT / v for k, v in cfg["paths"].items()}


def ensure_dirs(cfg: dict) -> None:
    for p in project_paths(cfg).values():
        p.mkdir(parents=True, exist_ok=True)


def get_logger(cfg: dict | None = None, name: str = "capm") -> logging.Logger:
    """控制台 + 文件双写日志（同一文件追加，便于事后追溯）。"""
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if cfg is not None:
        logs = ROOT / cfg["paths"]["logs"]
        logs.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logs / f"{name}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.info("=" * 78)

    _LOGGER = logger
    return logger


# --------------------------------------------------------------------------
# 溯源信息
# --------------------------------------------------------------------------
def git_rev(short: bool = True) -> str | None:
    """当前提交号，写进 manifest 以便任何数字都能追到代码版本。"""
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
        out = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True, timeout=10, check=False
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return None


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def package_versions() -> dict[str, str]:
    import importlib

    out = {}
    for mod in ("pandas", "numpy", "scipy", "statsmodels", "matplotlib", "akshare", "yaml"):
        try:
            m = importlib.import_module(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            out[mod] = "not-installed"
    out["python"] = sys.version.split()[0]
    return out


# --------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------
def save_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=index, encoding="utf-8-sig")
    return p


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def save_json(obj: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    tmp.replace(p)
    return p


def load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_manifest(cfg: dict, stage: str, extra: dict | None = None) -> dict:
    """每个阶段产出一份可追溯的运行记录。"""
    return {
        "stage": stage,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_rev": git_rev(),
        "config_sha256": sha256_file(CONFIG_PATH),
        "versions": package_versions(),
        "extra": extra or {},
    }


# --------------------------------------------------------------------------
# 时间轴
# --------------------------------------------------------------------------
def to_month_index(s: pd.Series | pd.Index) -> pd.PeriodIndex:
    """任意日期序列 -> 月度 PeriodIndex（以自然月为准）。"""
    if isinstance(s, pd.Series):
        s = pd.to_datetime(s)
    else:
        s = pd.to_datetime(pd.Index(s))
    return pd.PeriodIndex(s, freq="M")


def month_range(start: str, end: str) -> pd.PeriodIndex:
    return pd.period_range(pd.Period(start, freq="M"), pd.Period(end, freq="M"), freq="M")


def est_months(cfg: dict) -> pd.PeriodIndex:
    s = cfg["sample"]
    return month_range(s["est_start"], s["est_end"])


def test_months(cfg: dict) -> pd.PeriodIndex:
    s = cfg["sample"]
    return month_range(s["test_start"], s["test_end"])


def fmt_pct(x: float, nd: int = 2) -> str:
    return f"{x * 100:.{nd}f}%"


def describe_series(x: pd.Series) -> dict:
    """描述性统计，报告表格会用。"""
    x = x.dropna()
    if len(x) == 0:
        return {"n": 0}
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)),
        "min": float(x.min()),
        "median": float(x.median()),
        "max": float(x.max()),
    }


def allocate(counts: pd.Series, n: int) -> pd.Series:
    """按层规模比例分配名额（最大余数法 + 容量上限），抽样与分箱共用。"""
    import numpy as np

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
