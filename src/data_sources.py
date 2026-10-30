"""数据源：主分析每类数据统一一个来源，不做静默多级 fallback。

来源分工（均为免费接口）
  股票名单    akshare 交易所官方接口（含退市名单）
  行业分类    baostock 证监会行业表（一次调用返回全市场）
  个股行情    baostock 后复权月线；腾讯接口仅保留作历史来源诊断
  市场指数    中证指数官网 H00985（中证全指全收益）
  无风险利率  akshare.bond_china_yield（中债国债 1 年期）

抓取纪律
  每次请求之间加随机延迟，失败后退避重试；宁可慢，不要被封 IP。
"""
from __future__ import annotations

import os
import random
import time

# 抓数期间绕过系统代理，避免代理规则干扰（仅影响本进程）
os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")
# 关闭 akshare 内部的 tqdm 进度条，保持日志干净
os.environ.setdefault("TQDM_DISABLE", "1")

import akshare as ak  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
_session = requests.Session()
_session.headers.update(UA)

# 随机延迟区间（秒）：单次请求之间
DELAY = (0.3, 0.9)


class DataSourceError(RuntimeError):
    pass


def sleep_random(lo: float | None = None, hi: float | None = None) -> None:
    lo = DELAY[0] if lo is None else lo
    hi = DELAY[1] if hi is None else hi
    time.sleep(random.uniform(lo, hi))


def retry(fn, retries: int = 3, base: float = 2.0, label: str = ""):
    """指数退避重试，失败抛出最后一次异常。"""
    last: Exception | None = None
    for i in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            if i < retries:
                time.sleep(base * i + random.uniform(0, 0.8))
    raise DataSourceError(f"{label} 抓取失败（重试 {retries} 次）：{last}")


def ts_code(code: str) -> str:
    """6 位代码 -> 带市场前缀（sh600000 / sz000001）。"""
    code = str(code).zfill(6)
    return f"sh{code}" if code.startswith(("60", "68", "90")) else f"sz{code}"


def _pick(df: pd.DataFrame, names: list[str], required: bool = True):
    for n in names:
        if n in df.columns:
            return df[n]
    if required:
        raise KeyError(f"未找到候选列 {names}；实际列 {list(df.columns)}")
    return None


# --------------------------------------------------------------------------
# 股票名单
# --------------------------------------------------------------------------
def fetch_listed() -> pd.DataFrame:
    """在市 A 股：沪市主板 / 科创板 / 深市主板（含创业板）。"""
    frames = []
    for symbol, board in (("主板A股", "沪市主板"), ("科创板", "科创板")):
        df = retry(lambda s=symbol: ak.stock_info_sh_name_code(symbol=s), label=f"沪市{board}")
        frames.append(
            pd.DataFrame(
                {
                    "code": _pick(df, ["证券代码", "代码"]).astype(str).str.zfill(6),
                    "name": _pick(df, ["证券简称", "简称"]).astype(str),
                    "list_date": pd.to_datetime(_pick(df, ["上市日期"], required=False), errors="coerce"),
                    "exchange": "SH",
                    "board": board,
                }
            )
        )
        sleep_random()

    df = retry(lambda: ak.stock_info_sz_name_code(symbol="A股列表"), label="深市A股")
    board_raw = _pick(df, ["板块"], required=False)
    frames.append(
        pd.DataFrame(
            {
                "code": _pick(df, ["A股代码", "证券代码"]).astype(str).str.zfill(6),
                "name": _pick(df, ["A股简称", "证券简称"]).astype(str),
                "list_date": pd.to_datetime(_pick(df, ["A股上市日期"], required=False), errors="coerce"),
                "exchange": "SZ",
                "board": (board_raw.astype(str).str.contains("创业").map({True: "创业板", False: "深市主板"})
                          if board_raw is not None else "深市主板"),
            }
        )
    )

    listed = pd.concat(frames, ignore_index=True).drop_duplicates(subset="code", keep="first")
    listed["status"] = "listed"
    return listed.reset_index(drop=True)


def fetch_delisted() -> pd.DataFrame:
    """已退市股票（含退市日期，供按 asof 过滤）。"""
    frames = []
    df = retry(lambda: ak.stock_info_sh_delist(), label="沪市退市名单")
    frames.append(
        pd.DataFrame(
            {
                "code": _pick(df, ["公司代码", "证券代码"]).astype(str).str.zfill(6),
                "name": _pick(df, ["公司简称", "证券简称"]).astype(str),
                "list_date": pd.to_datetime(_pick(df, ["上市日期"], required=False), errors="coerce"),
                "delist_date": pd.to_datetime(_pick(df, ["暂停上市日期", "终止上市日期"], required=False), errors="coerce"),
                "exchange": "SH",
                "board": "退市股",
            }
        )
    )
    sleep_random()

    df = retry(lambda: ak.stock_info_sz_delist(symbol="终止上市公司"), label="深市退市名单")
    frames.append(
        pd.DataFrame(
            {
                "code": _pick(df, ["证券代码", "代码"]).astype(str).str.zfill(6),
                "name": _pick(df, ["证券简称", "公司简称"]).astype(str),
                "list_date": pd.to_datetime(_pick(df, ["上市日期"], required=False), errors="coerce"),
                "delist_date": pd.to_datetime(_pick(df, ["终止上市日期"], required=False), errors="coerce"),
                "exchange": "SZ",
                "board": "退市股",
            }
        )
    )

    delisted = pd.concat(frames, ignore_index=True).drop_duplicates(subset="code", keep="first")
    delisted["status"] = "delisted"
    return delisted.reset_index(drop=True)


# --------------------------------------------------------------------------
# 行业分类
# --------------------------------------------------------------------------
_CSRC_SECTOR = {
    "A": "农、林、牧、渔业", "B": "采矿业", "C": "制造业",
    "D": "电力、热力、燃气及水生产和供应业", "E": "建筑业",
    "F": "批发和零售业", "G": "交通运输、仓储和邮政业", "H": "住宿和餐饮业",
    "I": "信息传输、软件和信息技术服务业", "J": "金融业", "K": "房地产业",
    "L": "租赁和商务服务业", "M": "科学研究和技术服务业",
    "N": "水利、环境和公共设施管理业", "O": "居民服务、修理和其他服务业",
    "P": "教育", "Q": "卫生和社会工作", "R": "文化、体育和娱乐业", "S": "综合",
}


def fetch_industry(asof: str) -> pd.DataFrame:
    """返回带 updateDate 的历史行业快照，不退回当前行业分类。"""
    import baostock as bs

    login = retry(lambda: bs.login(), label="baostock 登录")
    try:
        rs = bs.query_stock_industry(date=asof)
        if rs.error_code != "0":
            raise DataSourceError(f"query_stock_industry: {rs.error_msg}")
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
    finally:
        try:
            bs.logout()
        except Exception:  # noqa: BLE001
            pass

    df = pd.DataFrame(rows, columns=rs.fields)
    if df.empty:
        raise DataSourceError(f"历史行业 {asof} 返回空表")
    return df


# --------------------------------------------------------------------------
# 个股行情：日频 -> 月频
# --------------------------------------------------------------------------
def fetch_stock_prices(code: str, start: str, end: str, adjust: str = "hfq") -> pd.DataFrame:
    """腾讯月 K 线，同源直接请求；绕过 AKShare 无超时的起始日期探测。

    一次请求覆盖完整月度窗口，保留实际末次交易日期，不补停牌月。
    640 根上限足以覆盖本作业 121 个月；更长窗口显式报错。
    """
    if len(pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M")) > 640:
        raise ValueError("腾讯月线窗口超过 640 月，需分段请求")
    symbol = ts_code(code)
    def get():
        response = _session.get(
            "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
            params={"param": f"{symbol},month,{pd.Timestamp(start):%Y-%m-%d},{pd.Timestamp(end):%Y-%m-%d},640,{adjust}"},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code", 0) != 0:
            raise DataSourceError(f"腾讯返回错误：{payload.get('msg')}")
        data = payload.get("data", {}).get(symbol, {})
        key = f"{adjust}month" if adjust else "month"
        bars = data.get(key)
        # 对要求复权的请求不接受未复权 month 字段，避免静默改变口径。
        if bars is None:
            if not data:
                return pd.DataFrame(columns=["date", "close"])
            raise DataSourceError(f"腾讯 {code} 缺少 {key}，实际字段 {list(data)}")
        return pd.DataFrame([{"date": bar[0], "close": bar[2]} for bar in bars])
    out = retry(get, label=f"{code} 腾讯月线")
    if out.empty:
        return pd.DataFrame(columns=["date", "close"])
    out["date"] = pd.to_datetime(out.date)
    out["close"] = pd.to_numeric(out.close, errors="coerce")
    out = out[out.date.between(pd.Timestamp(start), pd.Timestamp(end))]
    return out.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """日频 -> 月末收盘（每月最后一个交易日）。"""
    if daily.empty:
        return pd.DataFrame(columns=["period", "date", "close"])
    d = daily.copy()
    d["period"] = pd.PeriodIndex(pd.to_datetime(d["date"]), freq="M")
    cols = ["period", "date", "close"] + (["volume"] if "volume" in d else [])
    m = d.sort_values("date").groupby("period", as_index=False).last()[cols]
    return m.reset_index(drop=True)


def fetch_baostock_monthly(code: str, start: str, end: str) -> pd.DataFrame:
    """须由调用者登录；单一来源后复权月线，保留零成交量信息供清洗审计。"""
    import baostock as bs
    symbol = ts_code(code)
    rs = bs.query_history_k_data_plus(
        symbol[:2] + "." + symbol[2:], "date,code,close,volume",
        start_date=pd.Timestamp(start).strftime("%Y-%m-%d"),
        end_date=pd.Timestamp(end).strftime("%Y-%m-%d"), frequency="m", adjustflag="1")
    if rs.error_code != "0":
        raise DataSourceError(f"baostock {code}: {rs.error_code} {rs.error_msg}")
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    out = pd.DataFrame(rows, columns=rs.fields)
    if out.empty:
        return pd.DataFrame(columns=["date", "close", "volume"])
    out["date"] = pd.to_datetime(out.date)
    for col in ["close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out[["date", "close", "volume"]]


# --------------------------------------------------------------------------
# 市场指数（中证全指全收益）
# --------------------------------------------------------------------------
def fetch_market_index(code: str, start: str, end: str, url: str) -> pd.DataFrame:
    """中证指数官网日行情。注意日期参数为 YYYYMMDD。"""
    params = {
        "indexCode": code,
        "startDate": pd.Timestamp(start).strftime("%Y%m%d"),
        "endDate": pd.Timestamp(end).strftime("%Y%m%d"),
    }

    def _get():
        r = _session.get(url, params=params, timeout=60, headers={"Referer": "https://www.csindex.com.cn/"})
        r.raise_for_status()
        js = r.json()
        data = js.get("data") or []
        if not data:
            raise DataSourceError(f"中证 {code} 返回空数据: {js.get('msg')}")
        return pd.DataFrame(data)

    df = retry(_get, label=f"中证指数 {code}")
    df = df.rename(columns={"tradeDate": "date"})
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------
# 无风险利率
# --------------------------------------------------------------------------
def fetch_riskfree(cfg: dict) -> pd.DataFrame:
    """中债国债收益率曲线 1 年期（宽表：曲线名称 / 日期 / 3月 / 6月 / 1年 / ...）。

    注意：该接口单次查询跨度不能超过一年（10 年区间会返回空表），故按年循环取数。
    """
    s = cfg["sample"]
    y0, y1 = pd.Timestamp(s["est_start"]).year, pd.Timestamp(s["test_end"]).year

    frames = []
    for year in range(y0, y1 + 1):
        df = retry(
            lambda y=year: ak.bond_china_yield(start_date=f"{y}0101", end_date=f"{y}1231"),
            label=f"中债国债 {year}",
        )
        if len(df):
            frames.append(df)
        sleep_random(0.5, 1.2)

    if not frames:
        raise DataSourceError("中债国债收益率各年度均返回空表")

    df = pd.concat(frames, ignore_index=True)
    date_col = next((c for c in df.columns if "日期" in str(c)), df.columns[1])
    curve_col = next((c for c in df.columns if "曲线" in str(c)), None)
    term = cfg["riskfree"]["term_column"]
    if term not in df.columns:
        raise DataSourceError(f"缺少期限列 {term}；实际列 {list(df.columns)}")

    if curve_col is not None:
        mask = df[curve_col].astype(str).str.contains(cfg["riskfree"]["curve_keyword"], na=False)
        df = df[mask]
    if df.empty:
        raise DataSourceError("未匹配到国债收益率曲线")

    out = pd.DataFrame(
        {"date": pd.to_datetime(df[date_col], errors="coerce"),
         "y_annual": pd.to_numeric(df[term], errors="coerce")}
    )
    return out.dropna().drop_duplicates(subset="date", keep="last").sort_values("date").reset_index(drop=True)
