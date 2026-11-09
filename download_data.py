"""
数据获取模块
============
通过东方财富接口获取ETF历史日线行情。

注意：
1. 默认请求后复权行情，adjust="hfq"。
2. start 和 end 均包含当天，end=None 表示查询至当天。
3. 复权是否正确仍需另外核验。
"""

from json import load
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

DEFAULT_START = "2010-01-01"
DEFAULT_END = "2026-10-30"

EASTMONEY_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"

#涵盖 A 股、港股、美股、欧股、债券和黄金
ETF_POOL = {
    "沪深300": "sh510300",
    "中证500": "sh510500",
    "创业板": "sz159915",
    "上证红利": "sh510880",
    "恒生ETF": "sz159920",
    "纳斯达克100": "sh513100",
    "德国DAX": "sh513030",
    "5年期国债": "sh511010",
    "城投债": "sh511220",
    "黄金ETF": "sh518880",
}

LISTING_DATES = {
    "sh510300": "2012-05-28",
    "sh510500": "2013-03-15",
    "sz159915": "2011-12-09",
    "sh510880": "2007-01-18",
    "sz159920": "2012-10-22",
    "sh513100": "2013-05-15",
    "sh513030": "2014-09-05",
    "sh511010": "2013-03-25",
    "sh511220": "2014-12-16",
    "sh518880": "2013-07-29",
}


def _download_eastmoney_daily(ticker, start_date, end_date, adjust):
    """下载一只 ETF 在指定日期区间内的东方财富日线。"""

    # 东方财富用 1 表示沪市，0 表示深市。
    # 例如 sh513030 转成 1.513030，sz159915 转成 0.159915。
    market = "1" if ticker.startswith("sh") else "0"
    adjustment_codes = {"": "0", "qfq": "1", "hfq": "2"}

    params = {
        "secid": f"{market}.{ticker[2:]}",
        "klt": "101",
        "fqt": adjustment_codes[adjust],
        "beg": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "lmt": "6000",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56",
    }
    url = f"{EASTMONEY_URL}?{urlencode(params)}"

    for attempt in range(3):
        try:
            with urlopen(url, timeout=12) as response:
                payload = load(response)

            if payload.get("rc") != 0:
                raise ValueError(f"东方财富接口返回错误码 {payload.get('rc')}")

            data = payload.get("data") or {}
            klines = data.get("klines") or []

            # 每条日线是逗号分隔的字符串，拆成 get_etf() 原来接收的格式。
            return [line.split(",") for line in klines]

        except (OSError, ValueError) as exc:
            if attempt == 2:
                raise RuntimeError(f"{ticker} 的东方财富行情下载失败：{exc}") from exc


def get_etf(
        name_or_ticker: str,
        start: str = DEFAULT_START,
        end: str | None = DEFAULT_END,
        adjust: str = "hfq",
) -> pd.DataFrame:
    """下载单只 ETF，返回开盘、最高、最低、收盘价和成交量。"""
    if adjust not in ("hfq", "qfq", ""):
        raise ValueError('adjust 必须是 "hfq"、"qfq" 或 ""')

    ticker = ETF_POOL.get(name_or_ticker, name_or_ticker)
    if ticker not in LISTING_DATES:
        raise ValueError(f"固定资产池中没有 {name_or_ticker}")

    start_date = pd.Timestamp(start).normalize()
    if end is None:
        end_date = (
            pd.Timestamp.now(tz="Asia/Shanghai")
            .tz_localize(None)
            .normalize()
        )
    else:
        end_date = pd.Timestamp(end).normalize()

    if pd.isna(start_date) or pd.isna(end_date) or start_date > end_date:
        raise ValueError("请检查开始和结束日期")

    # 关键：每只 ETF 单独确定起点。
    listing_date = pd.Timestamp(LISTING_DATES[ticker])
    effective_start = max(start_date, listing_date)

    if effective_start > end_date:
        raise ValueError(f"{name_or_ticker} 在指定区间内尚未上市")

    rows = _download_eastmoney_daily(
        ticker, effective_start, end_date, adjust
    )

    # 东方财富字段顺序：日期、开盘、收盘、最高、最低、成交量。
    df = pd.DataFrame(
        [row[:6] for row in rows],
        columns=["Date", "Open", "Close", "High", "Low", "Volume"],
    )

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    if df.index.has_duplicates:
        raise ValueError(f"{name_or_ticker} 存在重复交易日期")

    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    price_columns = ["Open", "High", "Low", "Close"]
    if (
            np.isinf(df.to_numpy(dtype=float)).any()
            or (df[price_columns] <= 0).any().any()
            or (df["Volume"] < 0).any()
    ):
        raise ValueError(f"{name_or_ticker} 存在无效价格或成交量")

    df.attrs["source"] = "东方财富 newfqkline"
    df.attrs["ticker"] = ticker
    df.attrs["adjust_requested"] = adjust
    df.attrs["adjust_used"] = adjust

    return df


def get_etfs(
        names: list,
        start: str = DEFAULT_START,
        end: str | None = DEFAULT_END,
        column: str = "Close",
        adjust: str = "hfq",
) -> pd.DataFrame:
    """按日期并集合并，保留每只 ETF 自己的历史和缺失值。"""
    if isinstance(names, str):
        raise TypeError('names 应是列表，例如 ["沪深300", "黄金ETF"]')

    names = list(names)
    if not names:
        raise ValueError("至少需要一只 ETF")

    series_list = []
    adjust_used = {}

    for name in names:
        print(f"正在下载：{name}")

        df = get_etf(name, start=start, end=end, adjust=adjust)
        series = df[column].rename(name)

        first_valid = series.first_valid_index()
        last_valid = series.last_valid_index()
        if first_valid is None:
            raise ValueError(f"{name} 的 {column} 列没有有效数据")

        series_list.append(series)
        adjust_used[name] = df.attrs["adjust_used"]
        price_type = adjust_used[name] or "普通日线"

        print(
            f"  {first_valid:%Y-%m-%d} 至 {last_valid:%Y-%m-%d}，"
            f"{series.count()} 条有效记录，{price_type}"
        )

    # outer 取日期并集。某只 ETF 当天没有报价，就保留为空。
    prices = pd.concat(series_list, axis=1, join="outer", sort=True)
    prices.index.name = "Date"
    prices.attrs["source"] = "腾讯 newfqkline"
    prices.attrs["adjust_requested"] = adjust
    prices.attrs["adjust_used"] = adjust_used

    return prices


def list_etfs() -> dict:
    """返回固定资产池的副本。"""
    return dict(ETF_POOL)


if __name__ == "__main__":
    prices = get_etfs(list(ETF_POOL))

    output_path = Path(__file__).resolve().parent / "etf_prices.csv"
    prices.to_csv(output_path, encoding="utf-8-sig", na_rep="")

    print(f"\n已保存：{output_path}")
    print(f"共 {len(prices)} 个日期，{len(prices.columns)} 只 ETF")