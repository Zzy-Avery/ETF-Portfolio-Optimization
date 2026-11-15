"""
ETF 历史行情及权益记录获取。

数据来源：腾讯财经历史日线接口、天天基金分红及份额折算页面。
输出：etf_prices.csv（不复权收盘价）、etf_actions.csv（收益计算事件）、
etf_total_return.csv（总回报指数）。原始权益记录在内存中解析，不另行导出。
总回报假设：现金分红按场内除息日收盘价无成本再投资，允许小数份额，
不模拟红利到账延迟、税费及份额折算的整数取整。每只资产首个有效日为100。
价格查询包含起止日期，各资产历史起点不早于上市日期。
权益记录保留来源中的完整历史，可能包含价格样本期外事件。
已知事件日期和折算精度差异依据公告修正，来源保存在事件表中。
未修正事件采用数据源披露，不据此声称全部历史事件均经过独立公告核验。
缺报价日的原始价格与 TRI 均留空；相邻有效报价间的回报计入下一有效报价日，
跨缺口的回报是区间回报，不能直接视为完整的日频样本或自动认定为停牌。
依赖：numpy、pandas、lxml。
"""

from json import load
from time import sleep
from io import StringIO
from lxml import html as lxml_html
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import re

DEFAULT_START = "2010-01-01"
DEFAULT_END = "2026-10-30"

TENCENT_URL = "https://web.ifzq.gtimg.cn/appstock/app/newfqkline/get"
# 权益事件表的统一列名。
# 现金分红和拆分比例均以事件发生前的一份 ETF 份额为基准。
ACTION_COLUMNS = [
    "ETF",
    "Date",
    "CashDividend",
    "SplitRatio",
    "Source",
]

# 涵盖 A 股、港股、美股、欧股、债券和黄金
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


def _download_tencent_daily(ticker, start_date, end_date, adjust):
    """按两年区间分段获取腾讯日线，避免单次条数上限截断历史。"""
    rows = []
    key = f"{adjust}day" if adjust else "day"

    for year in range(start_date.year, end_date.year + 1, 2):
        period_start = max(start_date, pd.Timestamp(year, 1, 1))
        period_end = min(end_date, pd.Timestamp(year + 1, 12, 31))
        # 两个自然年的交易日数量低于接口每次 640 条的上限。
        params = {
            "param": (
                f"{ticker},day,{period_start:%Y-%m-%d},"
                f"{period_end:%Y-%m-%d},640,{adjust}"
            ),
        }
        # 两个入口均获取腾讯日线；发生连接或服务错误时切换入口重试。
        endpoints = (
            TENCENT_URL,
            "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
        )
        for attempt in range(4):
            try:
                endpoint = endpoints[attempt % len(endpoints)]
                request = Request(
                    f"{endpoint}?{urlencode(params)}",
                    headers={
                        "User-Agent": "Mozilla/5.0",
                        "Referer": "https://gu.qq.com/",
                    },
                )
                with urlopen(request, timeout=15) as response:
                    payload = load(response)
                if payload.get("code") != 0:
                    raise ValueError(f"腾讯接口错误：{payload.get('msg', payload.get('code'))}")
                data = (payload.get("data") or {}).get(ticker) or {}
                # 不以复权日线替代不复权日线，确保价格口径明确。
                if key not in data:
                    raise ValueError(f"响应缺少 {key} 行情字段")
                period_rows = [
                    row[:6] for row in data[key]
                    if period_start <= pd.Timestamp(row[0]) <= period_end
                ]
                # 长区间无报价视为数据异常；短区间可能完全处于休市期。
                if not period_rows and (period_end - period_start).days >= 30:
                    raise ValueError(f"{year} 年指定区间未返回行情")
                rows.extend(period_rows)
                break
            except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
                if attempt == 3:
                    raise RuntimeError(
                        f"{ticker} 腾讯行情下载失败（{year}）：{exc}"
                    ) from exc
                print(f"  {ticker} {year} 年请求失败（第 {attempt + 1}/4 次）：{exc}")
                sleep(attempt + 1)

        # 分段请求之间限速。
        sleep(0.5)

    if not rows:
        raise ValueError(f"{ticker} 指定区间未返回历史行情")
    return rows


def get_etf(
        name_or_ticker: str,
        start: str = DEFAULT_START,
        end: str | None = DEFAULT_END,
        adjust: str = "",
) -> pd.DataFrame:
    """
    下载单只 ETF 的历史行情，返回开盘、最高、最低、收盘价和成交量。

    默认使用不复权价格，保留各交易日的实际价格口径。
    总回报计算需结合经核验的现金分红及份额折算记录。
    """

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

    # 按各资产上市日期独立确定历史起点。
    listing_date = pd.Timestamp(LISTING_DATES[ticker])
    effective_start = max(start_date, listing_date)

    if effective_start > end_date:
        raise ValueError(f"{name_or_ticker} 在指定区间内尚未上市")

    rows = _download_tencent_daily(
        ticker, effective_start, end_date, adjust
    )

    # 腾讯日线字段顺序：日期、开盘、收盘、最高、最低、成交量（手）。
    df = pd.DataFrame(
        [row[:6] for row in rows],
        columns=["Date", "Open", "Close", "High", "Low", "Volume"],
    )

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    if df.empty or df.index.hasnans:
        raise ValueError(f"{name_or_ticker} 行情为空或存在无效日期")

    if df.index.has_duplicates:
        raise ValueError(f"{name_or_ticker} 存在重复交易日期")

    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    price_columns = ["Open", "High", "Low", "Close"]
    if (
            df.isna().any().any()
            or np.isinf(df.to_numpy(dtype=float)).any()
            or (df[price_columns] <= 0).any().any()
            or (df["Volume"] < 0).any()
    ):
        raise ValueError(f"{name_or_ticker} 存在无效价格或成交量")

    df.attrs["source"] = "腾讯财经历史日线 newfqkline/fqkline"
    df.attrs["ticker"] = ticker
    df.attrs["adjust_requested"] = adjust
    df.attrs["adjust_used"] = adjust

    return df


def get_etfs(
        names: list,
        start: str = DEFAULT_START,
        end: str | None = DEFAULT_END,
        column: str = "Close",
        adjust: str = "",
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
        price_type = adjust_used[name] or "不复权日线"

        print(
            f"  {first_valid:%Y-%m-%d} 至 {last_valid:%Y-%m-%d}，"
            f"{series.count()} 条有效记录，{price_type}"
        )

    # outer 取日期并集。某只 ETF 当天没有报价，就保留为空。
    prices = pd.concat(series_list, axis=1, join="outer", sort=True)
    prices.index.name = "Date"
    prices.attrs["source"] = "腾讯财经历史日线 newfqkline/fqkline"
    prices.attrs["adjust_requested"] = adjust
    prices.attrs["adjust_used"] = adjust_used

    return prices


def get_action_tables(name_or_ticker: str) -> dict:
    """
    读取单只 ETF 的分红、拆分原始表格。

    保留来源字段；事件日期及折算比例需与场内公告核验。
    """
    # 既支持 ETF 名称，也支持带市场前缀的代码。
    ticker = ETF_POOL.get(name_or_ticker, name_or_ticker)

    if ticker not in LISTING_DATES:
        raise ValueError(f"固定资产池中没有 {name_or_ticker}")

    # 例如 sh510300 转为网页需要的六位基金代码 510300。
    code = ticker[2:]
    url = f"https://fundf10.eastmoney.com/fhsp_{code}.html?code={code}"

    # 添加浏览器标识，构造网页请求。
    request = Request(
        url, headers={"User-Agent": "Mozilla/5.0", "Cache-Control": "no-cache"}
    )

    with urlopen(request, timeout=15) as response:
        html = response.read().decode("utf-8")

    # 校验页面基金身份，并限定到当前基金的权益详情区块。
    # 不扫描导航、推荐基金或其他基金查询结果中的同名字段。
    document = lxml_html.fromstring(html)
    title = document.findtext(".//title") or ""
    if not re.search(rf"[（(]\s*{re.escape(code)}\s*[）)]", title):
        raise ValueError(f"权益页面基金身份不符：请求 {code}，标题 {title}")
    action_tables = {}
    sections = {
        "dividends": ("分红送配详情", {"权益登记日", "除息日", "分红发放日"}),
        "splits": ("拆分详情", {"拆分折算日", "拆分类型", "拆分折算比例"}),
    }
    for key, (label, required_columns) in sections.items():
        nodes = document.xpath(
            "//div[contains(concat(' ', normalize-space(@class), ' '), ' boxitem ')]"
            "[h4/label[normalize-space(.)=$label]]/table",
            label=label,
        )
        if len(nodes) != 1:
            raise ValueError(f"{name_or_ticker} 的 {label} 区块无法唯一识别")
        section_html = lxml_html.tostring(nodes[0], encoding="unicode")
        tables = pd.read_html(StringIO(section_html))
        if len(tables) != 1 or not required_columns.issubset(tables[0].columns):
            raise ValueError(f"{name_or_ticker} 的 {label} 字段不完整")
        action_tables[key] = tables[0].copy()

    return action_tables


def parse_cash_dividend(text: str) -> float:
    """把“每N份派现金X元”转换为每份 ETF 对应的现金分红。"""

    # 去掉空白，避免文字中的空格或换行影响识别。
    text = re.sub(r"\s+", "", str(text))

    # 分别提取份数和现金金额；“每份”表示每1份。
    match = re.fullmatch(
        r"每(\d+(?:\.\d+)?)?份派现金(\d+(?:\.\d+)?)元",
        text,
    )

    # 遇到不认识的格式就报错，不能直接当作零分红。
    if match is None:
        raise ValueError(f"无法识别分红金额：{text}")

    # 第一组是份数；省略数字的“每份”按1份处理。
    units = float(match.group(1) or "1")

    # 第二组是这些份额合计获得的现金，单位为元。
    cash = float(match.group(2))

    if units <= 0:
        raise ValueError(f"分红对应的份数必须大于零：{text}")

    return cash / units


def parse_split_ratio(text: str) -> float:
    """把“旧份额:新份额”转换为每份旧份额对应的新份额数量。"""

    # 去掉空白，并把中文冒号统一为英文冒号。
    text = re.sub(r"\s+", "", str(text)).replace("：", ":")

    # 分别提取冒号前后的份额数量。
    match = re.fullmatch(
        r"(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)",
        text,
    )

    # 无法识别时直接报错，不能默认当作没有拆分。
    if match is None:
        raise ValueError(f"无法识别拆分比例：{text}")

    old_units = float(match.group(1))
    new_units = float(match.group(2))

    if old_units <= 0 or new_units <= 0:
        raise ValueError(f"拆分前后份额必须大于零：{text}")

    return new_units / old_units


def get_dividend_candidates(
    name_or_ticker: str, action_tables: dict | None = None,
) -> pd.DataFrame:
    """读取并整理分红记录，日期尚需核对场内公告。"""

    # 把名称或代码统一转换为资产池中的 ETF 名称。
    ticker = ETF_POOL.get(name_or_ticker, name_or_ticker)
    name = next(
        (name for name, code in ETF_POOL.items() if code == ticker),
        None,
    )
    if name is None:
        raise ValueError(f"固定资产池中没有 {name_or_ticker}")

    # 获取原始分红表。
    if action_tables is None:
        action_tables = get_action_tables(name)
    table = action_tables["dividends"]

    columns = ["ETF", "ReportedExDate", "CashDividend", "Source"]

    # 页面明确写着暂无分红，才返回没有记录的表。
    if len(table) == 1:
        row = table.iloc[0].dropna().astype(str)
        if not row.empty and row.str.contains("暂无分红信息", regex=False).all():
            return pd.DataFrame(columns=columns)

    # 按列名寻找金额列，兼容“每份分红”“每10份分红”等。
    amount_columns = [
        column for column in table.columns
        if re.fullmatch(r"每(?:\d+)?份分红", str(column))
    ]
    if table.empty or len(amount_columns) != 1:
        raise ValueError(f"{name} 的分红金额列无法确定")

    result = pd.DataFrame(index=table.index)
    result["ETF"] = name

    # 先保留网页提供的日期，不直接认定它就是场内除息日。
    result["ReportedExDate"] = pd.to_datetime(
        table["除息日"], errors="raise"
    )
    if result["ReportedExDate"].isna().any():
        raise ValueError(f"{name} 存在缺失的除息日")

    # 将现金分红统一为元/份。
    result["CashDividend"] = table[amount_columns[0]].map(
        parse_cash_dividend
    )

    result["Source"] = (
        f"https://fundf10.eastmoney.com/fhsp_{ticker[2:]}.html"
    )

    return result[columns].sort_values(
        "ReportedExDate"
    ).reset_index(drop=True)



def get_split_candidates(
    name_or_ticker: str, action_tables: dict | None = None,
) -> pd.DataFrame:
    """读取并整理拆分记录，日期和比例方向尚需核对公告。"""

    # 把名称或代码统一转换为资产池中的 ETF 名称。
    ticker = ETF_POOL.get(name_or_ticker, name_or_ticker)
    name = next(
        (name for name, code in ETF_POOL.items() if code == ticker),
        None,
    )
    if name is None:
        raise ValueError(f"固定资产池中没有 {name_or_ticker}")

    # 取出已经识别出的原始拆分表。
    if action_tables is None:
        action_tables = get_action_tables(name)
    table = action_tables["splits"]

    columns = ["ETF", "ReportedSplitDate", "SplitRatio", "Source"]

    # 页面明确写着暂无拆分，才返回没有记录的表。
    if len(table) == 1:
        row = table.iloc[0].dropna().astype(str)
        if not row.empty and row.str.contains("暂无拆分信息", regex=False).all():
            return pd.DataFrame(columns=columns)

    # 空表不能直接理解为没有拆分，需要检查数据来源。
    if table.empty:
        raise ValueError(f"{name} 的拆分表为空，无法确认拆分记录")

    result = pd.DataFrame(index=table.index)
    result["ETF"] = name

    # 保留网页提供的折算日期，之后核对实际生效时间。
    result["ReportedSplitDate"] = pd.to_datetime(
        table["拆分折算日"], errors="raise"
    )
    if result["ReportedSplitDate"].isna().any():
        raise ValueError(f"{name} 存在缺失的拆分日期")

    # 将“1:5”等文字转换为每份旧份额对应的新份额数量。
    result["SplitRatio"] = table["拆分折算比例"].map(
        parse_split_ratio
    )

    result["Source"] = (
        f"https://fundf10.eastmoney.com/fhsp_{ticker[2:]}.html"
    )

    return result[columns].sort_values(
        "ReportedSplitDate"
    ).reset_index(drop=True)


def get_all_action_candidates() -> tuple:
    """批量读取固定资产池的分红、拆分记录，尚未完成公告核验。"""

    # 两个列表分别保存各只 ETF 的分红表和拆分表。
    dividend_tables = []
    split_tables = []

    for name in ETF_POOL:
        print(f"\n正在读取权益记录：{name}")

        # 同一资产的分红和折算使用同一页面响应，避免重复请求。
        tables = get_action_tables(name)
        dividends = get_dividend_candidates(name, action_tables=tables)
        splits = get_split_candidates(name, action_tables=tables)

        dividend_tables.append(dividends)
        split_tables.append(splits)

        print(
            f"  读取到分红 {len(dividends)} 条，"
            f"拆分或折算 {len(splits)} 条"
        )

    # 把各只 ETF 的表格纵向拼接，重新生成连续行号。
    # 排除空表后合并，避免不同 pandas 版本对空列类型的推断差异。
    nonempty_dividends = [table for table in dividend_tables if not table.empty]
    nonempty_splits = [table for table in split_tables if not table.empty]
    all_dividends = (
        pd.concat(nonempty_dividends, ignore_index=True)
        if nonempty_dividends else dividend_tables[0].copy()
    )
    all_splits = (
        pd.concat(nonempty_splits, ignore_index=True)
        if nonempty_splits else split_tables[0].copy()
    )

    return all_dividends, all_splits


def prepare_actions(
    prices: pd.DataFrame,
    dividends: pd.DataFrame,
    splits: pd.DataFrame,
) -> pd.DataFrame:
    """
    整理收益计算事件，修正已知场内日期及折算精度差异。

    Date 为价格计算生效日，ReportedDate 为网页原始日期。
    停牌期间的份额折算在下一有效报价应用；现金分红必须有当日价格，
    不对缺失价格估计再投资份额。样本首日及之前、样本末日之后事件不计入。
    """
    # 日期、比例及依据均显式保留，避免依赖网页上的舍入值或登记日期。
    dividend_overrides = {
        ("城投债", "2015-06-18"): (
            "2015-06-19",
            "https://epaper.stcn.com/paper/zqsb/html/2015-06/15/content_700914.htm",
        ),
        ("城投债", "2015-09-21"): (
            "2015-09-22",
            "https://www.hftfund.com/contents/2015/9/15-045676a4d18c4622862ee699b8865b0e.html",
        ),
    }
    split_overrides = {
        ("中证500", "2015-04-14"): (
            "2015-04-14", 0.28032483,
            "https://www.sse.com.cn/disclosure/fund/announcement/c/2015-04-15/510500_20150415_2.pdf",
        ),
        # 2022-08-26 日终拆分，下一交易日价格采用新份额单位。
        ("中证500", "2022-08-26"): (
            "2022-08-29", 1.14539,
            "https://notice.10jqka.com.cn/api/pdf/3b6147102b640024.pdf",
        ),
        ("纳斯达克100", "2022-01-13"): (
            "2022-01-13", 5.0,
            "https://static.cninfo.com.cn/finalpage/2022-01-04/1212087910.PDF",
        ),
        ("城投债", "2023-01-13"): (
            "2023-01-16", 10.0,
            "https://www.hftfund.com/contents/2023/1/10-24d23c1abc4341348a561d4d43641659.html",
        ),
    }
    columns = ACTION_COLUMNS + ["ReportedDate", "EventType", "AdjustmentSource"]
    records = []
    for event_type, table, date_column in (
        ("dividend", dividends, "ReportedExDate"),
        ("split", splits, "ReportedSplitDate"),
    ):
        if table.duplicated(["ETF", date_column]).any():
            raise ValueError(f"存在重复的 {event_type} 记录")
        for _, row in table.iterrows():
            name = row["ETF"]
            if name not in prices.columns:
                raise ValueError(f"权益记录包含价格表外的资产：{name}")
            reported_date = pd.Timestamp(row[date_column]).normalize()
            if pd.isna(reported_date):
                raise ValueError(f"{name} 存在无效事件日期")
            date = reported_date
            cash, ratio = 0.0, 1.0
            adjustment_source = ""
            key = (name, f"{reported_date:%Y-%m-%d}")
            if event_type == "dividend":
                cash = float(row["CashDividend"])
                if key in dividend_overrides:
                    date_text, adjustment_source = dividend_overrides[key]
                    date = pd.Timestamp(date_text)
            else:
                ratio = float(row["SplitRatio"])
                if key in split_overrides:
                    date_text, ratio, adjustment_source = split_overrides[key]
                    date = pd.Timestamp(date_text)
            if not np.isfinite([cash, ratio]).all() or cash < 0 or ratio <= 0:
                raise ValueError(f"{name} {reported_date:%Y-%m-%d} 权益数值无效")

            valid_dates = prices[name].dropna().index
            if valid_dates.empty:
                raise ValueError(f"{name} 无有效报价")
            # 初始建仓前已发生的权益不属于样本期持有收益。
            if date <= valid_dates[0] or date > valid_dates[-1]:
                continue
            if date not in valid_dates:
                if event_type == "dividend":
                    raise ValueError(f"{name} {date:%Y-%m-%d} 除息日缺少再投资价格")
                date = valid_dates[valid_dates.searchsorted(date)]

            records.append({
                "ETF": name, "Date": date,
                "CashDividend": cash, "SplitRatio": ratio,
                "Source": row["Source"], "ReportedDate": reported_date,
                "EventType": event_type, "AdjustmentSource": adjustment_source,
            })

    actions = pd.DataFrame(records, columns=columns)
    # 同一价格日的多个事件涉及现金归属单位，需单独核验后再合并。
    if actions.duplicated(["ETF", "Date"]).any():
        raise ValueError("同一资产同日存在多个事件，需核对现金基准份额后合并")
    return actions.sort_values(["ETF", "Date"]).reset_index(drop=True)


def calculate_total_return_index(
    prices: pd.DataFrame,
    actions: pd.DataFrame,
    initial_value: float = 100.0,
) -> pd.DataFrame:
    """
    从不复权收盘价构建现金分红再投资总回报指数。

    相邻有效报价间增长因子为 (s_t * P_t + D_t) / P_prev，
    s_t 为每份旧份额对应的新份额数，D_t 为每份旧份额的现金红利。
    无事件时 s_t=1、D_t=0。缺报价日留空，不前向填补原始价格。
    """
    if not np.isfinite(initial_value) or initial_value <= 0:
        raise ValueError("指数初始值必须为有限正数")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("价格索引必须为 DatetimeIndex")
    if prices.empty or prices.index.hasnans or prices.index.has_duplicates:
        raise ValueError("价格表为空或存在无效、重复日期")
    if prices.columns.has_duplicates:
        raise ValueError("价格表存在重复资产列")
    prices = prices.sort_index()
    required = set(ACTION_COLUMNS)
    if not required.issubset(actions.columns):
        raise ValueError("事件表缺少必要字段")
    if not set(actions["ETF"]).issubset(prices.columns):
        raise ValueError("事件表包含价格表外的资产")
    actions = actions.copy()
    actions["Date"] = pd.to_datetime(actions["Date"], errors="raise")
    if actions["Date"].isna().any() or actions.duplicated(["ETF", "Date"]).any():
        raise ValueError("事件表存在无效日期或重复事件")
    tri = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    tri.index.name = "Date"
    for name in prices.columns:
        price = prices[name].dropna().astype(float)
        if price.empty or not np.isfinite(price).all() or (price <= 0).any():
            raise ValueError(f"{name} 的有效价格必须为有限正数")
        cash = pd.Series(0.0, index=price.index)
        ratio = pd.Series(1.0, index=price.index)
        events = actions.loc[actions["ETF"].eq(name)]
        for _, row in events.iterrows():
            date = row["Date"]
            if date <= price.index[0] or date > price.index[-1]:
                continue
            if date not in price.index:
                raise ValueError(f"{name} {date:%Y-%m-%d} 事件缺少对应价格")
            d, split = float(row["CashDividend"]), float(row["SplitRatio"])
            if not np.isfinite([d, split]).all() or d < 0 or split <= 0:
                raise ValueError(f"{name} 权益数值无效")
            cash.loc[date], ratio.loc[date] = d, split
        factors = (ratio * price + cash) / price.shift(1)
        factors.iloc[0] = 1.0
        values = initial_value * factors.cumprod()
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"{name} 总回报指数计算出现无效数值")
        tri.loc[price.index, name] = values
    tri.attrs["method"] = "除息日收盘价无成本再投资，允许小数份额，缺报价不填补"
    return tri


def list_etfs() -> dict:
    """返回固定资产池的副本。"""
    return dict(ETF_POOL)


if __name__ == "__main__":
    # 获取不复权收盘价及权益候选记录。
    prices = get_etfs(list(ETF_POOL), adjust="")
    dividends, splits = get_all_action_candidates()
    actions = prepare_actions(prices, dividends, splits)
    total_return = calculate_total_return_index(prices, actions)

    # 全部获取及计算成功后写入文件；下载或计算失败时保留现有 CSV。
    project_dir = Path(__file__).resolve().parent
    output_path = project_dir / "etf_prices.csv"
    action_path = project_dir / "etf_actions.csv"
    tri_path = project_dir / "etf_total_return.csv"

    # 行情日期保留为索引；事件表不额外写入行号。
    prices.to_csv(output_path, encoding="utf-8-sig", na_rep="")
    actions.to_csv(action_path, index=False, encoding="utf-8-sig")
    total_return.to_csv(tri_path, encoding="utf-8-sig", na_rep="")

    print(f"\n行情文件：{output_path}")
    print(f"共 {len(prices)} 个日期，{len(prices.columns)} 只 ETF")
    print("价格口径：不复权收盘价。")
    print(f"收益计算事件：{action_path}（{len(actions)} 条）")
    print(f"总回报指数：{tri_path}（各资产首个有效日为100）")
    print("总回报口径：分红除息日收盘价无成本再投资，计入份额折算。")
    print("缺失报价及对应指数保留为空；跨缺口收益为区间收益。")
