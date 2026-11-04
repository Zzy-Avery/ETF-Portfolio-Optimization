"""
数据获取模块
============
通过 AKShare 调用腾讯接口获取 ETF 历史日线行情。

用法：
    df = get_etf("沪深300")
    prices = get_etfs(["沪深300", "中证红利", "黄金ETF"])

多只 ETF 合并规则：
1. 保留各只 ETF 返回日期的并集。
2. 从各只 ETF 首个有效值中的最晚日期开始。
3. 缺失值使用此前最近的有效值前向填补（ffill）。
4. 下载失败或填补后仍有缺失时停止。

注意：
1. 默认请求后复权行情，adjust="hfq"。
2. start 和 end 均包含当天，end=None 表示查询至当天。
3. 复权是否正确仍需另外核验。
"""

from pathlib import Path
import akshare as ak
import numpy as np
import pandas as pd


# ==================== 常用 ETF 池 ====================
# 腾讯代码：sh 表示上海，sz 表示深圳
ETF_POOL = {
    # 红利类
    "中证红利": "sh515080",
    "红利低波": "sh512890",
    "红利ETF": "sh510880",

    # 债券类
    "国债ETF": "sh511010",
    "十年国债": "sh511260",

    # 商品
    "黄金ETF": "sh518880",

    # 宽基指数
    "沪深300": "sh510300",
    "中证500": "sh510500",
    "上证50": "sh510050",
    "创业板": "sz159915",

    # 行业
    "医药ETF": "sh512010",
    "消费ETF": "sz159928",
    "银行ETF": "sh512800",

    # 跨境
    "恒生ETF": "sz159920",
}


def _resolve_ticker(name: str) -> str:
    """根据ETF名称，从固定池中获取腾讯代码。"""
    if name not in ETF_POOL:
        raise ValueError(f"ETF池中没有：{name}")

    return ETF_POOL[name]


def _parse_date(value: str, label: str) -> pd.Timestamp:
    """转换并检查日期。"""
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}不是有效日期：{value}") from exc

    if pd.isna(result):
        raise ValueError(f"{label}不能为空日期")

    if result.tzinfo is not None:
        result = result.tz_localize(None)

    return result.normalize()


def get_etf(
    name_or_ticker: str,
    start: str = "2022-01-01",
    end: str = "2026-10-30",
    adjust: str = "hfq",
) -> pd.DataFrame:
    """
    获取单只 ETF 历史行情。

    参数：
        name_or_ticker:
            ETF_POOL中的中文简称，例如“沪深300”。

        start:
            开始日期，包含当天。

        end:
            结束日期，包含当天。
            None 表示查询至当天。

        adjust:
            "hfq"：请求后复权，默认
            "qfq"：请求前复权
            ""：请求不复权

    返回：
        以日期为索引的 DataFrame。

        基本列：
            Open、High、Low、Close、Volume

        接口提供时保留：
            Turnover、Amount

    本函数保留原始缺失值；
    多只 ETF 合并时由 get_etfs() 执行前向填补。
    """
    if adjust not in ("hfq", "qfq", ""):
        raise ValueError('adjust只能是 "hfq"、"qfq" 或 ""')

    ticker = _resolve_ticker(name_or_ticker)

    start_date = _parse_date(start, "start")

    if end is None:
        end_date = (
            pd.Timestamp.now(tz="Asia/Shanghai")
            .tz_localize(None)
            .normalize()
        )
    else:
        end_date = _parse_date(end, "end")

    if start_date > end_date:
        raise ValueError("start不能晚于end")

    # 调用腾讯历史行情接口
    raw = ak.stock_zh_a_hist_tx(
        symbol=ticker,
        start_date=start_date.strftime("%Y%m%d"),
        end_date=end_date.strftime("%Y%m%d"),
        adjust=adjust,
    )

    if raw is None or raw.empty:
        raise ValueError(
            f"无法获取 {name_or_ticker}（{ticker}）的数据，"
            "请检查代码、日期范围或接口支持情况"
        )

    df = raw.copy()

    # 检查必要字段
    required = {"date", "open", "high", "low", "close", "volume"}
    missing_columns = required.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"{ticker} 缺少必要字段：{sorted(missing_columns)}"
        )

    # 整理日期索引
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df["date"].isna().any():
        raise ValueError(f"{ticker} 存在无效日期")

    df = df.set_index("date").sort_index()
    df.index.name = "Date"

    if df.index.has_duplicates:
        raise ValueError(f"{ticker} 存在重复日期，需要检查")

    # 再次限制日期范围
    df = df.loc[start_date:end_date].copy()

    if df.empty:
        raise ValueError(f"{ticker} 在指定日期范围内没有数据")

    # 保留原模块使用的大写列名
    df = df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "turnover": "Turnover",
            "amount": "Amount",
        }
    )

    ordered_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Turnover",
        "Amount",
    ]

    df = df[
        [column for column in ordered_columns if column in df.columns]
    ].copy()

    # 转换为数字，无法转换的内容记为缺失值
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # 检查无穷值
    if np.isinf(df.to_numpy(dtype=float)).any():
        raise ValueError(f"{ticker} 存在正无穷或负无穷数据")

    # 非缺失价格必须为正
    price_columns = ["Open", "High", "Low", "Close"]

    if (df[price_columns] <= 0).any().any():
        raise ValueError(f"{ticker} 存在非正价格，需要检查")

    # 成交量、成交额和换手率不能为负
    for column in ("Volume", "Amount", "Turnover"):
        if column in df.columns and (df[column] < 0).any():
            raise ValueError(f"{ticker} 的 {column} 存在负值")

    # 记录来源与请求口径
    df.attrs["source"] = "腾讯 / AKShare"
    df.attrs["ticker"] = ticker
    df.attrs["adjust_requested"] = adjust

    return df


def get_etfs(
    names: list,
    start: str = "2022-01-01",
    end: str = "2026-10-30",
    column: str = "Close",
    adjust: str = "hfq",
) -> pd.DataFrame:
    """
    批量获取多只 ETF 的指定字段。

    默认获取后复权收盘价。

    合并规则：
    1. 保留各只 ETF 日期的并集。
    2. 从各列首个有效值中的最晚日期开始。
    3. 缺失值使用此前最近的有效值前向填补。
    4. 不使用未来数据向后填补。
    5. 下载失败或填补后仍有缺失时停止。

    返回：
        每列为一只 ETF，索引为日期的 DataFrame。
    """
    if isinstance(names, str):
        raise TypeError(
            'names应为列表，例如 ["沪深300", "黄金ETF"]'
        )

    names = list(names)

    if not names:
        raise ValueError("names不能为空")

    series_list = []
    failed = []

    for name in names:
        print(f"正在获取：{name}")

        try:
            df = get_etf(
                name,
                start=start,
                end=end,
                adjust=adjust,
            )

            if column not in df.columns:
                raise ValueError(f"接口未返回 {column} 列")

            series = df[column].copy()
            series.name = name

            if series.first_valid_index() is None:
                raise ValueError(f"{column} 列全部缺失")

            series_list.append(series)

            print(
                f"  获取成功：{len(df)}行，"
                f"{df.index.min():%Y-%m-%d} 至 "
                f"{df.index.max():%Y-%m-%d}"
            )

        except Exception as exc:
            failed.append(f"{name}：{exc}")

    # 有任意 ETF 下载失败，就停止
    if failed:
        raise ValueError(
            "以下ETF获取失败，停止合并：\n"
            + "\n".join(failed)
        )

    # 取各只ETF首个有效值日期中的最晚日期，作为共同分析起点
    common_start = max(
        series.first_valid_index()
        for series in series_list
    )

    # 按日期并集合并
    result = pd.concat(
        series_list,
        axis=1,
        join="outer",
    ).sort_index()

    # 先填补再截取，保留共同起点之前可用于填补的历史值
    missing_before_fill = result.isna()

    # 只统计最终保留区间中的缺失记录
    missing_counts = missing_before_fill.loc[common_start:].sum()

    if missing_counts.any():
        print("\n各只ETF需要前向填补的记录数：")
        print(missing_counts)

    # 用此前最近的有效值填补
    result = result.ffill()

    # 截掉共同起点之前的数据
    result = result.loc[common_start:].copy()
    result.index.name = "Date"

    if result.empty:
        raise ValueError("没有可合并的数据")

    # 填补后仍有缺失则停止
    if result.isna().any().any():
        raise ValueError("前向填补后仍有缺失值，请检查数据")

    result.attrs["source"] = "腾讯 / AKShare"
    result.attrs["adjust_requested"] = adjust
    result.attrs["fill_method"] = "ffill"

    return result


def list_etfs() -> dict:
    """返回内置 ETF 代码映射的副本。"""
    return dict(ETF_POOL)


if __name__ == "__main__":
    # 使用 ETF_POOL 中的全部 ETF
    names = list(ETF_POOL.keys())

    print(f"本次获取全部 {len(names)} 只ETF：")
    for name, ticker in ETF_POOL.items():
        print(f"  {name}：{ticker}")

    # 获取后复权收盘价，并自动前向填补缺失值
    prices = get_etfs(
        names=names,
        start="2022-01-01",
        end="2026-10-30",
        column="Close",
        adjust="hfq",
    )

    # 保存到当前Python文件所在的文件夹
    output_path = Path(__file__).resolve().parent / "etf_prices.csv"

    prices.to_csv(
        output_path,
        index=True,
        index_label="Date",
        encoding="utf-8-sig",
    )

    print(f"\n文件已保存到：{output_path}")
    print(f"共 {len(prices)} 行，{len(prices.columns)} 只ETF")

    # 在控制台完整显示价格表
    print(prices.to_string())