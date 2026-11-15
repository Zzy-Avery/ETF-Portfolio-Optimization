"""
基于 ETF 总回报指数的协方差估计与收益约束扩展实验。

比较等权、共同样本最小方差、共同样本收益约束最小方差、
LW 最小方差和 LW 收益约束最小方差五种组合。

所有优化组合采用相同的历史完整共同样本。
两类收益约束使用相同的均值和等权估计收益门槛。
收益约束仅作用于历史估计值，不保证未来实现收益。

已确认停牌日仅在持仓估值副本中沿用前一 TRI，
不填补参数估计输入。
"""

import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf
from pathlib import Path

from optimization import (
    calculate_returns,
    equal_weight,
    min_variance,
    min_variance_equal_return,
)

from backtest import (
    make_rebalance_schedule,
    run_backtest,
    calculate_performance,
)


# 1. 估计 Ledoit–Wolf 协方差矩阵
def estimate_lw_covariance(
    returns: pd.DataFrame,
    end_date: str,
    start_date: str | None = None,
) -> dict:
    """使用完整共同样本估计均值、样本协方差和 LW 协方差。"""

    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("收益率表必须使用日期索引")
    if returns.index.hasnans or returns.index.has_duplicates or returns.columns.has_duplicates:
        raise ValueError("收益率表存在无效日期、重复日期或重复资产")
    start = pd.Timestamp(start_date) if start_date is not None else None
    end = pd.Timestamp(end_date)
    if pd.isna(end) or (start is not None and (pd.isna(start) or start > end)):
        raise ValueError("历史窗口起止日期无效")
    # 截取截至信号日的历史收益率，不使用未来数据。
    window = returns.sort_index().loc[start:end].copy()
    if window.shape[1] == 0 or np.isinf(window.to_numpy(dtype=float)).any():
        raise ValueError("收益率窗口无资产或存在无穷值")

    # LW 需要完整样本：仅保留所有 ETF 均有有效收益率的日期
    # 这里只处理内存中的收益率表，不修改原始价格数据
    complete_returns = window.dropna(how="any")

    if len(complete_returns) < 2:
        raise ValueError(f"截至 {end:%Y-%m-%d} 的 LW 完整共同样本不足两条")

    # 根据历史样本，同时估计协方差矩阵和收缩系数。
    estimator = LedoitWolf().fit(complete_returns.to_numpy())

    # 转回 DataFrame，保留 ETF 名称和排列顺序
    covariance = pd.DataFrame(
        estimator.covariance_,
        index=complete_returns.columns,
        columns=complete_returns.columns,
    )

    return {
        "covariance": covariance,
        "shrinkage": float(estimator.shrinkage_),
        "sample_size": len(complete_returns),

        # 均值与两种协方差均使用同一份完整共同样本。
        "mean_returns": complete_returns.mean(),
        "sample_covariance": complete_returns.cov(),
    }


# 2. 计算每期对比组合的目标权重
def calculate_comparison_weights(
    returns: pd.DataFrame,
    schedule: pd.DataFrame,
) -> dict:
    """使用同一份完整共同样本，计算每期五种组合的目标权重。"""

    results = {}

    for number, row in schedule.iterrows():
        history_start = row["历史窗口起点"]
        signal_date = row["参数估计截止日"]
        execution_date = row["执行日"]
        start_date = None if pd.isna(history_start) else history_start

        # 均值、样本协方差和 LW 协方差使用相同日期和资产。
        statistics = estimate_lw_covariance(
            returns=returns,
            start_date=start_date,
            end_date=signal_date,
        )

        mu = statistics["mean_returns"]
        covariance = statistics["sample_covariance"]
        lw_covariance = statistics["covariance"]

        # 两种收益约束使用相同均值，等权收益门槛均为 mu.mean()。
        weights = pd.DataFrame({
            "等权": equal_weight(mu.index),

            "最小方差": min_variance(covariance),

            "LW最小方差": min_variance(lw_covariance),

            "等权收益约束最小方差": min_variance_equal_return(
                mu, covariance
            ),

            "等权收益约束LW最小方差": min_variance_equal_return(
                mu, lw_covariance
            ),
        })

        results[execution_date] = {
            "期数": number,
            "历史窗口起点": history_start,
            "参数估计截止日": signal_date,
            "目标权重": weights,
            "LW收缩系数": statistics["shrinkage"],
            "LW有效样本数量": statistics["sample_size"],
            "共同有效样本数量": statistics["sample_size"],
            "等权估计日收益门槛": float(mu.mean()),
        }

    return results


# 3. 执行一个指定区间的对比实验
def run_comparison(
    prices: pd.DataFrame,
    first_signal_month: str = "2025-12",
    last_signal_month: str = "2026-09",
    backtest_end: str = "2026-10-30",
    history_start: str | None = None,
    initial_capital: float = 100000.0,
    fee_rate: float = 0.0005,
    annual_risk_free_rate: float = 0.02,
) -> dict:
    """运行指定区间内五种组合的回测，并汇总绩效。"""

    # 根据完整 TRI 历史计算日收益，不按回测起点截断参数估计样本。
    returns = calculate_returns(prices)

    # 沿用原有扩展窗口、月度调仓规则
    schedule = make_rebalance_schedule(
        prices=prices,
        history_start=history_start,
        first_signal_month=first_signal_month,
        last_signal_month=last_signal_month,
        backtest_end=backtest_end,
    )

    # 计算每次调仓的目标权重
    target_weights = calculate_comparison_weights(
        returns=returns,
        schedule=schedule,
    )

    # 已确认的停牌日仅用于估值，不将缺报价填成参数估计样本。
    # 513100 拆分停牌及513030全天停牌公告分别见：
    # https://static.cninfo.com.cn/finalpage/2022-01-04/1212087910.PDF
    # https://www.huaan.com.cn/upload2010/2025/01/22/224505938_0_d128c6da-8383-3d4a-96ba-622ed0a4a5b2.pdf
    valuation_prices = prices.sort_index().copy()
    for name, date_text in (
        ("纳斯达克100", "2022-01-13"),
        ("德国DAX", "2025-01-23"),
    ):
        date = pd.Timestamp(date_text)
        if date not in valuation_prices.index or name not in valuation_prices.columns:
            continue
        if date in set(schedule["执行日"]):
            raise ValueError(f"{name} {date_text} 停牌，不能作为调仓执行日")
        if pd.isna(valuation_prices.loc[date, name]):
            position = valuation_prices.index.get_loc(date)
            if position == 0:
                raise ValueError(f"{name} 停牌日之前无可用估值")
            previous = valuation_prices[name].iloc[position - 1]
            if not np.isfinite(previous) or previous <= 0:
                raise ValueError(f"{name} 停牌日之前缺少有效 TRI")
            valuation_prices.loc[date, name] = previous

    # 沿用原有执行时点、持仓漂移和交易费用计算。
    backtest_results = run_backtest(
        prices=valuation_prices,
        schedule=schedule,
        target_weights=target_weights,
        initial_capital=initial_capital,
        fee_rate=fee_rate,
    )

    # 沿用原有绩效指标计算口径
    performance = calculate_performance(
        backtest_results=backtest_results,
        annual_risk_free_rate=annual_risk_free_rate,
    )

    return {
        "schedule": schedule,
        "target_weights": target_weights,
        "backtest_results": backtest_results,
        "performance": performance,
    }


# 4. 计算不同回测区间的指标
def run_period_comparisons(
    prices: pd.DataFrame,
    start_years: list,
    last_signal_month: str = "2026-09",
    backtest_end: str = "2026-10-30",
    initial_capital: float = 100000.0,
    fee_rate: float = 0.0005,
    annual_risk_free_rate: float = 0.02,
) -> dict:
    """从不同年份年初独立建仓回测，保存各区间结果并汇总绩效。"""

    # 保存各区间的完整结果，供后续绘图使用
    experiments = {}

    # 保存各区间的绩效表
    performance_tables = []

    for year in start_years:

        # 年初建仓，使用上一年 12 月末的信号
        first_signal_month = f"{year - 1}-12"

        # 每次调用都会从初始资金重新建仓
        # history_start=None：参数估计使用截至信号日的全部可用历史
        experiment = run_comparison(
            prices=prices,
            first_signal_month=first_signal_month,
            last_signal_month=last_signal_month,
            backtest_end=backtest_end,
            history_start=None,
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            annual_risk_free_rate=annual_risk_free_rate,
        )

        # 按起始年份保存完整结果
        experiments[year] = experiment

        # 提取当前区间的绩效表
        performance = experiment["performance"].reset_index()

        # 标记实际建仓日和回测结束日
        schedule = experiment["schedule"]

        performance.insert(0, "起始年份", year)
        performance.insert(
            1,
            "首次执行日",
            schedule["执行日"].iloc[0],
        )
        performance.insert(
            2,
            "结束日期",
            schedule["收益记录结束日"].iloc[-1],
        )

        performance_tables.append(performance)

    # 合并五个区间的绩效，使用“起始年份、组合”作为索引
    summary = (
        pd.concat(performance_tables, ignore_index=True)
        .set_index(["起始年份", "组合"])
    )

    return {
        "experiments": experiments,
        "performance": summary,
    }


if __name__ == "__main__":

    # 读取总回报指数，现金分红和份额折算已在数据模块处理。
    project_dir = Path(__file__).resolve().parent

    prices = pd.read_csv(
        project_dir / "etf_total_return.csv",
        index_col=0,
        parse_dates=True,
    ).sort_index()

    # 分别从五个年份年初独立建仓，统一回测至 2026-10-30
    # 每次调仓均使用截至信号日的全部可用历史重新估计参数
    period_results = run_period_comparisons(
        prices=prices,
        start_years=[2022, 2023, 2024, 2025, 2026],
        last_signal_month="2026-09",
        backtest_end="2026-10-30",
        initial_capital=100000.0,
        fee_rate=0.0005,
        annual_risk_free_rate=0.02,
    )

    # 打印五个区间、五种组合的绩效汇总。
    performance = period_results["performance"]

    display_columns = [
        "累计收益率",
        "年化收益率",
        "年化波动率",
        "夏普比率",
        "最大回撤",
        "平均双边调仓换手率（不含建仓）",
    ]

    print("\n不同回测起点的绩效对比：")
    print(performance[display_columns].to_string())