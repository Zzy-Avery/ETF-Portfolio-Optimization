import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf
from pathlib import Path

from optimization import (
    calculate_returns,
    estimate_statistics,
    equal_weight,
    min_variance,
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
    """使用指定历史区间的完整收益率样本，估计 LW 协方差矩阵。"""

    # 截取截至信号日的历史收益率，不使用未来数据
    window = returns.sort_index().loc[start_date:end_date].copy()

    # LW 需要完整样本：仅保留所有 ETF 均有有效收益率的日期
    # 这里只处理内存中的收益率表，不修改原始价格数据
    complete_returns = window.dropna(how="any")

    # 根据历史样本，同时估计协方差矩阵和收缩系数
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
    }


# 2. 计算每期对比组合的目标权重
def calculate_comparison_weights(
    returns: pd.DataFrame,
    schedule: pd.DataFrame,
) -> dict:
    """按照调仓计划，计算最小方差和 LW 最小方差的权重。"""

    results = {}

    for number, row in schedule.iterrows():

        # 读取本期历史窗口、信号日和执行日
        history_start = row["历史窗口起点"]
        signal_date = row["参数估计截止日"]
        execution_date = row["执行日"]

        # 起点为 None 时，使用截至信号日的全部可用历史
        start_date = None if pd.isna(history_start) else history_start

        # 原有最小方差：继续使用原来的统计量估计方法
        statistics = estimate_statistics(
            returns=returns,
            start_date=start_date,
            end_date=signal_date,
        )

        # LW 最小方差：使用 LW 协方差估计
        lw_statistics = estimate_lw_covariance(
            returns=returns,
            start_date=start_date,
            end_date=signal_date,
        )

        # 两种方法都调用 min_variance()，只改变输入的协方差矩阵
        weights = pd.DataFrame({
            "等权": equal_weight(returns.columns),
            "最小方差": min_variance(statistics["covariance"]),
            "LW最小方差": min_variance(lw_statistics["covariance"]),
        })

        # 沿用原回测函数需要的字典结构
        # 同时记录每期 LW 的收缩系数和有效样本数量
        results[execution_date] = {
            "期数": number,
            "历史窗口起点": history_start,
            "参数估计截止日": signal_date,
            "目标权重": weights,
            "LW收缩系数": lw_statistics["shrinkage"],
            "LW有效样本数量": lw_statistics["sample_size"],
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
    """运行指定区间内三种组合的回测，并汇总绩效。"""

    # 根据完整价格历史计算收益率
    # 不先按回测起点截断，以保留建仓前的参数估计样本
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

    # 沿用原有执行时点、持仓漂移和交易费用计算
    backtest_results = run_backtest(
        prices=prices,
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


#4. 计算不同回测区间的指标
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

    # 读取当前文件同目录下的价格数据
    project_dir = Path(__file__).resolve().parent

    prices = pd.read_csv(
        project_dir / "etf_prices.csv",
        index_col=0,
        parse_dates=True,
    ).sort_index()

    # 仅填补已经确认的两个历史价格缺口
    # 使用前一交易日收盘价，不修改原 CSV
    prices.loc["2022-01-13", "纳斯达克100"] = (
        prices.loc["2022-01-12", "纳斯达克100"]
    )

    prices.loc["2025-01-23", "德国DAX"] = (
        prices.loc["2025-01-22", "德国DAX"]
    )

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

    # 打印五个区间、两种组合的绩效汇总
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