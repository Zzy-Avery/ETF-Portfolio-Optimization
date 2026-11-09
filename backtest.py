import pandas as pd
#从optimization.py中导入已经定义好的7个函数
from optimization import (
    calculate_returns,
    estimate_statistics,
    equal_weight,
    min_variance,
    min_variance_equal_return,
    risk_parity,
    select_momentum_etfs,
)
import numpy as np


def make_rebalance_schedule(
    prices: pd.DataFrame,
    history_start: str | None = None,
    first_signal_month: str = "2025-12",
    last_signal_month: str = "2026-09",
    backtest_end: str = "2026-10-30",
) -> pd.DataFrame:
    """生成扩展窗口、月度调仓计划。"""

    # 检查日期索引
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("价格表必须使用日期索引")

    if prices.empty:
        raise ValueError("价格表不能为空")

    if prices.index.hasnans or prices.index.has_duplicates:
        raise ValueError("价格表日期不能缺失或重复")

    # 按日期排序，并只保留回测结束日之前的数据
    dates = prices.index.sort_values()
    end = pd.Timestamp(backtest_end)
    dates = dates[dates <= end]

    if len(dates) == 0:
        raise ValueError("回测结束日之前没有价格数据")

    # 避免数据不足时悄悄提前结束回测
    if dates[-1] != end:
        raise ValueError("回测结束日必须在价格表中存在")

    start = pd.Timestamp(history_start) if history_start is not None else pd.NaT
    first_month = pd.Period(first_signal_month, freq="M")
    last_month = pd.Period(last_signal_month, freq="M")

    if first_month > last_month:
        raise ValueError("首个信号月份不能晚于最后一个信号月份")

    if pd.notna(start) and start > first_month.start_time:
        raise ValueError("历史窗口起点不能晚于首个信号月份的月初")

    # 最后一个信号月份必须已经完整结束
    if end.normalize() < last_month.end_time.normalize():
        raise ValueError("回测结束日早于最后一个信号月份的月末")

    # 生成需要计算权重的月份：2025-12、2026-01……2026-09
    months = pd.period_range(
        start=first_month,
        end=last_month,
        freq="M",
    )

    records = []

    for month in months:
        # 筛选该月在价格表中的全部交易日期
        month_dates = dates[dates.to_period("M") == month]

        if len(month_dates) == 0:
            raise ValueError(f"{month}没有价格数据")

        # 当月最后一个交易日：参数估计截止日
        signal_date = month_dates[-1]

        # 找到参数估计截止日在全部日期中的位置
        signal_position = dates.get_loc(signal_date)

        # 后面至少需要一个执行日和一个收益记录日
        if signal_position + 2 >= len(dates):
            raise ValueError(f"{month}之后没有足够的执行及收益数据")

        # 下一交易日收盘执行
        execution_date = dates[signal_position + 1]

        # 执行后的下一个交易日，记录第一笔持仓日收益
        return_start = dates[signal_position + 2]

        records.append({
            "历史窗口起点": start,
            "参数估计截止日": signal_date,
            "执行日": execution_date,
            "收益记录开始日": return_start,
        })

    schedule = pd.DataFrame(records)

    # 本轮收益记录到下一次执行日，包含该日
    schedule["收益记录结束日"] = schedule["执行日"].shift(-1)

    # 最后一轮记录到回测结束日
    schedule.loc[
        schedule.index[-1], "收益记录结束日"
    ] = end

    schedule.index = pd.RangeIndex(
        start=1,
        stop=len(schedule) + 1,
        name="次数",
    )

    return schedule


def calculate_target_weights(
    returns: pd.DataFrame,
    schedule: pd.DataFrame,
    prices: pd.DataFrame,
) -> dict:
    """按照调仓计划，计算每一期八种组合的目标权重。"""

    # 创建空字典，用来保存每一期的计算结果
    results = {}

    # 每次读取调仓计划中的一行
    for number, row in schedule.iterrows():

        # 取出本期历史窗口的起点和终点
        history_start = row["历史窗口起点"]
        signal_date = row["参数估计截止日"]

        # 取出本期实际执行日期
        execution_date = row["执行日"]

        # 使用本期历史窗口估计参数
        statistics = estimate_statistics(
            returns=returns,
            start_date=None if pd.isna(history_start) else history_start,
            end_date=signal_date,
        )

        # 提取平均日收益率和日收益率协方差矩阵
        mu = statistics["mean_returns"]
        covariance = statistics["covariance"]

        # 用截至本期信号日的价格，选出过去 126 个交易日涨幅最高的 5 只 ETF
        selected = select_momentum_etfs(
            prices=prices,
            signal_date=signal_date,
            lookback=126,
            top_n=5,
        )

        # 从全池统计量中取出这 5 只 ETF 对应的均值和协方差
        selected_mu = mu.loc[selected]
        selected_covariance = covariance.loc[selected, selected]

        weights = pd.DataFrame({
            "等权": equal_weight(mu.index),
            "最小方差": min_variance(covariance),
            "等权收益约束最小方差": min_variance_equal_return(mu, covariance),
            "风险平价": risk_parity(covariance),

            "动量等权": equal_weight(selected),
            "动量最小方差": min_variance(selected_covariance),
            "动量收益约束最小方差": min_variance_equal_return(
                selected_mu, selected_covariance
            ),
            "动量风险平价": risk_parity(selected_covariance),
        }, index=mu.index).fillna(0.0)

        # 以执行日期为键，保存本期结果
        results[execution_date] = {
            "期数": number,
            "历史窗口起点": history_start,
            "参数估计截止日": signal_date,
            "样本数量": len(statistics["window_returns"]),
            "目标权重": weights,
            "动量入选ETF": selected,
        }

    # 全部期数计算完毕后，返回保存好的结果
    return results


def execute_rebalance(
    current_shares: pd.Series,
    cash: float,
    prices: pd.Series,
    target_weights: pd.Series,
    fee_rate: float = 0.0005,
) -> dict:
    """单词减仓/调仓的函数：按目标权重完成一次调仓，费用从现有资产中扣除。"""

    # 1. 按目标权重中的ETF顺序对齐数据
    assets = target_weights.index

    shares = current_shares.reindex(assets)
    prices = prices.reindex(assets)
    weights = target_weights.copy()

    if not 0 <= fee_rate < 1:
        raise ValueError("费率必须满足0 <= fee_rate < 1")

    if (
            (weights < 0).any()
            or abs(weights.sum() - 1.0) > 1e-8
    ):
        raise ValueError("目标权重必须非负，且总和为1")

    # 修正权重和的微小浮点误差
    weights = weights / weights.sum()

    # 2. 计算旧持仓市值和调仓前总资产
    # 旧持仓需要估值，目标权重为正的 ETF 需要有效价格来买入
    required = shares.ne(0) | weights.gt(0)
    invalid = required & (~np.isfinite(prices) | prices.le(0))
    if invalid.any():
        names = "、".join(prices.index[invalid])
        raise ValueError(f"调仓时缺少有效收盘价：{names}")

    old_values = shares * prices.where(shares.ne(0), 0.0)
    value_before = float(old_values.sum() + cash)

    # 金额容差，单位为元
    tolerance = max(1e-8, 1e-10 * value_before)

    # 3. 定义资金平衡方程
    def balance_error(amount):
        target_values = amount * weights
        traded_amount = float(
            (target_values - old_values).abs().sum()
        )

        return amount + fee_rate * traded_amount - value_before

    # 4. 二分求解扣费后的可配置金额
    lower = 0.0
    upper = value_before

    # 无费用或无需交易时，可以直接配置全部资产
    if balance_error(upper) == 0:
        investable_value = upper

    else:
        for _ in range(100):
            middle = lower + (upper - lower) / 2

            if balance_error(middle) > 0:
                upper = middle
            else:
                lower = middle

            # 取不超支的一侧，剩余资金小于容差时停止
            if abs(balance_error(lower)) <= tolerance:
                break
        else:
            raise RuntimeError("调仓金额求解未收敛")

        investable_value = lower

    # 5. 计算新目标市值及实际买卖金额
    target_values = investable_value * weights
    trade_values = target_values - old_values

    buy_amount = float(trade_values.clip(lower=0).sum())
    sell_amount = float((-trade_values).clip(lower=0).sum())

    fee = fee_rate * (buy_amount + sell_amount)

    # 6. 计算新份额和剩余现金，目标权重为零时，目标市值和新份额都应为零
    new_shares = target_values / prices.where(weights.gt(0), 1.0)

    new_cash = float(
        value_before - target_values.sum() - fee
    )

    # 只允许数值误差级的剩余现金
    if abs(new_cash) > tolerance:
        raise RuntimeError("调仓后现金余额异常")

    new_cash = max(new_cash, 0.0)

    value_after = float(
        (new_shares * prices.where(new_shares.ne(0), 0.0)).sum() + new_cash
    )

    # 7. 检查资金守恒
    if abs(value_before - value_after - fee) > tolerance:
        raise RuntimeError("调仓前后资金不守恒")

    return {
        "新持有份额": new_shares,
        "调仓后现金": new_cash,
        "各ETF交易金额": trade_values,
        "买入金额": buy_amount,
        "卖出金额": sell_amount,
        "交易费用": fee,
        "调仓前总资产": value_before,
        "调仓后总资产": value_after,
    }


def run_backtest(
    prices: pd.DataFrame,
    schedule: pd.DataFrame,
    target_weights: dict,
    initial_capital: float = 100000.0,
    fee_rate: float = 0.0005,
) -> dict:
    """按调仓计划进行每日回测，返回各组合的每日与交易记录。"""

    if initial_capital <= 0:
        raise ValueError("初始资金必须为正")

    # 按时间排序，避免日期顺序影响回测
    prices = prices.sort_index()
    schedule = schedule.sort_values("执行日")

    # 从第一次参数估计截止日开始记录，保留建仓前的资金基准
    start_date = schedule["参数估计截止日"].iloc[0]
    end_date = schedule["收益记录结束日"].iloc[-1]

    backtest_prices = prices.loc[start_date:end_date]

    # 所有ETF名称
    assets = prices.columns

    # 执行日期集合，用来判断当天是否需要调仓
    execution_dates = set(schedule["执行日"])

    # 从第一期目标权重表中获取八种组合的名称
    first_execution = schedule["执行日"].iloc[0]
    strategy_names = target_weights[
        first_execution
    ]["目标权重"].columns

    # 保存八种组合各自的回测结果
    results = {}

    # 八种组合分别回测
    for strategy in strategy_names:

        # 初始状态：没有ETF持仓，全部资金为现金
        shares = pd.Series(0.0, index=assets)
        cash = float(initial_capital)

        # 用于计算当天收益率
        previous_value = float(initial_capital)

        daily_records = []
        share_records = []
        weight_records = []
        trade_records = []

        # 按交易日依次模拟
        for date, daily_prices in backtest_prices.iterrows():

            # 1. 先按当天收盘价计算旧持仓市值
            held = shares.ne(0)
            invalid = held & (~np.isfinite(daily_prices) | daily_prices.le(0))
            if invalid.any():
                names = "、".join(daily_prices.index[invalid])
                raise ValueError(f"{date:%Y-%m-%d} {strategy} 持仓缺少有效收盘价：{names}")

            holding_values = shares * daily_prices.where(held, 0.0)
            value_before = float(holding_values.sum() + cash)

            # 普通持仓日没有交易费用
            daily_fee = 0.0

            # 2. 如果是执行日，则在收盘时建仓或调仓
            if date in execution_dates:

                # 取出当天、当前组合对应的目标权重
                weights = target_weights[
                    date
                ]["目标权重"][strategy]

                # 调用单次调仓函数
                trade = execute_rebalance(
                    current_shares=shares,
                    cash=cash,
                    prices=daily_prices,
                    target_weights=weights,
                    fee_rate=fee_rate,
                )

                # 更新持仓和现金
                shares = trade["新持有份额"]
                cash = trade["调仓后现金"]
                daily_fee = trade["交易费用"]

                # 保存本次交易记录
                trade_records.append({
                    "日期": date,
                    "参数估计截止日": target_weights[
                        date
                    ]["参数估计截止日"],
                    "调仓前总资产": trade["调仓前总资产"],
                    "买入金额": trade["买入金额"],
                    "卖出金额": trade["卖出金额"],
                    "买卖金额合计": (
                        trade["买入金额"] + trade["卖出金额"]
                    ),
                    "交易费用": daily_fee,
                    "调仓后总资产": trade["调仓后总资产"],
                })

            # 3. 计算当天交易完成后的资产
            holding_values = shares * daily_prices.where(shares.ne(0), 0.0)
            total_value = float(holding_values.sum() + cash)

            # 当天收益包含旧持仓涨跌和当天交易费用
            daily_return = total_value / previous_value - 1.0

            # 净值以初始资金为基准
            nav = total_value / initial_capital

            daily_records.append({
                "日期": date,
                "交易前总资产": value_before,
                "总资产": total_value,
                "现金": cash,
                "净值": nav,
                "日收益率": daily_return,
                "交易费用": daily_fee,
            })

            # 保存当天收盘后的份额和实际权重
            share_records.append(shares.copy())
            weight_records.append(
                holding_values / total_value
            )

            # 更新下一交易日使用的资产基准
            previous_value = total_value

        # 4. 将逐日记录整理成表格
        daily = pd.DataFrame(
            daily_records
        ).set_index("日期")

        share_history = pd.DataFrame(
            share_records,
            index=daily.index,
        )

        weight_history = pd.DataFrame(
            weight_records,
            index=daily.index,
        )

        trades = pd.DataFrame(
            trade_records
        ).set_index("日期")

        results[strategy] = {
            "每日记录": daily,
            "持有份额": share_history,
            "实际权重": weight_history,
            "交易记录": trades,
        }

    return results


def calculate_performance(
    backtest_results: dict,
    annual_risk_free_rate: float = 0.02,
    trading_days: int = 252,
) -> pd.DataFrame:
    """根据每日记录和交易记录，汇总各组合的回测指标。"""

    # 年无风险利率换算为等效日利率
    daily_risk_free_rate = (
        (1 + annual_risk_free_rate) ** (1 / trading_days) - 1
    )

    records = {}

    for strategy, result in backtest_results.items():

        daily = result["每日记录"].sort_index()
        trades = result["交易记录"].sort_index()

        nav = daily["净值"]

        # 第一行是初始资金基准，不是一个实际收益区间。去掉它，但保留首次建仓日的费用收益
        returns = daily["日收益率"].iloc[1:]

        n = len(returns)

        if n < 2:
            raise ValueError("计算绩效指标至少需要两个日收益区间")

        # 1. 累计收益率与复利年化收益率
        total_growth = float(nav.iloc[-1] / nav.iloc[0])

        total_return = total_growth - 1

        annual_return = (
            total_growth ** (trading_days / n) - 1
        )

        # 2. 年化波动率
        daily_volatility = float(returns.std(ddof=1))

        annual_volatility = (
            daily_volatility * np.sqrt(trading_days)
        )

        # 3. 夏普比率
        excess_returns = returns - daily_risk_free_rate

        if daily_volatility > 0:
            sharpe_ratio = float(
                excess_returns.mean()
                / daily_volatility
                * np.sqrt(trading_days)
            )
        else:
            sharpe_ratio = np.nan

        # 4. 每日回撤及最大回撤
        running_peak = nav.cummax()
        drawdown = nav / running_peak - 1

        # 显示为正数，例如0.08表示最大回撤8%
        max_drawdown = float(-drawdown.min())

        # 5. Calmar比率
        if max_drawdown > 0:
            calmar_ratio = annual_return / max_drawdown
        else:
            calmar_ratio = np.nan

        # 6. 双边换手率
        # 每次买卖金额合计 / 当次调仓前总资产
        turnover = (
            trades["买卖金额合计"]
            / trades["调仓前总资产"]
        )

        # 包含首次建仓
        cumulative_turnover = float(turnover.sum())

        # 排除首次建仓，只统计后续调仓
        if len(turnover) > 1:
            average_rebalance_turnover = float(
                turnover.iloc[1:].mean()
            )
        else:
            average_rebalance_turnover = np.nan

        # 7. 累计交易费用
        total_fees = float(trades["交易费用"].sum())

        # 保存当前组合的指标
        records[strategy] = {
            "期末净值": float(nav.iloc[-1]),
            "期末总资产": float(daily["总资产"].iloc[-1]),
            "累计收益率": total_return,
            "年化收益率": annual_return,
            "年化波动率": annual_volatility,
            "夏普比率": sharpe_ratio,
            "Calmar比率": calmar_ratio,
            "最大回撤": max_drawdown,
            "累计双边换手率（含建仓）": cumulative_turnover,
            "平均双边调仓换手率（不含建仓）": (
                average_rebalance_turnover
            ),
            "累计费用": total_fees,
        }

    performance = pd.DataFrame.from_dict(
        records,
        orient="index",
    )

    performance.index.name = "组合"

    return performance


if __name__ == "__main__":
    from pathlib import Path

    # 读取与 backtest.py 放在同一文件夹中的价格表
    project_dir = Path(__file__).resolve().parent
    prices = pd.read_csv(
        project_dir / "etf_prices.csv",
        index_col="Date",
        parse_dates=["Date"],
        encoding="utf-8-sig",
    )

    # 生成并打印调仓计划，保留原表的日期类型供后续计算
    schedule = make_rebalance_schedule(prices)
    display_schedule = schedule.apply(
        lambda column: pd.to_datetime(column).dt.strftime("%Y-%m-%d")
    )
    display_schedule["历史窗口起点"] = (
        display_schedule["历史窗口起点"].fillna("各 ETF 有效历史起点")
    )
    print("调仓计划")
    print(display_schedule.to_string())

    # 计算日收益率，再生成各期八种组合的目标权重
    returns = calculate_returns(prices)
    target_weights = calculate_target_weights(
        returns=returns,
        schedule=schedule,
        prices=prices,
    )

    # 对八种组合分别进行每日回测
    backtest_results = run_backtest(
        prices=prices,
        schedule=schedule,
        target_weights=target_weights,
        initial_capital=100000.0,
        fee_rate=0.0005,
    )

    # 计算并打印八种组合的绩效指标
    performance = calculate_performance(
        backtest_results=backtest_results,
        annual_risk_free_rate=0.02,
    )
    print("\n八种组合的回测绩效")
    print(performance.to_string(float_format=lambda value: f"{value:.6f}"))