import numpy as np
import pandas as pd
from scipy.optimize import minimize


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """将价格表转换为简单日收益率表。"""

    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("价格表必须使用日期索引")

    if prices.index.has_duplicates:
        raise ValueError("价格表存在重复日期")

    if prices.columns.has_duplicates:
        raise ValueError("价格表存在重复的 ETF 列名")

    if len(prices) < 2 or prices.shape[1] == 0:
        raise ValueError("价格表至少需要两行数据和一只 ETF")

    # 按日期从早到晚排序
    prices = prices.sort_index()

    # 当天收益率 = 当天价格 / 前一天价格 - 1
    returns = prices.pct_change(fill_method=None)

    # 第一行没有前一天价格，无法计算收益率
    return returns.iloc[1:].copy()


def estimate_statistics(
    returns: pd.DataFrame,
    end_date: str,
    start_date: str = "2022-01-01",
) -> dict:
    """计算从固定起点到指定截止日期的日度统计量。"""

    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("收益率表必须使用日期索引")

    if returns.index.has_duplicates:
        raise ValueError("收益率表存在重复日期")

    if returns.columns.has_duplicates:
        raise ValueError("收益率表存在重复的 ETF 列名")

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    if pd.isna(start) or pd.isna(end):
        raise ValueError("开始日期和截止日期不能为空")

    if start > end:
        raise ValueError("开始日期不能晚于截止日期")

    # 起点固定，截止日期随每次调仓更新
    window = returns.sort_index().loc[start:end].copy()

    if len(window) < 2 or window.shape[1] == 0:
        raise ValueError("历史窗口至少需要两条收益率记录和一只 ETF")

    # 不让缺失值或无穷值被忽略后继续计算
    if not np.isfinite(window.to_numpy(dtype=float)).all():
        raise ValueError("所选窗口的收益率存在缺失值或无穷值")

    # 每只 ETF 的平均日收益率
    mean_returns = window.mean()

    # 每只 ETF 的样本日波动率
    volatility = window.std(ddof=1)

    # 日收益率样本协方差矩阵
    covariance = window.cov()

    # 日收益率相关系数矩阵
    correlation = window.corr()

    return {
        "window_returns": window,
        "mean_returns": mean_returns,
        "volatility": volatility,
        "covariance": covariance,
        "correlation": correlation,
    }


def equal_weight(asset_names) -> pd.Series:
    """生成等权组合的目标权重。"""

    # 将 ETF 名单转换为列表
    asset_names = list(asset_names)

    # ETF 数量
    n = len(asset_names)

    if n == 0:
        raise ValueError("ETF 名单不能为空")

    if len(set(asset_names)) != n:
        raise ValueError("ETF 名称不能重复")

    # 每只 ETF 分配 1/n 的资金比例
    weights = pd.Series(
        data=1.0 / n,
        index=asset_names,
        name="weight",
    )

    return weights


def min_variance(covariance: pd.DataFrame) -> pd.Series:
    """计算最小方差组合权重。"""

    # 检查协方差矩阵的形状
    n = len(covariance)

    if n == 0 or covariance.shape != (n, n):
        raise ValueError("协方差矩阵必须是非空方阵")

    # 行和列必须对应相同的 ETF，且顺序一致
    if not covariance.index.equals(covariance.columns):
        raise ValueError("协方差矩阵的行列名称及顺序必须一致")

    if covariance.index.has_duplicates:
        raise ValueError("ETF 名称不能重复")

    # 转为 NumPy 数组，用于数值优化
    sigma = covariance.to_numpy(dtype=float)

    if not np.isfinite(sigma).all():
        raise ValueError("协方差矩阵存在缺失值或无穷值")

    if not np.allclose(sigma, sigma.T, rtol=1e-8, atol=1e-12):
        raise ValueError("协方差矩阵必须对称")

    # 消除浮点计算产生的微小不对称
    sigma = (sigma + sigma.T) / 2

    # 对目标函数缩放，避免日度方差数值太小影响求解，因为除以正常数不会改变最优权重
    scale = np.max(np.abs(sigma))

    # 所有协方差均为零时，所有可行权重的方差都为零
    if scale == 0:
        return equal_weight(covariance.index)

    scaled_sigma = sigma / scale

    # 协方差矩阵应为半正定，允许浮点计算带来的微小误差
    if np.linalg.eigvalsh(scaled_sigma).min() < -1e-10:
        raise ValueError("协方差矩阵不是半正定矩阵")

    # 目标函数：组合方差（缩放后）
    def objective(w):
        return w @ scaled_sigma @ w

    # 目标函数的梯度
    def gradient(w):
        return 2 * scaled_sigma @ w

    # 从等权组合开始迭代
    initial_weights = np.full(n, 1.0 / n)

    # 每只 ETF 的权重介于 0 和 1 之间
    bounds = [(0.0, 1.0)] * n

    # 所有权重之和等于 1
    constraints = {
        "type": "eq",
        "fun": lambda w: w.sum() - 1.0,
        "jac": lambda w: np.ones(n),
    }

    # 求解最小方差问题
    result = minimize(
        fun=objective,
        x0=initial_weights,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "ftol": 1e-12,
            "maxiter": 1000,
        },
    )

    if not result.success:
        raise RuntimeError(f"最小方差优化失败：{result.message}")

    w = result.x

    # 检查返回的权重是否满足约束
    if (
        not np.isfinite(w).all()
        or abs(w.sum() - 1.0) > 1e-8
        or (w < -1e-8).any()
        or (w > 1.0 + 1e-8).any()
    ):
        raise RuntimeError("优化结果不满足权重约束")

    # 只修正浮点误差造成的极小越界，并恢复权重和为 1
    w = np.clip(w, 0.0, 1.0)
    w = w / w.sum()

    return pd.Series(
        data=w,
        index=covariance.index,
        name="weight",
    )


def min_variance_equal_return(
    mean_returns: pd.Series,
    covariance: pd.DataFrame,
) -> pd.Series:
    """估计收益不低于等权组合时，计算最小方差组合权重。"""

    # 按平均收益率的 ETF 顺序排列协方差矩阵
    asset_names = mean_returns.index
    mu = mean_returns.to_numpy(dtype=float)
    sigma = covariance.loc[
        asset_names, asset_names
    ].to_numpy(dtype=float)

    n = len(mu)

    # 等权组合的估计平均日收益率
    target_return = mu.mean()

    # 缩放方差目标，避免日度方差数值太小
    variance_scale = np.max(np.abs(sigma))

    if variance_scale == 0:
        return equal_weight(asset_names)

    scaled_sigma = sigma / variance_scale

    # 缩放收益约束，不改变约束含义
    return_scale = np.max(np.abs(mu))

    if return_scale == 0:
        return_scale = 1.0

    excess_mu = (mu - target_return) / return_scale

    # 目标函数：最小化组合方差
    def objective(w):
        return w @ scaled_sigma @ w

    # 目标函数的梯度
    def gradient(w):
        return (scaled_sigma + scaled_sigma.T) @ w

    # 从等权组合开始迭代
    initial_weights = np.full(n, 1.0 / n)

    constraints = [
        {
            # 权重之和等于 1
            "type": "eq",
            "fun": lambda w: w.sum() - 1.0,
            "jac": lambda w: np.ones(n),
        },
        {
            # 组合估计收益不低于等权组合
            # SLSQP 的不等式要求函数值 >= 0
            "type": "ineq",
            "fun": lambda w: excess_mu @ w,
            "jac": lambda w: excess_mu,
        },
    ]

    result = minimize(
        fun=objective,
        x0=initial_weights,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=constraints,
        options={
            "ftol": 1e-12,
            "maxiter": 1000,
        },
    )

    # 求解失败时停止，避免把失败结果用于回测
    if not result.success:
        raise RuntimeError(
            f"收益约束最小方差优化失败：{result.message}"
        )

    w = result.x

    # 检查求解结果是否满足约束，允许微小浮点误差
    if (
        not np.isfinite(w).all()
        or abs(w.sum() - 1.0) > 1e-8
        or (w < -1e-8).any()
        or (w > 1.0 + 1e-8).any()
        or excess_mu @ w < -1e-8
    ):
        raise RuntimeError("优化结果不满足权重或收益约束")

    return pd.Series(
        data=w,
        index=asset_names,
        name="weight",
    )


if __name__ == "__main__":
    from pathlib import Path

    # 读取与 optimization.py 放在同一文件夹中的价格表
    project_dir = Path(__file__).resolve().parent

    prices = pd.read_csv(
        project_dir / "etf_prices.csv",
        index_col="Date",
        parse_dates=["Date"],
        encoding="utf-8-sig",
    )

    # 计算日收益率
    returns = calculate_returns(prices)

    # 第一次配置只使用 2022—2025 年的数据
    statistics = estimate_statistics(
        returns,
        end_date="2025-12-31",
    )

    mu = statistics["mean_returns"]
    covariance = statistics["covariance"]

    # 分别计算三种组合权重
    weights = pd.DataFrame({
        "等权": equal_weight(mu.index),
        "最小方差": min_variance(covariance),
        "等权收益约束最小方差": min_variance_equal_return(
            mu,
            covariance,
        ),
    })

    print("三种组合的目标权重：")
    print(weights.to_string(float_format=lambda x: f"{x:.4%}"))

    # 使用同一组历史参数，计算三种组合的估计收益和风险
    comparison = pd.DataFrame(
        index=weights.columns,
        columns=["估计平均日收益率", "估计日方差", "估计日波动率"],
        dtype=float,
    )

    sigma = covariance.to_numpy()

    for name in weights.columns:
        w = weights[name].to_numpy()

        portfolio_return = mu.to_numpy() @ w
        portfolio_variance = w @ sigma @ w

        comparison.loc[name] = [
            portfolio_return,
            portfolio_variance,
            np.sqrt(max(portfolio_variance, 0.0)),
        ]

    print("\n历史参数下的组合比较：")
    print(comparison.to_string(float_format=lambda x: f"{x:.8f}"))

    # 检查权重；允许微小浮点误差
    weight_tol = 1e-8

    # 收益、方差检查分别使用与输入尺度对应的容差
    return_tol = 1e-8 * max(float(mu.abs().max()), 1e-12)
    variance_tol = 1e-8 * max(float(np.abs(sigma).max()), 1e-12)

    eq_return = comparison.loc["等权", "估计平均日收益率"]
    constrained_return = comparison.loc[
        "等权收益约束最小方差", "估计平均日收益率"
    ]

    eq_variance = comparison.loc["等权", "估计日方差"]
    min_variance_value = comparison.loc["最小方差", "估计日方差"]
    constrained_variance = comparison.loc[
        "等权收益约束最小方差", "估计日方差"
    ]

    checks = {
        "所有权重均为有限数": np.isfinite(weights.to_numpy()).all(),
        "所有权重均非负": (weights.to_numpy() >= -weight_tol).all(),
        "各组合权重之和为1": (
            (weights.sum() - 1.0).abs() <= weight_tol
        ).all(),
        "最小方差组合的方差不高于等权": (
            min_variance_value <= eq_variance + variance_tol
        ),
        "最小方差组合的方差不高于收益约束组合": (
            min_variance_value <= constrained_variance + variance_tol
        ),
        "收益约束组合的估计收益不低于等权": (
            constrained_return >= eq_return - return_tol
        ),
        "收益约束组合的方差不高于等权": (
            constrained_variance <= eq_variance + variance_tol
        ),
    }

    print("\n检查结果：")
    for description, passed in checks.items():
        print(f"{description}：{'通过' if passed else '未通过'}")