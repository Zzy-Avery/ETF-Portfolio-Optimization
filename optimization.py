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
    start_date: str | None = None,
) -> dict:
    """计算从固定起点到指定截止日期的日度统计量。"""

    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("收益率表必须使用日期索引")

    if returns.index.has_duplicates:
        raise ValueError("收益率表存在重复日期")

    if returns.columns.has_duplicates:
        raise ValueError("收益率表存在重复的 ETF 列名")

    start = pd.Timestamp(start_date) if start_date is not None else None
    end = pd.Timestamp(end_date)

    if pd.isna(end) or (start is not None and pd.isna(start)):
        raise ValueError("开始日期和截止日期不能为空")
    if start is not None and start > end:
        raise ValueError("开始日期不能晚于截止日期")

    window = returns.sort_index().loc[start:end].copy()

    if window.shape[1] == 0:
        raise ValueError("历史窗口至少需要一只 ETF")
    if np.isinf(window.to_numpy(dtype=float)).any():
        raise ValueError("所选窗口的收益率存在无穷值")

    valid = window.notna().astype(np.int64)
    pair_counts = valid.T @ valid

    rows, cols = np.where(np.triu(pair_counts.to_numpy() < 2))
    if len(rows):
        pairs = [
            f"{window.columns[i]} / {window.columns[j]}"
            for i, j in zip(rows, cols)
        ]
        raise ValueError(
            f"截至 {end:%Y-%m-%d} 的有效共同收益不足两条："
            + "、".join(pairs)
        )

    # 每只 ETF 的平均日收益率
    mean_returns = window.mean()

    # 每只 ETF 的样本日波动率
    volatility = window.std(ddof=1)

    # 日收益率样本协方差矩阵
    pairwise_covariance = window.cov()

    # 日收益率相关系数矩阵
    correlation = window.corr()

    sigma = pairwise_covariance.to_numpy()
    covariance = pairwise_covariance.copy()

    min_eigenvalue = np.linalg.eigvalsh(sigma).min()
    if min_eigenvalue < -1e-10 * np.abs(sigma).max():
        diagonal = np.diag(np.diag(sigma))

        # 找到能保留的最大非对角线比例
        low, high = 0.0, 1.0
        for _ in range(50):
            weight = (low + high) / 2
            candidate = weight * sigma + (1 - weight) * diagonal
            if np.linalg.eigvalsh(candidate).min() >= 0:
                low = weight
            else:
                high = weight

        covariance.iloc[:, :] = low * sigma + (1 - low) * diagonal

    return {
        "window_returns": window,
        "mean_returns": mean_returns,
        "volatility": volatility,
        "covariance": covariance,
        "pairwise_covariance": pairwise_covariance,
        "pair_counts": pair_counts,
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


def risk_parity(covariance: pd.DataFrame) -> pd.Series:
    """计算只做多、等风险贡献的组合权重。"""
    if covariance.empty or not covariance.index.equals(covariance.columns):
        raise ValueError("风险平价要求协方差矩阵的行列资产一致且非空")
    if covariance.index.has_duplicates:
        raise ValueError("ETF 名称不能重复")

    sigma = covariance.to_numpy(dtype=float)
    if not np.isfinite(sigma).all() or not np.allclose(
        sigma, sigma.T, rtol=1e-8, atol=1e-12
    ):
        raise ValueError("风险平价要求协方差矩阵有限且对称")

    sigma = (sigma + sigma.T) / 2
    if (np.diag(sigma) <= 0).any():
        raise ValueError("等风险贡献要求每只 ETF 的方差为正")

    # 用各资产的波动率缩放坐标，减小波动率差异对迭代的影响。
    volatility = np.sqrt(np.diag(sigma))
    scaled = sigma / volatility[:, None] / volatility[None, :]
    if np.linalg.eigvalsh(scaled).min() < -1e-10:
        raise ValueError("风险平价要求协方差矩阵半正定")

    n = len(sigma)
    budget = 1.0 / n
    x = np.full(n, np.sqrt(budget))

    # 每一轮依次更新所有资产的辅助变量 x_i。
    for _ in range(5000):
        for i, row in enumerate(scaled):
            diagonal = row[i]
            other = row @ x - diagonal * x[i]

            # 解 diagonal * x_i**2 + other * x_i - budget = 0 的正根。
            root = np.hypot(other, 2 * np.sqrt(diagonal * budget))
            if other >= 0:
                x[i] = 2 * budget / (root + other)
            else:
                x[i] = (root - other) / (2 * diagonal)

        # 更新完一整轮，再检查所有资产的风险贡献。
        contributions = x * (scaled @ x)
        if np.allclose(contributions, budget, rtol=0, atol=1e-7 * budget):
            weights = x / volatility
            weights /= weights.sum()
            return pd.Series(weights, index=covariance.index, name="weight")

    raise RuntimeError("风险平价未收敛，无法验证各 ETF 的风险贡献相等")


def select_momentum_etfs(
    prices: pd.DataFrame,
    signal_date: str,
    lookback: int = 126,
    top_n: int = 5,
) -> pd.Index:
    """按截至信号日的累计涨幅排名，返回前 top_n 只 ETF。"""
    # 只使用信号日及之前的价格，取最后 127 行对应 126 个收益区间。
    history = prices.sort_index().loc[:pd.Timestamp(signal_date)]
    window = history.tail(lookback + 1)

    if len(window) < lookback + 1:
        raise ValueError(f"动量排名至少需要 {lookback + 1} 个交易日的价格")

    # 所有 ETF 使用相同的起止日期。
    start_prices = window.iloc[0]
    end_prices = window.iloc[-1]

    # 两端都有有效价格的 ETF 才参加排名。
    valid = (
        np.isfinite(start_prices)
        & np.isfinite(end_prices)
        & (start_prices > 0)
        & (end_prices > 0)
    )

    if valid.sum() < top_n:
        raise ValueError(f"动量排名只有 {valid.sum()} 只 ETF 有有效价格，少于 {top_n} 只")

    momentum = end_prices[valid] / start_prices[valid] - 1

    # 涨幅相同时，按价格表原来的列顺序决定先后。
    ranking = momentum.sort_values(ascending=False, kind="stable")
    return ranking.head(top_n).index


if __name__ == "__main__":
    from pathlib import Path

    # 读取与 optimization.py 放在同一文件夹的价格表
    project_dir = Path(__file__).resolve().parent
    prices = pd.read_csv(
        project_dir / "etf_prices.csv",
        index_col="Date",
        parse_dates=["Date"],
        encoding="utf-8-sig",
    )

    signal_date = "2025-12-31"

    # 全池参数估计仍使用各 ETF 截至信号日的有效历史。
    returns = calculate_returns(prices)
    statistics = estimate_statistics(returns, end_date=signal_date)

    mu = statistics["mean_returns"]
    covariance = statistics["covariance"]

    # 用最近 126 个交易日的涨幅选出前 5 只。
    selected = select_momentum_etfs(
        prices,
        signal_date=signal_date,
        lookback=126,
        top_n=5,
    )

    # 从全池估计结果中取出入选 ETF 的均值和协方差子矩阵。
    selected_mu = mu.loc[selected]
    selected_covariance = covariance.loc[selected, selected]

    # 原来的四种组合用全池，动量版的四种组合用同一份入选名单。
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

    print("\n动量入选 ETF（按涨幅从高到低）：")
    print("、".join(selected))

    print("\n八种组合的目标权重：")
    print(weights.to_string(float_format=lambda x: f"{x:.4%}"))

    # 按权重表的 ETF 顺序排列均值和协方差，再转成数组参与计算
    asset_order = weights.index
    mean_array = mu.loc[asset_order].to_numpy()
    sigma = covariance.loc[asset_order, asset_order].to_numpy()

    # 计算每种组合的估计收益、方差和波动率
    comparison = pd.DataFrame(
        index=weights.columns,
        columns=["估计平均日收益率", "估计日方差", "估计日波动率"],
        dtype=float,
    )

    for name in weights.columns:
        w = weights[name].to_numpy()
        portfolio_return = mean_array @ w
        portfolio_variance = w @ sigma @ w

        comparison.loc[name] = [
            portfolio_return,
            portfolio_variance,
            np.sqrt(max(portfolio_variance, 0.0)),
        ]

    print("\n历史参数下的组合比较：")
    print(comparison.to_string(float_format=lambda x: f"{x:.8f}"))

    # 先算出风险贡献占比，再在下面的 checks 中使用它
    rp_weights = weights["风险平价"].to_numpy()
    risk_contributions = rp_weights * (sigma @ rp_weights)
    risk_shares = risk_contributions / risk_contributions.sum()
    target_risk_share = 1.0 / len(risk_shares)

    print("\n风险平价组合的风险贡献占比：")
    print(
        pd.Series(risk_shares, index=asset_order).to_string(
            float_format=lambda x: f"{x:.4%}"
        )
    )

    # 检查权重、收益约束、方差关系及风险平价条件
    weight_tol = 1e-8
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

    # 取出三个动量组合的估计收益和方差
    momentum_eq = comparison.loc["动量等权"]
    momentum_min = comparison.loc["动量最小方差"]
    momentum_constrained = comparison.loc["动量收益约束最小方差"]

    # 动量等权只对入选 ETF 平均分配资金
    momentum_eq_weights = weights.loc[selected, "动量等权"].to_numpy()
    target_share = 1.0 / len(selected)

    # 动量风险平价只在入选 ETF 中计算风险贡献占比
    momentum_rp_weights = weights.loc[selected, "动量风险平价"].to_numpy()
    momentum_sigma = selected_covariance.to_numpy()
    risk_contributions = momentum_rp_weights * (momentum_sigma @ momentum_rp_weights)
    total_variance = risk_contributions.sum()

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
        "风险平价组合的风险贡献占比相等": np.allclose(
            risk_shares, target_risk_share, rtol=0, atol=1e-6
        ),
        "动量等权对入选ETF平均配权": np.allclose(
            momentum_eq_weights, target_share, rtol=0, atol=weight_tol
        ),
        "动量最小方差的方差不高于动量等权": (
                momentum_min["估计日方差"]
                <= momentum_eq["估计日方差"] + variance_tol
        ),
        "动量最小方差的方差不高于动量收益约束组合": (
                momentum_min["估计日方差"]
                <= momentum_constrained["估计日方差"] + variance_tol
        ),
        "动量收益约束组合的估计收益不低于动量等权": (
                momentum_constrained["估计平均日收益率"]
                >= momentum_eq["估计平均日收益率"] - return_tol
        ),
        "动量收益约束组合的方差不高于动量等权": (
                momentum_constrained["估计日方差"]
                <= momentum_eq["估计日方差"] + variance_tol
        ),
        "动量风险平价的入选ETF风险贡献占比相等": (
                total_variance > 0
                and np.allclose(
            risk_contributions / total_variance,
            target_share,
            rtol=0,
            atol=1e-6)
        ),
    }

    print("\n检查结果：")
    for description, passed in checks.items():
        print(f"{description}：{'通过' if passed else '未通过'}")