from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib.ticker import PercentFormatter

from optimization import calculate_returns

from backtest import (
    make_rebalance_schedule,
    calculate_target_weights,
    run_backtest,
    calculate_performance,
)


# ==================== 一、统一绘图设置 ====================

# 中文字体
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "DejaVu Sans",
]

# 正确显示负号
plt.rcParams["axes.unicode_minus"] = False

# 图片清晰度
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 200

# 字体大小
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 11

# 隐藏上边框和右边框
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# 图例不显示边框
plt.rcParams["legend.frameon"] = False


def format_date_axis(ax):
    """统一设置日期横轴和网格。"""

    locator = mdates.AutoDateLocator(
        minticks=5,
        maxticks=9,
    )

    ax.xaxis.set_major_locator(locator)

    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(locator)
    )

    ax.set_xlabel("日期")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    # 曲线两端不保留额外空白
    ax.margins(x=0)


# ==================== 二、绩效比较表 ====================

def plot_performance_table(performance):
    """将指标按行、组合按列排列，方便插入Word。"""

    columns = [
        "期末净值",
        "累计收益率",
        "年化收益率",
        "年化波动率",
        "夏普比率",
        "Calmar比率",
        "最大回撤",
        "累计费用",
    ]

    percentage_columns = [
        "累计收益率",
        "年化收益率",
        "年化波动率",
        "最大回撤",
    ]

    display = performance[columns].copy().astype(object)

    for strategy in display.index:
        for column in columns:

            value = performance.loc[strategy, column]

            if pd.isna(value):
                text = "—"

            elif column in percentage_columns:
                text = f"{value:.2%}"

            elif column == "期末净值":
                text = f"{value:.4f}"

            elif column == "累计费用":
                text = f"{value:,.2f}"

            else:
                text = f"{value:.3f}"

            display.loc[strategy, column] = text

    # 转置后：每行是一个指标，每列是一种组合
    table_data = display.T

    cell_text = [
        [
            "累计费用（元）" if indicator == "累计费用" else indicator
        ] + table_data.loc[indicator].tolist()
        for indicator in table_data.index
    ]

    # 长组合名称换行显示
    strategy_labels = [
        name.replace(
            "等权收益约束最小方差",
            "等权收益约束\n最小方差",
        )
        for name in table_data.columns
    ]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=["指标"] + strategy_labels,
        cellLoc="center",
        colWidths=[0.28, 0.22, 0.22, 0.28],
        bbox=[0, 0, 1, 1],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)

    for (row, column), cell in table.get_celld().items():

        cell.set_edgecolor("#DEE4E8")

        if row == 0:
            cell.set_facecolor("#466174")
            cell.set_text_props(
                color="white",
                weight="bold",
            )

        else:
            cell.set_facecolor(
                "#F1F5F7" if row % 2 else "white"
            )

    # 增加表头高度，容纳两行组合名称
    for column in range(4):
        table[(0, column)].set_height(0.08)

    fig.subplots_adjust(
        left=0.01,
        right=0.99,
        top=0.99,
        bottom=0.01,
    )

    print("\n绩效比较表：")
    print(display.to_string())

    return fig


# ==================== 三、净值曲线 ====================

def plot_nav(backtest_results, strategy_colors):
    """绘制三种组合扣费后的净值曲线。"""

    fig, ax = plt.subplots(figsize=(12, 5))

    for strategy, result in backtest_results.items():

        nav = (
            result["每日记录"]["净值"]
            .sort_index()
        )

        ax.plot(
            nav.index,
            nav,
            label=strategy,
            color=strategy_colors[strategy],
            linewidth=1.8,
        )

    # 初始净值基准
    ax.axhline(
        y=1,
        color="#9AA3AA",
        linestyle="--",
        linewidth=0.9,
    )

    ax.set_title("三种组合的净值比较（扣费后）")
    ax.set_ylabel("净值（初始资金 = 1）")

    format_date_axis(ax)

    ax.legend(loc="best")
    fig.tight_layout()

    return fig


# ==================== 四、回撤曲线 ====================

def plot_drawdown(backtest_results, strategy_colors):
    """绘制相对于各自历史最高净值的回撤。"""

    fig, ax = plt.subplots(figsize=(12, 5))

    for strategy, result in backtest_results.items():

        nav = (
            result["每日记录"]["净值"]
            .sort_index()
        )

        # 截至当天的历史最高净值
        running_peak = nav.cummax()

        # 每日回撤
        drawdown = nav / running_peak - 1

        ax.plot(
            drawdown.index,
            drawdown,
            label=strategy,
            color=strategy_colors[strategy],
            linewidth=1.6,
        )

    ax.axhline(
        y=0,
        color="#9AA3AA",
        linewidth=0.9,
    )

    ax.yaxis.set_major_formatter(
        PercentFormatter(xmax=1)
    )

    ax.set_title("三种组合的回撤比较")
    ax.set_ylabel("回撤")

    format_date_axis(ax)

    ax.legend(loc="best")
    fig.tight_layout()

    return fig


# ==================== 五、实际资产配置图 ====================

def plot_allocation(strategy, result):
    """按平均实际权重选出前五只ETF，其余合并。"""

    # 每日实际权重，而不是调仓目标权重
    weights = result["实际权重"].sort_index().copy()

    # 从首次建仓日开始，排除建仓前全为现金的基准日
    first_execution = (
        result["交易记录"].sort_index().index[0]
    )

    weights = weights.loc[first_execution:]

    # 每列求平均：得到每只ETF在建仓后期间的平均实际权重
    average_weights = weights.mean().sort_values(
        ascending=False,
        kind="stable",
    )

    # 每个组合分别选出平均权重最高的五只
    top_assets = average_weights.head(5).index

    # 按平均权重排名，保留这五只的每日权重
    grouped_weights = weights[top_assets].copy()

    # 其余ETF的名称
    other_assets = [
        asset
        for asset in weights.columns
        if asset not in top_assets
    ]

    # 将其余ETF合并为一层
    grouped_weights["其余ETF合计"] = (
        weights[other_assets].sum(axis=1)
    )

    # 前五只按平均权重排名依次分配颜色
    colors = [
        "#0066FF",  # 第一名：亮蓝
        "#FF8800",  # 第二名：亮橙
        "#00CC44",  # 第三名：亮绿
        "#AA33FF",  # 第四名：亮紫
        "#FF3344",  # 第五名：亮红
        "#E0E0E0",  # 其余ETF：浅灰
    ]

    fig, ax = plt.subplots(figsize=(12, 5.5))

    ax.stackplot(
        grouped_weights.index,
        grouped_weights.to_numpy().T,
        labels=grouped_weights.columns,
        colors=colors,
        alpha=1.0,
        linewidth=0.3,
        edgecolor="white",
    )

    ax.set_ylim(0, 1)

    ax.yaxis.set_major_formatter(
        PercentFormatter(xmax=1)
    )

    ax.set_title(f"{strategy}：每日实际资产配置")
    ax.set_ylabel("占总资产的权重")

    format_date_axis(ax)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
    )

    fig.text(
        0.08,
        0.02,
        "主要五只按建仓后平均实际权重从高到低选取；"
        "其余ETF合并为灰色。颜色表示排名，资产名称以图例为准。",
        fontsize=9,
        color="#667580",
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    return fig


# ==================== 六、调仓换手率图 ====================

def plot_turnover(backtest_results, strategy_colors):
    """比较每次交易的双边换手率，包括首次建仓。"""

    turnover_records = {}

    for strategy, result in backtest_results.items():

        trades = result["交易记录"].sort_index()

        turnover_records[strategy] = (
            trades["买卖金额合计"]
            / trades["调仓前总资产"]
        )

    # 行表示执行日，列表示组合
    turnover = pd.DataFrame(
        turnover_records
    ).sort_index()

    fig, ax = plt.subplots(figsize=(13, 5))

    x = np.arange(len(turnover))

    strategy_count = len(turnover.columns)

    # 同一天全部柱子的总宽度为0.8
    width = 0.8 / strategy_count

    for i, strategy in enumerate(turnover.columns):

        positions = (
            x
            + (i - (strategy_count - 1) / 2) * width
        )

        ax.bar(
            positions,
            turnover[strategy],
            width=width,
            label=strategy,
            color=strategy_colors[strategy],
        )

    labels = [
        date.strftime("%Y-%m-%d")
        for date in turnover.index
    ]

    labels[0] += "\n首次建仓"

    ax.set_xticks(x)

    ax.set_xticklabels(
        labels,
        rotation=35,
        ha="right",
    )

    ax.yaxis.set_major_formatter(
        PercentFormatter(xmax=1)
    )

    ax.set_title("各次交易双边换手率比较（含首次建仓）")
    ax.set_xlabel("实际执行日期")
    ax.set_ylabel("双边换手率")

    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    ax.legend(loc="best")

    fig.text(
        0.08,
        0.02,
        "双边换手率 =（买入金额 + 卖出金额）/ 调仓前总资产；"
        "三种组合的调仓日期相同，主要比较每次调整幅度。",
        fontsize=9,
        color="#667580",
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    return fig


# ==================== 七、统一绘图并保存 ====================

def plot_all(
    backtest_results,
    performance,
    output_dir,
):
    """生成五类图表，共七张图片。"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 三种组合在净值、回撤和换手率图中的固定颜色
    strategy_colors = {
        "等权": "#0066FF",  # 亮蓝
        "最小方差": "#FF8800",  # 亮橙
        "等权收益约束最小方差": "#00CC44",  # 亮绿
    }

    figures = {}

    # 第一类：绩效表
    figures["01_performance"] = plot_performance_table(
        performance
    )

    # 第二类：净值曲线
    figures["02_nav"] = plot_nav(
        backtest_results,
        strategy_colors,
    )

    # 第三类：回撤曲线
    figures["03_drawdown"] = plot_drawdown(
        backtest_results,
        strategy_colors,
    )

    # 第四类：每个组合各一张实际资产配置图
    for number, (strategy, result) in enumerate(
        backtest_results.items(),
        start=1,
    ):
        figures[f"04_allocation_{number}"] = (
            plot_allocation(
                strategy,
                result,
            )
        )

    # 第五类：调仓换手率
    figures["05_turnover"] = plot_turnover(
        backtest_results,
        strategy_colors,
    )

    # 保存图片
    for filename, fig in figures.items():

        fig.savefig(
            output_dir / f"{filename}.png",
            bbox_inches="tight",
        )

    print(f"\n图片已保存到：{output_dir.resolve()}")

    # 显示全部图片
    plt.show()


# ==================== 八、运行入口 ====================

if __name__ == "__main__":

    # 当前文件所在目录
    project_dir = Path(__file__).resolve().parent

    # 读取已经下载好的价格数据
    prices = pd.read_csv(
        project_dir / "etf_prices.csv",
        index_col="Date",
        parse_dates=["Date"],
        encoding="utf-8-sig",
    )

    # 生成调仓计划
    schedule = make_rebalance_schedule(prices)

    # 计算每日简单收益率
    returns = calculate_returns(prices)

    # 计算各期目标权重
    target_weights = calculate_target_weights(
        returns=returns,
        schedule=schedule,
    )

    # 运行每日回测
    backtest_results = run_backtest(
        prices=prices,
        schedule=schedule,
        target_weights=target_weights,
        initial_capital=100000.0,
        fee_rate=0.0005,
    )

    # 计算绩效指标
    performance = calculate_performance(
        backtest_results=backtest_results,
        annual_risk_free_rate=0.02,
    )

    # 绘制、保存并显示全部图表
    plot_all(
        backtest_results=backtest_results,
        performance=performance,
        output_dir=project_dir / "figures",
    )