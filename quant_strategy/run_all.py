"""
一键运行入口
  1. 生成仿真数据
  2. 运行默认参数回测
  3. 参数优化 (可选)
  4. 输出风险分析
"""

import sys
import io
from pathlib import Path

# Windows 控制台 UTF-8 输出
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 确保输出目录存在
Path("./output").mkdir(exist_ok=True)


def print_risk_analysis():
    """打印策略风险分析"""
    print("\n")
    print("█" * 60)
    print("  ⚠️  策略潜在风险分析")
    print("█" * 60)

    risks = [
        ("1. 过拟合风险", [
            "均线周期 (5/20) 和量能倍数 (1.5x) 基于经验设定，",
            "在仿真数据上优化可能导致参数过度拟合历史模式。",
            "→ 对策: Walk-forward验证 / 样本外测试 / 参数平原检验",
        ]),
        ("2. 滞后性风险", [
            "均线本质是滞后指标，金叉/死叉信号出现在趋势确立之后。",
            "震荡市中频繁假突破 → 高频止损 → 快速磨损本金。",
            "→ 对策: 加入ADX趋势过滤 (ADX>25才交易) / 使用EMA替代SMA",
        ]),
        ("3. 流动性假设风险", [
            "回测假设按收盘价成交 + 0.1%滑点，实际中小盘股冲击成本更大。",
            "涨停无法买入、跌停无法卖出在回测中未完全模拟。",
            "→ 对策: 仅选日均成交额 > 5000万的标的 / 加入涨跌停过滤",
        ]),
        ("4. 单一市场环境风险", [
            "2018-2025包含牛市(2019-2021)、熊市(2022-2023)、震荡(2024-2025)。",
            "但A股政策市特征明显，未来可能出现模型未覆盖的极端场景。",
            "→ 对策: 加入沪深300指数做牛熊过滤 / 分市场环境回测",
        ]),
        ("5. 仓位管理的执行风险", [
            "20%单股上限 + 60%总仓位限制在实际操作中可能因价格波动被突破。",
            "T+1制度下当日买入无法卖出，止损存在一天延迟。",
            "→ 对策: 预留10%安全边际 / 使用T+1标志位延迟开仓确认",
        ]),
        ("6. 成本低估风险", [
            "当前仅计入佣金+印花税，未包含：",
            "  - 资金成本 (融券/打新占用)",
            "  - 信息延迟 (散户无法获得tick级数据)",
            "  - 情绪成本 (连续止损后的执行偏差)",
            "→ 对策: 将交易成本假设翻倍做压力测试",
        ]),
        ("7. 幸存者偏差", [
            "回测标的池使用当前存续股票，未包含已退市标的。",
            "2018年存在大量爆雷股 (康美、康得新等)。",
            "→ 对策: 使用全A股含退市标的的完整数据集",
        ]),
    ]

    for title, lines in risks:
        print(f"\n  🔴 {title}")
        for line in lines:
            print(f"     {line}")

    print("\n" + "█" * 60)


def print_param_suggestions():
    """打印参数优化建议"""
    print("\n")
    print("█" * 60)
    print("  💡 参数优化建议")
    print("█" * 60)

    suggestions = [
        ("均线周期组合", [
            "默认: MA5 × MA20",
            "推荐测试: MA3×MA10 (激进短线) / MA5×MA20 (默认) / MA10×MA30 (稳健)",
            "关键: 快线 < 3 会产生大量噪音信号; 慢线 > 60 信号太少",
        ]),
        ("量能过滤阈值", [
            "默认: 成交量 > 20日均量 × 1.5",
            "推荐测试: 1.2x (宽松) / 1.5x (默认) / 2.0x (严格)",
            "建议: 可加入放量持续天数 (连续2天放量再确认)",
        ]),
        ("止损止盈比例", [
            "默认: -8% 止损 / +15% 止盈",
            "推荐测试: ATR自适应止损 (2×ATR) 优于固定百分比",
            "跟踪止盈: 从最高点回撤5%离场，锁定利润",
        ]),
        ("仓位管理", [
            "默认: 单股20% / 总仓位60%",
            "可选: 凯利公式动态仓位 / 波动率倒数加权",
            "保守: 单股10% / 总仓位50% (降低集中度风险)",
        ]),
        ("增加策略维度", [
            "趋势过滤: ADX > 25 时才允许开仓",
            "大盘过滤: 沪深300在MA60之上才允许做多",
            "板块轮动: 按行业动量排名选前3个行业",
        ]),
    ]

    for title, lines in suggestions:
        print(f"\n  📌 {title}")
        for line in lines:
            print(f"     {line}")

    print("\n" + "█" * 60)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="双均线交叉策略回测系统")
    parser.add_argument("--optimize", action="store_true", help="运行参数优化")
    parser.add_argument("--log", action="store_true", help="打印交易日志")
    parser.add_argument("--fast-opt", action="store_true", help="快速优化 (缩小参数空间)")
    args = parser.parse_args()

    print("\n" + "▓" * 60)
    print("  A股双均线交叉量化策略回测系统")
    print("  MA5 × MA20 | 量能过滤 | 仓位管理 | 止损止盈")
    print("▓" * 60 + "\n")

    # ── Step 1: 默认参数回测 ──
    from backtest_runner import run_backtest
    result = run_backtest(printlog=args.log)

    # ── Step 2: 风险分析 ──
    print_risk_analysis()

    # ── Step 3: 参数优化建议 ──
    print_param_suggestions()

    # ── Step 4: 可选参数优化 ──
    if args.optimize or args.fast_opt:
        from optimizer import grid_optimize
        if args.fast_opt:
            # 快速版: 只搜核心参数
            grid_optimize(
                param_grid={
                    "fast_period": [3, 5, 8],
                    "slow_period": [15, 20, 30],
                    "vol_mult": [1.3, 1.5, 2.0],
                    "stop_loss": [-0.08],
                    "take_profit": [0.15],
                },
                top_n=10,
            )
        else:
            grid_optimize(top_n=10)
    else:
        print("\n💡 提示: 加 --optimize 运行全量参数优化, 或 --fast-opt 快速优化")

    print("\n✅ 全部完成! 结果保存在 ./output/ 目录")


if __name__ == "__main__":
    main()
