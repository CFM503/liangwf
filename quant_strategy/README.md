# A股双均线交叉量化策略回测系统

## 策略逻辑
- **入场**: MA5 上穿 MA20 (金叉) + 成交量 > 20日均量 × 1.5
- **出场**: MA5 下穿 MA20 (死叉) / 止损 -8% / 止盈 +15% / 跟踪止盈 (从最高点回撤5%)
- **仓位**: 单股 ≤ 20%, 总仓位 ≤ 60%

## 文件说明
| 文件 | 说明 |
|------|------|
| `data_generator.py` | 生成仿真A股行情数据 |
| `strategy.py` | 核心策略逻辑 (Backtrader) |
| `backtest_runner.py` | 回测引擎 + 绩效分析 |
| `optimizer.py` | 参数优化 |
| `run_all.py` | 一键运行入口 |

## 运行
```bash
cd quant_strategy
python run_all.py
```

## 依赖
- backtrader, pandas, numpy, matplotlib
