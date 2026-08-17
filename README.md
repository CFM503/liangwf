# LiangWF — A股短线量化交易与信号系统

> 每日收盘后运行、输出"明日/短期大概率上涨 A 股候选列表"的量化信号系统。

---

## 📁 仓库目录结构

```
liangwf/
├── XiaoLiangTrader/          # 🚀 主生产系统（策略 + ML预测 + 每日运行 + 风控）
│   ├── main.py               # 统一运行入口 (--scan / --train / --backtest / --once)
│   ├── data/                 # 4级高可用容灾数据层 (Tencent/Akshare/Sina/合成)
│   ├── ml_model/             # 36 因子量化特征库 + ML预测器 (LightGBM/XGBoost)
│   ├── strategy/             # 选股打分引擎 + 均线动量策略
│   ├── backtest/             # 真实约束回测引擎 (严格 T+1 / 涨跌停 / 停牌冻结)
│   ├── bot/                  # 每日自动化调度编排 + 风控 (Kill Switch) + 邮件通知
│   ├── config/               # 系统配置文件 (config.yaml)
│   └── utils/                # 日志与加密工具
├── tests/                    # 🧪 自动化单元测试套件 (11/11 全通过)
│   ├── test_data_fetcher.py              # 数据层多源与涨跌停测试
│   ├── test_features_no_future_leak.py   # 特征工程与零未来函数扰动测试
│   └── test_xlt_backtest_realistic.py    # 真实交易约束回测测试
├── archive/                  # 📦 历史代码归档库
│   ├── trading_agent/        # 早期架构设计与配置参考（已归档，统一由 XiaoLiangTrader 承载）
│   └── legacy_scripts/       # 早期单策略原型脚本 (dual_ma_strategy, vectorbt 等)
├── CHANGELOG.md              # 📋 详细版本迭代日志
├── plan.md                   # 📝 项目改造规划
├── requirements.txt          # 依赖列表
├── DEPLOY.md                 # 🚀 操作系统级开机自启与后台定时服务部署指南 (Windows/Mac/Linux)
└── README.md
```

---

## 🚀 快速开始

```bash
# 1. 安装项目依赖
pip install -r requirements.txt

# 2. 启动本地 Streamlit Web 可视化看板
python -m streamlit run XiaoLiangTrader/webapp.py

# 3. 每日盘后执行短线信号扫描与通知
python XiaoLiangTrader/scripts/daily_signal.py --notify --notify-dingtalk

# 4. 实盘信号事后复盘检验 (统计真实胜率与收益)
python XiaoLiangTrader/scripts/review_live_signals.py

# 5. 运行全套自动化测试套件 (25/25 passed)
python -X utf8 -m unittest discover -s tests -p "test_*.py"
```

详细后台无人值守开机自启部署请参见 [`DEPLOY.md`](DEPLOY.md)。
详细模块文档与配置说明请参见 [`XiaoLiangTrader/README.md`](XiaoLiangTrader/README.md)。
