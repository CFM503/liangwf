#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║           XiaoLiangTrader — 校园股神量化系统              ║
║                                                          ║
║  "在宿舍里用一台笔记本，也能跑通量化交易的全流程"          ║
║                                                          ║
║  2008-2010，浙大，个人电脑，LightGBM + 双均线             ║
║  不依赖大模型，不依赖 GPU，纯粹的机器学习 + 技术分析       ║
║                                                          ║
║  声明: 仅供学习，不构成投资建议                            ║
╚══════════════════════════════════════════════════════════╝

用法:
    python main.py --scan             # 🔍 批量选股扫描（核心功能）
    python main.py --scan --all       # 🔍 全市场扫描（5000+ 只 A 股）
    python main.py --backtest         # 回测
    python main.py --train            # 训练 ML 模型
    python main.py --once             # 手动运行一次
    python main.py                    # 启动每日自动交易
    python main.py --stop             # 紧急停止
"""

import argparse
import signal
import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def cmd_scheduler(config_path):
    """启动定时调度"""
    try:
        import schedule
    except ImportError:
        print("请先安装: pip install schedule")
        sys.exit(1)

    from config.settings import load_config
    from bot.scheduler import TradingAgent
    from utils.logger import get_logger

    log = get_logger("xlt.main")
    cfg = load_config(config_path)
    agent = TradingAgent(cfg)

    log.info("=" * 60)
    log.info("🏫 XiaoLiangTrader 定时调度启动")
    log.info(f"   每日运行: {cfg.run_time}")
    log.info(f"   标的: {cfg.stocks}")
    log.info(f"   ML: {'✓ ' + cfg.ml.model_type if cfg.ml.enabled else '✗'}")
    log.info("   Ctrl+C 退出")
    log.info("=" * 60)

    schedule.every().day.at(cfg.run_time).do(agent.run_daily)

    def graceful_exit(signum, frame):
        log.info("正在退出...")
        agent.risk.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, graceful_exit)
    signal.signal(signal.SIGTERM, graceful_exit)

    import time
    while True:
        schedule.run_pending()
        time.sleep(30)


def cmd_once(config_path):
    from config.settings import load_config
    from bot.scheduler import TradingAgent
    cfg = load_config(config_path)
    TradingAgent(cfg).run_daily()


def cmd_backtest(config_path):
    from config.settings import load_config
    from backtest.engine import BacktestEngine
    cfg = load_config(config_path)
    engine = BacktestEngine(initial_cash=cfg.initial_cash)
    report = engine.run(
        stock_codes=cfg.stocks,
        start_date=cfg.data_start,
        fast_period=cfg.strategy.fast_period,
        slow_period=cfg.strategy.slow_period,
        vol_mult=cfg.strategy.vol_mult,
    )
    engine.print_report(report)


def cmd_train(config_path):
    """训练 ML 模型"""
    from config.settings import load_config
    from data.fetcher import fetch_stock
    from ml_model.predictor import MLPredictor

    cfg = load_config(config_path)
    model_type = cfg.ml.model_type
    print(f"正在训练 {model_type} 模型...")

    predictor = MLPredictor(
        model_type=model_type,
        forward_days=cfg.ml.forward_days,
        threshold=cfg.ml.threshold,
        n_estimators=cfg.ml.n_estimators,
        max_depth=cfg.ml.max_depth,
    )

    # 逐只股票训练（简单起见用第一只）
    symbol = cfg.stocks[0]
    df = fetch_stock(symbol, cfg.data_start, use_cache=True)
    if df.empty:
        print(f"无数据: {symbol}")
        return

    metrics = predictor.train(df)
    predictor.save()

    print(f"\n{'='*45}")
    print(f"  模型: {metrics['model_type']}")
    print(f"  训练集准确率: {metrics['train_accuracy']:.2%}")
    print(f"  验证集准确率: {metrics['val_accuracy']:.2%}")
    print(f"  训练样本: {metrics['train_samples']}")
    print(f"  验证样本: {metrics['val_samples']}")
    if metrics.get("top_features"):
        print(f"  Top5 特征:")
        for i, (feat, imp) in enumerate(list(metrics["top_features"].items())[:5]):
            print(f"    {i+1}. {feat} ({imp:.0f})")
    print(f"{'='*45}")


def cmd_status(config_path):
    from config.settings import load_config
    from bot.scheduler import TradingAgent
    cfg = load_config(config_path)
    status = TradingAgent(cfg).get_status()
    print(json.dumps(status, indent=2, ensure_ascii=False, default=str))


def cmd_scan(config_path, pool_override=None, scan_all=False):
    """
    批量选股扫描 — 这是"校园股神"的核心玩法
    扫描整个股票池，输出候选列表 + 信号强度 + 建议仓位
    """
    from config.settings import load_config
    from data.fetcher import DEFAULT_POOL
    from strategy.screener import StockScreener
    from ml_model.predictor import MLPredictor

    cfg = load_config(config_path)

    # 股票池：命令行指定 > 配置文件 > 默认池
    pool = pool_override or cfg.stocks or DEFAULT_POOL

    # ML 模型（可选）
    predictor = None
    if cfg.ml.enabled:
        predictor = MLPredictor(
            model_type=cfg.ml.model_type,
            forward_days=cfg.ml.forward_days,
            threshold=cfg.ml.threshold,
        )
        models = predictor.list_saved_models()
        if models:
            predictor.load(models[-1])
        else:
            print("⚠️  ML 已启用但无模型，先运行 --train，或仅用技术面选股")
            predictor = None

    screener = StockScreener(
        fast_period=cfg.strategy.fast_period,
        slow_period=cfg.strategy.slow_period,
        vol_mult=cfg.strategy.vol_mult,
        ml_predictor=predictor,
        ml_confidence=cfg.ml.confidence_threshold if cfg.ml.enabled else 0,
        min_tech_score=cfg.screener.min_tech_score if hasattr(cfg, 'screener') else 40.0,
        top_n=cfg.screener.top_n if hasattr(cfg, 'screener') else 10,
        max_position_pct=cfg.screener.max_position_pct if hasattr(cfg, 'screener') else 20.0,
    )

    if scan_all:
        # 全市场扫描模式
        print("🔍 全市场扫描模式（5000+ 只 A 股）")
        if predictor:
            print(f"   ML 模型: {cfg.ml.model_type} (阈值 {cfg.ml.confidence_threshold})")
        else:
            print(f"   ML: 未启用，仅技术面选股")
        results = screener.scan_all()
    else:
        # 指定股票池扫描
        print(f"🔍 扫描 {len(pool)} 只股票...")
        if predictor:
            print(f"   ML 模型: {cfg.ml.model_type} (阈值 {cfg.ml.confidence_threshold})")
        else:
            print(f"   ML: 未启用，仅技术面选股")
        results = screener.scan(pool)

    screener.print_results(results)


def cmd_encrypt(config_path):
    from utils.crypto import encrypt_yaml_value
    import yaml

    path = Path(config_path)
    if not path.exists():
        print(f"配置文件不存在: {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    changed = False
    for section in ["email", "broker"]:
        if section in cfg:
            for key, val in cfg[section].items():
                if isinstance(val, str) and val and not val.startswith("ENC:"):
                    cfg[section][key] = encrypt_yaml_value(val)
                    changed = True
                    print(f"  [加密] {section}.{key}")

    if changed:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        print(f"已加密保存: {path}")
    else:
        print("所有敏感字段已加密")


def cmd_stop():
    from bot.risk import RiskManager
    risk = RiskManager()
    risk.activate_kill_switch()
    print("🚨 Kill Switch 已激活！交易已停止。")
    print(f"   删除 {risk.kill_switch_file} 或 --resume 解除")


def cmd_resume():
    from bot.risk import RiskManager
    risk = RiskManager()
    risk.deactivate_kill_switch()
    print("✅ Kill Switch 已解除")


def main():
    parser = argparse.ArgumentParser(
        description="XiaoLiangTrader — 校园股神量化系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --scan              # 🔍 扫描指定股票池
  python main.py --scan --all        # 🔍 全市场扫描（5000+ 只 A 股）
  python main.py --scan --pool 600519 300750 002594  # 指定股票池
  python main.py --backtest          # 回测
  python main.py --train             # 训练 ML 模型
  python main.py --once              # 手动运行一次
  python main.py                     # 启动每日自动交易
  python main.py --stop              # 紧急停止
        """,
    )
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--scan", action="store_true", help="🔍 批量选股扫描")
    parser.add_argument("--all", action="store_true", dest="scan_all", help="全市场扫描（5000+ 只 A 股）")
    parser.add_argument("--pool", nargs="+", default=None, help="自定义股票池（代码列表）")
    parser.add_argument("--once", action="store_true", help="单次运行")
    parser.add_argument("--backtest", action="store_true", help="运行回测")
    parser.add_argument("--train", action="store_true", help="训练 ML 模型")
    parser.add_argument("--status", action="store_true", help="查看状态")
    parser.add_argument("--encrypt", action="store_true", help="加密配置")
    parser.add_argument("--stop", action="store_true", help="紧急停止")
    parser.add_argument("--resume", action="store_true", help="恢复交易")

    args = parser.parse_args()
    config_path = args.config or str(ROOT / "config" / "config.yaml")

    if args.encrypt:
        cmd_encrypt(config_path)
    elif args.stop:
        cmd_stop()
    elif args.resume:
        cmd_resume()
    elif args.scan:
        cmd_scan(config_path, pool_override=args.pool, scan_all=args.scan_all)
    elif args.backtest:
        cmd_backtest(config_path)
    elif args.train:
        cmd_train(config_path)
    elif args.status:
        cmd_status(config_path)
    elif args.once:
        cmd_once(config_path)
    else:
        cmd_scheduler(config_path)


if __name__ == "__main__":
    main()
