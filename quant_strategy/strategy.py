"""
双均线交叉策略 (Backtrader 实现)

核心逻辑:
  买入: MA5 上穿 MA20 且 成交量 > 20日均量 × vol_mult
  卖出: MA5 下穿 MA20 / 止损 / 止盈 / 跟踪止盈

仓位管理:
  - 单股 ≤ 总资产 × max_single_pct
  - 总仓位 ≤ 总资产 × max_total_pct
"""

import backtrader as bt
import backtrader.feeds as btfeeds
import pandas as pd


class MACrossStrategy(bt.Strategy):
    params = dict(
        fast_period=5,          # 短期均线
        slow_period=20,         # 长期均线
        vol_period=20,          # 均量周期
        vol_mult=1.5,           # 成交量倍数阈值
        stop_loss=-0.08,        # 止损比例
        take_profit=0.15,       # 止盈比例
        trail_pct=0.05,         # 跟踪止盈回撤比例
        max_single_pct=0.20,    # 单股最大仓位
        max_total_pct=0.60,     # 总仓位上限
        printlog=False,         # 是否打印交易日志
    )

    def __init__(self):
        self.order_dict = {}          # {data: order}
        self.entry_prices = {}        # {data: 入场价}
        self.max_prices = {}          # {data: 持仓期最高价}
        self.trade_log = []           # 交易记录

        # 为每只股票创建指标
        self.indicators = {}
        for d in self.datas:
            ma_fast = bt.indicators.SMA(d.close, period=self.p.fast_period)
            ma_slow = bt.indicators.SMA(d.close, period=self.p.slow_period)
            vol_ma  = bt.indicators.SMA(d.volume, period=self.p.vol_period)

            cross = bt.indicators.CrossOver(ma_fast, ma_slow)

            self.indicators[d._name] = {
                "ma_fast": ma_fast,
                "ma_slow": ma_slow,
                "vol_ma":  vol_ma,
                "cross":   cross,
            }

    def log(self, txt, dt=None):
        if self.p.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f"[{dt}] {txt}")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            d = order.data
            if order.isbuy():
                self.entry_prices[d._name] = order.executed.price
                self.max_prices[d._name] = order.executed.price
                self.log(f"买入 {d._name} @ {order.executed.price:.2f}  "
                         f"数量:{order.executed.size:.0f}  "
                         f"手续费:{order.executed.comm:.2f}")
            elif order.issell():
                entry = self.entry_prices.get(d._name, 0)
                pnl_pct = (order.executed.price / entry - 1) * 100 if entry else 0
                self.log(f"卖出 {d._name} @ {order.executed.price:.2f}  "
                         f"盈亏:{pnl_pct:+.2f}%  "
                         f"手续费:{order.executed.comm:.2f}")
                self.entry_prices.pop(d._name, None)
                self.max_prices.pop(d._name, None)
            self.order_dict.pop(d._name, None)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.order_dict.pop(order.data._name, None)

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trade_log.append({
                "code":    trade.data._name,
                "pnl":     trade.pnl,
                "pnlcomm": trade.pnlcomm,
                "barlen":  trade.barlen,
            })

    def _get_total_position_value(self):
        """当前总持仓市值"""
        return sum(d.close[0] * abs(self.getposition(d).size) for d in self.datas)

    def _can_open_position(self, data):
        """检查是否可以开仓"""
        # 已有该股持仓
        if self.getposition(data).size > 0:
            return False
        # 有未完成订单
        if data._name in self.order_dict:
            return False
        # 总仓位检查
        total_val = self._get_total_position_value()
        total_equity = self.broker.getvalue()
        if total_equity > 0 and total_val / total_equity >= self.p.max_total_pct:
            return False
        return True

    def _calc_buy_size(self, data):
        """计算买入数量 (A股100股整数倍)"""
        price = data.close[0]
        total_equity = self.broker.getvalue()

        # 单股上限
        max_amount = total_equity * self.p.max_single_pct
        # 扣除已有持仓
        current_val = price * self.getposition(data).size
        available = max_amount - current_val

        # 总仓位剩余空间
        total_val = self._get_total_position_value()
        total_room = total_equity * self.p.max_total_pct - total_val
        available = min(available, total_room)

        if available <= 0 or price <= 0:
            return 0

        size = int(available / price / 100) * 100  # 100股整数倍
        return max(size, 0)

    def _check_stop_conditions(self, data):
        """
        检查止损/止盈/跟踪止盈
        Returns: (should_exit: bool, reason: str)
        """
        name = data._name
        pos = self.getposition(data)
        if pos.size <= 0:
            return False, ""

        current_price = data.close[0]
        entry = self.entry_prices.get(name, 0)
        if entry <= 0:
            return False, ""

        # 更新持仓最高价
        if name in self.max_prices:
            self.max_prices[name] = max(self.max_prices[name], current_price)

        pnl_pct = current_price / entry - 1

        # 1) 固定止损
        if pnl_pct <= self.p.stop_loss:
            return True, f"止损({pnl_pct:.1%})"

        # 2) 固定止盈
        if pnl_pct >= self.p.take_profit:
            return True, f"止盈({pnl_pct:.1%})"

        # 3) 跟踪止盈 (已盈利 > 5% 后，从最高点回撤超过 trail_pct)
        max_p = self.max_prices.get(name, entry)
        if max_p > entry * 1.05:  # 至少盈利5%后才启动跟踪
            drawdown = 1 - current_price / max_p
            if drawdown >= self.p.trail_pct:
                return True, f"跟踪止盈(高点回撤{drawdown:.1%})"

        return False, ""

    def next(self):
        for d in self.datas:
            name = d._name
            ind = self.indicators[name]

            # ── 持仓管理：先检查止损止盈 ──
            should_exit, reason = self._check_stop_conditions(d)
            if should_exit:
                pos = self.getposition(d)
                if pos.size > 0 and name not in self.order_dict:
                    self.log(f"⚠ {name} {reason}")
                    self.order_dict[name] = self.sell(data=d, size=pos.size)
                continue  # 止盈止损后本bar不做其他操作

            # ── 死叉卖出 ──
            if ind["cross"][0] < 0:  # 死叉
                pos = self.getposition(d)
                if pos.size > 0 and name not in self.order_dict:
                    self.log(f"📉 {name} 死叉卖出")
                    self.order_dict[name] = self.sell(data=d, size=pos.size)
                continue

            # ── 金叉 + 量能过滤 → 买入 ──
            if ind["cross"][0] > 0:  # 金叉
                # 成交量过滤
                vol_ma = ind["vol_ma"][0]
                if vol_ma > 0 and d.volume[0] < vol_ma * self.p.vol_mult:
                    continue  # 量能不足，跳过

                if self._can_open_position(d):
                    size = self._calc_buy_size(d)
                    if size > 0:
                        self.log(f"📈 {name} 金叉买入 (量比:{d.volume[0]/vol_ma:.1f}x)")
                        self.order_dict[name] = self.buy(data=d, size=size)


class AShareCommission(bt.CommInfoBase):
    """A股佣金方案: 买入万2.5, 卖出万2.5+千1印花税"""
    params = (
        ("commission", 0.00025),  # 万2.5
        ("stamp_duty", 0.001),    # 卖出千1印花税
        ("stocklike", True),
        ("commtype", bt.CommInfoBase.COMM_PERC),
        ("percabs", True),
    )

    def _getcommission(self, size, price, pseudoexec):
        val = abs(size) * price
        comm = val * self.p.commission
        if size < 0:  # 卖出加印花税
            comm += val * self.p.stamp_duty
        return comm


def create_cerebro(strategy_params: dict = None, printlog: bool = False) -> bt.Cerebro:
    """创建并配置 Cerebro 引擎"""
    cerebro = bt.Cerebro()

    # 策略参数
    params = strategy_params or {}
    params.setdefault("printlog", printlog)
    cerebro.addstrategy(MACrossStrategy, **params)

    # A股佣金 (买入万2.5 + 卖出万2.5+千1印花税)
    cerebro.broker.addcommissioninfo(AShareCommission())

    # 初始资金 100万
    cerebro.broker.setcash(1_000_000)

    # 滑点 0.1%
    cerebro.broker.set_slippage_perc(0.001)

    return cerebro


def load_data_to_cerebro(
    cerebro: bt.Cerebro,
    data_dict: dict[str, pd.DataFrame],
    fromdate: str = "2018-01-01",
    todate: str = "2025-12-31",
):
    """将DataFrame字典加载为Backtrader数据源"""
    for code, df in data_dict.items():
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df = df[(df.index >= fromdate) & (df.index <= todate)]

        data_feed = btfeeds.PandasData(
            dataname=df,
            datetime=None,  # 使用index
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
            openinterest=-1,
            name=code,
        )
        cerebro.adddata(data_feed)
