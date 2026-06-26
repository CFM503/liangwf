"""
订单执行器 — 模拟盘（SQLite 持久化）+ 实盘接口（预留）
"""

import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .logger import log
from .strategy import Signal, Action


@dataclass
class OrderResult:
    success: bool
    order_id: str = ""
    symbol: str = ""
    action: str = ""
    price: float = 0.0
    size: int = 0
    amount: float = 0.0
    fee: float = 0.0
    message: str = ""
    timestamp: str = ""


class BaseExecutor(ABC):
    """执行器基类"""

    @abstractmethod
    def execute(self, signal: Signal) -> OrderResult:
        ...

    @abstractmethod
    def get_position(self, symbol: str) -> int:
        ...

    @abstractmethod
    def get_buy_price(self, symbol: str) -> float:
        ...

    @abstractmethod
    def get_max_price(self, symbol: str) -> float:
        ...

    @abstractmethod
    def get_cash(self) -> float:
        ...

    @abstractmethod
    def get_total_value(self) -> float:
        ...

    @abstractmethod
    def get_positions_summary(self) -> list[dict]:
        ...


class SimulatorExecutor(BaseExecutor):
    """
    模拟执行器 — 全部状态持久化到 SQLite
    重启后自动恢复持仓和资金状态
    """

    COMMISSION_RATE = 0.00025  # 佣金万2.5
    STAMP_TAX = 0.001          # 印花税千分之一（仅卖出）
    SLIPPAGE = 0.001           # 滑点 0.1%

    def __init__(self, db_path: str = "trading_agent.db", initial_cash: float = 1_000_000):
        self.db_path = db_path
        self.initial_cash = initial_cash
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # 账户表
        c.execute("""
            CREATE TABLE IF NOT EXISTS account (
                key TEXT PRIMARY KEY,
                value REAL
            )
        """)
        # 初始化资金（如果不存在）
        c.execute(
            "INSERT OR IGNORE INTO account (key, value) VALUES ('cash', ?)",
            (self.initial_cash,),
        )

        # 持仓表
        c.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                size INTEGER DEFAULT 0,
                buy_price REAL DEFAULT 0,
                max_price REAL DEFAULT 0,
                updated_at TEXT
            )
        """)

        # 订单记录
        c.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                symbol TEXT,
                action TEXT,
                price REAL,
                size INTEGER,
                amount REAL,
                fee REAL,
                reason TEXT,
                created_at TEXT
            )
        """)

        conn.commit()
        conn.close()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def execute(self, signal: Signal) -> OrderResult:
        if signal.action == Action.HOLD:
            return OrderResult(False, message="HOLD，不执行")

        if signal.action == Action.BUY:
            return self._execute_buy(signal)
        elif signal.action == Action.SELL:
            return self._execute_sell(signal)

        return OrderResult(False, message="未知操作")

    def _execute_buy(self, signal: Signal) -> OrderResult:
        conn = self._conn()
        c = conn.cursor()

        try:
            # 获取当前资金
            c.execute("SELECT value FROM account WHERE key='cash'")
            cash = c.fetchone()[0]

            # 模拟滑点
            price = signal.price * (1 + self.SLIPPAGE)

            # 计算可买数量（A股100股整数，单股≤20%总资产）
            c.execute("SELECT value FROM account WHERE key='cash'")
            total_value = self._get_total_value(conn)

            max_amount = total_value * 0.20  # 单股最大仓位
            available = min(cash * 0.95, max_amount)  # 留5%余量
            size = int(available / price / 100) * 100

            if size < 100:
                return OrderResult(False, symbol=signal.symbol, message="资金不足或仓位已满")

            amount = price * size
            fee = max(amount * self.COMMISSION_RATE, 5)  # 最低5元
            total_cost = amount + fee

            if total_cost > cash:
                size = int((cash - 5) / price / 100) * 100
                if size < 100:
                    return OrderResult(False, symbol=signal.symbol, message="资金不足")
                amount = price * size
                fee = max(amount * self.COMMISSION_RATE, 5)
                total_cost = amount + fee

            # 扣减资金
            c.execute(
                "UPDATE account SET value = value - ? WHERE key = 'cash'",
                (total_cost,),
            )

            # 更新持仓
            now = datetime.now().isoformat()
            c.execute("""
                INSERT INTO positions (symbol, size, buy_price, max_price, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    size = size + ?,
                    buy_price = (buy_price * size + ? * ?) / (size + ?),
                    max_price = MAX(max_price, ?),
                    updated_at = ?
            """, (
                signal.symbol, size, price, price, now,
                size, price, size, size,
                price, now,
            ))

            # 记录订单
            order_id = f"B_{signal.symbol}_{int(time.time())}"
            c.execute("""
                INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (order_id, signal.symbol, "BUY", price, size, amount, fee,
                  signal.reason, now))

            conn.commit()
            log.info(f"[执行] 买入 {signal.symbol} @ {price:.2f} x {size} = {amount:,.0f} (手续费{fee:.0f})")

            return OrderResult(
                True, order_id, signal.symbol, "BUY", price, size, amount, fee,
                signal.reason, now,
            )

        except Exception as e:
            conn.rollback()
            log.error(f"[执行] 买入失败 {signal.symbol}: {e}")
            return OrderResult(False, symbol=signal.symbol, message=str(e))
        finally:
            conn.close()

    def _execute_sell(self, signal: Signal) -> OrderResult:
        conn = self._conn()
        c = conn.cursor()

        try:
            # 查持仓
            c.execute("SELECT size, buy_price FROM positions WHERE symbol = ?", (signal.symbol,))
            row = c.fetchone()
            if not row or row[0] <= 0:
                return OrderResult(False, symbol=signal.symbol, message="无持仓")

            size = row[0]
            buy_price = row[1]

            # 模拟滑点（卖出方向）
            price = signal.price * (1 - self.SLIPPAGE)
            amount = price * size
            commission = max(amount * self.COMMISSION_RATE, 5)
            stamp_tax = amount * self.STAMP_TAX
            fee = commission + stamp_tax
            net_amount = amount - fee

            # 回笼资金
            c.execute(
                "UPDATE account SET value = value + ? WHERE key = 'cash'",
                (net_amount,),
            )

            # 清空持仓
            c.execute("DELETE FROM positions WHERE symbol = ?", (signal.symbol,))

            # 记录订单
            now = datetime.now().isoformat()
            order_id = f"S_{signal.symbol}_{int(time.time())}"
            pnl_pct = (price / buy_price - 1) * 100 if buy_price > 0 else 0
            reason = f"{signal.reason} | 盈亏{pnl_pct:+.2f}%"
            c.execute("""
                INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (order_id, signal.symbol, "SELL", price, size, amount, fee,
                  reason, now))

            conn.commit()
            log.info(
                f"[执行] 卖出 {signal.symbol} @ {price:.2f} x {size} = {amount:,.0f} "
                f"(手续费{fee:.0f}, 盈亏{pnl_pct:+.2f}%)"
            )

            return OrderResult(
                True, order_id, signal.symbol, "SELL", price, size, amount, fee,
                reason, now,
            )

        except Exception as e:
            conn.rollback()
            log.error(f"[执行] 卖出失败 {signal.symbol}: {e}")
            return OrderResult(False, symbol=signal.symbol, message=str(e))
        finally:
            conn.close()

    def get_position(self, symbol: str) -> int:
        conn = self._conn()
        c = conn.cursor()
        c.execute("SELECT size FROM positions WHERE symbol = ?", (symbol,))
        row = c.fetchone()
        conn.close()
        return row[0] if row else 0

    def get_buy_price(self, symbol: str) -> float:
        conn = self._conn()
        c = conn.cursor()
        c.execute("SELECT buy_price FROM positions WHERE symbol = ?", (symbol,))
        row = c.fetchone()
        conn.close()
        return row[0] if row else 0.0

    def get_max_price(self, symbol: str) -> float:
        conn = self._conn()
        c = conn.cursor()
        c.execute("SELECT max_price FROM positions WHERE symbol = ?", (symbol,))
        row = c.fetchone()
        conn.close()
        return row[0] if row else 0.0

    def update_max_price(self, symbol: str, current_price: float):
        """更新持仓最高价"""
        conn = self._conn()
        c = conn.cursor()
        c.execute(
            "UPDATE positions SET max_price = MAX(max_price, ?) WHERE symbol = ?",
            (current_price, symbol),
        )
        conn.commit()
        conn.close()

    def get_cash(self) -> float:
        conn = self._conn()
        c = conn.cursor()
        c.execute("SELECT value FROM account WHERE key='cash'")
        val = c.fetchone()[0]
        conn.close()
        return val

    def _get_total_value(self, conn=None) -> float:
        should_close = False
        if conn is None:
            conn = self._conn()
            should_close = True
        c = conn.cursor()
        c.execute("SELECT value FROM account WHERE key='cash'")
        cash = c.fetchone()[0]
        c.execute("SELECT SUM(size * buy_price) FROM positions")
        pos_val = c.fetchone()[0] or 0
        if should_close:
            conn.close()
        return cash + pos_val

    def get_total_value(self) -> float:
        return self._get_total_value()

    def get_positions_summary(self) -> list[dict]:
        conn = self._conn()
        c = conn.cursor()
        c.execute("SELECT symbol, size, buy_price, max_price FROM positions WHERE size > 0")
        rows = c.fetchall()
        conn.close()
        return [
            {"symbol": r[0], "size": r[1], "buy_price": r[2], "max_price": r[3]}
            for r in rows
        ]

    def get_recent_orders(self, limit: int = 20) -> list[dict]:
        conn = self._conn()
        c = conn.cursor()
        c.execute(
            "SELECT order_id, symbol, action, price, size, amount, fee, reason, created_at "
            "FROM orders ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        rows = c.fetchall()
        conn.close()
        return [
            {
                "order_id": r[0], "symbol": r[1], "action": r[2],
                "price": r[3], "size": r[4], "amount": r[5],
                "fee": r[6], "reason": r[7], "created_at": r[8],
            }
            for r in rows
        ]


# ═══════════════════════════════════════════════
# 实盘执行器（预留接口）
# ═══════════════════════════════════════════════
class LiveExecutor(BaseExecutor):
    """
    实盘执行器 — 需要对接券商 API

    接入指南：
    ────────────────────────────────────────
    1. 东方财富/QMT: pip install xtquant
    2. 同花顺: 需要 iFinD SDK
    3. 通用方案: easytrader 库（支持雪球/华泰/银河等）

    实现时需：
    - 在 __init__ 中连接券商
    - execute() 调用真实下单 API
    - 考虑订单确认回调、超时处理
    """

    def __init__(self, api_key: str = "", api_secret: str = "", account_id: str = ""):
        if not api_key:
            raise NotImplementedError(
                "实盘执行器需要配置券商 API。\n"
                "推荐方案：\n"
                "  1. easytrader (pip install easytrader) — 支持华泰/银河/雪球\n"
                "  2. xtquant (QMT) — 量化专用\n"
                "  3. 自行实现 BaseExecutor 接口\n"
                "请在 config.yaml 中配置 broker.api_key 等参数"
            )
        self.api_key = api_key
        self.api_secret = api_secret
        self.account_id = account_id
        # TODO: 连接券商
        raise NotImplementedError("实盘接口待实现，请继承 BaseExecutor 并实现 execute()")

    def execute(self, signal: Signal) -> OrderResult:
        raise NotImplementedError

    def get_position(self, symbol: str) -> int:
        raise NotImplementedError

    def get_buy_price(self, symbol: str) -> float:
        raise NotImplementedError

    def get_max_price(self, symbol: str) -> float:
        raise NotImplementedError

    def get_cash(self) -> float:
        raise NotImplementedError

    def get_total_value(self) -> float:
        raise NotImplementedError

    def get_positions_summary(self) -> list[dict]:
        raise NotImplementedError


def create_executor(mode: str = "simulator", **kwargs) -> BaseExecutor:
    """工厂函数"""
    if mode == "simulator":
        return SimulatorExecutor(**kwargs)
    elif mode == "live":
        return LiveExecutor(**kwargs)
    else:
        raise ValueError(f"未知执行器模式: {mode}")
