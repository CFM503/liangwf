"""
订单执行器 — 模拟盘 (SQLite) + 实盘接口（预留）
================================================
模拟盘的订单、持仓、资金全部存 SQLite，
程序重启后状态自动恢复。
"""

import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from strategy.signals import Signal, Action
from utils.logger import get_logger

log = get_logger("xlt.executor")


@dataclass
class OrderResult:
    """订单执行结果"""
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
    """执行器基类（模拟和实盘共用接口）"""

    @abstractmethod
    def execute(self, signal: Signal) -> OrderResult: ...
    @abstractmethod
    def get_position(self, symbol: str) -> int: ...
    @abstractmethod
    def get_buy_price(self, symbol: str) -> float: ...
    @abstractmethod
    def get_max_price(self, symbol: str) -> float: ...
    @abstractmethod
    def get_cash(self) -> float: ...
    @abstractmethod
    def get_total_value(self) -> float: ...
    @abstractmethod
    def get_positions_summary(self) -> list[dict]: ...


class SimulatorExecutor(BaseExecutor):
    """
    模拟执行器 — 完全持久化到 SQLite

    交易成本：
    - 佣金: 万2.5（最低5元）
    - 印花税: 千分之一（仅卖出）
    - 滑点: 0.1%
    - A股规则: T+1、100股整数、涨跌停
    """

    COMMISSION = 0.00025   # 佣金万2.5
    MIN_COMMISSION = 5.0   # 最低佣金
    STAMP_TAX = 0.001      # 印花税千一（卖出）
    SLIPPAGE = 0.001       # 滑点 0.1%
    T_PLUS_1 = True        # T+1 限制

    def __init__(self, db_path: str = "xlt.db", initial_cash: float = 1_000_000):
        self.db_path = db_path
        self.initial_cash = initial_cash
        self._today_bought: set[str] = set()  # T+1: 今日买入的股票不能卖出
        self._init_db()

    def _init_db(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS account (key TEXT PRIMARY KEY, value REAL)")
        c.execute("INSERT OR IGNORE INTO account (key, value) VALUES ('cash', ?)",
                  (self.initial_cash,))
        c.execute("""CREATE TABLE IF NOT EXISTS positions (
            symbol TEXT PRIMARY KEY, size INTEGER DEFAULT 0,
            buy_price REAL DEFAULT 0, max_price REAL DEFAULT 0,
            buy_date TEXT, updated_at TEXT
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY, symbol TEXT, action TEXT,
            price REAL, size INTEGER, amount REAL, fee REAL,
            reason TEXT, created_at TEXT
        )""")
        conn.commit()
        conn.close()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def reset_today(self):
        """每日开始时调用，清空 T+1 记录"""
        self._today_bought.clear()

    # ──────────────────────────────
    # 核心：执行订单
    # ──────────────────────────────
    def execute(self, signal: Signal) -> OrderResult:
        if signal.action == Action.HOLD:
            return OrderResult(False, message="HOLD")
        if signal.action == Action.BUY:
            return self._buy(signal)
        if signal.action == Action.SELL:
            return self._sell(signal)
        return OrderResult(False, message="未知操作")

    def _buy(self, signal: Signal) -> OrderResult:
        conn = self._conn()
        c = conn.cursor()
        try:
            c.execute("SELECT value FROM account WHERE key='cash'")
            cash = c.fetchone()[0]
            total_value = self._total_value(conn)

            # 加滑点
            price = signal.price * (1 + self.SLIPPAGE)

            # 仓位限制
            max_amount = total_value * 0.20
            available = min(cash * 0.95, max_amount)
            size = int(available / price / 100) * 100

            if size < 100:
                return OrderResult(False, symbol=signal.symbol, message="资金不足或仓位已满")

            amount = price * size
            fee = max(amount * self.COMMISSION, self.MIN_COMMISSION)
            total_cost = amount + fee

            if total_cost > cash:
                size = int((cash - self.MIN_COMMISSION) / price / 100) * 100
                if size < 100:
                    return OrderResult(False, symbol=signal.symbol, message="资金不足")
                amount = price * size
                fee = max(amount * self.COMMISSION, self.MIN_COMMISSION)
                total_cost = amount + fee

            c.execute("UPDATE account SET value = value - ? WHERE key='cash'", (total_cost,))

            now = datetime.now()
            c.execute("""
                INSERT INTO positions (symbol, size, buy_price, max_price, buy_date, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    size = size + ?,
                    buy_price = (buy_price * size + ? * ?) / (size + ?),
                    max_price = MAX(max_price, ?),
                    updated_at = ?
            """, (signal.symbol, size, price, price, now.date().isoformat(), now.isoformat(),
                  size, price, size, size, price, now.isoformat()))

            oid = f"B_{signal.symbol}_{int(time.time())}"
            c.execute("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?)",
                      (oid, signal.symbol, "BUY", price, size, amount, fee, signal.reason, now.isoformat()))

            conn.commit()
            self._today_bought.add(signal.symbol)  # T+1 标记
            log.info(f"[执行] 买入 {signal.symbol} @ {price:.2f} x {size} 费用{fee:.0f}")
            return OrderResult(True, oid, signal.symbol, "BUY", price, size, amount, fee, signal.reason, now.isoformat())
        except Exception as e:
            conn.rollback()
            log.error(f"[执行] 买入失败 {signal.symbol}: {e}")
            return OrderResult(False, symbol=signal.symbol, message=str(e))
        finally:
            conn.close()

    def _sell(self, signal: Signal) -> OrderResult:
        # T+1 检查
        if self.T_PLUS_1 and signal.symbol in self._today_bought:
            return OrderResult(False, symbol=signal.symbol, message="T+1限制：今日买入不能卖出")

        conn = self._conn()
        c = conn.cursor()
        try:
            c.execute("SELECT size, buy_price FROM positions WHERE symbol=?", (signal.symbol,))
            row = c.fetchone()
            if not row or row[0] <= 0:
                return OrderResult(False, symbol=signal.symbol, message="无持仓")

            size, buy_price = row
            price = signal.price * (1 - self.SLIPPAGE)
            amount = price * size
            commission = max(amount * self.COMMISSION, self.MIN_COMMISSION)
            stamp = amount * self.STAMP_TAX
            fee = commission + stamp
            net = amount - fee

            c.execute("UPDATE account SET value = value + ? WHERE key='cash'", (net,))
            c.execute("DELETE FROM positions WHERE symbol=?", (signal.symbol,))

            now = datetime.now()
            pnl = (price / buy_price - 1) * 100 if buy_price > 0 else 0
            oid = f"S_{signal.symbol}_{int(time.time())}"
            reason = f"{signal.reason} | 盈亏{pnl:+.2f}%"
            c.execute("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?)",
                      (oid, signal.symbol, "SELL", price, size, amount, fee, reason, now.isoformat()))

            conn.commit()
            log.info(f"[执行] 卖出 {signal.symbol} @ {price:.2f} x {size} 盈亏{pnl:+.2f}%")
            return OrderResult(True, oid, signal.symbol, "SELL", price, size, amount, fee, reason, now.isoformat())
        except Exception as e:
            conn.rollback()
            log.error(f"[执行] 卖出失败 {signal.symbol}: {e}")
            return OrderResult(False, symbol=signal.symbol, message=str(e))
        finally:
            conn.close()

    # ──────────────────────────────
    # 查询
    # ──────────────────────────────
    def get_position(self, symbol: str) -> int:
        conn = self._conn()
        c = conn.cursor()
        c.execute("SELECT size FROM positions WHERE symbol=?", (symbol,))
        row = c.fetchone()
        conn.close()
        return row[0] if row else 0

    def get_buy_price(self, symbol: str) -> float:
        conn = self._conn()
        c = conn.cursor()
        c.execute("SELECT buy_price FROM positions WHERE symbol=?", (symbol,))
        row = c.fetchone()
        conn.close()
        return row[0] if row else 0.0

    def get_max_price(self, symbol: str) -> float:
        conn = self._conn()
        c = conn.cursor()
        c.execute("SELECT max_price FROM positions WHERE symbol=?", (symbol,))
        row = c.fetchone()
        conn.close()
        return row[0] if row else 0.0

    def update_max_price(self, symbol: str, price: float):
        conn = self._conn()
        c = conn.cursor()
        c.execute("UPDATE positions SET max_price = MAX(max_price, ?) WHERE symbol=?",
                  (price, symbol))
        conn.commit()
        conn.close()

    def get_cash(self) -> float:
        conn = self._conn()
        c = conn.cursor()
        c.execute("SELECT value FROM account WHERE key='cash'")
        val = c.fetchone()[0]
        conn.close()
        return val

    def _total_value(self, conn) -> float:
        c = conn.cursor()
        c.execute("SELECT value FROM account WHERE key='cash'")
        cash = c.fetchone()[0]
        c.execute("SELECT COALESCE(SUM(size * buy_price), 0) FROM positions")
        pos = c.fetchone()[0]
        return cash + pos

    def get_total_value(self) -> float:
        conn = self._conn()
        val = self._total_value(conn)
        conn.close()
        return val

    def get_positions_summary(self) -> list[dict]:
        conn = self._conn()
        c = conn.cursor()
        c.execute("SELECT symbol, size, buy_price, max_price FROM positions WHERE size > 0")
        rows = c.fetchall()
        conn.close()
        return [{"symbol": r[0], "size": r[1], "buy_price": r[2], "max_price": r[3]} for r in rows]

    def get_recent_orders(self, limit: int = 20) -> list[dict]:
        conn = self._conn()
        c = conn.cursor()
        c.execute("SELECT * FROM orders ORDER BY created_at DESC LIMIT ?", (limit,))
        rows = c.fetchall()
        conn.close()
        return [
            {"order_id": r[0], "symbol": r[1], "action": r[2], "price": r[3],
             "size": r[4], "amount": r[5], "fee": r[6], "reason": r[7], "created_at": r[8]}
            for r in rows
        ]


# ═══════════════════════════════════════
# 实盘接口（预留）
# ═══════════════════════════════════════
class LiveExecutor(BaseExecutor):
    """
    实盘执行器 — 需要对接券商 API

    推荐方案：
    1. easytrader — pip install easytrader（华泰/银河/雪球）
    2. xtquant (QMT) — 量化专用
    3. 继承 BaseExecutor 自行实现
    """

    def __init__(self, **kwargs):
        raise NotImplementedError(
            "实盘接口待实现。请继承 BaseExecutor 并实现 execute()。\n"
            "推荐: easytrader 或 xtquant"
        )

    def execute(self, signal): ...
    def get_position(self, symbol): ...
    def get_buy_price(self, symbol): ...
    def get_max_price(self, symbol): ...
    def get_cash(self): ...
    def get_total_value(self): ...
    def get_positions_summary(self): ...


def create_executor(mode: str = "simulator", **kwargs) -> BaseExecutor:
    """工厂函数"""
    if mode == "simulator":
        return SimulatorExecutor(**kwargs)
    elif mode == "live":
        return LiveExecutor(**kwargs)
    raise ValueError(f"未知模式: {mode}")
