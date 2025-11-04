import os
from pathlib import Path
import sqlite3
from datetime import datetime
import hashlib
import secrets
from decimal import Decimal, ROUND_HALF_UP

# ----- string constants to avoid NameError -----
TP = "TP"
SL = "SL"
MANUAL = "MANUAL"

def _stable_db_path():
    base = os.getenv("ARXORA_DATA_DIR", os.path.expanduser("~/.arxora"))
    Path(base).mkdir(parents=True, exist_ok=True)
    return str(Path(base) / "trading_app.db")

class TradingDatabase:
    def __init__(self, db_name=None):
        self.db_name = db_name or _stable_db_path()
        self.init_database()
        self.migrate_schema()

    def _connect(self):
        conn = sqlite3.connect(self.db_name, timeout=30, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    # ---------- SCHEMA ----------

    def init_database(self):
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    initial_capital REAL DEFAULT 10000,
                    current_capital REAL DEFAULT 10000,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cur.execute('''
                CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username_nc 
                ON users(username COLLATE NOCASE)
            ''')
            cur.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    direction TEXT NOT NULL,                 -- LONG|SHORT
                    entry_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit_1 REAL,
                    take_profit_2 REAL,
                    take_profit_3 REAL,
                    tp1_probability REAL,
                    tp2_probability REAL,
                    tp3_probability REAL,
                    position_size REAL NOT NULL,
                    position_percent REAL DEFAULT 10,
                    remaining_percent REAL DEFAULT 100,
                    tp1_closed REAL DEFAULT 0,
                    tp2_closed REAL DEFAULT 0,
                    tp3_closed REAL DEFAULT 0,
                    sl_breakeven INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'ACTIVE',
                    close_reason TEXT,
                    close_price REAL,
                    total_pnl_percent REAL DEFAULT 0,
                    total_pnl_dollars REAL DEFAULT 0,
                    signal_date TEXT DEFAULT CURRENT_TIMESTAMP,
                    close_date TEXT,
                    model_used TEXT,
                    confidence INTEGER
                    -- новые поля добавятся миграцией
                )
            ''')
        finally:
            conn.close()

    def _col_missing(self, cur, table, col):
        cur.execute(f"PRAGMA table_info({table})")
        return all(r[1] != col for r in cur.fetchall())

    def migrate_schema(self):
        conn = self._connect()
        try:
            cur = conn.cursor()
            # orders
            cur.execute('''
                CREATE TABLE IF NOT EXISTS orders (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  trade_id INTEGER NOT NULL,
                  broker_order_id TEXT UNIQUE,
                  oco_group_id TEXT,
                  type TEXT NOT NULL,                 -- LIMIT|MARKET|STOP_MARKET|TAKE_PROFIT_LIMIT ...
                  reduce_only INTEGER DEFAULT 0,
                  status TEXT NOT NULL DEFAULT 'NEW', -- NEW|PARTIALLY_FILLED|FILLED|CANCELED|REJECTED|EXPIRED
                  qty REAL NOT NULL,
                  price REAL,
                  stop_price REAL,
                  trigger_price_type TEXT,            -- LAST|MARK|BID|ASK
                  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                  updated_at TEXT,
                  FOREIGN KEY(trade_id) REFERENCES trades(trade_id) ON DELETE CASCADE
                )
            ''')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_orders_trade_status ON orders(trade_id, status)')
            # idempotency
            cur.execute('''
                CREATE TABLE IF NOT EXISTS idempotency (
                  key TEXT PRIMARY KEY,
                  op TEXT NOT NULL,                   -- OPEN|PLACE_SL|PLACE_TP|CLOSE
                  trade_id INTEGER,
                  response TEXT,
                  created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # extra columns in trades
            add_cols = [
                ('broker_position_id','TEXT'),
                ('sl_order_id','TEXT'),
                ('tp1_order_id','TEXT'),
                ('tp2_order_id','TEXT'),
                ('tp3_order_id','TEXT'),
                ('oco_group_id','TEXT'),
                ('sl_triggered_at','TEXT'),
                ('closed_by','TEXT')  # TP | SL | MANUAL
            ]
            for col, typ in add_cols:
                if self._col_missing(cur, 'trades', col):
                    cur.execute(f'ALTER TABLE trades ADD COLUMN {col} {typ}')
            conn.commit()
        finally:
            conn.close()

    # ---------- USERS ----------

    def register_user(self, username, password, initial_capital=10000):
        conn = self._connect()
        try:
            cur = conn.cursor()
            conn.execute("BEGIN IMMEDIATE;")
            username = (username or "").strip()
            password = (password or "").strip()
            user_id = secrets.token_urlsafe(16)
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            cur.execute('''
                INSERT INTO users (user_id, username, password_hash, initial_capital, current_capital)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, username, password_hash, initial_capital, initial_capital))
            conn.commit()
            return user_id
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def login_user(self, username, password):
        conn = self._connect()
        try:
            cur = conn.cursor()
            username = (username or "").strip()
            password = (password or "").strip()
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            cur.execute('''
                SELECT rowid, user_id, username, initial_capital, current_capital
                FROM users 
                WHERE username = ? COLLATE NOCASE AND password_hash = ?
            ''', (username, password_hash))
            r = cur.fetchone()
            if not r:
                return None
            return {
                'user_id': r[1],
                'username': r[2],
                'initial_capital': r[3],
                'current_capital': r[4]
            }
        finally:
            conn.close()

    def get_user_info(self, user_id):
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute('''
                SELECT username, initial_capital, current_capital
                FROM users WHERE user_id = ?
            ''', (user_id,))
            r = cur.fetchone()
            if not r:
                return None
            return {
                'username': r[0],
                'initial_capital': r[1],
                'current_capital': r[2]
            }
        finally:
            conn.close()

    # ---------- TRADES ----------

    def can_add_trade(self, user_id, ticker):
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute('''
                SELECT COUNT(*) FROM trades 
                WHERE user_id = ? AND ticker = ? AND status = 'ACTIVE'
            ''', (user_id, ticker))
            return (cur.fetchone()[0] or 0) == 0
        finally:
            conn.close()

    def add_trade(self, user_id, signal_data, position_percent=10):
        if not self.can_add_trade(user_id, signal_data['ticker']):
            raise ValueError(f"Уже есть активная сделка по {signal_data['ticker']}")
        conn = self._connect()
        try:
            cur = conn.cursor()
            conn.execute("BEGIN IMMEDIATE;")

            cur.execute("SELECT current_capital FROM users WHERE user_id = ?", (user_id,))
            u = cur.fetchone()
            if not u:
                conn.rollback()
                raise ValueError("Пользователь не найден")
            current_capital = u[0] or 0.0
            position_size = (current_capital * position_percent) / 100.0

            cur.execute('''
                INSERT INTO trades (
                    user_id, ticker, direction, entry_price, stop_loss,
                    take_profit_1, take_profit_2, take_profit_3,
                    tp1_probability, tp2_probability, tp3_probability,
                    position_size, position_percent, model_used, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                signal_data['ticker'],
                signal_data['direction'],
                signal_data['entry_price'],
                signal_data['stop_loss'],
                signal_data.get('tp1'),
                signal_data.get('tp2'),
                signal_data.get('tp3'),
                signal_data.get('tp1_prob'),
                signal_data.get('tp2_prob'),
                signal_data.get('tp3_prob'),
                position_size,
                position_percent,
                signal_data.get('model', 'Octopus'),
                signal_data.get('confidence', 0)
            ))
            trade_id = cur.lastrowid
            conn.commit()
            return trade_id
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_active_trades(self, user_id):
        conn = self._connect()
        try:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute('''
                SELECT * FROM trades 
                WHERE user_id = ? AND status = 'ACTIVE'
                ORDER BY signal_date DESC
            ''', (user_id,))
            return [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()

    def get_closed_trades(self, user_id):
        conn = self._connect()
        try:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute('''
                SELECT * FROM trades 
                WHERE user_id = ? AND status = 'CLOSED'
                ORDER BY close_date DESC
            ''', (user_id,))
            return [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()

    # ---------- ORDERS & IDEMPOTENCY HELPERS ----------

    def record_order(self, trade_id, otype, qty, price=None, stop_price=None,
                     reduce_only=False, trigger_price_type=None, oco_group_id=None):
        conn = self._connect()
        try:
            cur = conn.cursor()
            conn.execute("BEGIN IMMEDIATE;")
            cur.execute('''
              INSERT INTO orders (trade_id, type, qty, price, stop_price, reduce_only, 
                                  trigger_price_type, oco_group_id)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (trade_id, otype, float(qty), price, stop_price,
                  1 if reduce_only else 0, trigger_price_type, oco_group_id))
            oid = cur.lastrowid
            conn.commit()
            return oid
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def attach_broker_order(self, order_id, broker_order_id):
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute('UPDATE orders SET broker_order_id = ?, updated_at = ? WHERE id = ?',
                        (broker_order_id, datetime.now().isoformat(), order_id))
            conn.commit()
        finally:
            conn.close()

    def update_order_status(self, broker_order_id, status):
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute('UPDATE orders SET status = ?, updated_at = ? WHERE broker_order_id = ?',
                        (status, datetime.now().isoformat(), broker_order_id))
            conn.commit()
        finally:
            conn.close()

    def put_idempotency(self, key, op, trade_id=None, response=None):
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute('INSERT OR IGNORE INTO idempotency (key, op, trade_id, response) VALUES (?, ?, ?, ?)',
                        (key, op, trade_id, (response or "")))
            conn.commit()
        finally:
            conn.close()

    def has_idempotency(self, key):
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute('SELECT 1 FROM idempotency WHERE key = ?', (key,))
            return cur.fetchone() is not None
        finally:
            conn.close()

    # ---------- PARTIAL / FULL CLOSE ----------

    def _dec(self, x):
        return Decimal(str(x)).quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)

    def partial_close_trade(self, trade_id, close_price, tp_level):
        conn = self._connect()
        try:
            cur = conn.cursor()
            conn.execute("BEGIN IMMEDIATE;")
            cur.execute('SELECT * FROM trades WHERE trade_id = ?', (trade_id,))
            row = cur.fetchone()
            if not row:
                conn.rollback()
                return 0.0

            cols = [d[0] for d in cur.description]
            trade = dict(zip(cols, row))

            entry = self._dec(trade['entry_price'])
            price = self._dec(close_price)
            if trade['direction'] == 'LONG':
                pnl_percent = (price - entry) / entry * self._dec(100)
            else:
                pnl_percent = (entry - price) / entry * self._dec(100)

            pct_map = {'tp1': self._dec(30), 'tp2': self._dec(30), 'tp3': self._dec(40)}
            percent_to_close = pct_map[tp_level]
            part_size = self._dec(trade['position_size']) * percent_to_close / self._dec(100)
            part_pnl_dollars = part_size * pnl_percent / self._dec(100)

            cur.execute(f'''
                UPDATE trades 
                SET {tp_level}_closed = ?, 
                    remaining_percent = remaining_percent - ?,
                    total_pnl_dollars = total_pnl_dollars + ?
                WHERE trade_id = ?
            ''', (float(percent_to_close), float(percent_to_close), float(part_pnl_dollars), trade_id))

            if tp_level == 'tp1':
                cur.execute('UPDATE trades SET sl_breakeven = 1, stop_loss = entry_price WHERE trade_id = ?', (trade_id,))

            cur.execute("SELECT current_capital FROM users WHERE user_id = ?", (trade['user_id'],))
            r = cur.fetchone()
            if not r:
                conn.rollback()
                raise ValueError("Пользователь не найден")
            new_capital = (r[0] or 0.0) + float(part_pnl_dollars)
            cur.execute("UPDATE users SET current_capital = ? WHERE user_id = ?", (new_capital, trade['user_id']))

            conn.commit()
            return float(part_pnl_dollars)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def full_close_trade(self, trade_id, close_price, close_reason):
        # close_reason должен быть одной из строк: TP, SL, MANUAL
        if close_reason not in (TP, SL, MANUAL):
            close_reason = MANUAL
        conn = self._connect()
        try:
            cur = conn.cursor()
            conn.execute("BEGIN IMMEDIATE;")
            cur.execute('SELECT * FROM trades WHERE trade_id = ?', (trade_id,))
            row = cur.fetchone()
            if not row:
                conn.rollback()
                return

            cols = [d[0] for d in cur.description]
            trade = dict(zip(cols, row))

            remaining_percent = trade['remaining_percent']
            if remaining_percent > 0:
                entry = self._dec(trade['entry_price'])
                price = self._dec(close_price)
                if trade['direction'] == 'LONG':
                    pnl_percent = (price - entry) / entry * self._dec(100)
                else:
                    pnl_percent = (entry - price) / entry * self._dec(100)

                remaining_size = self._dec(trade['position_size']) * self._dec(remaining_percent) / self._dec(100)
                remaining_pnl = remaining_size * pnl_percent / self._dec(100)

                cur.execute('''
                    UPDATE trades 
                    SET status = 'CLOSED',
                        close_reason = ?,
                        close_price = ?,
                        total_pnl_percent = ?,
                        total_pnl_dollars = total_pnl_dollars + ?,
                        close_date = ?,
                        closed_by = ?
                    WHERE trade_id = ?
                ''', (close_reason, float(price), float(pnl_percent), float(remaining_pnl),
                      datetime.now().isoformat(), close_reason, trade_id))

                cur.execute("SELECT current_capital FROM users WHERE user_id = ?", (trade['user_id'],))
                r = cur.fetchone()
                if not r:
                    conn.rollback()
                    raise ValueError("Пользователь не найден")
                new_capital = (r[0] or 0.0) + float(remaining_pnl)
                cur.execute("UPDATE users SET current_capital = ? WHERE user_id = ?", (new_capital, trade['user_id']))

            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ---------- STATS ----------

    def get_statistics(self, user_id):
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute('SELECT COUNT(*) FROM trades WHERE user_id = ?', (user_id,))
            total_trades = cur.fetchone()[0] or 0

            cur.execute('''
                SELECT COUNT(*), 
                       SUM(CASE WHEN total_pnl_percent > 0 THEN 1 ELSE 0 END),
                       AVG(total_pnl_percent),
                       SUM(total_pnl_dollars)
                FROM trades 
                WHERE user_id = ? AND status = 'CLOSED'
            ''', (user_id,))
            r = cur.fetchone() or (0, 0, 0.0, 0.0)
            closed_trades = r[0] or 0
            winning_trades = r[1] or 0
            avg_pnl = r[2] or 0.0
            total_pnl = r[3] or 0.0
            win_rate = (winning_trades / closed_trades * 100.0) if closed_trades > 0 else 0.0

            return {
                'total_trades': total_trades,
                'closed_trades': closed_trades,
                'active_trades': total_trades - closed_trades,
                'winning_trades': winning_trades,
                'losing_trades': closed_trades - winning_trades,
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'total_pnl': total_pnl
            }
        finally:
            conn.close()
