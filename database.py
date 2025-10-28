import sqlite3
from datetime import datetime
import hashlib
import secrets

class TradingDatabase:
    def __init__(self, db_name='trading_app.db'):
        self.db_name = db_name
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_name)
        cur = conn.cursor()
        
        # USERS
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
        
        # TRADES
        cur.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                ticker TEXT NOT NULL,
                direction TEXT NOT NULL,
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
                confidence INTEGER,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        conn.commit()
        conn.close()

    def register_user(self, username, password, initial_capital=10000):
        conn = sqlite3.connect(self.db_name)
        cur = conn.cursor()
        try:
            user_id = secrets.token_urlsafe(16)
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            cur.execute('''
                INSERT INTO users (user_id, username, password_hash, initial_capital, current_capital)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, username, password_hash, float(initial_capital), float(initial_capital)))
            conn.commit()
            return user_id
        except sqlite3.IntegrityError:
            return None
        finally:
            conn.close()

    def login_user(self, username, password):
        conn = sqlite3.connect(self.db_name)
        cur = conn.cursor()
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        cur.execute('''
            SELECT user_id, username, initial_capital, current_capital
            FROM users WHERE username = ? AND password_hash = ?
        ''', (username, password_hash))
        result = cur.fetchone()
        conn.close()
        if result:
            return {
                'user_id': result[0],
                'username': result[1],
                'initial_capital': float(result[2]),
                'current_capital': float(result[3])
            }
        return None

    def get_user_info(self, user_id):
        conn = sqlite3.connect(self.db_name)
        cur = conn.cursor()
        cur.execute('''
            SELECT username, initial_capital, current_capital
            FROM users WHERE user_id = ?
        ''', (user_id,))
        result = cur.fetchone()
        conn.close()
        if result:
            return {
                'username': result[0],
                'initial_capital': float(result[1]),
                'current_capital': float(result[2])
            }
        return None

    def update_capital(self, user_id, new_capital):
        conn = sqlite3.connect(self.db_name)
        cur = conn.cursor()
        # ИСПРАВЛЕНИЕ: явное приведение к float
        cur.execute(
            'UPDATE users SET current_capital = ? WHERE user_id = ?', 
            (float(new_capital), str(user_id))
        )
        conn.commit()
        conn.close()

    def can_add_trade(self, user_id, ticker):
        conn = sqlite3.connect(self.db_name)
        cur = conn.cursor()
        cur.execute('''
            SELECT COUNT(*) FROM trades 
            WHERE user_id = ? AND ticker = ? AND status = 'ACTIVE'
        ''', (user_id, ticker))
        count = cur.fetchone()[0]
        conn.close()
        return count == 0

    def add_trade(self, user_id, signal_data, position_percent=10):
        if not self.can_add_trade(user_id, signal_data['ticker']):
            raise ValueError(f"Уже есть активная сделка по {signal_data['ticker']}")
        
        conn = sqlite3.connect(self.db_name)
        cur = conn.cursor()
        user_info = self.get_user_info(user_id)
        current_capital = user_info['current_capital']
        position_size = (current_capital * position_percent) / 100
        
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
            float(signal_data['entry_price']),
            float(signal_data['stop_loss']),
            float(signal_data.get('tp1', 0)),
            float(signal_data.get('tp2', 0)),
            float(signal_data.get('tp3', 0)),
            float(signal_data.get('tp1_prob', 0)),
            float(signal_data.get('tp2_prob', 0)),
            float(signal_data.get('tp3_prob', 0)),
            float(position_size),
            float(position_percent),
            signal_data.get('model', 'Octopus'),
            int(signal_data.get('confidence', 0))
        ))
        
        trade_id = cur.lastrowid
        conn.commit()
        conn.close()
        return trade_id

    def get_active_trades(self, user_id):
        conn = sqlite3.connect(self.db_name)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute('''
            SELECT * FROM trades 
            WHERE user_id = ? AND status = 'ACTIVE'
            ORDER BY signal_date DESC
        ''', (user_id,))
        trades = [dict(row) for row in cur.fetchall()]
        conn.close()
        return trades

    def get_closed_trades(self, user_id):
        conn = sqlite3.connect(self.db_name)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute('''
            SELECT * FROM trades 
            WHERE user_id = ? AND status = 'CLOSED'
            ORDER BY close_date DESC
        ''', (user_id,))
        trades = [dict(row) for row in cur.fetchall()]
        conn.close()
        return trades

    def partial_close_trade(self, trade_id, close_price, tp_level):
        conn = sqlite3.connect(self.db_name)
        cur = conn.cursor()
        
        cur.execute('SELECT * FROM trades WHERE trade_id = ?', (trade_id,))
        row = cur.fetchone()
        if not row:
            conn.close()
            return 0.0
        
        # Создаем dict из row
        trade = dict(zip([d[0] for d in cur.description], row))
        
        if trade['direction'] == 'LONG':
            pnl_percent = ((close_price - trade['entry_price']) / trade['entry_price']) * 100
        else:
            pnl_percent = ((trade['entry_price'] - close_price) / trade['entry_price']) * 100
        
        percent_to_close = {'tp1': 30, 'tp2': 30, 'tp3': 40}[tp_level]
        part_size = trade['position_size'] * percent_to_close / 100
        part_pnl_dollars = (part_size * pnl_percent) / 100
        
        # Обновляем частичное закрытие
        cur.execute(f'''
            UPDATE trades 
            SET {tp_level}_closed = ?, 
                remaining_percent = remaining_percent - ?,
                total_pnl_dollars = total_pnl_dollars + ?
            WHERE trade_id = ?
        ''', (float(percent_to_close), float(percent_to_close), float(part_pnl_dollars), trade_id))
        
        # TP1: перенос SL в безубыток
        if tp_level == 'tp1':
            cur.execute(
                'UPDATE trades SET sl_breakeven = 1, stop_loss = entry_price WHERE trade_id = ?', 
                (trade_id,)
            )
        
        # Обновляем капитал
        user_info = self.get_user_info(trade['user_id'])
        new_capital = user_info['current_capital'] + part_pnl_dollars
        self.update_capital(trade['user_id'], new_capital)
        
        conn.commit()
        conn.close()
        return part_pnl_dollars

    def full_close_trade(self, trade_id, close_price, close_reason):
        conn = sqlite3.connect(self.db_name)
        cur = conn.cursor()
        
        cur.execute('SELECT * FROM trades WHERE trade_id = ?', (trade_id,))
        row = cur.fetchone()
        if not row:
            conn.close()
            return
        
        # ИСПРАВЛЕНИЕ: создаем dict правильно
        trade = dict(zip([d[0] for d in cur.description], row))
        
        remaining_percent = float(trade['remaining_percent'])
        if remaining_percent > 0:
            if trade['direction'] == 'LONG':
                pnl_percent = ((close_price - trade['entry_price']) / trade['entry_price']) * 100
            else:
                pnl_percent = ((trade['entry_price'] - close_price) / trade['entry_price']) * 100
            
            remaining_size = trade['position_size'] * remaining_percent / 100
            remaining_pnl = (remaining_size * pnl_percent) / 100
            
            cur.execute('''
                UPDATE trades 
                SET status = 'CLOSED',
                    close_reason = ?,
                    close_price = ?,
                    total_pnl_percent = ?,
                    total_pnl_dollars = total_pnl_dollars + ?,
                    close_date = ?
                WHERE trade_id = ?
            ''', (
                str(close_reason), 
                float(close_price), 
                float(pnl_percent), 
                float(remaining_pnl), 
                datetime.now().isoformat(), 
                trade_id
            ))
            
            # ИСПРАВЛЕНИЕ: обновляем капитал с явным приведением типов
            user_info = self.get_user_info(trade['user_id'])
            new_capital = float(user_info['current_capital']) + float(remaining_pnl)
            self.update_capital(trade['user_id'], new_capital)
        
        conn.commit()
        conn.close()

    def get_statistics(self, user_id):
        conn = sqlite3.connect(self.db_name)
        cur = conn.cursor()
        
        cur.execute('SELECT COUNT(*) FROM trades WHERE user_id = ?', (user_id,))
        total_trades = cur.fetchone()[0]
        
        cur.execute('''
            SELECT COUNT(*), 
                   SUM(CASE WHEN total_pnl_percent > 0 THEN 1 ELSE 0 END),
                   AVG(total_pnl_percent),
                   SUM(total_pnl_dollars)
            FROM trades 
            WHERE user_id = ? AND status = 'CLOSED'
        ''', (user_id,))
        
        result = cur.fetchone()
        closed_trades = result[0] or 0
        winning_trades = result[1] or 0
        avg_pnl = result[2] or 0
        total_pnl = result[3] or 0
        
        win_rate = (winning_trades / closed_trades * 100) if closed_trades > 0 else 0
        
        conn.close()
        
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
