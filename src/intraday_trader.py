"""
Intraday Paper Trader
Daytrading = Keine Übernachtung, Max Hold Time
"""
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class IntradayTrader:
    """
    Intraday-Spezifischer Trader:
    - Positionen werden am Tagende geschlossen
    - Time-Exit nach Max-Hold-Time
    - Keine Übernachtungs-Risiken
    """
    
    def __init__(self, db_path: str = "data/trades.db"):
        self.db_path = db_path
        self.position_size = 100  # Fix $100 pro Trade
        self.market_close = 21    # 21:00 UTC = 16:00 EST (Marktschluss)
        self.init_db()
    
    def init_db(self):
        """Intraday-optimierte Datenbank"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                strategy TEXT,
                signal_type TEXT,
                entry_price REAL,
                exit_price REAL,
                stop_loss REAL,
                take_profit REAL,
                position_size REAL,
                shares INTEGER,
                leverage INTEGER,
                confidence REAL,
                ml_probability REAL,
                setup_description TEXT,
                entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                exit_time TIMESTAMP,
                pnl_percent REAL,
                pnl_dollar REAL,
                status TEXT DEFAULT 'open',
                close_reason TEXT,
                max_hold_minutes INTEGER DEFAULT 60,
                target_profit_percent REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def open_position(self, signal) -> Optional[int]:
        """Öffnet Intraday-Position"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        shares = max(1, int(self.position_size / signal.entry_price))
        now = datetime.now()
        
        cursor.execute('''
            INSERT INTO trades 
            (ticker, strategy, signal_type, entry_price, stop_loss, take_profit, 
             position_size, shares, leverage, confidence, setup_description, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')
        ''', (
            signal.ticker, signal.strategy, signal.signal_type,
            signal.entry_price, signal.stop_loss, signal.take_profit,
            self.position_size, shares, signal.leverage,
            signal.confidence, signal.setup_description
        ))
        
        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return trade_id
    
    def check_and_close_positions(self, market_data: Dict) -> List[Dict]:
        """Prüft SL, TP, und Time-Exit"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, ticker, signal_type, entry_price, stop_loss, take_profit, shares, leverage
            FROM trades WHERE status = 'open'
        ''')
        
        open_trades = cursor.fetchall()
        closed_trades = []
        
        for trade in open_trades:
            trade_id, ticker, signal_type, entry, sl, tp, shares, leverage = trade
            
            if ticker not in market_data:
                continue
            
            current_price = market_data[ticker]['Close'].iloc[-1]
            close_reason = None
            
            # Stop Loss
            if signal_type == 'buy' and current_price <= sl:
                close_reason = 'STOP_LOSS'
            elif signal_type == 'sell' and current_price >= sl:
                close_reason = 'STOP_LOSS'
            # Take Profit
            elif signal_type == 'buy' and current_price >= tp:
                close_reason = 'TAKE_PROFIT'
            elif signal_type == 'sell' and current_price <= tp:
                close_reason = 'TAKE_PROFIT'
            
            if close_reason:
                if signal_type == 'buy':
                    pnl_percent = ((current_price - entry) / entry) * leverage * 100
                else:
                    pnl_percent = ((entry - current_price) / entry) * leverage * 100
                
                cursor.execute('''
                    UPDATE trades SET status='closed', exit_price=?, exit_time=?,
                    pnl_percent=?, close_reason=? WHERE id=?
                ''', (current_price, datetime.now().isoformat(), pnl_percent, close_reason, trade_id))
                
                closed_trades.append({'id': trade_id, 'ticker': ticker, 'pnl': pnl_percent, 'reason': close_reason})
        
        conn.commit()
        conn.close()
        return closed_trades