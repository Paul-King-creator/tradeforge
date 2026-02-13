"""
Paper Trader - Virtuelles Trading & Tracking
"""
import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import asdict
import pandas as pd

class PaperTrader:
    def __init__(self, db_path: str = "data/trades.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """SQLite Datenbank initialisieren"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Trades Tabelle
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                strategy TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                leverage INTEGER NOT NULL,
                position_size INTEGER DEFAULT 100,
                confidence REAL,
                setup_description TEXT,
                entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                exit_time TIMESTAMP,
                status TEXT DEFAULT 'open',
                pnl_percent REAL,
                pnl_absolute REAL,
                exit_reason TEXT
            )
        ''')
        
        # Performance Metriken
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE UNIQUE,
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                win_rate REAL,
                total_pnl REAL,
                avg_win REAL,
                avg_loss REAL,
                profit_factor REAL,
                portfolio_value REAL
            )
        ''')
        
        # Strategie-Performance
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT UNIQUE,
                total_trades INTEGER,
                wins INTEGER,
                losses INTEGER,
                win_rate REAL,
                avg_return REAL,
                total_pnl REAL,
                last_updated TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def open_position(self, signal, capital: float = 10000, risk_per_trade: float = 0.02):
        """Neue Position eröffnen"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Position Size berechnen
        risk_amount = capital * risk_per_trade
        price_risk = abs(signal.entry_price - signal.stop_loss)
        
        if price_risk > 0:
            position_size = int(risk_amount / price_risk)
        else:
            position_size = 10  # Fallback
        
        cursor.execute('''
            INSERT INTO trades 
            (ticker, signal_type, strategy, entry_price, stop_loss, take_profit, 
             leverage, position_size, confidence, setup_description, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')
        ''', (
            signal.ticker, signal.signal_type, signal.strategy,
            signal.entry_price, signal.stop_loss, signal.take_profit,
            signal.leverage, position_size, signal.confidence,
            signal.setup_description
        ))
        
        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return trade_id
    
    def check_and_close_positions(self, market_data: Dict[str, pd.DataFrame]):
        """Offene Positionen auf SL/TP prüfen"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, ticker, signal_type, entry_price, stop_loss, take_profit, leverage
            FROM trades WHERE status = 'open'
        ''')
        
        open_trades = cursor.fetchall()
        closed_trades = []
        
        for trade in open_trades:
            trade_id, ticker, signal_type, entry, sl, tp, leverage = trade
            
            if ticker not in market_data:
                continue
            
            current_price = market_data[ticker]['Close'].iloc[-1]
            
            # Prüfen ob SL oder TP erreicht
            hit_stop = False
            hit_target = False
            
            if signal_type == 'buy':
                hit_stop = current_price <= sl
                hit_target = current_price >= tp
            else:  # sell
                hit_stop = current_price >= sl
                hit_target = current_price <= tp
            
            if hit_stop or hit_target:
                # Trade schließen
                if signal_type == 'buy':
                    pnl_pct = ((current_price - entry) / entry) * leverage * 100
                else:
                    pnl_pct = ((entry - current_price) / entry) * leverage * 100
                
                exit_reason = 'stop_loss' if hit_stop else 'take_profit'
                
                cursor.execute('''
                    UPDATE trades 
                    SET status = 'closed', exit_price = ?, exit_time = ?,
                        pnl_percent = ?, exit_reason = ?
                    WHERE id = ?
                ''', (current_price, datetime.now(), pnl_pct, exit_reason, trade_id))
                
                closed_trades.append({
                    'id': trade_id,
                    'ticker': ticker,
                    'pnl': pnl_pct,
                    'reason': exit_reason
                })
        
        conn.commit()
        conn.close()
        
        return closed_trades
    
    def get_open_positions(self) -> List[Dict]:
        """Alle offenen Positionen"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM trades WHERE status = 'open' ORDER BY entry_time DESC
        ''')
        
        columns = [description[0] for description in cursor.description]
        trades = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return trades
    
    def get_daily_performance(self) -> Dict:
        """Tages-Performance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        today = datetime.now().date()
        
        # Heute geschlossene Trades
        cursor.execute('''
            SELECT COUNT(*), AVG(pnl_percent), SUM(CASE WHEN pnl_percent > 0 THEN 1 ELSE 0 END)
            FROM trades 
            WHERE DATE(exit_time) = ? AND status = 'closed'
        ''', (today,))
        
        result = cursor.fetchone()
        total_trades = result[0] or 0
        avg_pnl = result[1] or 0
        wins = result[2] or 0
        
        conn.close()
        
        return {
            'date': today,
            'total_trades': total_trades,
            'winning_trades': wins,
            'losing_trades': total_trades - wins,
            'win_rate': (wins / total_trades * 100) if total_trades > 0 else 0,
            'avg_pnl': avg_pnl
        }
    
    def get_strategy_performance(self) -> List[Dict]:
        """Performance pro Strategie"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT strategy,
                   COUNT(*) as total,
                   SUM(CASE WHEN pnl_percent > 0 THEN 1 ELSE 0 END) as wins,
                   AVG(pnl_percent) as avg_return
            FROM trades 
            WHERE status = 'closed'
            GROUP BY strategy
        ''')
        
        results = []
        for row in cursor.fetchall():
            strategy, total, wins, avg_ret = row
            win_rate = (wins / total * 100) if total > 0 else 0
            results.append({
                'strategy': strategy,
                'total_trades': total,
                'wins': wins,
                'losses': total - wins,
                'win_rate': win_rate,
                'avg_return': avg_ret
            })
        
        conn.close()
        return results