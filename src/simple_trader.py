"""
Simplifizierter Paper Trader
Fokus auf Qualität statt Quantität
"""
import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional

class SimplePaperTrader:
    """
    Einfacher Trader:
    - Fixe Position Size: $100 pro Trade
    - Kein Kapital-Limit (unendlich Geld)
    - Fokus auf Learning durch viele kleine Trades
    """
    
    def __init__(self, db_path: str = "data/trades.db"):
        self.db_path = db_path
        self.position_size = 100  # Fix: $100 pro Trade
        self.init_db()
    
    def init_db(self):
        """Datenbank initialisieren"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Einfache Trades Tabelle
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                strategy TEXT,
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
                close_reason TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def open_position(self, signal) -> Optional[int]:
        """
        Öffnet Position mit fixer $100 Größe
        Kein Kapital-Check mehr nötig!
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Shares berechnen basierend auf $100 Position
        shares = max(1, int(self.position_size / signal.entry_price))
        
        # ML Probability aus Signal extrahieren
        ml_prob = getattr(signal, 'ml_probability', None)
        
        cursor.execute('''
            INSERT INTO trades 
            (ticker, strategy, entry_price, stop_loss, take_profit, 
             position_size, shares, leverage, confidence, ml_probability, 
             setup_description, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')
        ''', (
            signal.ticker,
            signal.strategy,
            signal.entry_price,
            signal.stop_loss,
            signal.take_profit,
            self.position_size,
            shares,
            signal.leverage,
            signal.confidence,
            ml_prob,
            signal.setup_description
        ))
        
        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        print(f"   💰 Position: {shares} shares = ${shares * signal.entry_price:.2f}")
        
        return trade_id
    
    def check_and_close_positions(self, market_data: Dict) -> List[Dict]:
        """Prüft SL/TP und schließt Positionen"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, ticker, entry_price, stop_loss, take_profit, 
                   shares, leverage
            FROM trades WHERE status = 'open'
        ''')
        
        open_trades = cursor.fetchall()
        closed_trades = []
        
        for trade in open_trades:
            trade_id, ticker, entry, sl, tp, shares, leverage = trade
            
            if ticker not in market_data:
                continue
            
            current_price = market_data[ticker]['close'].iloc[-1]
            
            # SL oder TP hit?
            close_reason = None
            if current_price <= sl:
                close_reason = 'STOP_LOSS'
            elif current_price >= tp:
                close_reason = 'TAKE_PROFIT'
            
            if close_reason:
                # P&L berechnen
                pnl_percent = ((current_price - entry) / entry) * leverage * 100
                pnl_dollar = (current_price - entry) * shares
                
                cursor.execute('''
                    UPDATE trades 
                    SET status = 'closed',
                        exit_price = ?,
                        exit_time = CURRENT_TIMESTAMP,
                        pnl_percent = ?,
                        pnl_dollar = ?,
                        close_reason = ?
                    WHERE id = ?
                ''', (current_price, pnl_percent, pnl_dollar, close_reason, trade_id))
                
                closed_trades.append({
                    'id': trade_id,
                    'ticker': ticker,
                    'pnl': pnl_percent,
                    'pnl_dollar': pnl_dollar,
                    'reason': close_reason
                })
                
                emoji = "🟢" if pnl_percent > 0 else "🔴"
                print(f"   {emoji} {ticker} closed: {pnl_percent:+.2f}% (${pnl_dollar:+.2f}) - {close_reason}")
        
        conn.commit()
        conn.close()
        
        return closed_trades
    
    def get_open_positions(self) -> List[Dict]:
        """Alle offenen Positionen"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM trades WHERE status = 'open' ORDER BY entry_time DESC
        ''')
        
        positions = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return positions
    
    def get_stats(self) -> Dict:
        """Einfache Statistiken"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Gesamtstatistik
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN status = 'closed' THEN 1 ELSE 0 END) as closed,
                SUM(CASE WHEN status = 'open' THEN 1 ELSE 0 END) as open,
                SUM(CASE WHEN pnl_percent > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN pnl_percent <= 0 THEN 1 ELSE 0 END) as losses,
                AVG(CASE WHEN status = 'closed' THEN pnl_percent END) as avg_pnl,
                SUM(CASE WHEN status = 'closed' THEN pnl_dollar END) as total_pnl_dollar
            FROM trades
        ''')
        
        row = cursor.fetchone()
        conn.close()
        
        total_closed = row[1] or 0
        wins = row[3] or 0
        
        return {
            'total_trades': row[0] or 0,
            'closed_trades': total_closed,
            'open_trades': row[2] or 0,
            'wins': wins,
            'losses': row[4] or 0,
            'win_rate': (wins / total_closed * 100) if total_closed > 0 else 0,
            'avg_pnl': row[5] or 0,
            'total_pnl_dollar': row[6] or 0
        }