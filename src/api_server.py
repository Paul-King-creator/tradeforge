"""
API Server für TradeForge Dashboard
Stellt Endpoints bereit für Live-Daten
"""
from flask import Flask, jsonify
from flask_cors import CORS
import sqlite3
import json
from datetime import datetime, timedelta
import threading
import time

app = Flask(__name__)
CORS(app)  # Erlaubt Cross-Origin Requests vom Dashboard

DB_PATH = "data/trades.db"

def get_db_connection():
    """SQLite Verbindung"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/api/portfolio', methods=['GET'])
def get_portfolio():
    """Portfolio-Übersicht"""
    conn = get_db_connection()
    
    # Offene Positionen
    open_positions = conn.execute(
        "SELECT * FROM trades WHERE status = 'open'"
    ).fetchall()
    
    # Heutige geschlossene Trades
    today = datetime.now().date()
    closed_today = conn.execute(
        """SELECT * FROM trades 
           WHERE status = 'closed' AND DATE(exit_time) = ?""",
        (today,)
    ).fetchall()
    
    # Performance berechnen
    initial_capital = 10000
    total_pnl = sum(t['pnl_percent'] or 0 for t in closed_today)
    current_value = initial_capital * (1 + total_pnl / 100)
    
    conn.close()
    
    return jsonify({
        "totalValue": round(current_value, 2),
        "dayChange": round(current_value - initial_capital, 2),
        "dayChangePercent": round(total_pnl, 2),
        "totalReturn": round(total_pnl, 2),
        "openPositions": len(open_positions),
        "closedToday": len(closed_today),
        "initialCapital": initial_capital
    })

@app.route('/api/positions', methods=['GET'])
def get_positions():
    """Alle offenen Positionen"""
    conn = get_db_connection()
    
    positions = conn.execute(
        """SELECT id, ticker, signal_type as type, strategy, 
                  entry_price as entry, stop_loss as stopLoss, 
                  take_profit as takeProfit, leverage, 
                  confidence, setup_description as setupDescription,
                  entry_time as time
           FROM trades WHERE status = 'open' ORDER BY entry_time DESC"""
    ).fetchall()
    
    # Aktuelle Preise holen (Mock für jetzt, später echt)
    result = []
    for pos in positions:
        pos_dict = dict(pos)
        # Simulierter aktueller Preis (±2% vom Einstieg)
        import random
        variation = random.uniform(-0.02, 0.02)
        current = pos_dict['entry'] * (1 + variation)
        pos_dict['current'] = round(current, 2)
        
        # P&L berechnen
        if pos_dict['type'] == 'buy':
            pnl = ((current - pos_dict['entry']) / pos_dict['entry']) * pos_dict['leverage'] * 100
        else:
            pnl = ((pos_dict['entry'] - current) / pos_dict['entry']) * pos_dict['leverage'] * 100
        pos_dict['pnl'] = round(pnl, 2)
        
        result.append(pos_dict)
    
    conn.close()
    return jsonify(result)

@app.route('/api/trades/today', methods=['GET'])
def get_today_trades():
    """Alle Trades von heute"""
    conn = get_db_connection()
    today = datetime.now().date()
    
    trades = conn.execute(
        """SELECT ticker, signal_type as type, entry_price as price, 
                  pnl_percent as pnl, status
           FROM trades 
           WHERE DATE(entry_time) = ? OR DATE(exit_time) = ?
           ORDER BY entry_time DESC""",
        (today, today)
    ).fetchall()
    
    conn.close()
    return jsonify([dict(t) for t in trades])

@app.route('/api/watchlist', methods=['GET'])
def get_watchlist():
    """Watchlist mit aktuellen Preisen"""
    watchlist = [
        {"ticker": "NVDA", "price": 892.40, "change": 2.15},
        {"ticker": "AAPL", "price": 187.85, "change": 0.89},
        {"ticker": "MSFT", "price": 422.10, "change": 0.45},
        {"ticker": "AMZN", "price": 180.50, "change": 1.25},
        {"ticker": "GOOGL", "price": 175.20, "change": -0.35},
        {"ticker": "META", "price": 498.50, "change": 1.92},
        {"ticker": "TSLA", "price": 172.30, "change": -3.45},
        {"ticker": "AVGO", "price": 1385.20, "change": 0.78},
    ]
    return jsonify(watchlist)

@app.route('/api/strategies', methods=['GET'])
def get_strategy_stats():
    """Performance pro Strategie"""
    conn = get_db_connection()
    
    stats = conn.execute(
        """SELECT strategy as name, 
                  COUNT(*) as trades,
                  SUM(CASE WHEN pnl_percent > 0 THEN 1 ELSE 0 END) as wins,
                  AVG(pnl_percent) as avgReturn
           FROM trades 
           WHERE status = 'closed'
           GROUP BY strategy"""
    ).fetchall()
    
    conn.close()
    
    result = []
    for s in stats:
        s_dict = dict(s)
        s_dict['winRate'] = round((s_dict['wins'] / s_dict['trades'] * 100) if s_dict['trades'] > 0 else 0, 1)
        s_dict['avgReturn'] = round(s_dict['avgReturn'] or 0, 2)
        result.append(s_dict)
    
    return jsonify(result)

@app.route('/api/performance', methods=['GET'])
def get_performance():
    """Portfolio Performance über Zeit"""
    # Dies würde normalerweise aus der DB kommen
    # Für jetzt: Mock-Daten basierend auf Trades
    data = [
        {"time": "09:30", "value": 10000},
        {"time": "10:00", "value": 10120},
        {"time": "10:30", "value": 10050},
        {"time": "11:00", "value": 10230},
        {"time": "11:30", "value": 10410},
        {"time": "12:00", "value": 10520},
        {"time": "12:30", "value": 10680},
        {"time": "13:00", "value": 10750},
        {"time": "13:30", "value": 10847},
    ]
    return jsonify(data)

@app.route('/api/status', methods=['GET'])
def get_status():
    """Agent Status"""
    return jsonify({
        "status": "active",
        "lastUpdate": datetime.now().isoformat(),
        "version": "1.0.0"
    })

def run_api_server(host='0.0.0.0', port=5000):
    """Startet den API Server"""
    print(f"🌐 TradeForge API Server starting on http://{host}:{port}")
    app.run(host=host, port=port, debug=False)

if __name__ == '__main__':
    run_api_server()