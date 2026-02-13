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
    
    # Aktuelle Preise holen (Live von yfinance)
    import yfinance as yf
    result = []
    for pos in positions:
        pos_dict = dict(pos)
        try:
            # Echten aktuellen Preis holen
            stock = yf.Ticker(pos_dict['ticker'])
            info = stock.info
            current = info.get('regularMarketPrice', pos_dict['entry'])
            pos_dict['current'] = round(current, 2)
        except:
            # Fallback auf Entry-Preis wenn API nicht erreichbar
            pos_dict['current'] = pos_dict['entry']
        
        # P&L berechnen
        if pos_dict['type'] == 'buy':
            pnl = ((pos_dict['current'] - pos_dict['entry']) / pos_dict['entry']) * pos_dict['leverage'] * 100
        else:
            pnl = ((pos_dict['entry'] - pos_dict['current']) / pos_dict['entry']) * pos_dict['leverage'] * 100
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
    """Watchlist mit aktuellen Preisen von yfinance"""
    import yfinance as yf
    
    tickers = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'AVGO', 'JPM']
    watchlist = []
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            current_price = info.get('regularMarketPrice', 0)
            previous_close = info.get('regularMarketPreviousClose', 0)
            
            if current_price and previous_close:
                change_pct = ((current_price - previous_close) / previous_close) * 100
            else:
                change_pct = 0
            
            watchlist.append({
                "ticker": ticker,
                "price": round(current_price, 2),
                "change": round(change_pct, 2)
            })
        except Exception as e:
            # Fallback wenn Daten nicht verfügbar
            watchlist.append({
                "ticker": ticker,
                "price": 0,
                "change": 0
            })
    
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
    """Portfolio Performance über Zeit - aus DB generiert"""
    conn = get_db_connection()
    
    # Alle geschlossenen Trades nach Zeit sortiert
    trades = conn.execute(
        """SELECT exit_time, pnl_percent 
           FROM trades 
           WHERE status = 'closed' 
           ORDER BY exit_time ASC"""
    ).fetchall()
    
    conn.close()
    
    if not trades:
        # Keine Trades = flache Linie bei Startkapital
        return jsonify([{"time": "Start", "value": 10000}])
    
    # Performance-Kurve berechnen
    initial_capital = 10000
    data = []
    current_value = initial_capital
    
    for i, trade in enumerate(trades):
        pnl_decimal = (trade['pnl_percent'] or 0) / 100
        current_value = current_value * (1 + pnl_decimal)
        
        # Zeit formatieren
        exit_time = trade['exit_time']
        if exit_time:
            time_str = exit_time.split(' ')[1][:5] if ' ' in str(exit_time) else str(i)
        else:
            time_str = str(i)
        
        data.append({
            "time": time_str,
            "value": round(current_value, 2)
        })
    
    return jsonify(data)

@app.route('/api/learning', methods=['GET'])
def get_learning_status():
    """Adaptive Learning Status"""
    import sys
    sys.path.insert(0, 'src')
    from adaptive_optimizer import AdaptiveOptimizer
    
    optimizer = AdaptiveOptimizer()
    adjustments = optimizer.get_strategy_adjustments()
    
    return jsonify({
        "scores": adjustments['strategy_scores'],
        "recommendations": adjustments['recommendations'],
        "parameter_updates": adjustments['parameter_updates'],
        "timestamp": adjustments['timestamp']
    })

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