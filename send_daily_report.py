#!/usr/bin/env python3
"""
Daily Report Sender für TradeForge
Postet automatisch den Tagesreport in Telegram
"""
import sys
import sqlite3
import json
from datetime import datetime, timedelta

# Füge src hinzu
sys.path.insert(0, '/home/dev/.openclaw/workspace/tradeforge/src')

def get_report_data():
    """Holt alle Daten für den Report"""
    conn = sqlite3.connect('/home/dev/.openclaw/workspace/tradeforge/data/trades.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Portfolio Daten
    cursor.execute("SELECT COALESCE(SUM(pnl_percent), 0) FROM trades WHERE status = 'closed'")
    total_pnl = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'open'")
    open_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'closed'")
    closed_count = cursor.fetchone()[0]
    
    cursor.execute("""
        SELECT COUNT(*) FROM trades 
        WHERE status = 'closed' AND pnl_percent > 0
    """)
    wins = cursor.fetchone()[0]
    
    cursor.execute("""
        SELECT COUNT(*) FROM trades 
        WHERE status = 'closed' AND pnl_percent <= 0
    """)
    losses = cursor.fetchone()[0]
    
    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
    
    # Performance heute
    today = datetime.now().date()
    cursor.execute("""
        SELECT COUNT(*), COALESCE(SUM(pnl_percent), 0)
        FROM trades 
        WHERE status = 'closed' AND DATE(exit_time) = ?
    """, (today,))
    today_data = cursor.fetchone()
    today_trades = today_data[0]
    today_pnl = today_data[1] or 0
    
    # Offene Positionen
    cursor.execute("""
        SELECT ticker, entry_price, strategy, pnl_percent
        FROM trades 
        WHERE status = 'open'
        ORDER BY entry_time DESC
        LIMIT 5
    """)
    positions = cursor.fetchall()
    
    conn.close()
    
    return {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'time': datetime.now().strftime('%H:%M'),
        'portfolio_value': 10000 * (1 + total_pnl/100),
        'total_return': total_pnl,
        'open_positions': open_count,
        'closed_positions': closed_count,
        'win_rate': win_rate,
        'wins': wins,
        'losses': losses,
        'today_trades': today_trades,
        'today_pnl': today_pnl,
        'positions': positions
    }

def format_report(data):
    """Formatiert den Report für Telegram"""
    msg = f"""🤖 <b>TradeForge Daily Report</b>
📅 {data['date']} • {data['time']} UTC

💰 <b>Portfolio:</b> ${data['portfolio_value']:.2f} ({data['total_return']:+.2f}%)
📊 <b>Heute:</b> {data['today_trades']} trades • {data['today_pnl']:+.2f}%

📈 <b>Stats:</b>
• Open: {data['open_positions']} positions
• Closed: {data['closed_positions']} total
• Win Rate: {data['win_rate']:.1f}% ({data['wins']}W / {data['losses']}L)"""
    
    if data['positions']:
        msg += "\n\n📍 <b>Open Positions:</b>"
        for pos in data['positions']:
            pnl = pos[3] if pos[3] else 0
            emoji = "🟢" if pnl >= 0 else "🔴"
            msg += f"\n{emoji} {pos[0]} @ ${pos[1]:.2f} ({pos[2][:10]})"
        
        if len(data['positions']) == 5 and data['open_positions'] > 5:
            msg += f"\n   ... and {data['open_positions'] - 5} more"
    
    msg += "\n\n⏰ Nächster Report: Morgen 18:00 UTC"
    msg += "\n📊 Dashboard: http://100.113.212.123:3000"
    
    return msg

if __name__ == "__main__":
    try:
        data = get_report_data()
        report = format_report(data)
        print(report)
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        sys.exit(1)