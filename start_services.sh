#!/bin/bash
# TradeForge Startup Script
# Startet alle Services dauerhaft mit tmux

echo "🚀 TradeForge Services starten..."

# Alles killen
pkill -9 python 2>/dev/null
sleep 2

# Datenbank prüfen/erstellen
if [ ! -f "/home/dev/.openclaw/workspace/tradeforge/data/trades.db" ]; then
    echo "📊 Erstelle neue Datenbank..."
    cd /home/dev/.openclaw/workspace/tradeforge
    source venv/bin/activate
    python3 -c "
import sqlite3
conn = sqlite3.connect('data/trades.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE trades (
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
print('✅ Datenbank erstellt')
"
fi

# API Server
echo "🌐 Starte API Server..."
tmux kill-session -t tradeforge-api 2>/dev/null
tmux new-session -d -s tradeforge-api "cd /home/dev/.openclaw/workspace/tradeforge && source venv/bin/activate && python src/api_server.py"

sleep 2

# Dashboard
echo "📱 Starte Dashboard..."
tmux kill-session -t tradeforge-dashboard 2>/dev/null
tmux new-session -d -s tradeforge-dashboard "cd /home/dev/.openclaw/workspace/tradeforge-dashboard/dist && python3 -m http.server 3000 --bind 100.113.212.123"

sleep 2

# Trading Agent
echo "🤖 Starte Trading Agent..."
tmux kill-session -t tradeforge-agent 2>/dev/null
tmux new-session -d -s tradeforge-agent "cd /home/dev/.openclaw/workspace/tradeforge && source venv/bin/activate && python src/main_pro.py"

echo ""
echo "✅ Alle Services gestartet!"
echo ""
echo "Status:"
tmux list-sessions 2>/dev/null | grep tradeforge || echo "Keine Sessions"
echo ""
echo "URLs:"
echo "  Dashboard: http://100.113.212.123:3000"
echo "  API: http://100.113.212.123:5000"
echo ""
echo "Logs:"
echo "  tail -f /tmp/api.log"
echo "  tail -f /tmp/dashboard.log"