# TradeForge - Autonomer Trading Analyst

Ein autonomer Trading-Agent, der:
- Top 10 S&P 500 Werte analysiert
- Technische Strategien entwickelt und testet
- Paper-Trading mit virtuellen Trades durchführt
- Aus Trades lernt und Strategien optimiert
- Tägliche Reports generiert

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Nutzung

```bash
# Einmalige Analyse + Paper-Trading
python src/main.py

# Kontinuierlicher Modus (alle 15 Minuten)
python src/main.py --continuous

# Nur Report generieren
python src/main.py --report-only
```

## Konfiguration

Bearbeite `config.yaml`:
- `watchlist`: Zu überwachende Ticker
- `initial_capital`: Virtuelles Startkapital
- `risk_per_trade`: Risiko pro Trade (%)
- `leverage_max`: Maximaler Hebel

## Struktur

```
tradeforge/
├── src/
│   ├── main.py              # Hauptloop
│   ├── data_fetcher.py      # Marktdaten abrufen
│   ├── analyzer.py          # Technische Analyse
│   ├── strategy_engine.py   # Strategie-Logik
│   ├── paper_trader.py      # Virtuelles Trading
│   ├── learning_engine.py   # Auswertung & Optimierung
│   └── reporter.py          # Report-Generierung
├── strategies/              # Strategie-Definitionen
├── data/
│   ├── trades.db           # SQLite Datenbank
│   └── market_data/        # Historische Daten
└── reports/                # Generierte Reports
```

## Disclaimer

⚠️ **Dies ist ein Forschungsprojekt für Paper-Trading. Keine Finanzberatung. Der Agent lernt aus virtuellen Trades - es gibt keine Garantie für erfolgreiche Strategien im Live-Trading.**