"""
TradeForge - Hauptmodul
Autonomer Trading Analyst für Paper Trading
"""
import os
import sys
import yaml
import argparse
import time
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from data_fetcher import DataFetcher
from analyzer import TechnicalAnalyzer
from strategy_engine import StrategyEngine
from paper_trader import PaperTrader
from reporter import Reporter
from learning_engine import LearningEngine
from adaptive_optimizer import AdaptiveOptimizer

class TradeForge:
    def __init__(self, config_path: str = "config.yaml"):
        # Konfiguration laden
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Komponenten initialisieren
        self.data_fetcher = DataFetcher()
        self.analyzer = TechnicalAnalyzer()
        self.strategy_engine = StrategyEngine(self.config)
        self.paper_trader = PaperTrader()
        self.reporter = Reporter()
        self.learning_engine = LearningEngine()
        self.adaptive_optimizer = AdaptiveOptimizer()
        
        # Tracking
        self.todays_signals = []
        self.run_count = 0
        
        print("🚀 TradeForge initialisiert")
        print(f"📊 Watchlist: {', '.join(self.config['watchlist'])}")
        print(f"💰 Virtuelles Kapital: ${self.config['trading']['initial_capital']:,}")
    
    def run_analysis_cycle(self):
        """Ein kompletter Analyse-Zyklus"""
        print(f"\n{'='*60}")
        print(f"⏰ Analyse-Run #{self.run_count + 1} - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}\n")
        
        # 1. Daten abrufen
        print("📥 Lade Marktdaten...")
        market_data = self.data_fetcher.get_watchlist_data(
            self.config['watchlist'],
            period="5d",
            interval="15m"
        )
        
        if not market_data:
            print("❌ Keine Daten geladen")
            return
        
        print(f"✅ Daten für {len(market_data)} Werte geladen\n")
        
        # 2. Offene Positionen prüfen
        print("🔍 Prüfe offene Positionen...")
        closed_trades = self.paper_trader.check_and_close_positions(market_data)
        
        if closed_trades:
            print(f"📤 {len(closed_trades)} Positionen geschlossen:")
            for trade in closed_trades:
                emoji = "🟢" if trade['pnl'] > 0 else "🔴"
                print(f"   {emoji} {trade['ticker']}: {trade['pnl']:+.2f}% ({trade['reason']})")
        else:
            print("   Keine Positionen geschlossen")
        
        # 3. Neue Signale generieren
        print("\n🔮 Generiere Trading-Signale...")
        new_signals = []
        
        for ticker, df in market_data.items():
            signals = self.strategy_engine.analyze_all_strategies(ticker, df)
            
            for signal in signals:
                # Nur Signale mit hoher Konfidenz (>0.6) handeln
                if signal.confidence >= 0.6:
                    trade_id = self.paper_trader.open_position(
                        signal,
                        capital=self.config['trading']['initial_capital'],
                        risk_per_trade=self.config['trading']['risk_per_trade']
                    )
                    
                    signal_data = {
                        'ticker': signal.ticker,
                        'signal_type': signal.signal_type,
                        'strategy': signal.strategy,
                        'entry_price': signal.entry_price,
                        'confidence': signal.confidence,
                        'setup_description': signal.setup_description,
                        'timestamp': signal.timestamp.isoformat(),
                        'trade_id': trade_id
                    }
                    new_signals.append(signal_data)
                    
                    emoji = "🟢 LONG" if signal.signal_type == 'buy' else "🔴 SHORT"
                    print(f"   {emoji} {ticker} @ ${signal.entry_price:.2f} "
                          f"({signal.strategy}, Konfidenz: {signal.confidence:.0%})")
        
        if not new_signals:
            print("   Keine neuen Signale (Markt hat keine klaren Setups)")
        
        self.todays_signals.extend(new_signals)
        
        # 4. Status anzeigen
        open_positions = self.paper_trader.get_open_positions()
        print(f"\n📊 STATUS:")
        print(f"   Offene Positionen: {len(open_positions)}")
        print(f"   Neue Signale heute: {len(self.todays_signals)}")
        
        self.run_count += 1
    
    def generate_daily_report(self):
        """Tagesreport erstellen"""
        print(f"\n{'='*60}")
        print("📝 Generiere Tagesreport...")
        print(f"{'='*60}\n")
        
        performance = self.paper_trader.get_daily_performance()
        open_positions = self.paper_trader.get_open_positions()
        strategy_stats = self.paper_trader.get_strategy_performance()
        
        report = self.reporter.generate_daily_report(
            date=datetime.now(),
            performance=performance,
            open_positions=open_positions,
            strategy_stats=strategy_stats,
            new_signals=self.todays_signals
        )
        
        self.reporter.print_console_report(report)
        
        # 🧠 ADAPTIVE LEARNING: Analyse und Optimierung
        print(f"\n{'='*60}")
        print("🧠 ADAPTIVE LEARNING ANALYSE")
        print(f"{'='*60}\n")
        
        learning_report = self.adaptive_optimizer.generate_learning_report()
        print(learning_report)
        
        # Strategie-Parameter automatisch anpassen (nach 20 Trades)
        total_trades = sum(
            stat['total_trades'] 
            for stat in strategy_stats
        )
        
        if total_trades >= 20:
            print("\n⚙️  Optimiere Strategie-Parameter...")
            new_config = self.adaptive_optimizer.apply_learning_to_config(self.config)
            
            # Wenn sich etwas geändert hat, speichern
            if new_config != self.config:
                import yaml
                with open('config.yaml', 'w') as f:
                    yaml.dump(new_config, f)
                print("✅ Neue Parameter in config.yaml gespeichert!")
                print("   (Wirksam nach Neustart des Agents)")
        
        # Signale speichern
        if self.todays_signals:
            self.reporter.save_signals_json(self.todays_signals, datetime.now())
        
        # Zurücksetzen für nächsten Tag
        self.todays_signals = []
        
        return report
    
    def run_continuous(self):
        """Kontinuierlicher Modus"""
        print("\n🔄 KONTINUIERLICHER MODUS GESTARTET")
        print(f"   Intervall: alle {self.config['intervals']['analysis']} Minuten")
        print(f"   Tagesreport um: {self.config['intervals']['report_daily']}")
        print("   Strg+C zum Beenden\n")
        
        last_report_date = None
        
        try:
            while True:
                current_time = datetime.now()
                current_date = current_time.date()
                
                # Tagesreport prüfen
                report_time = datetime.strptime(
                    self.config['intervals']['report_daily'], 
                    "%H:%M"
                ).time()
                
                if (current_time.time() >= report_time and 
                    last_report_date != current_date):
                    self.generate_daily_report()
                    last_report_date = current_date
                
                # Analyse durchführen
                self.run_analysis_cycle()
                
                # Warten bis zum nächsten Intervall
                sleep_minutes = self.config['intervals']['analysis']
                print(f"\n⏳ Nächste Analyse in {sleep_minutes} Minuten...")
                time.sleep(sleep_minutes * 60)
                
        except KeyboardInterrupt:
            print("\n\n👋 TradeForge beendet")
            self.generate_daily_report()
    
    def run_once(self):
        """Einmaliger Durchlauf"""
        self.run_analysis_cycle()
        return self.generate_daily_report()


def main():
    parser = argparse.ArgumentParser(description='TradeForge - Autonomer Trading Analyst')
    parser.add_argument('--continuous', action='store_true', help='Kontinuierlicher Modus')
    parser.add_argument('--report-only', action='store_true', help='Nur Report generieren')
    parser.add_argument('--config', default='config.yaml', help='Pfad zur Config-Datei')
    
    args = parser.parse_args()
    
    # In das Verzeichnis wechseln
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(script_dir, '..'))
    
    forge = TradeForge(config_path=args.config)
    
    if args.report_only:
        forge.generate_daily_report()
    elif args.continuous:
        forge.run_continuous()
    else:
        forge.run_once()


if __name__ == "__main__":
    main()