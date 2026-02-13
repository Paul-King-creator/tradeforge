"""
TradeForge Pro - Qualitäts-orientierter Trading Agent
Fokus: Gute Trades lernen, nicht viele Trades machen
"""
import os
import sys
import yaml
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from data_fetcher import DataFetcher
from analyzer import TechnicalAnalyzer
from quality_strategy import QualityStrategyEngine, Signal
from simple_trader import SimplePaperTrader
from reporter import Reporter

# ML Imports
try:
    from ml_optimizer import MLStrategyOptimizer, ML_AVAILABLE
    if ML_AVAILABLE:
        print("🤖 ML-Optimizer bereit")
except ImportError:
    ML_AVAILABLE = False
    MLStrategyOptimizer = None
    print("⚠️  ML nicht verfügbar")

class TradeForgePro:
    """
    Pro Version:
    - Qualität > Quantität
    - Fix $100 pro Trade
    - ML-gesteuerte Entscheidungen
    - Lernen von jedem Trade
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_fetcher = DataFetcher()
        self.analyzer = TechnicalAnalyzer()
        self.strategy = QualityStrategyEngine(self.config)
        self.trader = SimplePaperTrader()
        self.reporter = Reporter()
        
        # ML
        self.ml_optimizer = None
        if ML_AVAILABLE:
            self.ml_optimizer = MLStrategyOptimizer()
            print("✅ ML System initialisiert")
        
        self.run_count = 0
        self.last_trade_date = None
        
        print("🚀 TradeForge Pro gestartet")
        print("   Fokus: Qualität statt Quantität")
        print("   Position Size: $100 fix")
    
    def run_analysis(self):
        """Ein Analyse-Durchlauf"""
        print(f"\n{'='*60}")
        print(f"🔍 Analyse #{self.run_count + 1} - {datetime.now().strftime('%H:%M')}")
        print(f"{'='*60}\n")
        
        # Tageswechsel checken
        today = datetime.now().date()
        if self.last_trade_date != today:
            self.strategy.reset_daily()
            self.last_trade_date = today
            print("📅 Neuer Tag - Trade-Counter reset")
        
        # 1. Marktdaten laden
        print("📊 Lade Marktdaten...")
        market_data = self.data_fetcher.get_multiple_data(
            self.config['watchlist'],
            period="2d",
            interval="15m"
        )
        
        if not market_data:
            print("❌ Keine Daten")
            return
        
        print(f"✅ {len(market_data)} Aktien geladen\n")
        
        # 2. Offene Positionen prüfen
        print("🔍 Prüfe Positionen...")
        closed = self.trader.check_and_close_positions(market_data)
        if closed:
            print(f"📤 {len(closed)} Positionen geschlossen")
            for c in closed:
                print(f"   {c['ticker']}: {c['pnl']:+.2f}%")
        
        # 3. Neue Signale suchen
        print("\n🔮 Suche nach Qualitäts-Setups...")
        new_trades = 0
        
        for ticker, df in market_data.items():
            # Indikatoren hinzufügen
            df = self.analyzer.add_indicators(df)
            
            # ML Vorhersage (wenn verfügbar)
            ml_pred = None
            if self.ml_optimizer and hasattr(self.ml_optimizer, 'predict_trade_success'):
                from ml_optimizer import TradeFeatures
                features = TradeFeatures(
                    ticker=ticker,
                    strategy='quality_setup',
                    entry_price=df['close'].iloc[-1],
                    leverage=2,
                    confidence=0.7
                )
                ml_pred = self.ml_optimizer.predict_trade_success(features)
            
            # Strategie analysieren
            signal = self.strategy.analyze(ticker, df, ml_pred)
            
            if signal:
                print(f"\n   🎯 SETUP GEFUNDEN: {ticker}")
                print(f"      Strategy: {signal.strategy}")
                print(f"      Confidence: {signal.confidence:.0%}")
                if ml_pred:
                    print(f"      ML Prediction: {ml_pred['success_probability']:.0%}")
                print(f"      Entry: ${signal.entry_price:.2f}")
                print(f"      SL/TP: ${signal.stop_loss:.2f} / ${signal.take_profit:.2f}")
                print(f"      R:R: 1:{(signal.take_profit - signal.entry_price) / (signal.entry_price - signal.stop_loss):.1f}")
                
                # Trade öffnen
                trade_id = self.trader.open_position(signal)
                if trade_id:
                    print(f"      ✅ Trade #{trade_id} eröffnet")
                    new_trades += 1
        
        if new_trades == 0:
            print("   ℹ️  Keine qualitativ hochwertigen Setups gefunden")
        else:
            print(f"\n📈 {new_trades} neue Trades eröffnet")
        
        # 4. Stats zeigen
        stats = self.trader.get_stats()
        print(f"\n📊 Stats:")
        print(f"   Total: {stats['total_trades']} | Closed: {stats['closed_trades']} | Open: {stats['open_trades']}")
        if stats['closed_trades'] > 0:
            print(f"   Win Rate: {stats['win_rate']:.1f}% ({stats['wins']}W / {stats['losses']}L)")
            print(f"   Avg P&L: {stats['avg_pnl']:+.2f}%")
            print(f"   Total $: ${stats['total_pnl_dollar']:+.2f}")
        
        self.run_count += 1
    
    def generate_report(self):
        """Tagesreport"""
        print(f"\n{'='*60}")
        print("📋 TAGESREPORT")
        print(f"{'='*60}\n")
        
        stats = self.trader.get_stats()
        
        print(f"📊 Trading Stats:")
        print(f"   Total Trades: {stats['total_trades']}")
        print(f"   Closed: {stats['closed_trades']}")
        print(f"   Open: {stats['open_trades']}")
        print(f"   Win Rate: {stats['win_rate']:.1f}%")
        print(f"   Avg Return: {stats['avg_pnl']:+.2f}%")
        print(f"   Total P&L: ${stats['total_pnl_dollar']:+.2f}")
        
        # ML Training
        if self.ml_optimizer:
            print("\n🤖 ML Training...")
            trained = self.ml_optimizer.train_models()
            if trained:
                print(self.ml_optimizer.generate_ml_report())
            else:
                print(f"   Brauche noch {20 - stats['closed_trades']} geschlossene Trades für ML")
        
        # Offene Positionen
        open_pos = self.trader.get_open_positions()
        if open_pos:
            print(f"\n📍 Offene Positionen ({len(open_pos)}):")
            for pos in open_pos[:5]:
                print(f"   {pos['ticker']}: ${pos['entry_price']:.2f} ({pos['strategy']})")
        
        print(f"\n{'='*60}")
    
    def run_continuous(self):
        """Hauptschleife"""
        print("\n🔄 KONTINUIERLICHER MODUS")
        print("   Alle 15 Minuten: Check auf Setups")
        print("   Um 18:00: Tagesreport\n")
        
        last_report_date = None
        
        try:
            while True:
                now = datetime.now()
                
                # Tagesreport um 18:00
                report_time = datetime.strptime("18:00", "%H:%M").time()
                if now.time() >= report_time and last_report_date != now.date():
                    self.generate_report()
                    last_report_date = now.date()
                
                # Analyse
                self.run_analysis()
                
                # Warten
                print(f"\n⏳ Nächster Check in 15 Minuten...")
                time.sleep(15 * 60)
                
        except KeyboardInterrupt:
            print("\n\n👋 Beendet")
            self.generate_report()

def main():
    forge = TradeForgePro()
    forge.run_continuous()

if __name__ == "__main__":
    main()