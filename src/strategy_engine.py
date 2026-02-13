"""
Strategy Engine - Trading Strategien
"""
import pandas as pd
from typing import List, Optional
from analyzer import Signal, TechnicalAnalyzer

class StrategyEngine:
    def __init__(self, config: dict):
        self.config = config
        self.analyzer = TechnicalAnalyzer()
        self.signals = []
    
    def analyze_all_strategies(self, ticker: str, df: pd.DataFrame) -> List[Signal]:
        """Alle aktiven Strategien auf Ticker anwenden"""
        signals = []
        
        # Daten vorbereiten
        df = self.analyzer.add_indicators(df)
        
        if len(df) < 50:
            return signals
        
        # Strategien ausführen
        if self.config['strategies']['momentum']['enabled']:
            sig = self._momentum_strategy(ticker, df)
            if sig:
                signals.append(sig)
        
        if self.config['strategies']['breakout']['enabled']:
            sig = self._breakout_strategy(ticker, df)
            if sig:
                signals.append(sig)
        
        if self.config['strategies']['mean_reversion']['enabled']:
            sig = self._mean_reversion_strategy(ticker, df)
            if sig:
                signals.append(sig)
        
        return signals
    
    def _momentum_strategy(self, ticker: str, df: pd.DataFrame) -> Optional[Signal]:
        """Momentum Strategie - Trendfolge mit RSI"""
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        
        trend = self.analyzer.analyze_trend(df)
        
        # Long Signal: Aufwärtstrend + RSI aus überverkauft + MACD Kreuzung
        long_condition = (
            trend in ['strong_uptrend', 'uptrend'] and
            last['RSI'] > 30 and prev['RSI'] <= 30 and
            last['MACD'] > last['MACD_Signal'] and
            last['Close'] > last['VWAP'] and
            last['Volume_Ratio'] > 1.2
        )
        
        if long_condition:
            stop_loss = last['Close'] - (last['ATR'] * 1.5)
            take_profit = last['Close'] + (last['ATR'] * 3)
            
            return Signal(
                ticker=ticker,
                signal_type='buy',
                confidence=self._calculate_confidence(df, 'long'),
                strategy='momentum_long',
                entry_price=last['Close'],
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=5,
                setup_description=f"Momentum Long: RSI={last['RSI']:.1f}, Trend={trend}",
                timestamp=last.name
            )
        
        return None
    
    def _breakout_strategy(self, ticker: str, df: pd.DataFrame) -> Optional[Signal]:
        """Breakout Strategie"""
        last = df.iloc[-1]
        high_20 = df['High'].tail(20).max()
        
        if last['Close'] > high_20 * 0.995 and last['Volume_Ratio'] > 1.5:
            stop_loss = last['Close'] - (last['ATR'] * 1.5)
            take_profit = last['Close'] + (last['ATR'] * 2.5)
            
            return Signal(
                ticker=ticker,
                signal_type='buy',
                confidence=0.7,
                strategy='breakout_long',
                entry_price=last['Close'],
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=3,
                setup_description=f"Breakout über {high_20:.2f}",
                timestamp=last.name
            )
        
        return None
    
    def _mean_reversion_strategy(self, ticker: str, df: pd.DataFrame) -> Optional[Signal]:
        """Mean Reversion Strategie"""
        last = df.iloc[-1]
        
        if last['Close'] <= last['BB_Lower'] * 1.01 and last['RSI'] < 25:
            stop_loss = last['Close'] - (last['ATR'] * 2)
            take_profit = last['BB_Middle']
            
            return Signal(
                ticker=ticker,
                signal_type='buy',
                confidence=0.6,
                strategy='mean_reversion_long',
                entry_price=last['Close'],
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=2,
                setup_description=f"Mean Reversion: RSI={last['RSI']:.1f}",
                timestamp=last.name
            )
        
        return None
    
    def _calculate_confidence(self, df: pd.DataFrame, direction: str) -> float:
        """Signal-Konfidenz berechnen"""
        last = df.iloc[-1]
        score = 0.5
        
        if direction == 'long':
            if last['RSI'] > 30 and last['RSI'] < 50:
                score += 0.1
            if last['Volume_Ratio'] > 1.5:
                score += 0.1
            if last['Close'] > last['EMA_9']:
                score += 0.1
        
        return min(score, 0.95)