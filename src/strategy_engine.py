"""
Strategy Engine - Trading Strategien
Mit erweiterter Chart-Analyse (Patterns, Support/Resistance, VWAP)
"""
import pandas as pd
from typing import List, Optional
from analyzer import Signal, TechnicalAnalyzer

# Versuche Advanced Analyzer zu importieren
try:
    from advanced_analyzer import AdvancedAnalyzer, PatternType
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False

class StrategyEngine:
    def __init__(self, config: dict):
        self.config = config
        self.analyzer = TechnicalAnalyzer()
        self.advanced_analyzer = AdvancedAnalyzer() if ADVANCED_AVAILABLE else None
        self.signals = []
    
    def analyze_all_strategies(self, ticker: str, df: pd.DataFrame) -> List[Signal]:
        """Alle aktiven Strategien auf Ticker anwenden"""
        signals = []
        
        # Daten vorbereiten
        df = self.analyzer.add_indicators(df)
        
        if len(df) < 50:
            return signals
        
        # Erweiterte Analyse (wenn verfügbar)
        advanced_data = None
        if self.advanced_analyzer:
            try:
                advanced_data = self.advanced_analyzer.analyze(df)
            except Exception as e:
                print(f"   ⚠️  Advanced analysis failed: {e}")
        
        # Strategien ausführen
        if self.config['strategies']['momentum']['enabled']:
            sig = self._momentum_strategy(ticker, df, advanced_data)
            if sig:
                signals.append(sig)
        
        if self.config['strategies']['breakout']['enabled']:
            sig = self._breakout_strategy(ticker, df, advanced_data)
            if sig:
                signals.append(sig)
        
        if self.config['strategies']['mean_reversion']['enabled']:
            sig = self._mean_reversion_strategy(ticker, df, advanced_data)
            if sig:
                signals.append(sig)
        
        # Pattern-basierte Signale (NEU)
        if advanced_data and advanced_data.get('patterns'):
            for pattern_data in advanced_data['patterns']:
                sig = self._pattern_based_signal(ticker, df, pattern_data, advanced_data)
                if sig:
                    signals.append(sig)
        
        return signals
    
    def _momentum_strategy(self, ticker: str, df: pd.DataFrame, advanced_data=None) -> Optional[Signal]:
        """Momentum Strategie - Trendfolge mit RSI + erweiterte Filter"""
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        
        trend = self.analyzer.analyze_trend(df)
        
        # Basis-Bedingungen
        long_condition = (
            trend in ['strong_uptrend', 'uptrend'] and
            last['RSI'] > 30 and prev['RSI'] <= 30 and
            last['MACD'] > last['MACD_Signal'] and
            last['Volume_Ratio'] > 1.2
        )
        
        # Erweiterte Filter (wenn verfügbar)
        if advanced_data and long_condition:
            # VWAP Filter: Preis sollte über VWAP sein (bullisch)
            if advanced_data['vwap']['above']:
                long_condition = True
                vwap_info = f"VWAP: +{advanced_data['vwap']['position']:.1f}%"
            else:
                # Unter VWAP ist vorsichtiger
                long_condition = False
            
            # Market Structure Filter
            if advanced_data['market_structure']['trend'] != 'bullish':
                long_condition = False
        else:
            vwap_info = ""
        
        if long_condition:
            stop_loss = last['Close'] - (last['ATR'] * 1.5)
            take_profit = last['Close'] + (last['ATR'] * 3)
            
            # Verwende Support Level als Stop Loss wenn verfügbar
            if advanced_data and advanced_data['support_resistance']['support']:
                nearest_support = max([s for s in advanced_data['support_resistance']['support'] if s < last['Close']], default=None)
                if nearest_support and nearest_support < last['Close'] * 0.98:
                    stop_loss = nearest_support
            
            setup_desc = f"Momentum: RSI={last['RSI']:.1f}, Trend={trend}"
            if vwap_info:
                setup_desc += f", {vwap_info}"
            
            return Signal(
                ticker=ticker,
                signal_type='buy',
                confidence=self._calculate_confidence(df, 'long'),
                strategy='momentum_long',
                entry_price=last['Close'],
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=3,
                setup_description=setup_desc,
                timestamp=last.name
            )
        
        return None
    
    def _breakout_strategy(self, ticker: str, df: pd.DataFrame, advanced_data=None) -> Optional[Signal]:
        """Breakout Strategie mit Volume Profile"""
        last = df.iloc[-1]
        high_20 = df['High'].tail(20).max()
        
        # Basis Breakout Bedingung
        breakout_condition = (
            last['Close'] > high_20 * 0.995 and 
            last['Volume_Ratio'] > 1.5
        )
        
        # Erweiterte Filter
        if advanced_data and breakout_condition:
            # VWAP Filter: Bei Breakout sollten wir über VWAP sein
            if not advanced_data['vwap']['above']:
                breakout_condition = False
        
        if breakout_condition:
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
    
    def _mean_reversion_strategy(self, ticker: str, df: pd.DataFrame, advanced_data=None) -> Optional[Signal]:
        """Mean Reversion Strategie mit Support Levels"""
        last = df.iloc[-1]
        
        # Basis Mean Reversion
        reversion_condition = (
            last['Close'] <= last['BB_Lower'] * 1.01 and 
            last['RSI'] < 30
        )
        
        # Erweiterte Filter
        if advanced_data and reversion_condition:
            # VWAP: Unter VWAP ist bärish für Mean Reversion Long
            if advanced_data['vwap']['above']:
                reversion_condition = False
        
        if reversion_condition:
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
    
    def _pattern_based_signal(self, ticker: str, df: pd.DataFrame, pattern_data: dict, advanced_data: dict) -> Optional[Signal]:
        """Erzeugt Signal basierend auf erkanntem Chart Pattern"""
        last = df.iloc[-1]
        
        pattern_type = pattern_data['type']
        confidence = pattern_data['confidence']
        
        # Nur hochkonfidente Patterns traden
        if confidence < 0.7:
            return None
        
        # Pattern-spezifische Logik
        if pattern_type == 'double_bottom':
            # Bullish Pattern - Long Signal
            return Signal(
                ticker=ticker,
                signal_type='buy',
                confidence=confidence,
                strategy='pattern_double_bottom',
                entry_price=last['Close'],
                stop_loss=pattern_data['stop_loss'],
                take_profit=pattern_data['target'],
                leverage=3,
                setup_description=f"Pattern: {pattern_data['description']}",
                timestamp=last.name
            )
        
        elif pattern_type == 'head_and_shoulders':
            # Bearish Pattern - Short Signal
            return Signal(
                ticker=ticker,
                signal_type='sell',
                confidence=confidence,
                strategy='pattern_head_shoulders',
                entry_price=last['Close'],
                stop_loss=pattern_data['stop_loss'],
                take_profit=pattern_data['target'],
                leverage=2,
                setup_description=f"Pattern: {pattern_data['description']}",
                timestamp=last.name
            )
        
        elif pattern_type == 'triangle_ascending':
            # Bullish Breakout erwartet
            return Signal(
                ticker=ticker,
                signal_type='buy',
                confidence=confidence,
                strategy='pattern_triangle_asc',
                entry_price=last['Close'],
                stop_loss=pattern_data['stop_loss'],
                take_profit=pattern_data['target'],
                leverage=2,
                setup_description=f"Pattern: {pattern_data['description']}",
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