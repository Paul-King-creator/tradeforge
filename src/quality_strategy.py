"""
Quality-First Strategy Engine
Nur die besten Setups traden
"""
import pandas as pd
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Signal:
    ticker: str
    signal_type: str  # 'buy' or 'sell'
    confidence: float
    strategy: str
    entry_price: float
    stop_loss: float
    take_profit: float
    leverage: int
    setup_description: str
    timestamp: datetime
    ml_probability: float = 0.0  # NEU: ML Vorhersage

class QualityStrategyEngine:
    """
    Fokus auf hochqualitative Trades:
    - Mindestens 2 konfluierende Signale
    - ML Probability > 60%
    - Gutes Risk/Reward (> 1:2)
    - Kein Overtrading (max 1 Trade pro Ticker pro Tag)
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.min_confidence = 0.7  # Höher!
        self.min_ml_probability = 0.6  # ML muss 60%+ sagen
        self.min_risk_reward = 2.0  # Mindestens 1:2
        self.today_trades = {}  # Ticker -> Anzahl heute
    
    def analyze(self, ticker: str, df: pd.DataFrame, ml_prediction: dict = None) -> Optional[Signal]:
        """
        Haupt-Analyse: Nur traden wenn ALLE Bedingungen erfüllt
        """
        if len(df) < 30:
            return None
        
        last = df.iloc[-1]
        
        # 1. Kein Overtrading (max 1 pro Tag pro Ticker)
        if self.today_trades.get(ticker, 0) >= 1:
            return None
        
        # 2. ML Prediction prüfen
        if ml_prediction:
            if not ml_prediction.get('should_trade', False):
                return None
            if ml_prediction.get('success_probability', 0) < self.min_ml_probability:
                return None
        
        # 3. Technische Analyse
        signals_found = []
        
        # Signal 1: RSI Oversold + Bounce
        if last.get('RSI', 50) < 30 and last['Close'] > last['Open']:
            signals_found.append(('RSI_OVERSOLD', 0.3))
        
        # Signal 2: MACD Kreuzung
        if last.get('MACD', 0) > last.get('MACD_Signal', 0):
            if df.iloc[-2].get('MACD', 0) <= df.iloc[-2].get('MACD_Signal', 0):
                signals_found.append(('MACD_CROSS', 0.3))
        
        # Signal 3: Bollinger Bounce
        if last.get('BB_Lower') and last['Close'] <= last['BB_Lower'] * 1.02:
            signals_found.append(('BB_BOUNCE', 0.2))
        
        # Signal 4: Volume Spike (berechnen falls nicht vorhanden)
        volume_ratio = last.get('Volume_Ratio', 1.0)
        if 'Volume' in last and len(df) > 20:
            avg_volume = df['Volume'].tail(20).mean()
            volume_ratio = last['Volume'] / avg_volume if avg_volume > 0 else 1.0
        
        if volume_ratio > 1.5:
            signals_found.append(('VOLUME_SPIKE', 0.2))
        
        # Signal 5: Trend Alignment
        trend_score = self._check_trend(df)
        if trend_score > 0:
            signals_found.append(('TREND', trend_score))
        
        # Mindestens 2 Signale müssen konfluieren
        if len(signals_found) < 2:
            return None
        
        # Gesamt-Confidence berechnen
        total_confidence = sum(s[1] for s in signals_found)
        if total_confidence < self.min_confidence:
            return None
        
        # Risk/Reward berechnen
        entry = last['Close']
        atr = last.get('ATR', entry * 0.02)
        
        # Stop Loss: ATR-basiert oder 2%
        stop_loss = entry - (atr * 1.5)
        
        # Take Profit: Mindestens 2:1 Reward
        risk = entry - stop_loss
        take_profit = entry + (risk * self.min_risk_reward)
        
        # Leverage basierend auf ML Confidence
        leverage = 2  # Default
        if ml_prediction:
            prob = ml_prediction.get('success_probability', 0.5)
            if prob > 0.75:
                leverage = 3
            elif prob > 0.85:
                leverage = 4
        
        # Setup Description
        setup_parts = [s[0] for s in signals_found]
        setup_desc = f"Quality: {' + '.join(setup_parts)} | Confidence: {total_confidence:.0%}"
        
        if ml_prediction:
            setup_desc += f" | ML: {ml_prediction['success_probability']:.0%}"
        
        # Trade Counter erhöhen
        self.today_trades[ticker] = self.today_trades.get(ticker, 0) + 1
        
        return Signal(
            ticker=ticker,
            signal_type='buy',
            confidence=min(total_confidence, 0.95),
            strategy='quality_setup',
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=leverage,
            setup_description=setup_desc,
            timestamp=last.name,
            ml_probability=ml_prediction.get('success_probability', 0.5) if ml_prediction else 0.5
        )
    
    def _check_trend(self, df: pd.DataFrame) -> float:
        """Prüft ob Trend bullisch ist"""
        if len(df) < 20:
            return 0
        
        # EMAs
        ema9 = df['Close'].ewm(span=9).mean().iloc[-1]
        ema21 = df['Close'].ewm(span=21).mean().iloc[-1]
        
        if ema9 > ema21:
            return 0.2
        return 0
    
    def reset_daily(self):
        """Reset für neuen Tag"""
        self.today_trades = {}