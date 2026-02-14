"""
True Intraday Strategy
Daytrading = Minuten bis Stunden, nicht Tage!
"""
import pandas as pd
import numpy as np
from typing import Optional, List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Signal:
    ticker: str
    signal_type: str
    confidence: float
    strategy: str
    entry_price: float
    stop_loss: float
    take_profit: float
    leverage: int
    setup_description: str
    timestamp: datetime
    
    # Intraday-spezifisch
    max_hold_time: int = 60  # Max Minuten im Trade
    target_profit: float = 0.5  # Ziel in % (0.5% = Daytrading)

class IntradayStrategy:
    """
    Echte Daytrading-Strategie:
    - Entry: Momentum- oder Mean-Reversion-Setup
    - Exit: Fixed Target (0.5-1%) oder Stop Loss
    - Hold Time: Max 60 Minuten (keine Übernachtung)
    - Keine "Take Profit auf Tage" - das ist Swing-Trading!
    """
    
    def __init__(self):
        self.min_volume = 1000000  # Min 1M Volumen
        self.max_spread = 0.002   # Max 0.2% Spread
        
    def analyze(self, ticker: str, df: pd.DataFrame) -> Optional[Signal]:
        """
        Suche nach Intraday-Setups
        """
        if len(df) < 20:
            return None
            
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        
        # Setup 1: Momentum Burst (1-Min-Scalping)
        momentum = self._check_momentum_burst(df)
        if momentum:
            return self._create_signal(
                ticker, last, momentum, 
                'momentum_scalp', 0.4, 3
            )
        
        # Setup 2: VWAP Bounce (5-Min-Setup)
        vwap = self._check_vwap_bounce(df)
        if vwap:
            return self._create_signal(
                ticker, last, vwap,
                'vwap_bounce', 0.6, 2
            )
        
        # Setup 3: Range Breakout (15-Min-Setup)
        breakout = self._check_range_breakout(df)
        if breakout:
            return self._create_signal(
                ticker, last, breakout,
                'breakout', 0.5, 2
            )
        
        return None
    
    def _check_momentum_burst(self, df: pd.DataFrame) -> dict:
        """
        1-Min Scalping: Schneller Momentum-Burst
        - Volume Spike > 200%
        - Candle > 0.2% in Richtung
        - RSI zwischen 40-70 (nicht überkauft)
        """
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Volume Spike
        vol_ratio = last.get('volume_ratio', 1.0)
        if vol_ratio < 2.0:
            return None
        
        # Momentum (1-Min-Candle)
        candle_pct = (last['Close'] - last['Open']) / last['Open']
        if abs(candle_pct) < 0.002:  # Min 0.2% Bewegung
            return None
            
        # Richtung
        direction = 'buy' if candle_pct > 0 else 'sell'
        
        # RSI Check (nicht extrem)
        rsi = last.get('RSI', 50)
        if direction == 'buy' and rsi > 70:
            return None
        if direction == 'sell' and rsi < 30:
            return None
            
        return { 'direction': direction, 'strength': abs(candle_pct) }
    
    def _check_vwap_bounce(self, df: pd.DataFrame) -> dict:
        """
        VWAP Bounce: Preis kommt zurück zum VWAP
        - Preis kreuzt VWAP
        - Volume bestätigt
        - Stop knapp unter/über VWAP
        """
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        vwap = last.get('vwap')
        if not vwap:
            return None
        
        # Kreuzung über VWAP (Long)
        if prev['Close'] < vwap and last['Close'] > vwap:
            direction = 'buy'
        # Kreuzung unter VWAP (Short)  
        elif prev['Close'] > vwap and last['Close'] < vwap:
            direction = 'sell'
        else:
            return None
            
        # Volume Check
        if last.get('volume_ratio', 1.0) < 1.2:
            return None
            
        return { 'direction': direction, 'vwap': vwap }
    
    def _check_range_breakout(self, df: pd.DataFrame) -> dict:
        """
        Range Breakout: Ausbruch aus 30-Min Range
        - High/Low der letzten 30 Min
        - Ausbruch mit Volume
        """
        if len(df) < 30:
            return None
            
        last = df.iloc[-1]
        
        # Letzte 30 Min Range
        recent = df.tail(30)
        range_high = recent['High'].max()
        range_low = recent['Low'].min()
        range_size = (range_high - range_low) / range_low
        
        # Range muss kompakt sein (< 1%)
        if range_size > 0.01:
            return None
            
        # Breakout Long
        if last['Close'] > range_high * 0.999:
            direction = 'buy'
        # Breakout Short
        elif last['Close'] < range_low * 1.001:
            direction = 'sell'
        else:
            return None
            
        # Volume bestätigt
        if last.get('volume_ratio', 1.0) < 1.3:
            return None
            
        return { 'direction': direction, 'break_level': last['Close'] }
    
    def _create_signal(self, ticker, last, setup, strategy, conf_base, leverage):
        """
        Erstellt Intraday-Signal mit FIXEN Exit-Regeln
        """
        entry = last['Close']
        direction = setup['direction']
        
        # Fester Stop: Max 0.3% Risk
        if direction == 'buy':
            stop_loss = entry * 0.997
            target_profit = 0.005  # 0.5% Target (Daytrading!)
        else:
            stop_loss = entry * 1.003
            target_profit = -0.005
            
        take_profit = entry * (1 + target_profit)
        
        # Max Hold Time: 60 Min (Intraday!)
        max_hold = 60
        
        # Description kurz & knapp
        desc = f"{strategy.upper()}: {direction} | Target {target_profit*100:.1f}% | Max {max_hold}min"
        
        return Signal(
            ticker=ticker,
            signal_type=direction,
            confidence=min(conf_base + 0.2, 0.9),
            strategy=strategy,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=leverage,
            setup_description=desc,
            timestamp=last.name,
            max_hold_time=max_hold,
            target_profit=abs(target_profit)
        )

# Hilfsfunktion für Time-Exit
def check_time_exit(entry_time: datetime, max_minutes: int = 60) -> bool:
    """
    Prüft ob Max-Hold-Time erreicht
    Returns True wenn Trade geschlossen werden soll
    """
    elapsed = (datetime.now() - entry_time).total_seconds() / 60
    return elapsed >= max_minutes