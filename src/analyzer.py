"""
Technical Analyzer - Technische Indikatoren
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Signal:
    ticker: str
    signal_type: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0-1
    strategy: str
    entry_price: float
    stop_loss: float
    take_profit: float
    leverage: int
    setup_description: str
    timestamp: pd.Timestamp

class TechnicalAnalyzer:
    def __init__(self):
        pass
    
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Alle relevanten Indikatoren hinzufügen"""
        df = df.copy()
        
        # Moving Averages
        df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volume
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # ATR (Average True Range) für Volatilität
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        # VWAP
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        df['VWAP'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
        
        return df
    
    def detect_support_resistance(self, df: pd.DataFrame, window: int = 20) -> tuple:
        """Support und Resistance Levels finden"""
        highs = df['High'].rolling(window=window, center=True).max()
        lows = df['Low'].rolling(window=window, center=True).min()
        
        resistance = df['High'][df['High'] == highs]
        support = df['Low'][df['Low'] == lows]
        
        # Letzte 3 Levels nehmen
        resistance_levels = resistance.tail(3).values
        support_levels = support.tail(3).values
        
        return support_levels, resistance_levels
    
    def analyze_trend(self, df: pd.DataFrame) -> str:
        """Trend-Richtung bestimmen"""
        if len(df) < 50:
            return 'unknown'
        
        last = df.iloc[-1]
        
        # EMA-Kreuzungen
        ema_bullish = last['EMA_9'] > last['EMA_21'] > last['EMA_50']
        ema_bearish = last['EMA_9'] < last['EMA_21'] < last['EMA_50']
        
        # Preis vs EMAs
        price_above_emas = last['Close'] > last['EMA_9']
        price_below_emas = last['Close'] < last['EMA_9']
        
        if ema_bullish and price_above_emas:
            return 'strong_uptrend'
        elif ema_bullish:
            return 'uptrend'
        elif ema_bearish and price_below_emas:
            return 'strong_downtrend'
        elif ema_bearish:
            return 'downtrend'
        else:
            return 'sideways'
    
    def calculate_position_size(self, capital: float, risk_percent: float, 
                                entry: float, stop_loss: float) -> int:
        """Positionsgröße basierend auf Risiko"""
        risk_amount = capital * risk_percent
        price_risk = abs(entry - stop_loss)
        
        if price_risk == 0:
            return 0
        
        shares = int(risk_amount / price_risk)
        return shares