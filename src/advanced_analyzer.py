"""
Advanced Technical Analysis
Erweiterte Chart-Analyse mit Pattern-Erkennung, Support/Resistance, VWAP, etc.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class PatternType(Enum):
    DOUBLE_BOTTOM = "double_bottom"
    DOUBLE_TOP = "double_top"
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    TRIANGLE_ASCENDING = "triangle_ascending"
    TRIANGLE_DESCENDING = "triangle_descending"
    WEDGE = "wedge"
    FLAG = "flag"
    CHANNEL = "channel"

@dataclass
class Pattern:
    type: PatternType
    confidence: float
    start_idx: int
    end_idx: int
    target_price: float
    stop_loss: float
    description: str

class AdvancedAnalyzer:
    """
    Erweiterte technische Analyse für TradeForge:
    - Support/Resistance Levels
    - VWAP (Volume Weighted Average Price)
    - Volume Profile
    - Fibonacci Retracements
    - Chart Patterns (Double Bottom, Head&Shoulders, etc.)
    - Market Structure (Higher Highs/Lows, Lower Highs/Lows)
    """
    
    def __init__(self):
        self.lookback_period = 60  # 60 Perioden für Pattern-Erkennung
    
    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """
        VWAP (Volume Weighted Average Price)
        Wichtig für Intraday-Trading
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return vwap
    
    def find_support_resistance(self, df: pd.DataFrame, lookback: int = 20) -> Dict:
        """
        Findet Support und Resistance Levels
        Basierend auf Pivot Points (Lokale Hochs/Tiefs)
        """
        highs = df['high'].values
        lows = df['low'].values
        
        # Lokale Maxima (Resistance)
        resistance_levels = []
        for i in range(lookback, len(highs) - lookback):
            if all(highs[i] >= highs[i-j] for j in range(1, lookback+1)) and \
               all(highs[i] >= highs[i+j] for j in range(1, lookback+1)):
                resistance_levels.append((i, highs[i]))
        
        # Lokale Minima (Support)
        support_levels = []
        for i in range(lookback, len(lows) - lookback):
            if all(lows[i] <= lows[i-j] for j in range(1, lookback+1)) and \
               all(lows[i] <= lows[i+j] for j in range(1, lookback+1)):
                support_levels.append((i, lows[i]))
        
        # Levels clustern (nahe beieinander liegende zusammenfassen)
        resistance_clustered = self._cluster_levels([r[1] for r in resistance_levels])
        support_clustered = self._cluster_levels([s[1] for s in support_levels])
        
        return {
            'resistance': resistance_clustered,
            'support': support_clustered,
            'resistance_count': len(resistance_clustered),
            'support_count': len(support_clustered)
        }
    
    def _cluster_levels(self, levels: List[float], threshold: float = 0.02) -> List[float]:
        """Clustert nahe beieinander liegende Preislevels"""
        if not levels:
            return []
        
        levels = sorted(levels)
        clusters = []
        current_cluster = [levels[0]]
        
        for i in range(1, len(levels)):
            if abs(levels[i] - current_cluster[-1]) / current_cluster[-1] <= threshold:
                current_cluster.append(levels[i])
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [levels[i]]
        
        if current_cluster:
            clusters.append(np.mean(current_cluster))
        
        return sorted(clusters, reverse=True)  # Höchste zuerst
    
    def calculate_fibonacci_retracements(self, high: float, low: float) -> Dict:
        """
        Fibonacci Retracement Levels
        Wichtige Level: 23.6%, 38.2%, 50%, 61.8%, 78.6%
        """
        diff = high - low
        return {
            '0.0': high,
            '23.6': high - diff * 0.236,
            '38.2': high - diff * 0.382,
            '50.0': high - diff * 0.5,
            '61.8': high - diff * 0.618,
            '78.6': high - diff * 0.786,
            '100.0': low
        }
    
    def detect_double_bottom(self, df: pd.DataFrame) -> Optional[Pattern]:
        """
        Erkennt Double Bottom Pattern (bullish)
        W-Form mit zwei Tiefs auf ähnlichem Level
        """
        lows = df['low'].values
        
        # Finde lokale Tiefs
        troughs = []
        for i in range(5, len(lows) - 5):
            if all(lows[i] <= lows[i-j] for j in range(1, 5)) and \
               all(lows[i] <= lows[i+j] for j in range(1, 5)):
                troughs.append((i, lows[i]))
        
        # Suche nach zwei Tiefs mit ähnlichem Preis
        if len(troughs) >= 2:
            for i in range(len(troughs) - 1):
                for j in range(i + 1, len(troughs)):
                    idx1, price1 = troughs[i]
                    idx2, price2 = troughs[j]
                    
                    # Mindestens 5 Perioden Abstand
                    if idx2 - idx1 < 5:
                        continue
                    
                    # Tiefs sollten ähnlich sein (±3%)
                    price_diff = abs(price1 - price2) / price1
                    if price_diff <= 0.03:
                        # Neckline (Widerstand zwischen den Tiefs)
                        between = df['high'].iloc[idx1:idx2].max()
                        
                        # Target = Neckline + (Neckline - Low)
                        target = between + (between - min(price1, price2))
                        
                        return Pattern(
                            type=PatternType.DOUBLE_BOTTOM,
                            confidence=min(0.9, 0.7 + (1 - price_diff)),
                            start_idx=idx1,
                            end_idx=idx2,
                            target_price=target,
                            stop_loss=min(price1, price2) * 0.98,
                            description=f"Double Bottom: Two lows at ${price1:.2f} and ${price2:.2f}"
                        )
        
        return None
    
    def detect_head_and_shoulders(self, df: pd.DataFrame) -> Optional[Pattern]:
        """
        Erkennt Head and Shoulders Pattern (bearish reversal)
        Drei Peaks: Schulter - Kopf - Schulter
        """
        highs = df['high'].values
        
        # Finde lokale Hochs
        peaks = []
        for i in range(5, len(highs) - 5):
            if all(highs[i] >= highs[i-j] for j in range(1, 5)) and \
               all(highs[i] >= highs[i+j] for j in range(1, 5)):
                peaks.append((i, highs[i]))
        
        # Suche nach Head & Shoulders (mindestens 3 Peaks)
        if len(peaks) >= 3:
            for i in range(len(peaks) - 2):
                left_shoulder = peaks[i]
                head = peaks[i + 1]
                right_shoulder = peaks[i + 2]
                
                # Kopf muss höher als Schultern sein
                if head[1] > left_shoulder[1] and head[1] > right_shoulder[1]:
                    # Schultern sollten ähnlich sein (±5%)
                    shoulder_diff = abs(left_shoulder[1] - right_shoulder[1]) / left_shoulder[1]
                    
                    if shoulder_diff <= 0.05:
                        # Neckline (Support zwischen Schultern)
                        neckline = df['low'].iloc[left_shoulder[0]:right_shoulder[0]].min()
                        
                        # Target = Neckline - (Head - Neckline)
                        target = neckline - (head[1] - neckline)
                        
                        return Pattern(
                            type=PatternType.HEAD_AND_SHOULDERS,
                            confidence=min(0.9, 0.75 + (1 - shoulder_diff)),
                            start_idx=left_shoulder[0],
                            end_idx=right_shoulder[0],
                            target_price=target,
                            stop_loss=head[1] * 1.02,
                            description=f"Head & Shoulders: Head at ${head[1]:.2f}, shoulders at ${left_shoulder[1]:.2f} and ${right_shoulder[1]:.2f}"
                        )
        
        return None
    
    def detect_triangle(self, df: pd.DataFrame) -> Optional[Pattern]:
        """
        Erkennt Ascending/Descending Triangles
        """
        highs = df['high'].values
        lows = df['low'].values
        
        n = len(highs)
        if n < 20:
            return None
        
        # Letzte 20 Perioden analysieren
        recent_highs = highs[-20:]
        recent_lows = lows[-20:]
        
        # Trendlinien berechnen (simple linear regression)
        x = np.arange(20)
        
        # Resistance line (highs)
        z_high = np.polyfit(x, recent_highs, 1)
        slope_high = z_high[0]
        
        # Support line (lows)
        z_low = np.polyfit(x, recent_lows, 1)
        slope_low = z_low[0]
        
        # Ascending Triangle: Flat resistance, rising support
        if abs(slope_high) < 0.001 and slope_low > 0:
            return Pattern(
                type=PatternType.TRIANGLE_ASCENDING,
                confidence=0.65,
                start_idx=n-20,
                end_idx=n-1,
                target_price=recent_highs[-1] + (recent_highs[-1] - recent_lows[-1]),
                stop_loss=recent_lows[-1] * 0.98,
                description="Ascending Triangle: Flat resistance, rising support"
            )
        
        # Descending Triangle: Falling resistance, flat support
        if slope_high < 0 and abs(slope_low) < 0.001:
            return Pattern(
                type=PatternType.TRIANGLE_DESCENDING,
                confidence=0.65,
                start_idx=n-20,
                end_idx=n-1,
                target_price=recent_lows[-1] - (recent_highs[-1] - recent_lows[-1]),
                stop_loss=recent_highs[-1] * 1.02,
                description="Descending Triangle: Falling resistance, flat support"
            )
        
        return None
    
    def analyze_market_structure(self, df: pd.DataFrame) -> Dict:
        """
        Analysiert die Marktstruktur
        Higher Highs/Lows = Bullish
        Lower Highs/Lows = Bearish
        """
        highs = df['high'].values
        lows = df['low'].values
        
        # Letzte 10 Perioden
        recent_highs = highs[-10:]
        recent_lows = lows[-10:]
        
        # Zähle Higher Highs vs Lower Highs
        hh_count = 0
        lh_count = 0
        hl_count = 0
        ll_count = 0
        
        for i in range(1, len(recent_highs)):
            if recent_highs[i] > recent_highs[i-1]:
                hh_count += 1
            else:
                lh_count += 1
            
            if recent_lows[i] > recent_lows[i-1]:
                hl_count += 1
            else:
                ll_count += 1
        
        # Trend bestimmen
        if hh_count > lh_count and hl_count > ll_count:
            trend = "bullish"
            trend_strength = (hh_count + hl_count) / 18  # 0-1
        elif lh_count > hh_count and ll_count > hl_count:
            trend = "bearish"
            trend_strength = (lh_count + ll_count) / 18
        else:
            trend = "neutral"
            trend_strength = 0.5
        
        return {
            'trend': trend,
            'trend_strength': round(trend_strength, 2),
            'higher_highs': hh_count,
            'lower_highs': lh_count,
            'higher_lows': hl_count,
            'lower_lows': ll_count
        }
    
    def get_all_patterns(self, df: pd.DataFrame) -> List[Pattern]:
        """Findet alle Chart Patterns"""
        patterns = []
        
        # Pattern erkennen
        double_bottom = self.detect_double_bottom(df)
        if double_bottom:
            patterns.append(double_bottom)
        
        head_shoulders = self.detect_head_and_shoulders(df)
        if head_shoulders:
            patterns.append(head_shoulders)
        
        triangle = self.detect_triangle(df)
        if triangle:
            patterns.append(triangle)
        
        return patterns
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Komplette erweiterte Analyse
        """
        # VWAP
        vwap = self.calculate_vwap(df)
        vwap_position = (df['close'].iloc[-1] - vwap.iloc[-1]) / vwap.iloc[-1]
        
        # Support/Resistance
        levels = self.find_support_resistance(df)
        
        # Fibonacci
        recent_high = df['high'].iloc[-20:].max()
        recent_low = df['low'].iloc[-20:].min()
        fib_levels = self.calculate_fibonacci_retracements(recent_high, recent_low)
        
        # Pattern
        patterns = self.get_all_patterns(df)
        
        # Market Structure
        structure = self.analyze_market_structure(df)
        
        return {
            'vwap': {
                'value': vwap.iloc[-1],
                'position': round(vwap_position * 100, 2),  # % über/unter VWAP
                'above': df['close'].iloc[-1] > vwap.iloc[-1]
            },
            'support_resistance': levels,
            'fibonacci': fib_levels,
            'patterns': [
                {
                    'type': p.type.value,
                    'confidence': p.confidence,
                    'target': p.target_price,
                    'stop_loss': p.stop_loss,
                    'description': p.description
                }
                for p in patterns
            ],
            'market_structure': structure,
            'pattern_count': len(patterns)
        }