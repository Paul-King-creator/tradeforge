#!/usr/bin/env python3
"""
TradeForge ML Pre-Training
Erstellt synthetische Trainingsdaten aus historischen Mustern
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pickle
import sqlite3
import os
import sys
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Watchlist
WATCHLIST = [
    'AAPL', 'MSFT', 'NVDA', 'AMD', 'AMZN', 'GOOGL', 'META', 'NFLX', 'TSLA', 'CRM',
    'AVGO', 'QCOM', 'INTC', 'MU', 'LRCX', 'KLAC',
    'JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'V', 'MA',
    'JNJ', 'PFE', 'UNH', 'ABBV', 'LLY', 'MRK', 'BMY',
    'XOM', 'CVX', 'COP', 'SLB',
    'BA', 'CAT', 'GE', 'HON', 'UPS',
    'WMT', 'HD', 'COST', 'NKE', 'SBUX', 'MCD',
    'SPY', 'QQQ', 'IWM', 'XLK', 'XLF', 'XLE', 'XLI'
]

class SyntheticDataGenerator:
    """Generiert synthetische Trainingsdaten aus historischen Preisdaten"""
    
    def __init__(self):
        self.successful_setups = []
        self.failed_setups = []
        self.neutral_setups = []
        
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Alle technischen Indikatoren hinzufügen"""
        df = df.copy()
        
        # Moving Averages
        df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
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
        df['Volume_Trend'] = df['Volume'].rolling(window=10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        df['ATR_Percent'] = df['ATR'] / df['Close'] * 100
        
        # VWAP
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        df['VWAP'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
        df['VWAP_Distance'] = (df['Close'] - df['VWAP']) / df['VWAP'] * 100
        
        # Trend Strength
        df['ADX'] = self._calculate_adx(df)
        
        # Momentum
        df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        df['Momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        
        # Volatility
        df['Volatility_20'] = df['Close'].rolling(window=20).std() / df['Close'].rolling(window=20).mean()
        
        # Price Distance from Moving Averages
        df['Dist_EMA9'] = (df['Close'] - df['EMA_9']) / df['EMA_9'] * 100
        df['Dist_EMA21'] = (df['Close'] - df['EMA_21']) / df['EMA_21'] * 100
        df['Dist_EMA50'] = (df['Close'] - df['EMA_50']) / df['EMA_50'] * 100
        
        return df
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average Directional Index für Trendstärke"""
        plus_dm = df['High'].diff()
        minus_dm = df['Low'].diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = minus_dm.abs()
        
        tr = np.maximum(df['High'] - df['Low'], 
                       np.maximum(abs(df['High'] - df['Close'].shift()), 
                                 abs(df['Low'] - df['Close'].shift())))
        
        atr = tr.rolling(window=period).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def identify_setups(self, ticker: str, df: pd.DataFrame):
        """Identifiziert erfolgreiche und fehlgeschlagene Setups"""
        if len(df) < 60:
            return
            
        df = self.add_indicators(df)
        
        # Für jeden möglichen Einstiegspunkt prüfen, was passiert wäre
        for i in range(50, len(df) - 20):  # Mindestens 50 Tage History, 20 Tage Forward
            row = df.iloc[i]
            
            # Nur bei ausreichenden Daten
            if pd.isna(row['RSI']) or pd.isna(row['MACD']):
                continue
            
            entry_price = row['Close']
            
            # Mögliche Setups identifizieren
            setups = self._detect_potential_setups(df, i)
            
            for setup_type in setups:
                # Forward-Test: Was ist passiert?
                future_return = self._calculate_forward_return(df, i)
                
                # Klassifizieren
                if future_return >= 0.02:  # +2% oder mehr
                    self.successful_setups.append(self._extract_features(ticker, df, i, setup_type, future_return, 1))
                elif future_return <= -0.01:  # -1% oder mehr
                    self.failed_setups.append(self._extract_features(ticker, df, i, setup_type, future_return, 0))
                else:
                    self.neutral_setups.append(self._extract_features(ticker, df, i, setup_type, future_return, 2))
    
    def _detect_potential_setups(self, df: pd.DataFrame, idx: int) -> List[str]:
        """Erkennt potenzielle Trade-Setups an einem Index"""
        setups = []
        row = df.iloc[idx]
        
        # Setup 1: RSI Oversold Bounce (RSI < 35)
        if row['RSI'] < 35:
            setups.append('rsi_oversold')
        
        # Setup 2: RSI Overbought Reversal (RSI > 70)
        if row['RSI'] > 70:
            setups.append('rsi_overbought')
        
        # Setup 3: MACD Crossover
        if idx > 0:
            prev_row = df.iloc[idx - 1]
            if prev_row['MACD'] < prev_row['MACD_Signal'] and row['MACD'] > row['MACD_Signal']:
                setups.append('macd_bullish_cross')
            if prev_row['MACD'] > prev_row['MACD_Signal'] and row['MACD'] < row['MACD_Signal']:
                setups.append('macd_bearish_cross')
        
        # Setup 4: Bollinger Bounce (Preis nahe unterer Band)
        if row['BB_Position'] < 0.1:
            setups.append('bb_lower_bounce')
        
        # Setup 5: Bollinger Reversal (Preis nahe oberes Band)
        if row['BB_Position'] > 0.9:
            setups.append('bb_upper_reversal')
        
        # Setup 6: EMA Trend Following (Preis über EMA21, EMA9 über EMA21)
        if row['Close'] > row['EMA_21'] and row['EMA_9'] > row['EMA_21']:
            setups.append('ema_trend_following')
        
        # Setup 7: Mean Reversion (Preis weit vom VWAP entfernt)
        if row['VWAP_Distance'] < -2:  # 2% unter VWAP
            setups.append('mean_reversion_long')
        if row['VWAP_Distance'] > 2:  # 2% über VWAP
            setups.append('mean_reversion_short')
        
        # Setup 8: High Volume Breakout
        if idx > 0 and row['Volume_Ratio'] > 1.5 and row['Close'] > df.iloc[idx-1]['High']:
            setups.append('volume_breakout')
        
        # Setup 9: Momentum
        if row['Momentum_10'] > 0.05 and row['Momentum_10'] < 0.15:
            setups.append('momentum_long')
        if row['Momentum_10'] < -0.05 and row['Momentum_10'] > -0.15:
            setups.append('momentum_short')
        
        # Wenn keine spezifischen Setups gefunden, aber gute technische Daten
        if not setups and row['RSI'] > 30 and row['RSI'] < 70:
            setups.append('general_technical')
        
        return setups if setups else ['no_setup']
    
    def _calculate_forward_return(self, df: pd.DataFrame, idx: int, days: int = 5) -> float:
        """Berechnet den Return über die nächsten N Tage"""
        if idx + days >= len(df):
            return 0
        
        entry = df.iloc[idx]['Close']
        
        # Max return und Min return im Zeitraum
        future_prices = df.iloc[idx+1:idx+days+1]['Close']
        max_price = future_prices.max()
        min_price = future_prices.min()
        
        # Verwende den besten Exit (für Long-Trades max, für Analysis allgemein max)
        return (max_price - entry) / entry
    
    def _extract_features(self, ticker: str, df: pd.DataFrame, idx: int, 
                          setup_type: str, future_return: float, label: int) -> Dict:
        """Extrahiert Features für ML"""
        row = df.iloc[idx]
        
        # Markttrend (letzte 20 Tage)
        trend_data = df.iloc[idx-20:idx]['Close']
        market_trend = (trend_data.iloc[-1] - trend_data.iloc[0]) / trend_data.iloc[0]
        
        # Higher Highs / Lower Lows
        highs = df.iloc[idx-10:idx]['High'].values
        lows = df.iloc[idx-10:idx]['Low'].values
        
        hh_count = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
        hl_count = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i-1])
        
        return {
            'ticker': ticker,
            'setup_type': setup_type,
            'entry_price': row['Close'],
            'future_return': future_return,
            'label': label,  # 1 = Erfolg (+2%), 0 = Misserfolg (-1%), 2 = Neutral
            
            # Technische Indikatoren
            'rsi': row['RSI'],
            'macd': row['MACD'],
            'macd_hist': row['MACD_Hist'],
            'macd_signal': row['MACD_Signal'],
            'bb_position': row['BB_Position'],
            'bb_width': row['BB_Width'],
            'volume_ratio': row['Volume_Ratio'],
            'volume_trend': row['Volume_Trend'],
            'atr_percent': row['ATR_Percent'],
            'vwap_distance': row['VWAP_Distance'],
            'adx': row['ADX'] if not pd.isna(row['ADX']) else 25,
            'momentum_10': row['Momentum_10'],
            'momentum_20': row['Momentum_20'],
            'volatility_20': row['Volatility_20'],
            
            # Moving Average Distanzen
            'dist_ema9': row['Dist_EMA9'],
            'dist_ema21': row['Dist_EMA21'],
            'dist_ema50': row['Dist_EMA50'],
            
            # Markttrend
            'market_trend_20d': market_trend,
            'higher_highs_count': hh_count,
            'higher_lows_count': hl_count,
            
            # Zeit-Features
            'day_of_week': row.name.weekday(),
            'month': row.name.month,
            'quarter': (row.name.month - 1) // 3 + 1,
        }


class MLModelTrainer:
    """Trainiert das ML-Modell mit den synthetischen Daten"""
    
    def __init__(self):
        self.classifier = None
        self.regressor = None
        self.scaler = None
        self.feature_names = None
        
    def prepare_training_data(self, successful: List[Dict], failed: List[Dict], 
                              neutral: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bereitet die Trainingsdaten vor"""
        
        # Kombiniere und balanciere Daten
        print(f"  Erfolgreiche Setups: {len(successful)}")
        print(f"  Fehlgeschlagene Setups: {len(failed)}")
        print(f"  Neutrale Setups: {len(neutral)}")
        
        # Balance die Klassen
        min_samples = min(len(successful), len(failed))
        if len(neutral) > 0:
            min_samples = min(min_samples, len(neutral))
        
        # Nehme zufällige Samples aus jeder Klasse
        np.random.seed(42)
        successful_sample = np.random.choice(len(successful), min_samples, replace=False)
        failed_sample = np.random.choice(len(failed), min_samples, replace=False)
        
        balanced_data = []
        for idx in successful_sample:
            balanced_data.append(successful[idx])
        for idx in failed_sample:
            balanced_data.append(failed[idx])
        
        print(f"  Balanceierte Trainingsdaten: {len(balanced_data)} (je {min_samples})")
        
        # Features extrahieren
        self.feature_names = [
            'rsi', 'macd', 'macd_hist', 'macd_signal', 'bb_position', 'bb_width',
            'volume_ratio', 'volume_trend', 'atr_percent', 'vwap_distance', 'adx',
            'momentum_10', 'momentum_20', 'volatility_20',
            'dist_ema9', 'dist_ema21', 'dist_ema50',
            'market_trend_20d', 'higher_highs_count', 'higher_lows_count',
            'day_of_week', 'month', 'quarter'
        ]
        
        X = []
        y_class = []  # 1 = Erfolg, 0 = Misserfolg
        y_reg = []    # Tatsächlicher Return
        
        for data in balanced_data:
            features = [data[f] for f in self.feature_names]
            X.append(features)
            y_class.append(data['label'])
            y_reg.append(data['future_return'])
        
        return np.array(X), np.array(y_class), np.array(y_reg)
    
    def train(self, successful: List[Dict], failed: List[Dict], neutral: List[Dict]):
        """Trainiert die Modelle"""
        print("\n🔧 Bereite Trainingsdaten vor...")
        X, y_class, y_reg = self.prepare_training_data(successful, failed, neutral)
        
        if len(X) < 100:
            print("❌ Nicht genug Trainingsdaten! Mindestens 100 erforderlich.")
            return False
        
        # Split in Train und Test
        X_train, X_test, y_train_class, y_test_class = train_test_split(
            X, y_class, test_size=0.2, random_state=42, stratify=y_class
        )
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X, y_reg, test_size=0.2, random_state=42
        )
        
        # Scaler fitten
        print("📝 Scaler wird trainiert...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Classifier trainieren (Erfolg vs Misserfolg)
        print("🤖 Training Classifier...")
        self.classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.classifier.fit(X_train_scaled, y_train_class)
        
        # Classifier evaluieren
        y_pred_class = self.classifier.predict(X_test_scaled)
        accuracy = accuracy_score(y_test_class, y_pred_class)
        precision = precision_score(y_test_class, y_pred_class, average='weighted', zero_division=0)
        recall = recall_score(y_test_class, y_pred_class, average='weighted', zero_division=0)
        f1 = f1_score(y_test_class, y_pred_class, average='weighted', zero_division=0)
        
        print(f"\n📊 Classifier Ergebnisse:")
        print(f"   Accuracy:  {accuracy:.3f}")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall:    {recall:.3f}")
        print(f"   F1-Score:  {f1:.3f}")
        
        # Regressor trainieren (Return-Vorhersage)
        print("\n📈 Training Regressor...")
        self.regressor = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.regressor.fit(X_train_scaled, y_train_reg)
        
        # Regressor evaluieren
        y_pred_reg = self.regressor.predict(X_test_scaled)
        mse = np.mean((y_test_reg - y_pred_reg) ** 2)
        mae = np.mean(np.abs(y_test_reg - y_pred_reg))
        
        print(f"\n📊 Regressor Ergebnisse:")
        print(f"   MSE: {mse:.6f}")
        print(f"   MAE: {mae:.6f}")
        
        # Feature Importance
        print("\n🔍 Feature Importance (Top 10):")
        importances = self.classifier.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        for i in indices:
            print(f"   {self.feature_names[i]}: {importances[i]:.3f}")
        
        return True
    
    def save_model(self, filepath: str):
        """Speichert das trainierte Modell"""
        model_data = {
            'classifier': self.classifier,
            'regressor': self.regressor,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'trained_at': datetime.now().isoformat(),
            'version': '1.0.0'
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n💾 Modell gespeichert unter: {filepath}")
    
    def predict(self, features: Dict) -> Dict:
        """Macht eine Vorhersage für ein Setup"""
        if self.classifier is None:
            return {'error': 'Modell nicht trainiert'}
        
        X = np.array([[features[f] for f in self.feature_names]])
        X_scaled = self.scaler.transform(X)
        
        # Klassifikation
        success_prob = self.classifier.predict_proba(X_scaled)[0][1]
        prediction = self.classifier.predict(X_scaled)[0]
        
        # Regression
        expected_return = self.regressor.predict(X_scaled)[0]
        
        return {
            'success_probability': round(success_prob, 3),
            'expected_return': round(expected_return, 4),
            'prediction': 'success' if prediction == 1 else 'fail',
            'confidence': 'high' if success_prob > 0.7 or success_prob < 0.3 else 'medium'
        }


def download_historical_data(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
    """Lädt historische Daten von Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty or len(df) < 60:
            print(f"   ⚠️  {ticker}: Nicht genug Daten")
            return None
        return df
    except Exception as e:
        print(f"   ❌ {ticker}: Fehler - {e}")
        return None


def main():
    print("=" * 70)
    print("🚀 TradeForge ML Pre-Training")
    print("=" * 70)
    print(f"\nStartzeit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Anzahl der Aktien: {len(WATCHLIST)}")
    print(f"Zeitraum: 1 Jahr historische Daten")
    
    # Initialisiere Generator
    generator = SyntheticDataGenerator()
    
    # Lade Daten für alle Aktien
    print("\n📥 Lade historische Daten...")
    success_count = 0
    
    for i, ticker in enumerate(WATCHLIST, 1):
        print(f"   [{i:2d}/{len(WATCHLIST)}] {ticker}...", end=' ')
        df = download_historical_data(ticker)
        if df is not None:
            generator.identify_setups(ticker, df)
            success_count += 1
            print(f"✅ ({len(df)} Tage)")
        else:
            print("❌")
    
    print(f"\n✅ Erfolgreich geladen: {success_count}/{len(WATCHLIST)} Aktien")
    
    # Zeige Statistik
    print("\n" + "=" * 70)
    print("📊 Generierte Trainingsdaten")
    print("=" * 70)
    print(f"  Erfolgreiche Setups (+2%+):  {len(generator.successful_setups)}")
    print(f"  Fehlgeschlagene Setups (-1%+): {len(generator.failed_setups)}")
    print(f"  Neutrale Setups:             {len(generator.neutral_setups)}")
    print(f"  Total:                        {len(generator.successful_setups) + len(generator.failed_setups) + len(generator.neutral_setups)}")
    
    # Trainiere Modelle
    print("\n" + "=" * 70)
    print("🎓 Training der ML-Modelle")
    print("=" * 70)
    
    trainer = MLModelTrainer()
    success = trainer.train(
        generator.successful_setups,
        generator.failed_setups,
        generator.neutral_setups
    )
    
    if not success:
        print("❌ Training fehlgeschlagen!")
        sys.exit(1)
    
    # Speichere Modell
    model_path = "models/pretrained_model.pkl"
    trainer.save_model(model_path)
    
    # Test-Vorhersage
    print("\n" + "=" * 70)
    print("🧪 Test-Vorhersage")
    print("=" * 70)
    
    test_features = {
        'rsi': 30,
        'macd': 0.5,
        'macd_hist': 0.3,
        'macd_signal': 0.2,
        'bb_position': 0.2,
        'bb_width': 0.15,
        'volume_ratio': 1.3,
        'volume_trend': 1000,
        'atr_percent': 2.0,
        'vwap_distance': -1.5,
        'adx': 25,
        'momentum_10': 0.03,
        'momentum_20': 0.05,
        'volatility_20': 0.02,
        'dist_ema9': -0.5,
        'dist_ema21': -1.0,
        'dist_ema50': -2.0,
        'market_trend_20d': 0.05,
        'higher_highs_count': 3,
        'higher_lows_count': 4,
        'day_of_week': 2,
        'month': 6,
        'quarter': 2
    }
    
    prediction = trainer.predict(test_features)
    print(f"\nTest-Features (RSI=30, VWAP=-1.5%):\n{prediction}")
    
    print("\n" + "=" * 70)
    print("✅ ML Pre-Training abgeschlossen!")
    print("=" * 70)
    print(f"\nDas Modell ist bereit für TradeForge!")
    print(f"Pfad: {model_path}")
    print(f"Endzeit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
