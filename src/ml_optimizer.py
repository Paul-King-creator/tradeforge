"""
ML-Based Strategy Optimizer
Verwendet scikit-learn für echtes Machine Learning
"""
import sqlite3
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import sys

# Versuche scikit-learn zu importieren
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️  scikit-learn nicht verfügbar - ML-Features deaktiviert")
    print("   Installieren mit: pip install scikit-learn pandas")


@dataclass
class TradeFeatures:
    """Features eines Trades für ML"""
    ticker: str
    strategy: str
    entry_price: float
    leverage: float
    confidence: float
    rsi: float = 50
    macd: float = 0
    bollinger_position: float = 0.5  # 0=Lower, 0.5=Middle, 1=Upper
    volume_ratio: float = 1.0  # Volume vs Average
    market_trend: float = 0  # -1 (bear) to 1 (bull)
    time_of_day: int = 12  # 0-23
    day_of_week: int = 0  # 0-6
    
    def to_vector(self) -> np.ndarray:
        """Konvertiert zu Feature-Vektor für ML"""
        return np.array([
            self.entry_price,
            self.leverage,
            self.confidence,
            self.rsi,
            self.macd,
            self.bollinger_position,
            self.volume_ratio,
            self.market_trend,
            self.time_of_day,
            self.day_of_week
        ])


class MLStrategyOptimizer:
    """
    ECHTES Machine Learning für TradeForge:
    
    1. Sammelt Trade-Daten mit Features
    2. Trainiert Predictive Models
    3. Sagt Trade-Erfolg voraus (bevor wir traden!)
    4. Optimiert Parameter durch Reinforcement Learning
    """
    
    def __init__(self, db_path: str = "data/trades.db"):
        self.db_path = db_path
        self.models = {}  # Ein Model pro Strategie
        self.scalers = {}
        self.min_samples_for_training = 20  # Mindestens 20 Trades
        self.performance_history = []
        
        if not ML_AVAILABLE:
            print("🚫 ML Optimizer initialisiert im Fallback-Modus")
        else:
            print("🤖 ML Optimizer bereit")
    
    def fetch_trade_data(self, strategy_name: str = None) -> List[Dict]:
        """Lade historische Trade-Daten mit allen Features"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT 
                ticker, strategy, entry_price, stop_loss, take_profit,
                leverage, confidence, pnl_percent, entry_time, exit_time,
                setup_description
            FROM trades 
            WHERE status = 'closed'
        '''
        
        if strategy_name:
            query += f" AND strategy = '{strategy_name}'"
        
        query += " ORDER BY entry_time DESC LIMIT 200"  # Letzte 200 Trades
        
        cursor.execute(query)
        columns = [col[0] for col in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        
        return results
    
    def extract_features_from_trade(self, trade: Dict) -> TradeFeatures:
        """Extrahiere Features aus Trade-Daten"""
        # Parse setup_description für Indikatoren
        setup = trade.get('setup_description', '')
        
        features = TradeFeatures(
            ticker=trade['ticker'],
            strategy=trade['strategy'],
            entry_price=trade['entry_price'],
            leverage=trade['leverage'],
            confidence=trade['confidence'],
            time_of_day=datetime.fromisoformat(trade['entry_time']).hour if trade.get('entry_time') else 12,
            day_of_week=datetime.fromisoformat(trade['entry_time']).weekday() if trade.get('entry_time') else 0
        )
        
        # Parse RSI aus Setup-Description (z.B. "RSI=13.2")
        if 'RSI=' in setup:
            try:
                rsi_str = setup.split('RSI=')[1].split(',')[0].split(')')[0]
                features.rsi = float(rsi_str)
            except:
                pass
        
        # Parse MACD
        if 'MACD=' in setup:
            try:
                macd_str = setup.split('MACD=')[1].split(',')[0]
                features.macd = float(macd_str)
            except:
                pass
        
        return features
    
    def train_models(self) -> bool:
        """Trainiere ML-Models für jede Strategie"""
        if not ML_AVAILABLE:
            return False
        
        print("\n🎓 Starte ML-Training...")
        
        strategies = ['momentum_long', 'breakout_long', 'mean_reversion_long']
        any_trained = False
        
        for strategy in strategies:
            trades = self.fetch_trade_data(strategy)
            
            if len(trades) < self.min_samples_for_training:
                print(f"   ⚠️  {strategy}: Nur {len(trades)} Trades (brauche {self.min_samples_for_training})")
                continue
            
            # Features und Labels vorbereiten
            X = []
            y_success = []  # 1 = Gewinn, 0 = Verlust
            y_return = []   # Tatsächlicher Return %
            
            for trade in trades:
                features = self.extract_features_from_trade(trade)
                X.append(features.to_vector())
                
                pnl = trade.get('pnl_percent', 0) or 0
                y_success.append(1 if pnl > 0 else 0)
                y_return.append(pnl)
            
            X = np.array(X)
            y_success = np.array(y_success)
            y_return = np.array(y_return)
            
            # Scaler fitten
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train/Test Split
            X_train, X_test, y_train_success, y_test_success = train_test_split(
                X_scaled, y_success, test_size=0.2, random_state=42
            )
            
            # Model 1: Klassifikation (Gewinn vs Verlust)
            clf = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
            clf.fit(X_train, y_train_success)
            
            # Accuracy testen
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test_success, y_pred)
            
            # Model 2: Regression (Wie viel % Return?)
            X_train_reg, X_test_reg, y_train_return, y_test_return = train_test_split(
                X_scaled, y_return, test_size=0.2, random_state=42
            )
            
            reg = RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
            reg.fit(X_train_reg, y_train_return)
            
            y_pred_return = reg.predict(X_test_reg)
            mse = mean_squared_error(y_test_return, y_pred_return)
            
            # Models speichern
            self.models[strategy] = {
                'classifier': clf,
                'regressor': reg,
                'accuracy': accuracy,
                'mse': mse,
                'trained_at': datetime.now().isoformat(),
                'samples': len(trades)
            }
            self.scalers[strategy] = scaler
            
            print(f"   ✅ {strategy}:")
            print(f"      Trainiert auf {len(trades)} Trades")
            print(f"      Vorhersage-Genauigkeit: {accuracy:.1%}")
            print(f"      Return-MSE: {mse:.2f}")
            any_trained = True
        
        return any_trained
    
    def predict_trade_success(self, features: TradeFeatures) -> Dict:
        """
        Sagt voraus ob ein Trade erfolgreich sein wird
        WIRD AUFGERUFEN BEVOR WIR TRADEN!
        """
        if not ML_AVAILABLE or features.strategy not in self.models:
            return {
                'success_probability': 0.5,
                'expected_return': 0,
                'confidence': 'low',
                'should_trade': True
            }
        
        model = self.models[features.strategy]
        scaler = self.scalers[features.strategy]
        
        X = features.to_vector().reshape(1, -1)
        X_scaled = scaler.transform(X)
        
        # Wahrscheinlichkeit für Gewinn
        success_prob = model['classifier'].predict_proba(X_scaled)[0][1]
        
        # Erwarteter Return
        expected_return = model['regressor'].predict(X_scaled)[0]
        
        # Confidence basierend auf Model-Accuracy
        confidence = 'high' if model['accuracy'] > 0.65 else 'medium' if model['accuracy'] > 0.55 else 'low'
        
        # Trade-Entscheidung
        should_trade = success_prob > 0.55 and expected_return > 0
        
        return {
            'success_probability': round(success_prob, 3),
            'expected_return': round(expected_return, 2),
            'confidence': confidence,
            'should_trade': should_trade,
            'model_accuracy': round(model['accuracy'], 3)
        }
    
    def optimize_leverage(self, strategy: str, base_leverage: int) -> int:
        """
        Optimiert Leverage basierend auf ML-Vorhersagen
        """
        if strategy not in self.models:
            return base_leverage
        
        model = self.models[strategy]
        accuracy = model['accuracy']
        
        # Wenn Model gut ist (>65%), können wir aggressiver sein
        if accuracy > 0.70:
            return min(base_leverage + 1, 5)  # Max 5x
        elif accuracy < 0.50:
            return max(base_leverage - 1, 1)  # Min 1x
        
        return base_leverage
    
    def generate_ml_report(self) -> str:
        """Generiert einen ML-Report"""
        if not self.models:
            return "🤖 ML Models: Noch nicht trainiert (brauche mindestens 20 Trades pro Strategie)"
        
        lines = [
            "=" * 60,
            "🤖 MACHINE LEARNING REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 60,
            ""
        ]
        
        for strategy, model in self.models.items():
            lines.extend([
                f"📊 {strategy.upper()}",
                f"   Samples: {model['samples']}",
                f"   Accuracy: {model['accuracy']:.1%}",
                f"   Last Trained: {model['trained_at'][:19]}",
                ""
            ])
        
        if not self.models:
            lines.append("   Noch keine Models trainiert")
        
        lines.append("=" * 60)
        return "\n".join(lines)


# Einfache Version für wenn ML nicht verfügbar ist
class SimpleRuleOptimizer:
    """Fallback: Einfache Regeln wenn kein ML verfügbar"""
    
    def __init__(self, db_path: str = "data/trades.db"):
        self.db_path = db_path
    
    def get_recommendation(self, strategy: str) -> str:
        return "ML nicht verfügbar - verwende Basis-Regeln"


# Factory
if ML_AVAILABLE:
    MLStrategyOptimizerClass = MLStrategyOptimizer
else:
    MLStrategyOptimizerClass = SimpleRuleOptimizer