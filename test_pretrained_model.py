#!/usr/bin/env python3
"""
Test für das vortrainierte ML-Modell
"""
import sys
sys.path.insert(0, '/home/dev/.openclaw/workspace/tradeforge/src')

from ml_optimizer import MLStrategyOptimizer, TradeFeatures

def test_pretrained_model():
    print("=" * 60)
    print("🧪 Test des vortrainierten ML-Modells")
    print("=" * 60)
    
    # Initialisiere Optimizer
    optimizer = MLStrategyOptimizer()
    
    # Test-Features erstellen
    test_features = TradeFeatures(
        ticker="AAPL",
        strategy="momentum_long",
        entry_price=150.0,
        leverage=2,
        confidence=0.7,
        rsi=30.0,  # Oversold - gutes Setup
        macd=0.5,
        bollinger_position=0.2,  # Nahe unterem Band
        volume_ratio=1.3,
        market_trend=0.05,
        time_of_day=10,
        day_of_week=2,
        vwap_position=-1.5,  # Unter VWAP
        distance_to_support=2.0,
        distance_to_resistance=5.0,
        market_structure_trend=0.3,
        pattern_detected=1,
        fib_38_2_position=0.4,
        higher_highs_count=3,
        higher_lows_count=4
    )
    
    # Vorhersage mit vortrainiertem Modell
    print("\n📊 Test-Vorhersage:")
    print(f"   Ticker: {test_features.ticker}")
    print(f"   RSI: {test_features.rsi} (Oversold)")
    print(f"   VWAP Position: {test_features.vwap_position}%")
    print(f"   Volume Ratio: {test_features.volume_ratio}")
    
    prediction = optimizer.predict_trade_success(test_features)
    
    print("\n🎯 Vorhersage-Ergebnis:")
    for key, value in prediction.items():
        print(f"   {key}: {value}")
    
    # Mehrere Test-Szenarien
    print("\n" + "=" * 60)
    print("📊 Verschiedene Test-Szenarien")
    print("=" * 60)
    
    scenarios = [
        ("Strong Buy", TradeFeatures(
            ticker="NVDA", strategy="momentum_long", entry_price=100.0,
            leverage=2, confidence=0.8, rsi=28.0, macd=0.8,
            bollinger_position=0.1, volume_ratio=1.8, market_trend=0.1,
            time_of_day=10, day_of_week=2, vwap_position=-2.5,
            distance_to_support=1.0, distance_to_resistance=8.0,
            market_structure_trend=0.5, pattern_detected=1,
            fib_38_2_position=0.3, higher_highs_count=4, higher_lows_count=5
        )),
        ("Neutral", TradeFeatures(
            ticker="TSLA", strategy="mean_reversion_long", entry_price=200.0,
            leverage=2, confidence=0.5, rsi=50.0, macd=0.0,
            bollinger_position=0.5, volume_ratio=1.0, market_trend=0.0,
            time_of_day=14, day_of_week=3, vwap_position=0.0,
            distance_to_support=3.0, distance_to_resistance=3.0,
            market_structure_trend=0.0, pattern_detected=0,
            fib_38_2_position=0.5, higher_highs_count=2, higher_lows_count=2
        )),
        ("Weak Signal", TradeFeatures(
            ticker="AMD", strategy="breakout_long", entry_price=50.0,
            leverage=2, confidence=0.4, rsi=75.0, macd=-0.5,
            bollinger_position=0.9, volume_ratio=0.8, market_trend=-0.05,
            time_of_day=15, day_of_week=4, vwap_position=3.0,
            distance_to_support=5.0, distance_to_resistance=1.0,
            market_structure_trend=-0.3, pattern_detected=0,
            fib_38_2_position=0.8, higher_highs_count=1, higher_lows_count=1
        ))
    ]
    
    for name, features in scenarios:
        pred = optimizer.predict_trade_success(features)
        status = "✅ TRADE" if pred.get('should_trade') else "❌ SKIP"
        print(f"\n{name}:")
        print(f"   Erfolgs-Wahrscheinlichkeit: {pred.get('success_probability', 0):.1%}")
        print(f"   Erwarteter Return: {pred.get('expected_return', 0):.2%}")
        print(f"   Confidence: {pred.get('confidence', 'unknown')}")
        print(f"   Entscheidung: {status}")
        print(f"   Modell: {pred.get('model_type', 'unknown')}")
    
    print("\n" + "=" * 60)
    print("✅ Test abgeschlossen!")
    print("=" * 60)

if __name__ == "__main__":
    test_pretrained_model()
