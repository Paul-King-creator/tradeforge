#!/usr/bin/env python3
"""
TradeForge Stock Research Report
Analysiert 50 Aktien aus der Watchlist für Day-Trading
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Die 50 Aktien aus der Watchlist
WATCHLIST = [
    'AAPL', 'MSFT', 'NVDA', 'AMD', 'AMZN', 'GOOGL', 'META', 'NFLX', 'TSLA', 'CRM',
    'AVGO', 'QCOM', 'INTC', 'MU', 'LRCX', 'KLAC',
    'JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'V', 'MA',
    'JNJ', 'PFE', 'UNH', 'ABBV', 'LLY', 'MRK', 'BMY',
    'XOM', 'CVX', 'COP', 'SLB',
    'BA', 'CAT', 'GE', 'HON', 'UPS',
    'WMT', 'HD', 'COST', 'NKE', 'SBUX', 'MCD',
    'SPY', 'QQQ', 'IWM', 'XLK'
]

def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(period).mean()
    return atr

def analyze_stock(symbol, spy_data=None):
    """Analyze a single stock"""
    print(f"Analyzing {symbol}...")
    
    try:
        # Download 60 days of data with intraday for best trading hours
        stock = yf.Ticker(symbol)
        
        # Get daily data for volatility and volume
        df_daily = stock.history(period="60d", interval="1d")
        
        if len(df_daily) < 30:
            print(f"  ⚠️  Not enough data for {symbol}")
            return None
        
        # Calculate metrics
        current_price = df_daily['Close'].iloc[-1]
        avg_volume = df_daily['Volume'].mean()
        
        # ATR and volatility
        atr = calculate_atr(df_daily)
        avg_atr = atr.mean()
        volatility_pct = (avg_atr / current_price) * 100
        
        # Price range analysis
        price_range = df_daily['High'] - df_daily['Low']
        avg_daily_range_pct = (price_range.mean() / current_price) * 100
        
        # Volume trend
        recent_volume = df_daily['Volume'].tail(10).mean()
        volume_ratio = recent_volume / avg_volume
        
        # Calculate returns for momentum analysis
        df_daily['Returns'] = df_daily['Close'].pct_change()
        
        # Best trading hours analysis (if intraday data available)
        try:
            df_intraday = stock.history(period="5d", interval="30m")
            if len(df_intraday) > 0:
                df_intraday['Hour'] = df_intraday.index.hour
                hourly_volume = df_intraday.groupby('Hour')['Volume'].mean()
                hourly_volatility = df_intraday.groupby('Hour').apply(
                    lambda x: (x['High'] - x['Low']).mean() / x['Close'].mean() * 100
                )
                
                # Best hours = high volume + high volatility
                hourly_score = (hourly_volume / hourly_volume.max()) + (hourly_volatility / hourly_volatility.max())
                best_hours = hourly_score.nlargest(3).index.tolist()
                best_hours_str = [f"{h}:00" for h in sorted(best_hours)]
            else:
                best_hours_str = ["09:30", "10:00", "15:30"]
        except:
            best_hours_str = ["09:30", "10:00", "15:30"]
        
        # Correlation with SPY
        correlation_with_spy = None
        if spy_data is not None and symbol != 'SPY':
            aligned_data = pd.concat([df_daily['Returns'], spy_data['Returns']], axis=1).dropna()
            if len(aligned_data) > 10:
                correlation_with_spy = aligned_data.corr().iloc[0, 1]
        
        # Determine recommended strategy
        if volatility_pct > 3.0 and volume_ratio > 1.0:
            strategy = "Breakout"
        elif volatility_pct > 2.0 and df_daily['Returns'].std() > 0.02:
            strategy = "Momentum"
        else:
            strategy = "Mean Reversion"
        
        # Calculate day trading score (0-100)
        # Higher volatility = better for day trading
        # Higher volume = better liquidity
        vol_score = min(volatility_pct * 10, 40)  # Max 40 points for volatility
        volume_score = min(volume_ratio * 20, 30)  # Max 30 points for volume
        range_score = min(avg_daily_range_pct * 5, 30)  # Max 30 points for range
        
        day_trading_score = vol_score + volume_score + range_score
        
        return {
            'symbol': symbol,
            'current_price': round(current_price, 2),
            'avg_volume': int(avg_volume),
            'recent_volume_ratio': round(volume_ratio, 2),
            'avg_daily_volatility_pct': round(volatility_pct, 2),
            'avg_daily_range_pct': round(avg_daily_range_pct, 2),
            'atr_14': round(avg_atr, 2),
            'best_trading_hours': best_hours_str,
            'correlation_with_spy': round(correlation_with_spy, 3) if correlation_with_spy else None,
            'recommended_strategy': strategy,
            'day_trading_score': round(day_trading_score, 1),
            '30d_return_pct': round(df_daily['Close'].pct_change(30).iloc[-1] * 100, 2) if len(df_daily) >= 30 else None,
            'data_quality': 'good' if len(df_daily) >= 50 else 'limited'
        }
        
    except Exception as e:
        print(f"  ❌ Error analyzing {symbol}: {e}")
        return None

def main():
    print("=" * 60)
    print("TradeForge Stock Research Report")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # First get SPY data for correlation
    print("\n📊 Loading SPY data for correlation analysis...")
    spy = yf.Ticker("SPY")
    spy_data = spy.history(period="60d", interval="1d")
    spy_data['Returns'] = spy_data['Close'].pct_change()
    
    # Analyze all stocks
    results = []
    print(f"\n🔍 Analyzing {len(WATCHLIST)} stocks...\n")
    
    for symbol in WATCHLIST:
        analysis = analyze_stock(symbol, spy_data)
        if analysis:
            results.append(analysis)
    
    # Create rankings
    print("\n📈 Creating rankings...")
    
    # Sort by day trading score
    top_day_trading = sorted(results, key=lambda x: x['day_trading_score'], reverse=True)[:10]
    
    # Sort by volatility
    most_volatile = sorted(results, key=lambda x: x['avg_daily_volatility_pct'], reverse=True)[:10]
    
    # Sort by volume
    highest_volume = sorted(results, key=lambda x: x['avg_volume'], reverse=True)[:10]
    
    # Create final report
    report = {
        'generated_at': datetime.now().isoformat(),
        'analysis_period_days': 60,
        'total_stocks_analyzed': len(results),
        'stock_analysis': results,
        'rankings': {
            'top_10_day_trading_stocks': [
                {
                    'rank': i + 1,
                    'symbol': s['symbol'],
                    'day_trading_score': s['day_trading_score'],
                    'volatility_pct': s['avg_daily_volatility_pct'],
                    'avg_volume': s['avg_volume'],
                    'strategy': s['recommended_strategy']
                }
                for i, s in enumerate(top_day_trading)
            ],
            'most_volatile': [s['symbol'] for s in most_volatile],
            'highest_volume': [s['symbol'] for s in highest_volume]
        },
        'summary': {
            'avg_volatility_all': round(np.mean([r['avg_daily_volatility_pct'] for r in results]), 2),
            'avg_volume_all': int(np.mean([r['avg_volume'] for r in results])),
            'best_strategy_distribution': {}
        }
    }
    
    # Count strategies
    strategy_counts = {}
    for r in results:
        strategy = r['recommended_strategy']
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    report['summary']['best_strategy_distribution'] = strategy_counts
    
    # Save to JSON
    output_path = '/home/dev/.openclaw/workspace/tradeforge/data/stock_research.json'
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Report saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("TOP 10 DAY-TRADING AKTIEN")
    print("=" * 60)
    print(f"{'Rank':<6}{'Symbol':<8}{'Score':<8}{'Vol%':<8}{'Strategy':<15}")
    print("-" * 60)
    for stock in top_day_trading:
        print(f"{top_day_trading.index(stock)+1:<6}{stock['symbol']:<8}{stock['day_trading_score']:<8}{stock['avg_daily_volatility_pct']:<8}{stock['recommended_strategy']:<15}")
    
    print("\n" + "=" * 60)
    print("STRATEGY DISTRIBUTION")
    print("=" * 60)
    for strategy, count in strategy_counts.items():
        print(f"  {strategy}: {count} stocks")
    
    print("\n" + "=" * 60)
    print("MARKET STATISTICS")
    print("=" * 60)
    print(f"  Average Volatility: {report['summary']['avg_volatility_all']:.2f}%")
    print(f"  Average Volume: {report['summary']['avg_volume_all']:,}")
    print(f"  Stocks Analyzed: {len(results)}")
    
    return report

if __name__ == "__main__":
    report = main()
