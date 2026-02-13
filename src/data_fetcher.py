"""
Data Fetcher - Marktdaten von Yahoo Finance
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import pickle

class DataFetcher:
    def __init__(self, cache_dir="data/market_data"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_intraday_data(self, ticker, period="5d", interval="15m"):
        """15-Minuten Daten für Intraday-Analyse"""
        cache_file = f"{self.cache_dir}/{ticker}_{period}_{interval}.pkl"
        
        # Cache prüfen (max 15 Min alt)
        if os.path.exists(cache_file):
            cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - cache_time < timedelta(minutes=15):
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)
            
            if df.empty:
                return None
            
            # Cache speichern
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)
            
            return df
        except Exception as e:
            print(f"❌ Fehler beim Laden von {ticker}: {e}")
            return None
    
    def get_daily_data(self, ticker, days=60):
        """Tägliche Daten für Swing-Analyse"""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=f"{days}d")
            return df
        except Exception as e:
            print(f"❌ Fehler beim Laden von {ticker}: {e}")
            return None
    
    def get_current_price(self, ticker):
        """Aktueller Preis"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return info.get('regularMarketPrice', None)
        except:
            return None
    
    def get_watchlist_data(self, watchlist, period="5d", interval="15m"):
        """Daten für alle Watchlist-Werte"""
        data = {}
        for ticker in watchlist:
            print(f"📊 Lade {ticker}...")
            df = self.get_intraday_data(ticker, period, interval)
            if df is not None:
                data[ticker] = df
        return data