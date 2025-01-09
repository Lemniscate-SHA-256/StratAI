from typing import Optional, List, Dict
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self):
        self.cache: Dict[str, pd.DataFrame] = {}
    
    def fetch_data(self, 
                  symbol: str, 
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None,
                  period: str = "1y",
                  interval: str = "1d") -> pd.DataFrame:
        """
        Fetch historical market data for a given symbol.
        
        Args:
            symbol: The stock symbol (e.g., 'AAPL')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            period: Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
            interval: Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        
        Returns:
            DataFrame with historical price data
        """
        try:
            # Create cache key
            cache_key = f"{symbol}_{start_date}_{end_date}_{period}_{interval}"
            
            # Check cache first
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Fetch data from yfinance
            ticker = yf.Ticker(symbol)
            
            if start_date and end_date:
                df = ticker.history(start=start_date, end=end_date, interval=interval)
            else:
                df = ticker.history(period=period, interval=interval)
            
            # Basic data validation
            if df.empty:
                raise ValueError(f"No data returned for symbol {symbol}")
            
            # Add technical indicators if needed
            # self._add_technical_indicators(df)
            
            # Cache the result
            self.cache[cache_key] = df
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise
    
    def fetch_multiple_symbols(self, 
                             symbols: List[str],
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None,
                             period: str = "1y",
                             interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple symbols.
        """
        result = {}
        for symbol in symbols:
            try:
                df = self.fetch_data(symbol, start_date, end_date, period, interval)
                result[symbol] = df
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {str(e)}")
                continue
        return result
    
    def clear_cache(self):
        """Clear the data cache"""
        self.cache.clear()
