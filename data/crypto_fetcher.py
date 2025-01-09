from typing import Dict, List, Optional
import pandas as pd
import requests
from datetime import datetime, timedelta
import logging
import time
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class CryptoDataFetcher:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
    def fetch_data(self,
                  symbol: str,
                  interval: str = "1d",
                  limit: int = 1000) -> pd.DataFrame:
        """
        Fetch cryptocurrency data from Binance API.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Kline interval (1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,1M)
            limit: Number of candles to fetch (max 1000)
        """
        try:
            cache_key = f"{symbol}_{interval}_{limit}"
            
            # Check cache
            if cache_key in self.cache:
                cached_data, cached_time = self.cache[cache_key]
                if time.time() - cached_time < self.cache_duration:
                    return cached_data
            
            # Prepare request
            endpoint = f"{self.base_url}/klines"
            params = {
                "symbol": symbol.upper().replace("/", ""),
                "interval": interval,
                "limit": limit
            }
            
            # Make request
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
                'taker_buy_quote_volume', 'ignore'
            ])
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Set index
            df.set_index('timestamp', inplace=True)
            
            # Cache the result
            self.cache[cache_key] = (df, time.time())
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching crypto data for {symbol}: {str(e)}")
            raise
    
    def fetch_multiple_symbols(self,
                             symbols: List[str],
                             interval: str = "1d",
                             limit: int = 1000) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols in parallel.
        """
        try:
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {
                    symbol: executor.submit(self.fetch_data, symbol, interval, limit)
                    for symbol in symbols
                }
                
                results = {}
                for symbol, future in futures.items():
                    try:
                        results[symbol] = future.result()
                    except Exception as e:
                        logger.error(f"Error fetching {symbol}: {str(e)}")
                        continue
                
                return results
                
        except Exception as e:
            logger.error(f"Error in parallel fetch: {str(e)}")
            raise
    
    def get_market_depth(self, symbol: str, limit: int = 100) -> Dict:
        """
        Fetch order book data for a symbol.
        """
        try:
            endpoint = f"{self.base_url}/depth"
            params = {
                "symbol": symbol.upper().replace("/", ""),
                "limit": limit
            }
            
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to DataFrames
            bids = pd.DataFrame(data['bids'], columns=['price', 'quantity'])
            asks = pd.DataFrame(data['asks'], columns=['price', 'quantity'])
            
            # Convert types
            for df in [bids, asks]:
                df['price'] = pd.to_numeric(df['price'])
                df['quantity'] = pd.to_numeric(df['quantity'])
            
            return {
                "bids": bids.to_dict(orient='records'),
                "asks": asks.to_dict(orient='records')
            }
            
        except Exception as e:
            logger.error(f"Error fetching market depth for {symbol}: {str(e)}")
            raise
    
    def get_recent_trades(self, symbol: str, limit: int = 500) -> pd.DataFrame:
        """
        Fetch recent trades for a symbol.
        """
        try:
            endpoint = f"{self.base_url}/trades"
            params = {
                "symbol": symbol.upper().replace("/", ""),
                "limit": limit
            }
            
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Convert types
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df['price'] = pd.to_numeric(df['price'])
            df['qty'] = pd.to_numeric(df['qty'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching recent trades for {symbol}: {str(e)}")
            raise
    
    def get_ticker_statistics(self, symbol: str) -> Dict:
        """
        Fetch 24-hour ticker statistics.
        """
        try:
            endpoint = f"{self.base_url}/ticker/24hr"
            params = {"symbol": symbol.upper().replace("/", "")}
            
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert numeric fields
            numeric_fields = [
                'priceChange', 'priceChangePercent', 'weightedAvgPrice',
                'prevClosePrice', 'lastPrice', 'lastQty', 'bidPrice', 'askPrice',
                'openPrice', 'highPrice', 'lowPrice', 'volume', 'quoteVolume'
            ]
            
            for field in numeric_fields:
                if field in data:
                    data[field] = float(data[field])
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching ticker statistics for {symbol}: {str(e)}")
            raise
