import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy

class MovingAverageCrossoverStrategy(BaseStrategy):
    def __init__(self, short_window: int = 20, long_window: int = 50):
        super().__init__(name="MA_Crossover")
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on Moving Average Crossover strategy.
        Returns 1 for buy signals, -1 for sell signals, and 0 for hold.
        """
        if len(data) < self.long_window:
            return pd.Series(0, index=data.index)
        
        # Calculate moving averages
        short_ma = data['Close'].rolling(window=self.short_window).mean()
        long_ma = data['Close'].rolling(window=self.long_window).mean()
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        
        # Crossover signals
        signals[short_ma > long_ma] = 1  # Golden Cross (buy)
        signals[short_ma < long_ma] = -1  # Death Cross (sell)
        
        # Remove signals during the initialization period
        signals[:self.long_window] = 0
        
        return signals
    
    def calculate_position_size(self, data: pd.DataFrame, capital: float) -> float:
        """
        Calculate position size based on volatility and available capital.
        """
        if len(data) < 20:
            return super().calculate_position_size(data, capital)
        
        # Calculate volatility using 20-day standard deviation
        volatility = data['Close'].rolling(window=20).std().iloc[-1]
        price = data['Close'].iloc[-1]
        
        # Adjust position size based on volatility
        volatility_factor = 1.0 / (1.0 + volatility/price)
        return capital * 0.02 * volatility_factor  # Adjust 2% rule based on volatility
