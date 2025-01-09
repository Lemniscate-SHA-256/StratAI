from abc import ABC, abstractmethod
from typing import Dict, Optional
import pandas as pd
import numpy as np

class BaseStrategy(ABC):
    def __init__(self, name: str):
        self.name = name
        self.position = 0  # Current position: 1 (long), -1 (short), 0 (neutral)
        self.positions: Dict[str, float] = {}  # Track multiple positions
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on the strategy logic.
        Returns a pandas Series with values: 1 (buy), -1 (sell), 0 (hold)
        """
        pass
    
    def calculate_position_size(self, data: pd.DataFrame, capital: float) -> float:
        """
        Calculate the position size based on available capital and risk parameters.
        Override this method in derived classes for custom position sizing.
        """
        return capital * 0.02  # Default to 2% of capital per trade
    
    def validate_signal(self, signal: int, current_price: float, 
                       available_capital: float) -> bool:
        """
        Validate if a trading signal should be executed based on current conditions.
        """
        # Basic validation logic
        if signal != 0 and self.position == signal:
            return False  # Already in the desired position
        
        if available_capital <= 0 and signal == 1:
            return False  # No capital available for long position
            
        return True
    
    def get_stop_loss(self, entry_price: float, signal: int) -> float:
        """
        Calculate stop loss price based on entry price and signal direction.
        """
        if signal == 1:  # Long position
            return entry_price * 0.98  # 2% stop loss
        elif signal == -1:  # Short position
            return entry_price * 1.02
        return 0.0
    
    def get_take_profit(self, entry_price: float, signal: int) -> float:
        """
        Calculate take profit price based on entry price and signal direction.
        """
        if signal == 1:  # Long position
            return entry_price * 1.06  # 6% take profit
        elif signal == -1:  # Short position
            return entry_price * 0.94
        return 0.0
