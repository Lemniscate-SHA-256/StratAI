from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from ..strategies.base_strategy import BaseStrategy
from datetime import datetime

class BacktestResult:
    def __init__(self, 
                 strategy_name: str,
                 symbol: str,
                 trades: pd.DataFrame,
                 equity_curve: pd.Series,
                 total_return: float,
                 sharpe_ratio: float,
                 max_drawdown: float):
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.trades = trades
        self.equity_curve = equity_curve
        self.total_return = total_return
        self.sharpe_ratio = sharpe_ratio
        self.max_drawdown = max_drawdown

class BacktestEngine:
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.commission = 0.001  # 0.1% commission per trade
        
    def run(self, 
            strategy: BaseStrategy,
            data: pd.DataFrame,
            symbol: str) -> BacktestResult:
        """
        Run backtest for a single symbol using the provided strategy.
        """
        # Initialize backtest variables
        capital = self.initial_capital
        position = 0
        trades = []
        equity_curve = []
        
        # Generate signals
        signals = strategy.generate_signals(data)
        
        # Simulate trading
        for i in range(len(data)):
            current_price = data['Close'].iloc[i]
            current_date = data.index[i]
            signal = signals.iloc[i]
            
            # Check for position exit
            if position != 0 and (
                (position == 1 and signal == -1) or
                (position == -1 and signal == 1)
            ):
                # Calculate profit/loss
                pnl = position * (current_price - entry_price) * position_size
                pnl -= position_size * current_price * self.commission  # Exit commission
                
                # Update capital
                capital += pnl
                
                # Record trade
                trades.append({
                    'Entry Date': entry_date,
                    'Exit Date': current_date,
                    'Entry Price': entry_price,
                    'Exit Price': current_price,
                    'Position': position,
                    'Size': position_size,
                    'PnL': pnl
                })
                
                position = 0
            
            # Check for new position entry
            if position == 0 and signal != 0:
                position = signal
                entry_price = current_price
                entry_date = current_date
                position_size = strategy.calculate_position_size(
                    data.iloc[:i+1], capital
                )
                
                # Deduct entry commission
                capital -= position_size * current_price * self.commission
            
            # Record equity
            equity_curve.append(capital)
        
        # Create trades DataFrame
        trades_df = pd.DataFrame(trades)
        
        # Calculate performance metrics
        equity_curve = pd.Series(equity_curve, index=data.index)
        total_return = (capital - self.initial_capital) / self.initial_capital
        
        # Calculate Sharpe Ratio (assuming risk-free rate of 0.02)
        returns = equity_curve.pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * (returns.mean() - 0.02/252) / returns.std()
        
        # Calculate Maximum Drawdown
        rolling_max = equity_curve.expanding().max()
        drawdowns = equity_curve / rolling_max - 1.0
        max_drawdown = drawdowns.min()
        
        return BacktestResult(
            strategy_name=strategy.name,
            symbol=symbol,
            trades=trades_df,
            equity_curve=equity_curve,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown
        )
    
    def run_multiple(self,
                    strategy: BaseStrategy,
                    data_dict: Dict[str, pd.DataFrame]) -> Dict[str, BacktestResult]:
        """
        Run backtest for multiple symbols using the same strategy.
        """
        results = {}
        for symbol, data in data_dict.items():
            results[symbol] = self.run(strategy, data, symbol)
        return results
