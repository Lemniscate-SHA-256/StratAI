from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class RiskAnalyzer:
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        
    def calculate_portfolio_metrics(self, 
                                 positions: List[Dict],
                                 historical_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Calculate comprehensive portfolio risk metrics.
        """
        try:
            # Calculate portfolio returns
            portfolio_value = sum(p['quantity'] * p['current_price'] for p in positions)
            weights = [p['quantity'] * p['current_price'] / portfolio_value for p in positions]
            
            # Calculate returns for each position
            returns_data = {}
            for symbol, data in historical_data.items():
                returns_data[symbol] = data['Close'].pct_change().dropna()
            
            # Combine returns into a DataFrame
            returns_df = pd.DataFrame(returns_data)
            
            # Calculate portfolio metrics
            portfolio_return = self._calculate_portfolio_return(positions)
            volatility = self._calculate_portfolio_volatility(returns_df, weights)
            var = self._calculate_value_at_risk(returns_df, weights, portfolio_value)
            sharpe = self._calculate_sharpe_ratio(portfolio_return, volatility)
            beta = self._calculate_portfolio_beta(returns_df, weights)
            
            # Calculate correlation matrix
            correlation_matrix = returns_df.corr().values.tolist()
            
            return {
                "portfolio_value": portfolio_value,
                "portfolio_return": portfolio_return,
                "volatility": volatility,
                "value_at_risk": var,
                "sharpe_ratio": sharpe,
                "beta": beta,
                "correlation_matrix": correlation_matrix,
                "risk_metrics": self._calculate_risk_metrics(positions, returns_df),
                "diversification_score": self._calculate_diversification_score(positions),
                "risk_concentration": self._calculate_risk_concentration(positions)
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {str(e)}")
            raise
    
    def _calculate_portfolio_return(self, positions: List[Dict]) -> float:
        """Calculate total portfolio return."""
        total_pnl = sum(p['pnl'] for p in positions)
        total_investment = sum(p['quantity'] * p['entry_price'] for p in positions)
        return total_pnl / total_investment if total_investment > 0 else 0
    
    def _calculate_portfolio_volatility(self, 
                                     returns_df: pd.DataFrame,
                                     weights: List[float]) -> float:
        """Calculate portfolio volatility using covariance matrix."""
        cov_matrix = returns_df.cov()
        portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
        return np.sqrt(portfolio_var * 252)  # Annualized volatility
    
    def _calculate_value_at_risk(self,
                               returns_df: pd.DataFrame,
                               weights: List[float],
                               portfolio_value: float,
                               confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk (VaR) using historical simulation."""
        portfolio_returns = np.sum(returns_df * weights, axis=1)
        var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        return abs(var * portfolio_value)
    
    def _calculate_sharpe_ratio(self,
                              portfolio_return: float,
                              portfolio_volatility: float) -> float:
        """Calculate Sharpe Ratio."""
        excess_return = portfolio_return - self.risk_free_rate
        return excess_return / portfolio_volatility if portfolio_volatility > 0 else 0
    
    def _calculate_portfolio_beta(self,
                                returns_df: pd.DataFrame,
                                weights: List[float]) -> float:
        """Calculate portfolio beta relative to a market index."""
        # Assuming the first column is the market index
        market_returns = returns_df.iloc[:, 0]
        portfolio_returns = np.sum(returns_df * weights, axis=1)
        
        covariance = np.cov(portfolio_returns, market_returns)[0][1]
        market_variance = np.var(market_returns)
        
        return covariance / market_variance if market_variance > 0 else 1
    
    def _calculate_risk_metrics(self,
                              positions: List[Dict],
                              returns_df: pd.DataFrame) -> Dict:
        """Calculate additional risk metrics."""
        portfolio_returns = returns_df.sum(axis=1)
        
        return {
            "max_drawdown": self._calculate_max_drawdown(portfolio_returns),
            "skewness": stats.skew(portfolio_returns),
            "kurtosis": stats.kurtosis(portfolio_returns),
            "tail_risk": self._calculate_tail_risk(portfolio_returns)
        }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        return abs(drawdowns.min())
    
    def _calculate_tail_risk(self, returns: pd.Series) -> float:
        """Calculate tail risk using Expected Shortfall (CVaR)."""
        confidence_level = 0.95
        var = np.percentile(returns, (1 - confidence_level) * 100)
        return abs(returns[returns <= var].mean())
    
    def _calculate_diversification_score(self, positions: List[Dict]) -> float:
        """Calculate portfolio diversification score."""
        # Group positions by asset type
        asset_types = {}
        total_value = 0
        
        for position in positions:
            asset_type = position['type']
            position_value = position['quantity'] * position['current_price']
            asset_types[asset_type] = asset_types.get(asset_type, 0) + position_value
            total_value += position_value
        
        # Calculate Herfindahl-Hirschman Index (HHI)
        if total_value == 0:
            return 0
            
        hhi = sum((value / total_value) ** 2 for value in asset_types.values())
        
        # Convert HHI to a 0-100 score where 100 is perfectly diversified
        return (1 - hhi) * 100
    
    def _calculate_risk_concentration(self, positions: List[Dict]) -> Dict:
        """Calculate risk concentration by asset type and individual position."""
        total_value = sum(p['quantity'] * p['current_price'] for p in positions)
        
        if total_value == 0:
            return {"asset_types": {}, "positions": {}}
        
        # Calculate concentration by asset type
        asset_types = {}
        for position in positions:
            asset_type = position['type']
            position_value = position['quantity'] * position['current_price']
            asset_types[asset_type] = asset_types.get(asset_type, 0) + (position_value / total_value) * 100
        
        # Calculate concentration by position
        position_concentration = {
            p['symbol']: (p['quantity'] * p['current_price'] / total_value) * 100
            for p in positions
        }
        
        return {
            "asset_types": asset_types,
            "positions": position_concentration
        }
