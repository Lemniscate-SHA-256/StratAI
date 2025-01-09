from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import ta
import logging

from ..strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class AIStrategyGenerator:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.trained = False
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare technical indicators as features for the AI model.
        """
        df = data.copy()
        
        # Add technical indicators
        # Trend
        df['sma_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['macd'] = ta.trend.macd_diff(df['Close'])
        
        # Momentum
        df['rsi'] = ta.momentum.rsi(df['Close'])
        df['stoch'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
        
        # Volatility
        df['bbands_width'] = ta.volatility.bollinger_wband(df['Close'])
        df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        
        # Volume
        df['volume_sma'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Price changes
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Remove NaN values
        df = df.dropna()
        
        return df
    
    def generate_labels(self, data: pd.DataFrame, threshold: float = 0.02) -> np.ndarray:
        """
        Generate trading signals based on future returns.
        1 for buy, -1 for sell, 0 for hold
        """
        future_returns = data['Close'].pct_change(5).shift(-5)  # 5-day future returns
        labels = np.zeros(len(data))
        labels[future_returns > threshold] = 1    # Buy signal
        labels[future_returns < -threshold] = -1  # Sell signal
        return labels[:-5]  # Remove last 5 days where we don't have future returns
    
    def train(self, data: pd.DataFrame, threshold: float = 0.02) -> Tuple[float, Dict]:
        """
        Train the AI model on historical data.
        """
        try:
            # Prepare features and labels
            df = self.prepare_features(data)
            feature_cols = ['sma_20', 'sma_50', 'macd', 'rsi', 'stoch', 
                          'bbands_width', 'atr', 'volume_sma', 'returns', 'volatility']
            X = df[feature_cols].values[:-5]  # Remove last 5 days
            y = self.generate_labels(df, threshold)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            accuracy = self.model.score(X_test_scaled, y_test)
            
            # Get feature importance
            self.feature_importance = dict(zip(feature_cols, self.model.feature_importances_))
            self.trained = True
            
            return accuracy, self.feature_importance
            
        except Exception as e:
            logger.error(f"Error training AI model: {str(e)}")
            raise
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals using the trained AI model.
        """
        if not self.trained:
            raise ValueError("Model must be trained before generating signals")
        
        try:
            df = self.prepare_features(data)
            feature_cols = ['sma_20', 'sma_50', 'macd', 'rsi', 'stoch', 
                          'bbands_width', 'atr', 'volume_sma', 'returns', 'volatility']
            X = df[feature_cols].values
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Generate predictions
            signals = self.model.predict(X_scaled)
            return pd.Series(signals, index=df.index)
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            raise
    
    def optimize_parameters(self, data: pd.DataFrame) -> Dict:
        """
        Optimize strategy parameters using machine learning.
        """
        try:
            # Prepare features
            df = self.prepare_features(data)
            
            # Test different thresholds
            thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]
            results = {}
            
            for threshold in thresholds:
                accuracy, _ = self.train(data, threshold)
                results[threshold] = accuracy
            
            # Find optimal threshold
            optimal_threshold = max(results.items(), key=lambda x: x[1])[0]
            
            return {
                "optimal_threshold": optimal_threshold,
                "accuracy_scores": results,
                "feature_importance": self.feature_importance
            }
            
        except Exception as e:
            logger.error(f"Error optimizing parameters: {str(e)}")
            raise

class AIGeneratedStrategy(BaseStrategy):
    def __init__(self, name: str = "AI_Generated"):
        super().__init__(name=name)
        self.ai_generator = AIStrategyGenerator()
        self.trained = False
    
    def train(self, data: pd.DataFrame):
        """Train the AI model with historical data."""
        accuracy, feature_importance = self.ai_generator.train(data)
        self.trained = True
        return accuracy, feature_importance
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals using the trained AI model."""
        if not self.trained:
            raise ValueError("Strategy must be trained before generating signals")
        return self.ai_generator.generate_signals(data)
