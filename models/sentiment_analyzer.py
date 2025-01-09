from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from textblob import TextBlob
import yfinance as yf
import requests
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.sentiment_cache = {}
        
    def analyze_news(self, symbol: str, days: int = 7) -> Dict:
        """
        Analyze news sentiment for a given symbol over the specified number of days.
        """
        try:
            # Get company news from yfinance
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            if not news:
                return {
                    "overall_sentiment": 0,
                    "sentiment_breakdown": {"positive": 0, "neutral": 0, "negative": 0},
                    "confidence": 0,
                    "news_count": 0
                }
            
            # Analyze sentiment for each news item
            sentiments = []
            for item in news:
                if 'title' in item:
                    blob = TextBlob(item['title'])
                    sentiments.append(blob.sentiment.polarity)
            
            if not sentiments:
                return {
                    "overall_sentiment": 0,
                    "sentiment_breakdown": {"positive": 0, "neutral": 0, "negative": 0},
                    "confidence": 0,
                    "news_count": 0
                }
            
            # Calculate sentiment metrics
            overall_sentiment = np.mean(sentiments)
            sentiment_breakdown = {
                "positive": len([s for s in sentiments if s > 0.1]) / len(sentiments),
                "neutral": len([s for s in sentiments if -0.1 <= s <= 0.1]) / len(sentiments),
                "negative": len([s for s in sentiments if s < -0.1]) / len(sentiments)
            }
            confidence = 1 - np.std(sentiments)  # Higher std means lower confidence
            
            return {
                "overall_sentiment": overall_sentiment,
                "sentiment_breakdown": sentiment_breakdown,
                "confidence": confidence,
                "news_count": len(sentiments)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing news for {symbol}: {str(e)}")
            return {
                "overall_sentiment": 0,
                "sentiment_breakdown": {"positive": 0, "neutral": 0, "negative": 0},
                "confidence": 0,
                "news_count": 0
            }
    
    def get_market_insights(self, symbol: str) -> Dict:
        """
        Get comprehensive market insights including technical and sentiment analysis.
        """
        try:
            # Get sentiment analysis
            sentiment = self.analyze_news(symbol)
            
            # Get technical indicators
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="6mo")
            
            if hist.empty:
                raise ValueError(f"No historical data available for {symbol}")
            
            # Calculate technical indicators
            current_price = hist['Close'].iloc[-1]
            sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
            sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1]
            rsi = self._calculate_rsi(hist['Close'])
            
            # Generate market insights
            insights = {
                "sentiment": sentiment,
                "technical": {
                    "trend": "bullish" if current_price > sma_50 > sma_200 else "bearish",
                    "strength": "strong" if abs(current_price - sma_50) / sma_50 > 0.05 else "weak",
                    "rsi": rsi
                },
                "summary": self._generate_summary(sentiment, current_price, sma_50, sma_200, rsi)
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting market insights for {symbol}: {str(e)}")
            raise
    
    def _calculate_rsi(self, prices: pd.Series, periods: int = 14) -> float:
        """Calculate RSI technical indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    def _generate_summary(self, sentiment: Dict, current_price: float, 
                         sma_50: float, sma_200: float, rsi: float) -> str:
        """Generate a human-readable summary of market insights."""
        summary_parts = []
        
        # Sentiment summary
        if sentiment['overall_sentiment'] > 0.2:
            summary_parts.append("Market sentiment is strongly positive")
        elif sentiment['overall_sentiment'] > 0:
            summary_parts.append("Market sentiment is slightly positive")
        elif sentiment['overall_sentiment'] < -0.2:
            summary_parts.append("Market sentiment is strongly negative")
        elif sentiment['overall_sentiment'] < 0:
            summary_parts.append("Market sentiment is slightly negative")
        else:
            summary_parts.append("Market sentiment is neutral")
        
        # Technical analysis summary
        if current_price > sma_50 > sma_200:
            summary_parts.append("Technical indicators suggest an upward trend")
        elif current_price < sma_50 < sma_200:
            summary_parts.append("Technical indicators suggest a downward trend")
        else:
            summary_parts.append("Technical indicators are mixed")
        
        # RSI interpretation
        if rsi > 70:
            summary_parts.append("The asset may be overbought")
        elif rsi < 30:
            summary_parts.append("The asset may be oversold")
        
        return " | ".join(summary_parts)
