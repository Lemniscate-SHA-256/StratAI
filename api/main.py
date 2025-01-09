from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd

from ..config import settings
from ..data.fetcher import DataFetcher
from ..strategies.moving_average_strategy import MovingAverageCrossoverStrategy
from ..models.strategy_generator import AIGeneratedStrategy
from ..models.sentiment_analyzer import SentimentAnalyzer
from ..backtesting.engine import BacktestEngine

app = FastAPI(
    title="StratAI API",
    description="AI-Powered Trading Strategy Platform",
    version="1.0.0"
)

data_fetcher = DataFetcher()
sentiment_analyzer = SentimentAnalyzer()

class BacktestRequest(BaseModel):
    symbols: List[str]
    start_date: str
    end_date: str
    strategy: str = "MA_Crossover"
    strategy_params: Dict = {"short_window": 20, "long_window": 50}
    initial_capital: float = 100000.0

class BacktestResponse(BaseModel):
    strategy_name: str
    symbol: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    number_of_trades: int

class MarketInsightResponse(BaseModel):
    symbol: str
    sentiment: Dict
    technical: Dict
    summary: str
    ai_prediction: Optional[str]

@app.get("/")
async def root():
    return {
        "message": "Welcome to StratAI API",
        "tagline": "Trade Smarter, Not Harder",
        "version": "1.0.0"
    }

@app.get("/symbols")
async def get_available_symbols():
    return {"symbols": settings.TRADING_PAIRS}

@app.post("/backtest")
async def run_backtest(request: BacktestRequest) -> Dict[str, BacktestResponse]:
    try:
        # Validate dates
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(request.end_date, "%Y-%m-%d")
        
        # Initialize strategy
        if request.strategy == "MA_Crossover":
            strategy = MovingAverageCrossoverStrategy(
                short_window=request.strategy_params.get("short_window", 20),
                long_window=request.strategy_params.get("long_window", 50)
            )
        elif request.strategy == "AI_Generated":
            strategy = AIGeneratedStrategy()
        else:
            raise HTTPException(status_code=400, detail=f"Strategy {request.strategy} not found")
        
        # Fetch data
        data_dict = data_fetcher.fetch_multiple_symbols(
            request.symbols,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        # Train AI strategy if selected
        if request.strategy == "AI_Generated":
            for symbol, data in data_dict.items():
                strategy.train(data)
        
        # Run backtest
        engine = BacktestEngine(initial_capital=request.initial_capital)
        results = engine.run_multiple(strategy, data_dict)
        
        # Format response
        response = {}
        for symbol, result in results.items():
            response[symbol] = BacktestResponse(
                strategy_name=result.strategy_name,
                symbol=symbol,
                total_return=result.total_return,
                sharpe_ratio=result.sharpe_ratio,
                max_drawdown=result.max_drawdown,
                number_of_trades=len(result.trades)
            )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market-data/{symbol}")
async def get_market_data(symbol: str, 
                         period: str = "1y",
                         interval: str = "1d"):
    try:
        df = data_fetcher.fetch_data(symbol, period=period, interval=interval)
        return df.to_dict(orient="index")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market-insights/{symbol}")
async def get_market_insights(symbol: str) -> MarketInsightResponse:
    try:
        # Get market insights including sentiment and technical analysis
        insights = sentiment_analyzer.get_market_insights(symbol)
        
        # Get recent data for AI prediction
        df = data_fetcher.fetch_data(symbol, period="6mo")
        
        # Initialize and train AI strategy
        ai_strategy = AIGeneratedStrategy()
        accuracy, feature_importance = ai_strategy.train(df)
        
        # Get latest prediction
        signals = ai_strategy.generate_signals(df)
        latest_signal = signals.iloc[-1]
        
        # Convert signal to prediction
        if latest_signal == 1:
            ai_prediction = "bullish"
        elif latest_signal == -1:
            ai_prediction = "bearish"
        else:
            ai_prediction = "neutral"
        
        return MarketInsightResponse(
            symbol=symbol,
            sentiment=insights["sentiment"],
            technical=insights["technical"],
            summary=insights["summary"],
            ai_prediction=ai_prediction
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/optimize-strategy/{symbol}")
async def optimize_strategy(symbol: str,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None):
    try:
        # Fetch historical data
        data = data_fetcher.fetch_data(
            symbol,
            start_date=start_date,
            end_date=end_date,
            period="1y" if not start_date else None
        )
        
        # Initialize AI strategy generator
        strategy_generator = AIGeneratedStrategy()
        
        # Optimize parameters
        optimization_results = strategy_generator.ai_generator.optimize_parameters(data)
        
        return {
            "symbol": symbol,
            "optimization_results": optimization_results,
            "recommendation": {
                "threshold": optimization_results["optimal_threshold"],
                "accuracy": optimization_results["accuracy_scores"][optimization_results["optimal_threshold"]],
                "important_features": dict(sorted(
                    optimization_results["feature_importance"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5])
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
