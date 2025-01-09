from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Data Sources
    YAHOO_FINANCE_ENABLED: bool = True
    ALPHA_VANTAGE_API_KEY: Optional[str] = None
    
    # Database Configuration
    DB_ENABLED: bool = False
    DB_URL: Optional[str] = None
    
    # Trading Configuration
    DEFAULT_TIMEFRAME: str = "1d"
    TRADING_PAIRS: list[str] = ["AAPL", "MSFT", "GOOGL"]
    BACKTEST_START_DATE: str = "2023-01-01"
    BACKTEST_END_DATE: str = "2023-12-31"
    
    # Risk Management
    MAX_POSITION_SIZE: float = 0.1  # 10% of portfolio
    STOP_LOSS_PCT: float = 0.02     # 2% stop loss
    TAKE_PROFIT_PCT: float = 0.06   # 6% take profit
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
