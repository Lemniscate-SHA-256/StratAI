import pandas as pd
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import argparse

print("HMM Model")

# Load and preprocess data
def load_and_preprocess_data(ticker, start_date, end_date):
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    
    if data.empty:
        raise ValueError(f"Failed to download data for {ticker}. Please check the ticker symbol and try again.")
    
    # Assuming the data has a 'Date' index and 'Close' and 'Volume' columns
    data = data[['Close', 'Volume']]
    
    # Calculating Daily Returns
    data['Returns'] = data['Close'].pct_change()
    
    # Calculating Rolling Volatility (e.g. 30-day rolling standard deviation of returns)
    data['Volatility'] = data['Returns'].rolling(window=30).std()
    
    # Calculating Daily Volume Change
    data['Volume_Change'] = data['Volume'].pct_change()
    
    # Drop NaN values that result from the pct_change and rolling calculations
    data.dropna(inplace=True)
    
    # Replace infinite values with NaN and then drop them
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    
    # Clip extreme values to a reasonable range
    data = data.clip(lower=-1e10, upper=1e10)
    
    # Preprocess the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[['Returns', 'Volatility', 'Volume_Change']])
    
    return scaled_data, data.index

# Fit HMM model
def fit_hmm_model(data):
    model = hmm.GaussianHMM(n_components=4, covariance_type="diag", n_iter=1000)
    model.fit(data)
    return model

# Predict hidden states
def predict_states(model, data):
    hidden_states = model.predict(data)
    return hidden_states

# Plot the Hidden States
def plot_hidden_states(dates, data, hidden_states, plotter, n_components):
    plt.figure(figsize=(12, 6))
    for i in range(n_components):
        state = data[hidden_states == i]
        if plotter == 'line':
            plt.plot(state.index, state['Close'], '.', label=f'State {i}')
        elif plotter == 'scatter':
            plt.scatter(state.index, state['Close'], label=f'State {i}')
    plt.legend()
    plt.title('Hidden States')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

# Main function
def main():
    parser = argparse.ArgumentParser(description='HMM Model for Asset Analysis')
    parser.add_argument('--ticker', type=str, required=True, help='Ticker symbol of the asset')
    parser.add_argument('--start_date', type=str, required=True, help='Start date for data fetching (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, required=True, help='End date for data fetching (YYYY-MM-DD)')
    parser.add_argument('--plotter', type=str, choices=['line', 'scatter'], default='line', help='Type of plot (line or scatter)')
    
    args = parser.parse_args()
    
    try:
        data, dates = load_and_preprocess_data(args.ticker, args.start_date, args.end_date)
        hmm_model = fit_hmm_model(data)
        hidden_states = predict_states(hmm_model, data)

        # Add Hidden States To The Original Data For Analysis
        asset_data = yf.download(args.ticker, start=args.start_date, end=args.end_date)
        asset_data = asset_data[['Close', 'Volume']]
        
        # Apply the same preprocessing steps to asset_data
        asset_data['Returns'] = asset_data['Close'].pct_change()
        asset_data['Volatility'] = asset_data['Returns'].rolling(window=30).std()
        asset_data['Volume_Change'] = asset_data['Volume'].pct_change()
        asset_data.dropna(inplace=True)
        asset_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        asset_data.dropna(inplace=True)
        asset_data = asset_data.clip(lower=-1e10, upper=1e10)
        
        asset_data['Hidden_States'] = np.nan
        
        # Align the dates and handle missing dates
        aligned_dates = asset_data.index.intersection(dates)
        aligned_hidden_states = hidden_states[:len(aligned_dates)]
        asset_data.loc[aligned_dates, 'Hidden_States'] = aligned_hidden_states

        print("HMM Model trained and states predicted successfully")
        print(asset_data.head())

        # Plot the Hidden States
        plot_hidden_states(aligned_dates, asset_data, aligned_hidden_states, args.plotter, hmm_model.n_components)

    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
