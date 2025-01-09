import pandas as pd
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import yfinance as yf

print("HMM Model")

# Load and preprocess data
def load_and_preprocess_data():
    print("Fetching gold data for the past 10 years...")
    gold_data = yf.download('GC=F', start='2013-01-01', end='2023-01-01')
    
    if gold_data.empty:
        raise ValueError("Failed to download gold data. Please check the ticker symbol and try again.")
    
    # Assuming the data has a 'Date' index and 'Close' and 'Volume' columns for gold prices and volume
    gold_data = gold_data[['Close', 'Volume']]
    
    # Calculating Daily Returns
    gold_data['Returns'] = gold_data['Close'].pct_change()
    
    # Calculating Rolling Volatility (e.g. 30-day rolling standard deviation of returns)
    gold_data['Volatility'] = gold_data['Returns'].rolling(window=30).std()
    
    # Calculating Daily Volume Change
    gold_data['Volume_Change'] = gold_data['Volume'].pct_change()
    
    # Drop NaN values that result from the pct_change and rolling calculations
    gold_data.dropna(inplace=True)
    
    # Replace infinite values with NaN and then drop them
    gold_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    gold_data.dropna(inplace=True)
    
    # Clip extreme values to a reasonable range
    gold_data = gold_data.clip(lower=-1e10, upper=1e10)
    
    # Preprocess the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(gold_data[['Returns', 'Volatility', 'Volume_Change']])
    
    return scaled_data, gold_data.index

# Fit HMM model
def fit_hmm_model(data):
    model = hmm.GaussianHMM(n_components=4, covariance_type="diag", n_iter=1000)
    model.fit(data)
    return model

# Predict hidden states
def predict_states(model, data):
    hidden_states = model.predict(data)
    return hidden_states

# Example usage
try:
    data, dates = load_and_preprocess_data()
    hmm_model = fit_hmm_model(data)
    hidden_states = predict_states(hmm_model, data)

    # Add Hidden States To The Original Data For Analysis
    gold_data = yf.download('GC=F', start='2013-01-01', end='2023-01-01')
    gold_data = gold_data[['Close', 'Volume']]
    gold_data['Hidden_States'] = np.nan
    gold_data.loc[dates, 'Hidden_States'] = hidden_states

    print("HMM Model trained and states predicted successfully")
    print(gold_data.head())

    # Plot the Hidden States
    plt.figure(figsize=(12, 6))
    for i in range(hmm_model.n_components):
        state = gold_data[gold_data['Hidden_States'] == i]
        plt.plot(state.index, state['Close'], '.', label=f'State {i}')
    plt.legend()
    plt.title('Hidden States')
    plt.xlabel('Date')
    plt.ylabel('Gold Price')
    plt.show()

except ValueError as e:
    print(e)