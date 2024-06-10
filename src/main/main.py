import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from datetime import date
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from dotenv import load_dotenv
import os
import tensorflow as tf

# Ensure TensorFlow uses the proper device and settings
tf.config.set_visible_devices([], 'GPU')  # Disable GPU if there are issues with CUDA

# Load environment variables from .env file
load_dotenv()

# Fetch historical data using yfinance
def fetch_historical_data():
    data = yf.download('^NSEBANK', start='2000-01-02', end=date.today().strftime('%Y-%m-%d'))
    data = data[['Adj Close']]
    data.columns = ['Price']
    return data

def plot_stock_price(data):
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Price'], color='blue', label='Bank Nifty Price')
    plt.title('Bank Nifty Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def get_technical_indicators(data):
    data['ma7'] = data['Price'].rolling(window=7).mean()
    data['ma21'] = data['Price'].rolling(window=21).mean()
    data['26ema'] = data['Price'].ewm(span=26).mean()
    data['12ema'] = data['Price'].ewm(span=12).mean()
    data['MACD'] = data['12ema'] - data['26ema']
    data['20sd'] = data['Price'].rolling(window=21).std()
    data['upper_band'] = data['ma21'] + (data['20sd']*2)
    data['lower_band'] = data['ma21'] - (data['20sd']*2)
    data['ema'] = data['Price'].ewm(com=0.5).mean()
    data['momentum'] = data['Price'].diff()
    data['momentum'].replace(0, np.nan, inplace=True)
    data['log_momentum'] = np.log(data['momentum'])
    data = data.dropna(subset=['log_momentum'])
    return data

def plot_technical_indicators(data, last_days):
    data = data[-last_days:]
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Price'], color='blue', label='Price')
    plt.plot(data.index, data['ma7'], color='red', label='MA 7 days')
    plt.plot(data.index, data['ma21'], color='green', label='MA 21 days')
    plt.plot(data.index, data['upper_band'], color='cyan', label='Upper Band')
    plt.plot(data.index, data['lower_band'], color='cyan', label='Lower Band')
    plt.fill_between(data.index, data['lower_band'], data['upper_band'], alpha=0.1)
    plt.title('Technical indicators for the last {} days'.format(last_days))
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def prepare_training_data(data):
    feature_columns = ['Price', 'ma7', 'ma21', '26ema', '12ema', 'MACD', '20sd', 'upper_band', 'lower_band', 'ema', 'momentum', 'log_momentum']
    data_features = data[feature_columns]
    data_target = data[['Price']]
    
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    scaled_features = feature_scaler.fit_transform(data_features)
    scaled_target = target_scaler.fit_transform(data_target)
    
    X_train, y_train = [], []
    for i in range(60, len(scaled_features)):
        X_train.append(scaled_features[i-60:i])
        y_train.append(scaled_target[i])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    return X_train, y_train, feature_scaler, target_scaler, feature_columns

def prepare_testing_data(data, feature_scaler, target_scaler, feature_columns):
    data_features = data[feature_columns]
    scaled_features = feature_scaler.transform(data_features)
    
    X_test, y_test = [], []
    for i in range(60, len(scaled_features)):
        X_test.append(scaled_features[i-60:i])
        y_test.append(scaled_features[i, 0])  # Assuming Price is the first column
    
    X_test, y_test = np.array(X_test), np.array(y_test)
    y_test = target_scaler.inverse_transform(y_test.reshape(-1, 1))
    return X_test, y_test

def build_lstm_model(input_shape):
    from keras.models import Sequential
    from keras.layers import LSTM, Dropout, Dense
    
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=60, activation='relu', return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=80, activation='relu', return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(units=120, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def plot_predictions(y_test, y_pred):
    plt.figure(figsize=(14, 7))
    plt.plot(y_test, color='red', label='Real Bank Nifty Price')
    plt.plot(y_pred, color='blue', label='Predicted Bank Nifty Price')
    plt.title('Bank Nifty Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def main():
    data = fetch_historical_data()
    plot_stock_price(data)
    
    print(f'There are {data.shape[0]} number of days in the dataset.')
    
    data_with_indicators = get_technical_indicators(data)
    data_with_indicators = data_with_indicators.dropna()
    
    plot_technical_indicators(data_with_indicators, 1000)
    
    X_train, y_train, feature_scaler, target_scaler, feature_columns = prepare_training_data(data_with_indicators)
    lstm_model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    lstm_model.summary()
    lstm_model.fit(X_train, y_train, epochs=50, batch_size=32)
    
    testing_data = data_with_indicators[-400:]  # Use the last 400 data points as testing data
    X_test, y_test = prepare_testing_data(testing_data, feature_scaler, target_scaler, feature_columns)
    y_pred = lstm_model.predict(X_test)
    y_pred = target_scaler.inverse_transform(y_pred)
    
    print("y_test (real prices):", y_test[:5])
    print("y_pred (predicted prices):", y_pred[:5])
    
    plot_predictions(y_test, y_pred)

if __name__ == "__main__":
    main()