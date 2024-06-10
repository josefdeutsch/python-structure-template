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
    plt.figure(figsize=(28, 12))
    plt.plot(data.index, data['Price'], label='Bank Nifty Price')
    plt.xlabel('Date')
    plt.ylabel('Rs')
    plt.title('Bank Nifty Price')
    plt.legend()
    plt.show()

def get_technical_indicators(dataset):
    dataset['ma7'] = dataset['Price'].rolling(window=7).mean()
    dataset['ma21'] = dataset['Price'].rolling(window=21).mean()
    dataset['26ema'] = dataset['Price'].ewm(span=26).mean()
    dataset['12ema'] = dataset['Price'].ewm(span=12).mean()
    dataset['MACD'] = dataset['12ema'] - dataset['26ema']
    dataset['20sd'] = dataset['Price'].rolling(window=21).std()
    dataset['upper_band'] = dataset['ma21'] + (dataset['20sd'] * 2)
    dataset['lower_band'] = dataset['ma21'] - (dataset['20sd'] * 2)
    dataset['ema'] = dataset['Price'].ewm(com=0.5).mean()
    dataset['momentum'] = dataset['Price'].diff()
    dataset['momentum'].replace(0, np.nan, inplace=True)  # Replace zero with NaN
    dataset['log_momentum'] = np.log(dataset['momentum'])
    dataset = dataset.dropna(subset=['log_momentum'])  # Drop rows with NaN in log_momentum
    return dataset

def plot_technical_indicators(dataset, last_days):
    shape_0 = dataset.shape[0]
    dataset = dataset.iloc[-last_days:, :]
    x_ = list(dataset.index)
    
    plt.figure(figsize=(30, 20))
    
    plt.subplot(2, 1, 1)
    plt.plot(dataset['ma7'], label='MA 7', color='g', linestyle='--')
    plt.plot(dataset['Price'], label='Closing Price', color='b')
    plt.plot(dataset['ma21'], label='MA 21', color='r', linestyle='--')
    plt.plot(dataset['upper_band'], label='Upper Band', color='c')
    plt.plot(dataset['lower_band'], label='Lower Band', color='c')
    plt.fill_between(x_, dataset['lower_band'], dataset['upper_band'], alpha=0.35)
    plt.title(f'Technical indicators for Bank Nifty - last {last_days} days.')
    plt.ylabel('Rs')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.title('MACD')
    plt.plot(dataset['MACD'], label='MACD', linestyle='-.')
    plt.plot(dataset['log_momentum'], label='Momentum', color='b', linestyle='-')
    plt.legend()
    
    plt.show()

from sklearn.preprocessing import MinMaxScaler
import numpy as np

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




def build_lstm_model(input_shape):
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
    lstm_model.fit(X_train, y_train, epochs=10, batch_size=32)
    
    testing_data = data_with_indicators[-400:]  # Use the last 400 data points as testing data
    X_test, y_test = prepare_testing_data(testing_data, feature_scaler, target_scaler, feature_columns)
    y_pred = lstm_model.predict(X_test)
    y_pred = target_scaler.inverse_transform(y_pred)
    
    print("y_test (real prices):", y_test[:5])
    print("y_pred (predicted prices):", y_pred[:5])
    
    plot_predictions(y_test, y_pred)

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()






