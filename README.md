# Financial Data Analysis and LSTM Modeling

This script explores critical tools and techniques for effective financial data analysis using Python. From importing essential libraries to developing and visualizing LSTM models, it covers the fundamentals of handling, analyzing, and predicting financial data trends.

Author: Nickolas Discolll  
Date: May 8, 2024

## Overview

This script provides a comprehensive guide to financial data analysis and predictive modeling using Python. It includes the following steps:
1. Importing historical stock data.
2. Plotting stock prices.
3. Calculating technical indicators.
4. Plotting technical indicators.
5. Creating a heatmap of correlations.
6. Preparing training data for LSTM model.
7. Building and compiling the LSTM model.
8. Preparing testing data.
9. Training the LSTM model.
10. Making predictions and plotting results.

## Functions

### `fetch_historical_data()`
Fetches historical stock prices for the Nifty Bank index from Yahoo Finance.

**Returns:**
- `pd.DataFrame`: A dataframe containing the adjusted closing prices.

### `plot_stock_price(data)`
Plots the Bank Nifty price against the date.

**Parameters:**
- `data` (`pd.DataFrame`): A dataframe containing stock prices.

### `get_technical_indicators(dataset)`
Calculates various technical indicators for the dataset.

**Parameters:**
- `dataset` (`pd.DataFrame`): A dataframe containing stock prices.

**Returns:**
- `pd.DataFrame`: The dataframe with technical indicators added.

### `plot_technical_indicators(dataset, last_days)`
Plots technical indicators for the dataset.

**Parameters:**
- `dataset` (`pd.DataFrame`): A dataframe containing stock prices and technical indicators.
- `last_days` (`int`): The number of days to plot.

### `create_heatmap(df)`
Creates a heatmap of the correlation matrix of the dataframe.

**Parameters:**
- `df` (`pd.DataFrame`): A dataframe containing stock prices and technical indicators.

### `prepare_training_data(data, training_size=0.8)`
Scales and prepares training data for the LSTM model.

**Parameters:**
- `data` (`pd.DataFrame`): A dataframe containing stock prices and technical indicators.
- `training_size` (`float`): The proportion of the dataset to be used for training.

**Returns:**
- `np.ndarray`: Scaled training data inputs and outputs.
- `MinMaxScaler`: Scaler used for normalizing the data.
- `pd.DataFrame`: Testing data.

### `build_lstm_model(input_shape)`
Builds and compiles an LSTM model.

**Parameters:**
- `input_shape` (`tuple`): The shape of the input data.

**Returns:**
- `keras.models.Sequential`: The compiled LSTM model.

### `prepare_testing_data(inputs, scaler)`
Prepares testing data for the LSTM model.

**Parameters:**
- `inputs` (`pd.DataFrame`): The dataframe containing stock prices and technical indicators.
- `scaler` (`sklearn.preprocessing.MinMaxScaler`): The scaler used for training data.

**Returns:**
- `np.ndarray`: Scaled testing data inputs and outputs.

### `plot_predictions(y_test, y_pred)`
Plots real vs predicted stock prices.

**Parameters:**
- `y_test` (`np.ndarray`): Real stock prices.
- `y_pred` (`np.ndarray`): Predicted stock prices.

### `main()`
Main function to run the financial data analysis and LSTM modeling.

## How to Use

1. **Install Required Libraries**
    Ensure you have the required libraries installed:
    ```sh
    pip install numpy pandas matplotlib seaborn pandas_datareader scikit-learn keras
    ```

2. **Run the Script**
    Execute the script:
    ```sh
    python financial_data_analysis.py
    ```

3. **Understanding the Workflow**
    - The script starts by fetching historical stock data.
    - It then plots the stock prices to visualize the data.
    - Technical indicators are calculated and added to the dataframe.
    - These indicators are plotted to provide further insights.
    - A heatmap of the correlation matrix is created.
    - Training data is prepared and scaled.
    - An LSTM model is built and compiled.
    - The model is trained using the prepared data.
    - Predictions are made on the test data.
    - Finally, the real vs predicted stock prices are plotted for comparison.

By following these steps, you can perform financial data analysis and predictive modeling using LSTM on the Nifty Bank index.
