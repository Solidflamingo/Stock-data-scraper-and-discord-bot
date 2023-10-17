import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, RegressorMixin
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
import quandl
import os
from dotenv import load_dotenv


# -------------------------------------------
# ------------- Environment Variables -------
# -------------------------------------------

# Load Quandl API key from environment variables
load_dotenv()
quandl.ApiConfig.api_key = os.environ.get('QUANDL_API_KEY')

# -------------------------------------------
# ------------- Data Collection -------------
# -------------------------------------------

# Fetch GDP data from FRED using Quandl
gdp_data = quandl.get("FRED/GDP", start_date="2010-01-01", end_date="2022-12-31")
# Define the number of steps for forecasting
forecast_steps = 10


# -------------------------------------------
# ---- Custom Estimator for Stock Prediction
# -------------------------------------------

# This class utilizes LSTM to predict stock prices. Inherits from scikit-learn's BaseEstimator and RegressorMixin.
class StockLSTM(BaseEstimator, RegressorMixin):
    def __init__(self, input_shape, epochs=1, batch_size=32):
        self.input_shape = input_shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(LSTM(units=20, return_sequences=True, input_shape=self.input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(units=20, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=20))
        model.add(Dense(units=forecast_steps))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def fit(self, X, y, **kwargs):
        history_obj = self.model.fit(X, y, **kwargs)
        return history_obj

    def predict(self, X):
        return self.model.predict(X)


# -------------------------------------------
# ------------- Raw Data Preparation ------------
# -------------------------------------------

# Function to fetch stock data using Yahoo Finance API through the yfinance package
def fetch_stock_data(symbol, period="5y"):
    ticker = yf.Ticker(symbol)
    return ticker.history(period=period)


# Function to normalize data to values between 0 and 1 for LSTM
def prepare_data(df, feature='Close'):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = df.copy()
    df_scaled[feature] = scaler.fit_transform(df[feature].values.reshape(-1, 1))
    return df_scaled[[feature]], scaler

# Fetch stock data for a particular symbol (replace TSLA with desired stock)
symbol = 'TSLA'
df_stock = fetch_stock_data(symbol)
scaled_stock_data, stock_scaler = prepare_data(df_stock)
df_stock.index = df_stock.index.tz_localize(None)

# Align GDP data with stock data without merging
gdp_data_resampled = gdp_data.resample('B').ffill()
gdp_data_resampled.index = gdp_data_resampled.index.tz_localize(None)
aligned_gdp_data = gdp_data_resampled.reindex(df_stock.index).ffill().bfill()
scaled_gdp_data_aligned, gdp_scaler = prepare_data(aligned_gdp_data, feature='Value')


# -------------------------------------------
# ------------- Dataset Split ----------------
# -------------------------------------------

# Divide the dataset into training and testing sets, adhering to an 80-20 ratio
train_size = int(len(scaled_stock_data) * 0.8)
train_stock_data = scaled_stock_data[:train_size]
test_stock_data = scaled_stock_data[train_size - 20:]
train_gdp_data = scaled_gdp_data_aligned[:train_size]
test_gdp_data = scaled_gdp_data_aligned[train_size - 20:]


# -------------------------------------------
# --------- LSTM Data Preparation ------------
# -------------------------------------------

# Function to reshape the dataset suitable for LSTM model
def create_dataset(stock_data, economic_data):
    X, y = [], []
    stock_data_array = stock_data.values
    economic_data_array = economic_data.values
    for i in range(20, len(stock_data_array) - forecast_steps):
        stock_segment = stock_data_array[i - 20:i, 0]
        economic_segment = economic_data_array[i - 20:i, 0]
        combined_data = np.column_stack((stock_segment, economic_segment))
        X.append(combined_data)
        y.append(stock_data_array[i:i + forecast_steps, 0])
    return np.array(X), np.array(y)

# Prepare LTSM data
train_size = int(len(scaled_stock_data) * 0.8)
X_train, y_train = create_dataset(train_stock_data, train_gdp_data)
X_test, y_test = create_dataset(test_stock_data, test_gdp_data)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 2))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 2))


# -------------------------------------------
# ---------- Model Training -----------------
# -------------------------------------------

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
model = StockLSTM(input_shape=(X_train.shape[1], 2))
model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[early_stopping], epochs=10, batch_size=16)

# -------------------------------------------
# ------- Hyperparameter Tuning --------------
# -------------------------------------------

# Loop over different combinations of epochs and batch_size to find the optimal hyperparameters
best_loss = np.inf
best_epochs = None
best_batch_size = None
for epochs in [10, 20, 50]:
    for batch_size in [16, 32, 64]:
        model = StockLSTM(input_shape=(X_train.shape[1], 2))
        history_obj = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)
        val_loss = min(history_obj.history['val_loss'])
        if val_loss < best_loss:
            best_loss = val_loss
            best_epochs = epochs
            best_batch_size = batch_size

# Print the best hyperparameters for future reference
print(f"Best epochs: {best_epochs}, best batch size: {best_batch_size}")

# Get the dates corresponding to the trading days to make the graph more interpretable
actual_dates = df_stock.index[20:]

# Use the model to predict stock prices on the test set and scale them back to the original value
test_predictions = model.predict(X_test)
test_predictions_flattened = test_predictions.reshape(test_predictions.shape[0] * forecast_steps, 1)
test_predictions_inversed = stock_scaler.inverse_transform(test_predictions_flattened)

# Extracting last forecasted value from each sequence for plotting
test_predictions_last_values = [item[-1] for item in test_predictions]
test_predictions_last_values_inversed = stock_scaler.inverse_transform(np.array(test_predictions_last_values).reshape(-1, 1))

# -------------------------------------------
# ---------- Visualization -------------------
# -------------------------------------------

# Create a figure for plotting
plt.figure(figsize=(16, 8))

# Plot the actual stock prices
plt.plot(df_stock['Close'].index, df_stock['Close'].values, color='blue', label='Actual Stock Price')

# Plot the model's predictions on the test dataset
actual_test_dates_last_values = df_stock.index[train_size:train_size+len(test_predictions_last_values_inversed)]
plt.plot(actual_test_dates_last_values, test_predictions_last_values_inversed, color='green', label='Predicted Stock Price on Test Data (Last Forecast Value)')

# -------------------------------------------
# ----- Future Stock Price Prediction --------
# -------------------------------------------

# Initialize future prediction parameters
future_days = 200  # Approximate number of trading days in a year
last_60_days_stock = scaled_stock_data[-20:].values
last_60_days_gdp = scaled_gdp_data_aligned[-20:].values
future_predictions = []

# Run the model for future prediction
for i in range(future_days):
    future_input_stock = last_60_days_stock[-20:]
    future_input_gdp = last_60_days_gdp[-20:]
    future_input_combined = np.column_stack((future_input_stock, future_input_gdp))
    future_input = future_input_combined.reshape(1, 20, 2)
    next_pred = model.predict(future_input)

    # Update future prediction and input data
    future_predictions.append(next_pred[0][-1])  # Append the last forecasted value
    last_60_days_stock = np.append(last_60_days_stock, next_pred[0][-1])
    last_60_days_gdp = np.append(last_60_days_gdp, aligned_gdp_data.iloc[-1])  # Here, we are just taking the last available GDP value.

    # Maintain the data length
    if len(last_60_days_stock) > 20:
        last_60_days_stock = last_60_days_stock[1:]
        last_60_days_gdp = last_60_days_gdp[1:]

# Inverse scale the future predictions to get the actual stock prices
future_predictions = stock_scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Generate future dates
future_dates = pd.date_range(start=actual_dates[-1], periods=future_days + 1)[1:]

# Plot the future stock prices
plt.plot(future_dates, future_predictions, color='red', label='Predicted Future Stock Price')

# -------------------------------------------
# ---------- Final Plot Customization -------
# -------------------------------------------

# Add title and labels
plt.title(f'{symbol} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')

# Add legend
plt.legend()

# Show the plot
plt.show()







