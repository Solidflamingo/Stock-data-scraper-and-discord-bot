# Stock Prediction & Discord Bot
## General Overview
Welcome to my project! This repository contains three scripts that contain a basic yahoo finance stock scraper, a Discord bot that delivers desired stock prices, and a stock prediction model based on 5 years history of a single stock.

## Technologies Used
* Python
* Pycharm
* Discord API
* Machine learning libraries (sckit-learn, pandas, keras, numpy)
* Quandl API
* Windows Task Scheduler
* Yfinance


## 1. Get Stock Data (Get stock data.py)
### Purpose:
To fetch the latest stock data. The desired stock symbols are manually entered and are attained from a Quandl API.

### Usage:
python "Get stock data.py"

### Key Features:
* Fetches real-time stock data (close price, timestamp, close %change)
* Can be configured to fetch data for specific stock symbols or market segments.
* Saves the data to a local mySQL server (you can configure your own SQl database to save the data).


## 2. Discord Integration (Discord Integration.py)
### Purpose:
This script allows for integration with the Discord API. It's designed to facilitate communication between my Get Stock Data app and Discord.

### Usage:
python "Discord Integration.py"

### Key Features:
* Establishes a connection to specified Discord servers.
* Listens for specific !price (stock symbol) commands to trigger a response that generates the last three days of stock data saved in the mySQL table.


## 3. Training Stock Data for Predictions (Training stock data for predictions.py)
### Purpose:
This script provides an integration of a stock price forecasting model using LSTM (Long Short-Term Memory) and economic indicators like GDP. The forecasting is applied on stock data retrieved from Yahoo Finance.

### Usage:
The script initiates by loading the Quandl API key from the environment variables. This is essential for fetching the GDP data later in the script. You will need a Quandl account.
python "Training stock data for predictions.py"

### Key Features:
* Data is fetched for GDP from the FRED database using Quandl. The number of steps for forecasting is set at this stage.
* A custom estimator is defined utilizing LSTM for predicting stock prices. It inherits functionalities from scikit-learn's BaseEstimator and RegressorMixin.
* Stock data for a chosen symbol (e.g., TSLA) is retrieved using Yahoo Finance API through the yfinance package. The data is then scaled between 0 and 1, which is a common practice for LSTM models.
* The dataset is divided into training and testing sets, adhering to an 80-20 ratio.
* A function is defined to reshape the dataset, making it suitable for LSTM models. The data is reshaped based on the previous 20 days, and the next 10 days are set as the target.
* An LSTM model is trained using the training data. Early stopping is implemented to prevent overfitting.
* A loop tests different combinations of epochs and batch sizes to find optimal hyperparameters.
* The script plots the actual stock prices and the model's predictions on the test dataset.
* A sequence of operations predicts stock prices for a set number of future days, using the trained LSTM model. The last available GDP value is used as a constant input for this prediction.
* Final touches to the plot include title, labels, legend, and display.



### To use these scripts, ensure you have the required packages installed and have set up the necessary configurations, especially in the .env file for sensitive credentials.
