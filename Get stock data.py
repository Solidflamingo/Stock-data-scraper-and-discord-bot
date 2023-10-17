import yfinance as yf
import pymysql
import configparser
from datetime import datetime
import os

#   Configuration and Environment Variables
# Read configuration settings from an external .ini file
config = configparser.ConfigParser()
config.read('config.ini')

# Fetch database credentials from configuration and environment variables
host = config['database']['host']
database = config['database']['database']
user = config['database']['user']
password = os.environ.get('MYSQL_PASSWORD')


#   Functions
# Fetch stock data for a given symbol over the past 7 days using yfinance
def fetch_7day_stock_data(symbol):
    stock = yf.Ticker(symbol)
    sevendays_data = stock.history(period='7d')
    return sevendays_data if not sevendays_data.empty else None


# Calculate the 7-day loss for a given stock
def calculate_7day_loss(stock_data):
    if stock_data.empty:
        return None
    start_price = stock_data['Close'].iloc[0]
    end_price = stock_data['Close'].iloc[-1]
    return ((end_price - start_price) / start_price) * 100


# Establish a connection to the MySQL database
def create_connection():
    try:
        connection = pymysql.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )
        return connection
    except pymysql.MySQLError as e:
        print(f"Error: {e}")
        return None


# Insert fetched stock data into the MySQL database
def insert_stock_data(connection, symbol, price, percentage_change):
    try:
        cursor = connection.cursor()
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        query = """
        INSERT INTO stocks (symbol, price, percentage_change, timestamp)
        VALUES (%s, %s, %s, %s)
        """
        cursor.execute(query, (symbol, price, percentage_change, current_time))
        connection.commit()
    except pymysql.MySQLError as e:
        print(f"Error: {e}")


#    Main Functionality
# The main function fetches stock data for a list of stock symbols,
# calculates their 7-day losses, and inserts the top 20 losing stocks
# into the MySQL database. For the purpose of this script, I have only included stocks
# I am interested in.
def main():
    stock_symbols = ["AAPL", "GOOGL", "AMZN", "MSFT", "TSLA", "BHP.AX", "RIO.AX", "ANZ.AX", "NAB.AX",
                     "XRO.AX", "MBLPC.AX", "LNW.AX", "REA.AX", "AMPPB.AX", "SEK.AX", "PMV.AX",
                     "PLTR", "AMD", "NVDA", "SNAP", "UBER", "RIVN", "INTC"]
    losses = {}

    for symbol in stock_symbols:
        stock_data = fetch_7day_stock_data(symbol)
        if stock_data is not None:
            loss = calculate_7day_loss(stock_data)
            if loss is not None:
                losses[symbol] = loss

    top_20_losing_stocks = sorted(losses, key=losses.get, reverse=True)[:20]

    conn = create_connection()
    if conn is None:
        return

    for symbol in top_20_losing_stocks:
        stock_data = fetch_7day_stock_data(symbol)
        if stock_data is not None:
            last_close_price = stock_data['Close'].iloc[-1]
            percentage_change = losses[symbol]
            insert_stock_data(conn, symbol, last_close_price, percentage_change)

    conn.close()


# Run the app
if __name__ == "__main__":
    main()
