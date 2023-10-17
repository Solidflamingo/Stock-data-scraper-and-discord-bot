import discord
from discord.ext import commands
import pymysql
import configparser
import os
from dotenv import load_dotenv

# Read configuration from ini file for database settings
config = configparser.ConfigParser()
config.read('config.ini')

# Fetch database credentials from configuration and environment variables
host = config['database']['host']
database = config['database']['database']
user = config['database']['user']
password = os.environ.get('MYSQL_PASSWORD')

# Initialize the bot with required intents
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)


# Function to establish a connection to the MySQL database
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


# Function to format date for better readability
def format_date(timestamp):
    formatted_date = timestamp.strftime("%B %d")
    day = timestamp.day
    suffix = "th" if 4 <= day <= 20 or 24 <= day <= 30 else ["st", "nd", "rd"][day % 10 - 1]
    return f"{formatted_date}{suffix}"


# Event to show that the bot is active
@bot.event
async def on_ready():
    print(f"We have logged in as {bot.user}")


# Command to fetch and display the latest stock prices from MYSQL database
@bot.command()
async def price(ctx, symbol):
    try:
        cursor = create_connection().cursor()

        # Check if the stock symbol exists in the database
        cursor.execute(f"SELECT symbol FROM stocks WHERE symbol = '{symbol}'")
        result = cursor.fetchone()
        if not result:
            await ctx.send(f"Symbol {symbol} not found in the database.")
            return

        # Fetch stock prices for the past week for the specified symbol
        cursor.execute(
            f"SELECT timestamp, FORMAT(price, 2) AS price FROM stocks WHERE symbol = '{symbol}' AND timestamp >= NOW() - INTERVAL 7 DAY ORDER BY timestamp DESC;")
        rows = cursor.fetchall()
        if not rows:
            await ctx.send(f"No price data found for {symbol} in the last week.")
            return

        # Format the data for displaying in Discord
        formatted_data = [
            "```",
            "Date          |   Price($)",
            "---------------------------"
        ]
        for row in rows:
            formatted_date = format_date(row[0])
            formatted_data.append(f"{formatted_date}  |  ${row[1]}")
        formatted_data.append("```")

        # Calculate and display the percentage change over the past week
        first_price = float(rows[0][1])
        last_price = float(rows[-1][1])
        percentage_change = ((first_price - last_price) / first_price) * 100
        formatted_data.append(f"Weekly Percentage Change: {percentage_change:.2f}%")

        message = '\n'.join(formatted_data)
        await ctx.send(message)
    except Exception as e:
        print(f"An error occurred: {e}")
        await ctx.send("An error occurred while fetching price data.")

# load and initiate the discord bot token via an env file
load_dotenv()
bot_token = os.getenv('DISCORD_BOT_TOKEN')
bot.run(bot_token)


