# OLStrader
Automating trading in a portfolio with Gold (IAU) and the S&amp;P500 index (VOO). Daily rebalancing based on whether the S&amp;P 500 is below (buy VOO, sell IAU) or above (sell VOO, buy IAU) its long-term trend.

## What is this

The code in this repository applies an ordinary least-squares (OLS) regression to the last 500 trading days of S&P 500 data to identify its (hypothesised) long-term trend. Based on this prediction, the algorithm determines whether the S&P 500 *currently* lies above or below its (hypothesised) long-term trend. If it is below, the algorithm sells gold (IAU) to buy S&P 500 (VOO). If it is above, it buys gold and sells S&P 500. The magnitude of the transfer depends on the relative difference between the long-term trend and the current S&P 500 value. This is a low-frequency trading bot, that should not be used to execute trades more than once a day. It relies on [Alpaca's trading API](https://alpaca.markets/algotrading) for market access, [Yahoo's Finance API](https://github.com/ranaroussi/yfinance) for market data, and [Discord's webhook](https://support.discord.com/hc/en-us/articles/228383668-Intro-to-Webhooks) for alerts.

## How to use it

The best way to use this script is by making a [PythonAnywhere](https://www.pythonanywhere.com/) account (free), and letting the script execute at your desired frequency. The time at which you choose to execute the script should be chosen to be within NYSE's trading hours (9:30 AM – 4:00 PM Eastern time). Make sure to replace the folder name (username) in the script and create a .env file with your credentials for Alpaca and Discord. If you want to switch from paper trading to real trading, just set (PAPER = False) and make sure your Alpaca account is funded.

## Disclaimer

Not Financial Advice.
This software is for educational purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. The author assumes no responsibility for your trading results. Always test algorithms extensively in a Paper Trading environment before going live.
