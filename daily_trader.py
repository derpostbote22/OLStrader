import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from datetime import timedelta, date
import time
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import os
import requests
from dotenv import load_dotenv
import sys

# ==========================================
# 1. ENVIRONMENT SETUP & CONFIGURATION
# ==========================================

# Force the script to look for the .env file in a specific directory.
# This is especially useful for cloud hosting like PythonAnywhere where 
# the script might be executed from a different working directory.
project_folder = os.path.expanduser('/home/username')  # Change to your username/path
load_dotenv(os.path.join(project_folder, '.env'))

# Load API keys and webhooks from the environment variables
API_KEY = os.getenv('ALPACA_KEY')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
DISCORD_WEBHOOK = os.getenv('DISCORD_WEBHOOK')

# Set PAPER to True for simulated trading, False for live money trading
PAPER = True

# Global variable to aggregate status messages for the final Discord alert
discord_message = ""

# ==========================================
# 2. CORE FUNCTIONS
# ==========================================

def send_alert(message):
    """
    Sends a formatted alert message to a configured Discord channel via Webhook.
    """
    if not DISCORD_WEBHOOK:
        print("No Discord Webhook found. Check .env file.")
        return

    # Package the message into the JSON payload format expected by Discord
    payload = {"content": f"**Trading Bot Update:**\n{message}"}

    try:
        response = requests.post(DISCORD_WEBHOOK, json=payload)
        # Discord returns an HTTP 204 (No Content) status code upon a successful webhook post
        if response.status_code == 204: 
            print("Successfully sent Discord alert")
        else:
            print(f"Failed to send Discord alert. Status: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Connection failed: {e}")


def check_market_open():
    """
    Validates if the US stock market is currently open using Alpaca's Clock API.
    Prevents the script from attempting to trade during closed hours.
    """
    global discord_message
    trading_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)

    try:
        clock = trading_client.get_clock()

        if clock.is_open:
            print("Market is OPEN. Proceeding with analysis...")
            return True
        else:
            print("Market is CLOSED.")
            
            # Format the next market open time for user-friendly reporting
            next_open = clock.next_open.strftime("%Y-%m-%d %H:%M:%S")
            discord_message = (
                f"⛔ **Market is CLOSED.**\n"
                f"No trades will be executed.\n"
                f"Next market open: {next_open} UTC"
            )
            
            # Send alert and terminate script execution to prevent errors
            send_alert(discord_message)
            sys.exit()

    except Exception as e:
        # Failsafe: if the API request fails, halt execution to avoid bad trades
        error_msg = f"⚠️ Error fetching market status: {e}. Script stopping for safety."
        print(error_msg)
        send_alert(error_msg)
        sys.exit()


def get_current_shares():
    """
    Fetches the current holdings for Gold (IAU) and the S&P 500 (VOO) from Alpaca.
    """
    global discord_message
    trading_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)

    # Default quantities to 0.0 in case we don't currently hold the assets
    curr_no_gd = 0.0
    curr_no_sp = 0.0

    try:
        positions = trading_client.get_all_positions()

        # Iterate through our portfolio to find specific asset quantities
        for p in positions:
            if p.symbol == 'IAU':
                curr_no_gd = float(p.qty)
            elif p.symbol == 'VOO':
                curr_no_sp = float(p.qty)

        print(f"Current VOO Shares (SP): {curr_no_sp}")
        print(f"Current IAU Shares (GD): {curr_no_gd}")

        discord_message += f"Current VOO Shares (SP): {curr_no_sp}, "
        discord_message += f"Current IAU Shares (GD): {curr_no_gd}"

        return curr_no_gd, curr_no_sp

    except Exception as e:
        discord_message += f"Error fetching positions: {e}"
        print(f"Error fetching positions: {e}")
        return 0.0, 0.0


def det_trans_inputs(curr_no_gd, curr_no_sp):
    """
    Downloads historical market data, runs a linear regression on log S&P 500 prices, 
    calculates prediction errors, and determines the optimal rebalancing trades based on prediction errors.
    """
    global discord_message
    
    # ---------------------------------------------------------
    # 0. Meta parameters
    # ---------------------------------------------------------
    sf = 2 # sf (scale factor): Adjusts sensitivity of our target reallocation
    no_days_back = 500 # Lookback period for historical data

    # ---------------------------------------------------------
    # 1. Fetch Historical Data from Yahoo Finance
    # ---------------------------------------------------------
    startdate = date.today() - timedelta(days=no_days_back)
    string_startdate = startdate.strftime("%Y-%m-%d")
    sp500 = yf.download(tickers="VOO", start=string_startdate)
    gold_data = yf.download(tickers='IAU', start=string_startdate)

    # ---------------------------------------------------------
    # 2. Data Alignment & Preprocessing
    # ---------------------------------------------------------
    # Ensure both dataframes have the exact same trading days (handles holidays)
    new_gd = gold_data[gold_data.index.isin(sp500.index)].copy()
    new_sp = sp500[sp500.index.isin(new_gd.index)].copy()
    
    # Calculate the natural log of the S&P 500 Open price to linearize exponential growth
    new_sp['logopen'] = np.log(new_sp['Open'])
    new_sp['Numbering'] = np.arange(len(new_sp.index)) # Create a numeric time index (x-axis)
    
    x = new_sp['Numbering'].to_list()
    y = new_sp['logopen'].to_list()
    dfi = pd.DataFrame({'y': y, 'x1': x})

    # ---------------------------------------------------------
    # 3. Model Training & Prediction
    # ---------------------------------------------------------
    # Fit a Generalized Linear Model (Gaussian) on the log prices over time
    regression = smf.glm('y ~ x1', data=dfi, family=sm.families.Gaussian(sm.families.links.identity())).fit()
    
    # Ensure the model slope isn't negative (we assume long-term market growth). 
    # If it is, constrain the slope to 0.
    if regression.params.x1 < 0:
        constraints = 'x1 = 0'
        regression = smf.glm('y ~ x1', data=dfi, family=sm.families.Gaussian(sm.families.links.identity())).fit_constrained(constraints)

    # Store predicted log prices
    new_sp['pospred_logopen'] = regression.predict()

    # ---------------------------------------------------------
    # 4. Error Analysis & Rebalance Calculation
    # ---------------------------------------------------------
    new_sp['logpos_error'] = new_sp['logopen'] - new_sp['pospred_logopen']
    new_sp['pospred_open'] = np.exp(new_sp['pospred_logopen']) # Convert log predictions back to normal prices
    new_sp['pospred_error'] = new_sp['Open']['VOO'] - new_sp['pospred_open']
    
    # Calculate percentage error relative to the predicted price
    new_sp['ppperc_error'] = new_sp['pospred_error'] / new_sp['pospred_open']

    # Calculate the error ratio using our recent price and the scale factor
    latest_voo_open = new_sp['Open']['VOO'].iloc[-1]
    latest_pred_open = new_sp['pospred_open'].iloc[-1]
    error_rnpp = (latest_voo_open - latest_pred_open) / (latest_voo_open * sf)
    
    # Determine what percentile our current error sits at historically
    percentile_current_error_pp = (np.array(new_sp['ppperc_error']) > error_rnpp).sum() / len(np.array(new_sp['ppperc_error']))

    # ---------------------------------------------------------
    # 5. Determine Trade Quantities
    # ---------------------------------------------------------
    # Calculate total portfolio value across both assets
    curr_inv_sp = curr_no_sp * latest_voo_open
    curr_inv_gd = curr_no_gd * new_gd['Open']['IAU'].iloc[-1]
    
    # Target investment in VOO based on our error percentile
    aim_inv_sp = percentile_current_error_pp * (curr_inv_sp + curr_inv_gd)
    
    # Calculate how much cash needs to be moved
    new_inv_withdraw_pp = aim_inv_sp - curr_inv_sp

    # Determine buy/sell direction based on the cash difference
    if new_inv_withdraw_pp < 0:
        sell_qty = new_inv_withdraw_pp * (-1) / latest_voo_open
        sell_stock = 'VOO'
        buy_stock = 'IAU'
        discord_message += f"\n Buying IAU while selling {sell_qty:.4f} shares of VOO."
    else:
        sell_qty = new_inv_withdraw_pp / new_gd['Open']['IAU'].iloc[-1]
        sell_stock = 'IAU'
        buy_stock = 'VOO'
        discord_message += f"\n Buying VOO while selling {sell_qty:.4f} shares of IAU."
        
    return sell_stock, sell_qty, buy_stock


def execute_modern_reinvestment(stock_to_sell, qty_to_sell, stock_to_buy):
    """
    Executes the rebalancing trade by first selling the designated stock, 
    calculating the net proceeds, and using that cash to buy the other stock.
    """
    global discord_message

    trading_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)
    print("Connected to Alpaca (alpaca-py).")
    discord_message += "\n Connected to Alpaca (alpaca-py)."

    # ---------------------------------------------------------
    # STEP 1: SELL ASSET
    # ---------------------------------------------------------
    print(f"Placing Market Sell Order for {qty_to_sell} shares of {stock_to_sell}...")

    sell_request = MarketOrderRequest(
        symbol=stock_to_sell,
        qty=qty_to_sell,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY
    )

    try:
        sell_order = trading_client.submit_order(order_data=sell_request)
        sell_order_id = str(sell_order.id)

        print("Waiting for sell execution...")
        start_time = time.time()
        timeout_seconds = 5 * 60  # 5 minutes maximum wait time

        while True:
            # 1. Timeout Check
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                print("Order timed out (5 mins). Canceling order...")
                try:
                    trading_client.cancel_order_by_id(sell_order_id)
                    cancel_msg = "Order canceled successfully."
                except Exception as e:
                    cancel_msg = f"Failed to cancel (might have filled): {e}"

                discord_message += f"\n ⚠️ Sell order timed out after 5 minutes. {cancel_msg}"
                return

            # 2. Fetch latest order status
            try:
                updated_order = trading_client.get_order_by_id(sell_order_id)
            except Exception as e:
                print(f"Error fetching order status: {e}")
                time.sleep(1)
                continue

            # 3. Evaluate Order State
            if updated_order.status == 'filled':
                break
            elif updated_order.status in ['canceled', 'expired', 'rejected']:
                print(f"Order failed. Status: {updated_order.status}")
                discord_message += f"\n Order failed. Status: {updated_order.status}"
                return

            # Prevent spamming the API limit
            time.sleep(1)

    except Exception as e:
        print(f"Error placing sell order: {e}")
        discord_message += f"\n Error placing sell order: {e}"
        return

    # ---------------------------------------------------------
    # STEP 2: CALCULATE CASH GENERATED
    # ---------------------------------------------------------
    # Alpaca returns numerics as strings to preserve precision; cast to float
    fill_price = float(updated_order.filled_avg_price)
    filled_qty = float(updated_order.filled_qty)
    gross_proceeds = fill_price * filled_qty

    net_cash = round(gross_proceeds, 2)

    print(f"Sold {stock_to_sell}. Avg Price: ${fill_price:.2f}")
    discord_message += f"\n Sold {stock_to_sell}. Avg Price: ${fill_price:.2f}"
    print(f"Net Cash Available for Reinvestment: ${net_cash:.2f}")
    discord_message += f"\n Net Cash Available for Reinvestment: ${net_cash:.2f}"

    # Alpaca requires a minimum of $1.00 for notional fractional orders
    if net_cash < 1.00:
        print("Not enough funds generated ($1.00 minimum required for fractional). Stopping.")
        discord_message += "\n Not enough funds generated ($1.00 minimum required for fractional). Stopping."
        return

    # ---------------------------------------------------------
    # STEP 3: BUY ASSET (FRACTIONAL/NOTIONAL)
    # ---------------------------------------------------------
    print(f"Placing Fractional Market Buy Order for ${net_cash} of {stock_to_buy}...")

    # 'notional' allows us to specify exactly how much dollar value we want to buy
    buy_request = MarketOrderRequest(
        symbol=stock_to_buy,
        notional=net_cash, 
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY
    )

    try:
        buy_order = trading_client.submit_order(order_data=buy_request)
        buy_order_id = str(buy_order.id)
        
        # Wait for buy order confirmation
        while True:
            updated_buy = trading_client.get_order_by_id(buy_order_id)
            if updated_buy.status == 'filled':
                fill_amt = float(updated_buy.filled_qty)
                fill_avg = float(updated_buy.filled_avg_price)
                print(f"Buy Order Status: {updated_buy.status}")
                discord_message += f"\n Buy Order Status: {updated_buy.status}"
                print(f"Successfully bought {fill_amt} shares of {stock_to_buy} at ${fill_avg:.2f}")
                discord_message += f"\n Successfully bought {fill_amt} shares of {stock_to_buy} at ${fill_avg:.2f}"
                break
            elif updated_buy.status in ['cancelled', 'rejected']:
                print(f"Buy order rejected. Reason: {updated_buy.status}")
                discord_message += f"\n Buy order rejected. Reason: {updated_buy.status}"
                break
            
            time.sleep(1)

    except Exception as e:
        print(f"Error placing buy order: {e}")
        discord_message += f"\n Error placing buy order: {e}"

# ==========================================
# 3. MAIN EXECUTION PIPELINE
# ==========================================

if __name__ == "__main__":
    check_market_open()
    cur_gd, cur_sp = get_current_shares()
    trans_inputs = det_trans_inputs(cur_gd, cur_sp)
    execute_modern_reinvestment(trans_inputs[0], trans_inputs[1], trans_inputs[2])
    send_alert(discord_message)
