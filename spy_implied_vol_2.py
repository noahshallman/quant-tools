#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 10:15:22 2025

@author: noahshallman
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Option Implied Volatility Surface Calculator

This script retrieves option data for a specified ticker, calculates the implied volatility
of call options using the Black-Scholes model, and plots the implied volatility surface.
"""

import argparse
import logging
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from scipy.stats import norm
from scipy.optimize import brentq


def black_scholes_call(S, K, T, r, sigma):
    """
    Compute the Black-Scholes price for a European call option.
    
    Parameters:
        S (float): Underlying asset price.
        K (float): Strike price.
        T (float): Time to expiration in years.
        r (float): Risk-free interest rate.
        sigma (float): Volatility.
            
    Returns:
        float: Theoretical call option price.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_volatility_call(C_market, S, K, T, r):
    """
    Compute the implied volatility of a call option using the Black-Scholes model.
    
    Parameters:
        C_market (float): Observed market price of the call.
        S (float): Underlying asset price.
        K (float): Strike price.
        T (float): Time to expiration in years.
        r (float): Risk-free rate.
            
    Returns:
        float: Implied volatility or np.nan if no solution is found.
    """
    def objective(sigma):
        return black_scholes_call(S, K, T, r, sigma) - C_market
    
    try:
        implied_vol = brentq(objective, 1e-6, 5.0)
        return implied_vol
    except (ValueError, RuntimeError) as e:
        logging.debug("Brent solver failed for S=%s, K=%s, T=%s, r=%s, C_market=%s: %s",
                      S, K, T, r, C_market, e)
        return np.nan


def fetch_option_data(ticker):
    """
    Fetch historical and options data for the specified ticker.
    
    Parameters:
        ticker (str): Ticker symbol.
        
    Returns:
        tuple: (Underlying asset price, list of expiration dates, yfinance Ticker object)
    """
    tkr = yf.Ticker(ticker)
    hist = tkr.history(period="1d")
    if hist.empty:
        logging.error("No historical data found for %s", ticker)
        return None, None, None
    S = hist['Close'].iloc[-1]
    exp_dates = tkr.options
    return S, exp_dates, tkr


def compute_iv_data(S, exp_dates, tkr, r):
    """
    Compute the implied volatilities for call options.
    
    Parameters:
        S (float): Underlying asset price.
        exp_dates (list): List of expiration date strings.
        tkr (yfinance.Ticker): Ticker object.
        r (float): Risk-free rate.
        
    Returns:
        tuple: Arrays of strike prices, times to expiration, and implied volatilities.
    """
    strikes_list = []
    T_list = []
    iv_list = []
    current_date = datetime.today()
    
    for exp in exp_dates:
        try:
            expiration = datetime.strptime(exp, '%Y-%m-%d')
        except Exception as e:
            logging.warning("Error parsing expiration date %s: %s", exp, e)
            continue
        T = (expiration - current_date).days / 365.0
        if T <= 0:
            continue  # Skip expired options
        
        try:
            opt_chain = tkr.option_chain(exp)
        except Exception as e:
            logging.warning("Error retrieving option chain for expiration %s: %s", exp, e)
            continue
        
        calls = opt_chain.calls
        for _, row in calls.iterrows():
            K = row['strike']
            # Use the mid-price if both bid and ask are available; otherwise use last price.
            price = (row['bid'] + row['ask']) / 2 if (row['bid'] > 0 and row['ask'] > 0) else row['lastPrice']
            if price <= 0:
                continue
            iv = implied_volatility_call(price, S, K, T, r)
            strikes_list.append(K)
            T_list.append(T)
            iv_list.append(iv)
    
    strikes = np.array(strikes_list)
    T_values = np.array(T_list)
    iv_values = np.array(iv_list)
    
    # Filter out entries where implied volatility calculation failed
    valid = ~np.isnan(iv_values)
    return strikes[valid], T_values[valid], iv_values[valid]


def plot_iv_surface(strikes, T_values, iv_values, ticker):
    """
    Plot the implied volatility surface.
    
    Parameters:
        strikes (np.array): Strike prices.
        T_values (np.array): Times to expiration in years.
        iv_values (np.array): Implied volatilities.
        ticker (str): Ticker symbol.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(strikes, T_values, iv_values, c=iv_values, cmap='viridis')
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Time to Expiration (years)')
    ax.set_zlabel('Implied Volatility')
    ax.set_title(f'Implied Volatility Surface for {ticker}')
    fig.colorbar(scatter, ax=ax, label='Implied Volatility')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Implied Volatility Surface Calculator")
    parser.add_argument("-t", "--ticker", type=str, default="SPY",
                        help="Ticker symbol (default: 'SPY')
    parser.add_argument("-r", "--rate", type=float, default=0.01,
                        help="Risk-free interest rate (default: 0.01)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Increase output verbosity")
    args = parser.parse_args()
    
    # Configure logging level based on verbosity
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    
    ticker = args.ticker.upper()
    r = args.rate
    
    logging.info("Fetching option data for ticker: %s", ticker)
    S, exp_dates, tkr = fetch_option_data(ticker)
    if S is None:
        return
    logging.info("Current underlying price: %.2f", S)
    
    logging.info("Computing implied volatilities...")
    strikes, T_values, iv_values = compute_iv_data(S, exp_dates, tkr, r)
    
    if len(iv_values) == 0:
        logging.error("No valid implied volatility data was computed.")
        return
    
    logging.info("Plotting implied volatility surface...")
    plot_iv_surface(strikes, T_values, iv_values, ticker)


if __name__ == "__main__":
    main()
