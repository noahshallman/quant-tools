# Quant Tools by Noah Shallman

Welcome! This is a growing collection of Python tools I've built to explore financial markets, combining quantitative signals, fundamental data, and machine learning. These projects reflect my passion for hedge fund strategy, data science, and financial economics.

---
## Quantitative & Fundamental ML: Earnings Surprise Predictor

This project is an end-to-end machine learning system for predicting earnings surprises using historical stock data and quarterly earnings data.

### Data Acquisition:
- Loads historical equity data using `yfinance`
- Simulates quarterly earnings data (ActualEPS & ConsensusEPS) with a dummy function (replaceable with real data sources).

### Feature Engineering:
- Computes technical indicators such as daily returns, log returns, multiple moving averages (SMA & EMA), volatility, RSI, and MACD.
- Merges stock data with earnings data by quarter and calculates the earnings surprise metric.

### Model Pipeline and Training:
- Construcs a scikit-learn pipeline with `StandardScaler` and a `RandomForestClassifier`. 
- Optimizes hyperparameters using `GridSearchCV` to improve model performance based on ROC-AUC. 

### Evaluation and Visualization:
- Evaluates the model with a classification report and ROC-AUC score.
- Visualizes feature importance using Seaborn to understand the impact of each feature.

### Main Execution:
- Full pipeline runs in a single executable Python script
- Includes modular structure for easy customization and real-world dataset integration

### Repo Link:
[View Code](https://github.com/noahshallman/quant-tools/blob/main/earnings_surprise_2.py)

## Implied Volatility Surface Modeler

This Python script retrieves option data for a specified ticker, calculates the implied volatility of call options using the Black-Scholes model, and visualizes the resulting implied volatility surface in a 3D plot. 

### Option Data Retrieval:
- Uses the `yfinance` library to fetch current stock price and option chain data for a given ticker.
- Extracts underlying asset price and available expiration dates for options.

### Implied Volatility Calculation:
- Implements the Black-Scholes formula to compute the theoretical call option price
- Uses Brent's method (`brentq` from SciPy) to solve for the implied volatility that equates the Black-Scholes price with the observed market price
- Includes error handling with logging to return `NaN` if the calculation fails

### Data Processing:
- Iterates over option chains for multiple expiration dates.
- Calculates time to expiration in years and computes mid-prices (or falls back to the last price) for call options.
- Filters out any invalid computed implied volatility values.

### Visualization:
- Uses Matplotlib's 3D plotting toolkit to render an implied volatility surface.
- Displays strike prices, time to expiration, and implied volatility values on a color-coded 3D scatter plot.
### Command Line Interface:
- Utilizes `argparse` to allow the user to specify:
  - Ticker symbol (default: `SPY`)
  - Risk-free rate (default: `0.01`)
  - Verbosity flag for debugging and logging
-Configures logging based on the verbosity flag
### Main Execution:
- Integrates data fetching, computation, and visualization into a single executable script.
- Designed as a standalone tool that can be run from the command line.
### Repo Link:
[View Code](https://github.com/noahshallman/quant-tools/blob/main/implied_vol.py)

---
