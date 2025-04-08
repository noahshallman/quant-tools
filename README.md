# Quant Tools by Noah Shallman

Welcome! This is a growing collection of Python tools I've built to explore financial markets, combining quantitative signals, fundamental data, and machine learning. These projects reflect my passion for hedge fund strategy, data science, and financial economics.

---

## Implied Volatility Surface Modeler

This Python script retrieves option data for a specified ticker, calculates the implied volatility of call options using the Black-Scholes model, and visualizes the resulting implied volatility surface in a 3D plot.

### Tools:
- Python, `yfinance`, SciPy, `matplotlib` (3D)

### Core Logic:
- Implements Brent’s method (`brentq` from SciPy) to solve for implied volatility from market prices
- Uses Black-Scholes pricing to calculate theoretical call prices and compares with market-observed prices

### Output:
- 3D surface plots showing strike price, time to expiration (in years), and implied volatility
- Includes filtering and error handling for invalid option quotes

### CLI Integration:
- Command-line interface with `argparse` support
  - Ticker symbol (default: `SPY`)
  - Risk-free rate (default: `0.01`)
  - Verbosity flag for debugging and logging

### Repo Link:
[View Code](https://github.com/noahshallman/quant-tools/blob/main/implied_vol.py)

---

## Quantamental ML: Earnings Surprise Predictor

This project is an end-to-end machine learning system for predicting earnings surprises using both technical market indicators and company-level fundamental metrics.

### Data Acquisition:
- Loads historical equity data via `yfinance`
- Simulates quarterly earnings data (ActualEPS & ConsensusEPS) using a placeholder function (can be replaced with real data sources)

### Feature Engineering:
- Calculates daily returns, log returns
- Technical indicators: SMA, EMA, RSI, MACD, and rolling volatility
- Merges quarterly stock data with earnings data and calculates the earnings surprise metric

### Model Pipeline:
- Scikit-learn pipeline with `StandardScaler` and `RandomForestClassifier`
- Hyperparameter tuning using `GridSearchCV` with ROC-AUC scoring

### Evaluation:
- ROC-AUC score and classification report for model performance
- Feature importance plotted with Seaborn

### Integration:
- Full pipeline runs in a single executable Python script
- Includes modular structure for easy customization and real-world dataset integration

### Repo Link:
[View Code](https://github.com/noahshallman/quant-tools/blob/main/earnings_suprise_2.py)

---

## 🧑‍💻 About Me

I'm Noah Shallman, a high school student passionate about financial economics, machine learning, and quantitative investing. I lead teams in the Wharton Investment & Data Science Competitions, build tools to model market behavior, and share my insights on:

- 📺 [YouTube – Noah Knows Markets](http://www.youtube.com/@noahknowsmarkets)
- 📰 [Substack – Noah Knows Markets](https://noahshallman.substack.com)


