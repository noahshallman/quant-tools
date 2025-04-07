#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 23:12:12 2025

@author: noahshallman
"""

# earnings_surprise_predictor.py
"""
Earnings Surprise Prediction System using scikit-learn

Loads historical stock data and (dummy) earnings data, computes technical indicators, 
merges quarterly earnings to calculate earnings surprise, and builds a model to predict if the 
earnings surprise is positive (binary classification). All code is contained in one file (for Spyder).
"""

#%% Imports
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

#%% Data Loading Functions
def load_stock_data(ticker: str, 
                   start_date: str = '2020-01-01',
                   end_date: str = None) -> pd.DataFrame:
    """
    Load stock data using yfinance.
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str, optional
        End date in YYYY-MM-DD format.
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing stock data.
    """
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def load_earnings_data(ticker: str) -> pd.DataFrame:
    """
    Load earnings data for the ticker.
    
    For demonstration, this function creates a dummy quarterly earnings DataFrame 
    with 'ActualEPS' and 'ConsensusEPS'. In a real implementation, this could be replaced 
    by scraping or calling an API.
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing earnings data with columns: Date, ActualEPS, ConsensusEPS.
    """
    # Generate quarterly dates (dummy earnings announcement dates)
    dates = pd.date_range(start='2020-03-31', periods=8, freq='Q')
    actual_eps = np.random.uniform(1.0, 3.0, len(dates))
    consensus_eps = np.random.uniform(1.0, 3.0, len(dates))
    earnings_df = pd.DataFrame({
       'Date': dates,
       'ActualEPS': actual_eps,
       'ConsensusEPS': consensus_eps
    })
    earnings_df['Date'] = pd.to_datetime(earnings_df['Date'])
    return earnings_df

#%% Feature Engineering Functions
def create_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create technical indicators for the stock data.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added technical features.
    """
    # Price-based features
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Moving averages
    for window in [5, 10, 20, 50]:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'EMA_{window}'] = df['Close'].ewm(span=window).mean()
    
    # Volatility
    df['Volatility_20'] = df['Returns'].rolling(window=20).std()
    
    # RSI calculation
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD and Signal Line
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df

#%% Model Pipeline
def create_model_pipeline() -> Pipeline:
    """
    Create scikit-learn pipeline for the model.
    
    Returns
    -------
    Pipeline
        scikit-learn pipeline object.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    return pipeline

def train_model(X_train: pd.DataFrame, 
                y_train: pd.Series,
                pipeline: Pipeline) -> Tuple[Pipeline, Dict]:
    """
    Train the model using GridSearchCV.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    pipeline : Pipeline
        Model pipeline.
        
    Returns
    -------
    Tuple[Pipeline, Dict]
        Trained pipeline and best parameters.
    """
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

#%% Evaluation Functions
def evaluate_model(model: Pipeline,
                   X_test: pd.DataFrame,
                   y_test: pd.Series) -> Dict:
    """
    Evaluate model performance.
    
    Parameters
    ----------
    model : Pipeline
        Trained model pipeline.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test labels.
        
    Returns
    -------
    Dict
        Dictionary containing evaluation metrics.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'classification_report': classification_report(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    return metrics

#%% Visualization Functions
def plot_feature_importance(model: Pipeline,
                            feature_names: List[str]) -> None:
    """
    Plot feature importance.
    
    Parameters
    ----------
    model : Pipeline
        Trained model pipeline.
    feature_names : List[str]
        List of feature names.
    """
    classifier = model.named_steps['classifier']
    importances = classifier.feature_importances_
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='importance', y='feature')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()

#%% Main Execution
if __name__ == "__main__":
    # Load stock data for the ticker
    ticker = "AAPL"
    df_stock = load_stock_data(ticker)
    
    # Create technical features from stock price data
    df_stock = create_technical_features(df_stock)
    
    # Load earnings data (dummy implementation) and compute quarter periods
    df_earnings = load_earnings_data(ticker)
    df_stock['Quarter'] = df_stock['Date'].dt.to_period('Q')
    df_earnings['Quarter'] = df_earnings['Date'].dt.to_period('Q')
    
    # Merge stock data with quarterly earnings data
    df_merged = pd.merge(df_stock, df_earnings, on="Quarter", how="inner")
    
    # Compute earnings surprise and create binary target: 1 if positive surprise, 0 otherwise
    df_merged['EarningsSurprise'] = (df_merged['ActualEPS'] - df_merged['ConsensusEPS']) / df_merged['ConsensusEPS']
    df_merged['Target'] = (df_merged['EarningsSurprise'] > 0).astype(int)
    
    # Define features by excluding non-feature columns and earnings-specific columns
    exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 
                    'ActualEPS', 'ConsensusEPS', 'EarningsSurprise', 'Target', 'Quarter']
    feature_cols = [col for col in df_merged.columns if col not in exclude_cols]
    
    # Remove rows with missing values
    df_merged = df_merged.dropna()
    
    # Prepare features and target for modeling
    X = df_merged[feature_cols]
    y = df_merged['Target']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train the model pipeline
    pipeline = create_model_pipeline()
    best_model, best_params = train_model(X_train, y_train, pipeline)
    
    # Evaluate model performance
    metrics = evaluate_model(best_model, X_test, y_test)
    print("\nModel Performance:")
    print(metrics['classification_report'])
    print(f"ROC AUC Score: {metrics['roc_auc']:.4f}")
    
    # Plot feature importance
    plot_feature_importance(best_model, feature_cols)
