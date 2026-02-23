import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# =========================
# Data + Feature Functions
# =========================

def get_price_history(ticker: str, lookback_years: int = 3) -> pd.DataFrame:
    """
    Download historical daily OHLC data for a ticker using yfinance.
    """
    end = datetime.today()
    start = end - timedelta(days=365 * lookback_years)
    data = yf.download(ticker, start=start, end=end)
    data.dropna(inplace=True)
    return data


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI) for a price series.
    """
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    gain_rol = pd.Series(gain, index=series.index).rolling(window=period).mean()
    loss_rol = pd.Series(loss, index=series.index).rolling(window=period).mean()

    rs = gain_rol / (loss_rol + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def build_features_v2(data: pd.DataFrame):
    """
    Build feature set and target for the ML model.
