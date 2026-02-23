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
    Target: 1 if next day's close > today's close, else 0.
    """
    df = data.copy()

    # Daily returns
    df["ret_1d"] = df["Close"].pct_change(1)
    df["ret_3d"] = df["Close"].pct_change(3)
    df["ret_5d"] = df["Close"].pct_change(5)

    # Moving averages
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_30"] = df["Close"].rolling(window=30).mean()
    df["SMA_ratio_10_30"] = df["SMA_10"] / (df["SMA_30"] + 1e-9)

    # RSI
    df["RSI_14"] = compute_rsi(df["Close"], period=14)

    # Rolling volatility
    df["vol_10d"] = df["ret_1d"].rolling(window=10).std()

    # Target: 1 if next day's close > today's close, else 0
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # Drop rows with NaNs (from rolling / shift)
    df.dropna(inplace=True)

    feature_cols = [
        "ret_1d", "ret_3d", "ret_5d",
        "SMA_10", "SMA_30", "SMA_ratio_10_30",
        "RSI_14", "vol_10d"
    ]

    X = df[feature_cols]
    y = df["target"]

    return df, X, y, feature_cols


def train_model_v2(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Train a RandomForest classifier and return model and basic performance metrics.
    Uses a time-based split (no shuffling).
    """
    # Time-based split: earlier data = train, later data = test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Evaluate on hold-out test set
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, acc


def predict_next_day_v2(model, X: pd.DataFrame):
    """
    Use a trained model and the latest feature row to predict next day UP/DOWN.
    Returns direction, confidence, and raw probabilities.
    """
    latest_features = X.iloc[[-1]]  # keep as DataFrame
    proba = model.predict_proba(latest_features)[0]  # [P(0), P(1)]

    p_down, p_up = proba[0], proba[1]
    direction = "UP" if p_up >= p_down else "DOWN"
    confidence = max(p_up, p_down)

    return direction, confidence, p_up, p_down


# =========================
# Streamlit App
# =========================

def main():
    st.set_page_config(page_title="Stock Predictor V2 (ML)", page_icon="📈")
    st.title("📈 Stock Predictor V2 (ML Model)")
    st.write(
        """
        This app uses a machine-learning model (Random Forest) to analyze recent price action 
        and predict whether a stock is more likely to go **UP or DOWN next trading day**, 
        along with a **confidence percentage**.
        """
    )

    # Sidebar configuration
    st.sidebar.header("Configuration")

    default_ticker = "SPY"
    ticker = st.sidebar.text_input("Ticker symbol", value=default_ticker).upper()

    lookback_years = st.sidebar.slider(
        "Years of history to use",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="Number of past years of daily data to pull for model training."
    )

    test_size = st.sidebar.slider(
        "Test size (for backtest split)",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
        help="Fraction of the most recent data used as a hold-out test set."
    )

    if st.sidebar.button("Run Prediction"):
        if not ticker:
            st.error("Please enter a ticker symbol.")
            return

        with st.spinner(f"Fetching data and training model for {ticker}..."):
            try:
                # 1) Get data
                data = get_price_history(ticker, lookback_years=lookback_years)
                if data.empty:
                    st.error("No data returned. Please check the ticker symbol.")
                    return

                # 2) Build features
                df, X, y, feature_cols = build_features_v2(data)

                if len(df) < 100:
                    st.warning(
                        "Not much historical data available after feature engineering. "
                        "Predictions may be unreliable."
                    )

                # 3) Train model
                model, acc = train_model_v2(X, y, test_size=test_size)

                # 4) Predict next day
                direction, confidence, p_up, p_down = predict_next_day_v2(model, X)

            except Exception as e:
                st.error(f"Something went wrong: {e}")
                return

        # =========================
        # Display Results
        # =========================
        latest_row = df.iloc[-1]
        last_date = latest_row.name.date()
        last_close = latest_row["Close"]

        st.subheader(f"Results for {ticker}")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Last close date", str(last_date))
            st.metric("Last close price", f"${last_close:,.2f}")
        with col2:
            st.metric("Backtest accuracy (hold-out)", f"{acc * 100:.1f}%")

        st.markdown("---")
        st.subheader("📊 Next-Day Prediction")

        st.write(f"**Direction:** `{direction}`")
        st.write(f"**Confidence:** `{confidence * 100:.1f}%`")

        st.write(
            f"- Probability stock will **go UP** next day: `{p_up * 100:.1f}%`  \n"
            f"- Probability stock will **go DOWN** next day: `{p_down * 100:.1f}%`"
        )

        st.info(
            "Interpretation: This is a statistical model based on past price behavior. "
            "It does **not** guarantee future performance and should not be treated as financial advice."
        )

        # =========================
        # Charts
        # =========================
        st.markdown("---")
        st.subheader("Price History")

        price_to_show = df[["Close"]].copy()
        price_to_show.index = price_to_show.index.tz_localize(None)  # handle tz issues for plotting
        st.line_chart(price_to_show)

        st.subheader("Recent Features Snapshot (Last 10 Days)")
        st.dataframe(df[feature_cols + ["target"]].tail(10))


if __name__ == "__main__":
    main()
