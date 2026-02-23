import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, brier_score_loss


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


# =========================
# Fundamental Factor Scoring
# =========================

def safe_get(info: dict, key: str, default=None):
    """
    Safe dictionary access for yfinance .info.
    """
    try:
        value = info.get(key, default)
        if value is None:
            return default
        if isinstance(value, float) and np.isnan(value):
            return default
        return value
    except Exception:
        return default


def score_pe(pe: float) -> float:
    """
    Heuristic scoring for P/E:
    Lower is generally better up to a point.
    Returns a score between 0 and 100.
    """
    if pe is None or pe <= 0:
        return 50.0  # neutral if invalid
    if pe < 10:
        return 90.0
    if pe < 20:
        return 80.0
    if pe < 30:
        return 65.0
    if pe < 50:
        return 50.0
    if pe < 80:
        return 40.0
    return 30.0


def score_margin(margin: float) -> float:
    """
    Score profit margins (0-1) into 0-100.
    """
    if margin is None:
        return 50.0
    margin = max(min(margin, 0.4), -0.4)   # cap between -40% and +40%
    return (margin + 0.4) / 0.8 * 100.0    # map [-0.4, 0.4] to [0, 100]


def score_growth(growth: float) -> float:
    """
    Score revenue or earnings growth into 0-100.
    """
    if growth is None:
        return 50.0
    growth = max(min(growth, 0.5), -0.5)   # cap between -50% and +50%
    return (growth + 0.5) / 1.0 * 100.0    # map [-0.5, 0.5] to [0, 100]


def score_debt_to_equity(de: float) -> float:
    """
    Score debt-to-equity: lower is better.
    """
    if de is None or de < 0:
        return 50.0
    if de < 0.5:
        return 85.0
    if de < 1.0:
        return 75.0
    if de < 2.0:
        return 60.0
    if de < 3.0:
        return 45.0
    return 30.0


def get_fundamental_scores(ticker: str):
    """
    Fetch some basic fundamentals via yfinance and convert them to 0-100 sub-scores.
    Returns (factor_score, details_dict).
    """
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
    except Exception:
        info = {}

    trailing_pe = safe_get(info, "trailingPE")
    forward_pe = safe_get(info, "forwardPE")
    profit_margin = safe_get(info, "profitMargins")
    revenue_growth = safe_get(info, "revenueGrowth")
    debt_to_equity = safe_get(info, "debtToEquity")

    # Compute sub-scores
    trailing_pe_score = score_pe(trailing_pe)
    forward_pe_score = score_pe(forward_pe)
    margin_score = score_margin(profit_margin)
    growth_score = score_growth(revenue_growth)
    de_score = score_debt_to_equity(debt_to_equity)

    scores = [trailing_pe_score, forward_pe_score, margin_score, growth_score, de_score]
    valid_scores = [s for s in scores if s is not None]

    if len(valid_scores) == 0:
        factor_score = 50.0
    else:
        factor_score = float(np.mean(valid_scores))

    details = {
        "trailingPE": trailing_pe,
        "trailingPE_score": trailing_pe_score,
        "forwardPE": forward_pe,
        "forwardPE_score": forward_pe_score,
        "profitMargins": profit_margin,
        "profitMargins_score": margin_score,
        "revenueGrowth": revenue_growth,
        "revenueGrowth_score": growth_score,
        "debtToEquity": debt_to_equity,
        "debtToEquity_score": de_score,
        "factor_score": factor_score,
    }

    return factor_score, details


# =========================
# ML Training + Brier Score
# =========================

def train_model_v2(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Train a RandomForest classifier and return model, accuracy, and Brier score.
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

    # Predictions on hold-out test set
    y_pred = model.predict(X_test)

    # Ensure y_test is 1D
    y_test_1d = np.ravel(y_test)

    # Accuracy
    acc = accuracy_score(y_test_1d, y_pred)

    # Brier score for probability calibration (class 1 = "UP")
    proba_test = model.predict_proba(X_test)[:, 1]
    proba_test_1d = np.ravel(proba_test)
    brier = brier_score_loss(y_test_1d, proba_test_1d)

    return model, acc, brier


def predict_next_day_v2(model, X: pd.DataFrame):
    """
    Use a trained model and the latest feature row to predict next day UP/DOWN.
    Returns direction, confidence (raw prob), and probabilities.
    """
    latest_features = X.iloc[[-1]]  # keep as DataFrame
    proba = model.predict_proba(latest_features)[0]  # [P(0), P(1)]

    p_down, p_up = proba[0], proba[1]
    direction = "UP" if p_up >= p_down else "DOWN"
    confidence = max(p_up, p_down)

    return direction, confidence, p_up, p_down


# =========================
# Streamlit App (Hybrid)
# =========================

def main():
    st.set_page_config(page_title="Hybrid Stock Predictor (V2 ML + Fundamentals)", page_icon="📈")
    st.title("📈 Hybrid Stock Predictor (ML + Fundamentals)")
    st.write(
        """
        This app uses a **hybrid approach**:
        - A machine-learning model (Random Forest) on recent price action, and
        - A simple fundamental factor score (P/E, margins, growth, debt)
        
        It predicts whether a stock is more likely to go **UP or DOWN next trading day**,
        and assigns a **1–100 rating** plus a **confidence percentage**.
        """
    )

    st.info("Phase 1 MVP focuses on the **1-day horizon** for a single ticker. Additional horizons and backtests will follow.")

    # Sidebar configuration
    st.sidebar.header("Configuration")

    default_ticker = "SPY"
    ticker = st.sidebar.text_input("Ticker symbol", value=default_ticker).upper()

    lookback_years = st.sidebar.slider(
        "Years of history to use (for training)",
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

        with st.spinner(f"Fetching data, training ML model, and computing scores for {ticker}..."):
            try:
                # 1) Get price data
                data = get_price_history(ticker, lookback_years=lookback_years)
                if data.empty:
                    st.error("No data returned. Please check the ticker symbol.")
                    return

                # 2) Build features
                df, X, y, feature_cols = build_features_v2(data)

                if len(df) < 150:
                    st.warning(
                        "Not much historical data available after feature engineering. "
                        "Predictions and calibration may be less reliable."
                    )

                # 3) Train model (ML)
                model, acc, brier = train_model_v2(X, y, test_size=test_size)

                # 4) Predict next day via ML
                direction_ml, confidence_ml, p_up, p_down = predict_next_day_v2(model, X)

                # 5) ML-based score (1–100) from p_up
                ml_score = p_up * 100.0

                # 6) Fundamentals factor score
                factor_score, factor_details = get_fundamental_scores(ticker)

                # 7) Hybrid final score (1–100) as weighted combination
                final_score = 0.7 * ml_score + 0.3 * factor_score
                final_score = float(np.clip(final_score, 0.0, 100.0))

                # 8) Direction from final score
                direction_final = "UP" if final_score > 50.0 else "DOWN"

                # 9) Confidence percentage: blend of ML margin and calibration (Brier)
                model_margin = abs(p_up - 0.5) * 2.0  # 0 to 1
                brier_clipped = float(np.clip(brier, 0.0, 1.0))
                calibration_factor = 1.0 - brier_clipped  # higher is better
                raw_confidence = model_margin * calibration_factor
                confidence_pct = float(np.clip(raw_confidence * 100.0, 0.0, 100.0))

                # 10) Build a one-row DataFrame for export and display
                latest_row = df.iloc[-1]
                last_date = latest_row.name.date()
                last_close = latest_row["Close"]

                result_row = {
                    "ticker": ticker,
                    "last_date": last_date,
                    "last_close": last_close,
                    "final_score_1d": final_score,
                    "direction_1d": direction_final,
                    "confidence_pct": confidence_pct,
                    "p_up_1d": p_up,
                    "p_down_1d": p_down,
                    "ml_score_1d": ml_score,
                    "factor_score": factor_score,
                    "backtest_accuracy": acc,
                    "brier_score": brier,
                }
                result_df = pd.DataFrame([result_row])

            except Exception as e:
                st.error(f"Something went wrong: {e}")
                return

        # =========================
        # Display Results
        # =========================
        st.subheader(f"Results for {ticker}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Last close date", str(last_date))
            st.metric("Last close price", f"${last_close:,.2f}")
        with col2:
            st.metric("Hybrid score (1-day, 1–100)", f"{final_score:.1f}")
            st.metric("Predicted direction (1-day)", direction_final)
        with col3:
            st.metric("Confidence % (hybrid)", f"{confidence_pct:.1f}%")
            st.metric("Brier score (hold-out)", f"{brier:.3f}")

        st.markdown("---")
        st.subheader("🔎 ML Probability & Factor Breakdown")

        st.write(f"**ML Probability stock goes UP next day (p_up):** `{p_up * 100:.1f}%`")
        st.write(f"**ML Probability stock goes DOWN next day (p_down):** `{p_down * 100:.1f}%`")
        st.write(f"**ML-only score (1–100):** `{ml_score:.1f}`")
        st.write(f"**Factor-only score (1–100):** `{factor_score:.1f}`")

        st.info(
            "The **hybrid score** combines the ML-based probability of going up with a basic fundamental score "
            "to produce a 1–100 rating, where 1 = very bearish, 50 = neutral, 100 = very bullish."
        )

        # Fundamental details table
        st.markdown("#### Fundamental factors (raw + scores)")
        factor_table = pd.DataFrame({
            "metric": [
                "trailingPE", "forwardPE", "profitMargins",
                "revenueGrowth", "debtToEquity"
            ],
            "raw_value": [
                factor_details["trailingPE"],
                factor_details["forwardPE"],
                factor_details["profitMargins"],
                factor_details["revenueGrowth"],
                factor_details["debtToEquity"]
            ],
            "score_0_100": [
                factor_details["trailingPE_score"],
                factor_details["forwardPE_score"],
                factor_details["profitMargins_score"],
                factor_details["revenueGrowth_score"],
                factor_details["debtToEquity_score"],
            ]
        })
        st.dataframe(factor_table)

        # =========================
        # Charts
        # =========================
        st.markdown("---")
        st.subheader("Price History (Close)")

        price_to_show = df[["Close"]].copy()
        try:
            price_to_show.index = price_to_show.index.tz_localize(None)
        except Exception:
            pass
        st.line_chart(price_to_show)

        st.subheader("Recent Features Snapshot (Last 10 Days)")
        st.dataframe(df[feature_cols + ["target"]].tail(10))

        # =========================
        # Downloadable Result
        # =========================
        st.markdown("---")
        st.subheader("Export Prediction")

        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download 1-day prediction as CSV",
            data=csv_bytes,
            file_name=f"{ticker}_hybrid_prediction_1d.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
