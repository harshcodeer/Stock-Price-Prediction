
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("stock_price_model.keras")
scaler = joblib.load("scaler.pkl")

st.title("📈 Stock Price Prediction")

stock = st.text_input("Enter Stock Symbol", "AAPL")

if st.button("Predict"):

    df = yf.download(stock, period="2y")
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA100"] = df["Close"].rolling(100).mean()
    delta = df["Close"].diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    df["RSI"] = 100 - (100 / (1 + rs))
    shortEMA = df["Close"].ewm(span=12).mean()

    longEMA = df["Close"].ewm(span=26).mean()

    df["MACD"] = shortEMA-longEMA

    df["Signal"] = df["MACD"].ewm(span=9).mean()

    data = df[['Close']]

    scaled = scaler.transform(data)

    x = []

    for i in range(100, len(scaled)):
        x.append(scaled[i-100:i])

    x = np.array(x)

    predictions = model.predict(x)

    predictions = scaler.inverse_transform(predictions)


    # Next Day Prediction
    next_day = predictions[-1][0]

    st.subheader("Next Day Predicted Price")
    st.success(f"${next_day:.2f}")

    # Actual Values
    actual = scaler.inverse_transform(scaled[100:])

    # Performance Metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import numpy as np

    rmse = np.sqrt(mean_squared_error(actual, predictions))
    mae = mean_absolute_error(actual, predictions)
    r2 = r2_score(actual, predictions)

    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("RMSE", f"{rmse:.2f}")
    col2.metric("MAE", f"{mae:.2f}")
    col3.metric("R² Score", f"{r2:.3f}")

    # Actual vs Predicted Chart
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12,6))

    ax.plot(actual, label="Actual Price")
    ax.plot(predictions, label="Predicted Price")

    ax.set_title("Actual vs Predicted Stock Price")
    ax.set_xlabel("Days")
    ax.set_ylabel("Closing Price")
    ax.legend()

    st.pyplot(fig)
    fig, ax = plt.subplots(figsize=(12,6))

    ax.plot(df["Close"], label="Close")
    ax.plot(df["MA50"], label="50 Day MA")
    ax.plot(df["MA100"], label="100 Day MA")

    ax.legend()

    st.pyplot(fig)
    st.subheader("Relative Strength Index (RSI)")
    st.line_chart(df["RSI"])
    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(df["MACD"],label="MACD")

    ax.plot(df["Signal"],label="Signal")

    ax.legend()

    st.pyplot(fig)

 