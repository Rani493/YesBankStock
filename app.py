import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load

st.title("Yes Bank Stock Price Prediction")

# Upload CSV
uploaded_file = st.file_uploader("Upload your Yes Bank stock data CSV")

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['Date'])
    df['Date'] = pd.to_datetime(df['Date'], format='%b-%d', errors='coerce') + pd.DateOffset(year=2023)
    df.set_index('Date', inplace=True)

    st.write("### Raw Data", df.tail())

    # Plot Closing Prices
    st.line_chart(df['Close'])

    # Load your trained model
    model = load("random_forest_model.joblib")

    # Preprocess
    df['High_Low_Diff'] = df['High'] - df['Low']
    df['Prev_Close'] = df['Close'].shift(1)
    df.dropna(inplace=True)

    # Predict
    features = df[['Open', 'High', 'Low', 'High_Low_Diff', 'Prev_Close']]
    prediction = model.predict(features)

    st.write("### Predicted Closing Prices")
    df['Predicted_Close'] = prediction
    st.line_chart(df[['Close', 'Predicted_Close']])
