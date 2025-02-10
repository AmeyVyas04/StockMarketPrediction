import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load the model (ensure you have a trained model saved)
MODEL_PATH = "C:\\Users\\AmeyV\\OneDrive\\Desktop\\AIML\\Projects material\\Stockmarketprediction\\Model-13.keras"
  # Change this if needed
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Streamlit UI
st.title("Stock Market Prediction using LSTM")

# Stock selection
stock = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, BHEL.BO):", "BHEL.BO")
start_date = st.date_input("Start Date", pd.to_datetime("2019-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2025-01-06"))

if st.button("Predict"):
    try:
        # Fetch stock data
        data = yf.download(stock, start=start_date, end=end_date)
        if data.empty:
            st.error("No data found for the given stock symbol and date range.")
            st.stop()
        
        st.write("Stock Data:", data.tail())
        
        # Preprocessing
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data[['Close']])
        
        # Prepare input for LSTM
        X_test = []
        for i in range(100, len(data_scaled)):
            X_test.append(data_scaled[i-100:i])
        X_test = np.array(X_test)
        
        if X_test.shape[0] == 0:
            st.error("Not enough data points for prediction. Try selecting a larger date range.")
            st.stop()
        
        # Predictions
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        
        # Plot results
        st.subheader("Actual vs Predicted Prices")
        fig, ax = plt.subplots()
        ax.plot(data.index[100:], data['Close'][100:], label="Actual Price", color='blue')
        ax.plot(data.index[100:], predictions, label="Predicted Price", color='red')
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
