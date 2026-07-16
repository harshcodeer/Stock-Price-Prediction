# Stock Price Prediction using Long Short-Term Memory (LSTM)

## Project Overview

The **Stock Price Prediction** project aims to forecast future stock prices using historical stock market data and Deep Learning techniques. Financial markets are highly dynamic and influenced by multiple factors, making accurate prediction a challenging task. This project utilizes a **Long Short-Term Memory (LSTM)** neural network to learn historical price patterns and generate future stock price predictions.

## Problem Statement

Stock prices fluctuate continuously due to economic conditions, investor sentiment, company performance, and global events. Investors and traders require reliable forecasting methods to support informed investment decisions. Traditional statistical methods often struggle to capture long-term temporal dependencies in financial data. This project applies an LSTM-based deep learning model to analyze historical stock prices and improve prediction accuracy.

## Objective

The primary objectives of this project are:

* Predict future stock prices using historical market data.
* Analyze historical trends and price movements.
* Build a deep learning model capable of learning sequential financial data.
* Visualize actual versus predicted stock prices.
* Develop a Streamlit application for real-time stock price prediction.
* Demonstrate an end-to-end deep learning workflow from data collection to deployment.

## Dataset

The dataset consists of historical stock market data downloaded from Yahoo Finance.

### Selected Features

The project primarily uses:

* Open Price
* High Price
* Low Price
* Close Price
* Adjusted Close Price (if available)
* Trading Volume

### Target Variable

* Closing Price

The **Closing Price** is selected as the prediction target because it represents the final market value of the stock for each trading day and is widely used for trend analysis and forecasting.

## Data Preprocessing

The following preprocessing steps were performed:

* Collected historical stock data.
* Removed missing values and irrelevant records.
* Selected the Closing Price as the target variable.
* Normalized the data using MinMaxScaler.
* Converted the time series into sequential input windows suitable for LSTM.
* Split the dataset into training and testing datasets.

## Why LSTM?

The project uses the **Long Short-Term Memory (LSTM)** network because stock prices are sequential data with temporal dependencies.

Advantages of LSTM include:

* Learns long-term dependencies in time-series data.
* Handles sequential information efficiently.
* Reduces the vanishing gradient problem found in traditional RNNs.
* Performs well on financial forecasting tasks.
* Captures historical trends and patterns more effectively than conventional regression models.

## Model Architecture

The deep learning model consists of:

* Input Layer
* Multiple LSTM Layers
* Dropout Layers to reduce overfitting
* Dense Output Layer for predicting stock prices

The model was implemented using TensorFlow/Keras.

## Model Training

The dataset was divided into:

* Training Data: 80%
* Testing Data: 20%

The model was trained using:

* Optimizer: Adam
* Loss Function: Mean Squared Error (MSE)
* Epochs: Multiple iterations until convergence
* Batch Size: Selected to balance learning speed and performance

## Model Evaluation

The model was evaluated using:

* Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)

Additionally, the predicted stock prices were compared with the actual prices using visualization graphs.

### Why RMSE?

RMSE measures the average prediction error in the same unit as the stock price. A lower RMSE indicates better prediction accuracy and allows easier interpretation of forecasting performance.

## Results

The LSTM model successfully captured historical stock price trends and generated predictions that closely followed actual market movements. Comparison plots of actual versus predicted prices demonstrated the model's effectiveness in learning sequential financial patterns.

## Deployment

The trained LSTM model was saved and integrated into a Streamlit web application. Users can input or select stock market data to generate predicted stock prices through an interactive interface. The project is version-controlled using Git and GitHub and deployed using Streamlit Community Cloud.

## Technologies Used

* Python
* TensorFlow
* Keras
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* yFinance
* Streamlit
* Jupyter Notebook
* Git & GitHub

## Future Enhancements

* Integrate live stock market data using the Yahoo Finance API.
* Predict prices for multiple stocks simultaneously.
* Include technical indicators such as RSI, MACD, Moving Averages, and Bollinger Bands.
* Compare LSTM with GRU, Transformer, and XGBoost models.
* Add confidence intervals and future trend forecasting.
* Deploy the application with real-time market updates.

## Conclusion

This project demonstrates the application of Deep Learning in financial forecasting using an LSTM neural network. It covers the complete machine learning lifecycle, including data collection, preprocessing, sequence generation, model training, evaluation, visualization, deployment, and real-time prediction. The project provides valuable insights into time-series forecasting and showcases practical skills in Deep Learning, financial analytics, and model deployment.
