# price_predictor.py

import pandas as pd
import numpy as np
import pickle # Used to save and load the trained model
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

def load_data(file_path):
    """
    Loads the Walmart sales data from a CSV file.
    
    Args:
        file_path (str): The path to the CSV file.
        
    Returns:
        pd.DataFrame: The loaded DataFrame, or None if an error occurs.
    """
    try:
        # Load the dataset
        df = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure the path is correct.")
        return None
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return None

def preprocess_and_train_model(df, store_id=1, dept_id=1, prediction_days=30):
    """
    Preprocesses the data, trains an ARIMA model, and saves the model.
    
    Args:
        df (pd.DataFrame): The raw data loaded from the CSV.
        store_id (int): The store ID to filter the data.
        dept_id (int): The department ID to filter the data.
        prediction_days (int): The number of days to predict into the future.
        
    Returns:
        tuple: A tuple containing the trained model, the historical data series, 
               and the future prediction series. Returns (None, None, None) on error.
    """
    print("Step 1: Preprocessing data for Store and Department...")
    
    # Filter for a specific store and department to simulate a single product
    # The 'Weekly_Sales' column will be treated as the 'Price' for this simulation.
    data = df[(df['Store'] == store_id) & (df['Dept'] == dept_id)]
    
    if data.empty:
        print(f"Error: No data found for Store {store_id} and Department {dept_id}.")
        return None, None, None

    # Convert the 'Date' column to a datetime object and set it as the index
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date').sort_index()

    # Create a time series from the 'Weekly_Sales' column
    # For a real price predictor, you would use a 'Price' column
    ts = data['Weekly_Sales']
    
    # Fill any missing values with the mean of the series
    ts = ts.fillna(ts.mean())

    print("Step 2: Training the ARIMA model...")
    # The order (p,d,q) for ARIMA is crucial. (5,1,0) is a common starting point
    # for non-seasonal time series data. You can tune this later.
    try:
        model = ARIMA(ts, order=(5, 1, 0))
        model_fit = model.fit()
        
        # Save the trained model to a file for later use by app.py
        with open('price_predictor.pkl', 'wb') as pkl_file:
            pickle.dump(model_fit, pkl_file)
        
        print("Model trained and saved as 'price_predictor.pkl'.")
    except Exception as e:
        print(f"Error during model training: {e}")
        return None, None, None
        
    # Make future predictions
    print(f"Step 3: Forecasting price for the next {prediction_days} days...")
    # The 'end' date for forecasting will be 'prediction_days' after the last date
    # in the dataset.
    last_date = ts.index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=prediction_days, freq='D')
    
    # Use the model to predict the future values
    forecast_series = model_fit.forecast(steps=prediction_days)
    forecast_series.index = future_dates
    
    return model_fit, ts, forecast_series

def visualize_results(historical_data, predictions):
    """
    Generates a plot showing historical data and future predictions.
    
    Args:
        historical_data (pd.Series): The historical time series.
        predictions (pd.Series): The future predictions.
    """
    print("Step 4: Generating visualization...")
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    plt.plot(historical_data.index, historical_data.values, label='Historical Price', color='blue', linewidth=2)
    
    # Plot predicted data
    plt.plot(predictions.index, predictions.values, label='Predicted Price', color='orange', linestyle='--', linewidth=2)
    
    plt.title('Product Price History and Future Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    
    # Save the plot to a file
    plt.savefig('static/images/price_chart.png')
    print("Chart saved to 'static/images/price_chart.png'.")
    plt.close()

def get_prediction(prediction_days=30):
    """
    Main function to run the full pipeline and get a prediction.
    
    Args:
        prediction_days (int): The number of days to predict.
        
    Returns:
        tuple: A tuple containing the historical series and the prediction series.
    """
    # Load the data from the 'data' folder
    file_path = 'data/product_price_data.csv' # Adjust path as per your directory structure
    df = load_data(file_path)
    
    if df is not None:
        model, historical, forecast = preprocess_and_train_model(df, prediction_days=prediction_days)
        if historical is not None and forecast is not None:
            visualize_results(historical, forecast)
            return historical, forecast
    return None, None

if __name__ == '__main__':
    # This block runs when the script is executed directly
    # It demonstrates the full pipeline from start to finish
    print("Running the price predictor as a standalone script...")
    historical_data, predicted_prices = get_prediction()
    if historical_data is not None and predicted_prices is not None:
        print("\nPipeline completed successfully.")
        print("Historical data last 5 rows:")
        print(historical_data.tail())
        print("\nPredicted prices:")
        print(predicted_prices)
