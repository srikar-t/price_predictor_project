# app.py

from flask import Flask, render_template, send_from_directory
import pickle
import pandas as pd
from price_predictor import get_prediction

# Create the Flask application instance
app = Flask(__name__)

@app.route('/')
def home():
    """
    Renders the homepage of the web application.
    This function will trigger the price prediction pipeline
    and pass the results to the HTML template.
    """
    # Run the full prediction pipeline from price_predictor.py
    # This will train the model, save it, generate the chart, and return the data.
    historical_data, predicted_prices = get_prediction()
    
    if historical_data is not None and predicted_prices is not None:
        # Extract key data points to display on the webpage
        current_price = historical_data.iloc[-1]
        predicted_price_in_30_days = predicted_prices.iloc[-1]
        
        # Format the data for a clean display
        historical_price_list = historical_data.tolist()
        predicted_price_list = predicted_prices.tolist()
        
        # Combine the labels for the chart
        all_labels = [str(d.date()) for d in historical_data.index] + [str(d.date()) for d in predicted_prices.index]
        
        # Render the HTML template and pass the data to it
        return render_template(
            'index.html', 
            current_price=f"${current_price:.2f}",
            predicted_price=f"${predicted_price_in_30_days:.2f}",
            historical_data=historical_price_list,
            predicted_data=predicted_price_list,
            labels=all_labels
        )
    else:
        # Handle cases where the data loading or model training failed
        return "An error occurred during data processing.", 500

@app.route('/static/images/<path:filename>')
def serve_image(filename):
    """
    Serves static images from the 'static/images' folder.
    This route is necessary for Flask to find and display the chart.
    """
    return send_from_directory('static/images', filename)

if __name__ == '__main__':
    # This is a standard entry point for a Flask application.
    # It starts the development server.
    app.run(debug=True)
