import os
import ee
import geemap
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from nixtla import NixtlaClient
from flask import Blueprint, render_template, request, jsonify, send_file, current_app
from datetime import datetime
from dotenv import load_dotenv
import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import json
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Blueprint with a url_prefix
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Load environment variables
load_dotenv()

# Initialize Earth Engine
try:
    ee.Initialize(project='ee-rohithreddy66666')
    logger.info("Earth Engine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Earth Engine. Error: {e}")
    logger.error("Please ensure you have authenticated with Earth Engine.")

# Initialize Nixtla client
nixtla_client = NixtlaClient(api_key=os.getenv('NIXTLA_API_KEY'))

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        y_pred = self.linear(lstm_out[:, -1])
        return y_pred

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

@api_bp.route('/')
def index():
    return render_template('index.html')

@api_bp.route('/get_methane_data', methods=['POST'])
def get_methane_data():
    try:
        data = request.json
        geometry = ee.Geometry.Rectangle(data['bounds'])
        start_date = data['start_date']
        end_date = data['end_date']
        
        logger.info(f"Fetching methane data for area: {data['bounds']}, from {start_date} to {end_date}")
        
        # Load the Sentinel-5P TROPOMI CH4 dataset
        s5p = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_CH4') \
            .select('CH4_column_volume_mixing_ratio_dry_air') \
            .filterDate(start_date, end_date)
        
        # Calculate the mean CH4 concentration for the selected area
        def calculate_mean(image):
            mean = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=1000,
                maxPixels=1e9
            )
            return image.set('mean', mean.get('CH4_column_volume_mixing_ratio_dry_air'))

        s5p_with_means = s5p.map(calculate_mean)
        
        # Get the time series data
        time_series = s5p_with_means.aggregate_array('mean').getInfo()
        dates = s5p_with_means.aggregate_array('system:time_start').getInfo()
        
        # Filter out None values
        valid_data = [(d, v) for d, v in zip(dates, time_series) if v is not None]
        
        if not valid_data:
            logger.warning("No valid data points found for the selected region and time period")
            return jsonify({
                'status': 'error', 
                'message': 'No valid data points found for the selected region and time period'
            })
        
        # Unzip the valid data
        valid_dates, valid_values = zip(*valid_data)
        
        # Create a pandas DataFrame
        df = pd.DataFrame({
            'date': [datetime.fromtimestamp(d/1000).strftime('%Y-%m-%d') for d in valid_dates],
            'CH4': valid_values
        })
        
        # Save to CSV
        csv_path = os.path.join(current_app.root_path, 'methane_data.csv')
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Methane data saved to {csv_path}")
        return jsonify({
            'status': 'success', 
            'message': 'Data saved to CSV', 
            'csv_path': csv_path
        })
    
    except Exception as e:
        logger.error(f"Error in get_methane_data: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error', 
            'message': f'An error occurred: {str(e)}'
        })

@api_bp.route('/analyze_data', methods=['POST'])
def analyze_data():
    csv_path = os.path.join(current_app.root_path, 'methane_data.csv')
    if not os.path.exists(csv_path):
        return jsonify({
            'status': 'error', 
            'message': 'CSV file not found. Please fetch data first.'
        })

    try:
        # Load the CSV file
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Get the latest date from our data
        latest_date = df['date'].max()
        
        # Generate dates for next 30 days
        forecast_dates = pd.date_range(
            start=latest_date + pd.Timedelta(days=1),
            periods=30,  # Changed to 30 days for one month
            freq='D'
        )
        
        logger.info(f"Latest date in data: {latest_date}")
        logger.info(f"Forecasting for dates: {forecast_dates}")

        # LSTM Analysis
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df['CH4'].values.reshape(-1, 1))

        # Prepare sequences for LSTM
        seq_length = 10
        X, y = create_sequences(scaled_data, seq_length)

        # Split into train and test sets
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test)

        # Initialize and train LSTM model
        model = LSTMModel()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        num_epochs = 100
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()

        # Generate LSTM predictions for the next 30 days
        model.eval()
        with torch.no_grad():
            last_sequence = torch.FloatTensor(scaled_data[-seq_length:]).reshape(1, seq_length, 1)
            lstm_predictions = []
            
            current_sequence = last_sequence.clone()
            for _ in range(30):  # Predict next 30 days
                pred = model(current_sequence)
                lstm_predictions.append(pred.item())
                
                # Update sequence for next prediction
                new_sequence = current_sequence.squeeze(0)[:, 0][1:]
                new_value = pred.reshape(1)
                current_sequence = torch.cat([new_sequence, new_value]).reshape(1, seq_length, 1)

        # Inverse transform LSTM predictions
        lstm_predictions = scaler.inverse_transform(np.array(lstm_predictions).reshape(-1, 1))

        # Calculate LSTM metrics
        model.eval()
        with torch.no_grad():
            test_predictions = model(X_test).detach().numpy()  # Fixed detach issue
            y_test_numpy = y_test.detach().numpy()  # Fixed detach issue
            lstm_rmse = np.sqrt(mean_squared_error(y_test_numpy, test_predictions))
            lstm_mae = mean_absolute_error(y_test_numpy, test_predictions)

        # Nixtla analysis
        df_nixtla = df.rename(columns={'date': 'ds', 'CH4': 'y'})
        fcst_df = nixtla_client.forecast(df_nixtla, h=30, level=[80, 90], freq='D')  # Changed to 30 days

        # Calculate Nixtla metrics
        nixtla_rmse = np.sqrt(mean_squared_error(df_nixtla['y'].tail(5), fcst_df['TimeGPT'][:5]))
        nixtla_mae = mean_absolute_error(df_nixtla['y'].tail(5), fcst_df['TimeGPT'][:5])

        # Calculate monthly mean
        lstm_mean = np.mean(lstm_predictions)
        nixtla_mean = np.mean(fcst_df['TimeGPT'])

        # Create visualization
        fig = go.Figure()

        # Plot historical data
        fig.add_trace(go.Scatter(
            x=df['date'].dt.strftime('%Y-%m-%d'),
            y=df['CH4'],
            mode='lines',
            name='Historical Data',
            line=dict(color='blue')
        ))

        # Plot LSTM forecast
        fig.add_trace(go.Scatter(
            x=forecast_dates.strftime('%Y-%m-%d'),
            y=lstm_predictions.flatten(),
            mode='lines+markers',
            name='LSTM Forecast',
            line=dict(color='green')
        ))

        # Plot Nixtla forecast
        fig.add_trace(go.Scatter(
            x=forecast_dates.strftime('%Y-%m-%d'),
            y=fcst_df['TimeGPT'],
            mode='lines+markers',
            name='Nixtla Forecast',
            line=dict(color='red')
        ))

        # Add horizontal lines for monthly means
        fig.add_hline(y=lstm_mean, 
                     line_dash="dash", 
                     line_color="green", 
                     annotation_text=f"LSTM Mean: {lstm_mean:.2f}")
        fig.add_hline(y=nixtla_mean, 
                     line_dash="dash", 
                     line_color="red", 
                     annotation_text=f"Nixtla Mean: {nixtla_mean:.2f}")

        # Update layout
        fig.update_layout(
            title='CH4 Time Series Forecast - 30 Days Prediction',
            xaxis_title='Date',
            yaxis_title='CH4 Concentration (ppb)',
            legend=dict(y=0.99, x=0.01, yanchor='top', xanchor='left'),
            hovermode='x unified',
            showlegend=True,
            height=600
        )

        # Convert the figure to JSON
        plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Prepare results
        results = {
            'status': 'success',
            'nixtla_forecast': fcst_df['TimeGPT'].tolist(),
            'lstm_forecast': lstm_predictions.flatten().tolist(),
            'dates': forecast_dates.strftime('%Y-%m-%d').tolist(),
            'latest_historical_date': latest_date.strftime('%Y-%m-%d'),
            'monthly_means': {
                'lstm': float(lstm_mean),
                'nixtla': float(nixtla_mean)
            },
            'plot_json': plot_json,
            'nixtla_metrics': {
                'rmse': float(nixtla_rmse),
                'mae': float(nixtla_mae)
            },
            'lstm_metrics': {
                'rmse': float(lstm_rmse),
                'mae': float(lstm_mae)
            }
        }

        return jsonify(results)

    except Exception as e:
        logger.error(f"Error in analyze_data: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error', 
            'message': f'Error during analysis: {str(e)}'
        })

@api_bp.route('/download_csv')
def download_csv():
    csv_path = os.path.join(current_app.root_path, 'methane_data.csv')
    if os.path.exists(csv_path):
        return send_file(
            csv_path,
            mimetype='text/csv',
            as_attachment=True,
            download_name='methane_data.csv'
        )
    else:
        return jsonify({
            'status': 'error',
            'message': 'CSV file not found'
        }), 404

if __name__ == '__main__':
    app.run(debug=True)