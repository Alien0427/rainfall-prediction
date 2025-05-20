import pandas as pd
import requests
import os
import numpy as np

def download_weather_data():
    """Download sample weather data from Kaggle's Australian Weather dataset."""
    print("Downloading sample weather data...")
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # URL for the sample dataset
    url = "https://raw.githubusercontent.com/datasets/weather-data/master/data/weather.csv"
    
    try:
        # Download the data
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Save the data
        with open('data/weather_data.csv', 'wb') as f:
            f.write(response.content)
        
        print("Data downloaded successfully!")
        
        # Load and display basic information about the dataset
        df = pd.read_csv('data/weather_data.csv')
        print("\nDataset Information:")
        print(f"Number of rows: {len(df)}")
        print(f"Number of columns: {len(df.columns)}")
        print("\nColumns:")
        print(df.columns.tolist())
        
    except Exception as e:
        print(f"Error downloading data: {str(e)}")
        print("\nAlternative: Creating synthetic weather data...")
        
        # Create synthetic data
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'Date': pd.date_range(start='2020-01-01', periods=n_samples),
            'Temperature': np.random.normal(25, 5, n_samples),
            'Humidity': np.random.normal(60, 15, n_samples),
            'WindSpeed': np.random.normal(15, 5, n_samples),
            'WindDirection': np.random.uniform(0, 360, n_samples),
            'Pressure': np.random.normal(1013, 5, n_samples),
            'Rainfall': np.random.exponential(2, n_samples)
        }
        
        # Create target variable based on features
        data['RainTomorrow'] = (
            (data['Temperature'] < 20) | 
            (data['Humidity'] > 70) | 
            (data['Rainfall'] > 5)
        ).astype(int)
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        df.to_csv('data/weather_data.csv', index=False)
        
        print("\nSynthetic Dataset Information:")
        print(f"Number of rows: {len(df)}")
        print(f"Number of columns: {len(df.columns)}")
        print("\nColumns:")
        print(df.columns.tolist())

if __name__ == "__main__":
    download_weather_data() 