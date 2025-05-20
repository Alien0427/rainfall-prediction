import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class DataPreparator:
    def __init__(self, data_path):
        """Initialize the DataPreparator with the path to the dataset."""
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.feature_importance = None

    def load_data(self):
        """Load the dataset from the specified path."""
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            return self.data
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None

    def explore_data(self):
        """Perform basic data exploration."""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return

        print("\nBasic Information:")
        print(self.data.info())
        
        print("\nMissing Values:")
        print(self.data.isnull().sum())
        
        print("\nBasic Statistics:")
        print(self.data.describe())

    def create_visualizations(self):
        """Create basic visualizations for data exploration."""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return

        # Create correlation heatmap (numeric columns only)
        numeric_data = self.data.select_dtypes(include=[np.number])
        plt.figure(figsize=(12, 8))
        sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')
        plt.close()

        # Create distribution plots for numerical features
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=self.data, x=col, hue='RainTomorrow', multiple="stack")
            plt.title(f'Distribution of {col} by Rain Tomorrow')
            plt.savefig(f'distribution_{col}.png')
            plt.close()

    def engineer_features(self):
        """Create advanced features for better prediction."""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return

        # Create interaction features
        if 'Temperature' in self.data.columns and 'Humidity' in self.data.columns:
            self.data['TempHumidity'] = self.data['Temperature'] * self.data['Humidity']
        
        if 'WindSpeed' in self.data.columns and 'WindDirection' in self.data.columns:
            self.data['WindSpeedDirection'] = self.data['WindSpeed'] * np.cos(np.radians(self.data['WindDirection']))

        # Create rolling statistics
        if 'Rainfall' in self.data.columns:
            self.data['Rainfall_3d_avg'] = self.data['Rainfall'].rolling(window=3).mean()
            self.data['Rainfall_7d_avg'] = self.data['Rainfall'].rolling(window=7).mean()

        # Create seasonal features
        if 'Date' in self.data.columns:
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            self.data['Month'] = self.data['Date'].dt.month
            self.data['Season'] = self.data['Date'].dt.month % 12 // 3 + 1
            self.data['DayOfYear'] = self.data['Date'].dt.dayofyear

        # Create lag features
        if 'Rainfall' in self.data.columns:
            self.data['Rainfall_lag1'] = self.data['Rainfall'].shift(1)
            self.data['Rainfall_lag2'] = self.data['Rainfall'].shift(2)

        # Create polynomial features for important numerical columns
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col != 'RainTomorrow':  # Don't create polynomial of target
                self.data[f'{col}_squared'] = self.data[col] ** 2

        # Handle missing values created by rolling and lag features
        self.data = self.data.fillna(method='bfill').fillna(method='ffill')

        print("Feature engineering completed.")
        print(f"New shape of dataset: {self.data.shape}")

    def analyze_feature_importance(self, model):
        """Analyze and visualize feature importance."""
        if hasattr(model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.data.drop('RainTomorrow', axis=1).columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            plt.figure(figsize=(12, 6))
            sns.barplot(data=self.feature_importance.head(10), x='importance', y='feature')
            plt.title('Top 10 Most Important Features')
            plt.tight_layout()
            plt.savefig('feature_importance_top10.png')
            plt.close()

    def create_advanced_visualizations(self):
        """Create advanced visualizations for better data understanding."""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return

        # Create pairplot for top correlated features
        corr_matrix = self.data.corr()
        top_corr_features = corr_matrix['RainTomorrow'].abs().sort_values(ascending=False).head(6).index
        sns.pairplot(self.data[top_corr_features], hue='RainTomorrow')
        plt.savefig('feature_pairplot.png')
        plt.close()

        # Create time series plot if date is available
        if 'Date' in self.data.columns:
            plt.figure(figsize=(15, 6))
            self.data.set_index('Date')['Rainfall'].plot()
            plt.title('Rainfall Time Series')
            plt.savefig('rainfall_timeseries.png')
            plt.close()

        # Create seasonal analysis
        if 'Month' in self.data.columns:
            monthly_rain = self.data.groupby('Month')['RainTomorrow'].mean()
            plt.figure(figsize=(10, 6))
            monthly_rain.plot(kind='bar')
            plt.title('Average Rainfall Probability by Month')
            plt.xlabel('Month')
            plt.ylabel('Probability of Rain')
            plt.savefig('monthly_rainfall.png')
            plt.close()

    def preprocess_data(self, target_column='RainTomorrow', test_size=0.2, random_state=42):
        """Preprocess the data for model training."""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return

        # Handle missing values
        self.data = self.data.fillna(self.data.mean())

        # Remove date columns if they exist
        date_cols = self.data.select_dtypes(include=['datetime64']).columns
        self.data = self.data.drop(date_cols, axis=1)

        # Separate features and target
        X = self.data.drop(target_column, axis=1)
        y = self.data[target_column]

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Scale the features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        print("Data preprocessing completed.")
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Testing set shape: {self.X_test.shape}")

        return self.X_train, self.X_test, self.y_train, self.y_test

if __name__ == "__main__":
    # Example usage
    preparator = DataPreparator("data/weather_data.csv")
    preparator.load_data()
    preparator.explore_data()
    preparator.create_visualizations()
    preparator.engineer_features()
    preparator.analyze_feature_importance()
    preparator.create_advanced_visualizations()
    preparator.preprocess_data() 