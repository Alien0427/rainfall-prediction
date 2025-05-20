# Rainfall Prediction Classifier

This project implements a machine learning classifier to predict rainfall based on historical weather data. The model is designed to predict whether it will rain tomorrow based on various weather features such as temperature, humidity, and wind speed.

## Project Structure

```
rain_prediction/
├── data/                  # Directory for dataset
├── data_preparation.py    # Data exploration and preprocessing
├── model.py              # Classifier pipeline implementation
├── evaluation.py         # Model evaluation and visualization
├── main.py              # Main execution script
└── requirements.txt      # Project dependencies
```

## Setup and Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your weather dataset in the `data/` directory
2. Run the main script:
```bash
python main.py
```

## Project Components

### Data Preparation
- Data loading and exploration
- Feature engineering
- Data cleaning and preprocessing
- Train-test split

### Model Pipeline
- Feature selection
- Model training
- Hyperparameter optimization
- Cross-validation

### Evaluation
- Performance metrics calculation
- Visualization of results
- Model interpretation

## Dependencies
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- xgboost 