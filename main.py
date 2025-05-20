import argparse
from data_preparation import DataPreparator
from model import RainfallClassifier
from evaluation import ModelEvaluator
import os

def create_output_directories():
    """Create directories for outputs if they don't exist."""
    directories = ['plots', 'models', 'reports']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Rainfall Prediction Classifier')
    parser.add_argument('--data_path', type=str, default='data/weather_data.csv',
                      help='Path to the weather dataset')
    parser.add_argument('--model_type', type=str, default='random_forest',
                      choices=['random_forest', 'logistic_regression', 'xgboost', 'ensemble'],
                      help='Type of model to train')
    parser.add_argument('--ensemble_method', type=str, default='voting',
                      choices=['voting', 'stacking'],
                      help='Ensemble method to use if model_type is ensemble')
    parser.add_argument('--optimize', action='store_true',
                      help='Whether to perform hyperparameter optimization')
    parser.add_argument('--calibrate', action='store_true',
                      help='Whether to calibrate the model')
    args = parser.parse_args()

    # Create output directories
    create_output_directories()

    # Step 1: Data Preparation
    print("\nStep 1: Data Preparation")
    print("=" * 50)
    preparator = DataPreparator(args.data_path)
    
    # Load and explore data
    data = preparator.load_data()
    preparator.explore_data()
    
    # Create basic visualizations
    preparator.create_visualizations()
    
    # Perform advanced feature engineering
    preparator.engineer_features()
    
    # Create advanced visualizations
    preparator.create_advanced_visualizations()
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preparator.preprocess_data()

    # Step 2: Model Training
    print("\nStep 2: Model Training")
    print("=" * 50)
    classifier = RainfallClassifier()
    
    if args.model_type == 'ensemble':
        # Train individual models first
        for model_name in ['random_forest', 'logistic_regression', 'xgboost']:
            if args.optimize:
                print(f"\nOptimizing {model_name}...")
                classifier.optimize_hyperparameters(X_train, y_train, model_name)
            else:
                print(f"\nTraining {model_name}...")
                classifier.train_model(X_train, y_train, model_name)
        
        # Create ensemble
        print(f"\nCreating {args.ensemble_method} ensemble...")
        classifier.create_ensemble(X_train, y_train, method=args.ensemble_method)
        
        # Analyze model agreement
        print("\nAnalyzing model agreement...")
        classifier.analyze_model_agreement(X_test, y_test)
    else:
        if args.optimize:
            print(f"\nOptimizing hyperparameters for {args.model_type}...")
            classifier.optimize_hyperparameters(X_train, y_train, args.model_type)
        else:
            print(f"\nTraining {args.model_type}...")
            classifier.train_model(X_train, y_train, args.model_type)

    # Step 3: Model Evaluation
    print("\nStep 3: Model Evaluation")
    print("=" * 50)
    evaluator = ModelEvaluator(classifier.best_model, classifier.best_model_name)
    
    # Calibrate model if requested
    if args.calibrate:
        print("\nCalibrating model...")
        evaluator.calibrate_model(X_train, y_train)
    
    # Generate comprehensive evaluation report
    evaluator.generate_evaluation_report(
        X_test, y_test,
        feature_names=preparator.data.drop('RainTomorrow', axis=1).columns
    )

    # Save the trained model
    model_path = os.path.join('models', f'rainfall_model_{args.model_type}.joblib')
    classifier.save_model(model_path)
    print(f"\nModel saved to {model_path}")

    # Print summary
    print("\nProject Summary")
    print("=" * 50)
    print(f"Model Type: {args.model_type}")
    if args.model_type == 'ensemble':
        print(f"Ensemble Method: {args.ensemble_method}")
    print(f"Hyperparameter Optimization: {'Yes' if args.optimize else 'No'}")
    print(f"Model Calibration: {'Yes' if args.calibrate else 'No'}")
    print("\nOutputs saved in:")
    print("- plots/: Visualizations and evaluation plots")
    print("- models/: Trained model files")
    print("- reports/: Evaluation reports and metrics")

if __name__ == "__main__":
    main() 