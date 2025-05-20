from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class RainfallClassifier:
    def __init__(self):
        """Initialize the RainfallClassifier with different model options."""
        self.models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'logistic_regression': LogisticRegression(random_state=42),
            'xgboost': XGBClassifier(random_state=42)
        }
        self.best_model = None
        self.best_model_name = None
        self.ensemble_model = None

    def train_model(self, X_train, y_train, model_name='random_forest'):
        """Train the specified model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")

        print(f"\nTraining {model_name}...")
        model = self.models[model_name]
        model.fit(X_train, y_train)
        self.best_model = model
        self.best_model_name = model_name
        print(f"{model_name} training completed.")

    def create_ensemble(self, X_train, y_train, method='voting'):
        """Create an ensemble of models using voting or stacking."""
        if method not in ['voting', 'stacking']:
            raise ValueError("Method must be either 'voting' or 'stacking'")

        print(f"\nCreating {method} ensemble...")
        
        if method == 'voting':
            self.ensemble_model = VotingClassifier(
                estimators=[
                    ('rf', self.models['random_forest']),
                    ('lr', self.models['logistic_regression']),
                    ('xgb', self.models['xgboost'])
                ],
                voting='soft'
            )
        else:  # stacking
            self.ensemble_model = StackingClassifier(
                estimators=[
                    ('rf', self.models['random_forest']),
                    ('lr', self.models['logistic_regression']),
                    ('xgb', self.models['xgboost'])
                ],
                final_estimator=LogisticRegression(),
                cv=5
            )

        self.ensemble_model.fit(X_train, y_train)
        self.best_model = self.ensemble_model
        self.best_model_name = f"{method}_ensemble"
        print(f"{method.capitalize()} ensemble training completed.")

    def get_model_predictions(self, X):
        """Get predictions from all individual models."""
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict_proba(X)[:, 1]
        return predictions

    def analyze_model_agreement(self, X_test, y_test):
        """Analyze agreement between different models."""
        predictions = self.get_model_predictions(X_test)
        
        # Calculate agreement matrix
        n_models = len(predictions)
        agreement_matrix = np.zeros((n_models, n_models))
        model_names = list(predictions.keys())
        
        for i, name1 in enumerate(model_names):
            for j, name2 in enumerate(model_names):
                agreement = np.mean(predictions[name1] > 0.5 == predictions[name2] > 0.5)
                agreement_matrix[i, j] = agreement

        # Create agreement heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(agreement_matrix, annot=True, fmt='.2f', 
                   xticklabels=model_names, yticklabels=model_names)
        plt.title('Model Agreement Matrix')
        plt.tight_layout()
        plt.savefig('model_agreement.png')
        plt.close()

        return agreement_matrix

    def optimize_hyperparameters(self, X_train, y_train, model_name='random_forest'):
        """Optimize hyperparameters for the specified model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")

        param_grids = {
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'class_weight': ['balanced', None]
            },
            'logistic_regression': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear'],
                'class_weight': ['balanced', None]
            },
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        }

        print(f"\nOptimizing hyperparameters for {model_name}...")
        grid_search = GridSearchCV(
            self.models[model_name],
            param_grids[model_name],
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        self.best_model = grid_search.best_estimator_
        self.best_model_name = model_name
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.3f}")

        return grid_search.best_params_

    def evaluate_model(self, X_test, y_test):
        """Evaluate the model on test data."""
        if self.best_model is None:
            raise ValueError("No model has been trained yet.")

        y_pred = self.best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f"\nModel Evaluation ({self.best_model_name}):")
        print(f"Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(report)

        return accuracy, report

    def save_model(self, filepath):
        """Save the trained model to disk."""
        if self.best_model is None:
            raise ValueError("No model has been trained yet.")

        joblib.dump(self.best_model, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a trained model from disk."""
        self.best_model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")

if __name__ == "__main__":
    # Example usage
    from data_preparation import DataPreparator

    # Load and preprocess data
    preparator = DataPreparator("data/weather_data.csv")
    preparator.load_data()
    X_train, X_test, y_train, y_test = preparator.preprocess_data()

    # Train and evaluate model
    classifier = RainfallClassifier()
    classifier.optimize_hyperparameters(X_train, y_train, 'random_forest')
    classifier.evaluate_model(X_test, y_test)
    classifier.save_model('rainfall_model.joblib') 