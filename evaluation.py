import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
    log_loss
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import shap

class ModelEvaluator:
    def __init__(self, model, model_name):
        """Initialize the ModelEvaluator with a trained model."""
        self.model = model
        self.model_name = model_name
        self.calibrated_model = None

    def plot_confusion_matrix(self, y_true, y_pred, save_path='confusion_matrix.png'):
        """Plot and save the confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(save_path)
        plt.close()

    def plot_roc_curve(self, y_true, y_pred_proba, save_path='roc_curve.png'):
        """Plot and save the ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic - {self.model_name}')
        plt.legend(loc="lower right")
        plt.savefig(save_path)
        plt.close()

    def plot_precision_recall_curve(self, y_true, y_pred_proba, save_path='precision_recall_curve.png'):
        """Plot and save the Precision-Recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        average_precision = average_precision_score(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {average_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {self.model_name}')
        plt.legend(loc="lower left")
        plt.savefig(save_path)
        plt.close()

    def plot_feature_importance(self, feature_names, save_path='feature_importance.png'):
        """Plot and save feature importance if the model supports it."""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]

            plt.figure(figsize=(10, 6))
            plt.title(f'Feature Importances - {self.model_name}')
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
        else:
            print("This model does not support feature importance visualization.")

    def calibrate_model(self, X_train, y_train):
        """Calibrate the model's probability estimates."""
        self.calibrated_model = CalibratedClassifierCV(self.model, cv=5)
        self.calibrated_model.fit(X_train, y_train)
        print("Model calibration completed.")

    def plot_calibration_curve(self, y_true, y_pred_proba, save_path='calibration_curve.png'):
        """Plot and save the calibration curve."""
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=10
        )

        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-",
                label=f"{self.model_name}")
        plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        plt.ylabel("Fraction of positives")
        plt.xlabel("Mean predicted value")
        plt.title(f"Calibration Curve - {self.model_name}")
        plt.legend(loc="lower right")
        plt.savefig(save_path)
        plt.close()

    def plot_prediction_distribution(self, y_true, y_pred_proba, save_path='prediction_distribution.png'):
        """Plot the distribution of predicted probabilities."""
        plt.figure(figsize=(10, 6))
        for label in [0, 1]:
            mask = y_true == label
            plt.hist(y_pred_proba[mask], bins=50, alpha=0.5,
                    label=f'Class {label}', density=True)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title(f'Prediction Distribution - {self.model_name}')
        plt.legend()
        plt.savefig(save_path)
        plt.close()

    def plot_shap_values(self, X_test, feature_names, save_path='shap_values.png'):
        """Plot SHAP values for model interpretation."""
        try:
            # Create SHAP explainer
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_test)

            # Plot summary
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                            show=False, plot_size=(12, 8))
            plt.title(f'SHAP Values - {self.model_name}')
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
        except Exception as e:
            print(f"Could not generate SHAP values: {str(e)}")

    def calculate_advanced_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate advanced evaluation metrics."""
        metrics = {
            'Brier Score': brier_score_loss(y_true, y_pred_proba),
            'Log Loss': log_loss(y_true, y_pred_proba),
            'AUC-ROC': roc_curve(y_true, y_pred_proba)[1].mean(),
            'Average Precision': average_precision_score(y_true, y_pred_proba)
        }
        
        print("\nAdvanced Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics

    def generate_evaluation_report(self, X_test, y_test, feature_names=None):
        """Generate a comprehensive evaluation report."""
        # Get predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Generate all plots
        self.plot_confusion_matrix(y_test, y_pred)
        self.plot_roc_curve(y_test, y_pred_proba)
        self.plot_precision_recall_curve(y_test, y_pred_proba)
        self.plot_calibration_curve(y_test, y_pred_proba)
        self.plot_prediction_distribution(y_test, y_pred_proba)
        
        if feature_names is not None:
            self.plot_feature_importance(feature_names)
            self.plot_shap_values(X_test, feature_names)

        # Calculate advanced metrics
        self.calculate_advanced_metrics(y_test, y_pred, y_pred_proba)

        print(f"\nEvaluation Report for {self.model_name}")
        print("=" * 50)
        print("Plots have been saved as:")
        print("- confusion_matrix.png")
        print("- roc_curve.png")
        print("- precision_recall_curve.png")
        print("- calibration_curve.png")
        print("- prediction_distribution.png")
        if feature_names is not None:
            print("- feature_importance.png")
            print("- shap_values.png")

if __name__ == "__main__":
    # Example usage
    from data_preparation import DataPreparator
    from model import RainfallClassifier

    # Load and preprocess data
    preparator = DataPreparator("data/weather_data.csv")
    preparator.load_data()
    X_train, X_test, y_train, y_test = preparator.preprocess_data()

    # Train model
    classifier = RainfallClassifier()
    classifier.train_model(X_train, y_train, 'random_forest')

    # Evaluate model
    evaluator = ModelEvaluator(classifier.best_model, classifier.best_model_name)
    evaluator.generate_evaluation_report(X_test, y_test, feature_names=preparator.data.columns[:-1]) 