"""
URL Model Training Script
Trains multiple ML models for URL phishing detection and selects the best one.
Uses URLFeatureExtractor to generate features from raw URL strings.
"""
import pandas as pd
import numpy as np
import joblib
import sys
import os
import warnings
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    f1_score, accuracy_score, precision_score, recall_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import RANDOM_STATE, TEST_SIZE, REPORTS_DIR
from src.url_detection.url_feature_extractor import URLFeatureExtractor

# Try importing XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[WARN] XGBoost not installed. Skipping XGBoost model.")

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = DATA_DIR / "models"

URL_DATA_PATH = RAW_DATA_DIR / "phishing_urls.csv"
URL_MODEL_PATH = MODELS_DIR / "url_classifier.pkl"
URL_FEATURE_NAMES_PATH = MODELS_DIR / "url_feature_names.pkl"
URL_MODEL_INFO_PATH = MODELS_DIR / "url_model_info.pkl"


class URLModelTrainer:
    """
    URL Phishing Detection Model Trainer.
    Trains and evaluates multiple ML models, selects the best by F1-score.
    """

    def __init__(self, random_state=RANDOM_STATE):
        """Initialize trainer with models."""
        self.random_state = random_state
        self.feature_extractor = URLFeatureExtractor()
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Define models
        self._init_models()

    def _init_models(self):
        """Initialize all models to compare."""
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200,
                random_state=self.random_state
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=self.random_state
            ),
        }
        if HAS_XGBOOST:
            self.models['XGBoost'] = XGBClassifier(
                n_estimators=200,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=self.random_state,
                verbosity=0
            )

    def load_data(self):
        """Load URL dataset and extract features."""
        print("\n" + "=" * 60)
        print("LOADING DATA & EXTRACTING FEATURES")
        print("=" * 60)

        # Load raw URLs
        df = pd.read_csv(URL_DATA_PATH)
        print(f"[OK] Loaded {len(df)} URLs")
        print(f"     Columns: {list(df.columns)}")
        print(f"     Label distribution:\n{df['label'].value_counts().to_string()}")

        # Extract features
        print("\nExtracting features from URLs...")
        X = self.feature_extractor.extract_batch(df['url'].tolist())
        y = df['label'].values

        print(f"[OK] Feature matrix shape: {X.shape}")

        # Check class imbalance
        ratio = y.sum() / max((1 - y).sum(), 1)
        print(f"     Imbalance ratio (phishing/legit): {ratio:.2f}")

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=self.random_state, stratify=y
        )
        print(f"     Train: {len(self.X_train)}, Test: {len(self.X_test)}")

        # Handle NaN / Inf
        self.X_train = self.X_train.fillna(0).replace([np.inf, -np.inf], 0)
        self.X_test = self.X_test.fillna(0).replace([np.inf, -np.inf], 0)

        return X, y

    def train_model(self, name, model):
        """Train a single model and return metrics."""
        print(f"\n  Training {name}...")

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='f1')

        # Fit on full training set
        model.fit(self.X_train, self.y_train)

        # Predict on test set
        y_pred = model.predict(self.X_test)

        # Metrics
        acc = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred, zero_division=0)
        rec = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)

        self.results[name] = {
            'model': model,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'cv_f1_mean': cv_scores.mean(),
            'cv_f1_std': cv_scores.std(),
            'y_pred': y_pred,
        }

        print(f"    Accuracy: {acc:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall: {rec:.4f}")
        print(f"    F1-Score: {f1:.4f}")
        print(f"    CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        return self.results[name]

    def train_all_models(self):
        """Train all models."""
        print("\n" + "=" * 60)
        print("TRAINING MODELS")
        print("=" * 60)

        for name, model in self.models.items():
            self.train_model(name, model)

    def select_best_model(self):
        """Select best model by F1-score."""
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)

        # Print comparison table
        print(f"\n{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'CV F1':>12}")
        print("-" * 80)
        for name, res in sorted(self.results.items(), key=lambda x: x[1]['f1'], reverse=True):
            print(f"{name:<25} {res['accuracy']:>10.4f} {res['precision']:>10.4f} "
                  f"{res['recall']:>10.4f} {res['f1']:>10.4f} "
                  f"{res['cv_f1_mean']:>6.4f}±{res['cv_f1_std']:.4f}")

        # Select best
        self.best_model_name = max(self.results, key=lambda k: self.results[k]['f1'])
        self.best_model = self.results[self.best_model_name]['model']

        print(f"\n[OK] Best model: {self.best_model_name} (F1: {self.results[self.best_model_name]['f1']:.4f})")

    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning on the best model."""
        print("\n" + "=" * 60)
        print(f"HYPERPARAMETER TUNING: {self.best_model_name}")
        print("=" * 60)

        param_grids = {
            'Random Forest': {
                'n_estimators': [200, 300],
                'max_depth': [None, 20, 30],
                'min_samples_split': [2, 5],
            },
            'XGBoost': {
                'n_estimators': [200, 300],
                'max_depth': [5, 7, 10],
                'learning_rate': [0.05, 0.1],
            },
            'Gradient Boosting': {
                'n_estimators': [200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.05, 0.1],
            },
            'Logistic Regression': {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l2'],
            },
        }

        if self.best_model_name not in param_grids:
            print("[SKIP] No parameter grid defined for this model.")
            return

        grid = param_grids[self.best_model_name]
        print(f"  Parameter grid: {grid}")

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        search = GridSearchCV(
            self.best_model, grid, cv=cv, scoring='f1',
            n_jobs=-1, verbose=0, refit=True
        )
        search.fit(self.X_train, self.y_train)

        # Update best model
        self.best_model = search.best_estimator_
        y_pred = self.best_model.predict(self.X_test)
        new_f1 = f1_score(self.y_test, y_pred, zero_division=0)

        print(f"  Best params: {search.best_params_}")
        print(f"  Tuned F1: {new_f1:.4f} (was {self.results[self.best_model_name]['f1']:.4f})")

        # Update results
        self.results[self.best_model_name]['f1'] = new_f1
        self.results[self.best_model_name]['model'] = self.best_model
        self.results[self.best_model_name]['y_pred'] = y_pred
        self.results[self.best_model_name]['accuracy'] = accuracy_score(self.y_test, y_pred)
        self.results[self.best_model_name]['precision'] = precision_score(self.y_test, y_pred, zero_division=0)
        self.results[self.best_model_name]['recall'] = recall_score(self.y_test, y_pred, zero_division=0)

    def plot_confusion_matrix(self):
        """Plot confusion matrix for the best model."""
        os.makedirs(str(REPORTS_DIR), exist_ok=True)

        y_pred = self.results[self.best_model_name]['y_pred']
        cm = confusion_matrix(self.y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Legitimate', 'Phishing'],
                    yticklabels=['Legitimate', 'Phishing'])
        plt.title(f'URL Detection — {self.best_model_name} Confusion Matrix',
                  fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        save_path = REPORTS_DIR / 'url_confusion_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved: {save_path}")

    def plot_roc_curve(self):
        """Plot ROC curve for all models."""
        os.makedirs(str(REPORTS_DIR), exist_ok=True)

        plt.figure(figsize=(10, 8))

        for name, res in self.results.items():
            model = res['model']
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(self.X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_proba = model.decision_function(self.X_test)
            else:
                continue

            fpr, tpr, _ = roc_curve(self.y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('URL Detection — ROC Curves', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = REPORTS_DIR / 'url_roc_curve.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved: {save_path}")

    def plot_feature_importance(self):
        """Plot top 15 feature importances for the best model."""
        os.makedirs(str(REPORTS_DIR), exist_ok=True)

        feature_names = self.feature_extractor.get_feature_names()

        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            importances = np.abs(self.best_model.coef_[0])
        else:
            print("[SKIP] Model does not expose feature importances.")
            return

        # Top 15
        indices = np.argsort(importances)[::-1][:15]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]

        plt.figure(figsize=(12, 8))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
        plt.barh(range(len(top_features)), top_importances[::-1], color=colors[::-1])
        plt.yticks(range(len(top_features)),
                   [f.replace('_', ' ').title() for f in reversed(top_features)])
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'URL Detection — Top 15 Feature Importances ({self.best_model_name})',
                  fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()

        save_path = REPORTS_DIR / 'url_feature_importance.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved: {save_path}")

    def generate_classification_report(self):
        """Print and save classification report."""
        y_pred = self.results[self.best_model_name]['y_pred']

        report = classification_report(
            self.y_test, y_pred,
            target_names=['Legitimate', 'Phishing'],
            digits=4
        )

        print("\n" + "=" * 60)
        print(f"CLASSIFICATION REPORT — {self.best_model_name}")
        print("=" * 60)
        print(report)

        # Save to file
        report_path = REPORTS_DIR / 'url_classification_report.txt'
        with open(report_path, 'w') as f:
            f.write(f"Classification Report — {self.best_model_name}\n")
            f.write("=" * 60 + "\n")
            f.write(report)
        print(f"[OK] Saved: {report_path}")

    def save_model(self):
        """Save the best model, feature names, and metadata."""
        os.makedirs(str(MODELS_DIR), exist_ok=True)

        # Save model
        joblib.dump(self.best_model, URL_MODEL_PATH)
        print(f"[OK] Model saved: {URL_MODEL_PATH}")

        # Save feature names
        feature_names = self.feature_extractor.get_feature_names()
        joblib.dump(feature_names, URL_FEATURE_NAMES_PATH)
        print(f"[OK] Feature names saved: {URL_FEATURE_NAMES_PATH}")

        # Save metadata
        res = self.results[self.best_model_name]
        info = {
            'model_name': self.best_model_name,
            'accuracy': res['accuracy'],
            'precision': res['precision'],
            'recall': res['recall'],
            'f1': res['f1'],
            'training_date': datetime.now().isoformat(),
            'num_features': len(feature_names),
            'feature_names': feature_names,
        }
        joblib.dump(info, URL_MODEL_INFO_PATH)
        print(f"[OK] Model info saved: {URL_MODEL_INFO_PATH}")


def main():
    """Main training pipeline."""
    print("\n" + "=" * 70)
    print("URL PHISHING DETECTION — MODEL TRAINING")
    print("=" * 70)

    trainer = URLModelTrainer()

    # Step 1: Load data & extract features
    trainer.load_data()

    # Step 2: Train all models
    trainer.train_all_models()

    # Step 3: Select best model
    trainer.select_best_model()

    # Step 4: Hyperparameter tuning
    trainer.hyperparameter_tuning()

    # Step 5: Generate reports
    trainer.generate_classification_report()
    trainer.plot_confusion_matrix()
    trainer.plot_roc_curve()
    trainer.plot_feature_importance()

    # Step 6: Save model
    trainer.save_model()

    print("\n" + "=" * 70)
    print("[OK] URL MODEL TRAINING COMPLETE!")
    print("=" * 70)
    print(f"  Best model: {trainer.best_model_name}")
    print(f"  F1-Score:   {trainer.results[trainer.best_model_name]['f1']:.4f}")
    print(f"  Accuracy:   {trainer.results[trainer.best_model_name]['accuracy']:.4f}")
    print(f"  Model file: {URL_MODEL_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
