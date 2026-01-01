"""
SMS Model Training Script
Trains multiple ML models and selects the best one
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, classification_report, confusion_matrix,
                            roc_auc_score, roc_curve)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import (SMS_FEATURES_DATA, SMS_MODEL_PATH, REPORTS_DIR, 
                        RANDOM_STATE, TEST_SIZE)

class SMSModelTrainer:
    """
    SMS Phishing Detection Model Trainer
    Trains and evaluates multiple ML models
    """
    
    def __init__(self, random_state=RANDOM_STATE):
        """Initialize trainer with models"""
        self.random_state = random_state
        
        # Initialize models
        self.models = {
            'Naive Bayes': MultinomialNB(),
            'Logistic Regression': LogisticRegression(
                max_iter=1000, 
                random_state=random_state,
                solver='liblinear'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=random_state,
                n_jobs=-1
            ),
            'SVM': SVC(
                kernel='linear',
                probability=True,
                random_state=random_state
            )
        }
        
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """Load and split dataset"""
        print("\n" + "="*70)
        print("LOADING DATASET")
        print("="*70)
        
        # Load features
        df = pd.read_csv(SMS_FEATURES_DATA)
        print(f"✓ Loaded {len(df)} samples")
        
        # Separate features and labels
        X = df.drop('label_encoded', axis=1)
        y = df['label_encoded']
        
        # Handle any remaining NaN values
        X = X.fillna(0)
        
        # Convert negative values to positive for Naive Bayes
        X = X.abs()
        
        print(f"✓ Features shape: {X.shape}")
        print(f"✓ Labels shape: {y.shape}")
        print(f"\nClass distribution:")
        print(f"  - Ham (0): {sum(y==0)} ({sum(y==0)/len(y)*100:.1f}%)")
        print(f"  - Spam (1): {sum(y==1)} ({sum(y==1)/len(y)*100:.1f}%)")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=self.random_state, stratify=y
        )
        
        print(f"\n✓ Training set: {len(self.X_train)} samples")
        print(f"✓ Test set: {len(self.X_test)} samples")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self, name, model):
        """Train a single model and return metrics"""
        print(f"\n{'─'*70}")
        print(f"Training: {name}")
        print(f"{'─'*70}")
        
        # Train
        model.fit(self.X_train, self.y_train)
        
        # Predict
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        
        # ROC AUC
        if y_pred_proba is not None:
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        else:
            roc_auc = None
        
        # Cross-validation
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='f1')
        
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        if roc_auc:
            print(f"ROC AUC:   {roc_auc:.4f}")
        print(f"CV F1:     {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'cv_scores': cv_scores,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def train_all_models(self):
        """Train all models"""
        print("\n" + "="*70)
        print("TRAINING ALL MODELS")
        print("="*70)
        
        for name, model in self.models.items():
            try:
                self.results[name] = self.train_model(name, model)
            except Exception as e:
                print(f"✗ Error training {name}: {e}")
                continue
        
        return self.results
    
    def compare_models(self):
        """Compare all trained models"""
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        
        # Create comparison dataframe
        comparison = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [r['accuracy'] for r in self.results.values()],
            'Precision': [r['precision'] for r in self.results.values()],
            'Recall': [r['recall'] for r in self.results.values()],
            'F1-Score': [r['f1_score'] for r in self.results.values()]
        })
        
        comparison = comparison.sort_values('F1-Score', ascending=False)
        print("\n", comparison.to_string(index=False))
        
        # Plot comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        for idx, (ax, metric, color) in enumerate(zip(axes.flat, metrics, colors)):
            data = comparison.sort_values(metric, ascending=True)
            ax.barh(data['Model'], data[metric], color=color, alpha=0.7)
            ax.set_xlabel(metric, fontsize=12, fontweight='bold')
            ax.set_xlim(0, 1)
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, v in enumerate(data[metric]):
                ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10)
        
        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Comparison plot saved to: {REPORTS_DIR / 'model_comparison.png'}")
        
        return comparison
    
    def select_best_model(self):
        """Select best model based on F1-score"""
        best_f1 = 0
        
        for name, result in self.results.items():
            if result['f1_score'] > best_f1:
                best_f1 = result['f1_score']
                self.best_model_name = name
                self.best_model = result['model']
        
        print("\n" + "="*70)
        print("BEST MODEL SELECTED")
        print("="*70)
        print(f"Model: {self.best_model_name}")
        print(f"F1-Score: {best_f1:.4f}")
        print("="*70)
        
        return self.best_model, self.best_model_name
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        n_models = len(self.results)
        n_cols = 2
        n_rows = (n_models + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5*n_rows))
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for idx, (name, result) in enumerate(self.results.items()):
            cm = confusion_matrix(self.y_test, result['predictions'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Ham', 'Spam'],
                       yticklabels=['Ham', 'Spam'],
                       cbar_kws={'label': 'Count'})
            
            axes[idx].set_title(f'{name}\nF1-Score: {result["f1_score"]:.4f}', 
                               fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Actual', fontsize=10)
            axes[idx].set_xlabel('Predicted', fontsize=10)
        
        # Remove empty subplots
        for idx in range(n_models, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / 'confusion_matrices_all.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Confusion matrices saved to: {REPORTS_DIR / 'confusion_matrices_all.png'}")
    
    def generate_classification_report(self):
        """Generate detailed classification report for best model"""
        print("\n" + "="*70)
        print(f"CLASSIFICATION REPORT - {self.best_model_name}")
        print("="*70)
        
        result = self.results[self.best_model_name]
        report = classification_report(
            self.y_test, 
            result['predictions'],
            target_names=['Ham', 'Spam'],
            digits=4
        )
        
        print(report)
        
        # Save to file
        report_path = REPORTS_DIR / 'classification_report.txt'
        with open(report_path, 'w') as f:
            f.write(f"Classification Report - {self.best_model_name}\n")
            f.write("="*70 + "\n")
            f.write(report)
        
        print(f"\n✓ Report saved to: {report_path}")
    
    def plot_roc_curves(self):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for name, result in self.results.items():
            if result['probabilities'] is not None:
                fpr, tpr, _ = roc_curve(self.y_test, result['probabilities'])
                plt.plot(fpr, tpr, label=f"{name} (AUC={result['roc_auc']:.3f})", linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curves - All Models', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ ROC curves saved to: {REPORTS_DIR / 'roc_curves.png'}")
    
    def save_best_model(self, filepath=SMS_MODEL_PATH):
        """Save the best model"""
        if self.best_model is None:
            print("✗ No model to save!")
            return
        
        # Save model
        joblib.dump(self.best_model, filepath)
        
        # Save model info
        model_info = {
            'name': self.best_model_name,
            'accuracy': self.results[self.best_model_name]['accuracy'],
            'f1_score': self.results[self.best_model_name]['f1_score'],
            'precision': self.results[self.best_model_name]['precision'],
            'recall': self.results[self.best_model_name]['recall'],
        }
        
        info_path = filepath.parent / 'sms_model_info.pkl'
        joblib.dump(model_info, info_path)
        
        print("\n" + "="*70)
        print("MODEL SAVED")
        print("="*70)
        print(f"✓ Model: {filepath}")
        print(f"✓ Info: {info_path}")
        print("="*70)
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning on best model"""
        print("\n" + "="*70)
        print(f"HYPERPARAMETER TUNING - {self.best_model_name}")
        print("="*70)
        
        if self.best_model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
            
        elif self.best_model_name == 'Logistic Regression':
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
            model = LogisticRegression(max_iter=1000, random_state=self.random_state)
            
        else:
            print(f"Hyperparameter tuning not configured for {self.best_model_name}")
            return None
        
        print("Starting grid search... (this may take a few minutes)")
        
        grid_search = GridSearchCV(
            model, 
            param_grid, 
            cv=5, 
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"\n✓ Best parameters: {grid_search.best_params_}")
        print(f"✓ Best CV F1-score: {grid_search.best_score_:.4f}")
        
        # Evaluate tuned model
        y_pred = grid_search.predict(self.X_test)
        tuned_f1 = f1_score(self.y_test, y_pred)
        
        print(f"✓ Test F1-score: {tuned_f1:.4f}")
        
        if tuned_f1 > self.results[self.best_model_name]['f1_score']:
            print("\n🎉 Tuned model is better! Updating best model...")
            self.best_model = grid_search.best_estimator_
            self.results[self.best_model_name]['f1_score'] = tuned_f1
        
        return grid_search.best_estimator_


def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print("SMS PHISHING DETECTION - MODEL TRAINING")
    print("="*70)
    
    # Create reports directory
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer
    trainer = SMSModelTrainer()
    
    # Load data
    trainer.load_data()
    
    # Train all models
    trainer.train_all_models()
    
    # Compare models
    trainer.compare_models()
    
    # Plot confusion matrices
    trainer.plot_confusion_matrices()
    
    # Plot ROC curves
    trainer.plot_roc_curves()
    
    # Select best model
    trainer.select_best_model()
    
    # Generate classification report
    trainer.generate_classification_report()
    
    # Hyperparameter tuning (optional)
    print("\n" + "="*70)
    response = input("Perform hyperparameter tuning? (y/n): ").lower()
    if response == 'y':
        trainer.hyperparameter_tuning()
    
    # Save best model
    trainer.save_best_model()
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print(f"  1. Model: {SMS_MODEL_PATH}")
    print(f"  2. Model comparison: {REPORTS_DIR / 'model_comparison.png'}")
    print(f"  3. Confusion matrices: {REPORTS_DIR / 'confusion_matrices_all.png'}")
    print(f"  4. ROC curves: {REPORTS_DIR / 'roc_curves.png'}")
    print(f"  5. Classification report: {REPORTS_DIR / 'classification_report.txt'}")
    print("\n✓ Ready for deployment!")


if __name__ == "__main__":
    main()