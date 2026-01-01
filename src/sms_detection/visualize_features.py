
"""
Feature Visualization Script
Creates comprehensive visualizations of extracted features
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import SMS_FEATURES_DATA, REPORTS_DIR

def plot_feature_distributions(df, features, save_path):
    """Plot distribution of features by class"""
    
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    for idx, feature in enumerate(features):
        if feature in df.columns:
            # Get data by class
            ham_data = df[df['label_encoded']==0][feature]
            spam_data = df[df['label_encoded']==1][feature]
            
            # Plot
            axes[idx].hist(ham_data, bins=30, alpha=0.6, label='Ham', color='#2ecc71', density=True)
            axes[idx].hist(spam_data, bins=30, alpha=0.6, label='Spam', color='#e74c3c', density=True)
            axes[idx].set_xlabel(feature.replace('num_', '').replace('_', ' ').title())
            axes[idx].set_ylabel('Density')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
    
    # Remove empty subplots
    for idx in range(n_features, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")

def plot_correlation_matrix(df, save_path):
    """Plot correlation matrix of features"""
    
    # Select numerical features
    numerical_cols = [col for col in df.columns if col.startswith('num_')]
    
    if len(numerical_cols) > 0:
        # Calculate correlation
        corr = df[numerical_cols].corr()
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_path}")

def plot_top_tfidf_terms(df, n_terms=20, save_path=None):
    """Plot top TF-IDF terms by class"""
    
    # Get TF-IDF columns
    tfidf_cols = [col for col in df.columns if col.startswith('tfidf_')]
    
    if len(tfidf_cols) > 0:
        # Calculate mean TF-IDF values by class
        ham_means = df[df['label_encoded']==0][tfidf_cols].mean()
        spam_means = df[df['label_encoded']==1][tfidf_cols].mean()
        
        # Get top terms for each class
        top_ham = ham_means.nlargest(n_terms)
        top_spam = spam_means.nlargest(n_terms)
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Ham
        top_ham.plot(kind='barh', ax=axes[0], color='#2ecc71')
        axes[0].set_title('Top TF-IDF Terms in HAM Messages', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Mean TF-IDF Score')
        axes[0].invert_yaxis()
        
        # Spam
        top_spam.plot(kind='barh', ax=axes[1], color='#e74c3c')
        axes[1].set_title('Top TF-IDF Terms in SPAM Messages', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Mean TF-IDF Score')
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved: {save_path}")
        else:
            plt.show()

def plot_feature_importance_comparison(df, top_n=15, save_path=None):
    """Compare feature values between ham and spam"""
    
    numerical_cols = [col for col in df.columns if col.startswith('num_')]
    
    if len(numerical_cols) > 0:
        # Calculate means
        ham_means = df[df['label_encoded']==0][numerical_cols].mean()
        spam_means = df[df['label_encoded']==1][numerical_cols].mean()
        
        # Calculate difference
        diff = (spam_means - ham_means).abs().sort_values(ascending=False)
        top_features = diff.head(top_n).index
        
        # Prepare data for plotting
        data = pd.DataFrame({
            'Ham': ham_means[top_features].values,
            'Spam': spam_means[top_features].values
        }, index=[f.replace('num_', '').replace('_', ' ').title() for f in top_features])
        
        # Plot
        ax = data.plot(kind='barh', figsize=(12, 8), color=['#2ecc71', '#e74c3c'], alpha=0.8)
        ax.set_title(f'Top {top_n} Most Discriminative Features', fontsize=14, fontweight='bold')
        ax.set_xlabel('Normalized Feature Value')
        ax.legend(title='Class', fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved: {save_path}")
        else:
            plt.show()

def main():
    """Generate all visualizations"""
    
    print("\n" + "="*60)
    print("FEATURE VISUALIZATION")
    print("="*60)
    
    # Load features
    print("\nLoading features...")
    df = pd.read_csv(SMS_FEATURES_DATA)
    print(f"✓ Loaded {len(df)} samples with {len(df.columns)} features")
    
    # Create reports directory
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Feature distributions
    print("\n1. Plotting feature distributions...")
    important_features = [
        'num_urgency_count', 'num_financial_count', 'num_action_count',
        'num_has_url', 'num_message_length', 'num_uppercase_ratio'
    ]
    plot_feature_distributions(
        df, important_features,
        REPORTS_DIR / 'feature_distributions.png'
    )
    
    # 2. Correlation matrix
    print("\n2. Plotting correlation matrix...")
    plot_correlation_matrix(df, REPORTS_DIR / 'correlation_matrix.png')
    
    # 3. Top TF-IDF terms
    print("\n3. Plotting top TF-IDF terms...")
    plot_top_tfidf_terms(df, n_terms=15, save_path=REPORTS_DIR / 'top_tfidf_terms.png')
    
    # 4. Feature importance comparison
    print("\n4. Plotting feature importance...")
    plot_feature_importance_comparison(df, top_n=15, save_path=REPORTS_DIR / 'feature_importance.png')
    
    print("\n" + "="*60)
    print("✓ ALL VISUALIZATIONS COMPLETE!")
    print("="*60)
    print(f"\nVisualizations saved in: {REPORTS_DIR}")
    print("Files created:")
    print("  1. feature_distributions.png")
    print("  2. correlation_matrix.png")
    print("  3. top_tfidf_terms.png")
    print("  4. feature_importance.png")

if __name__ == "__main__":
    main()