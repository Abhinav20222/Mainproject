
"""
Feature Extraction Module
Extracts TF-IDF features and combines with statistical features
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import joblib
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import (MAX_TFIDF_FEATURES, NGRAM_RANGE, MIN_DF, MAX_DF,
                        FEATURE_EXTRACTOR_PATH, SMS_FEATURES_DATA)

class FeatureExtractor:
    """
    Feature extraction class combining TF-IDF and statistical features
    """
    
    def __init__(self, max_features=MAX_TFIDF_FEATURES, ngram_range=NGRAM_RANGE):
        """
        Initialize feature extractor
        
        Args:
            max_features (int): Maximum number of TF-IDF features
            ngram_range (tuple): N-gram range for TF-IDF
        """
        print("Initializing Feature Extractor...")
        
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=MIN_DF,
            max_df=MAX_DF,
            lowercase=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{2,}',  # Words with at least 2 characters
            stop_words=None  # Already removed in preprocessing
        )
        
        self.scaler = StandardScaler()
        self.feature_names = None
        self.numerical_features = None
        
        print(f"✓ TF-IDF Vectorizer configured:")
        print(f"  - Max features: {max_features}")
        print(f"  - N-gram range: {ngram_range}")
        print(f"  - Min DF: {MIN_DF}")
        print(f"  - Max DF: {MAX_DF}\n")
    
    def fit_transform(self, df, text_column='processed_text'):
        """
        Fit and transform training data
        """
        print("="*60)
        print("FITTING AND TRANSFORMING FEATURES")
        print("="*60)
        
        # FIX: Fill NaN values with empty string before vectorization
        text_data = df[text_column].fillna('').astype(str)

        # 1. TF-IDF Features
        print("\n1. Extracting TF-IDF features...")
        # Use the sanitized text_data instead of df[text_column] directly
        tfidf_features = self.tfidf.fit_transform(text_data)
        print(f"   ✓ Shape: {tfidf_features.shape}")
        
        # Convert to DataFrame
        tfidf_df = pd.DataFrame(
            tfidf_features.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
        )
        
        # ... (rest of the method remains the same)
        # 2. Statistical Features
        print("\n2. Extracting statistical features...")
        self.numerical_features = [
            'message_length', 'word_count', 'avg_word_length',
            'special_char_count', 'digit_count', 'uppercase_count',
            'uppercase_ratio', 'digit_ratio', 'special_char_ratio',
            'has_url', 'has_email', 'has_phone', 'has_currency',
            'urgency_count', 'financial_count', 'action_count', 'threat_count',
            'excessive_caps', 'excessive_punctuation'
        ]
        
        # Check which features exist in dataframe
        available_features = [f for f in self.numerical_features if f in df.columns]
        missing_features = [f for f in self.numerical_features if f not in df.columns]
        
        if missing_features:
            print(f"   ⚠ Warning: Missing features: {missing_features}")
        
        numerical_data = df[available_features].values
        numerical_scaled = self.scaler.fit_transform(numerical_data)
        
        numerical_df = pd.DataFrame(
            numerical_scaled,
            columns=[f'num_{col}' for col in available_features]
        )
        print(f"   ✓ Shape: {numerical_df.shape}")
        
        # 3. Combine features
        print("\n3. Combining features...")
        features_df = pd.concat([tfidf_df, numerical_df], axis=1)
        
        self.feature_names = list(features_df.columns)
        
        print(f"   ✓ Final feature matrix shape: {features_df.shape}")
        print(f"   ✓ Total features: {len(self.feature_names)}")
        
        return features_df
    
    def transform(self, df, text_column='processed_text'):
        """
        Transform new data using fitted extractors
        """
        print("\nTransforming new data...")
        
        # FIX: Fill NaN values here as well
        text_data = df[text_column].fillna('').astype(str)

        # 1. TF-IDF Features
        tfidf_features = self.tfidf.transform(text_data)
        tfidf_df = pd.DataFrame(
            tfidf_features.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
        )
        
        # ... (rest of the method remains the same)
        
        # 2. Statistical Features
        available_features = [f for f in self.numerical_features if f in df.columns]
        numerical_data = df[available_features].values
        numerical_scaled = self.scaler.transform(numerical_data)
        
        numerical_df = pd.DataFrame(
            numerical_scaled,
            columns=[f'num_{col}' for col in available_features]
        )
        
        # 3. Combine
        features_df = pd.concat([tfidf_df, numerical_df], axis=1)
        
        print(f"✓ Transformed shape: {features_df.shape}")
        
        return features_df
    
    def get_top_tfidf_terms(self, n=20):
        """
        Get top N TF-IDF terms
        
        Args:
            n (int): Number of terms to return
            
        Returns:
            list: Top terms
        """
        if hasattr(self.tfidf, 'vocabulary_'):
            feature_names = self.tfidf.get_feature_names_out()
            return list(feature_names[:n])
        else:
            return []
    
    def get_feature_importance_by_class(self, df, label_column='label_encoded'):
        """
        Calculate mean feature values by class
        
        Args:
            df (DataFrame): Feature dataframe with labels
            label_column (str): Label column name
            
        Returns:
            DataFrame: Feature importance by class
        """
        print("\n" + "="*60)
        print("CALCULATING FEATURE IMPORTANCE BY CLASS")
        print("="*60)
        
        # Separate features and labels
        feature_cols = [col for col in df.columns if col != label_column]
        X = df[feature_cols]
        y = df[label_column]
        
        # Calculate mean by class
        ham_means = X[y == 0].mean()
        spam_means = X[y == 1].mean()
        
        # Calculate difference
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'ham_mean': ham_means.values,
            'spam_mean': spam_means.values,
            'difference': (spam_means - ham_means).values,
            'abs_difference': np.abs(spam_means - ham_means).values
        })
        
        # Sort by absolute difference
        importance_df = importance_df.sort_values('abs_difference', ascending=False)
        
        return importance_df
    
    def save(self, filepath=FEATURE_EXTRACTOR_PATH):
        """Save feature extractor"""
        joblib.dump(self, filepath)
        print(f"\n✓ Feature extractor saved to: {filepath}")
    
    @staticmethod
    def load(filepath=FEATURE_EXTRACTOR_PATH):
        """Load feature extractor"""
        extractor = joblib.load(filepath)
        print(f"✓ Feature extractor loaded from: {filepath}")
        return extractor


# Test the feature extractor
if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING FEATURE EXTRACTOR")
    print("="*60)
    
    from src.config import SMS_PROCESSED_DATA
    
    # Load preprocessed data
    print("\nLoading preprocessed data...")
    df = pd.read_csv(SMS_PROCESSED_DATA)
    print(f"✓ Loaded {len(df)} messages")
    print(f"Columns: {list(df.columns)}")
    
    # Initialize extractor
    extractor = FeatureExtractor(max_features=500)
    
    # Extract features
    features_df = extractor.fit_transform(df)
    
    # Add labels
    features_df['label_encoded'] = df['label_encoded'].values
    
    # Display sample
    print("\n" + "-"*60)
    print("SAMPLE FEATURES")
    print("-"*60)
    print(features_df.head())
    
    # Get top TF-IDF terms
    print("\n" + "-"*60)
    print("TOP 20 TF-IDF TERMS")
    print("-"*60)
    top_terms = extractor.get_top_tfidf_terms(20)
    for i, term in enumerate(top_terms, 1):
        print(f"{i:2d}. {term}")
    
    # Feature importance
    importance_df = extractor.get_feature_importance_by_class(features_df)
    print("\n" + "-"*60)
    print("TOP 20 MOST DISCRIMINATIVE FEATURES")
    print("-"*60)
    print(importance_df.head(20).to_string())
    
    # Save features
    print("\n" + "-"*60)
    print("SAVING FEATURES")
    print("-"*60)
    features_df.to_csv(SMS_FEATURES_DATA, index=False)
    print(f"✓ Features saved to: {SMS_FEATURES_DATA}")
    
    # Save extractor
    extractor.save()
    
    # Test loading
    print("\n" + "-"*60)
    print("TESTING LOAD")
    print("-"*60)
    loaded_extractor = FeatureExtractor.load()
    print("✓ Successfully loaded feature extractor!")
    
    print("\n✓ Feature extraction test complete!")