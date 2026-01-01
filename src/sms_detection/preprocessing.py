"""
SMS Preprocessing Module
Handles text cleaning, tokenization, and feature extraction
"""
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import (URGENCY_KEYWORDS, FINANCIAL_KEYWORDS, 
                        ACTION_KEYWORDS, THREAT_KEYWORDS)

class SMSPreprocessor:
    """
    Comprehensive SMS text preprocessor
    Handles cleaning, normalization, and feature extraction
    """
    
    def __init__(self, use_stemming=True):
        """
        Initialize preprocessor
        
        Args:
            use_stemming (bool): Use stemming instead of lemmatization
        """
        print("Initializing SMS Preprocessor...")
        
        # Load NLTK resources
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
        
        # Initialize stemmer/lemmatizer
        self.use_stemming = use_stemming
        if use_stemming:
            self.stemmer = PorterStemmer()
            print("✓ Using Porter Stemmer")
        else:
            self.lemmatizer = WordNetLemmatizer()
            print("✓ Using WordNet Lemmatizer")
        
        # Compile regex patterns (faster than re-compiling each time)
        self.url_pattern = re.compile(r'http\S+|www\.\S+|https\S+|\S+\.com|\S+\.org|\S+\.net')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.phone_pattern = re.compile(r'\d{10,}|\d{3}[-.\s]?\d{3}[-.\s]?\d{4}')
        self.number_pattern = re.compile(r'\d+')
        self.special_char_pattern = re.compile(r'[^a-zA-Z\s]')
        
        print("✓ Preprocessor initialized successfully\n")
    
    def clean_text(self, text):
        """
        Clean and normalize text
        
        Args:
            text (str): Raw message text
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs (replace with token)
        text = self.url_pattern.sub(' URL ', text)
        
        # Remove email addresses
        text = self.email_pattern.sub(' EMAIL ', text)
        
        # Remove phone numbers (replace with token)
        text = self.phone_pattern.sub(' PHONE ', text)
        
        # Remove other numbers
        text = self.number_pattern.sub(' NUMBER ', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_text(self, text):
        """
        Tokenize text into words
        
        Args:
            text (str): Cleaned text
            
        Returns:
            list: List of tokens
        """
        try:
            tokens = word_tokenize(text)
        except LookupError:
            nltk.download('punkt')
            tokens = word_tokenize(text)
        
        return tokens
    
    def remove_stopwords(self, tokens):
        """
        Remove stopwords from token list
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Filtered tokens
        """
        return [word for word in tokens if word not in self.stop_words and len(word) > 2]
    
    def stem_or_lemmatize(self, tokens):
        """
        Apply stemming or lemmatization
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Processed tokens
        """
        if self.use_stemming:
            return [self.stemmer.stem(word) for word in tokens]
        else:
            try:
                return [self.lemmatizer.lemmatize(word) for word in tokens]
            except LookupError:
                nltk.download('wordnet')
                nltk.download('omw-1.4')
                return [self.lemmatizer.lemmatize(word) for word in tokens]
    
    def preprocess_text(self, text):
        """
        Complete preprocessing pipeline
        
        Args:
            text (str): Raw message text
            
        Returns:
            str: Processed text ready for feature extraction
        """
        # Clean
        cleaned = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize_text(cleaned)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Stem/Lemmatize
        tokens = self.stem_or_lemmatize(tokens)
        
        # Join back to string
        return ' '.join(tokens)
    
    def extract_text_features(self, text):
        """
        Extract statistical features from text
        
        Args:
            text (str): Raw message text
            
        Returns:
            dict: Dictionary of features
        """
        features = {}
        text_lower = text.lower()
        
        # Basic statistics
        features['message_length'] = len(text)
        features['word_count'] = len(text.split())
        features['char_count'] = len(text)
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text else 0
        
        # Special characters
        features['special_char_count'] = sum(1 for char in text if char in string.punctuation)
        features['digit_count'] = sum(1 for char in text if char.isdigit())
        features['uppercase_count'] = sum(1 for char in text if char.isupper())
        
        # Ratios
        if len(text) > 0:
            features['uppercase_ratio'] = features['uppercase_count'] / len(text)
            features['digit_ratio'] = features['digit_count'] / len(text)
            features['special_char_ratio'] = features['special_char_count'] / len(text)
        else:
            features['uppercase_ratio'] = 0
            features['digit_ratio'] = 0
            features['special_char_ratio'] = 0
        
        # Pattern detection
        features['has_url'] = 1 if bool(self.url_pattern.search(text_lower)) else 0
        features['has_email'] = 1 if bool(self.email_pattern.search(text_lower)) else 0
        features['has_phone'] = 1 if bool(self.phone_pattern.search(text)) else 0
        features['has_currency'] = 1 if any(symbol in text for symbol in ['$', '£', '€', 'dollar', 'pound', 'rupee']) else 0
        
        # Keyword counts
        features['urgency_count'] = sum(1 for keyword in URGENCY_KEYWORDS if keyword in text_lower)
        features['financial_count'] = sum(1 for keyword in FINANCIAL_KEYWORDS if keyword in text_lower)
        features['action_count'] = sum(1 for keyword in ACTION_KEYWORDS if keyword in text_lower)
        features['threat_count'] = sum(1 for keyword in THREAT_KEYWORDS if keyword in text_lower)
        
        # Excessive patterns
        features['excessive_caps'] = 1 if features['uppercase_ratio'] > 0.3 else 0
        features['excessive_punctuation'] = 1 if features['special_char_ratio'] > 0.15 else 0
        
        return features
    
    def preprocess_dataset(self, df, text_column='message', label_column='label', verbose=True):
        """
        Preprocess entire dataset
        
        Args:
            df (DataFrame): Input dataframe
            text_column (str): Name of text column
            label_column (str): Name of label column
            verbose (bool): Print progress
            
        Returns:
            DataFrame: Preprocessed dataframe with additional features
        """
        if verbose:
            print("="*60)
            print("PREPROCESSING DATASET")
            print("="*60)
            print(f"\nInput shape: {df.shape}")
        
        # Create copy to avoid modifying original
        df_processed = df.copy()
        
        # 1. Clean text
        if verbose:
            print("\n1. Cleaning text...")
        df_processed['cleaned_text'] = df_processed[text_column].apply(self.clean_text)
        
        # 2. Preprocess for ML
        if verbose:
            print("2. Tokenizing and normalizing...")
        df_processed['processed_text'] = df_processed[text_column].apply(self.preprocess_text)
        
        # 3. Extract features
        if verbose:
            print("3. Extracting features...")
        
        from tqdm import tqdm
        tqdm.pandas(desc="Feature Extraction")
        feature_dicts = df_processed[text_column].progress_apply(self.extract_text_features)
        feature_df = pd.DataFrame(feature_dicts.tolist())
        
        # Combine
        df_processed = pd.concat([df_processed, feature_df], axis=1)
        
        # 4. Encode labels
        if label_column in df_processed.columns:
            if verbose:
                print("4. Encoding labels...")
            df_processed['label_encoded'] = df_processed[label_column].map({'ham': 0, 'spam': 1})
        
        if verbose:
            print(f"\n✓ Preprocessing complete!")
            print(f"Output shape: {df_processed.shape}")
            print(f"New columns added: {df_processed.shape[1] - df.shape[1]}")
            print(f"\nFeature columns: {list(feature_df.columns)}")
        
        return df_processed
    
    def get_feature_names(self):
        """Get list of all extracted feature names"""
        return [
            'message_length', 'word_count', 'char_count', 'avg_word_length',
            'special_char_count', 'digit_count', 'uppercase_count',
            'uppercase_ratio', 'digit_ratio', 'special_char_ratio',
            'has_url', 'has_email', 'has_phone', 'has_currency',
            'urgency_count', 'financial_count', 'action_count', 'threat_count',
            'excessive_caps', 'excessive_punctuation'
        ]


# Test the preprocessor
if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING SMS PREPROCESSOR")
    print("="*60)
    
    from src.config import SMS_RAW_DATA, SMS_PROCESSED_DATA
    
    # Load data
    print("\nLoading dataset...")
    df = pd.read_csv(SMS_RAW_DATA)
    print(f"✓ Loaded {len(df)} messages")
    
    # Initialize preprocessor
    preprocessor = SMSPreprocessor(use_stemming=True)
    
    # Test on single message
    print("\n" + "-"*60)
    print("TESTING ON SAMPLE MESSAGE")
    print("-"*60)
    
    sample_spam = df[df['label']=='spam']['message'].iloc[0]
    print(f"\nOriginal: {sample_spam}")
    print(f"\nCleaned: {preprocessor.clean_text(sample_spam)}")
    print(f"\nProcessed: {preprocessor.preprocess_text(sample_spam)}")
    print(f"\nFeatures: {preprocessor.extract_text_features(sample_spam)}")
    
    # Preprocess entire dataset
    print("\n" + "-"*60)
    print("PREPROCESSING ENTIRE DATASET")
    print("-"*60)
    
    df_processed = preprocessor.preprocess_dataset(df)
    
    # Save
    print(f"\nSaving to: {SMS_PROCESSED_DATA}")
    df_processed.to_csv(SMS_PROCESSED_DATA, index=False)
    print("✓ Saved successfully!")
    
    # Display sample
    print("\n" + "-"*60)
    print("SAMPLE PROCESSED DATA")
    print("-"*60)
    print(df_processed[['message', 'processed_text', 'urgency_count', 'financial_count', 'label']].head())
    
    # Statistics
    print("\n" + "-"*60)
    print("FEATURE STATISTICS")
    print("-"*60)
    print(df_processed[preprocessor.get_feature_names()].describe())
    
    print("\n✓ Preprocessing test complete!")