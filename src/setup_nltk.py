"""
NLTK Data Downloader
Downloads all required NLTK datasets for text preprocessing
"""
import nltk
import ssl

# Fix SSL certificate issue (common on Windows)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def download_nltk_data():
    """Download all required NLTK datasets"""
    
    datasets = [
        'punkt',        # Tokenizer
        'stopwords',    # Stop words
        'wordnet',      # Lemmatization
        'omw-1.4',      # Multilingual wordnet
        'averaged_perceptron_tagger',  # POS tagging
    ]
    
    print("Downloading NLTK datasets...")
    print("="*50)
    
    for dataset in datasets:
        try:
            print(f"Downloading {dataset}...", end=" ")
            nltk.download(dataset, quiet=True)
            print("✓ Done")
        except Exception as e:
            print(f"✗ Failed: {e}")
    
    print("="*50)
    print("NLTK setup complete!")
    
    # Verify downloads
    print("\nVerifying downloads...")
    try:
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        from nltk.stem import WordNetLemmatizer
        
        # Test
        test_text = "This is a test sentence for verification"
        tokens = word_tokenize(test_text)
        print(f"✓ Tokenization works: {tokens[:3]}...")
        
        stop_words = stopwords.words('english')
        print(f"✓ Stopwords loaded: {len(stop_words)} words")
        
        print("\n✓ All verifications passed!")
        
    except Exception as e:
        print(f"✗ Verification failed: {e}")

if __name__ == "__main__":
    download_nltk_data()