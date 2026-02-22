
# Comprehensive Algorithm Explanation — PhishGuard AI

> **A Complete Mathematical & Technical Deep-Dive into Every Algorithm, Data Transformation, and Decision Mechanism in the System**

---

## Table of Contents
1. [System-Level Algorithm (End-to-End Pipeline)](#1-system-level-algorithm)
2. [Data Flow & Transformations](#2-data-flow--transformations)
3. [SMS Detection Algorithm (NLP Pipeline)](#3-sms-detection-algorithm)
4. [TF-IDF Vectorization — Full Mathematical Derivation](#4-tf-idf-vectorization)
5. [URL Detection Algorithm (Structural Analysis Pipeline)](#5-url-detection-algorithm)
6. [Shannon Entropy — Mathematical Derivation](#6-shannon-entropy)
7. [Random Forest Classifier — Complete Internal Algorithm](#7-random-forest-classifier)
8. [How a Node is Declared/Split in Random Forest (Decision Tree Splitting)](#8-how-a-node-is-declaredsplit-in-random-forest)
9. [Visual Forensics Algorithm (pHash + SSIM)](#9-visual-forensics-algorithm)
10. [Weighted Fusion & Final Threat Score Algorithm](#10-weighted-fusion--final-threat-score)
11. [Model Evaluation Metrics — Mathematical Definitions](#11-model-evaluation-metrics)
12. [Cross-Validation Algorithm](#12-cross-validation-algorithm)
13. [Hyperparameter Tuning (GridSearchCV)](#13-hyperparameter-tuning)
14. [Complete Pseudocode — Full System](#14-complete-pseudocode)
15. [Why URL Gets Higher Weight (0.45) Than SMS (0.40) — Justification](#15-why-url-gets-higher-weight)
16. [SSIM — Detailed Step-by-Step Working with Worked Example](#16-ssim-detailed-working)
17. [How Frontend History & Diagrams Work Without a Database](#17-frontend-history-without-database)
18. [Image Storage for Visual Forensics — File System vs Database](#18-image-storage-for-visual-forensics)

---

## 1. System-Level Algorithm

**Algorithm Name:** Multi-Channel Hybrid Phishing Detection Pipeline

The system follows a **5-phase sequential pipeline**:

```
Phase 1: INPUT ACQUISITION
    User enters a message (SMS text) through the React Frontend.

Phase 2: CHANNEL DECOMPOSITION
    The API decomposes the input into 3 independent analysis channels:
        Channel A: SMS Text Content     → NLP + Random Forest
        Channel B: Extracted URL(s)      → Structural Feature Extraction + Random Forest
        Channel C: Visual Screenshot     → pHash + SSIM Comparison

Phase 3: INDEPENDENT CHANNEL ANALYSIS
    Each channel independently produces a threat_score ∈ [0, 1]

Phase 4: WEIGHTED SCORE FUSION
    Final_Score = (0.40 × SMS_Score) + (0.45 × URL_Score) + (0.15 × Visual_Score)
    (If a channel is not available, weights are re-normalized among active channels)

Phase 5: RISK CLASSIFICATION
    If Final_Score < 0.30  → "LOW" risk    (Green)
    If Final_Score < 0.60  → "MEDIUM" risk (Yellow)
    If Final_Score < 0.85  → "HIGH" risk   (Orange)
    If Final_Score ≥ 0.85  → "CRITICAL"    (Red)
```

---

## 2. Data Flow & Transformations

### 2.1 Complete Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                    USER INPUT (React Frontend)                       │
│           "URGENT! Your SBI account blocked. Click                   │
│            http://sbi-secure.xyz/login to verify"                    │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    FLASK REST API (api.py)                            │
│   POST /api/full-scan  receives JSON { "message": "..." }           │
│                                                                      │
│   Step 1: Extract URLs using regex from message text                 │
│           extracted_urls = ["http://sbi-secure.xyz/login"]           │
│   Step 2: Fork into 3 parallel analysis channels                     │
└────┬──────────────────────┬──────────────────────┬───────────────────┘
     │                      │                      │
     ▼                      ▼                      ▼
┌─────────────┐    ┌──────────────┐    ┌───────────────────┐
│ CHANNEL A:  │    │ CHANNEL B:   │    │ CHANNEL C:        │
│ SMS TEXT    │    │ URL STRUCT.  │    │ VISUAL FORENSICS  │
│ ANALYSIS    │    │ ANALYSIS     │    │                   │
└──────┬──────┘    └──────┬───────┘    └───────┬───────────┘
       │                  │                    │
       ▼                  ▼                    ▼
 ┌───────────┐    ┌────────────┐       ┌────────────┐
 │ STRING    │    │ 30 NUMERIC │       │ SCREENSHOT │
 │    ↓      │    │ FEATURES   │       │ via        │
 │ CLEAN     │    │ (floats)   │       │ SELENIUM   │
 │    ↓      │    │            │       │    ↓       │
 │ TOKENIZE  │    │            │       │ pHash      │
 │    ↓      │    │            │       │    ↓       │
 │ STOPWORDS │    │            │       │ SSIM       │
 │    ↓      │    │            │       │            │
 │ STEMMING  │    │            │       │            │
 │    ↓      │    │            │       │            │
 │ TF-IDF    │    │            │       │            │
 │ (500 feat)│    │            │       │            │
 │    +      │    │            │       │            │
 │ 19 STATS  │    │            │       │            │
 │ = 519 feat│    │            │       │            │
 └─────┬─────┘    └─────┬──────┘       └─────┬──────┘
       │                │                     │
       ▼                ▼                     ▼
 ┌───────────┐   ┌────────────┐       ┌────────────┐
 │ RF MODEL  │   │ RF MODEL   │       │ SIMILARITY │
 │ predict   │   │ predict    │       │ SCORE      │
 │ _proba()  │   │ _proba()   │       │ [0, 1]     │
 └─────┬─────┘   └─────┬──────┘       └─────┬──────┘
       │                │                     │
       ▼                ▼                     ▼
   SMS_Score         URL_Score          Visual_Score
   (0 to 1)          (0 to 1)           (0 to 1)
       │                │                     │
       └────────────────┼─────────────────────┘
                        ▼
              ┌──────────────────┐
              │  WEIGHTED FUSION │
              │ 0.40×S + 0.45×U  │
              │    + 0.15×V      │
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │  RISK LEVEL +    │
              │  THREAT SCORE    │
              │  → Frontend      │
              └──────────────────┘
```

### 2.2 Data Type Transformations at Each Stage

| Stage | Input Type | Output Type | Transformation |
|:------|:-----------|:------------|:---------------|
| Raw Input | `string` (human text) | `string` | None |
| Cleaning | `string` | `string` (lowercase, no punctuation) | Regex substitution |
| Tokenization | `string` | `list[string]` (tokens) | `text.split()` |
| Stopword Removal | `list[string]` | `list[string]` (shorter) | Filter against NLTK set |
| Stemming | `list[string]` | `list[string]` (root forms) | PorterStemmer |
| TF-IDF | `string` (joined tokens) | `float[500]` (sparse vector) | Matrix multiplication |
| Statistical Features | `string` (raw) | `float[19]` | Counting + ratios |
| StandardScaler | `float[19]` | `float[19]` (μ=0, σ=1) | `(x - μ) / σ` |
| Feature Merge | `float[500] + float[19]` | `float[519]` | Horizontal concatenation |
| RF Prediction | `float[519]` | `float[2]` probabilities | Ensemble voting |
| Threat Score | `float` (probability) | `int` (0–100) | `× 100` |

---

## 3. SMS Detection Algorithm — Step-by-Step

### Step 1: Text Cleaning (`preprocessing.py → clean_text()`)

```
ALGORITHM: CleanText(raw_message)
    INPUT:  raw_message = "URGENT! Your SBI Account BLOCKED!! Click http://bit.ly/xyz"
    
    1. text = lowercase(raw_message)
       → "urgent! your sbi account blocked!! click http://bit.ly/xyz"
    
    2. text = regex_replace(URLs → "URL")
       → "urgent! your sbi account blocked!! click URL"
    
    3. text = regex_replace(phone_numbers → "PHONE")
       → (no change in this case)
    
    4. text = regex_replace(numbers → "NUMBER")
       → (no change in this case)
    
    5. text = remove_punctuation(text)
       → "urgent your sbi account blocked click url"
    
    6. text = collapse_whitespace(text)
       → "urgent your sbi account blocked click url"
    
    OUTPUT: "urgent your sbi account blocked click url"
```

### Step 2: Tokenization (`preprocessing.py → tokenize_text()`)

```
ALGORITHM: Tokenize(cleaned_text)
    INPUT: "urgent your sbi account blocked click url"
    
    1. tokens = word_tokenize(text) using NLTK
       → ["urgent", "your", "sbi", "account", "blocked", "click", "url"]
    
    OUTPUT: ["urgent", "your", "sbi", "account", "blocked", "click", "url"]
```

### Step 3: Stopword Removal (`preprocessing.py → remove_stopwords()`)

```
ALGORITHM: RemoveStopwords(tokens)
    INPUT: ["urgent", "your", "sbi", "account", "blocked", "click", "url"]
    
    NLTK_STOPWORDS = {"the", "is", "at", "your", "in", "a", "an", ...}
    
    1. filtered = [t for t in tokens if t NOT IN NLTK_STOPWORDS]
       → ["urgent", "sbi", "account", "blocked", "click", "url"]
    
    OUTPUT: ["urgent", "sbi", "account", "blocked", "click", "url"]
```

### Step 4: Stemming (`preprocessing.py → stem_or_lemmatize()`)

```
ALGORITHM: Stem(tokens) using PorterStemmer
    INPUT: ["urgent", "sbi", "account", "blocked", "click", "url"]
    
    Porter Stemmer Rules (examples):
        "blocked"  → remove suffix "ed"  → "block"
        "running"  → remove suffix "ning" → "run"
        "urgent"   → no suffix to remove → "urgent"
    
    OUTPUT: ["urgent", "sbi", "account", "block", "click", "url"]
    
    FINAL: Join → "urgent sbi account block click url"
```

### Step 5: Statistical Feature Extraction (19 features)

From the **raw text** (before cleaning), extract:

| # | Feature Name | Formula | Example Value |
|:--|:-------------|:--------|:--------------|
| 1 | `message_length` | `len(text)` | 58 |
| 2 | `word_count` | `len(text.split())` | 9 |
| 3 | `char_count` | `len(text)` | 58 |
| 4 | `avg_word_length` | `mean([len(w) for w in words])` | 5.3 |
| 5 | `special_char_count` | `count of punctuation chars` | 4 |
| 6 | `digit_count` | `count of 0-9 digits` | 0 |
| 7 | `uppercase_count` | `count of A-Z chars` | 15 |
| 8 | `uppercase_ratio` | `uppercase_count / len(text)` | 0.26 |
| 9 | `digit_ratio` | `digit_count / len(text)` | 0.0 |
| 10 | `special_char_ratio` | `special_char_count / len(text)` | 0.07 |
| 11 | `has_url` | `1 if URL pattern found else 0` | 1 |
| 12 | `has_email` | `1 if email pattern found else 0` | 0 |
| 13 | `has_phone` | `1 if phone pattern found else 0` | 0 |
| 14 | `has_currency` | `1 if $,£,€ found else 0` | 0 |
| 15 | `urgency_count` | `count of words like "urgent", "immediately", "act now"` | 1 |
| 16 | `financial_count` | `count of words like "account", "bank", "credit"` | 1 |
| 17 | `action_count` | `count of words like "click", "verify", "login"` | 1 |
| 18 | `threat_count` | `count of words like "blocked", "suspended", "locked"` | 1 |
| 19 | `excessive_caps` | `1 if uppercase_ratio > 0.3 else 0` | 0 |

---

## 4. TF-IDF Vectorization — Full Mathematical Derivation

**TF-IDF** = **Term Frequency × Inverse Document Frequency**

### 4.1 Term Frequency (TF)

The frequency of term `t` in document `d`:

```
TF(t, d) = (Number of times term t appears in document d)
           ─────────────────────────────────────────────────
           (Total number of terms in document d)
```

**Example:** For document "urgent sbi account block click url" (6 words):
- TF("urgent", d) = 1/6 = 0.167
- TF("bank", d) = 0/6 = 0.000

### 4.2 Inverse Document Frequency (IDF)

Measures how rare/important a word is across ALL documents:

```
IDF(t) = log_e( (1 + n) / (1 + df(t)) ) + 1

Where:
    n    = total number of documents in the corpus (e.g., 5574 SMS messages)
    df(t) = number of documents containing term t
```

**Example** (corpus of 5574 messages):
- Word "the" appears in 4000 documents:
  `IDF("the") = ln((1+5574)/(1+4000)) + 1 = ln(1.394) + 1 = 1.332` → **Low importance**
- Word "urgent" appears in 120 documents:
  `IDF("urgent") = ln((1+5574)/(1+120)) + 1 = ln(46.08) + 1 = 4.83` → **High importance**

### 4.3 Final TF-IDF Score

```
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

**Example:** For "urgent" in our document:
```
TF-IDF("urgent") = 0.167 × 4.83 = 0.807
```
This high score means "urgent" is **frequent in this message** AND **rare across the corpus** → strong phishing indicator.

### 4.4 L2 Normalization

After computing all TF-IDF scores for a document, scikit-learn applies L2 normalization:

```
TF-IDF_normalized(t, d) = TF-IDF(t, d) / √(Σ TF-IDF(t', d)² for all t' in d)
```

This ensures each document vector has unit length = 1, making documents comparable regardless of length.

### 4.5 Configuration in Our System

```python
TfidfVectorizer(
    max_features=500,      # Keep only top 500 most important words
    ngram_range=(1, 2),    # Unigrams ("urgent") AND bigrams ("act now")
    min_df=2,              # Ignore words appearing in < 2 documents
    max_df=0.95            # Ignore words appearing in > 95% of documents
)
```

**Output:** Each SMS message → vector of 500 float values (one per word/bigram)

### 4.6 StandardScaler on Statistical Features

The 19 statistical features are normalized using StandardScaler:

```
z = (x - μ) / σ

Where:
    x = original feature value
    μ = mean of that feature across all training samples
    σ = standard deviation of that feature across all training samples
```

**Example:** `message_length`:
- Training mean μ = 80 characters, std σ = 45
- New message length = 58
- Scaled value = (58 - 80) / 45 = -0.489

**Result:** 500 TF-IDF features + 19 scaled statistical features = **519-dimensional feature vector** fed to Random Forest.

---

## 5. URL Detection Algorithm — 30 Structural Features

Unlike SMS (text → TF-IDF), URLs are analyzed by extracting **30 numerical features** from the URL string itself:

### 5.1 Feature Categories

**A. Length Features (4):**
| Feature | Formula | Safe Example (google.com) | Phishing Example (sbi-secure.xyz/login) |
|:--------|:--------|:--------------------------|:----------------------------------------|
| `url_length` | `len(url)` | 22 | 42 |
| `hostname_length` | `len(hostname)` | 14 | 14 |
| `path_length` | `len(path)` | 1 | 6 |
| `query_length` | `len(query)` | 0 | 12 |

**B. Character Count Features (9):**
`num_dots`, `num_hyphens`, `num_underscores`, `num_slashes`, `num_at_signs`, `num_question_marks`, `num_equals`, `num_percent`, `num_digits`

**C. Structural Pattern Features (9):**
`has_ip_address`, `has_https`, `has_http_in_domain`, `num_subdomains`, `has_port`, `has_double_slash_redirect`, `domain_has_digits`, `tld_length`, `is_shortened`

**D. Keyword Features (2):**
`has_suspicious_words` (login, verify, secure, update, confirm...), `has_brand_in_subdomain` (paypal in subdomain but not in real domain)

**E. Entropy Feature (1):**
`hostname_entropy` — Shannon Entropy of the hostname (see Section 6)

**F. Ratio Features (2):**
`digit_to_letter_ratio = num_digits / max(num_letters, 1)`
`special_char_ratio = num_special / max(len(url), 1)`

**G. Path Feature (1):**
`path_depth = count of "/" segments in URL path`

---

## 6. Shannon Entropy — Mathematical Derivation

Shannon Entropy measures the **randomness/disorder** of a string. Phishing domains tend to have high entropy (random characters like `x7k2m9.xyz`) while legitimate domains have low entropy (`google.com`).

### Formula:

```
H(X) = -Σ p(xᵢ) × log₂(p(xᵢ))    for all unique characters xᵢ

Where:
    p(xᵢ) = frequency of character xᵢ / total length of string
```

### Worked Example 1: "google" (legitimate)

| Char | Count | p(xᵢ) | -p × log₂(p) |
|:-----|:------|:-------|:--------------|
| g | 2 | 2/6 = 0.333 | -0.333 × log₂(0.333) = 0.528 |
| o | 2 | 2/6 = 0.333 | -0.333 × log₂(0.333) = 0.528 |
| l | 1 | 1/6 = 0.167 | -0.167 × log₂(0.167) = 0.431 |
| e | 1 | 1/6 = 0.167 | -0.167 × log₂(0.167) = 0.431 |

**H("google") = 0.528 + 0.528 + 0.431 + 0.431 = 1.918 bits** → **Low entropy (ordered)**

### Worked Example 2: "x7k2m9" (suspicious/random)

| Char | Count | p(xᵢ) | -p × log₂(p) |
|:-----|:------|:-------|:--------------|
| x | 1 | 1/6 = 0.167 | 0.431 |
| 7 | 1 | 1/6 = 0.167 | 0.431 |
| k | 1 | 1/6 = 0.167 | 0.431 |
| 2 | 1 | 1/6 = 0.167 | 0.431 |
| m | 1 | 1/6 = 0.167 | 0.431 |
| 9 | 1 | 1/6 = 0.167 | 0.431 |

**H("x7k2m9") = 6 × 0.431 = 2.585 bits** → **High entropy (random) = suspicious!**

### Code Implementation:

```python
def _shannon_entropy(text):
    if not text:
        return 0.0
    prob = {c: text.count(c) / len(text) for c in set(text)}
    return -sum(p * math.log2(p) for p in prob.values())
```

---

## 7. Random Forest Classifier — Complete Internal Algorithm

Random Forest is an **ensemble** of `B` independent Decision Trees. Our system uses **two separate Random Forest models** — one for SMS text analysis and one for URL structural analysis. The core algorithm is identical for both; only the **input features, number of trees, and class labels** differ.

| Aspect | SMS Random Forest | URL Random Forest |
|:-------|:------------------|:------------------|
| Number of Trees (B) | 100 | 200 |
| Input Features | 519 (500 TF-IDF + 19 statistical) | 30 (structural URL features) |
| Features per Node Split (√F) | √519 ≈ 23 | √30 ≈ 5 |
| Class Labels | ham (0) / spam (1) | legitimate (0) / phishing (1) |
| Feature Type | Text-derived (word frequencies + stats) | Numerical (URL structure patterns) |

### 7.1 Training Algorithm — SMS Case (B = 100 trees, 519 features)

```
ALGORITHM: TrainRandomForest_SMS(D_train, B=100)

INPUT:
    D_train = {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)}
    Each xᵢ = 519-dimensional vector (500 TF-IDF + 19 statistical features)
    Each yᵢ ∈ {0 = ham, 1 = spam}
    B = number of trees (100)

FOR b = 1 to 100:
    1. BOOTSTRAP SAMPLING:
       D_b = randomly sample n data points from D_train WITH replacement
       (some points appear multiple times, ~37% are left out → "Out-of-Bag")

    2. BUILD TREE:
       T_b = BuildDecisionTree(D_b)
       At each node, instead of testing ALL 519 features:
           - Randomly select m = √519 ≈ 23 features
           - Find the best split among ONLY those 23 features
           - Split the node

    3. Grow tree until:
       - Node has only 1 class (pure), OR
       - Node has fewer than min_samples_split (default=2) samples, OR
       - Maximum depth is reached

RETURN: SMS_Forest = {T₁, T₂, ..., T₁₀₀}
```

### 7.2 Training Algorithm — URL Case (B = 200 trees, 30 features)

```
ALGORITHM: TrainRandomForest_URL(D_train, B=200)

INPUT:
    D_train = {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)}
    Each xᵢ = 30-dimensional vector (url_length, num_dots, hostname_entropy, ...)
    Each yᵢ ∈ {0 = legitimate, 1 = phishing}
    B = number of trees (200)

FOR b = 1 to 200:
    1. BOOTSTRAP SAMPLING:
       D_b = randomly sample n URL data points from D_train WITH replacement
       (some URLs appear multiple times, ~37% are left out → "Out-of-Bag")

    2. BUILD TREE:
       T_b = BuildDecisionTree(D_b)
       At each node, instead of testing ALL 30 features:
           - Randomly select m = √30 ≈ 5 features
             (e.g., randomly pick: hostname_entropy, num_dots, has_https,
              url_length, has_suspicious_words — out of 30 total)
           - Find the best split among ONLY those 5 features
           - Split the node

    3. Grow tree until:
       - Node has only 1 class (pure), OR
       - Node has fewer than min_samples_split (default=2) samples, OR
       - Maximum depth is reached

RETURN: URL_Forest = {T₁, T₂, ..., T₂₀₀}
```

> **Why 200 trees for URL instead of 100?** URL features are purely numerical and structural (only 30 features), so each individual tree has less information to work with compared to the 519-feature SMS model. Using more trees (200) compensates by creating greater ensemble diversity, leading to more robust predictions on subtle phishing patterns.

### 7.3 Prediction Algorithm — SMS Case

```
ALGORITHM: PredictRandomForest_SMS(SMS_Forest, x_new)

INPUT: x_new = 519-dimensional feature vector of new SMS message
       (500 TF-IDF values + 19 statistical values)

FOR each tree T_b in SMS_Forest (100 trees):
    prediction_b = T_b.predict(x_new)    // traverse tree to leaf node

// MAJORITY VOTING:
count_spam = count(prediction_b == 1)     // e.g., 92 trees say "spam"
count_ham  = count(prediction_b == 0)     // e.g., 8 trees say "ham"

final_prediction = argmax(count_spam, count_ham)  // "spam" wins

// PROBABILITY (predict_proba):
P(spam) = count_spam / B = 92/100 = 0.92
P(ham)  = count_ham / B  = 8/100  = 0.08

RETURN: prediction = 1 (spam), probability = [0.08, 0.92]
        → SMS threat_score = 0.92 (92%)
```

### 7.4 Prediction Algorithm — URL Case

```
ALGORITHM: PredictRandomForest_URL(URL_Forest, x_new)

INPUT: x_new = 30-dimensional feature vector of new URL
       Example URL: "http://sbi-secure.xyz/login?id=38271"
       x_new = [42, 14, 6, 12, 2, 1, 0, 2, 0, 1, 1, 3, 0, 0, 1,
                0, 0, 1, 3, 1, 0, 1, 0.35, 0.05, 2, 2.85, 1, 0, 1, 1]
                ↑ These are the 30 structural features:
                url_length=42, hostname_length=14, path_length=6, ...
                hostname_entropy=2.85, has_suspicious_words=1, etc.

FOR each tree T_b in URL_Forest (200 trees):
    prediction_b = T_b.predict(x_new)    // traverse tree to leaf node

// MAJORITY VOTING:
count_phishing  = count(prediction_b == 1)  // e.g., 178 trees say "phishing"
count_legitimate = count(prediction_b == 0) // e.g., 22 trees say "legitimate"

final_prediction = argmax(count_phishing, count_legitimate)  // "phishing" wins

// PROBABILITY (predict_proba):
P(phishing)   = count_phishing / B  = 178/200 = 0.89
P(legitimate) = count_legitimate / B = 22/200 = 0.11

RETURN: prediction = 1 (phishing), probability = [0.11, 0.89]
        → URL threat_score = 0.89 (89%)
```

### 7.5 How a Single Tree Traverses — URL Example

When one tree in the URL Forest receives the feature vector for `http://sbi-secure.xyz/login?id=38271`:

```
                    [Root Node]
              hostname_entropy > 2.5?
                 /            \
              YES               NO
               ↓                 ↓
        [Node 2]            [Node 3]
     has_https == 0?       url_length > 50?
       /        \            /        \
     YES        NO         YES        NO
      ↓          ↓          ↓          ↓
  [Node 4]   [Leaf]     [Leaf]     [Leaf]
  num_dots>3? → legit   → phish    → legit
    /    \
  YES    NO
   ↓      ↓
[Leaf] [Leaf]
→phish →phish

For our URL: hostname_entropy=2.85 (>2.5) → YES → Node 2
             has_https=0 (no HTTPS) → YES → Node 4
             num_dots=2 (≤3) → NO → Leaf: PHISHING ✓

This tree votes: PHISHING
```

---

## 8. How a Node is Declared/Split in Random Forest

This is the **core mathematical mechanism** inside each Decision Tree. The splitting logic is the same for both SMS and URL models — only the features and feature count differ.

### 8.1 Node Structure

**SMS Tree Node:**
```
Node = {
    feature_index: which feature to test (e.g., "urgency_count")
    threshold:     split value (e.g., 0.5)
    left_child:    node for samples where feature ≤ threshold
    right_child:   node for samples where feature > threshold
    class_distribution: [count_ham, count_spam] at this node
}
```

**URL Tree Node:**
```
Node = {
    feature_index: which feature to test (e.g., "hostname_entropy")
    threshold:     split value (e.g., 2.5)
    left_child:    node for samples where feature ≤ threshold
    right_child:   node for samples where feature > threshold
    class_distribution: [count_legitimate, count_phishing] at this node
}
```

### 8.2 Splitting Criterion — Gini Impurity

The algorithm tries EVERY possible split and picks the one that **minimizes Gini Impurity**.

**Gini Impurity** measures how "mixed" a node is:

```
Gini(node) = 1 - Σ pᵢ²

Where pᵢ = proportion of class i in the node
```

**SMS Examples:**
- Pure node (all spam): Gini = 1 - (1.0² + 0.0²) = 1 - 1 = **0.0** (perfect)
- Mixed node (50/50):   Gini = 1 - (0.5² + 0.5²) = 1 - 0.5 = **0.5** (worst)
- Node with 90% spam:   Gini = 1 - (0.1² + 0.9²) = 1 - 0.82 = **0.18** (good)

**URL Examples:**
- Pure node (all phishing): Gini = 1 - (1.0² + 0.0²) = 1 - 1 = **0.0** (perfect)
- Mixed node (50/50):       Gini = 1 - (0.5² + 0.5²) = 1 - 0.5 = **0.5** (worst)
- Node with 85% phishing:   Gini = 1 - (0.15² + 0.85²) = 1 - (0.0225 + 0.7225) = **0.255**

### 8.3 The Splitting Algorithm — SMS Case (Step-by-Step)

```
ALGORITHM: FindBestSplit_SMS(node_samples)

INPUT: node_samples with n samples, total F = 519 features

1. RANDOM FEATURE SELECTION:
   m = √F = √519 ≈ 23 features (selected randomly from 519)
   Example random selection: [tfidf_word_42, tfidf_word_198, urgency_count,
                              uppercase_ratio, tfidf_word_301, ...] (23 total)

2. FOR each selected feature f in {f₁, f₂, ..., f₂₃}:
       Sort all n samples by feature f
       
       FOR each unique value v of feature f as potential threshold:
           Split samples into:
               Left  = {samples where f ≤ v}
               Right = {samples where f > v}
           
           Calculate weighted Gini of the split:
               Gini_split = (|Left|/n) × Gini(Left) + (|Right|/n) × Gini(Right)
           
           Calculate Information Gain:
               Gain = Gini(parent) - Gini_split

3. SELECT the split (feature, threshold) with HIGHEST Gain (lowest Gini_split)

4. CREATE child nodes:
       Left_child  ← samples where best_feature ≤ best_threshold
       Right_child ← samples where best_feature > best_threshold

5. RECURSE on both children until stopping conditions are met
```

### 8.4 The Splitting Algorithm — URL Case (Step-by-Step)

```
ALGORITHM: FindBestSplit_URL(node_samples)

INPUT: node_samples with n URL samples, total F = 30 features

1. RANDOM FEATURE SELECTION:
   m = √F = √30 ≈ 5 features (selected randomly from 30)
   Example random selection: [hostname_entropy, num_dots, has_https,
                              url_length, has_suspicious_words]

2. FOR each selected feature f in {f₁, f₂, ..., f₅}:
       Sort all n URL samples by feature f
       
       FOR each unique value v of feature f as potential threshold:
           Split samples into:
               Left  = {URLs where f ≤ v}
               Right = {URLs where f > v}
           
           Calculate weighted Gini of the split:
               Gini_split = (|Left|/n) × Gini(Left) + (|Right|/n) × Gini(Right)
           
           Calculate Information Gain:
               Gain = Gini(parent) - Gini_split

3. SELECT the split (feature, threshold) with HIGHEST Gain (lowest Gini_split)

4. CREATE child nodes:
       Left_child  ← URLs where best_feature ≤ best_threshold
       Right_child ← URLs where best_feature > best_threshold

5. RECURSE on both children until stopping conditions are met
```

### 8.5 Concrete Worked Example — SMS Case

**Parent node:** 100 samples (70 ham, 30 spam)
```
Gini(parent) = 1 - (0.7² + 0.3²) = 1 - (0.49 + 0.09) = 0.42
```

**Testing split:** Feature = `urgency_count`, Threshold = 0.5

| | urgency ≤ 0.5 (Left) | urgency > 0.5 (Right) |
|:---|:---|:---|
| Ham | 65 | 5 |
| Spam | 8 | 22 |
| Total | 73 | 27 |

```
Gini(Left)  = 1 - ((65/73)² + (8/73)²) = 1 - (0.793 + 0.012) = 0.195
Gini(Right) = 1 - ((5/27)² + (22/27)²)  = 1 - (0.034 + 0.664) = 0.302

Gini_split = (73/100) × 0.195 + (27/100) × 0.302 = 0.142 + 0.082 = 0.224

Information Gain = 0.42 - 0.224 = 0.196  ← SIGNIFICANT GAIN!
```

**The SMS node is DECLARED with:**
- **Feature:** `urgency_count`
- **Threshold:** 0.5
- **Rule:** "IF urgency_count ≤ 0.5 → go LEFT (likely ham), ELSE → go RIGHT (likely spam)"

### 8.6 Concrete Worked Example — URL Case

**Parent node:** 100 URL samples (55 legitimate, 45 phishing)
```
Gini(parent) = 1 - (0.55² + 0.45²) = 1 - (0.3025 + 0.2025) = 0.495
```
This is close to 0.5 → the node is very **mixed** (almost 50/50), so a good split is needed.

**Testing split:** Feature = `hostname_entropy`, Threshold = 2.5

*Logic: Legitimate domains like "google.com" have low entropy (~1.9), while phishing domains like "x7k2m9.xyz" have high entropy (~2.6+)*

| | entropy ≤ 2.5 (Left) | entropy > 2.5 (Right) |
|:---|:---|:---|
| Legitimate | 48 | 7 |
| Phishing | 12 | 33 |
| Total | 60 | 40 |

```
Gini(Left)  = 1 - ((48/60)² + (12/60)²) = 1 - (0.64 + 0.04) = 0.32
Gini(Right) = 1 - ((7/40)² + (33/40)²)   = 1 - (0.031 + 0.681) = 0.288

Gini_split = (60/100) × 0.32 + (40/100) × 0.288 = 0.192 + 0.115 = 0.307

Information Gain = 0.495 - 0.307 = 0.188  ← GOOD GAIN!
```

**Now test another URL feature:** Feature = `has_https`, Threshold = 0.5

*Logic: Phishing sites often lack HTTPS (has_https = 0)*

| | has_https ≤ 0.5 (Left = no HTTPS) | has_https > 0.5 (Right = has HTTPS) |
|:---|:---|:---|
| Legitimate | 10 | 45 |
| Phishing | 35 | 10 |
| Total | 45 | 55 |

```
Gini(Left)  = 1 - ((10/45)² + (35/45)²) = 1 - (0.049 + 0.605) = 0.346
Gini(Right) = 1 - ((45/55)² + (10/55)²)  = 1 - (0.669 + 0.033) = 0.298

Gini_split = (45/100) × 0.346 + (55/100) × 0.298 = 0.156 + 0.164 = 0.320

Information Gain = 0.495 - 0.320 = 0.175
```

**Comparing the two candidate splits:**

| Feature | Gini_split | Information Gain | Winner? |
|:--------|:-----------|:-----------------|:--------|
| `hostname_entropy > 2.5` | 0.307 | **0.188** | ✅ BEST |
| `has_https > 0.5` | 0.320 | 0.175 | |

**The URL node is DECLARED with:**
- **Feature:** `hostname_entropy`
- **Threshold:** 2.5
- **Rule:** "IF hostname_entropy ≤ 2.5 → go LEFT (likely legitimate), ELSE → go RIGHT (likely phishing)"

> **Key Insight:** In the URL model, the features being tested are structural properties of the URL (entropy, HTTPS presence, dots, length) rather than word frequencies. The math is identical — only the meaning of the features changes.

### 8.7 Why √F Features? (Random Feature Selection)

**SMS Model:** If all 100 trees tested ALL 519 features at every node, all trees would make the same splits → they'd be identical → no diversity → no benefit of ensemble. By randomly selecting only √519 ≈ **23 features** at each node, each tree sees different features → different trees → different "opinions" → better ensemble.

**URL Model:** Similarly, if all 200 trees tested ALL 30 features at every node, they'd all be identical. By randomly selecting only √30 ≈ **5 features** at each node:
- Tree 1 might see: `hostname_entropy`, `num_dots`, `has_https`, `url_length`, `path_depth`
- Tree 2 might see: `num_hyphens`, `has_ip_address`, `tld_length`, `is_shortened`, `num_digits`
- Tree 3 might see: `has_suspicious_words`, `hostname_entropy`, `num_subdomains`, `query_length`, `has_port`

Each tree learns different aspects of what makes a URL phishing → when combined, the 200 trees cover all 30 features thoroughly from many angles.

### 8.8 Deep Dive — Complete Node-by-Node Tree Construction (Every Internal Value)

This section shows **exactly what is stored inside every node** of a decision tree and how a complete tree is built level by level. This is the level of detail that demonstrates full understanding of Random Forest internals.

#### 8.8.1 Complete Internal Data Structure of a Node

Every single node in a scikit-learn Decision Tree stores these values:

```
Node_Internal_Structure = {
    // ─── SPLITTING INFORMATION ───
    feature_index:       int     // Index of the feature used to split (e.g., 14 = "urgency_count")
    threshold:           float   // The split value (e.g., 0.5)
    
    // ─── TREE POINTERS ───
    left_child:          Node    // Pointer to left child (samples where feature ≤ threshold)
    right_child:         Node    // Pointer to right child (samples where feature > threshold)
    
    // ─── NODE STATISTICS ───
    n_samples:           int     // Total number of training samples reaching this node
    n_samples_class:     array   // Class distribution: [count_class_0, count_class_1]
    weighted_n_samples:  float   // Weighted sample count (for class_weight adjustments)
    
    // ─── IMPURITY MEASURES ───
    impurity:            float   // Gini impurity at this node BEFORE splitting
    impurity_decrease:   float   // Weighted impurity decrease = Gain from this split
                                 // = (n/N) × [Gini(parent) - Gini_split]
    
    // ─── LEAF NODE VALUES (only for leaf nodes) ───
    value:               array   // Class vote counts: [n_class_0, n_class_1]
    predicted_class:     int     // argmax(value) → the class this leaf votes for
    
    // ─── DEPTH TRACKING ───
    depth:               int     // Level in the tree (root = 0, first children = 1, etc.)
    is_leaf:             bool    // True if no further splits possible
}
```

#### 8.8.2 Complete SMS Tree Construction — Node by Node (Tree #1 of 100)

Let's build a complete decision tree from the SMS Random Forest, showing **every value at every node**.

**Bootstrap sample for Tree #1:** 4000 samples drawn (with replacement) from 5574 total SMS messages. This particular bootstrap has 2800 ham, 1200 spam.

```
═══════════════════════════════════════════════════════════════
LEVEL 0: ROOT NODE (Node 0) — Building the first split
═══════════════════════════════════════════════════════════════

State at ROOT:
    n_samples = 4000
    n_samples_class = [2800 ham, 1200 spam]
    impurity (Gini) = 1 - (0.70² + 0.30²) = 1 - (0.49 + 0.09) = 0.42
    depth = 0

Random features selected (23 of 519):
    [tfidf_"urgent"(#42), tfidf_"click"(#88), tfidf_"verify"(#201),
     tfidf_"account"(#15), urgency_count(#501), has_url(#511),
     uppercase_ratio(#508), action_count(#517), threat_count(#518),
     tfidf_"bank"(#31), tfidf_"free"(#105), special_char_ratio(#510),
     ... 11 more random features]

Testing all 23 features × all unique thresholds:
    Feature: urgency_count,   threshold=0.5  → Gini_split=0.312, Gain=0.108
    Feature: tfidf_"urgent",  threshold=0.15 → Gini_split=0.298, Gain=0.122
    Feature: has_url,         threshold=0.5  → Gini_split=0.335, Gain=0.085
    Feature: uppercase_ratio, threshold=0.25 → Gini_split=0.361, Gain=0.059
    Feature: tfidf_"click",   threshold=0.08 → Gini_split=0.289, Gain=0.131 ← BEST!
    Feature: threat_count,    threshold=0.5  → Gini_split=0.305, Gain=0.115
    ... (testing all 23 features)

WINNER: tfidf_"click" with threshold 0.08

NODE 0 IS DECLARED:
    feature_index = 88 (tfidf_"click")
    threshold = 0.08
    impurity = 0.42
    impurity_decrease = (4000/4000) × (0.42 - 0.289) = 0.131
    n_samples = 4000
    n_samples_class = [2800, 1200]
    depth = 0
    is_leaf = False
    
    Split: samples where tfidf_"click" ≤ 0.08 → LEFT (3100 samples)
           samples where tfidf_"click" > 0.08 → RIGHT (900 samples)
```

```
═══════════════════════════════════════════════════════════════
LEVEL 1: LEFT CHILD (Node 1) — Messages WITHOUT "click"
═══════════════════════════════════════════════════════════════

State:
    n_samples = 3100
    n_samples_class = [2650 ham, 450 spam]
    impurity (Gini) = 1 - (0.855² + 0.145²) = 1 - (0.731 + 0.021) = 0.248
    depth = 1

NEW random features selected (different 23 from 519):
    [tfidf_"free"(#105), tfidf_"win"(#310), financial_count(#516),
     digit_ratio(#509), message_length(#500), tfidf_"offer"(#178),
     ... 17 more random features]
     
     NOTE: Different random selection at EVERY node!

Testing all features:
    WINNER: financial_count with threshold 1.5

NODE 1 IS DECLARED:
    feature_index = 516 (financial_count)
    threshold = 1.5
    impurity = 0.248
    impurity_decrease = (3100/4000) × (0.248 - 0.104) = 0.112
    n_samples = 3100
    n_samples_class = [2650, 450]
    depth = 1
    is_leaf = False
    
    Split: financial_count ≤ 1.5 → LEFT (2700 samples)
           financial_count > 1.5 → RIGHT (400 samples)
```

```
═══════════════════════════════════════════════════════════════
LEVEL 1: RIGHT CHILD (Node 2) — Messages WITH "click"
═══════════════════════════════════════════════════════════════

State:
    n_samples = 900
    n_samples_class = [150 ham, 750 spam]
    impurity (Gini) = 1 - (0.167² + 0.833²) = 1 - (0.028 + 0.694) = 0.278
    depth = 1

WINNER: has_url with threshold 0.5

NODE 2 IS DECLARED:
    feature_index = 511 (has_url)
    threshold = 0.5
    impurity = 0.278
    impurity_decrease = (900/4000) × (0.278 - 0.065) = 0.048
    n_samples = 900
    n_samples_class = [150, 750]
    depth = 1
    is_leaf = False
```

```
═══════════════════════════════════════════════════════════════
LEVEL 2: LEAF NODES — Where the tree stops splitting
═══════════════════════════════════════════════════════════════

Node 3 (Left child of Node 1):
    n_samples = 2700
    n_samples_class = [2600, 100]
    impurity = 0.071 (nearly pure — mostly ham)
    depth = 2
    is_leaf = True  ← STOPPING because Gini < min_impurity_decrease
    
    value = [2600, 100]
    predicted_class = 0 (HAM) ← because 2600 > 100
    confidence = 2600/2700 = 96.3%

Node 4 (Right child of Node 1):
    n_samples = 400
    n_samples_class = [50, 350]
    impurity = 0.219
    depth = 2
    is_leaf = True
    
    value = [50, 350]
    predicted_class = 1 (SPAM) ← because 350 > 50
    confidence = 350/400 = 87.5%

Node 5 (Left child of Node 2 — "click" but NO url):
    n_samples = 200
    n_samples_class = [140, 60]
    impurity = 0.42
    depth = 2
    is_leaf = False → continues splitting further...

Node 6 (Right child of Node 2 — "click" AND has url):
    n_samples = 700
    n_samples_class = [10, 690]
    impurity = 0.028 (nearly pure — almost all spam)
    depth = 2
    is_leaf = True
    
    value = [10, 690]
    predicted_class = 1 (SPAM) ← very high confidence
    confidence = 690/700 = 98.6%
```

**Complete SMS Tree #1 structure:**
```
                        [Node 0: Root]
                  tfidf_"click" > 0.08?
                  Gini=0.42, n=4000
                  [2800 ham, 1200 spam]
                 /                      \
              NO                        YES
               ↓                         ↓
        [Node 1]                    [Node 2]
   financial_count > 1.5?       has_url > 0.5?
   Gini=0.248, n=3100           Gini=0.278, n=900
   [2650 ham, 450 spam]         [150 ham, 750 spam]
      /           \                /           \
   NO             YES           NO              YES
    ↓              ↓             ↓               ↓
 [LEAF 3]      [LEAF 4]     [Node 5]         [LEAF 6]
 HAM 96.3%     SPAM 87.5%   (splits more)    SPAM 98.6%
 n=2700        n=400        n=200            n=700
 [2600,100]    [50,350]     [140,60]         [10,690]
```

#### 8.8.3 Complete URL Tree Construction — Node by Node (Tree #1 of 200)

**Bootstrap sample for URL Tree #1:** 8000 URL samples drawn (with replacement). This bootstrap has 4400 legitimate, 3600 phishing.

```
═══════════════════════════════════════════════════════════════
LEVEL 0: ROOT NODE (Node 0)
═══════════════════════════════════════════════════════════════

State at ROOT:
    n_samples = 8000
    n_samples_class = [4400 legitimate, 3600 phishing]
    impurity (Gini) = 1 - (0.55² + 0.45²) = 0.495
    depth = 0

Random features selected (5 of 30):
    [hostname_entropy(#25), has_https(#11), url_length(#0),
     num_dots(#4), has_suspicious_words(#22)]

Testing all 5 features:
    hostname_entropy > 2.8  → Gini_split=0.298, Gain=0.197 ← BEST!
    has_https > 0.5         → Gini_split=0.341, Gain=0.154
    url_length > 35         → Gini_split=0.372, Gain=0.123
    num_dots > 3            → Gini_split=0.401, Gain=0.094
    has_suspicious_words>0.5→ Gini_split=0.356, Gain=0.139

NODE 0 IS DECLARED:
    feature_index = 25 (hostname_entropy)
    threshold = 2.8
    impurity = 0.495
    impurity_decrease = (8000/8000) × (0.495 - 0.298) = 0.197
    n_samples = 8000
    n_samples_class = [4400, 3600]
    depth = 0
    
    Split: hostname_entropy ≤ 2.8 → LEFT (5200 samples)
           hostname_entropy > 2.8 → RIGHT (2800 samples)
```

```
═══════════════════════════════════════════════════════════════
LEVEL 1: LEFT CHILD (Node 1) — Low entropy domains
═══════════════════════════════════════════════════════════════

State:
    n_samples = 5200
    n_samples_class = [4000 legitimate, 1200 phishing]
    impurity = 1 - (0.769² + 0.231²) = 0.355
    depth = 1

NEW random 5 features: [has_https, path_depth, num_hyphens,
                         has_ip_address, digit_to_letter_ratio]

WINNER: has_https with threshold 0.5

NODE 1 IS DECLARED:
    feature_index = 11 (has_https)
    threshold = 0.5
    impurity = 0.355
    n_samples = 5200
    n_samples_class = [4000, 1200]
    depth = 1
```

```
═══════════════════════════════════════════════════════════════
LEVEL 1: RIGHT CHILD (Node 2) — High entropy domains (random chars)
═══════════════════════════════════════════════════════════════

State:
    n_samples = 2800
    n_samples_class = [400 legitimate, 2400 phishing]
    impurity = 1 - (0.143² + 0.857²) = 0.245
    depth = 1

NEW random 5 features: [url_length, num_subdomains, is_shortened,
                         tld_length, num_digits]

WINNER: url_length with threshold 55

NODE 2 IS DECLARED:
    feature_index = 0 (url_length)
    threshold = 55
    impurity = 0.245
    n_samples = 2800
    n_samples_class = [400, 2400]
    depth = 1
```

```
═══════════════════════════════════════════════════════════════
LEVEL 2: LEAF NODES
═══════════════════════════════════════════════════════════════

Node 3 (Left of Node 1 — low entropy + has HTTPS):
    n_samples = 3800
    n_samples_class = [3600, 200]
    is_leaf = True
    predicted_class = 0 (LEGITIMATE), confidence = 94.7%

Node 4 (Right of Node 1 — low entropy + no HTTPS):
    n_samples = 1400
    n_samples_class = [400, 1000]
    is_leaf = False → splits further on num_hyphens...

Node 5 (Left of Node 2 — high entropy + short URL):
    n_samples = 800
    n_samples_class = [350, 450]
    is_leaf = False → still mixed, splits further...

Node 6 (Right of Node 2 — high entropy + long URL):
    n_samples = 2000
    n_samples_class = [50, 1950]
    is_leaf = True
    predicted_class = 1 (PHISHING), confidence = 97.5%
```

**Complete URL Tree #1 structure:**
```
                        [Node 0: Root]
                  hostname_entropy > 2.8?
                  Gini=0.495, n=8000
                  [4400 legit, 3600 phish]
                 /                        \
              NO                          YES
               ↓                           ↓
        [Node 1]                      [Node 2]
     has_https > 0.5?              url_length > 55?
     Gini=0.355, n=5200            Gini=0.245, n=2800
     [4000 legit, 1200 phish]      [400 legit, 2400 phish]
        /           \                  /            \
     YES            NO              NO              YES
      ↓              ↓               ↓               ↓
   [LEAF 3]      [Node 4]        [Node 5]         [LEAF 6]
   LEGIT 94.7%   (splits more)   (splits more)    PHISH 97.5%
   n=3800        n=1400          n=800            n=2000
   [3600,200]    [400,1000]      [350,450]        [50,1950]
```

#### 8.8.4 How Different Trees Get DIFFERENT Structures (The Key Insight)

The guide's key question: **"Why do different trees in the same forest have different nodes?"**

Two mechanisms create tree diversity:

**Mechanism 1: Different Bootstrap Samples**
```
Original dataset: 8000 URL samples

Tree #1 bootstrap: Randomly draws 8000 WITH replacement
    → Gets samples [#14, #7, #7, #203, #14, #5001, ...]
    → Some samples appear 2-3 times, ~37% are missing entirely
    → Class ratio might be [4400 legit, 3600 phish]

Tree #2 bootstrap: Independently draws another 8000
    → Gets samples [#891, #2, #4507, #891, #12, ...]
    → DIFFERENT samples duplicated, DIFFERENT ones missing
    → Class ratio might be [4200 legit, 3800 phish]

→ Different training data = different splits!
```

**Mechanism 2: Different Random Feature Subsets at EVERY Node**
```
URL Tree #1, Root Node:                  URL Tree #2, Root Node:
  Random 5 features:                       Random 5 features:
  [hostname_entropy,                       [num_hyphens,
   has_https,                               is_shortened,
   url_length,                              num_subdomains,
   num_dots,                                has_port,
   has_suspicious_words]                    path_depth]

  Best split: hostname_entropy > 2.8       Best split: num_hyphens > 2
  Gain = 0.197                             Gain = 0.145

→ Tree #1 splits on entropy                → Tree #2 splits on hyphens
→ COMPLETELY DIFFERENT root nodes!          → COMPLETELY DIFFERENT tree structure!
```

**Resulting tree structures side by side:**
```
    URL Tree #1:                         URL Tree #2:
    
    hostname_entropy > 2.8?              num_hyphens > 2?
       /          \                         /          \
    has_https?   url_length>55?          is_shortened?  path_depth>3?
     /    \        /      \               /    \          /      \
   LEGIT  ...   ...    PHISH           PHISH  ...      ...    LEGIT

    URL Tree #3:                         URL Tree #4:
    
    has_ip_address > 0.5?                url_length > 42?
       /          \                         /          \
    num_dots?    PHISH 99%              has_https?   hostname_entropy?
     /    \                              /    \          /      \
   LEGIT  ...                         LEGIT  ...      ...    PHISH

Each of the 200 trees has a UNIQUE structure because:
  1. Different bootstrap sample (different training rows)
  2. Different random features at EVERY node (different columns)
  → 200 different "experts" that vote together
```

#### 8.8.5 What a Leaf Node Stores and How It Votes

When a node becomes a **leaf** (no more splitting), it stores final voting values:

```
LEAF NODE detailed structure for SMS Tree #1, Node 6:

Leaf = {
    is_leaf:          True
    depth:            2
    n_samples:        700
    n_samples_class:  [10 ham, 690 spam]
    impurity:         0.028        // Very pure node
    
    value:            [10, 690]    // Raw class counts
    
    // PREDICTION when a new sample reaches this leaf:
    predicted_class:  1 (SPAM)     // argmax([10, 690]) = index 1
    
    // PROBABILITY when predict_proba() is called:
    P(ham)  = 10 / 700  = 0.014   // 1.4% chance it's ham
    P(spam) = 690 / 700 = 0.986   // 98.6% chance it's spam
    
    // This leaf's VOTE in the forest:
    "I vote SPAM with 98.6% confidence"
}
```

**How 100 trees combine their leaf votes (SMS example):**
```
New SMS: "URGENT! Click http://bit.ly/xyz to verify your account"

Tree #1:  reaches Leaf [10, 690]   → P(spam) = 0.986
Tree #2:  reaches Leaf [5, 420]    → P(spam) = 0.988
Tree #3:  reaches Leaf [30, 180]   → P(spam) = 0.857
Tree #4:  reaches Leaf [200, 15]   → P(spam) = 0.070  ← disagrees!
Tree #5:  reaches Leaf [8, 502]    → P(spam) = 0.984
...
Tree #100: reaches Leaf [12, 388]  → P(spam) = 0.970

Final P(spam) = AVERAGE of all 100 trees' P(spam) values
             = (0.986 + 0.988 + 0.857 + 0.070 + 0.984 + ... + 0.970) / 100
             = 0.92

→ SMS threat_score = 0.92 (92%)

Note: Tree #4 disagreed (voted ham) because it saw different
features at its nodes. But the MAJORITY (92 out of 100) voted
spam → the ensemble is correct even when individual trees are wrong!
```

#### 8.8.6 Stopping Conditions — When a Node Becomes a Leaf

A node **stops splitting** and becomes a leaf when ANY of these conditions is met:

```
STOPPING CONDITIONS:
                                          
1. PURE NODE (Gini = 0.0)                            
   All samples belong to one class.                   
   Example: n_samples_class = [0, 350] → all spam     
   → No point splitting further                        
                                                       
2. MAX DEPTH REACHED                                  
   depth == max_depth parameter (default: None = unlimited)
   → Prevents overfitting by limiting tree complexity   
                                                       
3. MIN SAMPLES TO SPLIT                                
   n_samples < min_samples_split (default: 2)          
   → Only 1 sample left, can't split                   
                                                       
4. MIN SAMPLES IN LEAF                                 
   A split would create a child with < min_samples_leaf
   (default: 1) samples → split is rejected            
                                                       
5. NO IMPROVEMENT POSSIBLE                             
   Best possible split has Gain ≤ 0                    
   → Splitting would make things worse                 
                                                       
6. MAX LEAF NODES REACHED                              
   Total leaves in tree ≥ max_leaf_nodes               
   (default: None = unlimited)                         
```

## 9. Visual Forensics Algorithm (pHash + SSIM)

### How Does the System Know Which Trusted Image to Compare Against?

**The system does NOT know in advance which trusted image to compare against.** Instead, it uses a **two-stage funnel**:

```
Stage 1: COMPARE AGAINST ALL (Fast — pHash)
    The suspect screenshot's hash is compared against EVERY trusted image's hash.
    This is extremely fast because each comparison is just counting differing bits
    between two 256-bit numbers (a single integer subtraction).
    
    10 trusted images → 10 hash comparisons → takes < 1 millisecond total

    Result: Find the ONE closest match (lowest Hamming distance)

Stage 2: DEEP COMPARE AGAINST ONE (Slow — SSIM)
    ONLY if the closest match is "close enough" (distance ≤ 30),
    run the expensive SSIM comparison against THAT ONE best match only.
    
    This involves pixel-by-pixel analysis → takes ~200ms per comparison.
    We only run this ONCE (against the best match), NOT against all 10.
```

**Visual flow:**
```
Suspect URL: "http://sbi-secure.xyz/login"
                    │
                    ▼
        Capture screenshot via Selenium
                    │
                    ▼
        Compute pHash of suspect image
                    │
    ┌───────────────┼───────────────────────────────────────┐
    │               ▼                                       │
    │   Compare hash against ALL 10 trusted hashes:         │
    │                                                       │
    │   vs sbi_thumb.png      → distance = 12  ← CLOSEST!  │
    │   vs hdfc_thumb.png     → distance = 45              │
    │   vs icici_thumb.png    → distance = 52              │
    │   vs paypal_thumb.png   → distance = 78              │
    │   vs google_thumb.png   → distance = 91              │
    │   vs facebook_thumb.png → distance = 85              │
    │   vs amazon_thumb.png   → distance = 88              │
    │   vs axis_thumb.png     → distance = 61              │
    │   vs chase_thumb.png    → distance = 72              │
    │   vs wellsfargo_thumb   → distance = 69              │
    │                                                       │
    │   WINNER: sbi (distance = 12, which is ≤ 30)         │
    └───────────────┬───────────────────────────────────────┘
                    │
                    ▼
        Distance 12 ≤ 30 (PHASH_THRESHOLD)?  → YES
                    │
                    ▼
        Run SSIM ONLY against sbi.png (full screenshot)
                    │
                    ▼
        SSIM = 0.82 → spoofing_detected = True
        "This site is visually spoofing SBI!"
```

**Why this two-stage approach?**
- **pHash is fast:** Comparing two hashes = one integer subtraction. We can compare against hundreds of trusted images in milliseconds.
- **SSIM is slow:** It slides an 11×11 window across every pixel of two 1366×768 images. Running this against all 10 would take ~2 seconds.
- **Funnel logic:** Use the fast method to find the most likely match, then use the slow method to confirm and measure the exact similarity.

### Stage 1: Perceptual Hash (pHash) — Fast Pre-Filter

```
ALGORITHM: ComputePHash(image)

1. Resize image to 256×256 (normalize size)
2. Convert to grayscale
3. Apply Discrete Cosine Transform (DCT) 
4. Keep top-left 16×16 DCT coefficients (low-frequency = structure)
5. Compute median of coefficients
6. Hash: each coefficient → 1 if > median, else 0
7. Result: 256-bit binary hash (fingerprint of the image)

COMPARISON:
    Hamming_Distance(hash_A, hash_B) = count of differing bits
    
    If distance ≤ 30 (our PHASH_THRESHOLD) → images are SIMILAR → proceed to SSIM
    If distance > 30 → images are DIFFERENT → NOT spoofing (skip SSIM)
```

### Stage 2: SSIM (Structural Similarity Index Measure)

SSIM compares two images based on **three independent components** — luminance, contrast, and structure — then combines them into a single similarity score.

```
SSIM(x, y) = [l(x,y)]^α × [c(x,y)]^β × [s(x,y)]^γ

Where (with α = β = γ = 1, the default):

    1. LUMINANCE comparison (are they equally bright?):
       l(x,y) = (2μₓμᵧ + C₁) / (μₓ² + μᵧ² + C₁)
       
    2. CONTRAST comparison (do they have similar contrast?):
       c(x,y) = (2σₓσᵧ + C₂) / (σₓ² + σᵧ² + C₂)
       
    3. STRUCTURE comparison (do patterns/edges align?):
       s(x,y) = (σₓᵧ + C₃) / (σₓσᵧ + C₃)

Combined (simplified with C₃ = C₂/2):
    SSIM(x, y) = ((2μₓμᵧ + C₁)(2σₓᵧ + C₂)) / ((μₓ² + μᵧ² + C₁)(σₓ² + σᵧ² + C₂))

Where:
    μₓ, μᵧ     = mean pixel intensity of images x and y
    σₓ², σᵧ²   = variance of pixel intensities
    σₓᵧ        = covariance between x and y
    C₁ = (K₁ × L)² where K₁ = 0.01, L = 255 (pixel range) → C₁ = 6.5025
    C₂ = (K₂ × L)² where K₂ = 0.03, L = 255               → C₂ = 58.5225
    
    SSIM ∈ [-1, 1], where 1 = identical images
```

> **See Section 16 below** for a full step-by-step worked example with actual pixel values showing how SSIM is computed in our system.

```
Visual Threat Score:
    If SSIM ≥ 0.6 → spoofing_detected = True
    visual_threat_score = max(ssim_score, 1 - phash_distance/64)
```

---

## 10. Weighted Fusion & Final Threat Score Algorithm

```
ALGORITHM: ComputeFinalThreatScore(scores, analyses_performed)

INPUT:
    scores = {"sms": 0.92, "url": 0.87, "visual": 0.75}  (when all available)
    base_weights = {"sms": 0.40, "url": 0.45, "visual": 0.15}

1. IDENTIFY active channels (those that produced results)
   active = analyses_performed  // e.g., ["sms", "url"]

2. RE-NORMALIZE weights for active channels ONLY:
   total_active_weight = Σ base_weights[k] for k in active
   // e.g., if only SMS+URL: total = 0.40 + 0.45 = 0.85
   
   normalized_weight[k] = base_weights[k] / total_active_weight
   // SMS = 0.40/0.85 = 0.471, URL = 0.45/0.85 = 0.529

3. COMPUTE weighted sum:
   Final_Score = Σ scores[k] × normalized_weight[k]
   // = 0.92 × 0.471 + 0.87 × 0.529 = 0.433 + 0.460 = 0.893

4. CLASSIFY risk:
   If Final_Score < 0.30  → "LOW"
   If Final_Score < 0.60  → "MEDIUM"
   If Final_Score < 0.85  → "HIGH"
   If Final_Score ≥ 0.85  → "CRITICAL"
   
   // 0.893 → "CRITICAL"

OUTPUT: { combined_score: 0.893, risk_level: "CRITICAL" }
```

---

## 11. Model Evaluation Metrics — Mathematical Definitions

Based on the Confusion Matrix:

```
                        PREDICTED
                    Positive  |  Negative
              ┌──────────────┬──────────────┐
    Positive  │     TP       │     FN       │
ACTUAL        │ (True Pos)   │ (False Neg)  │
              ├──────────────┼──────────────┤
    Negative  │     FP       │     TN       │
              │ (False Pos)  │ (True Neg)   │
              └──────────────┴──────────────┘
```

| Metric | Formula | Meaning |
|:-------|:--------|:--------|
| **Accuracy** | `(TP + TN) / (TP + TN + FP + FN)` | Overall correctness |
| **Precision** | `TP / (TP + FP)` | Of all "spam" predictions, how many are actually spam? |
| **Recall** | `TP / (TP + FN)` | Of all actual spam, how many did we catch? |
| **F1-Score** | `2 × (Precision × Recall) / (Precision + Recall)` | Harmonic mean balancing P and R |
| **ROC-AUC** | Area under TPR vs FPR curve | Model's ability to discriminate classes |

---

## 12. Cross-Validation Algorithm (5-Fold Stratified)

```
ALGORITHM: StratifiedKFoldCV(model, D, k=5)

1. Divide dataset D into k=5 equal folds, preserving class ratio in each fold
   Fold1 = [samples 1-1115], Fold2 = [1116-2230], ...

2. FOR i = 1 to 5:
       Test_set  = Fold_i
       Train_set = All folds EXCEPT Fold_i
       
       model.fit(Train_set)
       predictions = model.predict(Test_set)
       score_i = f1_score(Test_set.labels, predictions)

3. Final_CV_Score = mean(score_1, score_2, ..., score_5)
   CV_Std = std(score_1, ..., score_5)

PURPOSE: Ensures the model performance is CONSISTENT, not just lucky on one split.
```

---

## 13. Hyperparameter Tuning (GridSearchCV)

```
ALGORITHM: GridSearchCV(model, param_grid, cv=5)

param_grid = {
    'n_estimators': [50, 100, 200],    // 3 values
    'max_depth':    [10, 20, None],    // 3 values
    'min_samples_split': [2, 5, 10]   // 3 values
}

Total combinations = 3 × 3 × 3 = 27

FOR each of 27 combinations:
    FOR each of 5 folds:
        Train model with this combination
        Evaluate on fold → get F1 score
    Average_F1 = mean of 5 fold scores

SELECT combination with highest Average_F1
RETRAIN final model with best parameters on full training set
```

---

## 14. Complete Pseudocode — Full System

```
ALGORITHM: PhishGuardAI_FullPipeline(user_input)

// ═══════════════════════════════════════════
// PHASE 1: INPUT PROCESSING
// ═══════════════════════════════════════════
message = user_input.text
urls = regex_extract_urls(message)

scores = {}
weights = {sms: 0.40, url: 0.45, visual: 0.15}

// ═══════════════════════════════════════════
// PHASE 2A: SMS ANALYSIS
// ═══════════════════════════════════════════
cleaned = clean_text(message)              // lowercase, remove punct
tokens = tokenize(cleaned)                 // split into words
tokens = remove_stopwords(tokens)          // remove "the", "is", etc.
tokens = stem(tokens)                      // "blocked" → "block"
processed_text = join(tokens)              // → "urgent sbi account block..."

tfidf_vector = loaded_tfidf.transform(processed_text)    // → float[500]
stat_features = extract_statistical(message)             // → float[19]
stat_scaled = loaded_scaler.transform(stat_features)     // → normalized float[19]
feature_vector = concat(tfidf_vector, stat_scaled)       // → float[519]

probabilities = sms_model.predict_proba(feature_vector)  // → [P(ham), P(spam)]
scores.sms = probabilities[1]                            // P(spam) = e.g., 0.92

// ═══════════════════════════════════════════
// PHASE 2B: URL ANALYSIS (for each URL found)
// ═══════════════════════════════════════════
FOR each url in urls:
    features = extract_30_url_features(url)   // length, dots, entropy, etc.
    vector = [features[name] for name in saved_feature_names]  // → float[30]
    vector = replace_nan_inf(vector, 0.0)
    
    proba = url_model.predict_proba(vector)   // → [P(legit), P(phish)]
    url_score = proba[1]                      // P(phishing)
    
scores.url = max(url_scores)  // worst URL determines the score

// ═══════════════════════════════════════════
// PHASE 2C: VISUAL ANALYSIS (optional)
// ═══════════════════════════════════════════
screenshot = selenium_capture(urls[0])
suspect_hash = phash(resize(screenshot, 256×256), hash_size=16)

FOR each trusted_site in trusted_database:
    distance = hamming(suspect_hash, trusted_hash)
    IF distance < best_distance:
        best_match = trusted_site

IF best_distance ≤ 30:  // Similar enough → run SSIM
    ssim_score = SSIM(grayscale(screenshot), grayscale(trusted_image))
    scores.visual = max(ssim_score, 1 - best_distance/64)
ELSE:
    scores.visual = 0.0  // Not spoofing any known site

// ═══════════════════════════════════════════
// PHASE 3: WEIGHTED FUSION
// ═══════════════════════════════════════════
active_weight_sum = Σ weights[k] for k in scores.keys()
combined = Σ (scores[k] × weights[k] / active_weight_sum)

// ═══════════════════════════════════════════
// PHASE 4: RISK CLASSIFICATION
// ═══════════════════════════════════════════
IF combined < 0.30 → risk = "LOW"
ELIF combined < 0.60 → risk = "MEDIUM"  
ELIF combined < 0.85 → risk = "HIGH"
ELSE → risk = "CRITICAL"

RETURN {
    combined_score: combined,
    risk_level: risk,
    sms_analysis: {score, prediction, confidence, features},
    url_analysis: {score, is_phishing, top_risk_features},
    visual_analysis: {spoofing_detected, ssim_score, heatmap}
}
```

---

## 15. Why URL Gets Higher Weight (0.45) Than SMS (0.40) — Justification

Our system assigns: **SMS = 0.40, URL = 0.45, Visual = 0.15**. The guide's question: *"Why give more preference to URL than other channels?"*

### 15.1 The Core Argument

The URL is the **attack vector itself** — it is the actual mechanism through which phishing damage occurs. A phishing message is only dangerous **because** it contains a malicious URL. Without the URL, the text alone cannot steal credentials, install malware, or redirect users to fake websites.

### 15.2 Detailed Justification

| Reason | Explanation | Weight Impact |
|:-------|:-----------|:--------------|
| **1. URL is the attack payload** | The URL is WHERE the victim gets phished. Even a perfectly crafted phishing SMS is harmless without a malicious link. The URL is the weapon; the text is just the delivery method. | URL should outweigh SMS |
| **2. URL features are objective** | URL structural features (length, entropy, HTTPS, IP address, suspicious TLD) are **mathematical and deterministic** — they don't depend on language, tone, or cultural context. SMS text analysis can be fooled by clever wording. | URL is more reliable |
| **3. Zero-day detection capability** | Our URL model analyzes the **structure** of the URL, not blacklists. This means it can detect **never-seen-before phishing URLs** (zero-day attacks) based on structural anomalies like random characters, suspicious TLDs (.xyz, .top), IP addresses in domains, etc. | URL is more future-proof |
| **4. Research-backed priority** | According to the Anti-Phishing Working Group (APWG), **95%+ of phishing attacks** rely on a malicious URL as the primary attack vector. Academic research consistently shows URL features as the strongest single predictor of phishing. | Industry standard |
| **5. Lower false positive rate** | URL structural analysis has a lower false positive rate than text analysis. A legitimate message might contain urgent language ("Your OTP is...") triggering the SMS model, but its URL (bank's real domain) would correctly show as safe. | URL corrects SMS errors |

### 15.3 Why Not Equal Weights (33/33/33)?

```
Scenario: Legitimate bank OTP message
    SMS text: "URGENT: Your OTP is 483921. Do not share. -SBI Bank"
    URL:      (none)
    Visual:   (none)

With equal weights:
    SMS says "spam" (0.75) → 0.75 × 0.33 = 0.248
    URL says nothing       → channel skipped
    Visual says nothing    → channel skipped
    Final = 0.248 / 0.33 = 0.75 → HIGH risk → FALSE ALARM on a real OTP!

With our weights (0.40, 0.45, 0.15):
    Same scenario, but when URL IS present:
    SMS says "spam" (0.75) → 0.75 × 0.40
    URL says "safe" (0.10) → 0.10 × 0.45  ← URL's higher weight CORRECTS the error
    Final = (0.30 + 0.045) / 0.85 = 0.406 → MEDIUM risk → More appropriate!
```

### 15.4 Why Visual Gets Only 15%?

- Visual analysis is **optional** — it only works when a URL is present AND Selenium can capture a screenshot
- It is a **supplementary check**, not a primary detector — it confirms spoofing but can't detect all phishing types (e.g., credential harvesting on unique-looking pages)
- Screenshot capture adds **latency** (3-5 seconds), so it should enhance but not dominate the score
- It depends on a **finite trusted database** — we can only compare against sites we have captured screenshots of

### 15.5 Summary Table

| Channel | Weight | Role | Justification |
|:--------|:-------|:-----|:--------------|
| **SMS Text** | 0.40 | **Content Analysis** — detects urgency, threats, and phishing language patterns | Strong but can be fooled by clever wording; language-dependent |
| **URL Structure** | 0.45 | **Attack Vector Analysis** — detects the actual malicious link | Objective, mathematical, language-independent, detects zero-day attacks |
| **Visual Forensics** | 0.15 | **Confirmation Analysis** — confirms brand spoofing visually | Optional, dependent on screenshot availability and trusted DB coverage |

---

## 16. SSIM — Detailed Step-by-Step Working with Worked Example

SSIM (Structural Similarity Index Measure) compares two images by evaluating **three perceptual qualities** that the human eye uses to judge similarity.

### 16.1 What SSIM Actually Measures

```
Human eyes judge image similarity based on:
    1. LUMINANCE → Are the images equally bright overall?
    2. CONTRAST  → Do they have similar light-dark variation?
    3. STRUCTURE → Do the edges, patterns, and shapes match?

SSIM combines all three into ONE score between -1 and +1.
```

### 16.2 How SSIM Works in Our System — Step by Step

```
STEP 1: Load both images
    suspect_image  = screenshot of "http://sbi-secure.xyz" (the URL being scanned)
    trusted_image  = saved screenshot of real "https://www.sbi.co.in" 
                     (from data/trusted_screenshots/sbi.png)

STEP 2: Convert to grayscale
    Both images → single-channel grayscale (0 = black, 255 = white)
    Why? SSIM measures structure, not color. Grayscale removes color noise.

STEP 3: Resize suspect to match trusted dimensions (1366×768)
    Both images must be the same size for pixel-by-pixel comparison.

STEP 4: Slide an 11×11 window across BOTH images simultaneously
    At each position, compute local statistics for both images:
    
    Window on suspect image → compute μₓ (mean), σₓ² (variance)
    Window on trusted image → compute μᵧ (mean), σᵧ² (variance)
    Cross-correlation        → compute σₓᵧ (covariance)
    
    Then compute LOCAL SSIM at this window position.

STEP 5: Average all local SSIM values → Final SSIM score
```

### 16.3 Worked Numerical Example (Simplified 4×4 Window)

Let's compute SSIM for a small patch (in reality, the window is 11×11 across the entire image):

**Suspect image patch (from fake SBI site):**
```
Patch x = [180, 185, 190, 175,
           178, 182, 188, 179,
           185, 190, 195, 182,
           170, 175, 180, 172]
```

**Trusted image patch (from real SBI site):**
```
Patch y = [182, 187, 192, 177,
           180, 184, 190, 181,
           183, 188, 193, 180,
           172, 177, 182, 174]
```

**Step A — Compute means:**
```
μₓ = (180+185+190+175+178+182+188+179+185+190+195+182+170+175+180+172) / 16
   = 2906 / 16 = 181.625

μᵧ = (182+187+192+177+180+184+190+181+183+188+193+180+172+177+182+174) / 16
   = 2922 / 16 = 182.625
```

**Step B — Compute variances:**
```
σₓ² = (1/16) × Σ(xᵢ - μₓ)² = (1/16) × [(180-181.625)² + (185-181.625)² + ...]
    = 47.734

σᵧ² = (1/16) × Σ(yᵢ - μᵧ)² = (1/16) × [(182-182.625)² + (187-182.625)² + ...]
    = 37.234
```

**Step C — Compute covariance:**
```
σₓᵧ = (1/16) × Σ(xᵢ - μₓ)(yᵢ - μᵧ)
    = (1/16) × [(180-181.625)(182-182.625) + (185-181.625)(187-182.625) + ...]
    = 41.359
```

**Step D — Compute SSIM constants:**
```
L = 255 (pixel range for 8-bit images)
K₁ = 0.01,  K₂ = 0.03

C₁ = (K₁ × L)² = (0.01 × 255)² = (2.55)² = 6.5025
C₂ = (K₂ × L)² = (0.03 × 255)² = (7.65)² = 58.5225
```

**Step E — Compute SSIM:**
```
Numerator   = (2 × 181.625 × 182.625 + 6.5025)(2 × 41.359 + 58.5225)
            = (2 × 33162.28 + 6.5025)(82.718 + 58.5225)
            = (66331.06)(141.24)
            = 9,368,179.1

Denominator = (181.625² + 182.625² + 6.5025)(47.734 + 37.234 + 58.5225)
            = (32987.64 + 33372.13 + 6.5025)(143.49)
            = (66366.27)(143.49)
            = 9,523,106.8

SSIM = 9,368,179.1 / 9,523,106.8 = 0.9837
```

**Result: SSIM = 0.984** → The patches are **very similar** → this part of the fake site closely mimics the real SBI site.

### 16.4 What Different SSIM Scores Mean

| SSIM Range | Interpretation | Example |
|:-----------|:---------------|:--------|
| 0.95 – 1.00 | **Near-identical** — likely a direct clone/spoof | Pixel-perfect copy of SBI login page |
| 0.80 – 0.95 | **Very similar** — same layout, minor differences | Same design but different text or images |
| 0.60 – 0.80 | **Moderately similar** — suspicious, likely spoofing | Same color scheme and logo placement |
| 0.30 – 0.60 | **Low similarity** — different pages | A banking site vs a shopping site |
| 0.00 – 0.30 | **Very different** — unrelated | Google.com vs a random phishing page |

### 16.5 Our Code Implementation (from `image_comparator.py`)

```python
# Step 1: Read both images with OpenCV
suspect_cv = cv2.imread(suspect_screenshot_path)
trusted_cv = cv2.imread(str(trusted_full_path))

# Step 2: Convert to grayscale
suspect_gray = cv2.cvtColor(suspect_cv, cv2.COLOR_BGR2GRAY)
trusted_gray = cv2.cvtColor(trusted_cv, cv2.COLOR_BGR2GRAY)

# Step 3: Resize suspect to match trusted dimensions
h, w = trusted_gray.shape
suspect_gray = cv2.resize(suspect_gray, (w, h))

# Step 4 & 5: Compute SSIM (scikit-image handles the sliding window internally)
ssim_score, diff = structural_similarity(
    trusted_gray, suspect_gray, full=True
)   # full=True also returns the per-pixel difference map

# Step 6: Generate visual difference heatmap
diff_normalized = ((1 - diff) * 255).astype(np.uint8)
heatmap = cv2.applyColorMap(diff_normalized, cv2.COLORMAP_JET)
# Blue = identical regions, Red = different regions
```

### 16.6 The Difference Heatmap

The `diff` array returned by SSIM contains per-pixel similarity values. We convert it into a **color heatmap**:

```
Blue regions  → SSIM ≈ 1.0 → These areas are IDENTICAL (e.g., copied logo)
Green regions → SSIM ≈ 0.5 → These areas are SIMILAR (e.g., same layout)
Red regions   → SSIM ≈ 0.0 → These areas are DIFFERENT (e.g., changed text)

This heatmap is saved to data/temp/diff_heatmap.png and shown on the frontend
to visually demonstrate WHERE the spoofing differences exist.
```

---

## 17. How Frontend History & Diagrams Work Without a Database

### 17.1 The Guide's Question

*"Your frontend has a history panel and diagrams — how does this work without a database after full implementation?"*

### 17.2 Answer: React Component State (In-Memory)

The scan history and diagrams in our frontend are stored **entirely in React component state** — JavaScript memory within the browser. There is **no database** involved for the frontend history.

```
How it works:

    1. User submits a message for scanning
    2. Frontend sends POST request to Flask API
    3. API returns analysis results (JSON)
    4. Frontend adds the result to a React state array:
       
       scanHistory = [...previousScans, newScanResult]
       
    5. The history panel RENDERS from this state array
    6. The threat chart RENDERS from the same state array

    LIFECYCLE:
        Page Load    → scanHistory = []        (empty)
        Scan 1       → scanHistory = [result1]
        Scan 2       → scanHistory = [result1, result2]
        Scan 3       → scanHistory = [result1, result2, result3]
        Page Refresh → scanHistory = []        (reset — data is lost)
```

### 17.3 Why This Is Acceptable (Not a Limitation)

| Concern | Answer |
|:--------|:-------|
| "Data is lost on refresh" | This is **intentional** — our system is a **real-time detection tool**, not a monitoring dashboard. Each session is independent. Users scan suspicious messages as they receive them. |
| "No persistent history" | For a production deployment, adding `localStorage` or a database would be a straightforward enhancement. The current design prioritizes **zero configuration** — users don't need to set up a database to use the tool. |
| "What about the chart?" | The threat history chart dynamically renders from the session's scan array. Each new scan adds a data point. It shows the **current session's threat trend**, not historical data. |

### 17.4 The Data Flow (Frontend → Backend → Frontend)

```
┌─────────────────────────────────────────────────────────┐
│                   REACT FRONTEND                         │
│                                                          │
│   const [scanHistory, setScanHistory] = useState([])     │
│   const [threatData, setThreatData] = useState([])       │
│                                                          │
│   On scan submit:                                        │
│     1. POST /api/full-scan { message: "..." }           │
│     2. Receive JSON response from Flask API              │
│     3. setScanHistory(prev => [...prev, response])       │
│     4. setThreatData(prev => [...prev, {                 │
│            time: new Date(),                             │
│            score: response.combined_score                │
│        }])                                               │
│                                                          │
│   History Panel:   renders scanHistory.map(...)           │
│   Threat Chart:    renders threatData as line chart       │
│   Stats Cards:     computed from scanHistory              │
│                                                          │
│   ALL DATA LIVES IN JavaScript MEMORY ONLY               │
│   No localStorage, no database, no cookies               │
└─────────────────────────────────────────────────────────┘
```

### 17.5 For Full Production Implementation

If persistent history is needed in a production deployment, two options exist:

```
Option A: localStorage (Simple, no backend changes)
    - Store scan results in browser's localStorage
    - Data persists across page refreshes
    - Data is per-browser (not shared between devices)
    - No server-side database needed

Option B: Database (Full production)
    - Add SQLite/PostgreSQL to Flask backend
    - Store scan results with timestamps and user IDs
    - Add GET /api/history endpoint
    - Frontend fetches history on page load
    - Enables multi-device access and analytics
```

> **Our current design choice:** We deliberately chose stateless in-memory storage to keep the system **lightweight, zero-configuration, and focused on the core ML detection** rather than data management infrastructure.

---

## 18. Image Storage for Visual Forensics — File System vs Database

### 18.1 The Guide's Question

*"Is a database needed for image comparison? The images must be stored somewhere for comparison."*

### 18.2 Answer: File System Storage (No Database Required)

Our system stores trusted reference screenshots as **PNG files on the local file system** — not in a database. This is a deliberate architectural decision.

```
Storage Structure:
    phishing_detection/
    └── data/
        └── trusted_screenshots/
            ├── sbi.png              (1366×768 full screenshot)
            ├── sbi_thumb.png        (256×256 thumbnail for pHash)
            ├── hdfc.png
            ├── hdfc_thumb.png
            ├── icici.png
            ├── icici_thumb.png
            ├── paypal.png
            ├── paypal_thumb.png
            ├── google.png
            ├── google_thumb.png
            └── ... (10 trusted sites × 2 versions = 20 files)
```

### 18.3 How the Images Are Created

The `build_trusted_db.py` script captures reference screenshots:

```
ALGORITHM: BuildTrustedDatabase()

1. DEFINE trusted sites dictionary:
   TRUSTED_SITES = {
       "sbi":      "https://www.sbi.co.in",
       "hdfc":     "https://www.hdfcbank.com",
       "icici":    "https://www.icicibank.com",
       "paypal":   "https://www.paypal.com",
       "google":   "https://accounts.google.com",
       "facebook": "https://www.facebook.com",
       "amazon":   "https://www.amazon.com",
       ... (10 total)
   }

2. START headless Chrome browser (via Selenium)
   Window size: 1366×768

3. FOR each site in TRUSTED_SITES:
       a. Navigate to the URL
       b. Wait 5 seconds for full page load
       c. Capture full screenshot → save as {site_key}.png
       d. Create 256×256 thumbnail → save as {site_key}_thumb.png

4. CLOSE browser

This is run ONCE during setup (python build_visual_db.py)
The screenshots are then used as the trusted baseline for all future comparisons.
```

### 18.4 How Images Are Used During Comparison

```
When a user scans a message with a URL:

1. Selenium captures a LIVE screenshot of the suspicious URL
   → saved to data/temp/suspect_screenshot.png (temporary)

2. pHash Stage:
   - Load suspect_thumb (256×256 resize)
   - Compare against ALL trusted thumbnails (sbi_thumb.png, hdfc_thumb.png, ...)
   - Find closest match by Hamming distance

3. SSIM Stage (if pHash distance ≤ 30):
   - Load full suspect screenshot (1366×768)
   - Load full trusted screenshot of the best match (e.g., sbi.png)
   - Compute SSIM between the two full images
   - Generate difference heatmap

4. Temporary suspect screenshot is overwritten on next scan
   (no permanent storage of suspect images)
```

### 18.5 Why File System and Not a Database?

| Aspect | File System (Our Choice) | Database (Alternative) |
|:-------|:------------------------|:-----------------------|
| **Speed** | Direct file I/O is **fastest** for large binary files like images | DB requires encoding/decoding overhead (BLOB storage) |
| **Simplicity** | No database setup needed — just a folder with PNG files | Requires DB installation, schema, connection handling |
| **OpenCV compatibility** | OpenCV's `cv2.imread()` reads files directly from disk | Would need to extract from DB → save to temp → read with OpenCV |
| **pHash library** | `imagehash` library works directly with PIL/file paths | Same extraction overhead as OpenCV |
| **Scale** | We have ~10 trusted sites = ~20 files (~5 MB total) — a database is overkill | Database makes sense for 1000+ images |
| **Portability** | Copy the folder → system works anywhere | Need to export/import database |
| **Updates** | Simply re-run `build_visual_db.py` to refresh screenshots | Need migration scripts |

### 18.6 When Would a Database Be Needed?

```
A database WOULD be needed if:
    1. Scaling to 100+ trusted sites → file management becomes complex
    2. Multiple users submitting their own trusted baselines
    3. Version tracking (keeping old screenshots for comparison over time)
    4. Distributed deployment (multiple servers need access to the same images)
    5. Storing scan results + screenshots for audit/compliance

For our project scope (10 trusted sites, single-server):
    File system is the OPTIMAL choice — simpler, faster, and equally reliable.
```

> **Key point for the guide:** "The images are stored as PNG files in the `data/trusted_screenshots/` directory — not in a database. We chose file system storage because OpenCV and imagehash libraries read directly from disk, making it the fastest option. For our scale of 10 trusted sites (~5 MB total), a database would add unnecessary complexity without any benefit."

---

> **Summary for Guide:** "Sir/Ma'am, our system uses a **Multi-Channel Pipeline** where each input is decomposed into 3 independent analysis channels. SMS text undergoes an NLP pipeline (cleaning → tokenization → stopword removal → stemming → TF-IDF vectorization into 500 features + 19 statistical features = 519 features), which is fed into a Random Forest of 100 trees. Each tree node is split using **Gini Impurity minimization** on a random subset of √519 ≈ 23 features. URLs are analyzed by extracting 30 structural features (including Shannon Entropy for randomness detection) and fed into a separate Random Forest of 200 trees. Visual analysis uses a two-stage approach: pHash for fast pre-filtering and SSIM for deep structural comparison. The three channel scores are fused using normalized weighted averaging (SMS 40%, URL 45%, Visual 15%) to produce the final threat score. URL receives the highest weight because it is the **actual attack vector** — the mechanism through which phishing damage occurs. SSIM works by sliding an 11×11 window across grayscale images, computing local luminance, contrast, and structure similarity at each position, then averaging. Our frontend history is stored in React state (JavaScript memory) — no database needed. Trusted screenshots are stored as PNG files on the file system, not in a database, because OpenCV and pHash libraries read directly from disk."

