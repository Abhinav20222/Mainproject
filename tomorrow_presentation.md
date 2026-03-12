# Tomorrow Presentation — Model Explanations & Random Forest

---

## 1. Naive Bayes — Simple Explanation

**Analogy**: A detective who counts clue frequencies.

```
How it works:
  1. During training, it counts:
     - How often each word appears in SPAM messages
     - How often each word appears in HAM messages

  2. For a new message "urgent click verify account":
     - P(spam | urgent) = 0.85  (85% of spam has "urgent")
     - P(spam | click)  = 0.72
     - P(spam | verify) = 0.90
     - P(spam | account)= 0.65
     
     Multiply: 0.85 × 0.72 × 0.90 × 0.65 = high probability → SPAM

  Formula: P(spam | words) = P(words | spam) × P(spam) / P(words)
           (Bayes Theorem)
```

**Weakness**: Assumes every word is **independent** — but "click" + "verify" together is WAY more suspicious than each alone. Naive Bayes misses this.

---

## 2. Logistic Regression — Simple Explanation

**Analogy**: A scoring system with weights.

```
How it works:
  1. Assigns a WEIGHT to each feature:
     urgency_count × 2.5 + has_url × 3.1 + digit_ratio × 1.2 + ... = total score

  2. Passes total score through sigmoid function:
     P(spam) = 1 / (1 + e^(-score))
     
     If score = +5  → P(spam) = 0.99 → SPAM
     If score = -5  → P(spam) = 0.01 → HAM
     If score =  0  → P(spam) = 0.50 → uncertain

  3. It draws a STRAIGHT LINE to separate spam from ham
```

**Weakness**: Can only draw a **straight line** boundary. If phishing patterns are complex (e.g., "URL + urgency = spam, but URL alone = not spam"), it can't capture this.

---

## 3. SVM (Support Vector Machine) — Simple Explanation

**Analogy**: Finding the widest road between two villages.

```
How it works:
  1. Plots all messages as points in 519-dimensional space
  
  2. Finds the BEST line (hyperplane) that:
     - Separates spam points from ham points
     - Maximizes the MARGIN (gap) between the closest points

     HAM points ●  ●  ●  |  MARGIN  |  ▲  ▲  ▲ SPAM points
                ●  ●     |◄───────►|     ▲  ▲
                   ●     |  widest  |  ▲
                         |  gap     |
                    ← HAM side    SPAM side →

  3. The closest points to the line are called "Support Vectors"
```

**Weakness**: Very **slow** on large datasets (519 features × 5572 messages). Also can't easily handle non-linear patterns with a linear kernel.

---

## 4. Random Forest — Complete Algorithm (Step by Step)

**Analogy**: 100 doctors voting on a diagnosis.

```
RANDOM FOREST ALGORITHM:

TRAINING:
═══════════════════════════════════════════════════
Step 1: Create 100 empty trees (SMS) / 200 trees (URL)

Step 2: For EACH tree (Tree 1, Tree 2, ... Tree 100):
   │
   ├── 2a. BOOTSTRAP: Randomly sample messages WITH replacement
   │        (some messages picked multiple times, some left out)
   │        This gives each tree a DIFFERENT training set
   │
   ├── 2b. BUILD TREE (starting from root):
   │    │
   │    └── At EACH node:
   │         ├── Randomly select 23 features (√519) from 519 total
   │         │   (for URL: √28 ≈ 5 features from 28 total)
   │         │
   │         ├── For each of the 23 features, find best split point:
   │         │     "Is urgency_count > 2?"  → Gini improvement = 0.15
   │         │     "Is tfidf_127 > 0.3?"    → Gini improvement = 0.22 ← BEST
   │         │     "Is has_url > 0?"         → Gini improvement = 0.18
   │         │     ... test all 23 ...
   │         │
   │         ├── USE the best split: tfidf_127 > 0.3?
   │         │     ├── YES → go to left child node (repeat splitting)
   │         │     └── NO  → go to right child node (repeat splitting)
   │         │
   │         └── STOP when:
   │              - Node is pure (all spam or all ham, Gini = 0)
   │              - Or min_samples_split reached
   │              → This becomes a LEAF node (final answer)
   │
   └── 2c. Tree is complete! (might have 50-200+ nodes)

Step 3: Save all 100 trees


PREDICTION (for a new message):
═══════════════════════════════════════════════════
Step 1: Extract 519 features from the new message
        (500 TF-IDF + 19 statistical features)

Step 2: Send features through ALL 100 trees:
   Tree 1:  features → traverse nodes → reaches leaf → SPAM
   Tree 2:  features → traverse nodes → reaches leaf → HAM
   Tree 3:  features → traverse nodes → reaches leaf → SPAM
   Tree 4:  features → traverse nodes → reaches leaf → SPAM
   ...
   Tree 100: features → traverse nodes → reaches leaf → SPAM

Step 3: MAJORITY VOTE:
   SPAM votes: 87 trees
   HAM votes:  13 trees
   → Final answer: SPAM (87% confidence)
```

### Key Parameters:
| Parameter | SMS Model | URL Model |
|-----------|-----------|-----------|
| n_estimators (trees) | 100 | 200 |
| Total features | 519 | 28 |
| Features per node (√total) | 23 | 5 |
| max_depth | None (unlimited) | None (unlimited) |
| class_weight | default | 'balanced' |

---

## 5. Why Random Forest Over Others? (SMS — 4 models compared)

| Criteria | Naive Bayes | Logistic Regression | SVM | **Random Forest** ✅ |
|----------|------------|-------------------|-----|------------------|
| Handles word combinations? | ❌ Assumes independent | ❌ Linear only | ❌ Linear kernel | ✅ **Trees capture interactions** |
| Works with mixed features? (TF-IDF + statistics) | ❌ Struggles | ⚠️ Needs scaling | ⚠️ Needs scaling | ✅ **Handles naturally** |
| Explains WHY it flagged? | ❌ | ❌ | ❌ | ✅ **feature_importances_** |
| Speed for real-time API? | ✅ Fast | ✅ Fast | ❌ Slow | ✅ **Fast (parallel)** |
| Overfitting protection? | ⚠️ Naive assumption | ⚠️ Can overfit | ⚠️ Can overfit | ✅ **100 trees average out errors** |
| Cross-validation stability? | Moderate | Moderate | Low | ✅ **Most stable** |

---

## 6. Why Random Forest Over Others? (URL — 3-4 models compared)

| Criteria | Logistic Regression | Gradient Boosting | XGBoost | **Random Forest** ✅ |
|----------|-------------------|------------------|---------|------------------|
| Handles non-linear URL patterns? | ❌ Linear only | ✅ Yes | ✅ Yes | ✅ **Yes** |
| Training speed? | ✅ Fast | ❌ Slow (sequential) | ⚠️ Medium | ✅ **Fast (parallel)** |
| Overfitting risk? | Low but underfits | ⚠️ High risk | ⚠️ Moderate | ✅ **Low (bagging)** |
| Handles class imbalance? | ⚠️ Needs tuning | ❌ No built-in | ⚠️ Needs tuning | ✅ **class_weight='balanced'** |
| Feature importance? | ⚠️ Coefficients only | ✅ Yes | ✅ Yes | ✅ **Yes** |
| Prediction speed for API? | ✅ Fast | ⚠️ Sequential trees | ⚠️ Sequential | ✅ **Parallel trees** |

---

## 7. Quick Answers for Common Follow-up Questions

### "How many epochs?"
Random Forest does NOT use epochs. Epochs are for neural networks (deep learning). Random Forest uses `n_estimators` (number of trees): 100 for SMS, 200 for URL. Each tree trains in a single pass.

### "How does a tree know when to stop?"
A tree stops splitting when:
- The node is **pure** (Gini = 0, all samples are same class)
- Or `min_samples_split` is reached (default = 2)
- Or no feature split can improve Gini further

### "What are the 23 features per node?"
- Total features: 519 (SMS) or 28 (URL)
- At each node: √519 ≈ 23 random features selected (SMS) or √28 ≈ 5 (URL)
- This is NOT the number of nodes — each tree has many nodes
- Each node picks a NEW random set of features

### "Is the 500-word vocabulary fixed?"
Yes. The 500 TF-IDF words are selected ONCE during training, saved in `feature_extractor.pkl`, and reused for every prediction. They never change unless you retrain.

### "Why so many zeros in sms_features.csv?"
Normal! Each message uses only ~10-20 words out of 500. The other ~480 columns are 0 because those words don't appear in that message. This is called a "sparse matrix."

---

## 8. Summary — One Line Answers

| Topic | Answer |
|-------|--------|
| Naive Bayes | Probability-based, counts word frequencies, assumes independence |
| Logistic Regression | Weighted sum + sigmoid, draws straight line boundary |
| SVM | Finds widest margin hyperplane, slow on large data |
| Random Forest | 100 trees × majority vote, each node picks 23 random features |
| Why RF for SMS? | Non-linear, mixed features, feature importance, fast, stable |
| Why RF for URL? | Parallel, handles imbalance, low overfitting, interpretable |
| Epochs? | None — RF uses n_estimators (trees), not epochs |
| Tree stopping? | When node is pure (Gini=0) or can't improve further |

---

# URL TRAINING — Models Explained (Same Style as SMS)

> **Key Difference**: URL training uses **28 structural features** (url_length, num_dots, has_ip_address, etc.) — NOT TF-IDF words. The models receive 28 numbers, not 519.

---

## URL Model 1: Logistic Regression — Simple Explanation

**Same concept as SMS**, but applied to URL features.

```
How it works for URL:
  1. Assigns a WEIGHT to each of the 28 URL features:
     url_length × 0.8 + num_dots × 1.5 + has_ip_address × 4.2 
     + has_https × (-2.1) + num_subdomains × 1.9 + ... = total score

  2. Passes total score through sigmoid function:
     P(phishing) = 1 / (1 + e^(-score))
     
     If score = +5  → P(phishing) = 0.99 → PHISHING
     If score = -5  → P(phishing) = 0.01 → LEGITIMATE
     If score =  0  → P(phishing) = 0.50 → uncertain

  3. Draws a STRAIGHT LINE to separate phishing from legit URLs

  Config: max_iter=1000, class_weight='balanced'
  'balanced' = gives more importance to the minority class
```

**Weakness for URL**: Phishing patterns are NON-LINEAR. Example:
- `url_length=30 + has_https=0` → might be phishing
- `url_length=30 + has_https=1` → probably legitimate
- Logistic Regression can't capture this interaction — it treats each feature independently

---

## URL Model 2: Gradient Boosting — Simple Explanation

**Analogy**: A team of teachers where each new teacher corrects the mistakes of the previous one.

```
How it works:
  1. Tree 1: Tries to classify all URLs → makes some mistakes
     Correctly classified: 85%
     Incorrectly classified: 15% ← these become the FOCUS

  2. Tree 2: Focuses MORE on the 15% that Tree 1 got wrong
     → Corrects some of Tree 1's mistakes
     Still wrong: 8% ← these become the new focus

  3. Tree 3: Focuses MORE on the 8% that Trees 1+2 still got wrong
     → Corrects more mistakes

  ... continues for 200 trees (n_estimators=200) ...

  Final prediction = weighted combination of ALL 200 trees

  Key difference from Random Forest:
  - Random Forest: Trees built INDEPENDENTLY (parallel)
  - Gradient Boosting: Trees built SEQUENTIALLY (each depends on previous)
```

```
  Random Forest:                    Gradient Boosting:
  Tree1  Tree2  Tree3  Tree4       Tree1 → Tree2 → Tree3 → Tree4
    ↓      ↓      ↓      ↓         "fix    "fix    "fix    "fix
    ↓      ↓      ↓      ↓          tree1   tree2   tree3   tree4
  INDEPENDENT (parallel)             errors" errors" errors" errors"
  → MAJORITY VOTE                    SEQUENTIAL (one after another)
                                     → WEIGHTED SUM
```

**Weakness**: 
- **Slow training** (trees are sequential, can't use parallel processing)
- **High overfitting risk** — it tries SO hard to fix mistakes that it can memorize training data
- **No built-in class imbalance handling** (doesn't have `class_weight='balanced'`)

---

## URL Model 3: XGBoost (eXtreme Gradient Boosting) — Simple Explanation

**Analogy**: An OPTIMIZED version of Gradient Boosting — same idea, but faster and smarter.

```
How it works:
  Same as Gradient Boosting, BUT with extra optimizations:

  1. Built-in REGULARIZATION:
     - L1 (Lasso): forces some feature weights to zero → ignores useless features
     - L2 (Ridge): keeps weights small → prevents overfitting
     Gradient Boosting doesn't have this built-in

  2. Handles MISSING VALUES automatically:
     - If a URL has a missing feature, XGBoost decides which branch to go
     - Gradient Boosting would crash on missing values

  3. FASTER because:
     - Uses "histogram-based" splitting (groups similar values into bins)
     - Parallel computation of each tree's splits
     - Cache-aware data access

  Config: n_estimators=200, eval_metric='logloss', verbosity=0
```

**Weakness**: 
- Still **sequential** trees (each tree depends on previous)
- Needs **careful hyperparameter tuning** (learning_rate, max_depth, etc.)
- **Optional** in our project — only used if `xgboost` package is installed

---

## URL Model 4: Random Forest ✅ (Winner for URL)

**Same algorithm as SMS** but with different parameters:

```
URL Random Forest Algorithm:

TRAINING:
═══════════════════════════════════════════════════
Step 1: Create 200 empty trees (more trees than SMS because fewer features)

Step 2: For EACH tree:
   ├── 2a. BOOTSTRAP: Randomly sample URLs WITH replacement
   │        from the training dataset
   │
   ├── 2b. BUILD TREE:
   │    └── At EACH node:
   │         ├── Randomly select 5 features (√28) from 28 total
   │         │   (SMS uses 23 from 519, URL uses 5 from 28)
   │         │
   │         ├── For each of the 5 features, find best split:
   │         │     "Is url_length > 54?"           → Gini = 0.12
   │         │     "Is has_ip_address > 0?"         → Gini = 0.25 ← BEST
   │         │     "Is num_subdomains > 2?"         → Gini = 0.18
   │         │     "Is has_suspicious_words > 0?"   → Gini = 0.20
   │         │     "Is hostname_entropy > 3.5?"     → Gini = 0.15
   │         │
   │         ├── USE best: has_ip_address > 0?
   │         │     ├── YES → left child (likely phishing)
   │         │     └── NO  → right child (keep splitting)
   │         │
   │         └── STOP when node is pure
   │
   └── 2c. Tree complete!

Step 3: Save all 200 trees

PREDICTION (for a new URL):
═══════════════════════════════════════════════════
Step 1: Extract 28 features from the URL
        (url_length, num_dots, has_ip, has_https, entropy, etc.)

Step 2: Send through ALL 200 trees:
   Tree 1:   28 features → traverse → PHISHING
   Tree 2:   28 features → traverse → PHISHING
   Tree 3:   28 features → traverse → LEGITIMATE
   ...
   Tree 200: 28 features → traverse → PHISHING

Step 3: MAJORITY VOTE:
   PHISHING votes:   168 trees (84%)
   LEGITIMATE votes:  32 trees (16%)
   → Final answer: PHISHING (84% confidence)

EXTRA: class_weight='balanced'
   - Our dataset has MORE phishing URLs than legitimate ones
   - 'balanced' gives HIGHER weight to the minority class (legitimate)
   - Without this, model would just predict "phishing" for everything
```

---

## Why Random Forest Over Others for URL?

### Problem 1: Phishing URL patterns are NON-LINEAR
```
Example:
  url_length=25, has_https=1, num_dots=1  → google.com = LEGIT
  url_length=85, has_https=0, num_dots=5  → paypal.secure-login.evil.xyz = PHISHING

  But also:
  url_length=15, has_https=0, num_dots=1  → bit.ly/x3k = SUSPICIOUS (shortened!)

  Logistic Regression draws ONE straight line → can't handle this complexity
  Random Forest trees can create complex if-then rules for each case
```

### Problem 2: Class Imbalance (more phishing than legit)
```
  Logistic Regression: needs manual adjustment
  Gradient Boosting:   no built-in support
  XGBoost:            needs scale_pos_weight parameter
  Random Forest:      class_weight='balanced' ← one line, automatic!
```

### Problem 3: Training Speed
```
  Gradient Boosting: Tree1 → Tree2 → Tree3 → ... → Tree200 (sequential, SLOW)
  XGBoost:          Tree1 → Tree2 → Tree3 → ... → Tree200 (faster but still sequential)
  Random Forest:    Tree1, Tree2, Tree3, ..., Tree200 ALL at once (parallel, FAST)
                    n_jobs=-1 uses ALL CPU cores
```

### Problem 4: Overfitting
```
  Gradient Boosting: Tries hard to fix every mistake → memorizes noise → overfits
  XGBoost:          Better (has regularization) but still sequential → moderate risk
  Logistic Regression: Underfits (too simple for URL patterns)
  Random Forest:    Each tree overfits a little, but 200 trees AVERAGE OUT the noise
                    → Bagging = Bootstrap Aggregation = overfitting protection
```

### Final Comparison for URL:

| What Matters | Logistic Reg | Gradient Boosting | XGBoost | **Random Forest ✅** |
|-------------|-------------|------------------|---------|---------------------|
| Non-linear patterns | ❌ | ✅ | ✅ | ✅ |
| Class imbalance | ⚠️ manual | ❌ none | ⚠️ manual | ✅ automatic |
| Training speed | ✅ fast | ❌ slow | ⚠️ medium | ✅ fast (parallel) |
| Overfitting risk | underfits | ⚠️ high | ⚠️ moderate | ✅ low (bagging) |
| Feature importance | ⚠️ | ✅ | ✅ | ✅ |
| API prediction speed | ✅ | ⚠️ sequential | ⚠️ sequential | ✅ parallel |
| **WINNER?** | ❌ | ❌ | ❌ | ✅ **YES** |

---

# THREAT SCORE CALCULATION — Line by Line

**File**: `src/api.py` → `/api/full-scan` endpoint (Lines 266–411)

---

## Step 1: Define the Weight System (Line 315)

```python
weights = {'sms': 0.40, 'url': 0.45, 'visual': 0.15}
scores = {}
```

```
Why these weights?
  ┌─────────────────────────────────────────────┐
  │  URL Analysis:  45% (most important)         │
  │  → URL structure is the strongest indicator  │
  │  → A phishing URL is definitive proof        │
  │                                               │
  │  SMS Analysis:  40% (second most important)   │
  │  → Text content reveals manipulation tactics │
  │  → Urgency words, threats, financial keywords │
  │                                               │
  │  Visual:        15% (supplementary)           │
  │  → Screenshot comparison is optional          │
  │  → Detects visual spoofing (looks like bank) │
  └─────────────────────────────────────────────┘
  Total: 0.40 + 0.45 + 0.15 = 1.00 (100%)
```

---

## Step 2: SMS Threat Score (Lines 318–328)

```python
if message and model_cache.is_ready:
    sms_result = model_cache.predict(message)
    # model_cache.predict() does:
    #   1. Preprocess message (clean, tokenize, stem)
    #   2. Extract features (TF-IDF 500 + 19 statistical = 519)
    #   3. Random Forest predicts: 0 (ham) or 1 (spam)
    #   4. predict_proba() gives confidence: e.g., [0.13, 0.87]
    #      → 87 trees said spam, 13 said ham
    #   5. Returns threat_score = 0.87 (probability of spam)

    sms_score = sms_result.get('threat_score', 0)
    # Gets the threat score from the prediction result

    if sms_score > 1:
        sms_score = sms_score / 100.0
    # NORMALIZE: Some predictions return 0-100, some return 0-1
    # If score is 87, divide by 100 → 0.87
    # If score is 0.87, keep as is

    scores['sms'] = sms_score
    # scores = {'sms': 0.87}
```

**SMS Threat Score = probability that the message is spam (from Random Forest's predict_proba)**

---

## Step 3: URL Threat Score (Lines 332–354)

```python
if extracted_urls:
    predictor = get_url_predictor()
    if predictor and predictor.is_ready:
        best_url_result = None
        best_url_score = -1
        # Start with -1 so any real score will be higher

        for u in extracted_urls:
            url_result = predictor.predict(u)
            # predictor.predict() does:
            #   1. Extract 28 structural features from URL
            #   2. Random Forest predicts: 0 (legit) or 1 (phishing)
            #   3. predict_proba() gives: e.g., [0.06, 0.94]
            #      → 188 trees said phishing, 12 said legit
            #   4. Returns threat_score = 0.94

            score = url_result.get('threat_score', 0)

            if score > best_url_score:
                best_url_score = score
                best_url_result = url_result
            # KEEP THE WORST (highest) URL score
            # If message has 3 URLs:
            #   URL1: 0.20 (safe)
            #   URL2: 0.94 (phishing!) ← KEPT
            #   URL3: 0.35 (moderate)
            # best_url_score = 0.94

        scores['url'] = best_url_score
        # scores = {'sms': 0.87, 'url': 0.94}
```

**URL Threat Score = probability of the MOST dangerous URL being phishing**

**Why keep the worst?** A message is as dangerous as its most dangerous URL. If even ONE URL is malicious, the whole message is a threat.

---

## Step 4: Visual Threat Score (Lines 356–368) — OPTIONAL

```python
if extracted_urls and include_visual:
    # Only runs if user checked "Include Visual Analysis" checkbox

    capturer = get_screenshot_capturer()
    comparator = get_image_comparator()

    screenshot_path = capturer.capture(extracted_urls[0])
    # Uses Selenium WebDriver to:
    #   1. Open a headless Chrome browser
    #   2. Navigate to the first URL
    #   3. Wait for page to load
    #   4. Take a screenshot → saves as PNG file
    # Only captures FIRST URL (screenshots are slow)

    vis_result = comparator.compare(screenshot_path)
    # Compares screenshot against database of trusted sites using:
    #   SSIM (Structural Similarity Index):
    #     - Compares pixel patterns between images
    #     - High similarity to a bank site = suspicious!
    #   pHash (Perceptual Hash):
    #     - Creates a fingerprint of the image
    #     - Compares fingerprints for visual similarity
    #
    # If phishing site LOOKS like SBI bank → visual_threat_score = 0.80
    # If site looks original/unique → visual_threat_score = 0.10

    scores['visual'] = vis_result.get('visual_threat_score', 0)
    # scores = {'sms': 0.87, 'url': 0.94, 'visual': 0.80}
```

---

## Step 5: COMBINE — The Final Score Calculation (Lines 370–380)

This is the MOST IMPORTANT part:

```python
if scores:
    active_weights = {k: weights[k] for k in scores}
    # Only include weights for analyses that were actually performed
    #
    # If ALL 3 were done:
    #   active_weights = {'sms': 0.40, 'url': 0.45, 'visual': 0.15}
    #
    # If only SMS + URL done (no visual):
    #   active_weights = {'sms': 0.40, 'url': 0.45}
    #
    # If only SMS done (no URLs in message):
    #   active_weights = {'sms': 0.40}

    total_weight = sum(active_weights.values())
    # Total of active weights:
    #   All 3: 0.40 + 0.45 + 0.15 = 1.00
    #   SMS + URL: 0.40 + 0.45 = 0.85
    #   SMS only: 0.40

    combined_score = sum(
        scores[k] * (active_weights[k] / total_weight)
        for k in scores
    )
    # RE-NORMALIZE weights so they sum to 1.0
    # Then multiply each score by its normalized weight
```

---

## THE MATH — All 3 Scenarios:

### Scenario 1: All 3 analyses performed (SMS + URL + Visual)
```
scores  = {'sms': 0.87, 'url': 0.94, 'visual': 0.80}
weights = {'sms': 0.40, 'url': 0.45, 'visual': 0.15}
total_weight = 0.40 + 0.45 + 0.15 = 1.00

combined = sms_score × (0.40/1.00) + url_score × (0.45/1.00) + visual × (0.15/1.00)
combined = 0.87 × 0.40    +    0.94 × 0.45    +    0.80 × 0.15
combined = 0.348           +    0.423           +    0.120
combined = 0.891

→ Risk Level: CRITICAL (≥ 0.85)
```

### Scenario 2: SMS + URL only (no visual checkbox)
```
scores  = {'sms': 0.87, 'url': 0.94}
weights = {'sms': 0.40, 'url': 0.45}           ← visual excluded
total_weight = 0.40 + 0.45 = 0.85

RE-NORMALIZE: weights must sum to 1.0
  sms_weight = 0.40 / 0.85 = 0.4706 (47.06%)
  url_weight = 0.45 / 0.85 = 0.5294 (52.94%)
                              ──────
                              1.0000 ✓

combined = 0.87 × 0.4706  +  0.94 × 0.5294
combined = 0.4094          +  0.4976
combined = 0.907

→ Risk Level: CRITICAL (≥ 0.85)
```

### Scenario 3: SMS only (no URLs found in message)
```
scores  = {'sms': 0.87}
weights = {'sms': 0.40}                         ← url and visual excluded
total_weight = 0.40

RE-NORMALIZE:
  sms_weight = 0.40 / 0.40 = 1.0 (100%)

combined = 0.87 × 1.0
combined = 0.87

→ Risk Level: HIGH (0.60–0.85)
```

### Scenario 4: Safe message "Hey, how are you?"
```
scores  = {'sms': 0.05}                         ← very low threat
total_weight = 0.40

combined = 0.05 × (0.40/0.40) = 0.05 × 1.0 = 0.05

→ Risk Level: LOW (< 0.30)
```

---

## Step 6: Determine Risk Level (Lines 382–390)

```python
if combined_score < 0.3:
    risk_level = "LOW"        # 0% to 29% — safe message
elif combined_score < 0.6:
    risk_level = "MEDIUM"     # 30% to 59% — suspicious
elif combined_score < 0.85:
    risk_level = "HIGH"       # 60% to 84% — likely phishing
else:
    risk_level = "CRITICAL"   # 85% to 100% — definitely phishing
```

```
Risk Level Scale:
0%──────────30%──────────60%──────────85%──────────100%
│    LOW     │   MEDIUM   │    HIGH    │  CRITICAL  │
│   Safe ✅  │ Suspicious │  Likely ⚠️ │ Phishing 🚨│
│  (green)   │  (yellow)  │  (orange)  │   (red)    │
```

---

## Step 7: Return Results (Lines 394–405)

```python
return jsonify({
    'success': True,
    'combined_threat_score': round(combined_score, 4),  # e.g., 0.891
    'risk_level': risk_level,                            # e.g., "CRITICAL"
    'sms_analysis': sms_analysis,       # Full SMS result details
    'url_analysis': url_analysis,       # Full URL result details
    'visual_analysis': visual_analysis, # Full visual result (if done)
    'analyses_performed': analyses_performed,  # ['sms', 'url', 'visual']
    'score_weights': {                  # Actual weights used after normalization
        'sms': 0.40,
        'url': 0.53,                    # Re-normalized if visual not done
    },
    'total_analysis_time_ms': 125.5,    # Total processing time
})
# This JSON response is sent back to the React frontend
# Frontend reads combined_threat_score and risk_level to display results
```

---

## Complete Data Flow Diagram:

```
Message: "URGENT! Click http://paypal.secure-login.xyz to verify account"

STEP 1: Extract URL from message
   → Found: ["http://paypal.secure-login.xyz"]

STEP 2: SMS Analysis (weight: 40%)
   "URGENT! Click http://paypal.secure-login.xyz to verify account"
    → Preprocess → TF-IDF + features → Random Forest
    → threat_score = 0.87 (87% probability of spam)

STEP 3: URL Analysis (weight: 45%)
   "http://paypal.secure-login.xyz"
    → Extract 28 features → Random Forest
    → threat_score = 0.94 (94% probability of phishing)

STEP 4: Visual Analysis (weight: 15%) — if enabled
   Screenshot of paypal.secure-login.xyz
    → Compare with real PayPal screenshot
    → visual_threat_score = 0.80 (looks 80% similar to PayPal)

STEP 5: Combine (re-normalized)
   ┌────────────────────────────────────────────────────────────┐
   │ combined = 0.87×0.40 + 0.94×0.45 + 0.80×0.15             │
   │ combined = 0.348 + 0.423 + 0.120                          │
   │ combined = 0.891                                           │
   │                                                            │
   │ Risk Level: CRITICAL (0.891 ≥ 0.85)                       │
   └────────────────────────────────────────────────────────────┘

STEP 6: Send to Frontend → Red "CRITICAL" alert displayed
```
 Naive Bayes — Simple Explanation
How it works:
  1. During training, it counts:
     - How often each word appears in SPAM messages
     - How often each word appears in HAM messages

  2. For a new message "urgent click verify account":
     - P(spam | urgent) = 0.85  (85% of spam has "urgent")
     - P(spam | click)  = 0.72
     - P(spam | verify) = 0.90
     - P(spam | account)= 0.65
     
     Multiply: 0.85 × 0.72 × 0.90 × 0.65 = high probability → SPAM

  Formula: P(spam | words) = P(words | spam) × P(spam) / P(words)
           (Bayes Theorem)

           Weakness: Assumes every word is independent — but "click" + "verify" together is WAY more suspicious than each alone. Naive Bayes misses this.



Logistic Regression 

How it works:
  1. Assigns a WEIGHT to each feature:
     urgency_count × 2.5 + has_url × 3.1 + digit_ratio × 1.2 + ... = total score

  2. Passes total score through sigmoid function:
     P(spam) = 1 / (1 + e^(-score))
     
     If score = +5  → P(spam) = 0.99 → SPAM
     If score = -5  → P(spam) = 0.01 → HAM
     If score =  0  → P(spam) = 0.50 → uncertain

  3. It draws a STRAIGHT LINE to separate spam from ham
  Weakness: Can only draw a straight line boundary. If phishing patterns are complex (e.g., "URL + urgency = spam, but URL alone = not spam"), it can't capture this.

  SVM (Support Vector Machine) — Simple Explanation
  How it works:
  1. Plots all messages as points in 519-dimensional space
  
  2. Finds the BEST line (hyperplane) that:
     - Separates spam points from ham points
     - Maximizes the MARGIN (gap) between the closest points

     HAM points ●  ●  ●  |  MARGIN  |  ▲  ▲  ▲ SPAM points
                ●  ●     |◄───────►|     ▲  ▲
                   ●     |  widest  |  ▲
                         |  gap     |
                    ← HAM side    SPAM side →

  3. The closest points to the line are called "Support Vectors"
  Weakness: Very slow on large datasets (519 features × 5572 messages). Also can't easily handle non-linear patterns with a linear kernel.

  Random Forest — Complete Algorithm (Step by Step)
  Analogy: 100 doctors voting on a diagnosis.

  RANDOM FOREST ALGORITHM:

  TRAINING:
  ═══════════════════════════════════════════════════
  Step 1: Create 100 empty trees (SMS) / 200 trees (URL)

  Step 2: For EACH tree (Tree 1, Tree 2, ... Tree 100):
   │
   ├── 2a. BOOTSTRAP: Randomly sample messages WITH replacement
   │        (some messages picked multiple times, some left out)
   │        This gives each tree a DIFFERENT training set
   │
   ├── 2b. BUILD TREE (starting from root):
   │    │
   │    └── At EACH node:
   │         ├── Randomly select 23 features (√519) from 519 total
   │         │   (for URL: √28 ≈ 5 features from 28 total)
   │         │
   │         ├── For each of the 23 features, find best split point:
   │         │     "Is urgency_count > 2?"  → Gini improvement = 0.15
   │         │     "Is tfidf_127 > 0.3?"    → Gini improvement = 0.22 ← BEST
   │         │     "Is has_url > 0?"         → Gini improvement = 0.18
   │         │     ... test all 23 ...
   │         │
   │         ├── USE the best split: tfidf_127 > 0.3?
   │         │     ├── YES → go to left child node (repeat splitting)
   │         │     └── NO  → go to right child node (repeat splitting)
   │         │
   │         └── STOP when:
   │              - Node is pure (all spam or all ham, Gini = 0)
   │              - Or min_samples_split reached
   │              → This becomes a LEAF node (final answer)
   │
   └── 2c. Tree is complete! (might have 50-200+ nodes)

  Step 3: Save all 100 trees


  PREDICTION (for a new message):
  ═══════════════════════════════════════════════════
  Step 1: Extract 519 features from the new message
        (500 TF-IDF + 19 statistical features)

  Step 2: Send features through ALL 100 trees:
   Tree 1:  features → traverse nodes → reaches leaf → SPAM
   Tree 2:  features → traverse nodes → reaches leaf → HAM
   Tree 3:  features → traverse nodes → reaches leaf → SPAM
   Tree 4:  features → traverse nodes → reaches leaf → SPAM
   ...
   Tree 100: features → traverse nodes → reaches leaf → SPAM

  Step 3: MAJORITY VOTE:
   SPAM votes: 87 trees
   HAM votes:  13 trees
   → Final answer: SPAM (87% confidence)

  Key Parameters:
  Parameter | SMS Model | URL Model
  -----------|-----------|-----------
  n_estimators (trees) | 100 | 200
  Total features | 519 | 28
  Features per node (√total) | 23 | 5
  max_depth | None (unlimited) | None (unlimited)
  class_weight | default | 'balanced'

---

# 🎯 EVALUATION PRESENTATION — How to Start & What to Say

---

## Opening (30 seconds)

> *"Good morning. Our project is a **Multi-Channel Phishing Detection System** that uses **Machine Learning** to detect phishing attacks through three channels — **SMS text analysis, URL structural analysis, and Visual forensic analysis**. The core algorithm is **Random Forest**, and I'll explain exactly how it works from data input to threat score output."*

---

## Recommended Presentation Flow:

### 1️⃣ Start with the Dataset (2 min)
> *"We use two datasets:*
> - *SMS: UCI SMS Spam Collection — 5,572 real messages (4,825 ham + 747 spam) from Kaggle, the most cited SMS benchmark*
> - *URL: Custom dataset — 4,775 URLs (2,775 phishing from PhishTank + 2,000 legitimate from Alexa Top Sites, including Indian banking URLs)"*

### 2️⃣ Feature Extraction (3 min)
> *"For SMS, we extract 519 features:*
> - *500 TF-IDF word features — top 500 words auto-selected from vocabulary*
> - *19 statistical features — urgency count, URL presence, uppercase ratio, etc."*
>
> *"For URL, we extract 28 structural features:*
> - *url_length, num_dots, has_ip_address, hostname_entropy, has_suspicious_words, etc.*
> - *No TF-IDF for URL — we analyze structure, not content."*

### 3️⃣ Random Forest Algorithm — THE MAIN PART (5 min)
> *"We train 4 models for SMS (Naive Bayes, Logistic Regression, SVM, Random Forest) and 4 for URL (Logistic Regression, Gradient Boosting, XGBoost, Random Forest). Random Forest performed best in both. Here's how it works:"*
>
> - 100 trees (SMS) / 200 trees (URL)
> - Each tree gets a bootstrap sample of ALL training messages (not just one message)
> - At each node: randomly select √features (23 for SMS, 5 for URL), pick best split using Gini impurity
> - Tree grows until leaves are pure (Gini = 0)
> - Prediction = majority vote of all trees

### 4️⃣ Why Random Forest Over Others? (2 min)
> *"Naive Bayes assumes word independence — misses word combinations like 'click + verify'. Logistic Regression draws a straight line boundary — can't handle complex patterns. SVM is slow on 519 features. Random Forest captures non-linear patterns, handles mixed features naturally, provides feature importance rankings, and resists overfitting through bagging (100 trees average out errors)."*

### 5️⃣ Threat Score Calculation (2 min)
> *"The final threat score combines three weights:*
> - *SMS analysis: 40% weight*
> - *URL analysis: 45% weight (highest because URL structure is strongest indicator)*
> - *Visual analysis: 15% weight (supplementary)*
> - *Weights are re-normalized if any analysis is skipped*
> - *Risk levels: LOW (<30%), MEDIUM (30-60%), HIGH (60-85%), CRITICAL (85%+)"*

### 6️⃣ Close with Results
> *"SMS model achieves ~97% accuracy. URL model achieves ~95% F1-score with 0.98 ROC AUC. Random Forest was selected through cross-validation as the most stable and accurate model for both tasks."*

---

## ⚡ Quick Ready-Answers for Tough Questions:

| If they ask... | Say this... |
|---|---|
| "How many epochs?" | "Random Forest doesn't use epochs — it uses n_estimators (100/200 trees). Epochs are for neural networks, not traditional ML." |
| "Why 0.95 F1 but 0.98 AUC?" | "F1 measures at one threshold (0.5), AUC measures across all possible thresholds. AUC is almost always ≥ F1. Both confirm RF is the best model." |
| "What are 23 features per node?" | "At each node, √519 ≈ 23 random features are sampled. This is NOT the number of nodes — each tree has hundreds of nodes, and each node picks a NEW random 23." |
| "Is TF-IDF vocabulary fixed?" | "Yes, 500 words selected once during training, saved in feature_extractor.pkl, reused for every prediction." |
| "Why so many zeros in features?" | "Sparse matrix — each message has only 10-20 words out of 500 vocabulary. Zeros mean that word doesn't appear in that message." |
| "How does tree know to stop?" | "When the node is pure (Gini = 0, all samples are same class) or min_samples_split is reached." |
| "Does each tree train on one message?" | "No! Each tree trains on ALL messages (bootstrap sample of ~4,458). During prediction, a single message goes through all 100 trees." |
| "Where is the dataset from?" | "SMS: UCI/Kaggle (5,572 messages, most cited benchmark). URL: PhishTank + Alexa Top Sites (4,775 URLs, used by Firefox browser)." |
| "What is class_weight balanced?" | "Gives higher importance to the minority class so the model doesn't just predict the majority class every time." |
| "What tools for training?" | "Python scripts (.py files), NOT Jupyter/Colab. Scikit-learn for ML, NLTK for NLP, Pandas for data, Flask for API." |