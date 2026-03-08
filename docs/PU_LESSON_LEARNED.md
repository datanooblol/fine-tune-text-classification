# PU Learning: Practical Lessons Learned

## Overview

This document summarizes practical insights from implementing Modified Logistic Regression for Positive-Unlabeled (PU) learning on the Enron spam dataset (31,716 emails).

**Key Reference:** `package/models/modified_logistic_regression.py` and `06_semi_supervised_idea.ipynb`

---

## 1. Feature Engineering: TF-IDF vs Embeddings

### TF-IDF (Winner for Single Language)

**Performance on Enron dataset:**
- Accuracy: 79%
- Spam recall: 59%
- Ham recall: 99%
- F1-score: 0.74
- Estimated c: 0.4440 (true: 0.2500)

**Why it works:**
- Captures lexical patterns specific to spam ("FREE", "CLICK HERE", "LIMITED TIME")
- Maintains clear separation between spam/ham vocabulary
- Low false positive rate (1% ham misclassified as spam)

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2)  # Unigrams + bigrams
)
X = vectorizer.fit_transform(texts)
```

### BGE Embeddings (Raw) - Failed

**Performance:**
- Accuracy: 53%
- Spam recall: 8% ❌
- Ham recall: 100%
- F1-score: 0.14
- Estimated c: 0.6811 (172% overestimation)

**Why it failed:**
- Semantic similarity collapsed spam/ham into similar embedding space
- Model defaulted to predicting ham (safer bet)
- 384 dimensions too dense for PU learning's c estimation

### BGE + PCA (Improved but Not Ideal)

**Performance with 100 components:**
- Accuracy: 75%
- Spam recall: 98% ✓
- Ham recall: 51% ❌
- F1-score: 0.80
- Estimated c: 0.1904

**Trade-off:**
- Catches 98% of spam but flags 49% of ham as spam
- High false positive rate unacceptable for production

**Conclusion:** Use TF-IDF for single-language tasks. Only use BGE+PCA if multilingual support is required.

---

## 2. Multilingual Text: Mixed Thai-English

For text containing both Thai and technical English terms:

```python
import re
from pythainlp.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

def mixed_thai_english_tokenizer(text):
    tokens = []
    for segment in text.split():
        if re.search(r'[\u0E00-\u0E7F]', segment):
            # Thai characters - use pythainlp
            tokens.extend(word_tokenize(segment, engine='newmm'))
        else:
            # English/technical terms - keep as-is
            tokens.append(segment.lower())
    return tokens

vectorizer = TfidfVectorizer(
    tokenizer=mixed_thai_english_tokenizer,
    max_features=1000,
    ngram_range=(1, 2)
)
```

**Example:**
```
Input:  "โปรโมชั่น iPhone 15 Pro ลดราคา 50% API integration ฟรี"
Output: ['โปรโมชั่น', 'iphone', '15', 'pro', 'ลดราคา', '50%', 'api', 'integration', 'ฟรี']
```

---

## 3. Training Dynamics

### Epoch Selection

**Experiment results:**

| Epochs | Loss  | Estimated c | Spam Recall | Ham Recall | Accuracy |
|--------|-------|-------------|-------------|------------|----------|
| 500    | 0.3825| 0.4440      | 59%         | 99%        | 79%      |
| 1000   | 0.3459| 0.4440      | 59%         | 99%        | 79%      |
| 10000  | 0.3574| 0.2077      | 99%         | 63%        | 81%      |

**Key insights:**
- Loss decreases steadily but performance plateaus after 1000 epochs
- 10K epochs improves spam recall (59% → 99%) but hurts ham recall (99% → 63%)
- Diminishing returns after epoch 1000
- **Recommendation:** Use 500-1000 epochs for balanced performance

### Learning Rate

```python
model = ModifiedLogisticRegressionPU(
    lr=0.01,      # Default works well
    epochs=1000,
    verbose=True
)
```

---

## 4. Understanding Model Predictions

### Probability Distribution Analysis

**After 10K epochs (TF-IDF):**

```
              count      mean       std       min       25%       50%       75%       max
true_label                                                              
Ham (0)      15553.0  0.406212  0.316947  0.000741  0.120275  0.321314  0.688344  0.999707
Spam (1)     16163.0  0.929075  0.110782  0.065571  0.920287  0.970818  0.991164  0.999980
```

**Key observations:**

1. **Spam detection (excellent):**
   - Mean probability: 0.929 (very confident)
   - 75th percentile: 0.991 (most spam >99% confidence)
   - Tight distribution (std: 0.11)

2. **Ham detection (uncertain):**
   - Mean probability: 0.406 (uncertain)
   - Wide distribution (std: 0.32)
   - 25th percentile: 0.12, 75th percentile: 0.69

**Why this happens:**
- PU learning trains on "labeled spam" vs "unlabeled mix"
- Model learns spam patterns well (clear positive signal)
- Ham patterns are diffuse (hidden in unlabeled set)
- **This is expected behavior for PU learning**

---

## 5. Pseudo-Labeling Strategy

### Conservative Thresholds

```python
import pandas as pd

# Get predictions
proba = model.predict_proba(X)[:, 1]

# Create results DataFrame
results = pd.DataFrame({
    'text': texts,
    'original_label': y_pu,  # 0=unlabeled, 1=labeled positive
    'predicted_proba': proba
})

# Conservative pseudo-labeling
pseudo_spam = results[
    (results['original_label'] == 0) &  # Was unlabeled
    (results['predicted_proba'] >= 0.90)  # High confidence
]

pseudo_ham = results[
    (results['original_label'] == 0) &  # Was unlabeled
    (results['predicted_proba'] <= 0.10)  # High confidence (hard negatives)
]

# Uncertain zone - skip these
uncertain = results[
    (results['predicted_proba'] > 0.10) & 
    (results['predicted_proba'] < 0.90)
]
```

### Threshold Guidelines

| Threshold | Use Case | Expected Precision |
|-----------|----------|-------------------|
| ≥ 0.95 | Production labeling | Very high |
| ≥ 0.90 | Initial pseudo-labeling | High |
| ≥ 0.80 | Exploratory analysis | Medium |
| ≤ 0.10 | Hard negatives | High |
| ≤ 0.20 | Confident ham | Medium |

**Start conservative (0.90/0.10) and gradually relax if needed.**

---

## 6. Finding Hard Negatives

Hard negatives are unlabeled examples with low probabilities (confident ham predictions):

```python
# Find hard negatives
hard_negatives = results[
    (results['original_label'] == 0) &  # Unlabeled
    (results['predicted_proba'] <= 0.10)  # Low probability
].sort_values('predicted_proba')

print(f"Found {len(hard_negatives)} hard negatives")
print(hard_negatives[['text', 'predicted_proba']].head(10))
```

**Why hard negatives are valuable:**
- Help balance dataset
- Improve model robustness
- Reduce false positives in next training iteration

---

## 7. Positive-Unlabeled Framing

### Label Convention

**Use this framing:**
```python
y = 1  # Labeled positive (spam)
y = 0  # Unlabeled (contains both spam and ham)
```

**NOT this:**
```python
y = 1  # Positive
y = -1 # Negative  ❌ Don't use when you don't trust negative labels
```

**Benefits:**
- Cleaner conceptual model
- Matches PU learning assumptions
- Easier to explain to stakeholders
- Reusable code across different PU scenarios

---

## 8. Inspecting Predictions

### View Unlabeled Predicted as Positive

```python
# Filter: unlabeled that model predicts as spam
unlabeled_spam = results[
    (results['original_label'] == 0) &  # Was unlabeled
    (results['predicted_proba'] >= 0.90)  # Predicted as spam
].sort_values('predicted_proba', ascending=False)

# Export for manual review
unlabeled_spam.to_csv('pseudo_spam_labels.csv', index=False)

# View top predictions
print(unlabeled_spam[['text', 'predicted_proba']].head(20))
```

### Calibration Check

```python
# Compare estimated vs true labeling rate
print(f"Estimated c (ĉ): {model.c_hat_:.4f}")
print(f"True c: {(y_pu == 1).mean():.4f}")

# Check if close (within 20%)
error = abs(model.c_hat_ - (y_pu == 1).mean()) / (y_pu == 1).mean()
if error > 0.2:
    print(f"⚠️  Warning: c estimation error {error:.1%}")
    print("Consider: more training data, different features, or regularization")
```

---

## 9. Self-Training Loop (Optional)

Iteratively improve model by adding high-confidence pseudo-labels:

```python
import numpy as np

# Initial training
model = ModifiedLogisticRegressionPU(lr=0.01, epochs=1000)
model.fit(X_labeled, y_pu)

# Self-training iterations
for iteration in range(3):  # 2-3 iterations max
    # Get predictions on unlabeled
    proba = model.predict_proba(X_unlabeled)[:, 1]
    
    # High-confidence predictions only
    high_conf_mask = (proba >= 0.90) | (proba <= 0.10)
    pseudo_labels = (proba >= 0.90).astype(int)
    
    # Add to training set
    X_combined = np.vstack([X_labeled, X_unlabeled[high_conf_mask]])
    y_combined = np.hstack([y_pu, pseudo_labels[high_conf_mask]])
    
    # Retrain
    model.fit(X_combined, y_combined)
    
    print(f"Iteration {iteration+1}: Added {high_conf_mask.sum()} pseudo-labels")
```

**Warning:** Diminishing returns after 2-3 iterations. Monitor validation performance.

---

## 10. Key Takeaways

### ✅ Do This

1. **Use TF-IDF for single-language tasks** - Better than embeddings for PU learning
2. **Start with conservative thresholds** - 0.90 for spam, 0.10 for ham
3. **Always inspect pseudo-labels** - Manual review before committing
4. **Monitor c estimation** - Should be close to true labeling rate
5. **Use Positive-Unlabeled framing** - Cleaner than positive/negative
6. **Stop training at 500-1000 epochs** - Diminishing returns after
7. **Leverage hard negatives** - Low probability unlabeled examples

### ❌ Avoid This

1. **Don't use raw embeddings** - Too dense for c estimation
2. **Don't trust all predictions** - Only use high-confidence (>0.90 or <0.10)
3. **Don't over-train** - 10K epochs doesn't help much vs 1K
4. **Don't ignore ham recall** - Balance spam/ham performance
5. **Don't skip calibration checks** - Verify estimated c vs true c
6. **Don't do too many self-training iterations** - 2-3 max

### 🎯 Expected Performance

**With TF-IDF + 1000 epochs:**
- Spam recall: 60-80%
- Ham recall: 95-99%
- Overall accuracy: 75-85%
- F1-score: 0.70-0.80

**Model is best for:**
- Finding spam in unlabeled data (high recall)
- Identifying hard negatives (confident ham)

**Model struggles with:**
- Confident ham classification (wide probability distribution)
- Perfect c estimation (expect 10-30% error)

---

## 11. Complete Example

```python
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
from package.models.modified_logistic_regression import ModifiedLogisticRegressionPU

# 1. Load data
dataset = load_dataset("SetFit/enron_spam")
train_data = dataset['train']

# 2. Create PU scenario (hide 75% of spam)
spam_indices = [i for i, label in enumerate(train_data['label']) if label == 1]
np.random.seed(42)
labeled_spam = np.random.choice(spam_indices, size=len(spam_indices)//4, replace=False)

y_pu = np.zeros(len(train_data))
y_pu[labeled_spam] = 1

# 3. Extract features
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X = vectorizer.fit_transform(train_data['text'])

# 4. Train model
model = ModifiedLogisticRegressionPU(lr=0.01, epochs=1000, verbose=True)
model.fit(X, y_pu)

# 5. Get predictions
proba = model.predict_proba(X)[:, 1]

# 6. Create results DataFrame
results = pd.DataFrame({
    'text': train_data['text'],
    'true_label': train_data['label'],
    'original_label': y_pu,
    'predicted_proba': proba
})

# 7. Pseudo-labeling
pseudo_spam = results[
    (results['original_label'] == 0) & 
    (results['predicted_proba'] >= 0.90)
]

pseudo_ham = results[
    (results['original_label'] == 0) & 
    (results['predicted_proba'] <= 0.10)
]

print(f"Estimated c: {model.c_hat_:.4f}")
print(f"True c: {y_pu.mean():.4f}")
print(f"\nFound {len(pseudo_spam)} high-confidence spam")
print(f"Found {len(pseudo_ham)} high-confidence ham")

# 8. Evaluate
y_pred = (proba >= 0.5).astype(int)
print("\nClassification Report:")
print(classification_report(results['true_label'], y_pred, target_names=['Ham', 'Spam']))
```

---

## References

- **Paper:** Jaskie, Elkan, Spanias - "A Modified Logistic Regression for Positive and Unlabeled Learning"
- **Implementation:** `package/models/modified_logistic_regression.py`
- **Experiments:** `06_semi_supervised_idea.ipynb`
- **Tokenization:** `07_tokenization.ipynb` (for mixed Thai-English)
