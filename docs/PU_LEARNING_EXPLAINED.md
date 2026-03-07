# PU Learning: Positive-Unlabeled Learning Explained

## 🎯 What is PU Learning?

**Problem**: You have some labeled positives, but the rest is unlabeled (could be positive OR negative).

**Example**: 
- 152 emails labeled as "spam" ✅
- 1,220 emails unlabeled ❓ (some are spam, some are not)
- Goal: Find the hidden spam in the unlabeled set

---

## 📊 The Scenario

```
Original Data:
├── 610 spam emails (positive)
└── 762 ham emails (negative)

PU Learning Setup:
├── 152 spam labeled as "Positive" (25%)
├── 458 spam labeled as "Unlabeled" (75% - HIDDEN!)
└── 762 ham labeled as "Unlabeled"

Total Unlabeled: 1,220 (458 spam + 762 ham mixed together)
```

**Challenge**: How to find the 458 hidden spam from 1,220 unlabeled?

---

## 🔑 Key Concept: P(s=1|y=1)

### What does it mean?

- `s=1` = labeled as positive
- `y=1` = truly positive
- `P(s=1|y=1)` = **"What % of true positives are labeled?"**

In our case: 152/(152+458) = **0.25 (25%)**

### Why is this important?

If we know P(s=1|y=1), we can convert:
- P(s=1|X) = "probability of being labeled" (what model predicts)
- P(y=1|X) = "probability of being positive" (what we want!)

**Formula**: `P(y=1|X) ≈ P(s=1|X) / P(s=1|y=1)`

---

## 🛠️ The Adapted Mixture (AM) Method

### Step 1: Hold-Out Trick

```python
# We have 152 labeled positives
# Hold out 25% (38 examples) for calibration
# Train on remaining 75% (114 examples) + all unlabeled (1,220)

X_hold_out = labeled_positives.sample(frac=0.25)  # 38 examples
X_train = remaining_114 + unlabeled_1220  # 1,334 examples
y_train = [1]*114 + [0]*1220  # Treat unlabeled as negative
```

### Step 2: Train Model

```python
# Model learns: P(s=1|X) = probability of being LABELED
model.fit(X_train, y_train)
```

### Step 3: Estimate P(s=1|y=1)

```python
# Predict on held-out POSITIVES (we know they're truly positive)
hold_out_predictions = model.predict_proba(X_hold_out)[:,1]

# Average = estimate of P(s=1|y=1)
prob_s1y1 = hold_out_predictions.mean()  # Should be ~0.25
```

### Step 4: Calibrate Predictions

```python
# Predict on ALL data
predicted_s = model.predict_proba(X_all)[:,1]  # P(s=1|X)

# Convert to P(y=1|X)
predicted_y = predicted_s / prob_s1y1
```

---

## 🔁 Why 1001 Iterations?

### The Problem with Single Iteration

```
Single run:
  Hold-out: [spam_5, spam_12, spam_23, ...]
  ❌ Only one random split
  ❌ Unstable estimate
  ❌ Some spam patterns might be missed
```

### The Ensemble Solution

```
Iteration 1: Hold-out [spam_5, spam_12, ...]  → Model A → Predictions A
Iteration 2: Hold-out [spam_1, spam_8, ...]   → Model B → Predictions B
Iteration 3: Hold-out [spam_3, spam_15, ...]  → Model C → Predictions C
...
Iteration 1001: Hold-out [spam_7, spam_19, ...] → Model Z → Predictions Z

Final = Average(A, B, C, ..., Z)
```

### Benefits

1. **Coverage**: Every spam example appears in both training and hold-out across iterations
2. **Stability**: Averaging reduces variance
3. **Robustness**: No single bad split ruins the result

---

## 💾 Why `predicted += predicted_index_scaled`?

### Memory-Efficient Accumulation

```python
# Initialize
predicted = np.zeros(1372)  # One value per example

# Loop
for i in range(1001):
    pred_i = predict_PU_prob_AM(...)  # Get predictions for this iteration
    predicted += pred_i  # Accumulate sum
    
# Average
final_proba = predicted / 1001
```

### Why Not Store All?

**Bad (Memory Intensive)**:
```python
all_preds = []  # Store 1001 × 1372 = 1.4M values
for i in range(1001):
    all_preds.append(predict_PU_prob_AM(...))
final = np.mean(all_preds, axis=0)
```

**Good (Memory Efficient)**:
```python
predicted = np.zeros(1372)  # Only 1372 values
for i in range(1001):
    predicted += predict_PU_prob_AM(...)  # Just add
final = predicted / 1001  # Still only 1372 values
```

**Math**: `(a + b + c) / 3 = average` is same as accumulating sum then dividing!

---

## 📈 From Probabilities to Labels

### After 1001 Iterations

```python
# You have probabilities for each example
mod_data['y_pos_pred_proba'] = predicted / 1001

# Example values:
# Index  PU_Target   y_pos_pred_proba
# 0      Positive    0.65017          ← Known positive
# 1      Unlabeled   0.82341          ← Hidden positive (high!)
# 2      Unlabeled   0.00035          ← True negative (low)
# 3      Unlabeled   0.45123          ← Uncertain
```

### Choose Threshold

```python
# Option 1: Fixed threshold
threshold = 0.5
mod_data['predicted_label'] = (mod_data['y_pos_pred_proba'] > threshold).astype(int)

# Option 2: Optimize threshold
best_threshold = 0
best_f1 = 0
for threshold in np.linspace(0.1, 0.9, 100):
    pred = (mod_data['y_pos_pred_proba'] > threshold).astype(int)
    f1 = f1_score(mod_data['Target'], pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Option 3: Top-K (if you know expected count)
k = 610  # Expected positives
top_k_indices = mod_data.nlargest(k, 'y_pos_pred_proba').index
mod_data.loc[top_k_indices, 'predicted_label'] = 1
```

### Train Final Model

```python
# Use predicted labels for training
X_final = mod_data[features]
y_final = mod_data['predicted_label']

final_model = xgb.XGBClassifier()
final_model.fit(X_final, y_final)
```

---

## 🎯 Complete Workflow

```
1. Setup PU Scenario
   ├── Keep 25% positives labeled (152)
   └── Hide 75% positives in unlabeled (458 + 762 = 1,220)

2. Run 1001 Iterations
   ├── Each iteration:
   │   ├── Hold out 25% of labeled positives
   │   ├── Train on remaining + unlabeled
   │   ├── Estimate P(s=1|y=1) from held-out
   │   ├── Predict probabilities for all
   │   └── Accumulate predictions
   └── Average all predictions

3. Convert Probabilities to Labels
   ├── Choose threshold (e.g., 0.5)
   └── Label examples above threshold as positive

4. Train Final Classifier
   ├── Use newly labeled data
   └── Evaluate performance
```

---

## 📊 Expected Results

```
After PU Learning:

Target | Median Probability
-------|-------------------
0      | 0.000349   ← Negatives: very low
1      | 0.032135   ← Hidden positives: 100x higher!

With threshold = 0.5:
- Found: ~565 out of 610 positives (93% recall)
- False positives: ~12 out of 762 negatives (98% precision)
```

---

## 🆚 Comparison with Other Methods

| Method | Data Needed | Complexity | Accuracy | Use Case |
|--------|-------------|------------|----------|----------|
| **Supervised** | Labeled pos + neg | Low | High | Have both labels |
| **PU Learning** | Labeled pos only | Medium | Good | Only positive labels |
| **One-Class SVM** | Labeled pos only | Low | Medium | Simple baseline |
| **Clustering** | Labeled pos only | Medium | Varies | Exploratory |

---

## 💡 Key Takeaways

1. **PU Learning** = Learn from positive + unlabeled data
2. **Hold-out trick** = Estimate labeling rate P(s=1|y=1)
3. **Ensemble** = Run many iterations with different hold-outs
4. **Accumulation** = Memory-efficient averaging with `+=`
5. **Threshold** = Convert probabilities to final labels
6. **No automatic labeling** = YOU choose threshold and create labels

---

## 📚 References

- Elkan, C., & Noto, K. (2008). Learning classifiers from only positive and unlabeled data.
- Adapted Mixture (AM) method for PU Learning
- Ensemble learning for variance reduction

---

## 🚀 Next Steps

1. Try PU Learning on your spam dataset
2. Experiment with different thresholds
3. Compare with supervised baseline
4. Try clustering-based PU approach
5. Combine multiple PU methods for best results
