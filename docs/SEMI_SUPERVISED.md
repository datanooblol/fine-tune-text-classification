# Semi-Supervised Learning

## Overview

**Semi-supervised learning** bridges supervised and unsupervised learning by using both labeled and unlabeled data for training. This is useful when labeling data is expensive or time-consuming.

**Key Idea**: Leverage large amounts of unlabeled data to improve model performance beyond what's possible with limited labeled data alone.

---

## Learning Paradigms Comparison

| Paradigm | Labeled Data | Unlabeled Data | Use Case |
|----------|--------------|----------------|----------|
| **Supervised** | ✅ Lots | ❌ None | Traditional classification |
| **Unsupervised** | ❌ None | ✅ Lots | Clustering, dimensionality reduction |
| **Semi-Supervised** | ✅ Some | ✅ Lots | Limited labels, abundant unlabeled data |

---

## Semi-Supervised Learning Variants

### 1. Self-Training (Bootstrapping)

**How it works:**
1. Train initial model on labeled data
2. Predict labels for unlabeled data
3. Add high-confidence predictions to training set
4. Retrain model
5. Repeat until convergence

**Pros:**
- Simple to implement
- Works with any classifier
- Iteratively improves

**Cons:**
- Can reinforce errors (confirmation bias)
- Sensitive to confidence threshold
- May not converge

**Example:**
```python
# Iteration 1: Train on 100 labeled
# Iteration 2: Add 50 high-confidence predictions → Train on 150
# Iteration 3: Add 30 more → Train on 180
# Continue...
```

**Use when:** You have a small labeled set and want to gradually expand it.

---

### 2. Co-Training

**How it works:**
1. Split features into 2+ independent views (e.g., text + metadata)
2. Train separate models on each view
3. Each model labels unlabeled data
4. Models teach each other by adding confident predictions
5. Retrain and repeat

**Pros:**
- Reduces error propagation (models check each other)
- Works well with multi-view data
- More robust than self-training

**Cons:**
- Requires independent feature views
- More complex implementation
- Computationally expensive

**Example:**
```python
# View 1: Email text content
# View 2: Email metadata (sender, time, subject length)
# Model A (text) and Model B (metadata) train each other
```

**Use when:** Your data has multiple independent feature representations.

---

### 3. Pseudo-Labeling

**How it works:**
1. Train model on labeled data
2. Generate pseudo-labels for ALL unlabeled data
3. Train new model on labeled + pseudo-labeled data together
4. Optionally weight pseudo-labels lower than true labels

**Pros:**
- Simple one-shot approach
- Can use all unlabeled data at once
- Works well with deep learning

**Cons:**
- Quality depends on initial model
- No iterative refinement (unless repeated)
- Can amplify initial errors

**Example:**
```python
# Train on 100 labeled
# Predict on 10,000 unlabeled → pseudo-labels
# Train on 100 labeled + 10,000 pseudo-labeled (weighted)
```

**Use when:** You have a decent initial model and lots of unlabeled data.

---

### 4. PU Learning (Positive-Unlabeled)

**How it works:**
1. Only positive examples are labeled
2. Unlabeled set contains hidden positives + negatives
3. Use specialized algorithms to handle this asymmetry:
   - **Spy technique**: Hide some positives in unlabeled to find reliable negatives
   - **Class prior estimation**: Estimate proportion of positives in unlabeled
   - **Cost-sensitive**: Weight unlabeled examples differently

**Pros:**
- Designed for one-class scenarios
- Handles hidden positives in unlabeled
- Theoretically grounded

**Cons:**
- More complex than binary classification
- Requires assumptions about data distribution
- Fewer off-the-shelf implementations

**Example:**
```python
# 20 known spam emails (positive)
# 80 unlabeled emails (some are spam, some are not)
# Goal: Identify all spam without labeled "not spam" examples
```

**Use when:** You only have positive labels and unlabeled data (no confirmed negatives).

---

## Comparison Table

| Method | Labeled Data | Unlabeled Data | Iterations | Complexity | Best For |
|--------|--------------|----------------|------------|------------|----------|
| **Self-Training** | Small | Large | Multiple | Low | General purpose |
| **Co-Training** | Small | Large | Multiple | Medium | Multi-view data |
| **Pseudo-Labeling** | Small | Large | Single/Few | Low | Deep learning |
| **PU Learning** | Positive only | Large | Varies | High | One-class problems |

---

## When to Use Each

### Use Self-Training when:
- You have limited labeled data
- You want iterative improvement
- You can set a good confidence threshold

### Use Co-Training when:
- Your data has multiple independent views
- You want robustness against errors
- You have computational resources

### Use Pseudo-Labeling when:
- You have a strong initial model
- You want a simple one-shot approach
- You're using deep learning (neural networks)

### Use PU Learning when:
- You only have positive labels
- Unlabeled data may contain hidden positives
- You're doing anomaly/outlier detection

---

## Practical Example: Text Classification

**Scenario:** Classify customer support tickets as "urgent" vs "not urgent"

**Available Data:**
- 50 labeled urgent tickets
- 20 labeled not-urgent tickets  
- 5,000 unlabeled tickets

**Approach Options:**

1. **Self-Training:**
   - Train on 70 labeled
   - Predict on 5,000 unlabeled
   - Add top 100 confident predictions
   - Retrain, repeat

2. **Co-Training:**
   - View 1: Ticket text
   - View 2: Metadata (time, customer tier, category)
   - Train 2 models, let them teach each other

3. **Pseudo-Labeling:**
   - Train on 70 labeled
   - Pseudo-label all 5,000
   - Train on 70 + 5,000 (weight pseudo-labels at 0.3)

4. **PU Learning (if only urgent labeled):**
   - 50 urgent tickets (positive)
   - 5,020 unlabeled (contains both urgent and not-urgent)
   - Use PU algorithm to identify all urgent tickets

---

## Implementation Tips

1. **Always validate:** Hold out labeled data for validation
2. **Monitor performance:** Track metrics across iterations
3. **Set thresholds carefully:** Confidence thresholds greatly impact results
4. **Start conservative:** Begin with high confidence thresholds, relax gradually
5. **Check for drift:** Ensure pseudo-labels don't degrade over iterations
6. **Use class weights:** Handle imbalanced data properly

---

## References

- Chapelle, O., Schölkopf, B., & Zien, A. (2006). Semi-supervised learning.
- Blum, A., & Mitchell, T. (1998). Combining labeled and unlabeled data with co-training.
- Lee, D. H. (2013). Pseudo-label: The simple and efficient semi-supervised learning method.
- Elkan, C., & Noto, K. (2008). Learning classifiers from only positive and unlabeled data.
