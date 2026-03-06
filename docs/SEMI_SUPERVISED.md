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

**Code Example:**
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Initial labeled data
labeled_texts = ["spam text 1", "ham text 1", ...]
labeled_labels = [1, 0, ...]
unlabeled_texts = ["unknown text 1", "unknown text 2", ...]  # Large unlabeled set

# Load model
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Self-training loop
for iteration in range(5):
    # 1. Train on current labeled data
    dataset = Dataset.from_dict({'text': labeled_texts, 'labels': labeled_labels})
    dataset = dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding=True), batched=True)
    
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir='./model', num_train_epochs=3),
        train_dataset=dataset,
        processing_class=tokenizer
    )
    trainer.train()
    
    # 2. Predict on unlabeled data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    inputs = tokenizer(unlabeled_texts, truncation=True, padding=True, return_tensors="pt").to(device)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    predictions = torch.argmax(probs, dim=-1)
    confidences = torch.max(probs, dim=-1).values
    
    # 3. Add high-confidence predictions to labeled set
    confidence_threshold = 0.9
    for i, (text, pred, conf) in enumerate(zip(unlabeled_texts, predictions, confidences)):
        if conf > confidence_threshold:
            labeled_texts.append(text)
            labeled_labels.append(pred.item())
            unlabeled_texts.pop(i)
    
    print(f"Iteration {iteration}: Labeled set size = {len(labeled_texts)}")
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

**Code Example:**
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import torch

# Data with two views
texts = ["email body 1", "email body 2", ...]  # View 1: Text
metadata = [[10, 1, 5], [50, 0, 2], ...]  # View 2: [length, is_reply, num_links]
labels = [1, 0, ...]  # Small labeled set
unlabeled_texts = ["unlabeled email 1", ...]
unlabeled_metadata = [[30, 1, 3], ...]

# Model A: Text-based
tokenizer_a = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model_a = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Model B: Metadata-based (simplified - would use a different architecture)
model_b = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Co-training loop
for iteration in range(5):
    # 1. Train Model A on text + current labels
    dataset_a = Dataset.from_dict({'text': texts, 'labels': labels})
    dataset_a = dataset_a.map(lambda x: tokenizer_a(x['text'], truncation=True, padding=True), batched=True)
    # ... train model_a ...
    
    # 2. Train Model B on metadata + current labels
    # ... train model_b on metadata ...
    
    # 3. Model A predicts on unlabeled data
    inputs_a = tokenizer_a(unlabeled_texts, truncation=True, padding=True, return_tensors="pt")
    preds_a = torch.argmax(model_a(**inputs_a).logits, dim=-1)
    
    # 4. Model B predicts on unlabeled data
    # preds_b = model_b.predict(unlabeled_metadata)
    
    # 5. Add examples where both models agree with high confidence
    for i, (text, meta, pred_a) in enumerate(zip(unlabeled_texts, unlabeled_metadata, preds_a)):
        # if pred_a == pred_b and confidence > threshold:
        texts.append(text)
        metadata.append(meta)
        labels.append(pred_a.item())
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

**Code Example:**
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Data
labeled_texts = ["spam 1", "ham 1", ...]  # 100 labeled
labeled_labels = [1, 0, ...]
unlabeled_texts = ["unknown 1", "unknown 2", ...]  # 10,000 unlabeled

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Step 1: Train on labeled data
dataset = Dataset.from_dict({'text': labeled_texts, 'labels': labeled_labels})
dataset = dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding=True), batched=True)

trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir='./model', num_train_epochs=3),
    train_dataset=dataset,
    processing_class=tokenizer
)
trainer.train()

# Step 2: Generate pseudo-labels for unlabeled data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
inputs = tokenizer(unlabeled_texts, truncation=True, padding=True, return_tensors="pt").to(device)
outputs = model(**inputs)
pseudo_labels = torch.argmax(outputs.logits, dim=-1).cpu().tolist()

# Step 3: Combine labeled + pseudo-labeled data
all_texts = labeled_texts + unlabeled_texts
all_labels = labeled_labels + pseudo_labels

# Create sample weights (true labels = 1.0, pseudo-labels = 0.3)
sample_weights = [1.0] * len(labeled_texts) + [0.3] * len(unlabeled_texts)

# Step 4: Retrain on combined data
combined_dataset = Dataset.from_dict({
    'text': all_texts, 
    'labels': all_labels,
    'weight': sample_weights
})
combined_dataset = combined_dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding=True), batched=True)

# Retrain with weighted loss (requires custom Trainer)
trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir='./model_pseudo', num_train_epochs=3),
    train_dataset=combined_dataset,
    processing_class=tokenizer
)
trainer.train()
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

**Code Example:**
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# PU Learning data
positive_texts = ["spam 1", "spam 2", ...]  # 20 known positives
unlabeled_texts = ["unknown 1", "unknown 2", ...]  # 80 unlabeled (mix of positive and negative)

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Step 1: Initial training - treat unlabeled as negative (with lower weight)
train_texts = positive_texts + unlabeled_texts
train_labels = [1] * len(positive_texts) + [0] * len(unlabeled_texts)

# Create sample weights (positives = 1.0, unlabeled = 0.1)
sample_weights = [1.0] * len(positive_texts) + [0.1] * len(unlabeled_texts)

dataset = Dataset.from_dict({
    'text': train_texts,
    'labels': train_labels,
    'weight': sample_weights
})
dataset = dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding=True), batched=True)

trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir='./pu_model', num_train_epochs=3),
    train_dataset=dataset,
    processing_class=tokenizer
)
trainer.train()

# Step 2: Predict on unlabeled data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
inputs = tokenizer(unlabeled_texts, truncation=True, padding=True, return_tensors="pt").to(device)
outputs = model(**inputs)
probs = torch.softmax(outputs.logits, dim=-1)
predictions = torch.argmax(probs, dim=-1)
confidences = probs[:, 1]  # Confidence for positive class

# Step 3: Identify reliable positives and negatives
reliable_positives = []
reliable_negatives = []

for text, pred, conf in zip(unlabeled_texts, predictions, confidences):
    if pred == 1 and conf > 0.9:  # High confidence positive
        reliable_positives.append(text)
    elif pred == 0 and conf < 0.1:  # High confidence negative
        reliable_negatives.append(text)

# Step 4: Retrain with reliable labels
final_texts = positive_texts + reliable_positives + reliable_negatives
final_labels = [1] * (len(positive_texts) + len(reliable_positives)) + [0] * len(reliable_negatives)

final_dataset = Dataset.from_dict({'text': final_texts, 'labels': final_labels})
final_dataset = final_dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding=True), batched=True)

trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir='./pu_model_final', num_train_epochs=3),
    train_dataset=final_dataset,
    processing_class=tokenizer
)
trainer.train()

print(f"Reliable positives found: {len(reliable_positives)}")
print(f"Reliable negatives found: {len(reliable_negatives)}")
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
