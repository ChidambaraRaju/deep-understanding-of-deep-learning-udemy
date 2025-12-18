# Accuracy, Precision, Recall, and F1 Score

When evaluating a classification model, **accuracy alone is often not enough**.  
To properly understand model behavior—especially with **imbalanced datasets**—we use additional metrics derived from the **confusion matrix**.

---

## 1. Confusion Matrix (Foundation)

All classification metrics are built from four basic outcomes:

| Actual \ Predicted | Positive | Negative |
|-------------------|----------|----------|
| **Positive** | True Positive (TP) | False Negative (FN) |
| **Negative** | False Positive (FP) | True Negative (TN) |

- **TP**: Model correctly predicts positive  
- **TN**: Model correctly predicts negative  
- **FP**: Model predicts positive, but actually negative  
- **FN**: Model predicts negative, but actually positive  

---

## 2. Accuracy

### Definition
Accuracy measures **overall correctness** of the model.

### Formula
$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

### Intuition
- Counts **all correct predictions**
- Treats all errors equally
- Gives a single overall performance score

### When Accuracy Works Well
- Dataset is **balanced**
- Cost of false positives and false negatives is similar

### Limitation
- Can be **misleading for imbalanced datasets**
- Does not reveal **model bias**
- Example:
  - If 99% of data is negative, predicting "negative" always gives 99% accuracy

---

## 3. Precision

### Definition
Precision answers the question:

> **When the model predicts “positive”, how often is it correct?**

### Formula
$$
\text{Precision} = \frac{TP}{TP + FP}
$$

### Intuition
- Focuses on **positive predictions only**
- Penalizes **false positives**
- Measures **bias toward saying “YES”**

### High Precision Means
- Few false positives
- Model is cautious when predicting positive

### When Precision Is Important
- False positives are **costly**
- Examples:
  - Cancer diagnosis
  - Fraud detection
  - Spam filtering

### Key Insight
A model that says “YES” too often will have **low precision**.

---

## 4. Recall (Sensitivity)

### Definition
Recall answers the question:

> **Of all actual positives, how many did the model correctly find?**

### Formula
$$
\text{Recall} = \frac{TP}{TP + FN}
$$

### Intuition
- Focuses on **actual positives**
- Penalizes **false negatives**
- Measures **bias toward saying “NO”**

### High Recall Means
- Few false negatives
- Model rarely misses positive cases

### When Recall Is Important
- Missing a positive is **dangerous**
- Examples:
  - Disease screening (COVID, cancer)
  - Safety systems
  - Intrusion detection

### Alternative Name
- Recall is also called **Sensitivity**
- Same formula, different terminology

---

## 5. Precision vs Recall (Key Difference)

| Metric | Focus | Penalizes | Bias Detected |
|------|------|----------|---------------|
| Precision | Predicted Positives | False Positives | Saying "YES" too often |
| Recall | Actual Positives | False Negatives | Saying "NO" too often |

- Precision and recall **trade off** against each other
- Improving one often reduces the other

---

## 6. F1 Score

### Definition
F1 Score provides a **single metric** that balances **precision and recall**.

### Formula
$$
\text{F1 Score} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}
$$

(Equivalent to harmonic mean of precision and recall)

### Intuition
- High only when **both precision and recall are high**
- Penalizes extreme imbalance between them
- Ignores true negatives explicitly, but they are implicitly accounted for

### When F1 Score Is Useful
- Dataset is **imbalanced**
- Both false positives and false negatives matter
- Need a **single balanced metric**

### Interpretation
- F1 ≈ 1 → Very good model
- F1 ≈ 0 → Very poor model

---

## 7. Summary Comparison

| Metric | Best Use Case | Main Weakness |
|------|--------------|---------------|
| Accuracy | Balanced data | Misleading with imbalance |
| Precision | FP is costly | Ignores FN |
| Recall | FN is costly | Ignores FP |
| F1 Score | Balanced error importance | Does not explain bias direction |

---

## 8. Final Takeaways

- **Accuracy** gives an overall view, but hides bias
- **Precision** detects false alarms
- **Recall** detects missed positives
- **F1 Score** balances both types of errors
- Always inspect **precision and recall together**, not in isolation

---

> **Rule of thumb:**  
> If accuracy looks good, but the model behaves poorly in practice — check precision, recall, and F1.
