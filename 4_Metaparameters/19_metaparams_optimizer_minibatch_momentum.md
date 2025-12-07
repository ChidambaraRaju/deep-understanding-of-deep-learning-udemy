# 📘 Optimizers: Mini-batch & Momentum

## 1️⃣ What Are Optimizers?

Optimizers in deep learning are not determining *new* ways to learn; they are **modifications of the standard Gradient Descent algorithm**.

- **Goal:** To smooth the descent towards the minimum of the error landscape.
- **Function:** They adjust the weights during backpropagation to make learning **faster** and **more efficient**.
- **Relationship:** All modern optimizers (Momentum, RMSprop, Adam) are extensions of Stochastic Gradient Descent (SGD).

---

## 2️⃣ The Problem with Stochastic Gradient Descent (SGD)

### ✅ How SGD Works
SGD updates the model weights after **every single data point**.
$$
W_{new} = W_{old} - \eta \cdot \nabla Loss
$$
*(Where $\eta$ is the learning rate)*

### ✅ The Issue: Volatility
While effective, SGD is highly sensitive to the specific data point being processed.

| Scenario | SGD Behavior |
|----------|--------------|
| **Homogeneous Data** | Works great! (e.g., sorted medical records) |
| **Heterogeneous Data** | Volatile! (e.g., diverse images, outliers) |

**The "Outlier" Problem:**
If the model encounters a non-representative data point (an outlier), SGD calculates a massive gradient based solely on that point. This causes the weights to bounce wildly, potentially moving the model *away* from the optimal solution.

---

## 3️⃣ Solution 1: Mini-batch Gradient Descent

Instead of updating weights after *every* sample (SGD) or *all* samples (Batch GD), we update after a small **batch** of samples.

- **Batch Size ($N$):** Typically a power of 2 (e.g., 16, 32, 64, 128).
- **Mechanism:** The loss is **averaged** across the $N$ samples in the batch.

### ✅ Why it Helps
1.  **Smoothing:** Outliers are averaged with "normal" data points, reducing their individual impact.
2.  **Stability:** Prevents drastic weight changes based on a single bad example.

> **Note:** For very simple/homogeneous datasets, pure SGD might actually be faster. Mini-batch shines when data is complex and noisy.

---

## 4️⃣ Solution 2: The Momentum Optimizer

### ✅ Core Concept
Momentum speeds up training by taking a **weighted average** of previous gradients. It biases the weight trajectory based on its **previous direction**.

### ✅ The Analogy
Imagine running as fast as you can. If a strong crosswind (a noisy gradient from a bad data point) hits you:
- **Without Momentum:** You get blown completely off course.
- **With Momentum:** Your previous forward speed keeps you moving mostly forward; the wind only slightly adjusts your path.

### ✅ The Math: Weighted Average

Momentum introduces a velocity term ($v$) that accumulates past gradients.

$$
v_t = \beta v_{t-1} + (1 - \beta) g_t
$$

$$
W_{new} = W_{old} - \eta v_t
$$

* $v_t$: Current velocity (accumulated gradient).
* $v_{t-1}$: Previous velocity.
* $g_t$: Current gradient.
* $\beta$ (Beta): The **momentum parameter** (friction).
    * Controls how much history to keep.
    * **Typical Value:** $0.9$ to $0.99$.

### ✅ How Beta ($\beta$) Works

| Beta Value | Effect |
|------------|--------|
| $\beta = 0$ | **Pure SGD** (No history is kept). |
| $\beta = 0.9$ | Current gradient is only 10% of the update; 90% is previous velocity. |

---

## 5️⃣ Summary Comparison

| Optimizer | Update Frequency | Pros | Cons |
|-----------|------------------|------|------|
| **SGD** | Every sample | Frequent updates, good for simple data | Noisy, volatile with outliers |
| **Mini-batch** | Every $N$ samples | Stable, parallelizable | Requires tuning batch size |
| **Momentum** | Every batch | Smooths trajectory, dampens oscillations | Adds a hyperparameter ($\beta$) |

---

## 6️⃣ Key Points

- ✅ **Optimizers** are just tweaks to gradient descent to smooth the error landscape traversal.
- ✅ **Mini-batch** stabilizes learning by averaging losses over a group of samples (powers of 2).
- ✅ **Momentum** prevents zig-zagging by using a weighted average of past gradients.
- ✅ **Beta ($\beta$)** in momentum is usually set to **0.9**, meaning we trust the history much more than the current single gradient.

---

## ✅ One-Line Summary

> **Mini-batch stabilizes updates by averaging outliers, while Momentum accelerates learning by accumulating previous gradient directions to smooth out the path to the minimum.**

---
