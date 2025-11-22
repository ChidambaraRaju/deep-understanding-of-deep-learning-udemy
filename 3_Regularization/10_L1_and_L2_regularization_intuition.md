# L1 and L2 Regularization — Quick Reference

Regularization helps prevent overfitting by penalizing large weights, encouraging the network to learn simpler and more generalizable patterns.

---

## 📌 Why Regularization?
Overfitting happens when a model becomes too flexible and memorizes noise.  
Large or sensitive weights often cause this.  
Regularization keeps weights controlled.

---

## 🔵 L2 Regularization (Ridge)

### **Formula**

$$\mathcal{L}_{L2} = \mathcal{L} + \lambda \sum_i w_i^2$$


### **Intuition**
- Squares grow fast → large weights are heavily penalized.
- Encourages **small, smooth weights** but **not exactly zero**.
- Reduces sensitivity to input noise.

### **Gradient Update**
$$w \leftarrow w - \eta\left(\frac{\partial \mathcal{L}}{\partial w} + 2\lambda w\right)$$

The extra term $2\lambda w$ acts like a proportional pull towards zero (weight decay)

### ✔ Effects
- Smooth model
- Better generalization
- No sparsity (weights stay non-zero)

---

## 🔴 L1 Regularization (Lasso)

### **Formula**

$$\mathcal{L}_{L1} = \mathcal{L} + \lambda \sum_i |w_i|$$


### **Intuition**
- Absolute value penalty grows linearly.
- Creates a **constant push** toward zero.
- Many weights become **exactly zero** → sparsity.

### **Gradient Update**

$$w \leftarrow w - \eta\left(\frac{\partial \mathcal{L}}{\partial w} + \lambda\,\mathrm{sign}(w)\right)$$

Because the subgradient of $|w|$ is $\pm1$, small weights cross zero and stay zero — hence sparsity.

### ✔ Effects
- Sparse model
- Feature selection
- Good for high-dimensional data

---

## 🧠 Geometric Intuition (2D)

- **L2 penalty region** → circle → smooth edges → weights shrink but don’t reach zero  
- **L1 penalty region** → diamond → sharp corners → solutions land exactly at zero

---

## 🔥 Summary Table

| Regularization | Penalty | Weight Behavior | Produces Sparsity? | Best Use |
|----------------|---------|-----------------|---------------------|----------|
| **L1** | \(|w|\) | Many weights become **0** | ✔ Yes | Feature selection |
| **L2** | \(w^2\) | Weights shrink smoothly | ✘ No | Stable generalization |

---

## ✔ When to Use What?
- Use **L1** when you want sparsity or interpretability.  
- Use **L2** when you want stable training and smoother models.  
- Use both (Elastic Net) for the best of both worlds.

---
