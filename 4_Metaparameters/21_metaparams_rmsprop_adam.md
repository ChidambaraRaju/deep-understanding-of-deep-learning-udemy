# 📘 RMSProp & Adam Optimizers in Deep Learning

## 1️⃣ Why Do We Need Advanced Optimizers?

All modern optimizers are **extensions of basic Gradient Descent**.

| Optimizer | Purpose |
|----------|----------|
| Gradient Descent | Basic learning |
| Momentum | Faster & smoother updates |
| RMSProp | Adaptive learning rate |
| **Adam** | **Momentum + RMSProp (Best of both)** |

✅ These optimizers:
- Improve training stability
- Speed up convergence
- Reduce manual tuning of learning rates

---

## 2️⃣ RMSProp Optimizer

### ✅ What Does RMSProp Stand For?
- **RMS** → Root Mean Square
- **Prop** → Propagation (from backpropagation)

RMSProp adapts the **learning rate based on recent gradient magnitudes**.

---

### ✅ Root Mean Square (RMS) Formula

$$
RMS(x) = \sqrt{\frac{1}{n} \sum x^2}
$$

- RMS measures the **energy or magnitude** of a signal.
- Closely related to **standard deviation**

---

### ✅ Core Idea of RMSProp

Instead of using a fixed learning rate, RMSProp:

- Maintains a **running average of squared gradients**
- Adjusts learning rate for each weight individually

| Gradient Behavior | RMSProp Response |
|------------------|------------------|
| Large gradients | Smaller step size |
| Small gradients | Larger step size |

✅ This prevents:
- Exploding gradients
- Vanishing gradients

---

### ✅ RMSProp Equations

Let:
- $g_t$ = Gradient at time step $t$
- $s_t$ = Running average of squared gradients
- $\eta$ = Learning rate
- $\epsilon$ = Small constant (≈ 1e-8)

#### 1️⃣ Running Average Update
$$
s_t = \beta s_{t-1} + (1 - \beta) g_t^2
$$

#### 2️⃣ Weight Update Rule
$$
w_t = w_{t-1} - \frac{\eta}{\sqrt{s_t} + \epsilon} g_t
$$

✅ Each weight receives its **own adaptive learning rate**
✅ Very robust to poor learning-rate initialization

---

## 3️⃣ Adam Optimizer (Adaptive Moment Estimation)

Adam combines:

✅ **Momentum** → Smooth gradient direction  
✅ **RMSProp** → Adaptive learning rate  

> ✅ **Adam = Momentum + RMSProp**

---

### ✅ What Adam Tracks

| Term | Meaning |
|------|---------|
| $v_t$ | First moment (mean of gradients) |
| $s_t$ | Second moment (variance of gradients) |

---

### ✅ Adam Equations

#### 1️⃣ Momentum Update
$$
v_t = \beta_1 v_{t-1} + (1 - \beta_1) g_t
$$

#### 2️⃣ RMSProp Update
$$
s_t = \beta_2 s_{t-1} + (1 - \beta_2) g_t^2
$$

---

### ✅ Bias Correction in Adam

Since both $v_t$ and $s_t$ start at zero, they are biased initially. Adam corrects this using:

$$
\hat{v}_t = \frac{v_t}{1 - \beta_1^t}
$$

$$
\hat{s}_t = \frac{s_t}{1 - \beta_2^t}
$$

✅ Effect:
- Large learning steps in early training
- Smaller steps later for fine tuning

---

### ✅ Final Adam Weight Update

$$
w_t = w_{t-1} - \frac{\eta}{\sqrt{\hat{s}_t} + \epsilon} \hat{v}_t
$$

---

### ✅ Default Adam Hyperparameters

| Parameter | Value |
|-------------|--------|
| Learning Rate $\eta$ | 0.001 |
| $\beta_1$ (Momentum) | 0.9 |
| $\beta_2$ (RMSProp) | 0.999 |
| $\epsilon$ | 1e-8 |

✅ These defaults work well for most problems.

---

## 4️⃣ RMSProp vs Adam

| Feature | RMSProp | Adam |
|---------|----------|-------|
| Momentum | ❌ | ✅ |
| Adaptive Learning Rate | ✅ | ✅ |
| Bias Correction | ❌ | ✅ |
| Tracks Gradient Mean | ❌ | ✅ |
| Tracks Gradient Variance | ✅ | ✅ |
| Widely Used Today | ❌ | ✅ |

---

## 5️⃣ Why Adam Is the Most Popular Optimizer

✅ Faster convergence  
✅ Stable training  
✅ Minimal learning rate tuning  
✅ Works well for:
- Deep Neural Networks
- CNNs
- Transformers
- LLMs

⚠️ However, for **small/simple datasets**, classic SGD can sometimes outperform Adam.

---

## 6️⃣ Key Points

- ✅ RMSProp adapts learning rate using gradient energy
- ✅ Adam = Momentum + RMSProp
- ✅ Bias correction enables faster early learning
- ✅ Adam is the default optimizer in most deep learning projects

---

## ✅ One-Line Summary

> **RMSProp adapts learning rates using gradient magnitudes, while Adam combines both momentum and adaptive learning rates, making it the most powerful and widely used optimizer today.**

---
