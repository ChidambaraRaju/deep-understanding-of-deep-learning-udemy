# Deep Learning: Data Normalization vs. Batch Normalization

In Deep Learning, **normalization** is the process of adjusting the scale of data to make training faster and more stable. This document breaks down the two most critical types: **Data Normalization** (applied to raw inputs) and **Batch Normalization** (applied to hidden layers).

---

## 1. Data Normalization (Input Scaling)

Data normalization acts on the **raw input features** before they are fed into the neural network.

### The Problem: Unscaled Data
Real-world datasets often contain features with vastly different scales.
* **Feature A (Age):** 0 to 100
* **Feature B (Salary):** 20,000 to 200,000

If we feed this directly into a network, the weights associated with "Salary" will likely explode or cause the gradients to oscillate wildly. The loss landscape becomes elongated (like a narrow ravine), making it difficult for the optimizer (like SGD or Adam) to find the minimum.



### The Solution
We scale all features to a similar range, typically $[0, 1]$ or a distribution with $\mu=0$ and $\sigma=1$. This turns the loss landscape into a symmetrical "bowl," allowing the optimizer to take larger, straighter steps toward the solution.

### Common Techniques

#### A. Min-Max Scaling (Normalization)
Rescales data to a fixed range, usually $[0, 1]$.
$$x' = \frac{x - \min(x)}{\max(x) - \min(x)}$$

#### B. Standardization (Z-Score Normalization)
Rescales data so it has a mean ($\mu$) of 0 and a standard deviation ($\sigma$) of 1. This is generally preferred for deep learning.
$$x' = \frac{x - \mu}{\sigma}$$

---

## 2. Batch Normalization (Hidden Layer Scaling)

While Data Normalization fixes the inputs, what happens **deep inside** the network?

### The Problem: Internal Covariate Shift
As the network trains, the weights in the early layers change. This changes the distribution of the outputs from those layers.
Consequently, the later layers (deeper in the network) are constantly trying to adapt to a "moving target." This phenomenon is called **Internal Covariate Shift**. It forces us to use lower learning rates and careful parameter initialization.




### The Solution: Batch Normalization (BN)
Invented in 2015 by Ioffe and Szegedy, Batch Normalization applies normalization **between the layers** of the network. It normalizes the output of a previous layer by subtracting the batch mean and dividing by the batch standard deviation.

However, it adds two learnable parameters per feature:
1.  **Scale ($\gamma$):** Allows the network to expand the range.
2.  **Shift ($\beta$):** Allows the network to shift the mean.

$$y = \gamma \cdot \hat{x} + \beta$$

*Why add learnable parameters?* If the network determines that the raw, un-normalized data was actually better for learning, $\gamma$ and $\beta$ allow the network to "undo" the normalization.

### Benefits of Batch Normalization
1.  **Faster Convergence:** Allows for much higher learning rates.
2.  **Regularization Effect:** Because batch statistics are slightly noisy estimates of the population, BN adds a slight noise that reduces overfitting (similar to Dropout).
3.  **Independence:** Makes the network less sensitive to weight initialization.

---

## 3. Comparison Summary

| Feature | Data Normalization | Batch Normalization |
| :--- | :--- | :--- |
| **Where is it applied?** | On the raw input dataset (Pre-processing). | Between hidden layers inside the model. |
| **When is it calculated?** | Before training starts. | Dynamically during training (per batch). |
| **Main Goal** | Ensure all input features contribute equally. | Stabilize the inputs to deeper layers. |
| **Parameters** | None (uses static dataset statistics). | Learnable parameters ($\gamma$, $\beta$). |
| **Inference Behavior** | Uses saved scalar/mean from training set. | Uses "Running Mean/Var" tracked during training. |

---
