# 01: The Foundations of Artificial Neural Networks (ANN)

### 🎯 Objective
This document provides a complete, ground-up explanation of the core components of an Artificial Neural Network (ANN). We will build from the simplest unit, the **Perceptron**, and assemble it into a full network that can learn through **Backpropagation**.

### 🧠 What is an ANN?
An Artificial Neural Network is a computing system inspired by the biological neural networks that constitute animal brains. It's not a direct copy, but rather a mathematical framework for learning patterns from data.

The network is built from simple, interconnected processing units called **neurons** (or perceptrons), which are organized into **layers**. The goal of an ANN is to learn to map inputs (like an image of a handwritten digit) to desired outputs (like the number "7").

---

## 1. The Building Block: The Perceptron

The simplest form of a neural network is the **Perceptron**, which is just a single neuron. It was developed in the 1950s and is the foundation for all modern ANNs.

A perceptron takes multiple binary inputs, $x_1, x_2, ..., x_n$, and produces a single binary output.



It works in four steps:
1.  **Receive Inputs ($x$):** It receives a set of input features. For example, $x_1 = 5$ and $x_2 = 3$.
2.  **Weigh Inputs ($w$):** Each input is multiplied by a **weight** ($w$). A weight represents the *importance* of that specific input. A higher weight means the network considers that input more important.
3.  **Summation ($\sum$):** All the weighted inputs are summed together. This is the **weighted sum**.
4.  **Activation Function ($f(z)$):** The weighted sum is passed through an **activation function**, which decides what the final output should be. In the original Perceptron, this was a simple *step function* (if the sum is > 0, output 1; else, output 0).

This process can be written as a mathematical formula. First, we calculate the weighted sum, which we'll call $z$:

$$z = (x_1 w_1) + (x_2 w_2) + ... + (x_n w_n)$$

This brings us to a crucial, often-missed component: the bias.

---

## 2. 💡 The Role of the Bias (Intercept)

If you look at the formula above, $z$ is just a combination of inputs and weights. What if all inputs $x$ are 0? Then $z$ will be 0, and the output will always be the same, regardless of the weights.

We need a way to shift the output *independently* of the inputs. This is the job of the **bias ($b$)**, which is just a constant number added to the weighted sum.

Think of the familiar equation for a line: $y = mx + b$.
* $m$ is the slope (the **weight**), deciding how much $x$ influences $y$.
* $b$ is the intercept (the **bias**), deciding where the line starts on the y-axis.

Without a bias, the line $y = mx$ *must* pass through the origin (0,0). With a bias, the line can be shifted up or down, making it much more flexible.

Our new, more robust formula for the weighted sum $z$ is:

$$z = (x_1 w_1 + x_2 w_2 + ... + x_n w_n) + b$$

In vector notation, this is written beautifully as:

$$z = \mathbf{w} \cdot \mathbf{x} + b$$

---

## 3. Linear vs. Non-Linear Activation Functions

After calculating $z$, the neuron passes it to an **activation function**. This is where the real power of ANNs comes from.

### The Problem with Linear Functions
What if we just used a simple linear function, like $f(z) = z$?
* Layer 1 (hidden): $z_1 = \mathbf{w_1} \cdot \mathbf{x} + b_1$
* Layer 2 (output): $z_2 = \mathbf{w_2} \cdot z_1 + b_2$

If we substitute $z_1$ into $z_2$, we get:
$z_2 = \mathbf{w_2} \cdot (\mathbf{w_1} \cdot \mathbf{x} + b_1) + b_2$
$z_2 = (\mathbf{w_2} \mathbf{w_1}) \cdot \mathbf{x} + (\mathbf{w_2} b_1 + b_2)$

This entire two-layer network is just a *new* linear function. We could have just used a single layer with different weights. **A stack of linear layers is mathematically the same as a single linear layer.** This means a linear network can *only* learn simple, linear patterns.

### The Power of Non-Linearity
To model complex, real-world data (like images, sound, or language), we need to introduce non-linearity. We do this by using **non-linear activation functions**.



These functions "bend" the data, allowing the network to approximate any complex, non-linear relationship.

* **Sigmoid:** $f(z) = \frac{1}{1 + e^{-z}}$
    * Squashes any number into a range between 0 and 1.
    * Useful for the *output* layer when predicting probabilities (e.g., "is this a cat or not?").
* **ReLU (Rectified Linear Unit):** $f(z) = \max(0, z)$
    * If $z$ is positive, it returns $z$. If $z$ is negative, it returns 0.
    * This is the most popular activation function for *hidden layers*. It's simple, fast, and helps prevent a problem called the "vanishing gradient."

---

## 4. ⚙️ The Math Behind an ANN (Forward Propagation)

Now let's assemble these perceptrons into a full network with one hidden layer. This process of passing data from input to output is called **Forward Propagation**.

**Our Network:**
* **Input Layer ($L_0$):** Has 2 features, $x_1, x_2$.
* **Hidden Layer ($L_1$):** Has 2 neurons.
* **Output Layer ($L_2$):** Has 1 neuron (predicting a single value).

**Step 1: From Input to Hidden Layer ($L_1$)**

* **Neuron 1 ($h_1$):**
    * Calculates its weighted sum, $z_1^{[1]}$:
        $$z_1^{[1]} = w_{11}^{[1]} x_1 + w_{12}^{[1]} x_2 + b_1^{[1]}$$
    * Applies activation function (e.g., ReLU): $a_1^{[1]} = \text{ReLU}(z_1^{[1]})$

* **Neuron 2 ($h_2$):**
    * Calculates its weighted sum, $z_2^{[1]}$:
        $$z_2^{[1]} = w_{21}^{[1]} x_1 + w_{22}^{[1]} x_2 + b_2^{[1]}$$
    * Applies activation function: $a_2^{[1]} = \text{ReLU}(z_2^{[1]})$

*(Note on notation: $w_{ji}^{[l]}$ is the weight from neuron $i$ in layer $l-1$ to neuron $j$ in layer $l$. $a_j^{[l]}$ is the activation (output) of neuron $j$ in layer $l$.)*

**Step 2: From Hidden Layer ($L_1$) to Output Layer ($L_2$)**

The *outputs* of the hidden layer ($a_1^{[1]}$ and $a_2^{[1]}$) are now the *inputs* for the output layer.

* **Output Neuron ($o_1$):**
    * Calculates its weighted sum, $z_1^{[2]}$:
        $$z_1^{[2]} = w_{11}^{[2]} a_1^{[1]} + w_{12}^{[2]} a_2^{[1]} + b_1^{[2]}$$
    * Applies activation function (e.g., Sigmoid, for probability):
        $$\hat{y} = a_1^{[2]} = \text{Sigmoid}(z_1^{[2]})$$

The final value, $\hat{y}$ (y-hat), is the network's **prediction**.

---

## 5. 📈 Loss Function vs. Cost Function

Our network has made a prediction, $\hat{y}$. Now we must compare it to the *actual* value, $y$, from our training data. This is how we measure **error**.

### Loss Function
A **Loss Function** measures the error for a **single training example**.
* **$y$**: The true label (e.g., 1, because the image *is* a cat)
* **$\hat{y}$**: The network's prediction (e.g., 0.2, or "20% sure it's a cat")
* **Loss ($L$):** The penalty for this mistake.

A common loss function for regression is **Squared Error**: $L = (y - \hat{y})^2$

### Cost Function
A **Cost Function ($J$)** is simply the **average of the loss** over the *entire training batch* (or dataset).
* $m$ = number of training examples.

A common cost function is **Mean Squared Error (MSE)**:
$$J(W, b) = \frac{1}{m} \sum_{i=1}^{m} L^{(i)} = \frac{1}{m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2$$

**The single most important goal of training is to find the weights ($W$) and biases ($b$) that minimize this Cost Function $J$.**

---

## 6. 📉 Learning: Backpropagation and the Chain Rule

We know our error (from the Cost Function). How do we use it to update every single weight and bias in the network?

The answer is **Backpropagation**.

The main tool we use is **Gradient Descent**. The "gradient" is just a derivative (a slope) that tells us: "If I *increase* this weight, does the Cost ($J$) go up or down, and by how much?"

We want to find the derivative of the Cost $J$ with respect to *every* weight $w$ and bias $b$ in the network (e.g., $\frac{\partial J}{\partial w_{11}^{[1]}}$).

We then update the weight using this rule:
**Weight Update Rule:** $w := w - \alpha \frac{\partial J}{\partial w}$
* $w$ is the current weight.
* $\alpha$ (alpha) is the **learning rate**—a small number (e.g., 0.01) that controls how big our update steps are.
* $\frac{\partial J}{\partial w}$ is the gradient. We subtract it because if the gradient is positive (meaning increasing $w$ increases the error), we want to move $w$ in the *negative* direction.

### The Problem: Finding the Gradient
Finding $\frac{\partial J}{\partial w_{11}^{[2]}}$ (a weight in the last layer) is easy. But how do we find $\frac{\partial J}{\partial w_{11}^{[1]}}$ (a weight buried in the *first* layer)?

The weight $w_{11}^{[1]}$ doesn't affect the Cost $J$ directly. It affects $z_1^{[1]}$, which affects $a_1^{[1]}$, which affects $z_1^{[2]}$, which affects $\hat{y}$, which *finally* affects the Cost $J$.

### The Solution: The Chain Rule
Calculus gives us a tool for this: the **Chain Rule**. It lets us find the derivative of a "chain" of functions by multiplying their individual derivatives.

To find the gradient of a weight in the first layer, $\frac{\partial J}{\partial w_{11}^{[1]}}$, we "back-propagate" the error:

$$\frac{\partial J}{\partial w_{11}^{[1]}} = \frac{\partial J}{\partial \hat{y}} \times \frac{\partial \hat{y}}{\partial z_1^{[2]}} \times \frac{\partial z_1^{[2]}}{\partial a_1^{[1]}} \times \frac{\partial a_1^{[1]}}{\partial z_1^{[1]}} \times \frac{\partial z_1^{[1]}}{\partial w_{11}^{[1]}}$$

Let's read this chain backward (which is how it's calculated):
1.  **$\frac{\partial z_1^{[1]}}{\partial w_{11}^{[1]}}$:** How much does the weighted sum ($z$) change when we change the weight ($w$)? (This is just $x_1$).
2.  **$\frac{\partial a_1^{[1]}}{\partial z_1^{[1]}}$:** How much does the activation ($a$) change when we change the weighted sum ($z$)? (This is the derivative of the ReLU function).
3.  **$\frac{\partial z_1^{[2]}}{\partial a_1^{[1]}}$:** How much does the *next* layer's sum change when we change this neuron's activation? (This is just $w_{11}^{[2]}$).
4.  **...and so on...** all the way back from the final Cost.

**Backpropagation** is simply the name of the algorithm that efficiently computes this chain rule for all weights in the network, starting from the final error and working its way backward.

---

## 🔄 Summary: The Full Cycle of Learning

1.  **Initialize:** Start with random values for all weights ($W$) and biases ($b$).
2.  **Forward Propagation:** Take a batch of training data. Feed it through the network to get predictions, $\hat{y}$.
3.  **Calculate Cost:** Compare the predictions $\hat{y}$ to the true labels $y$ using the Cost Function $J$.
4.  **Backpropagation:** Calculate the gradient of the Cost $J$ with respect to every weight and bias in the network using the chain rule.
5.  **Gradient Descent:** Update all weights and biases using the update rule: $w := w - \alpha \frac{\partial J}{\partial w}$.
6.  **Repeat:** Go back to Step 2 and repeat this process for many "epochs" (passes over the entire dataset).

Over time, the weights and biases will slowly converge to values that minimize the Cost Function, and your network will become an expert at its given task.