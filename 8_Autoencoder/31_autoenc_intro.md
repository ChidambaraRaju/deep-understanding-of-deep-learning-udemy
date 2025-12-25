# Autoencoders in Deep Learning



## 1. Introduction

Autoencoders are a special class of neural networks designed to **learn efficient representations of data**.
Instead of mapping inputs to external labels, autoencoders are trained to **reconstruct their own input**.

They are foundational models for:
- Dimensionality reduction
- Representation learning
- Data compression
- Denoising
- Anomaly detection
- Pretraining deep networks

Understanding autoencoders builds intuition for many modern architectures used in computer vision and generative modeling.

---

## 2. What Is an Autoencoder?

The term *autoencoder* comes from two words:
- **Auto** → self
- **Encode** → transform data into another form

An autoencoder is a neural network that:
- Takes an input vector
- Encodes it into a compressed representation
- Decodes it back to reconstruct the original input

The learning objective is:

$$
\hat{x} \approx x
$$

where:
- $x$ is the input
- $\hat{x}$ is the reconstructed output

---

## 3. Core Architecture

A typical autoencoder has a symmetric structure:

```
Input → Encoder → Bottleneck → Decoder → Output
```

### Main Components

**Input Layer**
- Receives the original data
- Size equals the number of input features

**Encoder**
- A stack of hidden layers
- Gradually reduces dimensionality
- Learns compressed representations

**Bottleneck (Latent / Code Layer)**
- The smallest layer in the network
- Stores the compressed representation
- Forces the network to learn meaningful structure

**Decoder**
- Mirrors the encoder
- Gradually increases dimensionality
- Reconstructs the input

**Output Layer**
- Produces reconstructed data
- Typically same size as the input layer

---

## 4. Encoder–Decoder Abstraction

Autoencoders are often visualized abstractly as:

$$
x \;\xrightarrow{\text{Encoder}}\; z \;\xrightarrow{\text{Decoder}}\; \hat{x}
$$

where:
- $x$ is the input
- $z$ is the latent (compressed) representation
- $\hat{x}$ is the reconstructed output

---

## 5. Layer Size Relationships

Let:
- $K$ = number of input features
- $M$ = number of units in the bottleneck layer
- $N$ = number of units in encoder/decoder hidden layers

Typical properties:
- Output size $\approx K$
- Bottleneck size $M \ll K$
- Encoder and decoder are often mirrors

Key constraint:

$$
K > M
$$

This prevents the model from simply learning the identity function.

---

## 6. What Does an Autoencoder Learn?

Autoencoders learn:
- Compact representations of data
- Latent features that capture essential structure
- Non‑linear dimensionality reduction

Example (MNIST):
- Input size: $784$
- Bottleneck size: $50$
- Compression ratio $\approx 16 \times$

Despite heavy compression, reconstructed images remain visually similar to the originals.

---

## 7. Training Objective

The training goal is to **minimize reconstruction error**:

$$
\min \; \lVert x - \hat{x} \rVert
$$

The model adjusts encoder and decoder weights so that the output matches the input as closely as possible.

---

## 8. Loss Functions for Autoencoders

### 8.1 Mean Squared Error (MSE)

The most common loss function:

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (x_i - \hat{x}_i)^2
$$

Properties:
- Penalizes large reconstruction errors
- Suitable for continuous data
- Commonly used for images

---

### 8.2 Cross‑Entropy Loss

Used when:
- Inputs represent probabilities
- Data values are scaled between $0$ and $1$

Binary Cross‑Entropy:

$$
\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^{N}
\left[x_i \log(\hat{x}_i) + (1 - x_i)\log(1 - \hat{x}_i)\right]
$$

Choice of loss function is often empirical.

---

## 9. Are Autoencoders Supervised or Unsupervised?

Autoencoders are often called **unsupervised**, because:
- No external labels are provided

However:
- The input itself serves as the target
- The model is explicitly supervised to match $x$ and $\hat{x}$

More accurately, autoencoders belong to:

> **Self‑supervised learning**

---

## 10. Applications of Autoencoders

### 10.1 Dimensionality Reduction
- Learn non‑linear embeddings
- Alternative to PCA

### 10.2 Data Compression
- Reduce storage for images or videos
- Task‑specific compression

### 10.3 Denoising
- Remove noise from corrupted inputs
- Learn robust representations

### 10.4 Feature Extraction
- Use latent vectors $z$ as features
- Feed into classifiers or clustering algorithms

### 10.5 Anomaly Detection
- Normal data reconstructs well
- Anomalies have high reconstruction error

### 10.6 Pretraining Deep Networks
- Initialize deep models with learned representations
- Useful in image processing and vision tasks

---

## 11. Key Takeaways

- Autoencoders reconstruct their own inputs
- Compression occurs at the bottleneck layer
- Bottleneck size controls information capacity
- Mean Squared Error is the most common loss
- Latent space captures meaningful structure
- Autoencoders are powerful representation learners

---

