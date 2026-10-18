# Iris Classification: Scientific Methodology

## 1. Theoretical Formulation

The classification framework employs a **Gaussian Naive Bayes (GNB)** generative probabilistic model.

Given a continuous feature vector $\mathbf{x} = [x_1, x_2, x_3, x_4]^T \in \mathbb{R}^4$ representing:
1. $x_1$: Sepal Length ($\text{cm}$)
2. $x_2$: Sepal Width ($\text{cm}$)
3. $x_3$: Petal Length ($\text{cm}$)
4. $x_4$: Petal Width ($\text{cm}$)

And a discrete class variable $C_k \in \{\text{setosa}, \text{versicolor}, \text{virginica}\}$ where $k \in \{0, 1, 2\}$, Bayes' Theorem yields:

$$P(C_k \mid \mathbf{x}) = \frac{P(C_k) \, P(\mathbf{x} \mid C_k)}{P(\mathbf{x})} = \frac{P(C_k) \prod_{i=1}^{4} P(x_i \mid C_k)}{\sum_{j=0}^{2} P(C_j) \prod_{i=1}^{4} P(x_i \mid C_j)}$$

Under the conditional independence assumption, each feature likelihood $P(x_i \mid C_k)$ follows a univariate normal distribution:

$$P(x_i \mid C_k) = \frac{1}{\sqrt{2\pi \sigma_{ki}^2}} \exp\left( -\frac{(x_i - \mu_{ki})^2}{2\sigma_{ki}^2} \right)$$

Where:
- $\mu_{ki}$ is the empirical sample mean of feature $i$ for class $C_k$.
- $\sigma_{ki}^2$ is the empirical sample variance of feature $i$ for class $C_k$.
- $P(C_k) = \frac{N_k}{N} = \frac{50}{150} = \frac{1}{3}$ represents balanced prior probabilities.

Decision classification is executed via the Maximum A Posteriori (MAP) decision rule:

$$\hat{y} = \arg\max_{k \in \{0, 1, 2\}} \left( \ln P(C_k) - \frac{1}{2} \sum_{i=1}^4 \ln(2\pi \sigma_{ki}^2) - \sum_{i=1}^4 \frac{(x_i - \mu_{ki})^2}{2\sigma_{ki}^2} \right)$$

---

## 2. Model Training & Serialization Pipeline

1. **Partitioning:** Evaluated using standard cross-validation and historical 80/20 train/test splits.
2. **Parameter Estimation:** Closed-form maximum likelihood estimation (MLE) of class-conditional means and variances across all 4 continuous dimensions.
3. **Artifact Persistence:** Model weights and prior parameters serialized using `joblib` into `iris_model.pkl` for sub-millisecond production inference.
