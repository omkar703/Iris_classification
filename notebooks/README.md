# Iris Classification Research Notebooks

This directory contains the computational research notebooks for botanical morphometric feature extraction, exploratory distribution analysis, Gaussian Naive Bayes model training, and probabilistic inference validation.

---

## 1. Notebook Index

| Notebook | Focus | Primary Inputs | Key Outputs |
| :--- | :--- | :--- | :--- |
| [`01_iris_morphometric_classification.ipynb`](01_iris_morphometric_classification.ipynb) | End-to-end exploratory data analysis, class separability, GaussianNB training, probability calibration, and model serialization | `Iris_classification/data/raw/iris.csv` or `sklearn.datasets.load_iris` | Serialized model (`iris_model.pkl`), validation metrics, decision boundary evaluations |

---

## 2. Research Workflow & Execution Order

1. **Data Acquisition & Sanity Check:**
   - Loads the 150-specimen Fisher (1936) morphometric dataset.
   - Verifies 4 continuous morphological features ($X \in \mathbb{R}^{150 \times 4}$) and balanced tri-class targets ($y \in \{0, 1, 2\}$).
2. **Exploratory Morphometrics:**
   - Evaluates petal and sepal length/width distributions, inter-feature Pearson correlations, and pairwise class distributions.
   - Confirms linear separability of *Iris setosa* and non-linear boundary characteristics between *Iris versicolor* and *Iris virginica*.
3. **Probabilistic Classifier Training:**
   - Instantiates `sklearn.naive_bayes.GaussianNB()`.
   - Computes empirical class priors $P(C_k)$ and feature likelihood parameters ($\mu_{ki}, \sigma^2_{ki}$).
4. **Validation & Inference Testing:**
   - Validates multi-class prediction probabilities $\hat{P}(C_k \mid x)$ using maximum a posteriori (MAP) estimation.
   - Exports the verified estimator directly to `iris_model.pkl`.

---

## 3. Environment & Execution Instructions

Launch the Jupyter kernel within the project virtual environment:

```bash
# From the repository root or project directory
./.venv/bin/jupyter lab notebooks/01_iris_morphometric_classification.ipynb
```

Alternatively, execute headlessly via `pytest` or `nbconvert`:

```bash
./.venv/bin/jupyter nbconvert --to notebook --execute notebooks/01_iris_morphometric_classification.ipynb
```

All cell execution outputs are preserved for academic reproducibility.
