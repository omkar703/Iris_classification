# Iris Botanical Morphometry Dataset

This directory archives the primary botanical dataset underlying the taxonomic classification models.

---

## 1. Dataset Provenance & Attribution

- **Dataset Name:** Iris Flower Morphometric Dataset
- **Originating Collector:** Dr. Edgar Anderson (Gaspé Peninsula, Quebec, Canada)
- **Seminal Publication:** Fisher, R. A. (1936). *The use of multiple measurements in taxonomic problems*. Annals of Eugenics, 7(2), 179–188.
- **Specimens:** 150 total observed flowers (50 per taxon).
- **Taxa:**
  - `setosa` (*Iris setosa*)
  - `versicolor` (*Iris versicolor*)
  - `virginica` (*Iris virginica*)

---

## 2. Feature Schema & Physical Dimensions

All morphological attributes represent continuous measurements recorded in centimeters ($\text{cm}$):

| Feature Name | Short Alias | Type | Range ($\text{cm}$) | Mean $\pm$ Std ($\text{cm}$) | Taxonomic Significance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `sepal length (cm)` | `sepal_length` | Float | 4.30 – 7.90 | $5.84 \pm 0.83$ | Calyx elongation |
| `sepal width (cm)` | `sepal_width` | Float | 2.00 – 4.40 | $3.06 \pm 0.44$ | Outer perianth breadth |
| `petal length (cm)` | `petal_length` | Float | 1.00 – 6.90 | $3.76 \pm 1.76$ | Inner perianth blade length |
| `petal width (cm)` | `petal_width` | Float | 0.10 – 2.50 | $1.20 \pm 0.76$ | Inner perianth blade width |
| `target` | `class` | Integer | 0, 1, 2 | Balanced (50/50/50) | Class label identifier |
| `species` | `taxon` | String | setosa, versicolor, virginica | Balanced (50/50/50) | Binomial botanical name |

---

## 3. Storage Layout

```
data/
├── README.md
└── raw/
    └── iris.csv      # Raw tabular dataset (150 rows x 6 columns)
```

No missing values or null entries exist within this historical benchmark dataset.
