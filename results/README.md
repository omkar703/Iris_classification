# Iris Classification Experimental Results & Artifacts

This directory stores empirical evaluation metrics, classification reports, sample batch predictions, and research figures generated from the Gaussian Naive Bayes taxonomic model.

---

## 1. Directory Structure

```
results/
├── README.md
├── figures/
│   ├── README.md
│   ├── iris_specimen_morphology.png    # High-resolution scatter matrix & decision regions
│   └── iris_botany_anatomy.svg         # Morphological perianth landmark architecture
├── metrics/
│   └── classification_report.json      # Empirical precision, recall, F1, and confusion matrix
└── predictions/
    ├── README.md
    └── sample_predictions.json         # Real model inference outputs with posterior probabilities
```

---

## 2. Empirical Performance Summary

Evaluated on the full 150-specimen Fisher dataset with verified GaussianNB model parameters:

- **Overall Accuracy:** $96.0\%$ ($144 / 150$ correct specimens)
- **Macro Average F1-Score:** $0.96$
- **Weighted Average F1-Score:** $0.96$
- **Class-Specific Breakdown:**
  - *Setosa:* Precision = $1.00$, Recall = $1.00$, F1 = $1.00$ ($100\%$ linear separation)
  - *Versicolor:* Precision = $0.94$, Recall = $0.94$, F1 = $0.94$ ($47 / 50$)
  - *Virginica:* Precision = $0.94$, Recall = $0.94$, F1 = $0.94$ ($47 / 50$)

Full serialized metrics are available in [`metrics/classification_report.json`](metrics/classification_report.json).
