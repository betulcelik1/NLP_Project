# EXP-01 — TF-IDF + Logistic Regression (English Track 1)

## Dataset and features
- Train: 3,042 examples; Dev: 660 examples; four NLI labels.
- text1 and text2 were concatenated in order with Text1: and Text2: prefixes.
- TF-IDF word unigrams and bigrams, max_features=50000; 8,847 features fitted on Train only.
- Logistic Regression max_iter=1000, class_weight=balanced.

## Development results
| Metric | Value |
|---|---:|
| Accuracy | 0.4500 |
| Macro F1 | 0.4449 |
| Weighted F1 | 0.4612 |
| Correct / incorrect | 297 / 363 |

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| BACKWARD_ENTAILMENT | 0.56 | 0.44 | 0.50 |
| EQUIVALENCE | 0.49 | 0.45 | 0.47 |
| FORWARD_ENTAILMENT | 0.54 | 0.47 | 0.50 |
| NEGATIVE_OTHER | 0.24 | 0.43 | 0.31 |

## Errors and reversed pairs
- 142 of 363 errors (39.1%) were false NEGATIVE_OTHER predictions from other classes.
- 56 examples confused FORWARD_ENTAILMENT and BACKWARD_ENTAILMENT (31 backward to forward, 25 forward to backward).
- Dev has 277 pair_id groups of size 2 and 106 singleton groups.
- Of paired groups, 105 had both correct, 41 had only one correct, and 131 had both incorrect.
- Accuracy over the 554 paired-group examples: 0.4531; this is NOT an official SoftCons or HardCons score.

## Files and next steps
- Notebook: [01_TFIDF_LogisticRegression_Baseline.ipynb](../notebooks/01_TFIDF_LogisticRegression_Baseline.ipynb)
- Colab generated metrics.json, dev_predictions.csv and error_analysis.csv under /content/NLP_Project/results/baseline/. These outputs have not been committed yet.
- Next: run the official scorer on properly formatted Dev predictions; record Weighted F1, SoftCons, HardCons, then improve representation and compare with Transformers.
