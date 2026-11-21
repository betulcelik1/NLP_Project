# EXP-05 — Clean grouped validation

## Purpose
Evaluate EXP-05 on an internal split without sharing pair IDs or identical normalized unordered text pairs between train and validation.

## Split and leakage controls
- Source: 3,042 training examples.
- Group construction: union rows sharing `pair_id` or the same unordered text pair after lowercasing and whitespace normalization.
- GroupShuffleSplit(test_size=0.20, random_state=42).
- Internal train: 2,434 rows; validation: 608 rows.
- Shared pair IDs: **0**.
- Shared normalized unordered text pairs: **0**.
- Validation labels: FORWARD_ENTAILMENT 175, BACKWARD_ENTAILMENT 175, EQUIVALENCE 170, NEGATIVE_OTHER 88.
- This blocks exact normalized pair duplicates, but not semantic near-duplicates.

## Modeling
- EXP-05 architecture: separate text1/text2 word TF-IDF (1,2), separate text1/text2 character TF-IDF (2,5; max_features=10000; min_df=2), and Overlap1, Overlap2, normalized LengthDiff lexical features.
- TF-IDF vectorizers refit **only** on the 2,434 internal training rows.
- LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42).
- Train matrix: (2434, 29349); validation matrix: (608, 29349).

## Validation results
| Evaluation | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| EXP-05 official Dev (660 rows) | 0.6212 | 0.5979 | 0.6263 |
| First pair_id-only internal validation (609 rows) | 0.6174 | 0.5885 | 0.6190 |
| **Clean grouped internal validation (608 rows)** | **0.6266** | **0.6024** | **0.6340** |

### Clean validation class metrics
| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| BACKWARD_ENTAILMENT | 0.7544 | 0.7371 | 0.7457 | 175 |
| EQUIVALENCE | 0.5696 | 0.5294 | 0.5488 | 170 |
| FORWARD_ENTAILMENT | 0.7673 | 0.6971 | 0.7305 | 175 |
| NEGATIVE_OTHER | 0.3333 | 0.4545 | 0.3846 | 88 |

## Interpretation
Performance remains similar when shared pair IDs and exact normalized unordered text-pair duplicates are prevented across internal train/validation. NEGATIVE_OTHER remains the weakest class. Different splits and training sizes mean that the 0.6340 internal score and 0.6263 official Dev score are not a controlled head-to-head comparison. EXP-05 architecture was previously selected using official Dev, so internal validation is an additional robustness check, **not** a fully independent test. Consider repeated grouped cross-validation and a final untouched test set.

Metrics are from user-run Colab output. No official scorer SoftCons/HardCons was run on this internal split.
