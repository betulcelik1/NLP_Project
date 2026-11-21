# EXP-05 — Five-fold grouped cross-validation and OOF error analysis

## Protocol
- Source: official Track 1 training data, 3,042 examples.
- Grouping: union-find components sharing either `pair_id` or an identical unordered text pair after lowercase and whitespace normalization.
- 1,725 unique groups; GroupKFold(n_splits=5).
- Every fold had **0 shared pair IDs** and **0 shared normalized unordered text pairs** between training and validation.
- Each fold has 2,433–2,434 training examples and 608–609 validation examples.
- EXP-05 architecture: separate word TF-IDF (1,2) and character TF-IDF (2,5; max_features=10000; min_df=2) for text1 and text2, plus three lexical overlap/length features; LogisticRegression(max_iter=1000,class_weight="balanced",random_state=42).
- TF-IDF vectorizers refitted on each fold's training portion only.
- **Reproducibility caveat:** the CV notebook reimplemented lexical tokenization with `re.findall(r"\b\w+\b", text.lower())`; it has not yet been checked for exact equivalence to the original EXP-05 notebook's lexical feature function. This may affect strict head-to-head comparability.

## Fold scores
| Fold | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| 1 | 0.6355 | 0.6151 | 0.6386 |
| 2 | 0.5944 | 0.5639 | 0.5928 |
| 3 | 0.6118 | 0.5923 | 0.6198 |
| 4 | 0.6316 | 0.6029 | 0.6343 |
| 5 | 0.5526 | 0.5307 | 0.5634 |
| **Mean** | **0.6052** | **0.5810** | **0.6098** |
| **Sample SD** | **0.0337** | **0.0339** | **0.0315** |

## Out-of-fold (OOF) results
- All 3,042 training instances have one OOF prediction; 0 missing.
- OOF Accuracy: 0.6052; Macro F1: 0.5816; Weighted F1: 0.6107.
- OOF Weighted F1 is not identical to the unweighted average of fold-level Weighted F1 because of nonlinear aggregation and differing support.

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| BACKWARD_ENTAILMENT | 0.7245 | 0.7015 | 0.7128 | 881 |
| EQUIVALENCE | 0.5443 | 0.5212 | 0.5325 | 802 |
| FORWARD_ENTAILMENT | 0.7337 | 0.6879 | 0.7100 | 881 |
| NEGATIVE_OTHER | 0.3345 | 0.4163 | 0.3709 | 478 |

### OOF confusion matrix (rows = gold; columns = predicted)
| Gold / Pred | BACKWARD_ENTAILMENT | EQUIVALENCE | FORWARD_ENTAILMENT | NEGATIVE_OTHER |
|---|---:|---:|---:|---:|
| BACKWARD_ENTAILMENT | 618 | 104 | 43 | 116 |
| EQUIVALENCE | 114 | 418 | 111 | 159 |
| FORWARD_ENTAILMENT | 44 | 110 | 606 | 121 |
| NEGATIVE_OTHER | 77 | 136 | 66 | 199 |

## Fold 5
- Validation: 608; errors: 272.
- Accuracy: 0.5526; Macro F1: 0.5307; Weighted F1: 0.5634.
- Fold 5 F1: BACKWARD_ENTAILMENT 0.6587, EQUIVALENCE 0.5000, FORWARD_ENTAILMENT 0.6706, NEGATIVE_OTHER 0.2933.
- Frequent errors: EQUIVALENCE→NEGATIVE_OTHER (34); BACKWARD_ENTAILMENT→NEGATIVE_OTHER (33); FORWARD_ENTAILMENT→NEGATIVE_OTHER (31); NEGATIVE_OTHER→EQUIVALENCE (28).
- Example failures: `Conservative opposition` vs `Conservatives` (FORWARD_ENTAILMENT predicted NEGATIVE_OTHER); `Canadian government` vs `Canada` (FORWARD_ENTAILMENT predicted BACKWARD_ENTAILMENT); `bubbles` vs `a bubble` (FORWARD_ENTAILMENT predicted EQUIVALENCE).

## Interpretation
EXP-05's performance varies by fold (Weighted F1 0.5634–0.6386), and NEGATIVE_OTHER is weakest. Distinguishing EQUIVALENCE from NEGATIVE_OTHER is a major source of errors. The official Dev Weighted F1 was 0.6263, but it is not a controlled comparison with the CV mean, since the evaluation data and training sizes differ. These results motivate testing a contextual transformer as EXP-06. Model selection on repeatedly inspected Dev data remains a caveat.

Results supplied from the user's Colab execution. CV notebook should be committed separately after verification.
