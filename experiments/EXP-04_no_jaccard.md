# EXP-04 — Remove Jaccard feature (Track 1 English)

## Motivation
The EXP-03 leave-one-feature-out ablation showed a small increase in Dev Weighted F1 when the Jaccard overlap feature was removed.

## Setup
- Train: 3,042 instances; Dev: 660 instances; four DiCo-NLI labels.
- Separate text1/text2 word unigram/bigram TF-IDF (11,036 dimensions).
- Retained three lexical features: Overlap1 (intersection/text1 unique token count), Overlap2 (intersection/text2 unique token count), and normalized absolute difference in unique-token counts (LengthDiff).
- Removed: Jaccard.
- Final feature dimension: 11,039.
- LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42).

## Classification metrics on Dev
| Metric | EXP-03 | EXP-04 |
|---|---:|---:|
| Accuracy | 0.6061 | **0.6091** |
| Macro F1 | 0.5884 | **0.5921** |
| Weighted F1 | 0.6151 | **0.6180** |

## Official DiCo-NLI scorer on Dev
Submission has 660 rows and two columns (`instance_id`, `label`), zero duplicate IDs. Official scorer returned exit code 0.

| Metric | EXP-03 | EXP-04 | Difference |
|---|---:|---:|---:|
| weighted_f1 | 0.6151 | **0.6180** | +0.0029 |
| soft_cons | 0.7762 | 0.7762 | 0.0000 |
| hard_cons | 0.6029 | 0.6029 | 0.0000 |

Metrics are rounded to four decimal places as printed by the scorer.

## EXP-03 feature ablation (Dev Weighted F1)
| Removed feature | Weighted F1 | Delta vs full EXP-03 |
|---|---:|---:|
| Jaccard | 0.6180 | +0.0029 |
| Overlap1 | 0.6107 | -0.0044 |
| Overlap2 | 0.5972 | -0.0179 |
| LengthDiff | 0.6019 | -0.0132 |

## Interpretation and caveats
EXP-04 is the current best **Dev Weighted F1** candidate, with unchanged reported SoftCons and HardCons. Its improvement is small, and the same Dev set was used to select the feature to remove and evaluate the resulting model. Avoid interpreting this as evidence of improved generalization until evaluated on held-out data. Ablation effects may reflect feature interactions.

## Reproducibility
- Notebook in user's Colab contains new experiment cells; the GitHub notebook may not yet be synced.
- Colab outputs: `/content/NLP_Project/results/exp_04/dev_submission.csv` and `official_scores/scores.json` (not committed).
- Official scorer outputs supplied by the user.
