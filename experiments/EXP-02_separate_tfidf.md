# EXP-02 — Separate Text1/Text2 TF-IDF + Logistic Regression (English Track 1)

## Goal
Test whether encoding text1 and text2 in separate feature spaces improves direction-sensitive NLI classification and pair consistency relative to EXP-01.

## Data
- Train: 3,042 instances; Dev: 660 instances.
- Four labels: BACKWARD_ENTAILMENT, EQUIVALENCE, FORWARD_ENTAILMENT, NEGATIVE_OTHER.
- Training-only TF-IDF fitting; Dev transformed without fitting.

## Method
- Two independent TfidfVectorizer instances, each with word ngram_range=(1,2) and max_features=50000.
- text1 vocabulary: 5,509; text2 vocabulary: 5,527.
- Concatenated sparse feature matrix: Train (3042,11036), Dev (660,11036).
- LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42).
- Separate model/feature variables from EXP-01.

## Development classification metrics
| Metric | EXP-01 | EXP-02 |
|---|---:|---:|
| Accuracy | 0.4500 | 0.4818 |
| Macro F1 | 0.4449 | 0.4627 |
| Weighted F1 | 0.4612 | 0.4844 |

## Official DiCo-NLI evaluation (Dev)
Official evaluation returned exit code 0; submission contained 660 rows, exactly instance_id and label columns, and zero duplicate IDs.

| Metric | EXP-01 | EXP-02 | Delta (percentage points, rounded) |
|---|---:|---:|---:|
| weighted_f1 | 0.4612 | 0.4844 | +2.32 |
| soft_cons | 0.5957 | 0.7653 | +16.96 |
| hard_cons | 0.3791 | 0.4765 | +9.74 |

Note: EXP-02 official metrics are recorded to four decimal places from the user's scorer output. EXP-01 has higher-precision values in EXP-01_baseline.md.

## Interpretation
Keeping text1 and text2 in separate TF-IDF spaces improves all three official Dev metrics. The largest gain is in SoftCons, suggesting stronger directional consistency under argument reversal. This is an observed Dev result, not a guarantee of test-set improvement.

## Reproducibility
- Notebook: notebooks/01_TFIDF_LogisticRegression_Baseline.ipynb (the local Colab notebook includes EXP-02 cells; GitHub notebook may need syncing).
- Local Colab output: results/iteration_3/dev_submission.csv and results/iteration_3/official_scores/scores.json.
- The local output files have not been committed to GitHub.
