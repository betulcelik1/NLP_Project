# EXP-05 — Word + character TF-IDF with lexical features (Track 1 English)

## Objective
Assess whether character-level n-grams improve performance beyond EXP-04's separate word TF-IDF and three lexical features.

## Data and setup
- Train: 3,042 rows; Dev: 660 rows; four NLI labels.
- EXP-04 features: separate text1/text2 word unigram/bigram TF-IDF plus Overlap1, Overlap2 and normalized LengthDiff (11,039 features).
- Two independently fitted character TF-IDF vectorizers, one per text field, analyzer=char, ngram_range=(2,5), max_features=10000, min_df=2.
- 20,000 additional character features; total 31,039 features. Train shape (3042,31039); Dev shape (660,31039).
- Vectorizers fitted on training data only.
- LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42).

## Classification results on Dev
| Metric | EXP-04 | EXP-05 |
|---|---:|---:|
| Accuracy | 0.6091 | **0.6212** |
| Macro F1 | 0.5921 | **0.5979** |
| Weighted F1 | 0.6180 | **0.6263** |

### EXP-05 per-class scores
| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| BACKWARD_ENTAILMENT | 0.7403 | 0.7053 | 0.7224 | 190 |
| EQUIVALENCE | 0.5439 | 0.5345 | 0.5391 | 174 |
| FORWARD_ENTAILMENT | 0.7624 | 0.7263 | 0.7439 | 190 |
| NEGATIVE_OTHER | 0.3543 | 0.4245 | 0.3863 | 106 |

## Official DiCo-NLI scorer (Dev)
Submission: (660, 2), 0 duplicate instance IDs. Return code: 0.

| Experiment | Weighted F1 | SoftCons | HardCons |
|---|---:|---:|---:|
| EXP-01 | 0.4612 | 0.5957 | 0.3791 |
| EXP-02 | 0.4844 | 0.7653 | 0.4765 |
| EXP-03 | 0.6151 | 0.7762 | 0.6029 |
| EXP-04 | 0.6180 | 0.7762 | 0.6029 |
| **EXP-05** | **0.6263** | **0.7906** | **0.6318** |

Compared with EXP-04: +0.0083 Weighted F1, +0.0144 SoftCons, +0.0289 HardCons. All metrics are reported at four-decimal precision from the official scorer output.

## Interpretation and limitations
EXP-05 achieves the highest Dev scores of the five experiments. EQUIVALENCE F1 improved relative to EXP-04 (0.4840 to 0.5391), whereas NEGATIVE_OTHER F1 decreased slightly (0.3984 to 0.3863). The same Dev set was repeatedly used to choose features, so these results may overstate generalization. A held-out, pair-grouped validation strategy should be considered before further model selection.

## Reproducibility
- Notebook: `notebooks/01_TFIDF_LogisticRegression_Baseline.ipynb` (the latest Colab cells may not yet be synchronized to GitHub).
- Local Colab results: `/content/NLP_Project/results/exp_05/dev_submission.csv` and `official_scores/scores.json` (not committed).
- Metrics provided from the user's Colab execution.
