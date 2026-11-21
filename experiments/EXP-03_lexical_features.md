# EXP-03 — Separate TF-IDF + lexical overlap features (Track 1 English)

## Motivation
EXP-02 improved directional consistency but struggled on NEGATIVE_OTHER. Investigate whether simple token-overlap and relative-length features help.

## Data and features
- Train: 3,042 instances; Dev: 660 instances.
- Base representation: independent text1/text2 word unigram/bigram TF-IDF matrices (EXP-02), 11,036 features.
- Added four features calculated from lowercase whitespace-split token **sets**:
  1. Jaccard overlap: |A ∩ B| / |A ∪ B|.
  2. text1 overlap: |A ∩ B| / |A|.
  3. text2 overlap: |A ∩ B| / |B|.
  4. Normalized absolute unique-token count difference: ||A| − |B|| / max(|A|, |B|, 1).
- Final sparse matrices: Train (3042, 11040); Dev (660, 11040).
- Model: LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42).
- No change to previous EXP-01 and EXP-02 models.

## Development classification metrics
| Metric | EXP-01 | EXP-02 | EXP-03 |
|---|---:|---:|---:|
| Accuracy | 0.4500 | 0.4818 | **0.6061** |
| Macro F1 | 0.4449 | 0.4627 | **0.5884** |
| Weighted F1 | 0.4612 | 0.4844 | **0.6151** |

## Official DiCo-NLI scorer (Dev)
Submission: 660 rows, columns `instance_id,label`, 0 duplicate IDs. Scorer return code: 0.

| Metric | EXP-01 | EXP-02 | EXP-03 |
|---|---:|---:|---:|
| weighted_f1 | 0.4612 | 0.4844 | **0.6151** |
| soft_cons | 0.5957 | 0.7653 | **0.7762** |
| hard_cons | 0.3791 | 0.4765 | **0.6029** |

Relative to EXP-02, EXP-03 gains +0.1307 weighted F1, +0.0109 SoftCons, and +0.1264 HardCons (all approximate due to rounded scorer values).

## Interpretation and limitations
Lexical overlap and relative-length features are associated with substantially better Dev classification and pair-level correctness, with a smaller gain in directional consistency. These results do not establish test-set generalization. A useful follow-up is a feature ablation and error analysis to identify which of the four new features matter.

## Reproducibility
- Colab notebook: `notebooks/01_TFIDF_LogisticRegression_Baseline.ipynb` (new EXP-03 cells are in the user's Colab session and may not yet be synced to GitHub).
- Local results: `/content/NLP_Project/results/exp_03/dev_submission.csv`, `official_scores/scores.json` (not committed).
- This document records user-reported scorer outputs.
