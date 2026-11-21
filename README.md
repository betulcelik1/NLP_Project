# NLP_Project# NLP_Project
# DiCo-NLI — SWE420 Natural Language Processing Project

## Project Overview

This project investigates Directional-Consistent Fine-Grained Natural Language Inference (DiCo-NLI), focusing on the classification of ordered phrase pairs and the consistency of predictions when pair direction is reversed.

## Task

SemEval-2027 Task 2: Directional-Consistent Fine-Grained Natural Language Inference (DiCo-NLI)

Official Task:
https://inigolopezgazpio.net/SemEval-2027-Task-2-DiCo-NLI/

## Labels

- EQUIVALENCE
- FORWARD_ENTAILMENT
- BACKWARD_ENTAILMENT
- NEGATIVE_OTHER

## Dataset

Track: English (Track 1)

- Training data
- Development data
- Reference files for official local scoring

Dataset source:
https://inigolopezgazpio.net/SemEval-2027-Task-2-DiCo-NLI/final_data/

## Project Objectives

1. Analyze the DiCo-NLI dataset.
2. Develop a TF-IDF + Logistic Regression baseline.
3. Fine-tune Transformer-based models.
4. Investigate directional consistency.
5. Evaluate models using task-specific metrics.
6. Analyze classification and consistency errors.

## Experimental Pipeline

Dataset Analysis
→ Preprocessing
→ TF-IDF Baseline
→ Transformer Fine-tuning
→ Directional Consistency Analysis
→ Evaluation
→ Error Analysis

## Repository Structure

- `notebooks/`: Experiments and analyses
- `src/`: Reusable source code
- `data/`: Dataset instructions
- `results/`: Experimental outputs
- `docs/`: Literature review and research plan
- `references/`: Sources

## Research Questions

- RQ1: How effectively can traditional machine learning methods classify fine-grained NLI relations compared to Transformer-based models?
- RQ2: Which Transformer model performs better on directional-consistent fine-grained NLI?
- RQ3: To what extent do models maintain consistency when premise and hypothesis are reversed?
- RQ4: Can reverse-pair-aware training improve directional consistency without sacrificing classification performance?

## Current Progress

- [ ] Literature review
- [ ] Dataset analysis
- [ ] Baseline
- [ ] Transformer
- [ ] Directional consistency
- [ ] Official evaluation
- [ ] Error analysis

## Team Contributions

- Member 1: Literature and task analysis
- Member 2: Dataset and preprocessing
- Member 3: Baseline model
- Member 4: Transformer experiments
- Member 5: Directional consistency and evaluation

## References

- DiCo-NLI Official Repository
- DiCo-NLI Pilot Paper
- BERT Paper
- SNLI
- MultiNLI