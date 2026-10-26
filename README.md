# PhishGuard AI — Machine Learning

## Overview
PhishGuard AI uses Logistic Regression to classify URLs as legitimate or phishing based on 18 URL-based features.

## Final Model
- Algorithm: Logistic Regression
- Preprocessing: StandardScaler
- Features: 18
- Model parameters: phishguard_model_final.json
- Prediction threshold: 0.5
- Labels: 0 = Legitimate, 1 = Phishing

## Evaluation
The selected Model A was evaluated using a hostname-separated test set.

- Accuracy: 99.382%
- Phishing precision: 99.951%
- Phishing recall: 98.619%
- F1-score: 99.28%

These results are from the PhiUSIIL dataset and do not guarantee the same performance on real-world URLs.

## Feature Extraction
The feature_extraction.py file calculates the 18 URL features used by the model.

## Project Files
- feature_extraction.py — URL feature extraction
- phishguard_model_final.json — final model parameters
- requirements.txt — Python dependencies

## Important
The final JSON was verified against Model A predictions for 1,000 test examples. Browser integration and JavaScript prediction parity still need testing.

The training dataset and virtual environments are intentionally excluded from this repository.
