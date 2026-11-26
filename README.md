# Edge AI-Based Financial Fraud Detection System

An Edge AI-based financial transaction fraud detection and risk assessment prototype deployed on the **NVIDIA Jetson Orin Nano**.

The project combines a Random Forest classifier, a Python-based risk policy, and a locally running Gemma 3 4B language model to provide fraud scores, risk assessments, and human-readable explanations.

## Project Overview

Financial fraud detection involves balancing two important objectives: identifying more fraudulent transactions and minimizing false alarms.

This project explores that trade-off using the PaySim synthetic mobile-money dataset and a locally deployed Edge AI application.

The system accepts transaction details through a Streamlit dashboard, processes them using FastAPI, generates fraud scores through Random Forest, assigns risk levels using Python rules, and generates explanations using Gemma 3 4B through Ollama.

**Note:** This is an academic prototype, not a production banking or payment authorization system.

## Key Features

- Random Forest-based fraud classification.
- Threshold-based fraud detection.
- Python-based risk assessment.
- Local LLM explanations using Gemma 3 4B.
- Streamlit dashboard.
- FastAPI backend.
- SQLite transaction history.
- Docker-based deployment.
- NVIDIA Jetson Orin Nano edge computing.

## System Architecture

```text
Transaction Input
       |
       v
Streamlit Dashboard
       |
       v
     FastAPI
       |
       v
Feature Engineering
       |
       v
Random Forest Model
       |
       v
Fraud Score
       |
       v
V5 Classification Threshold
       |
       v
Python Risk Policy
       |
       +------------------+
       |                  |
       v                  v
   Gemma 3 4B           SQLite
   Explanation          History
       |
       v
Dashboard Results
```

Gemma generates explanations and suggested next steps. It does not override the Random Forest classification or Python risk policy.

## Technology Stack

| Component | Technology |
|---|---|
| Edge Hardware | NVIDIA Jetson Orin Nano |
| Programming Language | Python |
| Machine Learning | Scikit-learn, Random Forest |
| Backend | FastAPI |
| Frontend | Streamlit |
| LLM Runtime | Ollama |
| Language Model | Gemma 3 4B |
| Database | SQLite |
| Containerization | Docker |
| Dataset | PaySim |

## Dataset

We use the PaySim synthetic mobile-money transaction dataset.

- Total transactions: 6,362,620
- Legitimate transactions: 6,354,407
- Fraudulent transactions: 8,213
- Fraud prevalence: approximately 0.129%

### Dataset Splitting

| Split | Total | Fraud | Legitimate |
|---|---:|---:|---:|
| Training | 4,463,587 | 3,643 | 4,459,944 |
| Validation | 943,289 | 560 | 942,729 |
| Testing | 955,744 | 4,010 | 951,734 |

The dataset was divided chronologically to simulate training on earlier transactions and evaluation on later transactions.

### Training Strategy

For V4 training:

- All 3,643 fraudulent training samples were retained.
- 250,000 legitimate samples were randomly selected.
- Total training samples: 253,643.

### Model Features

The Random Forest model uses five features:

1. Transaction amount.
2. Log-transformed amount.
3. Transaction type.
4. Transaction hour.
5. Recipient category.

The transaction hour is derived using `(step - 1) % 24`, representing simulated time.

## Random Forest Configuration

| Parameter | Value |
|---|---:|
| Number of trees | 80 |
| Maximum depth | 14 |
| Minimum samples per leaf | 5 |
| Class weighting | balanced_subsample |
| CPU workers | 4 |
| Random state | 42 |

## Model Evaluation: V4 vs V5

Both versions were evaluated on the same held-out test dataset.

| Metric | V4 | V5 |
|---|---:|---:|
| Precision | 72.58% | 52.69% |
| Recall | 22.44% | 36.18% |
| F1-score | 0.3429 | 0.4290 |
| True Positives | 900 | 1,451 |
| False Positives | 340 | 1,303 |
| False Negatives | 3,110 | 2,559 |
| True Negatives | 951,394 | 950,431 |

V4 test ROC-AUC: **0.9407**

V4 test PR-AUC: **0.3840**

V5 ROC-AUC and PR-AUC were not recorded in the available experiment results.

### Threshold Optimization

V4 threshold: `0.990471`

Selected by maximizing validation F1-score.

V5 threshold: `0.960688`

Selected under a validation false-positive-rate constraint of 0.10%.

V5 reuses the V4-trained Random Forest. It is not a newly trained model. The change involves the classification threshold and associated risk-assessment workflow.

On the test set, V5 detected 551 additional fraudulent transactions compared with V4, but also produced more false positives.

This demonstrates the trade-off between fraud detection and false alarms.

## Risk Assessment Policy

The system uses a separate Python-based policy:

| Condition | Risk | Recommended Action |
|---|---|---|
| Prediction is FRAUD | FRAUD | FLAG |
| Score >= 0.80 | HIGH | REVIEW |
| Score >= 0.50 | MEDIUM | REVIEW |
| Score < 0.50 | LOW | ALLOW |

A transaction may be classified as legitimate but still receive a MEDIUM or HIGH risk assessment.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Application root |
| GET | `/health` | Health status |
| POST | `/predict` | Transaction fraud assessment |
| GET | `/history` | Transaction history |

## Local LLM Integration

The project integrates Gemma 3 4B using Ollama.

Its responsibilities include:

- Explaining transaction assessments.
- Generating understandable risk summaries.
- Suggesting additional verification steps.

The Random Forest remains responsible for fraud classification, and the Python policy determines the risk level and recommended action.

## Edge Deployment

The application is deployed locally on NVIDIA Jetson Orin Nano using Docker.

V5 deployment details:

- Docker image: `edge-fraud-scanner:v5`
- Container: `edge-fraud-v5`
- API port: `8001`
- Dashboard port: `8502`

The current Random Forest implementation uses CPU parallelism. GPU acceleration for Random Forest inference is not claimed.

## Limitations

- PaySim is synthetic and may not represent all real banking fraud patterns.
- The model currently uses five transaction features.
- V5 fraud recall is 36.18%, so further improvement is necessary.
- V5 generates more false positives than V4.
- Fraud scores are not established as calibrated real-world probabilities.
- Further security and real-world validation would be required before production use.

## Future Scope

- Increase training data and compare model performance.
- Introduce customer behavioural features.
- Explore additional transaction datasets.
- Improve fraud recall while controlling false positives.
- Investigate model calibration.
- Optimize inference for edge hardware.
- Perform additional deployment and security testing.

## Author

**Srushti Kulkarni**

B.Tech Computer Science Engineering  
Specialization: Artificial Intelligence and Edge Computing

## Disclaimer

This project is developed for academic and demonstration purposes. It should not be used for real financial decisions without extensive validation, security review, and appropriate human oversight.
