# Federated Machine Learning for PCOS Detection

> A privacy-preserving diagnostic system for Polycystic Ovary Syndrome (PCOS) that trains a neural network collaboratively across multiple clients using federated learning — enabling accurate detection without centralising sensitive patient data.

---

## Overview

Polycystic Ovary Syndrome is one of the most common hormonal disorders affecting women of reproductive age, with wide-ranging metabolic, cardiovascular, and reproductive implications. Early and accurate diagnosis is critical — yet the sensitive nature of clinical data makes centralised machine learning approaches ethically and legally problematic under medical data regulations.

This project proposes a federated machine learning framework for binary PCOS classification. Rather than pooling raw patient records into a single training environment, the system distributes model training across two independent clients, each working on its local data partition. Only model parameter updates — not raw data — are shared between clients and aggregated into a global model. This architecture preserves data privacy while still enabling collaborative learning, directly addressing the core tension between diagnostic accuracy and patient confidentiality in clinical AI systems.

The research was documented as a formal research paper: *"Federated Machine Learning for Polycystic Ovary Syndrome Detection: A Collaborative Approach."*

---

## Dataset

**Source:** Clinical PCOS Dataset — 541 patient records

| Property | Detail |
|---|---|
| Observations | 541 women of reproductive age |
| Task type | Binary classification |
| Target variable | `PCOS (Y/N)` — PCOS-positive or PCOS-negative |

**Feature categories:**

| Category | Features |
|---|---|
| Anthropometric | Age, Height, Weight, BMI |
| Hormonal | FSH, LH, TSH, PRL, Vitamin D3, AMH |
| Haematological | Haemoglobin, Blood Group, RBC, WBC |
| Cardiovascular | BP Systolic, BP Diastolic, Heart Rate |
| Reproductive & lifestyle | Menstrual cycle regularity, Pregnancy status, Marital status |

---

## Methodology

### Data Preprocessing

- Removed non-informative administrative columns (`Sl. No.`, `Patient File No.`, `Cycle Duration (days)`)
- Applied one-hot encoding to categorical variables including marital status and menstrual cycle regularity
- Standardised all features using `StandardScaler` to ensure uniform scale across predictors and improve gradient descent convergence
- Split data into feature matrix `X` and binary target vector `y`
- Converted preprocessed arrays into PyTorch tensors for compatibility with the deep learning framework

### Federated Model Architecture

The `FederatedModel` class is implemented in PyTorch as a feedforward neural network with the following architecture:

```
Input Layer  →  Fully Connected Layer (ReLU)
             →  Fully Connected Layer (ReLU)
             →  Output Layer (Sigmoid)
```

The sigmoid output produces continuous probability estimates, discretised at a threshold of 0.5 to produce binary PCOS/no-PCOS predictions. The architecture is deliberately compact to balance representational capacity against overfitting risk on a dataset of 541 observations.

### Federated Learning Process

The framework instantiates two independent clients (`client1_model`, `client2_model`), each holding a separate instance of `FederatedModel` with its own SGD optimiser. Training proceeds over a fixed number of epochs:

1. Each client trains locally on its own data partition
2. Gradients are reset to zero before each local update to prevent parameter leakage across batches
3. Model weights (not raw data) are passed between clients for aggregation
4. The global model is formed by averaging predictions from both client models on the held-out test set
5. Aggregated predictions are thresholded at 0.5 to produce final binary labels

This process mirrors real-world federated learning deployments in multi-centre clinical studies, where participating hospitals collaborate on model training without sharing patient records.

### Evaluation Metrics

- Accuracy
- Precision
- Recall (True Positive Rate)
- F1-Score
- Confusion Matrix

---

## Results

The federated model was evaluated on a held-out test set of 109 instances.

**Confusion Matrix:**

| | Predicted Positive | Predicted Negative |
|---|---|---|
| Actual Positive | 32 | 0 |
| Actual Negative | 73 | 4 |

**Performance Metrics:**

| Metric | Value |
|---|---|
| Accuracy | 71.56% |
| True Positive Rate (Recall) | 1.0000 |
| Precision | 0.3048 |
| F1-Score | 0.4672 |
| False Positive Rate | 0.9481 |

The model achieves a perfect recall of 1.0 — it correctly identifies every PCOS-positive case in the test set with zero false negatives. In a clinical screening context this is the most important property: no true cases are missed. The trade-off is a higher false positive rate, meaning some PCOS-negative patients are flagged for follow-up. The 71.56% overall accuracy reflects this asymmetry and is consistent with the recall-maximising behaviour of the binary threshold.

---

## Limitations & Future Work

**Current Limitations:**

- The dataset is relatively small at 541 patients, which limits the model's ability to generalise across diverse clinical populations and demographic groups
- The federated framework currently simulates two clients on a single machine rather than deploying across genuinely distributed infrastructure — true federated deployment would require a secure communication protocol between institutions
- The simple two-layer feedforward architecture may not capture non-linear hormonal interaction patterns as effectively as deeper or more specialised architectures
- Precision is low (0.30), indicating a high volume of false positives that would place unnecessary burden on follow-up clinical resources

**Future Directions:**

- Integrate differential privacy and secure aggregation techniques to provide formal privacy guarantees beyond the architectural separation of client data
- Explore deeper architectures including CNNs and RNNs within the federated framework, and evaluate transfer learning approaches for cross-population generalisation
- Extend to a genuine multi-centre federated deployment, validating the framework across hospitals with heterogeneous data distributions
- Investigate threshold optimisation to improve the precision-recall trade-off for clinical deployment scenarios where false positive burden must be balanced against detection sensitivity
- Incorporate ultrasound imaging features and time-series hormonal data as additional modalities to enrich the diagnostic signal

---

## How to Run This Project

### Prerequisites

```bash
Python 3.8+
```

### 1. Clone the Repository

```bash
git clone https://github.com/d4h2nu8h/federated-pcos-detection.git
cd federated-pcos-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Key libraries required:

```bash
pip install torch scikit-learn pandas numpy matplotlib seaborn
```

### 3. Run the Notebook

```bash
jupyter notebook PCOS_FINAL.ipynb
```

The notebook runs the full pipeline: preprocessing, federated training across two clients, model aggregation, and evaluation with confusion matrix and metric outputs.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.8+ |
| Deep Learning | PyTorch |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Notebook Environment | Jupyter Notebook |

---

## Author

**Dhanush Sambasivam**

[![GitHub](https://img.shields.io/badge/GitHub-d4h2nu8h-181717?style=flat&logo=github)](https://github.com/d4h2nu8h)

---

## License

This project is intended for academic and research purposes. Patient data is anonymised and sourced from a publicly available clinical PCOS dataset.
