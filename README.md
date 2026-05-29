# Federated Prototype Learning with Adaptive Client Selection for Heterogeneous Software Defect Prediction
## Description

Enhanced Federated Prototype Learning (FPLPA) for Heterogeneous Software Defect Prediction with Adaptive Client Selection Strategies, evaluated on NASA, ReLink, and AEEEM datasets using prototype-based federated learning.

## Overview

This project extends the IEEE research paper **"Heterogeneous Defect Prediction Based on Federated Prototype Learning (FPLPA)"** by introducing an adaptive client selection framework for prototype aggregation in federated learning environments.

The objective is to improve software defect prediction across heterogeneous software projects while preserving data privacy and reducing the impact of noisy or low-quality clients during federated aggregation.

Instead of aggregating prototypes from all participating clients, the proposed approach intelligently selects clients based on multiple performance indicators before global prototype aggregation.

---

## Problem Statement

Software defect prediction aims to identify defective software modules using software metrics collected from different software projects.

Traditional centralized approaches require data sharing, which raises privacy concerns and becomes impractical when organizations maintain independent software repositories.

Federated Learning addresses this issue by enabling collaborative model training without sharing raw data. However, existing federated approaches often assume that all clients contribute equally during aggregation.

In heterogeneous environments, clients may differ significantly in:

* Dataset size
* Class distribution
* Data quality
* Feature distributions
* Training performance

Aggregating information from all clients may therefore reduce overall model effectiveness.

---

## Proposed Contribution

This project extends the original FPLPA framework by introducing an adaptive server-side client selection mechanism.

### Key Contributions

* Implementation of Federated Prototype Learning for heterogeneous software defect prediction.
* Introduction of adaptive client selection before prototype aggregation.
* Comparative evaluation of multiple client selection strategies.
* Statistical significance analysis using paired t-tests.
* Evaluation on heterogeneous software defect datasets from NASA, ReLink, and AEEEM.

---

## Methodology

### Data Preprocessing

* Label normalization and encoding
* Missing value handling
* Feature standardization
* Mutual Information based feature selection
* Feature reshaping for CNN input

### Local Model Training

Each client trains a CNN-based Prototype Network (CPN) locally.

The model generates:

* Classification predictions
* Latent feature embeddings

Local prototypes are computed as the average embedding vectors for each class.

### Federated Prototype Learning

1. Clients train local models.
2. Local class prototypes are generated.
3. The server selects participating clients.
4. Selected prototypes are aggregated.
5. Global prototypes are distributed back to clients.
6. Prototype regularization aligns local and global representations.

---

## Client Selection Strategies

### Random Selection

Randomly selects k clients for aggregation.

### FedCS

Selects clients with larger dataset sizes.

### PowerChoice

Prioritizes clients exhibiting higher training loss.

### MultiMetric Selection (Proposed)

The proposed client selection strategy evaluates clients using:

* Performance Improvement (Delta)
* Consistency of Improvement
* Diversity of Selection
* Exploration using Upper Confidence Bound (UCB)

A weighted scoring mechanism ranks clients and selects the top-k candidates for prototype aggregation.

---

## Datasets

### ReLink

* Apache
* Safe
* ZXing

### NASA

* CM1
* MW1
* PC1
* PC3
* PC4

### AEEEM

* AEEEM Dataset

Each dataset is treated as an independent federated client.

---

## Model Architecture

CNN-Based Prototype Network (CPN)

Input Features
→ Convolution Layers
→ Embedding Layer
→ Classification Layer

The embedding layer is used for prototype generation and prototype alignment.

---

## Evaluation Metrics

The following metrics are used to evaluate model performance:

* AUC (Area Under ROC Curve)
* G-Mean
* MCC (Matthews Correlation Coefficient)
* F1 Score
* Balanced Accuracy

---

## Experimental Setup

* Communication Rounds: 20
* Local Epochs: 5
* Multiple Independent Runs: 10
* Subset Sizes Evaluated: 3, 5, 7, and 8 clients
* Prototype Aggregation across selected clients

---

## Results

The proposed adaptive client selection framework was evaluated against multiple baseline selection strategies.

Results indicate that performance-aware client selection can improve the quality of global prototype aggregation and provide more robust learning under heterogeneous software defect datasets.

Statistical significance was further validated using paired t-tests across multiple evaluation metrics.

---

## Future Work

* Dynamic client weighting during aggregation.
* Reinforcement Learning based client selection.
* Trust-aware federated aggregation.
* Attention-based prototype aggregation.
* Evaluation on larger industrial software repositories.

---

## Repository Structure

```text
├── data/
├── notebooks/
├── src/
├── results/
├── figures/
├── README.md
└── requirements.txt
```

---

## Reference

M. B. R. Pandit and N. Varma,
"Heterogeneous Defect Prediction Based on Federated Prototype Learning",
IEEE Access, 2023.

---

## Authors

Chandu Cherupally
Bachelor of Technology (Computer Science and Engineering)

Final Year Research Project
