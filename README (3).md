# 🚗 carPricePredictor

**An end-to-end machine learning pipeline for predicting car prices — built with DVC, MLflow, XGBoost, FastAPI, and Docker.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)]()
[![DVC](https://img.shields.io/badge/DVC-Pipeline-945DD6.svg)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-served-009688.svg)]()

---

## Overview

`carPricePredictor` is a reproducible ML system that predicts the price of a car from its attributes. It's built less like a notebook experiment and more like production infrastructure — every stage of the pipeline (ingestion, preprocessing, training, evaluation, and registration) is versioned, tracked, and reproducible with a single command.

The trained model is served through a **FastAPI** application, containerized with **Docker**, with data and model artifacts versioned via **DVC** on **AWS S3**, and experiments logged with **MLflow**.

## Features

- 📦 **Reproducible pipeline** — every stage defined in `dvc.yaml`, re-runnable with `dvc repro`
- 🌲 **XGBoost regression model** for price prediction
- 📊 **Experiment tracking** with MLflow (metrics, params, model registry)
- ☁️ **Remote data/model versioning** via DVC + AWS S3
- ⚡ **FastAPI inference service** for real-time predictions
- 🐳 **Dockerized** for consistent deployment
- 🔁 **CI workflow** via GitHub Actions

## Pipeline Architecture

```
Raw Data → Data Ingestion → Data Preprocessing → Model Building (XGBoost)
                                                        │
                                                        ▼
                                          Model Evaluation → Model Registration (MLflow)
```

| Stage | Script | Output |
|---|---|---|
| Data Ingestion | `src/data/data_ingestion.py` | `data/raw/` |
| Data Preprocessing | `src/data/data_preprocessing.py` | `data/interim/` |
| Model Building | `src/model/model_building.py` | `xgb_model.pkl` |
| Model Evaluation | `src/model/model_evaluation.py` | `experiment_info.json` |
| Model Registration | `src/model/register_model.py` | Registered model in MLflow |

## Project Structure

```
carPricePredictor/
├── .dvc/                  # DVC internal config
├── .github/workflows/     # CI/CD pipelines
├── Notebooks/             # EDA and experimentation notebooks
├── app/                   # FastAPI inference service
├── data/                  # Raw & processed data (DVC-tracked)
├── src/                   # Pipeline source code
├── Dockerfile             # Container definition
├── dvc.yaml               # DVC pipeline stages
├── params.yaml            # Pipeline hyperparameters
├── requirements.txt       # Python dependencies
└── setup.py               # Package config
```

## Getting Started

### Prerequisites
- Python 3.11
- Conda (recommended) or venv
- AWS credentials (for DVC remote storage)
- Docker (optional, for containerized deployment)

### 1. Clone the repository

```bash
git clone https://github.com/jwsadman/carPricePredictor.git
cd carPricePredictor
```

### 2. Set up the environment

```bash
conda create -n price python=3.11
conda activate price
pip install -r requirements.txt
```

### 3. Configure AWS access

DVC uses an S3 remote to store versioned data and models.

```bash
aws configure
```

### 4. Reproduce the pipeline

```bash
dvc init
dvc repro
```

This runs every stage — ingestion, preprocessing, training, evaluation, and registration — end to end.

### 5. Run the API

```bash
uvicorn app.main:app --reload
```

Or with Docker:

```bash
docker build -t car-price-predictor .
docker run -p 8000:8000 car-price-predictor
```

## Tech Stack

| Category | Tools |
|---|---|
| Modeling | XGBoost, scikit-learn |
| Data & Pipeline Versioning | DVC, DVC-S3 |
| Experiment Tracking | MLflow |
| Serving | FastAPI, Uvicorn |
| Storage | AWS S3, boto3 |
| Data Handling | pandas, NumPy |
| Visualization | matplotlib, seaborn |
| Deployment | Docker |

## Roadmap

- [ ] Add automated model evaluation gating in CI
- [ ] Expand test coverage
- [ ] Add a lightweight frontend for interactive predictions
- [ ] Publish inference API docs

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Author

**jwsadman** — [GitHub](https://github.com/jwsadman)
