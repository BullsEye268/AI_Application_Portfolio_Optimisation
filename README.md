# Portfolio Optimization Model Training

This directory contains code for training an optimal portfolio allocation model using historical stock data. The model optimizes portfolio weights to maximize the Sharpe ratio.

## Features

- Data versioning with DVC
- Modern portfolio theory implementation
- Bayesian optimization with MLflow and hyperopt
- Experiment tracking with MLflow
- MongoDB integration for storing models

## Requirements

Install the required packages:

```
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file with the following variables:

```
MONGO_URI=mongodb://username:password@host:port/database
```

## DVC Setup

Initialize DVC for the first time:

```
dvc init
```

To set up a remote storage (optional):

```
dvc remote add -d myremote s3://mybucket/path
```

## Usage

Run the training pipeline:

```
dvc repro
```

This will:

1. Fetch stock data for the predefined symbols
2. Train the portfolio optimization model using Bayesian optimization
3. Track experiments with MLflow
4. Save the model and metrics
5. Track all artifacts with DVC

## Model Outputs

- **Model weights**: Stored in `models/portfolio_model.joblib`
- **Performance metrics**: Tracked in `results/metrics.json`
- **Visualization**: Portfolio allocation chart in `results/portfolio_allocation.png`
- **MLflow artifacts**: Stored in `mlruns` directory

## Tracking Experiments

View metrics from your DVC experiments:

```
dvc metrics show
```

Compare different DVC runs:

```
dvc metrics diff
```

View DVC plots:

```
dvc plots show
```

## MLflow Experiment Tracking

You can view detailed experiment information using the MLflow UI:

```
mlflow ui
```

This will start a web server at http://localhost:5000 where you can:

- View all runs and experiments
- Compare parameter values and metrics across runs
- Examine the learning curves
- View and download artifacts
