# Food Spoilage Risk Prediction Using Machine Learning

## Overview
An educational machine-learning project that predicts recorded spoilage values from experimental treatment and storage factors.

## Dataset
- Source: Minitab Food Spoilage dataset
- Experimental runs: 16
- Target: `Spoilage`
- Features: `Preservative`, `VacuumPress`, `ContaminationLevel`, `CoolTemp`

## Preliminary Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Mean Baseline | 8.350 | 10.090 | -0.138 |
| Random Forest | 4.445 | 5.776 | 0.627 |

Evaluation used Leave-One-Out Cross-Validation on 16 observations. Results are preliminary and do not establish real-world generalization.

## Project Structure
- `data/food_spoilage.csv`: dataset
- `src/train_model.py`: training and evaluation script
- `notebooks/food_spoilage_experiment.ipynb`: experiment notebook
- `results/baseline_metrics.json`: model metrics
- `results/predictions.csv`: cross-validated predictions
- `results/actual_vs_predicted.png`: prediction plot

## Limitations
The dataset is small and contains experimental runs rather than timestamped sensor histories for individual food items. The current model is not a validated food-safety classifier. Advanced models, dashboard features, and cloud integration remain future work.

## Data Attribution
Source: https://support.minitab.com/en-us/datasets/doe-data-sets/food-spoilage-data/

Check redistribution permissions before publishing the dataset.

## Disclaimer
This is an educational prototype, not a certified food-safety system.
