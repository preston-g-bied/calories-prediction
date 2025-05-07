# Calories Burnt Prediction - Kaggle Competition

## Overview
This repository contains the solution for the 2025 Kaggle Playground Series competition on predicting calories burned during workouts. The goal is to predict the continuous target "Calories" based on various biological and workout-related features.

## Evaluation Metric
Root Mean Squared Logarithmic Error (RMSLE)

## Data
- **Competition Dataset**: Provided by Kaggle (train.csv, test.csv)
- **Original Dataset**: The original "Calories Burnt Prediction" dataset which can be used as supplementary training data

## Features (from original dataset)
- User_Id
- Gender
- Age
- Height
- Weight
- Duration
- Heart_rate
- Body_temp

## Project Structure
- `data/`: Raw and processed datasets
- `notebooks/`: Jupyter notebooks for exploratory analysis and modeling
- `src/`: Source code for the project
- `models/`: Saved model files
- `submissions/`: Competition submission files
- `configs/`: Configuration files for models and features
- `logs/`: Training logs

## Getting Started
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download the competition data and place it in the `data/raw/` directory
4. Run the notebooks in the `notebooks/` directory in sequence

## Solution Approach
1. **Exploratory Data Analysis**: Understand the data, distributions, and relationships
2. **Feature Engineering**: Create new features and transform existing ones
3. **Model Development**: Train various models and optimize hyperparameters
4. **Ensembling**: Combine predictions from different models
5. **Submission**: Generate final predictions

## Models
- Baseline: Linear Regression, Random Forest
- Advanced: Gradient Boosting (XGBoost, LightGBM, CatBoost)
- Deep Learning: Neural Networks (if applicable)

## Requirements
See `requirements.txt` for a list of dependencies.