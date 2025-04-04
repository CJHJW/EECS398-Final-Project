---
layout: default
title: EECS398 Practical Data Science - University of Michigan
---

# 🎮 Forecasting Victory: 2024 League of Legends Worlds Matches Predictions
This data science project explores 2024 League of Legends match data from [Oracle's Elixir](https://oracleselixir.com/), focusing on how in-game resources influence victory and how side selection (🔵 Blue vs. 🔴 Red) impacts team performance. Through a combination of statistical analysis and machine learning, the project ultimately builds a predictive model to forecast match outcomes.


## Introduction

- Predicting win rates in League of Legends Competition
- Baseline model: Logistic Regression
- Final model: Random Forest with hyperparameter tuning

## 🔍 Contents

- [Introduction](#1.introduction)
- [Data Cleaning and Exploratory Data Analysis](#2.data_cleaning_and_analysis)
- [Framing a Prediction Problem](#3.problem_framing)
- [Baseline Model](#4.baseline_model)
- [Final Model](#5.final_model)

## 📊 Model Comparison

Here we show the accuracy and confusion matrix of the baseline vs final model...

## 🎯 Feature Engineering

- `StandardScaler` on gold diff
- `QuantileTransformer` on XP diff

## 🖼️ Visualizations

![Confusion Matrix](/assets/img/conf_matrix.png)

## ✅ Conclusion

The final model outperformed the baseline in both accuracy and stability.
