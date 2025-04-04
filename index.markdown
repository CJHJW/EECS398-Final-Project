---
layout: default
title: EECS398 Practical Data Science - University of Michigan
---

# 🎮 Forecasting Victory: 2024 League of Legends Worlds Matches Predictions
This data science project explores **2024 League of Legends match data** from [Oracle's Elixir](https://oracleselixir.com/), focusing on **how in-game resources influence victory** and **how side selection (🔵 Blue vs. 🔴 Red) impacts team performance**. Through a combination of statistical analysis and machine learning, the project ultimately builds a predictive model to forecast match outcomes.

## Introduction
The raw data from [Oracle's Elixir](https://oracleselixir.com/) contains **117,576 records (rows)** and **161 features (columns)**.  

Each **12 consecutive records** correspond to one match:
- The **first 5 records** represent players data from the blue side.
- The **next 5 records** represent players data from the red side.
- The **final 2 records** provide team-level overviews for both sides.

Therefore, the dataset covers a total of **9,798 matches**.

The **161 features** can be categorized into **three main groups**:  
- **Team/Player Information**:  
Includes identifiers such as player names, team names, league affiliations, and match timestamps.  
- **Match Overview**:  
Contains high-level game details like match outcome (victory/loss), side selection (🔵blue/🔴red), champion picks and bans, total game duration.  
- **In-Game Performance Metrics**:  
Captures gameplay stats such as kills, deaths, assists, objective control (dragons🐲, barons😈, towers🗼), and team differences in XP and gold across different time intervals.

The Blue🔵 side refers to the team located on the **bottom left of the map** and always **gets first pick in the draft**. The Red🔴 side is positioned on the **top right corner**. In competitive play, the team with Side Selection Privilege chooses the side for Game 1, and then the losing team picks the side for the next game. (Source: [LOL Worlds 2024 Fantasy - E-Go App](https://leagueoflegends.fandom.com/wiki/Draft_Pick#:~:text=The%20Blue%20team%20always%20has,second%20team%20picks%20two%20champions))

This makes side selection a strategic tool — a subtle but important factor that can influence match outcomes. Surprisingly, this goes against the common belief that both sides should be equally fair in terms of gameplay.

In practice, **Blue Side teams consistently perform better**. One contributing factor is the **camera perspective advantage**: although both sides appear symmetrical, the Blue side benefits from a slight downward tilt in the in-game camera. This offers a clearer view of flanks, jungle movements, and overall map activity — making it easier to react and make informed decisions. (Source: [Is red stronger than blue in League of Legends? - Eloking](https://eloking.com/blog/is-red-stronger-than-blue-in-lol#:~:text=The%20Blue%20Side%20team%20has,movements%20from%20the%20Red%20Side.))

In the analysis, the project uses match data to explore the question: **How does side selection (🔵 Blue vs. 🔴 Red) impact team performance?**



## Contents

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
