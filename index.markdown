---
layout: default
title: EECS398 Practical Data Science - University of Michigan
---

# 🎮 **Forecasting Victory: 2024 League of Legends Worlds Matches Predictions**
This data science project explores **2024 League of Legends match data** from [Oracle's Elixir](https://oracleselixir.com/), focusing on **how in-game resources influence victory** and **how side selection (🔵 Blue vs. 🔴 Red) impacts team performance**. Through a combination of statistical analysis and machine learning, the project ultimately builds a predictive model to forecast match outcomes.

## **Introduction**
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

To better understand **side selection privilege**, this project analyzes match data to explore the question: **How does side selection (🔵 Blue vs. 🔴 Red) impact team performance?**  

Below lists the used features and their description:  

| Features | Description |
|----------|----------|
| result | 1(Win), 0(Lose) |
| league | Which league helds this match |
| side | red, blue |
| firstblood | Whether the team took the first kill, 1(Yes), 0(No) |
| firstdragon | Whether the team took the first dragon, 1(Yes), 0(No) |
| firstbaron | Whether the team took the first baron, 1(Yes), 0(No) |
| firsttower | Whether the team took the first tower, 1(Yes), 0(No) |
| firstmidtower | Whether the team took the first mid-tower, 1(Yes), 0(No) |
| firsttothreetowers | Whether the team took the first third-tower, 1(Yes), 0(No) |
| gamelength | How long the match lasted in seconds |
| golddiffat(10/15/20) | Gold difference between two teams at 10/15/20 minutes |
| xpdiffat(10/15/20) | XP difference between two teams at 10/15/20 mintues |


## **Data Cleaning and Exploratory Data Analysis**

### **Data Cleaning**

#### **Extract team data and target columns**
- Selected only the team-level records, excluding individual player-level rows  
- Extracted the key features listed in the previous section

After filtering and selection, the dataset contains:  
- **19596 rows (2 teams × 9798 matches)**  
- **16 columns** (side, result, objectives, and performance features, etc.)  

#### **Check and modify NaN**
Below shows the result of Null value checking. It revealed that at least **2,822** team records contain incomplete data. Since filling in simulated values wouldn’t make sense in a competitive esports context—and the missing data accounts for only **~15%** of the entire dataset—**dropping the rows with NaN values** is a reasonable and efficient solution. After deleting NaN data, the dataset contains:  
- **16774 rows (2 teams × 8387 matches)**
- **16 columns**  

Losing only **~15%** data is acceptable  
<p align="center">
  <img src="image/NaN Checking.png" alt="NaN Checking" width="200"/>
</p>

#### **Categorize `Gamelength`**
The `gamelength` ranges from 1143 to 3482 seconds. Below shows the distribution of `gamelength`:  
<iframe src="/_site/assets/gamelength_hist.html" 
        width="100%" 
        height="600" 
        frameborder="0">
</iframe>

#### Recategorize result as win

### Univariate Analysis


## 📊 Model Comparison

Here we show the accuracy and confusion matrix of the baseline vs final model...

## 🎯 Feature Engineering

- `StandardScaler` on gold diff
- `QuantileTransformer` on XP diff

## 🖼️ Visualizations

![Confusion Matrix](/assets/img/conf_matrix.png)

## ✅ Conclusion

The final model outperformed the baseline in both accuracy and stability.
