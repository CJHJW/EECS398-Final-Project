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
<iframe src="assets/gamelength_hist.html" 
        width="120%" 
        height="450" 
        frameborder="0">
</iframe>

Instead of focusing on specific game lengths in seconds, our analysis is more concerned with the **relationship between general time periods (in minutes)** and other features. Therefore, the gamelength column needs to be categorized into **time periods**, and drop the original `gamelength`.

Below are the results after categorizing

| Time Period | Count |
|----------|----------|
| 30-35(mins) | 5522 |
| 25-30(mins) | 5348 |
| 35-40(mins) | 2714 |
| <=25(mins) | 1786 |
| >=40(mins) | 1404 |

<iframe src="assets/gameduration_hist.html" 
        width="120%" 
        height="450" 
        frameborder="0">
</iframe>

#### **Recategorize `result` as Boolean (Win: True / Lose: False)**
To improve readability and make the data more intuitive, `result` — originally encoded as 1 for a win and 0 for a loss — is recategorized into a Boolean type:  
- 1 → True (Win)
- 0 → False (Lose)

This transformation allows easier logical filtering and improves clarity in visualizations and model interpretation.

#### **Dateset overview**
Below is a preview of the dataset after cleaning  
Now, the dataset contains **16774 rows** and **16 columns**  

<div style="overflow-x: auto;">
| | result | league | side | firstblood | firstdragon | firstbaron | firsttower | firstmidtower | firsttothreetowers | golddiffat10 | golddiffat15 | golddiffat20 | xpdiffat10 | xpdiffat15 | xpdiffat20 | time_label |
|---:|---------:|:---------|:-------|-------------:|--------------:|-------------:|-------------:|----------------:|---------------------:|---------------:|---------------:|---------------:|-------------:|-------------:|-------------:|:-------------|
| 30 |        1 | TSC      | Blue   |            0 |             1 |            1 |            1 |               1 |                    1 |           1364 |           2293 |           4248 |          557 |          949 |         2138 | <=25(mins)   |
| 31 |        0 | TSC      | Red    |            1 |             0 |            0 |            0 |               0 |                    0 |          -1364 |          -2293 |          -4248 |         -557 |         -949 |        -2138 | <=25(mins)   |
| 32 |        1 | TSC      | Blue   |            0 |             0 |            0 |            0 |               0 |                    0 |            -88 |            -75 |            777 |          625 |         1092 |         2722 | 35-40(mins)  |
| 33 |        0 | TSC      | Red    |            1 |             1 |            1 |            1 |               1 |                    1 |             88 |             75 |           -777 |         -625 |        -1092 |        -2722 | 35-40(mins)  |
| 34 |        1 | TSC      | Blue   |            0 |             1 |            1 |            0 |               0 |                    0 |          -2583 |           -561 |          -1528 |        -1718 |          410 |         -722 | 30-35(mins)  |
</div>

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
