---
layout: default
title: EECS398 Practical Data Science - University of Michigan
---

<link rel="stylesheet" href="/EECS398-Final-Project/assets/table.css">

# 🎮 **Forecasting Victory: 2024 League of Legends Worlds Matches Predictions**
This data science project explores **2024 League of Legends match data** from [Oracle's Elixir](https://oracleselixir.com/), focusing on **how in-game resources influence victory** and **how side selection (🔵 Blue vs. 🔴 Red) impacts team performance**. Through a combination of statistical analysis and machine learning, the project ultimately builds a predictive model to forecast match outcomes.

## **Table of Contents**
- [Introduction](#introduction)
- [Data Cleaning and Exploratory Data Analysis](#data-cleaning-and-exploratory-data-analysis)
- [Framing a Prediction Problem](#framing-a-prediction-problem)
- [Baseline Model](#baseline-model)
- [Final Model](#final-model)

## **[Introduction](#table-of-contents)**
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
| result | 1 (Win), 0 (Lose) |
| side | red, blue |
| firstblood | Whether the team took the first kill, 1 (Yes), 0 (No) |
| firstdragon | Whether the team took the first dragon, 1 (Yes), 0 (No) |
| firstbaron | Whether the team took the first baron, 1 (Yes), 0 (No) |
| firsttower | Whether the team took the first tower, 1 (Yes), 0 (No) |
| firstmidtower | Whether the team took the first mid-tower, 1 (Yes), 0 (No) |
| firsttothreetowers | Whether the team took the first third-tower, 1 (Yes), 0 (No) |
| gamelength | How long the match lasted in seconds |
| golddiffat(10/15/20) | Gold difference between two teams at 10/15/20 minutes |
| xpdiffat(10/15/20) | XP difference between two teams at 10/15/20 mintues |



## **[Data Cleaning and Exploratory Data Analysis](#table-of-contents)**

### **Data Cleaning**

#### **Extract team data and target columns**
- Selected only the team-level records, excluding individual player-level rows  
- Extracted the key features listed in the previous section

After filtering and selection, the dataset contains:  
- **19596 rows (2 teams × 9798 matches)**  
- **15 columns** (side, result, objectives, and performance features, etc.)  

#### **Check and modify NaN**
Below shows the result of Null value checking. It revealed that at least **2,822** team records contain incomplete data. Since filling in simulated values wouldn’t make sense in a competitive esports context—and the missing data accounts for only **~15%** of the entire dataset—**dropping the rows with NaN values** is a reasonable and efficient solution. After deleting NaN data, the dataset contains:  
- **16774 rows (2 teams × 8387 matches)**
- **15 columns**  

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
| 30-35 | 5522 |
| 25-30 | 5348 |
| 35-40 | 2714 |
| <=25 | 1786 |
| >=40 | 1404 |

<iframe src="assets/gameduration_hist.html" 
        width="120%" 
        height="450" 
        frameborder="0">
</iframe>

#### **Dateset overview**
Below is a preview of the dataset after cleaning  

<div class="table-wrapper" markdown="1">

|    | side   |   firstblood |   firstdragon |   firstbaron |   firsttower |   firstmidtower |   firsttothreetowers |   golddiffat10 |   golddiffat15 |   golddiffat20 |   xpdiffat10 |   xpdiffat15 |   xpdiffat20 | time_label   | win   |
|---:|:-------|-------------:|--------------:|-------------:|-------------:|----------------:|---------------------:|---------------:|---------------:|---------------:|-------------:|-------------:|-------------:|:-------------|:------|
| 30 | Blue   |            0 |             1 |            1 |            1 |               1 |                    1 |           1364 |           2293 |           4248 |          557 |          949 |         2138 | <=25(mins)   | True  |
| 31 | Red    |            1 |             0 |            0 |            0 |               0 |                    0 |          -1364 |          -2293 |          -4248 |         -557 |         -949 |        -2138 | <=25(mins)   | False |
| 32 | Blue   |            0 |             0 |            0 |            0 |               0 |                    0 |            -88 |            -75 |            777 |          625 |         1092 |         2722 | 35-40(mins)  | True  |
| 33 | Red    |            1 |             1 |            1 |            1 |               1 |                    1 |             88 |             75 |           -777 |         -625 |        -1092 |        -2722 | 35-40(mins)  | False |
| 34 | Blue   |            0 |             1 |            1 |            0 |               0 |                    0 |          -2583 |           -561 |          -1528 |        -1718 |          410 |         -722 | 30-35(mins)  | True  |

</div>

### **Univariate Analysis**

🔴 For red side teams, 95% of XP differences range from **-2129 to 1903**, with a median of **-63**.  
🔵 For blue side teams, 95% of XP differences range from **-1903 to 2129**, with a median of **63**.  

These results suggest that **the blue side has a slight advantage in XP gain during the early game**, likely contributing to better early-game momentum.  

The plot below shows the **distribution of XP difference at 10 minutes** for red side team:  

<iframe src="assets/xp10_red.html" 
        width="120%" 
        height="450" 
        frameborder="0">
</iframe>

### **Bivariate Analysis**

#### **Win Rate for each side and firstblood**

The plot below shows the win rates based on **team side** (🔵 blue vs 🔴 red) and **whether the team secured first blood**:  

Teams that secured first blood had a win rate approximately **18.6% higher** than those that did not.  
🔵 Blue side teams showed an average **4.9% higher** win rate compared to 🔴 red side teams.  

These insights highlight the strategic importance of first blood and support the observed advantage of blue side teams.  

<iframe src="assets/win_blood.html" 
        width="120%" 
        height="450" 
        frameborder="0">
</iframe>

#### **Win Rate by Side and First Objective Secured**

From the analysis above, it's clear that the first resource secured (such as first blood, tower, baron) has a significant impact on a team's chance of winning. However, **the strength of this impact varies by objective**.  

The plot below compares win rates for each side (🔵 blue and 🔴 red) based on whether they secured key objectives first. **It ranks these objectives by their positive influence on win rate, in ascending order**.

Key insights:  
- **Securing First Baron or First to Three Towers** shows the **strongest** correlation with winning, for both sides.  
- 🔵 Blue side consistently gains slightly higher win rates from each objective compared to 🔴 red side.

<iframe src="assets/win_side_tower.html" 
        width="1000" 
        height="400" 
        frameborder="0">
</iframe>

#### **Difference in Gold and XP at 10 Minutes Across Game Lengths**
The two plots below illustrate how gold and XP differences at 10 minutes vary across different game duration groups:

- The violin plot shows that **the spread of XP differences narrows as game length increases**. This makes sense — longer matches tend to be more competitive, so the performance gap between the two teams is usually smaller early on.


### **Interesting Aggregates**
Table 1 shows the quantified differences in **win rate**, **first objective secured rate**, and **gold/XP difference** between the two sides (🔵 Blue vs 🔴 Red):

The results illustrate that except **first dragon rate**, **🔵 Blue teams consistently outperform 🔴 Red teams** across all key indicators.  
Blue side teams not only have a **higher win rate**, but also **secure early objectives more often** and maintain a **stronger lead in both gold and XP**.   

<div class="table-wrapper" markdown="1">

| side   |   firstblood |   firstdragon |   firstbaron |   firsttower |   firstmidtower |   firsttothreetowers |   golddiffat10 |   golddiffat15 |   golddiffat20 |   xpdiffat10 |   xpdiffat15 |   xpdiffat20 |      win |
|:-------|-------------:|--------------:|-------------:|-------------:|----------------:|---------------------:|---------------:|---------------:|---------------:|-------------:|-------------:|-------------:|---------:|
| Blue   |     0.516275 |      0.384643 |     0.501967 |     0.548706 |        0.572314 |             0.571837 |        144.923 |        331.158 |        523.683 |      66.8972 |      94.4559 |       95.871 | 0.527483 |
| Red    |     0.483725 |      0.61488  |     0.456421 |     0.451294 |        0.427686 |             0.428163 |       -144.923 |       -331.158 |       -523.683 |     -66.8972 |     -94.4559 |      -95.871 | 0.472517 |

</div>

Table 2 shows the quantified differences in **win rate** between 🔵 Blue and 🔴 Red sides across **different game durations**:

- The results indicate that **🔵 Blue teams consistently outperform 🔴 Red teams at all game lengths**.
- Notably, in **shorter matches (≤ 25 minutes)**, Blue teams win over **60%** of the time — a significant advantage.
- However, in **longer matches (> 25 minutes)**, the win rate difference between the two sides narrows to within **6%**, suggesting the side advantage becomes less impactful as the game progresses.  

<div class="table-wrapper" markdown="1">

| side   |   <=25(mins) |   25-30(mins) |   30-35(mins) |   35-40(mins) |   >=40(mins) |
|:-------|-------------:|--------------:|--------------:|--------------:|-------------:|
| Blue   |     0.601344 |      0.522438 |      0.516117 |      0.511422 |      0.52849 |
| Red    |     0.398656 |      0.477562 |      0.483883 |      0.488578 |      0.47151 |

</div>

### **Imputation**
Imputation is not required in this case, as the cleaned dataset contains no **missing (NaN)** values.

## **[Framing a Prediction Problem](#table-of-contents)**
We aim to predict **whether a team wins or loses a match** based on their **in-game performance features collected by the 20-minute mark**, as analyzed in the sections above.  
- Prediction Type: **Binary Classification**
- Response Variable: **win**(True=Win, False=Lose), the only variable represents the match outcomes, interpretable
- Evaluation Matrics: confusion matrix, accuracy, **ROC curve**, **AUC score**. Unlike accuracy and precision, which depend on a specific classification threshold typically 0.5, **ROC AUC evaluates model performance across all possible thresholds**. This gives a more complete view of the classifier’s ability to separate the two classes. Also, in the dataset, there may be a slight imbalance in match outcomes (blue side winning more often). **ROC AUC is robust to class imbalance**, whereas accuracy may be misleading in such cases.  
- Except for `time_label`, all the used features are known at the time of prediction(before the game end).

## **[Baseline Model](#table-of-contents)**
The baseline model uses **logistic regression** to predict whether a team will win or lose a match, based on early-game features available by the 20-minute mark.  

Based on insights from the exploratory data analysis (EDA), the features `side` and `firstbaron` showed strong influence on match outcomes. Therefore, the baseline model uses these two categorical features along with `xpdiffat10` — a quantitative feature representing early XP advantage — to train and make predictions.  

The table below shows features description:

| Feature   |  Type |   Description |   Method |
|:-------|-------------:|--------------:|--------------:|
| side  |     Nominal |   Team side: Blue or Red |      One-Hot Encoding |
| firstbaron  |    Nominal |     Whether the team took first Baron (0/1)	 |      One-Hot Encoding |
| xpdiffat10  |     Quantative |      XP difference between two teams at 10 min |      Standard Scaler |

The model uses 30% data as test data. One-hot encoding is applied to the nominal features using `OneHotEncoder(drop='first')` to avoid multicollinearity, and StandardScaler() is applied to ensure fair contribution in the logistic regression model.  

Plots below shows model performance:

The first plot shows the confusion matrix with **0.8281** accuracy.  

<img src="assets/base_cm.png" alt="Confusion Matrix" style="width:100%; max-width:600px;">

The second plot shows the ROC curve with **0.88** AUC score.  

<img src="assets/base_roc.png" alt="ROC curve" style="width:100%; max-width:600px;">

The performance of the baseline model isn't perfect, but it is strong given its **simplicity**. The model uses only three features yet achieves an **accuracy of 0.8281** and an **AUC score of 0.88**, indicating that side selection, early objective control, and XP advantage are all strongly correlated with winning.  

However, there is still room for improvement:  
- While 82.81% accuracy and 0.88 AUC socre is promising, it's likely that incorporating more in-game features could further boost performance.  
- Using logistic regression might be too simple in this case. The model oversimplifies the true complexity. Also, logistic regression doesn't consider the correlation between features. For example, team taking the first baron might take a lead in xp or gold. Therefore, exploring more complex models could potentially improve classification performance.  

## **[Final Model](#table-of-contents)**

### **Feature Engineering**  
`firstdragon` and `firstblood`  are included in the model because they capture early-game advantages that strongly correlate with match outcomes as shown in previous [EDA section](#Win-Rate-by-Side-and-First-Objective-Secured). Moreover, following new features are created:  

<div class="table-wrapper" markdown="1">
  
| Feature   |  Input Columns |  What It Captures |  Why It Matters |
|:-------|-------------:|--------------:|--------------:|
| xp_per_min  | `xpdiffat10`, `xpdiffat15`, `xpdiffat20` | XP difference per minute | Considers XP difference at all time periods to reflect leveling (dis)advantage |
| gold_per_min  | `golddiffat10`, `golddiffat15`, `golddiffat20` | Gold difference per minute | Considers Gold difference at all time periods to reflect economic (dis)advantage |
| tower_score  | `firsttower`, `firstmidtower`, `firsttothreetowers` | How many kinds of tower a team firstly taken in total(0-4) |  Measures overall map pressure and early tower control |
| gold_drop_1015  | `golddiffat10`, `golddiffat15` | Gold lead change (10–15 mins) |  Indicates gold economy shift from 10 to 15 mins |
| gold_drop_1520  | `golddiffat15`, `golddiffat20` | Gold lead change (15–20 mins) |  Indicates gold economy shift from 15 to 20 mins |
| xp_drop_1015  | `xpdiffat10`, `xpdiffat15` | XP lead change (10–15 mins) |  Indicates xp advantage shift from 10 to 15 mins |
| xp_drop_1520  | `xpdiffat15`, `xpdiffat20` | XP lead change (15–20 mins) |  Indicates xp advantage shift from 15 to 20 mins |

</div>

In addition to **logistic regression**, we also trained models using **Random Forest** and **Decision Tree classifiers** to explore the impact of non-linear relationships and feature interactions on prediction performance.  

### **Tuning Hyperparameters**
We use **GridSearchCV** to find the optimal tree depth for Random Forest and Decision Tree. Tuning `max_depth` helps control model complexity and reduces the risk of overfitting by limiting how deeply the trees can grow. The train result shows that Random Forest's optimal tree depth is **6**, Decision Tree's optimal tree depth is **5**.  

### **Models Performance**

The **Logistic Regression** model performs **85.08** accuracy and **0.93** AUC. Below shows the confusion matrix and ROC curve:  

<img src="assets/final_logistic_cm.png" alt="Confusion Matrix" style="width:100%; max-width:600px;">

<img src="assets/final_logistic_roc.png" alt="ROC curve" style="width:100%; max-width:600px;">

The **Random Forest** model performs **84.98** accuracy and **0.92** AUC. Below shows the confusion matrix and ROC curve:  

<img src="assets/final_rf_cm.png" alt="Confusion Matrix" style="width:100%; max-width:600px;">

<img src="assets/final_rf_roc.png" alt="ROC curve" style="width:100%; max-width:600px;">

The **Decision Tree** model performs **83.75** accuracy and **0.91** AUC. Below shows the confusion matrix and ROC curve:  

<img src="assets/final_dt_cm.png" alt="Confusion Matrix" style="width:100%; max-width:600px;">

<img src="assets/final_dt_roc.png" alt="ROC curve" style="width:100%; max-width:600px;">

### **Models Comparison**

Below shows the comparison of three model's AUC score. The final Logistic Regression model has the best performance not only on accuracy but also on AUC score. 

<img src="assets/all_models_roc.png" alt="ROC curve" style="width:100%; max-width:600px;">

Logistic Regression model playing better than other two tree models suggests that:  
- The relationship between features and target is mostly linear
- Feature engineering captured key patterns well
- Tree-based models may have overfit

As a result, the **final Logistic Regression model** is selected as the final model since it has the highest accuracy and AUC score while it's also simple and easy to interpret.  

Compared to the base logistic regression model, the final model demonstrates a notable improvement in predictive performance:  
- Accuracy increased from **82.81%** to **85.08%**
- AUC score improved from **0.88** to **0.93**

Overall, **85.08%** accuracy is not perfect for prediction model. But **0.93** AUC indicates excellent performance, with the model having a high ability to distinguish between classes. The final model now is more confident and accurate in ranking match outcomes.  

### **Feature Importance**

In the logistic regression model, after regularizing each variable, the feature coefficients become comparable. A positive coefficient increases the probability of winning, while a negative one decreases it. Moreover, the larger the absolute value of a coefficient, the greater its influence on the predicted probability. Therefore, we interpret feature importance based on the magnitude and sign of these coefficients.  

Below shows each feature's importance for the base model and the final model:  

<img src="assets/base_fi.png" alt="feature importance" style="width:100%; max-width:600px;">

<img src="assets/final_fi.png" alt="feature importance" style="width:100%; max-width:600px;">

**Base Model Feature Importance**:  The base model relies heavily on `firstbaron`, it lacks depth in understanding game trends over time, relying mainly on early or one-time events.  
**Final Model Feature Importance**: The final model benefits from a richer set of engineered features, it captures not only early-game objectives but also trends over time (like XP and gold changes).  

One surprising result from the logistic regression feature importance is the **minimal impact of the side feature on match outcome**. This contrasts with our earlier exploratory data analysis (EDA), where we observed noticeable performance differences between teams starting on the blue vs. red side.  

---

Thanks for reading!  
[⬆️ Back to Top](#)

