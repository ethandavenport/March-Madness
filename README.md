# Project Overview
This repository contains code to estimate win probabilities for NCAA men’s tournament games from 2017–2025 and to turn those probabilities into a bracket strategy for upcoming tournaments. The workflow covers data collection, merging and cleaning, feature selection with grouped penalties, model training and evaluation, and bracket construction rules that balance model predictions with historical seeding outcomes.

This project was done purely for my enjoyment and hopeful success in March Madness bracket groups, and is not associated with any class, employer, or other formal commitment.

# Data Collection and Sources
Tournament results and core identifiers come from the official March Machine Learning Mania Kaggle competition (detailed results, seeds, and team metadata). Additional team strength metrics are pulled from several external sources covering efficiency, resume quality, and advanced scouting stats for recent seasons.
​
### Key sources:

- **[Kaggle NCAA tourney data (MM Mania)](https://www.kaggle.com/competitions/march-machine-learning-mania-2025/data)**: play-by-play–level box score stats and tournament seeds
- **[Kaggle CBB dataset](https://www.kaggle.com/datasets/andrewsundberg/college-basketball-dataset)**: regular-season box score–based advanced metrics and adjusted efficiencies
- **​[KenPom / Barttorvik](https://www.kaggle.com/datasets/nishaanamin/march-madness-data?select=KenPom+Barttorvik.csv)**: tempo-free efficiency, tempo, height, experience, talent ratings, and other advanced team-level metrics
- **[Resumes](https://www.kaggle.com/datasets/nishaanamin/march-madness-data?select=Resumes.csv)**: “resume” statistics summarizing quality wins and losses by quadrant and other selection committee-style descriptors
- **​[KPI rankings](https://faktorsports.com/)**: resume-based rating and SOS metrics

# Building the Modeling Dataset
The modeling dataset is constructed at the game level, with each row corresponding to a single NCAA tournament game. The pipeline starts from the detailed tournament results and augments them with seeds and team names, restricted to the main bracket rounds for seasons where external sources are available.

Key steps:
- **Tournament results**
  - Load tournament results and drop unneeded box score columns, keeping scoring, team IDs, and day numbers
  - Map DayNum to human-readable rounds (Play-in, Round 1–6) using a custom function that accounts for the 2021 schedule quirk, double checking round counts per season
- **Seeds and naming**
  - Strip regional prefixes from seeds to create numeric seeds (1–16) and merge them onto winning and losing teams
  - Normalize ordering by renaming the alphabetically first team as ATeam and the other as BTeam, then realign scores, seeds, and IDs so the “A” side is not always the winner
- **Filtering seasons**
  - Restrict to seasons where all metrics are available, ultimately focusing the modeling window on 2017–2025

<p align="center">
  <img width="701" height="182" alt="image" src="https://github.com/user-attachments/assets/44369a3b-451a-43b6-a369-b85ceaa8ac05">
</p>

# Merging External Team Features
For each season, external datasets are reshaped into team-season–level tables and merged twice onto the game dataset: once for ATeam and once for BTeam. This creates a symmetric design where every numeric attribute exists in A and B versions.

​Examples of merged features:
- From **CBB**: offensive and defensive efficiencies, pace, shooting splits, offensive/defensive rebounding, turnover rates, and ranking-based stats
- From **KenPom / Barttorvik**: adjusted offense/defense, tempo, average height, effective height, experience, talent, free-throw rates, and efficiency-based SOS measures
- From **Resumes**: quad-level win/loss counts, quality-win indices, and other committee-style resume indicators
- From **KPI**: KPI rating, SOS value, and SOS ranking for each team-season

Because sources cover different year ranges, each file is filtered to the overlapping seasons and then concatenated. The final merged DataFrame includes 100+ columns per game, with consistent A/B feature pairs that maintain symmetry between teams. The following represents a conceptual mapping of  the final training data set.

<p align="center">
  <img width="701" height="173" alt="image" src="https://github.com/user-attachments/assets/5a456561-c01f-4de5-80be-133e23eb9557">
</p>

# Feature Engineering and Grouped Selection
The main target variable is a binary flag AWon indicating whether ATeam won the game. To ensure fair modeling, the design matrix is built so that each underlying stat appears as a pair: one column for ATeam and one for BTeam.

Feature selection is performed using a **group-lasso–style approach**:
- Start from the curated set of base team features from sources listed above
- For each base feature, create both A and B columns
- Run repeated fits of a logistic regression with L1-type penalty to select stable feature groups, selecting optimal alpha value across resamples
- Enforce symmetry by requiring that **if a stat is selected for A, the corresponding B stat is also included**; this prevents the model from exploiting arbitrary naming of the two teams

<p align="center">
  <img width="659" height="393" alt="image" src="https://github.com/user-attachments/assets/79f7e6f1-5c61-417f-b30b-b1faee9c219e">
</p>

# Model Training and Hyperparameter Tuning

Four complementary models are trained using the selected feature set:

1. **LASSO Logistic Regression**
    - Binary logistic regression with L1 penalty to encourage sparsity with the grouped feature selection
2. **Elastic Net Logistic Regression**
    - Logistic regression with elastic net penalty (convex combination of L1 and L2) implemented via saga solver
    - Cross-validated grid search over values of inverse-regularization strength C and l1-ratio, with a diagnostic plot of validation log loss to select hyperparameters
3. **Gradient Boosting Classifier**
    - Tree-based model fit on the selected features
    - Optuna optimizes `n_estimators`, `learning_rate`, `max_depth`, `min_samples_leaf`, and `subsample` using log loss on held-out validation sets
4. **Neural Network**
    - Small fully connected network built with TensorFlow/Keras
    - Optuna tunes the `number of neurons per layer`, `learning rate`, `L2 regularization`, and `batch size`, using early stopping on validation loss

### Stabilizing Evaluation
Because the dataset is **relatively small** (roughly a few hundred games across tournaments), a single train/test split can give noisy estimates of performance. To stabilize evaluation:
- Each model is trained and evaluated across many random splits (i.e., 20–100 repetitions) with stratification on the outcome variable
- For each model, the mean and standard deviation of test log loss are reported, alongside training log loss

<p align="center">
  <img width="527" height="527" alt="image" src="https://github.com/user-attachments/assets/d1735c42-a2da-4ee8-8a21-af66499aa9af">
</p>

# Seed-Based Baseline and Historical Upsets
Before trusting ML to drive bracket picks, the project establishes a **seeding-only baseline**:
- Fit a simple logistic model using only the log ratio of seeds, log(BSeed/ASeed), to predict ATeam win probability
- Use cross-validation to **estimate typical win probabilities for every 1–16 vs. 1–16 pairing** and visualize them as a 16×16 probability matrix
- This serves as a benchmark for how much predictive power is available without any advanced metrics

Historical upset rates by round and seed matchup (i.e., 12-over-5, 13-over-4) are computed from the same historical window, providing a reference for how aggressive ML-driven upsets should be. The idea is to avoid a bracket that is out-of-line with historical frequencies.

<p align="center">
  <img width="460" height="393" alt="image" src="https://github.com/user-attachments/assets/8d0dd33f-2109-47d1-aa1f-5130e1da1974">
</p>

# Bracket Construction Strategy
The bracket logic revolves around a **lift score** that compares two perspectives on each game:
- The log‑odds from the full machine‑learning model
- The log‑odds implied by the simple, seed‑based baseline

When this lift is **negative**, it signals that the model thinks the underdog is more dangerous than the seeding alone would suggest, flagging a potential upset. For each early round, historical data is used to translate typical upset rates into a round‑specific lift cutoff:
- Historical games are ranked by lift (from most underdog‑friendly to most favorite‑friendly)
- The cutoff is chosen so that the fraction of games below that cutoff matches how often underdogs have actually won in that round
- In a future tournament, whenever a matchup’s lift for the worse seed is below the relevant cutoff, the bracket intentionally picks the upset, even if the favorite still has the higher raw win probability

This procedure is repeated separately for every round through the Elite Eight, giving each stage its own lift threshold. Early rounds tend to allow more upset picks, while later rounds are more conservative, reflecting how rarely big surprises occur deep in the tournament. Because decisions are driven by these numeric thresholds rather than a fixed quota of surprises:
- The **number of predicted upsets is allowed to vary** from year to year
- Over many tournaments, the average upset rate naturally lines up with history, but **any given bracket can lean more chaotic or more chalky** depending on how strongly the model disagrees with the seed baseline.

In the Final Four and title game, the bracket simply takes the team with the higher modeled win probability, avoiding extra thresholding when comparable historical data is limited and matchups are few.
​



​
