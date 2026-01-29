# Project Overview
This repository contains code to estimate win probabilities for NCAA men’s tournament games from 2017–2025 and to turn those probabilities into a bracket strategy for upcoming tournaments. The workflow covers data collection, merging and cleaning, feature selection with grouped penalties, model training and evaluation, and bracket construction rules that balance model predictions with historical seeding outcomes.

**This project was done purely for my enjoyment and hopeful success in March Madness bracket groups, and is not associated with any class, employer, or other formal commitment.

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

The raw tournament results are cleaned and reshaped into a game-level table with team IDs, seeds, scores, and rounds, but no team statistics attached yet.

<p align="center">
  <img width="628" height="229" alt="image" src="https://github.com/user-attachments/assets/e45bf5b8-3e5a-4717-b70a-abd60b71be31">
</p>

### Merging External Team Features
For each season, external datasets are reshaped into team-season–level tables and merged twice onto the game dataset: once for ATeam and once for BTeam. This creates a symmetric design where every numeric attribute exists in A and B versions.

​Examples of merged features:
- From **CBB**: offensive and defensive efficiencies, pace, shooting splits, offensive/defensive rebounding, turnover rates, and ranking-based stats
- From **KenPom / Barttorvik**: adjusted offense/defense, tempo, average height, effective height, experience, talent, free-throw rates, and efficiency-based SOS measures
- From **Resumes**: quad-level win/loss counts, quality-win indices, and other committee-style resume indicators
- From **KPI**: KPI rating, SOS value, and SOS ranking for each team-season

The external sources are stacked and standardized into a unified team-season table so that any team in any year has a full set of candidate metrics.

<p align="center">
  <img width="720" height="191" alt="image" src="https://github.com/user-attachments/assets/1c1809e0-b03a-42fb-94c6-c6d6ad12ad9e">
</p>

Because sources cover different year ranges, each file is filtered to the overlapping seasons and then concatenated. The team-season table is left-joined onto the game-level results using year and team ID, attaching the appropriate stats to each side of every matchup.

<p align="center">
  <img width="700" height="337" alt="image" src="https://github.com/user-attachments/assets/c0fcf85e-cb8f-4159-80c7-d1afb7835c01">
  <br/>
</p>

The final merged DataFrame includes 100+ columns per game, with consistent A/B feature pairs that maintain symmetry between teams. The following represents a conceptual mapping of the final training data set, where each row is a single game with aligned features for both teams, ready to feed into the machine learning pipeline.

<p align="center">
  <br/>
  <img width="1061" height="238" alt="image" src="https://github.com/user-attachments/assets/fca30f66-cb61-4656-be35-5b39afe43ffa">
</p>

# Feature Engineering and Grouped Selection
The main target variable is a binary flag AWon indicating whether ATeam won the game. To ensure fair modeling, the design matrix is built so that each underlying stat appears as a pair: one column for ATeam and one for BTeam.

Feature selection is performed using a **group-lasso–style approach**:
- Start from the curated set of base team features from sources listed above
- For each base feature, create both A and B columns
- Run repeated fits of a logistic regression with L1-type penalty to select stable feature groups, selecting optimal alpha value across resamples
- Enforce symmetry by requiring that **if a stat is selected for A, the corresponding B stat is also included**; this prevents the model from exploiting arbitrary naming of the two teams

The alpha vs. cross‑validated log loss and feature count plot is used to choose the regularization strength at the point where log loss is minimized, balancing predictive performance with model sparsity.

<p align="center">
  <img width="659" height="393" alt="image" src="https://github.com/user-attachments/assets/79f7e6f1-5c61-417f-b30b-b1faee9c219e">
</p>

# LASSO Feature Selection Results

LASSO ultimately keeps a compact subset of the most informative team metrics from the full pool of 60 features, focusing heavily on efficiency, schedule strength, and resume quality.
- **ADJDE** – Adjusted defensive efficiency, capturing how many points a team allows per possession accounting for opponent quality and tempo
- **ADJOE** – Adjusted offensive efficiency, measuring how effectively a team scores per possession on a tempo‑ and opponent‑adjusted basis
- **BADJ EM** – Barttorvik-style adjusted efficiency margin (offense minus defense), summarizing overall team strength on a single scale
- **BADJ O** – Barttorvik adjusted offensive efficiency, an alternative tempo-free measure of how strong the team is on offense
- **BARTHAG** – Barttorvik’s overall power rating or “win probability vs. an average team,” representing general team quality
- **ELITE SOS** – Strength-of-schedule metric emphasizing games against elite opponents, indicating how battle-tested a team is
- **KADJ EM** – KenPom adjusted efficiency margin, a widely used overall strength indicator combining offense and defense
- **KADJ O** – KenPom adjusted offensive efficiency, quantifying offensive quality in KenPom’s framework
- **KPI #** – KPI ranking value, a resume-based measure that blends performance and schedule to mimic selection-committee evaluations
- **Q1 PLUS Q2 W** – Total wins in Quadrant 1 and Quadrant 2 games, summarizing how often the team beats high- and mid-tier competition
- **Q1 W** – Wins specifically against top-tier (Quadrant 1) opponents
- **R SCORE** – Resume score index that compresses quality wins, bad losses, and schedule into a single resume strength metric
- **TALENT** – Talent rating based on recruiting or roster quality, approximating the underlying player skill level on the team
- **WAB** – “Wins Above Bubble,” estimating how many more games a team has won compared with a typical bubble team given its schedule

The selected features are almost all efficiency, schedule-strength, and resume metrics, which directly capture how good a team is and who they’ve proven it against, rather than simpler descriptors like seed. This suggests the model is learning that underlying power ratings (KenPom/Barttorvik), resume scores (KPI, WAB, Q1/Q2 wins), and talent levels provide more predictive signal than seeding, so when regularization forces it to choose, seed is redundant and gets dropped. The model keeps these broad “catch‑all” indicators and discards more niche stats, such as 2‑ and 3‑point shooting splits, tempo, height, rebounding, free throw percentage, and more, implying that their effects are largely absorbed by the higher-level composite metrics.

# Model Training and Hyperparameter Tuning

Four complementary models are trained using the selected feature set:

1. **LASSO Logistic Regression**
    - Binary logistic regression with L1 penalty to encourage sparsity with the grouped feature selection
2. **Elastic Net Logistic Regression**
    - Logistic regression with elastic net penalty (convex combination of L1 and L2) implemented via saga solver
    - Cross-validated grid search over values of inverse-regularization strength *C* and L1-ratio, with a diagnostic plot of validation log loss to select hyperparameters
3. **Gradient Boosting Classifier**
    - Tree-based model fit on the selected features
    - Optuna optimizes `n_estimators`, `learning_rate`, `max_depth`, `min_samples_leaf`, and `subsample` using log loss on held-out validation sets
4. **Neural Network**
    - Small fully connected network built with TensorFlow/Keras
    - Optuna tunes the `number of neurons per layer`, `learning rate`, `L2 regularization`, and `batch size`, using early stopping on validation loss

### Stabilizing Evaluation
Because the dataset is **relatively small** (roughly a few hundred games across tournaments), a single train/test split can give noisy estimates of performance. To stabilize evaluation:
- Each model is trained and evaluated across many random splits (100+ iterations) with stratification on the outcome variable
- For each model, the mean and standard deviation of test log loss across iterations are reported, alongside training log loss

The model performance plot compares out-of-sample log loss across each model type, as well as the baseline, defined as the predicted probability for every game being equal to the underlying prevalence in the data, illustrating to what extent each model is able to improve the predicted probabilities.

<p align="center">
  <img width="527" height="327" alt="image" src="https://github.com/user-attachments/assets/dcc3c1d9-371a-4b43-8dc9-87ed4f30a57b">
</p>

The calibration plot compares predicted win probabilities to actual outcomes for all four models, illustrating how well each model’s probability estimates line up with observed frequencies across the probability range.

<p align="center">
  <img width="527" height="527" alt="image" src="https://github.com/user-attachments/assets/d1735c42-a2da-4ee8-8a21-af66499aa9af">
</p>

# Final Model Selection

**Elastic Net** was ultimately chosen for deployment because it provided the best balance of accuracy, stability, and bracket realism. The neural network underperformed, which is no surprise for such a small dataset. LASSO was marginally more accurate but noticeably more volatile and prone to extreme probabilities, and Elastic Net delivered competitive log loss while keeping predictions close to the seed‑based baseline. To underscore this, some out‑of‑sample model predictions were clearly too aggressive relative to seeding:

- LASSO assigned 5‑seed Colorado a **15.7%** win probability against 12‑seed Georgetown in the 2021 first round
- LASSO also gave 2‑seed Louisville only a **25.4%** chance to beat 7‑seed Michigan in the 2017 second round
- Boosting placed 1‑seed Arizona at **31.4%** to beat 5‑seed Houston in the 2022 Sweet Sixteen
- Boosting similarly rated 3‑seed Creighton at **29.7%** to defeat 11‑seed Oregon in the 2024 second round

These examples highlight why a slightly “tamer” but more consistent model like Elastic Net is preferable for a deployed bracket strategy.

<p align="center">
  <img width="527" height="327" alt="image" src="https://github.com/user-attachments/assets/fd9b8687-f1c5-4461-97f5-442860f8582d">
</p>

# Seed-Based Baseline and Historical Upsets
Before trusting ML to drive bracket picks, the project establishes a **seeding-only baseline**:
- Fit a simple logistic model using only the log ratio of seeds, log(BSeed/ASeed), to predict ATeam win probability
- Use this simple model to **estimate typical win probabilities for every 1–16 vs. 1–16 pairing** and visualize them as a 16×16 probability matrix
- This serves as a benchmark for how much predictive power is available without any advanced metrics

Historical upset rates by seed matchup are trained from the same historical window, providing a reference for how aggressive ML-driven upsets should be. The idea is to avoid a bracket that is out-of-line with historical frequencies. The 16×16 seed‑baseline probability matrix visualizes the seed‑only model’s estimated win probabilities for every possible seed matchup, highlighting only cases where a seed is equal or favored to show how strongly the baseline expects better seeds to advance.

<p align="center">
  <img width="518" height="443" alt="image" src="https://github.com/user-attachments/assets/8d0dd33f-2109-47d1-aa1f-5130e1da1974">
</p>

# Bracket Construction Strategy
The bracket logic revolves around a **lift score** that compares two perspectives on each game:
- The log‑odds from the full machine‑learning model
- The log‑odds implied by the simple, seed‑based baseline

When this lift is **negative**, it signals that the model thinks the underdog is more dangerous than the seeding alone would suggest, flagging a potential upset. For each early round, historical data is used to translate typical upset rates into a round‑specific lift cutoff:
- Historical games are ranked by lift (from most underdog‑friendly to most favorite‑friendly)
- The cutoff is chosen so that the fraction of games above that cutoff matches how often underdogs have actually won in that round
- In a future tournament, whenever a matchup’s lift for the worse seed is below the relevant cutoff, the bracket intentionally picks the upset, even if the favorite still has the higher raw win probability

This procedure is repeated separately for every round through the Elite Eight, giving each stage its own lift threshold. Early rounds tend to allow more upset picks, while later rounds are more conservative, reflecting how rarely big upsets occur deep in the tournament. Because decisions are driven by these numeric thresholds rather than a fixed quota of upsets:
- The **number of predicted upsets is allowed to vary** from year to year
- Over many tournaments, the average upset rate naturally lines up with history, but **any given bracket can lean more chaotic or more chalky** depending on how strongly the model disagrees with the seed baseline.

In the Final Four and title game, the bracket simply takes the team with the higher modeled win probability, avoiding extra thresholding when comparable historical data is limited and matchups are few.
​



​
