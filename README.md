# Project Overview
This repository contains code to estimate win probabilities for NCAA men's tournament games from 2017–2025 and to turn those probabilities into a bracket strategy for upcoming tournaments. The workflow covers data collection, merging and cleaning, feature selection with grouped penalties, model training and evaluation, and bracket construction rules that balance model predictions with historical seeding outcomes.

A visual telling of this project can be found at the following [Streamlit app](https://march-madness.streamlit.app/)

This project exists in my relentless pursuit of reaching my high of the 2021 Tournament again. That year, I took home first place in my Dad's work bracket pool, with 150+ entries. It was glorious. I return to the bracket-making starting blocks, this time armed with the power of data and machine learning.

***This project was done purely for my enjoyment and hopeful success in March Madness bracket groups, and is not associated with any class, employer, or other formal commitment.*

# Data Collection and Sources
Tournament results and core identifiers come from the official March Machine Learning Mania Kaggle competition (detailed results, seeds, and team metadata). Additional team strength metrics are pulled from several external sources covering efficiency, resume quality, and advanced scouting stats for recent seasons.
​
### Key sources:

- **[Kaggle NCAA tourney data (MM Mania)](https://www.kaggle.com/competitions/march-machine-learning-mania-2025/data)**: play-by-play–level box score stats and tournament seeds
- **[Kaggle CBB dataset](https://www.kaggle.com/datasets/andrewsundberg/college-basketball-dataset)**: regular-season box score–based advanced metrics and adjusted efficiencies
- **​[KenPom / Barttorvik](https://www.kaggle.com/datasets/nishaanamin/march-madness-data?select=KenPom+Barttorvik.csv)**: tempo-free efficiency, tempo, height, experience, talent ratings, and other advanced team-level metrics
- **[Resumes](https://www.kaggle.com/datasets/nishaanamin/march-madness-data?select=Resumes.csv)**: "resume" statistics summarizing quality wins and losses by quadrant and other selection committee-style descriptors
- **​[KPI rankings](https://faktorsports.com/)**: resume-based rating and SOS metrics

# Building the Modeling Dataset
The modeling dataset is constructed at the game level, with each row corresponding to a single NCAA tournament game. The pipeline starts from the detailed tournament results and augments them with seeds and team names, restricted to the main bracket rounds for seasons where external sources are available.

Key steps:
- **Tournament results**
  - Load tournament results and drop unneeded box score columns, keeping scoring, team IDs, and day numbers
  - Map DayNum to human-readable rounds (Play-in, Round 1–6) using a custom function that accounts for the 2021 schedule quirk, double checking round counts per season
- **Seeds and naming**
  - Strip regional prefixes from seeds to create numeric seeds (1–16) and merge them onto winning and losing teams
  - Normalize ordering by renaming the alphabetically first team as ATeam and the other as BTeam, then realign scores, seeds, and IDs so the "A" side is not always the winner
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

# Grouped Feature Selection
The main target variable is a binary flag AWon indicating whether ATeam won the game. To ensure fair modeling, the design matrix is built so that each underlying stat appears as a pair: one column for ATeam and one for BTeam.

Feature selection is performed using a **group-lasso–style approach**:
- Start from the curated set of base team features from sources listed above
- Run repeated fits of a logistic regression with L1-type penalty to select stable feature groups, selecting optimal alpha value across resamples
- Enforce symmetry by requiring that **if a stat is selected for A, the corresponding B stat is also included**; this prevents the model from exploiting arbitrary naming of the two teams
- The regularization strength selected here serves as a starting point; it is further varied as a hyperparameter during each individual model's tuning stage

The alpha vs. cross‑validated log loss and feature count plot is used to choose the regularization strength at the point where log loss is minimized, balancing predictive performance with model sparsity.

<p align="center">
  <img width="659" height="393" alt="image" src="https://github.com/user-attachments/assets/79f7e6f1-5c61-417f-b30b-b1faee9c219e">
</p>

# Model Training and Hyperparameter Tuning

Five complementary models are trained using the selected feature set:

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
5. **Mixture of Experts (MoE)**
    - An ensemble of logistic regression "experts," each trained on a random subset of features, with a logistic gating function that learns which experts to trust for a given matchup
    - Optuna tunes the key hyperparameters: `alpha` (L1 regularization applied during grouped feature selection before experts are created), `n_experts` (number of individual expert models), `n_features` (how many randomly selected features each expert sees), `C_expert` (inverse L2 regularization strength within each expert's logistic regression), and `C_meta` (inverse L2 regularization strength for the gating function that combines expert outputs)

### Model Evaluation
Because the dataset is **relatively small** (roughly a few hundred games across tournaments), a single train/test split can give noisy estimates of performance. To stabilize evaluation:
- Each model is trained and evaluated across many random splits (100+ iterations) with stratification on the outcome variable
- For each model, the mean and standard deviation of test log loss across iterations are reported, alongside training log loss

The model performance plot compares out-of-sample log loss across each model type, as well as the baseline (defined as the predicted probability for every game being equal to the underlying prevalence in the data) illustrating to what extent each model is able to improve the predicted probabilities. The Mixture of Experts model is selected for this task since it achieves competitive log loss while maintaining strong generalization across splits.

<p align="center">
  <img width="534" height="331" alt="image" src="https://github.com/user-attachments/assets/634c745c-a553-4e7d-bd74-26408e633c3c">
</p>

The calibration plot compares predicted win probabilities to actual outcomes for all five models, illustrating how well each model's probability estimates line up with observed frequencies across the probability range. No particular model shows any significant deviation from the baseline, indicating that all models produce reasonably well-calibrated probabilities.

<p align="center">
  <img width="545" height="527" alt="image" src="https://github.com/user-attachments/assets/aa1e5482-bebb-4e10-bcc4-186f41d27b7b">
</p>

# Round Probabilities and Monte Carlo Simulation
To move from individual game predictions to full tournament projections, the pipeline constructs a **64×64 probability matrix** estimating the win probability for every possible team-vs-team matchup in the bracket. This matrix is built using the trained model's predictions, giving a complete picture of how any two teams in the field would fare head-to-head.

From there, a **Monte Carlo simulation** is run 10,000 times: each simulation plays out the entire tournament bracket by sampling game outcomes according to the 64×64 probability matrix. Across all 10,000 iterations, the pipeline records each team's probability of advancing to each subsequent round from the Round of 32 all the way through the championship. These round-by-round probabilities provide a more nuanced view than single-game predictions alone, capturing how a team's path difficulty, bracket position, and potential opponents all factor into their chances of making a deep run.

# Seed-Based Baseline and Historical Upsets
Before trusting ML to drive bracket picks, the project establishes a **seeding-only baseline**:
- Fit a simple logistic model using only the log ratio of seeds, log(BSeed/ASeed), to predict ATeam win probability
- Use this simple model to **estimate typical win probabilities for every 1–16 vs. 1–16 pairing** and visualize them as a 16×16 probability matrix

Historical upset rates by seed matchup are trained from the same historical window, providing a reference for how aggressive ML-driven upsets should be. The idea is to avoid a bracket that is out-of-line with historical frequencies. The 16×16 seed‑baseline probability matrix visualizes the seed‑only model's estimated win probabilities for every possible seed matchup, highlighting only cases where a seed is equal or favored to show how strongly the baseline expects better seeds to advance.

<p align="center">
  <img width="518" height="443" alt="image" src="https://github.com/user-attachments/assets/8d0dd33f-2109-47d1-aa1f-5130e1da1974">
</p>

# Bracket Construction Strategy
At the end of the day, the goal is to build a bracket that doesn't just rubber-stamp the higher seed in every game. Where's the fun in that? We want a bracket that's willing to go out on a limb and call upsets when the data says they're worth the risk.

The bracket logic revolves around a **lift score** that compares two perspectives on each game:
- The log‑odds from the full machine‑learning model
- The log‑odds implied by the simple, seed‑based baseline

When this lift is **negative**, it signals that the model thinks the underdog is more dangerous than the seeding alone would suggest, flagging a potential upset. For each early round, historical data is used to translate typical upset rates into a round‑specific lift cutoff:
- Historical games are ranked by lift (from most underdog‑friendly to most favorite‑friendly)
- The cutoff is chosen so that the fraction of games above that cutoff matches how often underdogs have actually won in that round
- In a future tournament, whenever a matchup's lift for the worse seed is below the relevant cutoff, the bracket intentionally picks the upset, **even if the favorite still has the higher raw win probability**
- This strategy naturally results in a distribution of first-round victories that aligns with history. 13-16 seeds are rarely, but not never, selected to win while 9-12 seeds are selected to win at a relatively common rate

This procedure is repeated separately for every round through the Elite Eight, giving each stage its own lift threshold. Early rounds tend to allow more upset picks, while later rounds are more conservative, reflecting how rarely big upsets occur deep in the tournament. Because decisions are driven by these numeric thresholds rather than a fixed quota of upsets:
- The **number of predicted upsets is allowed to vary** from year to year
- Over many tournaments, the average upset rate naturally lines up with history, but **any given bracket can lean more chaotic or more chalky** depending on how strongly the model disagrees with the seed baseline

In the Final Four and title game, the bracket simply takes the team with the higher modeled win probability, avoiding extra thresholding when comparable historical data is limited and matchups are few.

### Historical Rate Adjustment
After implementing the initial bracket strategy, the results were **a bit too upset-happy,** especially in later rounds. The root cause of this was that the lift thresholds are calibrated against real historical games, where the teams that advance to later rounds include a fair amount of randomness. But in the hypothetical bracket, the **"analytically strong underdogs" from earlier rounds are the ones advancing**, which means later-round matchups are more likely to feature teams that are genuinely worthy of another upset pick. This creates a compounding effect where the bracket keeps picking upsets deeper into the tournament at a higher rate than history would support.

To correct for this, an **additional conservative factor** of `std(lift thresholds) / 2` is added to the lift thresholds for Rounds 2–4 (Round of 32 through Elite Eight). This nudges the bar for calling an upset slightly higher in later rounds, dampening the compounding effect. The specific value of half a standard deviation was chosen because it brought the upset rate in line with what felt right relative to historical frequencies. Not exactly a mathematically derived optimum, but a pragmatic adjustment. This is a potential area for future study and refinement.

# Explaining Predictions with SHAP
To make the model's predictions interpretable at the matchup level, **SHAP (SHapley Additive exPlanations)** values are computed for every game. In the Bracket tab, hovering over any matchup surfaces a SHAP waterfall plot that breaks down exactly **why** the Mixture of Experts model arrived at a given win probability.

<p align="center">
  <img width="555" height="444" alt="SHAP example: North Carolina vs Mississippi" src="https://github.com/user-attachments/assets/2ad785e7-2c27-476b-bfda-93c9d12bc45d">
</p>
 
Take the North Carolina vs. Ole Miss example above. The model gives UNC a **62.8%** chance to win:
- **Blue bars** push the prediction toward the blue team (North Carolina); **red bars** push it toward the red team (Mississippi)
- **ADJOE (adjusted offensive efficiency) for UNC** is the biggest driver, shifting the prediction **11 percentage points** in their favor
- **ADJDE (adjusted defensive efficiency) for UNC** pushes **8 points back** toward Ole Miss, signaling that the Tar Heels are giving up buckets at a high rate
- **TALENT for UNC** contributes another **5 points** in their favor, which is an indicator of overall player talent level
- You can continue down the waterfall, but the **top few bars** tell you what matters most for any given game and how the model arrived at its number
 
### Prediction Storytelling
If you listen to college basketball analysts break down tournament matchups, their reasoning tends to be very specific and narrative-driven:
- *"The underdog will want to push the pace, force turnovers, and get transition buckets, which isn't the favorite's style of play and will make them uncomfortable"*
- *"I don't think they have the size and physicality to match up in the paint."*

These are stories built on particular box score traits, such as tempo, turnover rate, rebounding, and height. The SHAP plots tell a different story:
- The features that **dominate predictions** across nearly every matchup are **catch-all adjusted efficiency metrics**: adjusted offensive and defensive efficiency, overall efficiency margins, power ratings like BARTHAG, and resume-quality indicators like KPI and WAB
- The specific box score stats that analysts love to build narratives around don't often crack the top of the waterfall
- When I experimented with **removing the catch-all efficiency metrics** to force the model onto those more granular, "storytelling-friendly" features, the **predictions became noticeably less reliable**
 
Why? It likely comes down to what these composite metrics actually represent:
- A stat like adjusted offensive efficiency is **already integrating** a team's shooting, turnover rate, offensive rebounding, free throw rate, and the quality of defenses they've faced into a single tempo- and opponent-adjusted number
- Asking the model to **re-derive that same signal** from the raw components, especially with a small dataset, introduces noise without adding new information
- The composites are more predictive precisely because they're **more stable** and less susceptible to small-sample weirdness in any one stat
 
In short, the model's best predictions come from knowing ***how good*** a team is overall, not from dissecting ***how*** they're good. The narrative might be less colorful than what you'd hear on a studio show, but the probabilities are sharper for it.
 
## Thank You
If you've made it this far, I appreciate you taking an interest in the process behind all of this. I hope you enjoy the project, whether you're here to geek out over the modeling, steal some bracket strategy, or just see if the machine can beat your gut.
 
If you have any follow-up questions or just want to talk March Madness, feel free to reach out: **ethandavenport@utexas.edu**​



​
