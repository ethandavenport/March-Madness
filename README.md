# Project Overview
This repository contains code to estimate win probabilities for NCAA men’s tournament games from 2017–2025 and to turn those probabilities into a bracket strategy for upcoming tournaments. The workflow covers data collection, merging and cleaning, feature selection with grouped penalties, model training and evaluation, and bracket construction rules that balance model predictions with historical seeding outcomes.

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
The modeling dataset is constructed at the game level, with each row corresponding to a single NCAA tournament game between 2017 and 2024. The pipeline starts from the detailed tournament results and augments them with seeds and team names, restricted to the main bracket rounds for seasons where external sources are available.
​
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
  <img width="701" height="173" alt="image" src="https://github.com/user-attachments/assets/5a456561-c01f-4de5-80be-133e23eb9557">
</p>


​
