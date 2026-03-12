"""
tournament_simulator.py
=======================
March Madness bracket simulator built on top of MixtureOfExperts.

Region / Final Four conventions
--------------------------------
Your seed strings look like  "W01", "X02", "Y16", "Z11".
  - Within each region: 1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15
  - Final Four semi-finals:  W winner  vs  X winner
                             Y winner  vs  Z winner
  - Championship:            WX winner vs  YZ winner

df is the matchup DataFrame with _A / _B feature columns.
seed_df must have columns: Season, TeamName, Seed  (e.g. "W01", "Z16")

stats_df is derived automatically from df — you do NOT need to pass it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from moe_classifier import MixtureOfExperts, split_n_scale

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROUND_NAMES = [
    "Round of 64",
    "Round of 32",
    "Sweet 16",
    "Elite 8",
    "Final Four",
    "Championship",
    "Champion",
]

# Round-of-64 seed matchups within every region (stronger seed first)
_R64_PAIRS = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]

# How R64 winners fold into R32 (index pairs from the 8-team survivors list)
_R32_FOLD = [(0, 1), (2, 3), (4, 5), (6, 7)]

# Final Four semi-final pairings: (region_a, region_b)
# W plays X, Y plays Z — championship is winner of each semi
_FF_PAIRS = [("W", "X"), ("Y", "Z")]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_seed(seed_str: str) -> Tuple[str, int]:
    """
    "W01"  -> ("W",  1)
    "Z16"  -> ("Z", 16)
    "MW11" -> ("MW", 11)
    """
    seed_str = str(seed_str).strip()
    for i, ch in enumerate(seed_str):
        if ch.isdigit():
            return seed_str[:i], int(seed_str[i:])
    raise ValueError(f"Cannot parse seed: {seed_str!r}")


def stats_df_from_matchups(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Derive a (Season, TeamName, <features>) table from a matchup DataFrame
    that has columns ending in _A and _B.

    Returns (stats_df, base_feature_cols).
    """
    meta = {"Season", "ATeamName", "BTeamName", "AWon"}

    a_feat_cols = [c for c in df.columns if c.endswith("_A") and c not in meta]
    b_feat_cols = [c for c in df.columns if c.endswith("_B") and c not in meta]
    base_cols = [c[:-2] for c in a_feat_cols]   # strip "_A"

    side_a = df[["Season", "ATeamName"] + a_feat_cols].copy()
    side_a.columns = ["Season", "TeamName"] + base_cols

    side_b = df[["Season", "BTeamName"] + b_feat_cols].copy()
    side_b.columns = ["Season", "TeamName"] + base_cols

    stats = (
        pd.concat([side_a, side_b], ignore_index=True)
        .drop_duplicates(subset=["Season", "TeamName"])
        .reset_index(drop=True)
    )
    return stats, base_cols


def build_bracket(seed_df_season: pd.DataFrame) -> Tuple[Dict, Dict]:
    """
    Build the bracket structure for a single season.

    Seed format: first character = region letter, next two characters = seed number.
    e.g. "W01" -> region "W", seed 1

    Returns
    -------
    bracket : dict  {region -> list of 8 ((team, seed_num), (team, seed_num))}
    team_map : dict  {(region, seed_num) -> team_name}
    """
    team_map = {}
    for _, row in seed_df_season.iterrows():
        seed_str = str(row["Seed"]).strip()
        region = seed_str[0]
        num = int(seed_str[1:3])
        team_map[(region, num)] = row["TeamName"]

    regions = sorted({k[0] for k in team_map})

    bracket = {}
    for region in regions:
        matchups = []
        for s1, s2 in _R64_PAIRS:
            t1 = team_map.get((region, s1), f"TBD_{region}_{s1}")
            t2 = team_map.get((region, s2), f"TBD_{region}_{s2}")
            matchups.append(((t1, s1), (t2, s2)))
        bracket[region] = matchups

    return bracket, team_map


# ---------------------------------------------------------------------------
# Build 64x64 probability matrix
# ---------------------------------------------------------------------------

def build_prob_matrix(
    season: int,
    df: pd.DataFrame,
    seed_df: pd.DataFrame,
    model_kwargs: Optional[dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, MixtureOfExperts]:
    """
    Train MoE on all seasons except `season`, then compute P(i beats j)
    for every pair of teams in that season's tournament.

    Parameters
    ----------
    season : int
        Tournament year to simulate (held out from training).
    df : pd.DataFrame
        Full matchup DataFrame with _A / _B feature columns.
    seed_df : pd.DataFrame
        Columns: Season, TeamName, Seed.
        Seed format: first char = region (W/X/Y/Z), next two chars = seed number (01-16).
    model_kwargs : dict, optional
        Passed to MixtureOfExperts().

    Returns
    -------
    prob_matrix : pd.DataFrame  (n_teams x n_teams)
        prob_matrix.loc[A, B] = P(A beats B)
    seed_info : pd.DataFrame
        TeamName, Seed, Region, SeedNum for the given season.
    model : MixtureOfExperts
        The fitted model.
    """
    if model_kwargs is None:
        model_kwargs = {}

    # --- Derive stats table from matchup df ---
    stats_df, base_cols = stats_df_from_matchups(df)

    # --- Train on all years except target season ---
    train_df_raw = df[df["Season"] != season].copy()
    X_tr, _, y_tr, _, _, _ = split_n_scale(train_df_raw)

    model = MixtureOfExperts(**model_kwargs)
    model.fit(X_tr, y_tr)

    # --- Parse season seed info ---
    # Seed format guaranteed: char 0 = region letter, chars 1-2 = zero-padded seed number
    season_seeds = seed_df[seed_df["Season"] == season].copy().reset_index(drop=True)
    season_seeds["Region"] = season_seeds["Seed"].str[0]
    season_seeds["SeedNum"] = season_seeds["Seed"].str[1:3].astype(int)

    # Keep only teams that actually played in df that season (eliminates play-in losers)
    season_df = df[df["Season"] == season]
    teams_in_df = set(season_df["ATeamName"]).union(set(season_df["BTeamName"]))
    season_seeds = season_seeds[season_seeds["TeamName"].isin(teams_in_df)].reset_index(drop=True)

    teams = season_seeds["TeamName"].tolist()

    # --- Fit scaler on training stats only ---
    train_seasons = df[df["Season"] != season]["Season"].unique()
    train_stats = stats_df[stats_df["Season"].isin(train_seasons)]

    paired_cols = [f"{f}_A" for f in base_cols] + [f"{f}_B" for f in base_cols]

    # Each training team row becomes a scaler-fitting row with itself on both sides
    scaler_rows = train_stats[base_cols].rename(
        columns={f: f"{f}_A" for f in base_cols}
    ).copy()
    for f in base_cols:
        scaler_rows[f"{f}_B"] = train_stats[f].values
    scaler = StandardScaler()
    scaler.fit(scaler_rows[paired_cols])

    # --- Index stats for fast lookup ---
    stats_index = stats_df.set_index(["Season", "TeamName"])

    # --- Fill n x n matrix (upper triangle only, mirror below) ---
    n = len(teams)
    matrix = pd.DataFrame(np.full((n, n), np.nan), index=teams, columns=teams)
    np.fill_diagonal(matrix.values, 0.5)

    for i, team_a in enumerate(teams):
        for j, team_b in enumerate(teams):
            if i >= j:
                continue

            try:
                row_a = stats_index.loc[(season, team_a)]
                row_b = stats_index.loc[(season, team_b)]
            except KeyError:
                matrix.loc[team_a, team_b] = 0.5
                matrix.loc[team_b, team_a] = 0.5
                continue

            matchup_data = {f"{f}_A": row_a[f] for f in base_cols}
            matchup_data.update({f"{f}_B": row_b[f] for f in base_cols})

            matchup_scaled = pd.DataFrame(
                scaler.transform(pd.DataFrame([matchup_data])[paired_cols]),
                columns=paired_cols,
            )

            p_a = model.predict_proba(matchup_scaled)[0, 1]
            matrix.loc[team_a, team_b] = p_a
            matrix.loc[team_b, team_a] = 1.0 - p_a

    return matrix, season_seeds, model


# ---------------------------------------------------------------------------
# Simulate one bracket
# ---------------------------------------------------------------------------

def _simulate_once(
    bracket: Dict,
    prob_matrix: pd.DataFrame,
    rng: np.random.Generator,
) -> Dict[str, int]:
    """
    Simulate a single tournament.

    Round index stored per team (highest round reached):
        0 = lost in Round of 64
        1 = lost in Round of 32
        2 = lost in Sweet 16
        3 = lost in Elite 8
        4 = lost in Final Four semi
        5 = lost in Championship game
        6 = Champion
    """

    def play(t1: str, t2: str) -> Tuple[str, str]:
        p = prob_matrix.loc[t1, t2]
        return (t1, t2) if rng.random() < p else (t2, t1)

    results: Dict[str, int] = {}
    region_champs: Dict[str, str] = {}

    for region, matchups in bracket.items():

        # Round of 64
        r64_winners = []
        for (t1, _), (t2, _) in matchups:
            w, l = play(t1, t2)
            results[l] = 0
            r64_winners.append(w)

        # Round of 32: pairs (0,1),(2,3),(4,5),(6,7)
        r32_winners = []
        for a, b in _R32_FOLD:
            w, l = play(r64_winners[a], r64_winners[b])
            results[l] = 1
            r32_winners.append(w)

        # Sweet 16: (0,1) and (2,3)
        s16_winners = []
        for a, b in [(0, 1), (2, 3)]:
            w, l = play(r32_winners[a], r32_winners[b])
            results[l] = 2
            s16_winners.append(w)

        # Elite 8
        w, l = play(s16_winners[0], s16_winners[1])
        results[l] = 3
        region_champs[region] = w

    # Final Four: W vs X, then Y vs Z
    ff_winners = []
    for reg_a, reg_b in _FF_PAIRS:
        t1 = region_champs.get(reg_a)
        t2 = region_champs.get(reg_b)
        if t1 is None and t2 is None:
            continue
        if t1 is None:
            ff_winners.append(t2)
            continue
        if t2 is None:
            ff_winners.append(t1)
            continue
        w, l = play(t1, t2)
        results[l] = 4
        ff_winners.append(w)

    # Championship
    if len(ff_winners) == 2:
        w, l = play(ff_winners[0], ff_winners[1])
        results[l] = 5
        results[w] = 6
    elif len(ff_winners) == 1:
        results[ff_winners[0]] = 6

    return results


# ---------------------------------------------------------------------------
# Run N simulations
# ---------------------------------------------------------------------------

def simulate_tournament(
    prob_matrix: pd.DataFrame,
    seed_info: pd.DataFrame,
    n_sims: int = 1000,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run N Monte-Carlo bracket simulations.

    Returns
    -------
    advancement_df : pd.DataFrame
        Rows = teams sorted by region then seed.
        Columns = Seed, Region, SeedNum, Round of 64, Round of 32, ..., Champion.
        Values = fraction of simulations that team reached that round.
    sim_detail : pd.DataFrame
        Shape (n_sims, n_teams). Raw highest-round-reached values (0-6).
    """
    rng = np.random.default_rng(random_state)
    bracket, _ = build_bracket(seed_info)
    teams = seed_info["TeamName"].tolist()

    sim_records = [_simulate_once(bracket, prob_matrix, rng) for _ in range(n_sims)]

    sim_detail = (
        pd.DataFrame(sim_records)
        .reindex(columns=teams)
        .fillna(0)
        .astype(int)
    )

    thresholds = {
        "Round of 64":  0,
        "Round of 32":  1,
        "Sweet 16":     2,
        "Elite 8":      3,
        "Final Four":   4,
        "Championship": 5,
        "Champion":     6,
    }

    adv_data = {rnd: (sim_detail >= thr).mean() for rnd, thr in thresholds.items()}
    advancement_df = pd.DataFrame(adv_data, index=teams)

    seed_lookup = seed_info.set_index("TeamName")[["Seed", "Region", "SeedNum"]]
    advancement_df = (
        seed_lookup.join(advancement_df)
        .sort_values(["Region", "SeedNum"])
    )

    return advancement_df, sim_detail


# ---------------------------------------------------------------------------
# End-to-end wrapper
# ---------------------------------------------------------------------------

def run_tournament_analysis(
    season: int,
    df: pd.DataFrame,
    seed_df: pd.DataFrame,
    model_kwargs: Optional[dict] = None,
    n_sims: int = 1000,
    random_state: Optional[int] = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, MixtureOfExperts]:
    """
    Full pipeline: train MoE -> build 64x64 matrix -> simulate -> return results.

    Parameters
    ----------
    season : int
        Tournament year to simulate (held out from training).
    df : pd.DataFrame
        Matchup DataFrame with _A/_B feature columns. stats_df is derived
        automatically — no need to pass separately.
    seed_df : pd.DataFrame
        Columns: Season, TeamName, Seed  (e.g. "W01", "Z16").
        Final Four pairings: W vs X, Y vs Z.
    model_kwargs : dict, optional
        MixtureOfExperts hyperparameters.
    n_sims : int
        Monte-Carlo simulations.
    random_state : int or None

    Returns
    -------
    advancement_df : pd.DataFrame
        Team advancement probabilities by round, sorted by region and seed.
    prob_matrix : pd.DataFrame
        64x64 P(row beats col) matrix with team names as index/columns.
    model : MixtureOfExperts
        Fitted model.

    Example
    -------
    >>> adv, matrix, model = run_tournament_analysis(
    ...     season=2024,
    ...     df=df,
    ...     seed_df=seed_df,
    ...     model_kwargs=dict(alpha=0.01, n_features=5, n_experts=20),
    ...     n_sims=1000,
    ... )
    >>> print(adv[["Seed", "Sweet 16", "Final Four", "Champion"]]
    ...       .sort_values("Champion", ascending=False).head(10))
    >>> print(f"P(UConn beats Purdue) = {matrix.loc['Connecticut', 'Purdue']:.3f}")
    """
    print(f"[{season}] Building probability matrix (training on all other years)...")
    prob_matrix, seed_info, model = build_prob_matrix(
        season=season,
        df=df,
        seed_df=seed_df,
        model_kwargs=model_kwargs,
    )
    n = prob_matrix.shape[0]
    print(f"[{season}] Matrix built — {n} teams, {n*(n-1)//2} unique matchup probabilities.")

    print(f"[{season}] Simulating {n_sims:,} tournaments...")
    advancement_df, sim_detail = simulate_tournament(
        prob_matrix=prob_matrix,
        seed_info=seed_info,
        n_sims=n_sims,
        random_state=random_state,
    )
    print(f"[{season}] Done.")

    return advancement_df, prob_matrix, model
