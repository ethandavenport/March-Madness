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

df is the matchup DataFrame with _A / _B feature columns (used for training only).
seed_df must have columns: Season, TeamName, Seed  (e.g. "W01", "Z16")
team_data must have columns: Year, TeamID, Team, <stat cols>
    — used to look up team stats when building the probability matrix.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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
    meta = {"Season", "ATeamName", "BTeamName", "AWon",
            "ATeamID", "BTeamID", "Region_A", "Region_B", "Round", "SlotID"}

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
    team_data: pd.DataFrame,
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
        Used only for model training.
    seed_df : pd.DataFrame
        Columns: Season, TeamName, Seed.
        Seed format: first char = region (W/X/Y/Z), next two chars = seed number (01-16).
    team_data : pd.DataFrame
        Team-level stats with columns Year, TeamID, Team, <stat cols>.
        Used to look up stats for every team when building the prob matrix.
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

    # --- Ensure season is always a string ---
    season = str(season)

    # --- Train on all years except target season ---
    train_df_raw = df[df["Season"] != season].copy()
    X_tr, _, y_tr, _, _, _, scaler = split_n_scale(train_df_raw)

    model = MixtureOfExperts(**model_kwargs)
    model.fit(X_tr, y_tr)

    # --- Feature columns (same order the scaler was fit on) ---
    paired_cols = list(X_tr.columns)
    base_cols = [c[:-2] for c in paired_cols if c.endswith("_A")]

    # --- Build season stats lookup from team_data, joined via TeamID ---
    td_season = team_data[team_data["Year"].astype(str) == season].copy()
    td_season["TeamID"] = td_season["TeamID"].astype(str)

    # Only keep stat columns that match base_cols from the model
    stat_cols_td = [c for c in td_season.columns if c not in ("Year", "TeamID", "Team")]
    common_cols = [c for c in base_cols if c in stat_cols_td]

    # Build a TeamID -> stats mapping
    td_stats = td_season[["TeamID"] + common_cols].copy()
    td_stats = td_stats.drop_duplicates(subset=["TeamID"])

    # --- Parse season seed info ---
    season_seeds = seed_df[seed_df["Season"] == season].copy().reset_index(drop=True)
    season_seeds["Region"] = season_seeds["Seed"].str[0]
    season_seeds["SeedNum"] = season_seeds["Seed"].str[1:3].astype(int)
    season_seeds["TeamID"] = season_seeds["TeamID"].astype(str)

    # Join stats onto seeds via TeamID, keyed by TeamName for the matrix lookup
    seed_with_stats = season_seeds.merge(td_stats, on="TeamID", how="left")
    season_stats = seed_with_stats[["TeamName"] + common_cols].copy()
    # Seed comes from seed_df, not team_data
    if "Seed" in base_cols and "Seed" not in season_stats.columns:
        season_stats["Seed"] = seed_with_stats["SeedNum"].values
    # Add any missing base_cols as NaN
    for c in base_cols:
        if c not in season_stats.columns:
            season_stats[c] = np.nan
    season_stats = season_stats.drop_duplicates(subset=["TeamName"]).set_index("TeamName")

    # --- Resolve play-in games ---
    # Known play-in winners for past tournaments (keyed by int year)
    _known_playin_winners = {
        2017: ["1243", "1291", "1413", "1425"],
        2018: ["1347", "1382", "1393", "1411"],
        2019: ["1113", "1125", "1192", "1295"],
        2021: ["1179", "1313", "1411", "1417"],
        2022: ["1231", "1323", "1411", "1460"],
        2023: ["1113", "1192", "1338", "1394"],
        2024: ["1160", "1161", "1212", "1447"],
        2025: ["1106", "1291", "1314", "1462"],
    }

    # Detect play-in games and build metadata
    playin_mask = season_seeds["Seed"].str.contains(r"[ab]$", regex=True)
    playin_games = []  # list of (region, seed_num, team_a_name, team_b_name)

    if playin_mask.any():
        playin_seeds = season_seeds[playin_mask].copy()
        main_seeds = season_seeds[~playin_mask].copy()

        # Group play-in pairs
        for (region, seed_num), grp in playin_seeds.groupby(["Region", "SeedNum"]):
            if len(grp) == 2:
                names = grp["TeamName"].tolist()
                playin_games.append((region, seed_num, names[0], names[1]))

        if int(season) in _known_playin_winners:
            # Past year: keep only known winners, drop losers entirely
            known = set(str(t) for t in _known_playin_winners[int(season)])
            winners = playin_seeds[playin_seeds["TeamID"].astype(str).isin(known)].copy()
            winners["Seed"] = winners["Seed"].str.replace(r"[ab]$", "", regex=True)
            winners["SeedNum"] = winners["Seed"].str[1:3].astype(int)
            season_seeds = pd.concat([main_seeds, winners], ignore_index=True)
            playin_games = []  # all resolved, no variable games
        else:
            # Future year: resolve 16-seeds (lower TeamID wins), keep both non-16-seeds
            resolved_list = []
            remaining_playin_games = []

            for (region, seed_num, t1, t2) in playin_games:
                grp = playin_seeds[
                    (playin_seeds["Region"] == region) &
                    (playin_seeds["SeedNum"] == seed_num)
                ].copy()

                if seed_num == 16:
                    # Pick lower TeamID as winner
                    grp["_tid_int"] = grp["TeamID"].astype(int)
                    winner_row = grp.loc[[grp["_tid_int"].idxmin()]].drop(columns=["_tid_int"])
                    winner_row = winner_row.copy()
                    winner_row["Seed"] = winner_row["Seed"].str.replace(r"[ab]$", "", regex=True)
                    winner_row["SeedNum"] = winner_row["Seed"].str[1:3].astype(int)
                    resolved_list.append(winner_row)
                else:
                    # Keep BOTH teams — they'll all be in the prob matrix
                    both = grp.copy()
                    both["Seed"] = both["Seed"].str.replace(r"[ab]$", "", regex=True)
                    both["SeedNum"] = both["Seed"].str[1:3].astype(int)
                    resolved_list.append(both)
                    remaining_playin_games.append((region, seed_num, t1, t2))

            all_resolved = pd.concat(resolved_list, ignore_index=True) if resolved_list else pd.DataFrame()
            season_seeds = pd.concat([main_seeds, all_resolved], ignore_index=True)
            playin_games = remaining_playin_games  # only unresolved (non-16) games

    teams = season_seeds["TeamName"].tolist()

    # Teams present in seed info but missing from team_data (e.g. forfeits, TBD slots)
    # are treated as auto-losses against any real team (prob 0.0) and 0.5 vs each other
    valid_teams  = [t for t in teams if t in season_stats.index and not t.startswith("TBD_")]
    missing_teams = set(teams) - set(valid_teams)
    if missing_teams:
        print(f"  Note: {len(missing_teams)} teams missing from stats (forfeit/TBD): {missing_teams}")

    # --- Build all matchup rows at once, then transform + predict in one shot ---
    pairs        = []    # (i, j, t_a, t_b, flipped, missing)
    matchup_rows = []

    for i, team_a in enumerate(teams):
        for j, team_b in enumerate(teams):
            if i >= j:
                continue

            a_missing = team_a in missing_teams
            b_missing = team_b in missing_teams

            if a_missing or b_missing:
                pairs.append((i, j, team_a, team_b, False, True))
                continue

            # Alphabetical A/B to match training convention
            if team_a <= team_b:
                t_a, t_b, flipped = team_a, team_b, False
            else:
                t_a, t_b, flipped = team_b, team_a, True

            row_a = season_stats.loc[t_a, base_cols]
            row_b = season_stats.loc[t_b, base_cols]

            matchup_rows.append(
                {**{f"{f}_A": row_a[f] for f in base_cols},
                 **{f"{f}_B": row_b[f] for f in base_cols}}
            )
            pairs.append((i, j, t_a, t_b, flipped, False))

    # Single transform + predict call for all valid matchups
    n = len(teams)
    matrix = pd.DataFrame(np.full((n, n), 0.5), index=teams, columns=teams)

    if matchup_rows:
        X_all    = pd.DataFrame(matchup_rows)[paired_cols]
        X_scaled = pd.DataFrame(scaler.transform(X_all), columns=paired_cols)
        probs    = model.predict_proba(X_scaled)[:, 1]

    prob_idx = 0
    for i, j, t_a, t_b, flipped, missing in pairs:
        team_a, team_b = teams[i], teams[j]
        if missing:
            a_missing = team_a in missing_teams
            b_missing = team_b in missing_teams
            if a_missing and not b_missing:
                # Real team beats TBD/forfeit with certainty
                matrix.iloc[i, j] = 0.0
                matrix.iloc[j, i] = 1.0
            elif b_missing and not a_missing:
                matrix.iloc[i, j] = 1.0
                matrix.iloc[j, i] = 0.0
            else:
                # Both missing — coin flip
                matrix.iloc[i, j] = 0.5
                matrix.iloc[j, i] = 0.5
            continue

        p_a = probs[prob_idx]
        prob_idx += 1

        if not flipped:
            matrix.iloc[i, j] = p_a
            matrix.iloc[j, i] = 1.0 - p_a
        else:
            matrix.iloc[i, j] = 1.0 - p_a
            matrix.iloc[j, i] = p_a

    return matrix, season_seeds, model, playin_games


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
    playin_assignments: Optional[Dict[Tuple[str, int], str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run N Monte-Carlo bracket simulations.

    Parameters
    ----------
    playin_assignments : dict, optional
        {(region, seed_num): winner_team_name} for unresolved play-in games.
        When provided, the bracket is built with these specific teams in
        those slots. Teams not assigned are excluded from results.

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

    # If play-in assignments are given, build a filtered seed_info with
    # only the assigned winners in the contested slots
    if playin_assignments:
        # Remove the non-assigned play-in teams from seed_info
        losers = set()
        for (region, seed_num), winner in playin_assignments.items():
            # Find all teams in this slot and mark non-winners as losers
            slot_teams = seed_info[
                (seed_info["Region"] == region) &
                (seed_info["SeedNum"] == seed_num)
            ]["TeamName"].tolist()
            for t in slot_teams:
                if t != winner:
                    losers.add(t)
        sim_seed_info = seed_info[~seed_info["TeamName"].isin(losers)].reset_index(drop=True)
    else:
        sim_seed_info = seed_info

    bracket, _ = build_bracket(sim_seed_info)
    teams = sim_seed_info["TeamName"].tolist()

    # Find any TBD slots in the bracket not present in prob_matrix
    # (e.g. VCU forfeit in 2021 creates "TBD_X_10")
    all_bracket_teams = {
        team
        for matchups in bracket.values()
        for (t1, _), (t2, _) in matchups
        for team in (t1, t2)
    }
    tbd_teams = [t for t in all_bracket_teams if t not in prob_matrix.index]
    if tbd_teams:
        # Add TBD teams to matrix: 0.0 probability of beating anyone (auto-loss)
        for tbd in tbd_teams:
            prob_matrix[tbd] = 1.0       # all real teams beat TBD with prob 1.0
            prob_matrix.loc[tbd] = 0.0   # TBD beats nobody
            prob_matrix.loc[tbd, tbd] = 0.5

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

    seed_lookup = sim_seed_info.set_index("TeamName")[["Seed", "Region", "SeedNum"]]
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
    team_data: pd.DataFrame,
    model_kwargs: Optional[dict] = None,
    n_sims: int = 1000,
    random_state: Optional[int] = 42,
) -> Tuple[Dict, pd.DataFrame, MixtureOfExperts, List]:
    """
    Full pipeline: train MoE -> build NxN matrix -> simulate -> return results.

    For past seasons (all play-ins known), returns one simulation.
    For future seasons with unresolved non-16-seed play-ins, returns one
    simulation per combination of play-in winners.

    Returns
    -------
    results : dict
        { playin_key: advancement_df }
        playin_key is a tuple of winner names for the variable play-in
        games, or () if no variable play-ins.
    prob_matrix : pd.DataFrame
        NxN P(row beats col) matrix (includes all play-in teams).
    model : MixtureOfExperts
        Fitted model.
    playin_games : list
        List of (region, seed_num, team_a_name, team_b_name) for
        unresolved play-in games. Empty for past seasons.
    """
    print(f"[{season}] Building probability matrix (training on all other years)...")
    prob_matrix, seed_info, model, playin_games = build_prob_matrix(
        season=season,
        df=df,
        seed_df=seed_df,
        team_data=team_data,
        model_kwargs=model_kwargs,
    )
    n = prob_matrix.shape[0]
    print(f"[{season}] Matrix built — {n} teams, {n*(n-1)//2} unique matchup probabilities.")

    # Build all combinations of unresolved play-in winners
    if playin_games:
        from itertools import product as _product
        combos = list(_product(*[(t1, t2) for (_, _, t1, t2) in playin_games]))
    else:
        combos = [()]

    print(f"[{season}] Simulating {n_sims:,} tournaments × {len(combos)} play-in combo(s)...")

    results = {}
    for combo in combos:
        # Build playin_assignments: {(region, seed_num): winner_name}
        if combo:
            assignments = {}
            for i, (region, seed_num, t1, t2) in enumerate(playin_games):
                assignments[(region, seed_num)] = combo[i]
        else:
            assignments = None

        advancement_df, _ = simulate_tournament(
            prob_matrix=prob_matrix.copy(),
            seed_info=seed_info,
            n_sims=n_sims,
            random_state=random_state,
            playin_assignments=assignments,
        )
        results[combo] = advancement_df

    print(f"[{season}] Done — {len(results)} simulation(s) complete.")

    return results, prob_matrix, model, playin_games


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def save_advancement_csv(
    all_results: Dict[int, Tuple[Dict, List]],
    path: str = "adv_all.csv",
    playin_meta_path: str = "playin_meta.csv",
):
    """
    Save advancement DataFrames and play-in metadata to CSV.

    Parameters
    ----------
    all_results : dict
        { season: (results_dict, playin_games_list) }
        results_dict: { combo_tuple: advancement_df }
        playin_games_list: [(region, seed_num, team_a, team_b), ...]
    path : str
        Output path for the main advancement CSV.
    playin_meta_path : str
        Output path for play-in game metadata CSV.
    """
    import json

    adv_frames = []
    meta_rows = []

    for season, (results, playin_games) in all_results.items():
        # Save play-in metadata
        for region, seed_num, t1, t2 in playin_games:
            meta_rows.append({
                "Season": int(season),
                "Region": region,
                "SeedNum": seed_num,
                "TeamA": t1,
                "TeamB": t2,
            })

        for combo_key, adv_df in results.items():
            frame = adv_df.reset_index()  # TeamName index -> column
            # Ensure TeamName column exists (index name varies)
            if "TeamName" not in frame.columns and frame.columns[0] not in ("Seed", "Region", "SeedNum"):
                frame = frame.rename(columns={frame.columns[0]: "TeamName"})
            frame["Season"] = int(season)
            # PlayinKey: JSON-serialized list of winner names, or "" if none
            frame["PlayinKey"] = json.dumps(list(combo_key)) if combo_key else ""
            adv_frames.append(frame)

    combined = pd.concat(adv_frames, ignore_index=True)
    combined.to_csv(path, index=False)
    print(f"Saved {path} ({len(combined)} rows)")

    if meta_rows:
        meta_df = pd.DataFrame(meta_rows)
        meta_df.to_csv(playin_meta_path, index=False)
        print(f"Saved {playin_meta_path} ({len(meta_rows)} play-in games)")
    else:
        # Write empty file so app doesn't error
        pd.DataFrame(columns=["Season", "Region", "SeedNum", "TeamA", "TeamB"]).to_csv(
            playin_meta_path, index=False
        )
        print(f"Saved empty {playin_meta_path} (no unresolved play-in games)")