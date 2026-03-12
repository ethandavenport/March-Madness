"""
fill_bracket.py
===============
Fills out a complete March Madness bracket for a given year.

Inputs
------
year            : str or int, tournament year (e.g. "2024")
team_data       : DataFrame (Year, TeamID, Team, <stat cols>)
                  — same one built in Cell 32 of the notebook
mm_seeds_full   : DataFrame (Season, Seed, TeamID, TeamName)
                  — includes play-in seeds like "W16a", "W16b"
logit           : fitted LogisticRegression from Cell 97 (seed baseline)
seed_probs      : DataFrame (16x16 matrix of seed win probabilities, %)
                  — built in Cell 101 of the notebook
lift_thresholds : dict {'Round 1': float, ...} from Cell 114
model_kwargs    : dict of MixtureOfExperts hyperparameters
                  (alpha, n_experts, n_features, C_expert, C_meta)
df              : the full historical matchup DataFrame used to train the model
                  — needed for training; stats come from team_data not df

Play-in handling
----------------
Seeds like "W16a" / "W16b" indicate play-in games. There are 4 such matchups
(2 for the 16-seeds, 2 for the 11-seeds in recent years), producing 16 possible
combinations of winners. fill_bracket returns a dict:
    { (winner1_id, winner2_id, winner3_id, winner4_id) : bracket_df }
Keys are tuples of 4 TeamIDs sorted by region (W < X < Y < Z) then seed number.

Output df columns (63 rows, one per game)
------------------------------------------
Round, Season, ATeamName, BTeamName, ATeamID, BTeamID,
Seed_A, Seed_B, <stat>_A, <stat>_B, ...,
AProb, FProb, SProb, Lift, Selected
"""

from __future__ import annotations
from itertools import product
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from moe_classifier import MixtureOfExperts, split_n_scale, lasso_cols

# ---------------------------------------------------------------------------
# Bracket structure constants
# ---------------------------------------------------------------------------

# First-round seed matchups within each region (fav_seed, dog_seed)
_R1_SEED_PAIRS = [(1,16), (8,9), (5,12), (4,13), (6,11), (3,14), (7,10), (2,15)]

# How R1 winners fold into R2: indices into the 8-element R1 winner list
_R2_FOLD = [(0,1), (2,3), (4,5), (6,7)]

# How R2 winners fold into Sweet 16
_R3_FOLD = [(0,1), (2,3)]

# How S16 winners fold into Elite 8
_R4_FOLD = [(0,1)]

# Final Four pairings by region
_FF_PAIRS = [("W", "X"), ("Y", "Z")]

ROUND_LABELS = [
    "Round 1",
    "Round 2",
    "Round 3 (Sweet Sixteen)",
    "Round 4 (Elite Eight)",
    "Final Four",
    "Championship",
]
UPSET_ROUNDS = set(ROUND_LABELS[:4])

EPSILON = 1.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_seed_num(seed_str: str) -> int:
    """Extract numeric seed from strings like 'W01', 'W16a', 'Y11b'."""
    return int("".join(c for c in seed_str[1:] if c.isdigit()))


def _sprob_from_matrix(seed_a: int, seed_b: int, seed_probs: pd.DataFrame) -> float:
    """
    Seed-baseline P(favorite wins) from the seed_probs matrix.
    seed_probs values are in %, index/columns are seed numbers 1-16.
    Returns probability as a float in [0,1].
    Favorite = lower seed number; value is always >= 0.5.
    """
    fav = min(seed_a, seed_b)
    dog = max(seed_a, seed_b)
    raw = seed_probs.loc[fav, dog]   # P(fav wins) in %
    return float(raw) / 100.0


def _fprob(prob_a: float, seed_a: int, seed_b: int) -> float:
    """Convert P(A wins) to P(favorite wins). Favorite = lower seed."""
    if seed_a <= seed_b:
        return prob_a
    else:
        return 1.0 - prob_a


def _lift(fprob: float, sprob: float) -> float:
    """Log-odds lift of model vs seed baseline (epsilon=1)."""
    return (
        np.log((EPSILON + fprob)  / (EPSILON + 1.0 - fprob)) -
        np.log((EPSILON + sprob)  / (EPSILON + 1.0 - sprob))
    )


def _stat_cols(team_data: pd.DataFrame) -> list:
    """Return the stat column names from team_data (everything except Year/TeamID/Team)."""
    return [c for c in team_data.columns if c not in ("Year", "TeamID", "Team")]


def _build_matchup_rows(
    pairs: list[tuple],          # list of (TeamID_a, TeamID_b) — A already alphabetical
    seed_lookup: pd.DataFrame,   # indexed by TeamID: Seed, SeedNum, TeamName
    stats_lookup: pd.DataFrame,  # indexed by TeamID: stat cols
    stat_cols: list,
    season,
    round_label: str,
) -> pd.DataFrame:
    """
    Build a matchup DataFrame in the same format as df.
    Each pair is (tid_a, tid_b) where A is alphabetically first.
    Returns DataFrame with columns matching df exactly.
    """
    rows = []
    for tid_a, tid_b in pairs:
        name_a = str(seed_lookup.loc[tid_a, "TeamName"]) if pd.notna(seed_lookup.loc[tid_a, "TeamName"]) else str(tid_a)
        name_b = str(seed_lookup.loc[tid_b, "TeamName"]) if pd.notna(seed_lookup.loc[tid_b, "TeamName"]) else str(tid_b)
        seed_a = int(seed_lookup.loc[tid_a, "SeedNum"])
        seed_b = int(seed_lookup.loc[tid_b, "SeedNum"])

        stats_a = stats_lookup.loc[tid_a, stat_cols]
        stats_b = stats_lookup.loc[tid_b, stat_cols]

        row = {
            "Season":    season,
            "ATeamName": name_a,
            "BTeamName": name_b,
            "ATeamID":   tid_a,
            "BTeamID":   tid_b,
            "Region_A":  seed_lookup.loc[tid_a, "Region"],
            "Region_B":  seed_lookup.loc[tid_b, "Region"],
            "Seed_A":    float(seed_a),
            "Seed_B":    float(seed_b),
        }

        for c in stat_cols:
            row[f"{c}_A"] = stats_a[c]
            row[f"{c}_B"] = stats_b[c]

        rows.append(row)

    result = pd.DataFrame(rows)
    for c in result.columns:
        if c not in ("Season", "ATeamName", "BTeamName", "ATeamID", "BTeamID", "Region_A", "Region_B"):
            result[c] = result[c].astype("float64")
    return result.reset_index(drop=True)


def _alphabetical_pair(tid1, tid2, seed_lookup):
    n1 = seed_lookup.loc[tid1, "TeamName"]
    n2 = seed_lookup.loc[tid2, "TeamName"]
    n1 = str(tid1) if pd.isna(n1) else str(n1)
    n2 = str(tid2) if pd.isna(n2) else str(n2)
    return (tid1, tid2) if n1 <= n2 else (tid2, tid1)


def _predict_and_annotate(
    matchup_df: pd.DataFrame,
    feat_cols: list,
    scaler: StandardScaler,
    model: MixtureOfExperts,
    seed_probs: pd.DataFrame,
    lift_thresholds: dict,
    round_label: str,
    seed_lookup: pd.DataFrame,
) -> tuple[pd.DataFrame, list]:
    """
    Scale, predict, compute FProb/SProb/Lift, pick winners.
    Returns (annotated_df, list_of_winner_TeamIDs).
    """
    X = matchup_df[feat_cols].astype(float)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feat_cols)
    probs_a = model.predict_proba(X_scaled)[:, 1]

    matchup_df = matchup_df.copy()
    matchup_df["AProb"] = probs_a

    # FProb: P(favorite wins)
    matchup_df["FProb"] = np.where(
        matchup_df["Seed_A"] <= matchup_df["Seed_B"],
        matchup_df["AProb"],
        1.0 - matchup_df["AProb"],
    )

    # SProb: seed-baseline P(favorite wins) from seed_probs matrix
    matchup_df["SProb"] = matchup_df.apply(
        lambda r: _sprob_from_matrix(int(r["Seed_A"]), int(r["Seed_B"]), seed_probs),
        axis=1,
    )

    # Lift
    matchup_df["Lift"] = matchup_df.apply(
        lambda r: _lift(r["FProb"], r["SProb"]), axis=1
    )

    # Pick winners
    threshold = lift_thresholds.get(round_label) if round_label in UPSET_ROUNDS else None

    winner_tids = []
    selected_names = []
    for _, row in matchup_df.iterrows():
        tid_a = row["ATeamID"]
        tid_b = row["BTeamID"]
        fav_tid = tid_a if row["Seed_A"] <= row["Seed_B"] else tid_b
        dog_tid = tid_b if fav_tid == tid_a else tid_a

        fav_seed = min(int(row["Seed_A"]), int(row["Seed_B"]))
        dog_seed = max(int(row["Seed_A"]), int(row["Seed_B"]))
        never_upset = (fav_seed == 1 and dog_seed == 16)

        if threshold is not None and row["Lift"] < threshold and not never_upset:
            winner_tid = dog_tid
        else:
            winner_tid = fav_tid

        winner_tids.append(winner_tid)
        selected_names.append(seed_lookup.loc[winner_tid, "TeamName"])

    matchup_df["Selected"] = selected_names
    matchup_df["Round"] = round_label
    return matchup_df, winner_tids


def _run_bracket(
    region_order: list,
    seed_lookup: pd.DataFrame,
    team_map: dict,        # (region, seed_num) -> TeamID
    stats_lookup: pd.DataFrame,
    stat_cols: list,
    feat_cols: list,
    scaler: StandardScaler,
    model: MixtureOfExperts,
    seed_probs: pd.DataFrame,
    lift_thresholds: dict,
    season,
) -> pd.DataFrame:
    """
    Run a full bracket simulation for one play-in outcome.
    Returns a 63-row DataFrame.
    """
    all_dfs = []

    # ------------------------------------------------------------------
    # Rounds 1-4: within-region
    # ------------------------------------------------------------------
    region_champs = {}   # region -> TeamID

    for region in region_order:
        # Build R1 pairs for this region
        r1_pairs = [
            _alphabetical_pair(team_map[(region, s1)], team_map[(region, s2)], seed_lookup)
            for s1, s2 in _R1_SEED_PAIRS
        ]

        matchup_df = _build_matchup_rows(r1_pairs, seed_lookup, stats_lookup,
                                         stat_cols, season, "Round 1")
        matchup_df, r1_winners = _predict_and_annotate(
            matchup_df, feat_cols, scaler, model, seed_probs,
            lift_thresholds, "Round 1", seed_lookup
        )
        all_dfs.append(matchup_df)

        # R2
        r2_pairs = [_alphabetical_pair(r1_winners[a], r1_winners[b], seed_lookup)
                    for a, b in _R2_FOLD]
        matchup_df = _build_matchup_rows(r2_pairs, seed_lookup, stats_lookup,
                                         stat_cols, season, "Round 2")
        matchup_df, r2_winners = _predict_and_annotate(
            matchup_df, feat_cols, scaler, model, seed_probs,
            lift_thresholds, "Round 2", seed_lookup
        )
        all_dfs.append(matchup_df)

        # Sweet 16
        r3_pairs = [_alphabetical_pair(r2_winners[a], r2_winners[b], seed_lookup)
                    for a, b in _R3_FOLD]
        matchup_df = _build_matchup_rows(r3_pairs, seed_lookup, stats_lookup,
                                         stat_cols, season, "Round 3 (Sweet Sixteen)")
        matchup_df, r3_winners = _predict_and_annotate(
            matchup_df, feat_cols, scaler, model, seed_probs,
            lift_thresholds, "Round 3 (Sweet Sixteen)", seed_lookup
        )
        all_dfs.append(matchup_df)

        # Elite 8
        r4_pairs = [_alphabetical_pair(r3_winners[a], r3_winners[b], seed_lookup)
                    for a, b in _R4_FOLD]
        matchup_df = _build_matchup_rows(r4_pairs, seed_lookup, stats_lookup,
                                         stat_cols, season, "Round 4 (Elite Eight)")
        matchup_df, r4_winners = _predict_and_annotate(
            matchup_df, feat_cols, scaler, model, seed_probs,
            lift_thresholds, "Round 4 (Elite Eight)", seed_lookup
        )
        all_dfs.append(matchup_df)

        region_champs[region] = r4_winners[0]

    # ------------------------------------------------------------------
    # Final Four
    # ------------------------------------------------------------------
    ff_pairs = [
        _alphabetical_pair(region_champs[ra], region_champs[rb], seed_lookup)
        for ra, rb in _FF_PAIRS
        if ra in region_champs and rb in region_champs
    ]
    matchup_df = _build_matchup_rows(ff_pairs, seed_lookup, stats_lookup,
                                     stat_cols, season, "Final Four")
    matchup_df, ff_winners = _predict_and_annotate(
        matchup_df, feat_cols, scaler, model, seed_probs,
        lift_thresholds, "Final Four", seed_lookup
    )
    all_dfs.append(matchup_df)

    # ------------------------------------------------------------------
    # Championship
    # ------------------------------------------------------------------
    champ_pair = [_alphabetical_pair(ff_winners[0], ff_winners[1], seed_lookup)]
    matchup_df = _build_matchup_rows(champ_pair, seed_lookup, stats_lookup,
                                     stat_cols, season, "Championship")
    matchup_df, _ = _predict_and_annotate(
        matchup_df, feat_cols, scaler, model, seed_probs,
        lift_thresholds, "Championship", seed_lookup
    )
    all_dfs.append(matchup_df)

    result = pd.concat(all_dfs, ignore_index=True)
    result["Round"] = pd.Categorical(result["Round"], categories=ROUND_LABELS, ordered=True)
    return result.sort_values(["Round", "ATeamName"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fill_bracket(
    year,
    df: pd.DataFrame,
    team_data: pd.DataFrame,
    mm_seeds_full: pd.DataFrame,
    logit,              # kept for API compatibility but seed_probs is used instead
    seed_probs: pd.DataFrame,
    lift_thresholds: dict,
    model_kwargs: Optional[dict] = None,
) -> dict:
    """
    Fill out all possible brackets for `year` (one per play-in outcome).

    Returns
    -------
    dict  { (tid1, tid2, tid3, tid4) : bracket_df }
          Keys are tuples of 4 play-in winner TeamIDs, sorted W->X->Y->Z
          then by seed number. If there are no play-in games, the dict
          has a single key of an empty tuple: { (): bracket_df }.
    """

    # --- Normalise types ---
    season = str(year)
    mm_seeds_full = mm_seeds_full.copy()
    mm_seeds_full["Season"]  = mm_seeds_full["Season"].astype(str)
    mm_seeds_full["TeamID"]  = mm_seeds_full["TeamID"].astype(str)
    team_data = team_data.copy()
    team_data["Year"]   = team_data["Year"].astype(str)
    team_data["TeamID"] = team_data["TeamID"].astype(str)
    df = df.copy()
    df["Season"]   = df["Season"].astype(str)
    df["ATeamID"]  = df["ATeamID"].astype(str)
    df["BTeamID"]  = df["BTeamID"].astype(str)

    # --- Filter mm_seeds_full for this year ---
    seeds_year = mm_seeds_full[mm_seeds_full["Season"] == season].copy()
    seeds_year["Region"]  = seeds_year["Seed"].str[0]
    seeds_year["SeedNum"] = seeds_year["Seed"].apply(_to_seed_num)
    seeds_year = seeds_year.dropna(subset=["TeamID"]).reset_index(drop=True)

    # --- Find play-in seeds (those with 'a'/'b' suffix) ---
    playin_mask = seeds_year["Seed"].str.contains(r"[ab]$", regex=True)
    playin_seeds = seeds_year[playin_mask].copy()
    main_seeds   = seeds_year[~playin_mask].copy()

    # Group play-in pairs: same region + same seed number = one game
    playin_games = []   # list of [tid1, tid2, region, seed_num]
    for (region, seed_num), grp in playin_seeds.groupby(["Region", "SeedNum"]):
        tids = grp["TeamID"].tolist()
        if len(tids) == 2:
            playin_games.append((region, seed_num, tids[0], tids[1]))

    # --- Build team_data stats lookup for this year ---
    # Use team_data filtered to this year (the data represents current-season stats)
    td_year = team_data[team_data["Year"] == season].copy()
    stat_cols = _stat_cols(team_data)

    # --- Build feature column list from df (preserves df column order) ---
    meta = {"Season", "ATeamName", "BTeamName", "ATeamID", "BTeamID", "AWon"}
    feat_cols = [c for c in df.columns if c not in meta]
    # Stat base cols: those that exist in both df (_A suffix) and team_data
    base_cols_in_df   = [c[:-2] for c in feat_cols if c.endswith("_A")]
    stat_cols_to_use  = [c for c in base_cols_in_df if c in stat_cols]
    # Rebuild feat_cols to only include cols backed by team_data + Seed
    feat_cols = []
    for c in stat_cols_to_use:
        feat_cols.extend([f"{c}_A", f"{c}_B"])
    # Insert Seed_A / Seed_B at the position they appear in df
    all_feat = [c for c in df.columns if c not in meta]
    seed_feat = [c for c in all_feat if c in ("Seed_A", "Seed_B")]
    feat_cols = seed_feat + [c for c in feat_cols if c not in seed_feat]
    # Reorder to match df column order
    feat_cols = [c for c in all_feat if c in feat_cols]

    # --- Train MoE on df excluding this year ---
    print(f"[{season}] Training model...")
    train_df = df[df["Season"] != season].copy()
    X_tr, _, y_tr, _, _, _ = split_n_scale(train_df)

    model = MixtureOfExperts(**(model_kwargs or {}))
    model.fit(X_tr, y_tr)

    scaler = StandardScaler()
    scaler.fit(train_df[feat_cols].astype(float))
    print(f"[{season}] Model trained.")

    # --- Build seed_lookup (indexed by TeamID) for main seeds ---
    seed_lookup_base = main_seeds.set_index("TeamID")[["TeamName", "Seed", "SeedNum", "Region"]]

    # # --- Generate all play-in outcomes ---
    # # Each play-in game produces 2 possible winners
    # if playin_games:
    #     playin_options = list(product(*[(g[2], g[3]) for g in playin_games]))
    # else:
    #     playin_options = [()]

    # Known play-in winners for past tournaments
    known_playin_winners = {
        "2017": ["1243", "1291", "1413", "1425"],
        "2018": ["1347", "1382", "1393", "1411"],
        "2019": ["1113", "1125", "1192", "1295"],
        "2021": ["1179", "1313", "1411", "1417"],
        "2022": ["1231", "1323", "1411", "1460"],
        "2023": ["1113", "1192", "1338", "1394"],
        "2024": ["1160", "1161", "1212", "1447"],
        "2025": ["1106", "1291", "1314", "1462"],
    }

    if playin_games and season in known_playin_winners:
        # For past years, only simulate the one bracket with actual winners
        known_winners = known_playin_winners[season]
        single_option = tuple(
            known_winners[i] for i, _ in enumerate(playin_games)
            # match each play-in game to its known winner
        )
        # More robustly: for each play-in game, pick whichever of its two teams is a known winner
        single_option = tuple(
            g[2] if g[2] in known_winners else g[3]
            for g in playin_games
        )
        playin_options = [single_option]
    else:
        playin_options = list(product(*[(g[2], g[3]) for g in playin_games])) if playin_games else [()]

    results = {}

    for option in playin_options:
        # Build seed_lookup for this combination of play-in winners
        sl = seed_lookup_base.copy()

        # Add play-in winners: they slot into the main seed slot
        for i, (region, seed_num, tid1, tid2) in enumerate(playin_games):
            winner_tid = option[i]
            loser_tid  = tid2 if winner_tid == tid1 else tid1

            # Add winner to seed_lookup with the plain seed (e.g. "W16")
            winner_row = playin_seeds[playin_seeds["TeamID"] == winner_tid].iloc[0]
            sl.loc[winner_tid] = {
                "TeamName": winner_row["TeamName"],
                "Seed":     f"{region}{seed_num:02d}",
                "SeedNum":  seed_num,
                "Region":   region,
            }

        # Build team_map: (region, seed_num) -> TeamID
        team_map = {
            (r["Region"], int(r["SeedNum"])): tid
            for tid, r in sl.iterrows()
        }

        # Stats lookup: filter td_year to tournament teams only
        tourney_tids = set(sl.index.tolist())
        stats_lookup = (td_year[td_year["TeamID"].isin(tourney_tids)]
                .drop_duplicates("TeamID")
                .set_index("TeamID"))

        missing = tourney_tids - set(stats_lookup.index)
        if missing:
            missing_names = [sl.loc[t, "TeamName"] for t in missing if t in sl.index]
            print(f"  WARNING: teams missing from team_data for {season}: {missing_names}")

        regions = sorted(sl["Region"].unique())

        bracket = _run_bracket(
            region_order=regions,
            seed_lookup=sl,
            team_map=team_map,
            stats_lookup=stats_lookup,
            stat_cols=stat_cols_to_use,
            feat_cols=feat_cols,
            scaler=scaler,
            model=model,
            seed_probs=seed_probs,
            lift_thresholds=lift_thresholds,
            season=season,
        )

        # Key: tuple of play-in winner TeamIDs sorted W->X->Y->Z then seed num
        if option:
            key_tids = sorted(
                option,
                key=lambda t: (sl.loc[t, "Region"], sl.loc[t, "SeedNum"])
            )
            key = tuple(key_tids)
        else:
            key = ()

        results[key] = bracket

    return results
