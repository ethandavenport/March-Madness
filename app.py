import streamlit as st
import pandas as pd
import base64
import os
import glob
from fill_bracket import SLOT_GAME_MAP

st.set_page_config(page_title="March Madness", layout="wide")

# ── Discover available bracket files (bracket_YYYY.csv) ──────────────────────
_bracket_files = {}
for f in sorted(glob.glob("bracket_*.csv")):
    name = os.path.basename(f)
    # Match bracket_2017.csv .. bracket_2025.csv, skip bracket_all.csv etc.
    if name.startswith("bracket_") and name[8:-4].isdigit():
        yr = int(name[8:-4])
        _bracket_files[yr] = f

if not _bracket_files:
    st.error("No bracket files found (expected bracket_YYYY.csv).")
    st.stop()

_bracket_years = sorted(_bracket_files.keys(), reverse=True)

ROUND_ORDER = [
    "Round 1",
    "Round 2",
    "Round 3 (Sweet Sixteen)",
    "Round 4 (Elite Eight)",
    "Final Four",
    "Championship",
]
ROUND_SHORT = {
    "Round 1":                  "R64",
    "Round 2":                  "R32",
    "Round 3 (Sweet Sixteen)":  "S16",
    "Round 4 (Elite Eight)":    "E8",
    "Final Four":               "FF",
    "Championship":             "Champion",
}

# ── Region layout per year ────────────────────────────────────────────────────
# Order: (TL, BL, TR, BR) — matching ESPN/CBS bracket layout.
# W always plays X in FF, Y always plays Z in FF.
# Left side = TL + BL, right side = TR + BR.
# Within each side, first = top region, second = bottom region.
_REGION_LAYOUTS = {
    2017: ("W", "X", "Y", "Z"),
    2018: ("Y", "Z", "W", "X"),
    2019: ("W", "X", "Z", "Y"),
    2021: ("X", "W", "Z", "Y"),
    2022: ("X", "W", "Z", "Y"),
    2023: ("X", "W", "Y", "Z"),
    2024: ("W", "X", "Z", "Y"),
    2025: ("Y", "Z", "W", "X"),
    2026: ("W", "X", "Z", "Y"),
}
# Default layout if year not in map
_DEFAULT_LAYOUT = ("W", "X", "Y", "Z")


def _get_layout(year):
    """Return (TL, BL, TR, BR) region layout for a given year."""
    return _REGION_LAYOUTS.get(year, _DEFAULT_LAYOUT)


@st.cache_data(ttl=60)
def _load_bracket_year(year):
    """Load bracket_YYYY.csv for a single year and build caches.
    Returns dict: { playin_key_str: (bracket_df, shap_cache, results_cache) }
    For years with no play-in choices, the only key is "".
    """
    path = _bracket_files[year]
    raw = pd.read_csv(path)
    raw["Round"] = pd.Categorical(raw["Round"], categories=ROUND_ORDER, ordered=True)
    raw = raw.sort_values("Round")

    if "PlayinKey" not in raw.columns:
        raw["PlayinKey"] = ""
    raw["PlayinKey"] = raw["PlayinKey"].fillna("")

    result = {}
    for pk in raw["PlayinKey"].unique():
        df = raw[raw["PlayinKey"] == pk].copy()

        # SHAP cache
        if "SHAPPlot" in df.columns:
            sc = (
                df[["MatchID", "SHAPPlot"]]
                .dropna(subset=["SHAPPlot"])
                .set_index("MatchID")["SHAPPlot"]
                .to_dict()
            )
        else:
            sc = {}

        # Results cache
        has_results = "ActualA" in df.columns
        rc_cols = ["MatchID","ATeamID","BTeamID",
                   "ActualA","ActualASeed","ActualATid",
                   "ActualB","ActualBSeed","ActualBTid"]
        for wc in ["ActualWinner","ActualWinnerSeed","ActualWinnerTid"]:
            if wc in df.columns:
                rc_cols.append(wc)
        if has_results:
            rc = (
                df[rc_cols]
                .set_index("MatchID")
                .to_dict(orient="index")
            )
        else:
            rc = {}

        result[pk] = (df, sc, rc)

    return result


# Initialise with the most recent year (will be overridden in bracket tab)
_init_bracket_data = _load_bracket_year(_bracket_years[0])
_init_key = list(_init_bracket_data.keys())[0]
bracket, shap_cache, results_cache = _init_bracket_data[_init_key]

# ── Helpers ────────────────────────────────────────────────────────────────────

def _tid(v):
    """Normalise a TeamID to a clean string like '1181' (no '.0' suffix)."""
    if v is None:
        return None
    if isinstance(v, float):
        if pd.isna(v):
            return None
        return str(int(v))
    s = str(v).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s

def prob_color(p):
    if pd.isna(p):
        return "#aaa"
    r = int(210 * (1 - p))
    g = int(180 * p)
    return f"rgb({r},{g},40)"

def get_winner_seed(row):
    return int(row["Seed_A"]) if row["Selected"] == row["ATeamName"] else int(row["Seed_B"])

# ── Bracket slot system ────────────────────────────────────────────────────────
#
# Slot indices 0-7 correspond to R1 matchups in top-to-bottom order:
#   0: 1v16,  1: 8v9,  2: 5v12,  3: 4v13
#   4: 6v11,  5: 3v14, 6: 7v10,  7: 2v15
#
# Each team in a region gets a slot_position (float):
#   Top team of slot i  → i + 0.0
#   Bot team of slot i  → i + 0.5
#
# This position is carried forward through the bracket.
# In each matchup the team with the LOWER slot_position appears on top.

R1_TOP_SEEDS = [1, 8, 5, 4, 6, 3, 7, 2]  # lower seed of each R1 slot (slot 0..7)


def build_slot_positions(region):
    """
    Returns dict: team_name -> float slot_position (0.0 – 7.5)
    based on which R1 game and position they occupied.
    """
    r1 = bracket[
        ((bracket["Region_A"] == region) | (bracket["Region_B"] == region)) &
        (bracket["Round"] == "Round 1")
    ].reset_index(drop=True)

    pos = {}
    for slot_idx, top_seed in enumerate(R1_TOP_SEEDS):
        for _, row in r1.iterrows():
            sa, sb = int(row["Seed_A"]), int(row["Seed_B"])
            if min(sa, sb) == top_seed:
                # lower seed → top of slot, higher seed → bottom
                if sa < sb:
                    pos[row["ATeamName"]] = slot_idx + 0.0
                    pos[row["BTeamName"]] = slot_idx + 0.5
                else:
                    pos[row["BTeamName"]] = slot_idx + 0.0
                    pos[row["ATeamName"]] = slot_idx + 0.5
                break
    return pos


def games_for_ordered(region, round_name, slot_pos):
    """
    Return list of game-dicts for a region+round, sorted by bracket slot order.
    Each dict has: row, top_name, top_seed, bot_name, bot_seed, sort_key.
    top = team with lower slot_position (came from higher up in bracket).
    """
    mask = (
        ((bracket["Region_A"] == region) | (bracket["Region_B"] == region)) &
        (bracket["Round"] == round_name)
    )
    games = bracket[mask].copy()

    result = []
    for _, row in games.iterrows():
        pa = slot_pos.get(row["ATeamName"], 999)
        pb = slot_pos.get(row["BTeamName"], 999)
        if pa <= pb:
            top_name, top_seed = row["ATeamName"], int(row["Seed_A"])
            bot_name, bot_seed = row["BTeamName"], int(row["Seed_B"])
            top_tid  = _tid(row["ATeamID"])
            bot_tid  = _tid(row["BTeamID"])
            sort_key = pa
        else:
            top_name, top_seed = row["BTeamName"], int(row["Seed_B"])
            bot_name, bot_seed = row["ATeamName"], int(row["Seed_A"])
            top_tid  = _tid(row["BTeamID"])
            bot_tid  = _tid(row["ATeamID"])
            sort_key = pb
        result.append({
            "row": row,
            "top_name": top_name, "top_seed": top_seed, "top_tid": top_tid,
            "bot_name": bot_name, "bot_seed": bot_seed, "bot_tid": bot_tid,
            "sort_key": sort_key,
        })

    result.sort(key=lambda x: x["sort_key"])
    return result


# ── Layout constants ───────────────────────────────────────────────────────────
GAME_H  = 80   # px height of one game card
GAME_MB = 3    # margin-bottom on .game
COL_PAD = 20   # px padding on the "inner" side of each round-col (connector lives here)
# The vertical bar of connectors is drawn at x = COL_PAD/2, centered in the gap between columns.

R1_TOTAL = 8 * GAME_H + 7 * GAME_MB  # total column height = 663px


def game_center_y(slot_idx, n):
    """Pixel y-center of game at slot_idx when n games fill R1_TOTAL."""
    sh = R1_TOTAL / n
    spacer = (sh - GAME_H) / 2
    return slot_idx * sh + spacer + GAME_H / 2


# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"] { background: #f5f3ef; color: #1a1a1a; }
[data-testid="stAppViewContainer"] { padding: 0; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding: 2rem 0.25rem 4rem 0.25rem; max-width: 100%; }

h1 {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em;
    color: #c97b00 !important;
    margin: 0 0 0.1em 0;
    line-height: 1;
    text-align: center;
    text-transform: uppercase;
}
.subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    color: #888;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    text-align: center;
    margin: 0;
    line-height: 1.6;
}
.subtitle-block {
    margin-bottom: 1.5rem;
    text-align: center;
}

/* ── Round headers ── */
.round-headers-row {
    display: flex;
    align-items: flex-end;
    margin-bottom: 4px;
}
.round-headers-left  { display: flex; flex: 1; min-width: 0; }
.round-headers-right { display: flex; flex: 1; min-width: 0; flex-direction: row-reverse; }
/* Header cells: border-bottom only under the content area, gaps between rounds */
.round-header-cell {
    flex: 1;
    min-width: 0;
    box-sizing: border-box;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 0.80rem;
    letter-spacing: 0.10em;
    color: #999;
    text-align: center;
    padding-bottom: 5px;
    /* no border-bottom here — applied via inline style in Python to match box width */
}

/* ── Main bracket wrapper ── */
.bracket-wrapper {
    display: flex;
    align-items: stretch;
    width: 100%;
    position: relative;  /* needed for absolute champ-col */
}
.side-half {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 3px;
}
.region-block { min-width: 0; }
.rounds-row     { display: flex; align-items: flex-start; }
.rounds-row.rtl { flex-direction: row-reverse; }

.round-col {
    flex: 1;
    min-width: 0;
    position: relative;
    box-sizing: border-box;
}

.game-spacer { flex-shrink: 0; }

.game {
    background: #ffffff;
    border: 1px solid #ddd9d2;
    border-radius: 6px;
    overflow: visible;
    margin-bottom: 0;
    flex-shrink: 0;
    transition: border-color 0.15s, box-shadow 0.15s;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    position: relative;
}
.game:hover {
    border-color: #c97b0077;
    box-shadow: 0 2px 8px rgba(201,123,0,0.10);
    z-index: 1000;
}

/* ── Wrong-prediction styling ── */
.team-name.wrong   { text-decoration: line-through; color: #ccc; }
.team-name.correct { color: #2e7d32; font-weight: 700; }
.actual-winner {
    position: absolute;
    left: 0; right: 0;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    color: #d32f2f;
    padding: 1px 7px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    z-index: 1;
}
.actual-winner.above { bottom: 100%; padding-bottom: 2px; }
.actual-winner.below { top: 100%;    padding-top: 2px; }

/* ── SHAP tooltip — pure CSS, no JS needed ── */
.shap-tooltip {
    display: none;
    position: absolute;
    left: 105%;
    top: 50%;
    transform: translateY(-50%);
    width: 560px;
    background: #fff;
    border: 1px solid #ddd9d2;
    border-radius: 10px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.22);
    padding: 8px;
    pointer-events: none;
    z-index: 9999;
}
.shap-tooltip.tip-left {
    left: auto;
    right: 105%;
}
.shap-tooltip img {
    width: 100%;
    height: auto;
    display: block;
    border-radius: 6px;
}
.game:hover .shap-tooltip { display: block; }

.team {
    display: grid;
    grid-template-columns: 16px 1fr auto auto;
    align-items: center;
    padding: 5px 7px;
    gap: 5px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem;
    color: #333;
    border-bottom: 1px solid #eee9e2;
    min-height: 27px;
    overflow: hidden;
}
.team:last-child { border-bottom: none; }
.seed { font-size: 0.60rem; color: #c97b00; font-weight: 700; text-align: right; white-space: nowrap; }
.team-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-weight: 500; min-width: 0; }
.pct { font-size: 0.60rem; font-weight: 700; padding: 1px 4px; border-radius: 3px;
       background: #f0ece3; min-width: 28px; text-align: center; white-space: nowrap; }

.prob-header {
    display: grid;
    grid-template-columns: 16px 1fr auto auto;
    gap: 5px;
    padding: 2px 7px 1px 7px;
    font-size: 0.52rem;
    color: #bbb;
    font-family: 'DM Sans', sans-serif;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    background: #faf8f4;
    border-bottom: 1px solid #eee9e2;
}

/* ── Championship centre ── */
.champ-col {
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    width: 600px;
    z-index: 2;
    /* This container is centered on the bracket midpoint.
       champ-inner (the games row) is the natural content.
       champ-bottom hangs below via absolute positioning
       so it doesn't shift the vertical centering. */
}
.champ-inner {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 8px;
    width: 100%;
    padding: 0 4px;
}
.champ-bottom {
    position: absolute;
    left: 0; right: 0;
    top: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding-top: 4px;
}
.champ-ff-col  { flex: 1; min-width: 0; }
.champ-ncg-col {
    flex: 1; min-width: 0;
    display: flex; flex-direction: column; align-items: stretch;
}
.champ-game {
    background: #ffffff;
    border: 1px solid #c97b0055;
    border-radius: 8px;
    overflow: visible;
    width: 100%;
    box-shadow: 0 0 18px rgba(201,123,0,0.09);
    position: relative;
}
.champ-game .team { padding: 8px 7px; min-height: 32px; }
.champ-game .prob-header { padding: 3px 7px 2px 7px; }
/* extra bottom margin on the NCG game card to make room for
   the red actual-winner labels that sit below the card */
.champ-ncg-col .champ-game { margin-bottom: 6px; }

.champion-box {
    margin-top: 14px;
    background: linear-gradient(135deg, #fff8ec 0%, #fff3d8 100%);
    border: 2px solid #c97b00;
    border-radius: 8px;
    padding: 6px 14px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(201,123,0,0.15);
    /* match the NCG column width: 1 of 3 equal cols minus gaps */
    width: calc((100% - 16px) / 3);
}
.champion-box .champ-label {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    color: #c97b00;
    margin-bottom: 2px;
}
.champion-box .champ-name {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: #333;
}
.champion-box .champ-name.wrong {
    text-decoration: line-through;
    color: #ccc;
}
.champion-box .champ-name.correct {
    color: #2e7d32;
}
.actual-champ {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.76rem;
    font-weight: 700;
    color: #d32f2f;
    text-align: center;
    padding-top: 5px;
}

/* ── Mobile-friendly adjustments ── */
@media (max-width: 768px) {
    .block-container { padding: 1rem 0.15rem 2rem 0.15rem; }

    /* Let the entire page be wide enough for the bracket.
       The body/viewport scrolls naturally — no nested scroll containers. */
    .bracket-wrapper {
        min-width: 1960px;
        position: relative;
    }

    /* Each side-half must be wide enough for 4 readable round columns */
    .bracket-wrapper > .side-half {
        min-width: 640px;
    }

    /* Game cards: wide enough to show full team names */
    .round-col { min-width: 155px; }
    .game { min-width: 145px; }

    /* Champ col stays centered on the bracket */
    .champ-col { width: 560px; }

    /* Round headers row: same total width so they align */
    .round-headers-row {
        min-width: 1960px;
    }
    .round-headers-left, .round-headers-right { min-width: 640px; }
}
</style>
""", unsafe_allow_html=True)

# Viewport meta — allow pinch-to-zoom on mobile
st.markdown('<meta name="viewport" content="width=device-width, initial-scale=1.0, minimum-scale=0.25, maximum-scale=5.0, user-scalable=yes">', unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("<h1>MARCH MADNESS</h1>", unsafe_allow_html=True)
st.markdown("""<div class="subtitle-block">
<p class="subtitle">Author: Ethan Davenport</p>
<p class="subtitle">Model Predictions Made Using Mixture of Experts</p>
</div>""", unsafe_allow_html=True)

# ── Game card renderers ────────────────────────────────────────────────────────

def team_row_html(name, seed, model_p, seed_p, state="neutral", show_seed_prob=True):
    """state: 'correct', 'wrong', or 'neutral'"""
    mc = prob_color(model_p)
    mp = f"{model_p*100:.0f}%" if not pd.isna(model_p) else "—"
    name_class = f"team-name {state}" if state != "neutral" else "team-name"
    if show_seed_prob:
        sc = prob_color(seed_p)
        sp = f"{seed_p*100:.0f}%"  if not pd.isna(seed_p)  else "—"
        return (
            f'<div class="team">'
            f'<span class="seed">{seed}</span>'
            f'<span class="{name_class}">{name}</span>'
            f'<span class="pct" style="color:{mc};">{mp}</span>'
            f'<span class="pct" style="color:{sc};">{sp}</span>'
            f'</div>'
        )
    else:
        return (
            f'<div class="team" style="grid-template-columns:16px 1fr auto;">'
            f'<span class="seed">{seed}</span>'
            f'<span class="{name_class}">{name}</span>'
            f'<span class="pct" style="color:{mc};">{mp}</span>'
            f'</div>'
        )

def game_card_parts(top_name, top_seed, bot_name, bot_seed, fp, sp,
                    match_id=None, tooltip_side="right",
                    actual_top=None, actual_top_seed=None, actual_top_tid=None,
                    actual_bot=None, actual_bot_seed=None, actual_bot_tid=None,
                    top_tid=None, bot_tid=None,
                    show_seed_prob=True,
                    prob_is_top=False):
    """
    actual_top/actual_bot: the two actual teams from df for this slot.
    actual_top = df's A team, actual_bot = df's B team (not top/bot ordered).
    We check whether each predicted team (top_tid, bot_tid) appears in
    {actual_top_tid, actual_bot_tid}. If yes → green. If no → strikethrough
    and show the actual team that should be there in red.

    prob_is_top: if True, fp is already P(top team wins); use directly.
                 if False (default), fp is P(favorite wins) and needs
                 seed-based reorientation.
    """
    if prob_is_top:
        model_top = fp if not pd.isna(fp) else float("nan")
        model_bot = (1 - fp) if not pd.isna(fp) else float("nan")
        seed_top_p = sp if not pd.isna(sp) else float("nan")
        seed_bot_p = (1 - sp) if not pd.isna(sp) else float("nan")
    elif top_seed <= bot_seed:
        model_top, model_bot = fp, 1 - fp
        seed_top_p, seed_bot_p = sp, 1 - sp
    else:
        model_top, model_bot = 1 - fp, fp
        seed_top_p, seed_bot_p = 1 - sp, sp
    if show_seed_prob:
        hdr = '<div class="prob-header"><span></span><span></span><span>Model</span><span>Seed</span></div>'
    else:
        hdr = '<div class="prob-header" style="grid-template-columns:16px 1fr auto;"><span></span><span></span><span>Model</span></div>'

    actual_tids = {t for t in [actual_top_tid, actual_bot_tid] if t is not None}

    def _state(tid):
        if not actual_tids:
            return "neutral"
        return "correct" if _tid(tid) in actual_tids else "wrong"

    state_top = _state(top_tid)
    state_bot = _state(bot_tid)

    # Determine which actual team to show for each wrong prediction.
    # actual_top = upper-source-slot winner (bracket top position)
    # actual_bot = lower-source-slot winner (bracket bot position)
    # When only ONE predicted team is wrong, show the actual team that the
    # other predicted team is NOT (i.e. the one that was displaced).
    # When BOTH predicted teams are wrong, actual_top replaces the display-top
    # and actual_bot replaces the display-bot (positional match).
    above_html = below_html = ""
    if state_top == "wrong" and state_bot == "wrong":
        # Both wrong: positional — actual_top above top, actual_bot below bot
        if actual_top:
            seed_str = f"{int(actual_top_seed)} " if actual_top_seed is not None else ""
            above_html = f'<div class="actual-winner above">{seed_str}{actual_top}</div>'
        if actual_bot:
            seed_str = f"{int(actual_bot_seed)} " if actual_bot_seed is not None else ""
            below_html = f'<div class="actual-winner below">{seed_str}{actual_bot}</div>'
    else:
        if state_top == "wrong":
            # Top is wrong, bot is correct → show whichever actual team isn't the bot
            if actual_top_tid and _tid(bot_tid) != actual_top_tid:
                name, seed_val = actual_top, actual_top_seed
            else:
                name, seed_val = actual_bot, actual_bot_seed
            if name:
                seed_str = f"{int(seed_val)} " if seed_val is not None else ""
                above_html = f'<div class="actual-winner above">{seed_str}{name}</div>'
        if state_bot == "wrong":
            # Bot is wrong, top is correct → show whichever actual team isn't the top
            if actual_bot_tid and _tid(top_tid) != actual_bot_tid:
                name, seed_val = actual_bot, actual_bot_seed
            else:
                name, seed_val = actual_top, actual_top_seed
            if name:
                seed_str = f"{int(seed_val)} " if seed_val is not None else ""
                below_html = f'<div class="actual-winner below">{seed_str}{name}</div>'

    tooltip_html = ""
    if match_id and match_id in shap_cache:
        b64 = shap_cache[match_id]
        tip_class = "shap-tooltip tip-left" if tooltip_side == "left" else "shap-tooltip"
        tooltip_html = (
            f'<div class="{tip_class}">'
            f'<img src="data:image/png;base64,{b64}" alt="SHAP explanation"/>'
            f'</div>'
        )

    return (
        f'<div class="game">'
        f'{above_html}{hdr}'
        f'{team_row_html(top_name, top_seed, model_top, seed_top_p, state=state_top, show_seed_prob=show_seed_prob)}'
        f'{team_row_html(bot_name, bot_seed, model_bot, seed_bot_p, state=state_bot, show_seed_prob=show_seed_prob)}'
        f'{below_html}{tooltip_html}'
        f'</div>'
    )

def game_card(gd, tooltip_side="right"):
    row      = gd["row"]
    fp       = row.get("FProb", float("nan"))
    sp       = row.get("SProb", float("nan"))
    match_id = str(row["MatchID"]) if "MatchID" in row and not pd.isna(row["MatchID"]) else None
    top_tid  = gd["top_tid"]
    bot_tid  = gd["bot_tid"]

    at_n = at_s = at_t = ab_n = ab_s = ab_t = None

    if match_id and match_id in results_cache:
        rc = results_cache[match_id]
        def _clean(v):
            return None if (v is None or (isinstance(v, float) and pd.isna(v))) else v

        # ActualA = upper-source-slot winner (bracket top)
        # ActualB = lower-source-slot winner (bracket bot)
        # These are already in bracket-position order from fill_bracket.
        # Pass them directly as actual_top / actual_bot — they always
        # correspond to display top / bot regardless of A/B swap,
        # because both orderings derive from the same bracket structure.
        at_n = _clean(rc.get("ActualA"))
        at_s = _clean(rc.get("ActualASeed"))
        at_t = _tid(rc.get("ActualATid"))
        ab_n = _clean(rc.get("ActualB"))
        ab_s = _clean(rc.get("ActualBSeed"))
        ab_t = _tid(rc.get("ActualBTid"))

    return game_card_parts(
        gd["top_name"], gd["top_seed"], gd["bot_name"], gd["bot_seed"],
        fp, sp, match_id, tooltip_side,
        actual_top=at_n, actual_top_seed=at_s, actual_top_tid=at_t,
        actual_bot=ab_n, actual_bot_seed=ab_s, actual_bot_tid=ab_t,
        top_tid=top_tid, bot_tid=bot_tid,
    )

# ── Connector SVG ──────────────────────────────────────────────────────────────

def make_connector_svg(prev_n, curr_n, rtl):
    """
    Bracket connector lines between prev_n parent games and curr_n child games.
    SVG lives in the COL_PAD-wide padding strip between columns.

    For each child game i connecting parents 2i and 2i+1:
      - Short horizontal stub from each parent's right/left edge → vertical bar
      - Vertical bar connecting the two parent stub endpoints (at x = COL_PAD/2)
      - Horizontal line from midpoint of vertical bar → child game edge

    LTR: parent edge = x=0, vertical bar at x=COL_PAD/2, child edge = x=COL_PAD
    RTL: parent edge = x=COL_PAD, vertical bar at x=COL_PAD/2, child edge = x=0
    """
    w  = COL_PAD
    h  = R1_TOTAL
    xv = w / 2
    x_parent = 0      if not rtl else w   # edge where parent col ends
    x_child  = w      if not rtl else 0   # edge where child col begins
    lines = []
    c  = "#ccc8c0"
    sw = "1.2"

    for i in range(curr_n):
        cy  = game_center_y(i, curr_n)
        p1y = game_center_y(i * 2,     prev_n)
        p2y = game_center_y(i * 2 + 1, prev_n)
        my  = (p1y + p2y) / 2

        # Horizontal stubs: parent edge → vertical bar, at each parent's center y
        lines.append(f'<line x1="{x_parent}" y1="{p1y:.1f}" x2="{xv}" y2="{p1y:.1f}" stroke="{c}" stroke-width="{sw}"/>')
        lines.append(f'<line x1="{x_parent}" y1="{p2y:.1f}" x2="{xv}" y2="{p2y:.1f}" stroke="{c}" stroke-width="{sw}"/>')
        # Vertical bar connecting the two stub endpoints
        lines.append(f'<line x1="{xv}" y1="{p1y:.1f}" x2="{xv}" y2="{p2y:.1f}" stroke="{c}" stroke-width="{sw}"/>')
        # Horizontal line from midpoint → child game
        lines.append(f'<line x1="{xv}" y1="{my:.1f}" x2="{x_child}" y2="{cy:.1f}" stroke="{c}" stroke-width="{sw}"/>')

    pos_side = "left:0" if not rtl else "right:0"
    return (
        f'<svg style="position:absolute;top:0;{pos_side};width:{w}px;height:{h}px;'
        f'pointer-events:none;overflow:visible;"'
        f' viewBox="0 0 {w} {h}" preserveAspectRatio="none">'
        + "".join(lines) + "</svg>"
    )

# ── Region HTML ────────────────────────────────────────────────────────────────
REGION_ROUNDS = ["Round 1", "Round 2", "Round 3 (Sweet Sixteen)", "Round 4 (Elite Eight)"]


def region_html(region, rtl=False):
    slot_pos  = build_slot_positions(region)
    direction = "rtl" if rtl else ""

    html = f'<div class="region-block"><div class="rounds-row {direction}">'

    prev_n = None
    for rnd_idx, rnd in enumerate(REGION_ROUNDS):
        games = games_for_ordered(region, rnd, slot_pos)
        n     = len(games)

        # Padding: first column (R64) gets no connector padding.
        # Subsequent columns get COL_PAD on their inward side for the connector SVG.
        if rnd_idx == 0:
            pad_style = ""
        else:
            pad_style = f"padding-left:{COL_PAD}px;" if not rtl else f"padding-right:{COL_PAD}px;"

        col_html = f'<div class="round-col" style="height:{R1_TOTAL}px;{pad_style}">'

        if n > 0:
            sh     = R1_TOTAL / n
            spacer = int((sh - GAME_H) / 2)
            gap    = int(sh - GAME_H)

            col_html += f'<div class="game-spacer" style="height:{spacer}px;"></div>'
            tooltip_side = "left" if rtl else "right"
            for idx, gd in enumerate(games):
                col_html += game_card(gd, tooltip_side)
                if idx < n - 1:
                    col_html += f'<div class="game-spacer" style="height:{gap}px;"></div>'

            # Connector SVG: anchored to left:0 (or right:0 for rtl) of this column's padding
            if rnd_idx > 0 and prev_n == n * 2:
                col_html += make_connector_svg(prev_n, n, rtl)

        col_html += '</div>'
        html += col_html
        prev_n = n

    html += '</div></div>'
    return html

# ── Championship centre ────────────────────────────────────────────────────────

def champ_html(layout):
    """layout = (TL, BL, TR, BR) region letters."""
    ff_games   = bracket[bracket["Round"] == "Final Four"].reset_index(drop=True)
    champ_game = bracket[bracket["Round"] == "Championship"].reset_index(drop=True)

    # Left side = TL + BL, right side = TR + BR
    left_set  = {layout[0], layout[1]}
    right_set = {layout[2], layout[3]}
    top_left_region  = layout[0]   # TL goes on top in left FF
    top_right_region = layout[2]   # TR goes on top in right FF

    ff_left = ff_right = None
    for _, row in ff_games.iterrows():
        regions = {str(row.get("Region_A", "")), str(row.get("Region_B", ""))}
        if regions & left_set:
            ff_left = row
        else:
            ff_right = row
    if ff_left is None and len(ff_games) > 0:
        ff_left = ff_games.iloc[0]
    if ff_right is None and len(ff_games) > 1:
        ff_right = ff_games.iloc[1]

    def _get_actuals_raw(mid):
        """Return raw (at_n,at_s,at_t, ab_n,ab_s,ab_t) from results_cache.
        These are in source-slot order (upper/lower), NOT layout order."""
        if not mid or mid not in results_cache:
            return None,None,None, None,None,None
        rc = results_cache[mid]
        def _clean(v):
            return None if (v is None or (isinstance(v, float) and pd.isna(v))) else v
        return (
            _clean(rc.get("ActualA")),    _clean(rc.get("ActualASeed")), _tid(rc.get("ActualATid")),
            _clean(rc.get("ActualB")),    _clean(rc.get("ActualBSeed")), _tid(rc.get("ActualBTid")),
        )

    def _get_actuals_for_ff(mid, row, top_region):
        """Return actuals reordered so actual_top = team from top_region,
        actual_bot = team from the other region."""
        at_n,at_s,at_t, ab_n,ab_s,ab_t = _get_actuals_raw(mid)
        if at_t is None and ab_t is None:
            return at_n,at_s,at_t, ab_n,ab_s,ab_t

        # Look up which region each actual team came from.
        # The FF sources are two E8 winners. We need to check which region
        # each actual team belongs to using the bracket data.
        # Simpler: check the df row's actual game — ActualA/ActualB correspond
        # to source slot order (upper/lower from SLOT_GAME_MAP).
        # For FF: WX_FF sources are (W_E8_1v2, X_E8_1v2).
        # We know the source regions from the slot names.
        sid = row.get("SlotID") if row is not None else None
        if sid and sid in _ff_source_regions:
            src_top_region, src_bot_region = _ff_source_regions[sid]
            # ActualA = winner of first source, ActualB = winner of second source
            # If first source's region == top_region, ActualA is display top
            if src_top_region == top_region:
                return at_n,at_s,at_t, ab_n,ab_s,ab_t
            else:
                # Swap: ActualB should be display top
                return ab_n,ab_s,ab_t, at_n,at_s,at_t

        return at_n,at_s,at_t, ab_n,ab_s,ab_t

    def _get_actuals_for_ncg(mid, row):
        """Return actuals reordered so actual_top = team from left side,
        actual_bot = team from right side."""
        at_n,at_s,at_t, ab_n,ab_s,ab_t = _get_actuals_raw(mid)
        if at_t is None and ab_t is None:
            return at_n,at_s,at_t, ab_n,ab_s,ab_t

        # NCG sources are (WX_FF, YZ_FF).
        # ActualA = winner of WX_FF, ActualB = winner of YZ_FF.
        # We need ActualA on top if WX is on the left, ActualB on top if YZ is on left.
        # Check: is WX_FF's pair on the left side?
        wx_regions = {"W", "X"}
        if wx_regions & left_set:
            # WX is on the left → ActualA (WX winner) is top
            return at_n,at_s,at_t, ab_n,ab_s,ab_t
        else:
            # YZ is on the left → ActualB (YZ winner) is top
            return ab_n,ab_s,ab_t, at_n,at_s,at_t

    # Precompute FF source regions from SLOT_GAME_MAP
    # WX_FF sources = (W_E8_1v2, X_E8_1v2) → regions W, X
    # YZ_FF sources = (Y_E8_1v2, Z_E8_1v2) → regions Y, Z
    _ff_source_regions = {}
    for ff_slot in ["WX_FF", "YZ_FF"]:
        sources = SLOT_GAME_MAP.get(ff_slot)
        if sources:
            # Extract region letter from source slot ID (first char)
            _ff_source_regions[ff_slot] = (sources[0][0], sources[1][0])

    def ff_card_html(row, top_region):
        if row is None:
            return ""
        aprob = row.get("AProb", float("nan"))  # P(A team wins)
        fp = row.get("FProb", float("nan"))
        sp = row.get("SProb", float("nan"))
        sa, sb = int(row["Seed_A"]), int(row["Seed_B"])
        ra = str(row.get("Region_A", ""))
        rb = str(row.get("Region_B", ""))
        mid = str(row["MatchID"]) if "MatchID" in row and not pd.isna(row["MatchID"]) else None

        # Determine top/bot based on region layout, and compute
        # top_prob directly from AProb so it follows the team correctly.
        if ra == top_region:
            # A team is on top
            top_n, top_s, bot_n, bot_s = row["ATeamName"], sa, row["BTeamName"], sb
            top_id, bot_id = _tid(row["ATeamID"]), _tid(row["BTeamID"])
            top_prob = aprob
        elif rb == top_region:
            # B team is on top
            top_n, top_s, bot_n, bot_s = row["BTeamName"], sb, row["ATeamName"], sa
            top_id, bot_id = _tid(row["BTeamID"]), _tid(row["ATeamID"])
            top_prob = 1 - aprob if not pd.isna(aprob) else aprob
        elif sa <= sb:
            top_n, top_s, bot_n, bot_s = row["ATeamName"], sa, row["BTeamName"], sb
            top_id, bot_id = _tid(row["ATeamID"]), _tid(row["BTeamID"])
            top_prob = aprob
        else:
            top_n, top_s, bot_n, bot_s = row["BTeamName"], sb, row["ATeamName"], sa
            top_id, bot_id = _tid(row["BTeamID"]), _tid(row["ATeamID"])
            top_prob = 1 - aprob if not pd.isna(aprob) else aprob

        at_n,at_s,at_t, ab_n,ab_s,ab_t = _get_actuals_for_ff(mid, row, top_region)
        return game_card_parts(top_n, top_s, bot_n, bot_s, top_prob, sp, mid,
                               actual_top=at_n, actual_top_seed=at_s, actual_top_tid=at_t,
                               actual_bot=ab_n, actual_bot_seed=ab_s, actual_bot_tid=ab_t,
                               top_tid=top_id, bot_tid=bot_id,
                               show_seed_prob=False, prob_is_top=True)

    html = '<div class="champ-col"><div class="champ-inner">'
    html += f'<div class="champ-ff-col">{ff_card_html(ff_left, top_left_region)}</div>'

    # Championship
    html += '<div class="champ-ncg-col">'
    if not champ_game.empty:
        row  = champ_game.iloc[0]
        aprob = row.get("AProb", float("nan"))
        fp   = row.get("FProb", float("nan"))
        sp   = row.get("SProb", float("nan"))
        sa, sb = int(row["Seed_A"]), int(row["Seed_B"])
        ra = str(row.get("Region_A", ""))
        rb = str(row.get("Region_B", ""))
        mid = str(row["MatchID"]) if "MatchID" in row and not pd.isna(row["MatchID"]) else None
        if ra in left_set:
            tn, ts, bn, bs = row["ATeamName"], sa, row["BTeamName"], sb
            t_id, b_id = _tid(row["ATeamID"]), _tid(row["BTeamID"])
            top_prob = aprob
        elif rb in left_set:
            tn, ts, bn, bs = row["BTeamName"], sb, row["ATeamName"], sa
            t_id, b_id = _tid(row["BTeamID"]), _tid(row["ATeamID"])
            top_prob = 1 - aprob if not pd.isna(aprob) else aprob
        elif sa <= sb:
            tn, ts, bn, bs = row["ATeamName"], sa, row["BTeamName"], sb
            t_id, b_id = _tid(row["ATeamID"]), _tid(row["BTeamID"])
            top_prob = aprob
        else:
            tn, ts, bn, bs = row["BTeamName"], sb, row["ATeamName"], sa
            t_id, b_id = _tid(row["BTeamID"]), _tid(row["ATeamID"])
            top_prob = 1 - aprob if not pd.isna(aprob) else aprob

        at_n,at_s,at_t, ab_n,ab_s,ab_t = _get_actuals_for_ncg(mid, row)

        card = game_card_parts(tn, ts, bn, bs, top_prob, sp, mid,
                               actual_top=at_n, actual_top_seed=at_s, actual_top_tid=at_t,
                               actual_bot=ab_n, actual_bot_seed=ab_s, actual_bot_tid=ab_t,
                               top_tid=t_id, bot_tid=b_id,
                               show_seed_prob=False, prob_is_top=True)

        # Apply champ-game styling to the card (gold border)
        card = card.replace('<div class="game">', '<div class="champ-game game">', 1)
        html += card

    html += '</div>'  # close champ-ncg-col

    html += f'<div class="champ-ff-col">{ff_card_html(ff_right, top_right_region)}</div>'
    html += '</div>'  # close champ-inner (the row)

    # ── Champion box below the row, absolutely positioned so it
    #    doesn't affect the vertical centering of the games row ──
    if not champ_game.empty:
        row  = champ_game.iloc[0]
        mid  = str(row["MatchID"]) if "MatchID" in row and not pd.isna(row["MatchID"]) else None
        winner   = row["Selected"]
        win_seed = get_winner_seed(row)

        # Check if predicted champ matches actual champ
        actual_winner = None
        actual_winner_seed = None
        if mid and mid in results_cache:
            rc = results_cache[mid]
            def _clean_w(v):
                return None if (v is None or (isinstance(v, float) and pd.isna(v))) else v
            actual_winner      = _clean_w(rc.get("ActualWinner"))
            actual_winner_seed = _clean_w(rc.get("ActualWinnerSeed"))

        # Determine if predicted champion is correct
        if actual_winner is not None:
            champ_correct = (winner == actual_winner)
            champ_cls = "champ-name correct" if champ_correct else "champ-name wrong"
        else:
            champ_cls = "champ-name"
            champ_correct = True  # no data to compare

        html += '<div class="champ-bottom">'
        html += '<div class="champion-box">'
        html += '<div class="champ-label">🏆 CHAMPION</div>'
        html += f'<div class="{champ_cls}">{win_seed} {winner}</div>'
        html += '</div>'

        # Show actual champion below if prediction was wrong
        if actual_winner is not None and not champ_correct:
            act_seed_str = f"{int(actual_winner_seed)} " if actual_winner_seed is not None else ""
            html += f'<div class="actual-champ">🏆 {act_seed_str}{actual_winner}</div>'

        html += '</div>'  # close champ-bottom

    html += '</div>'  # close champ-col
    return html

# ── Assemble ───────────────────────────────────────────────────────────────────

# Round header cells — border only under box content, gaps between rounds.
# The last cell (E8, index 3) also gets the side-half inner padding so it aligns
# with the E8 box and leaves a gap at the center.
SIDE_PAD = 36  # px — inner padding on each side-half, also applied to header row

def make_header_cells(rounds, rtl=False):
    cells = []
    for i, rnd in enumerate(rounds):
        is_last = (i == len(rounds) - 1)
        # Left margin: connector gap for cols 1+, but on the last (E8) col the
        # side-half padding already provides that gap — so use SIDE_PAD instead of
        # COL_PAD+SIDE_PAD to keep all underlines the same length.
        if not rtl:
            ml = SIDE_PAD if is_last else (COL_PAD if i > 0 else 0)
            mr = 0
        else:
            ml = 0
            mr = SIDE_PAD if is_last else (COL_PAD if i > 0 else 0)

        margin_style = ""
        if ml: margin_style += f"margin-left:{ml}px;"
        if mr: margin_style += f"margin-right:{mr}px;"

        inner = (
            f'<span style="display:block;border-bottom:2px solid #d8d4cc;'
            f'padding-bottom:5px;text-align:center;{margin_style}">'
            f'{ROUND_SHORT[rnd]}</span>'
        )
        cells.append(f'<div class="round-header-cell">{inner}</div>')
    return "".join(cells)

# ── Custom tab styling ─────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Make Streamlit tabs span full width as solid rectangles */
.stTabs [data-baseweb="tab-list"] {
    gap: 0px;
    border-bottom: 2px solid #ddd9d2;
}
.stTabs [data-baseweb="tab"] {
    flex: 1;
    justify-content: center;
    padding: 12px 0;
    border-radius: 0;
    border: 1px solid #ddd9d2;
    border-bottom: none;
    background: #f5f3ef;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    color: #888;
    margin: 0;
}
.stTabs [aria-selected="true"] {
    background: #fff !important;
    color: #c97b00 !important;
    border-top: 2px solid #c97b00 !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-border"]    { display: none; }

/* ── Bracket title bar ── */
.bracket-title-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 6px;
    padding: 0 4px;
}
.bracket-title-bar .bt-spacer { width: 80px; }
.bracket-title-bar .bt-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #c97b00;
    text-align: center;
    flex: 1;
}

/* ── Style the Streamlit selectbox in bracket tab ── */
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    border-color: #ddd9d2;
    background: #faf8f4;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    color: #333;
    border-radius: 6px;
}

/* ── Play-in toggle buttons (bracket tab) ── */
/* Secondary (unselected) */
[data-testid="stButton"] button[kind="secondary"] {
    border: 1px solid #ddd9d2 !important;
    border-radius: 4px !important;
    background: #faf8f4 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.62rem !important;
    font-weight: 500 !important;
    color: #555 !important;
    padding: 0px 5px !important;
    min-height: 0 !important;
    max-height: 22px !important;
    height: 22px !important;
    line-height: 1 !important;
    margin: 0 !important;
    transition: all 0.15s !important;
}
[data-testid="stButton"] button[kind="secondary"]:hover {
    border-color: #c97b00 !important;
    color: #c97b00 !important;
}
/* Primary (selected) */
[data-testid="stButton"] button[kind="primary"] {
    background: #c97b00 !important;
    color: #fff !important;
    border: 1px solid #c97b00 !important;
    border-radius: 4px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.62rem !important;
    font-weight: 700 !important;
    padding: 0px 5px !important;
    min-height: 0 !important;
    max-height: 22px !important;
    height: 22px !important;
    line-height: 1 !important;
    margin: 0 !important;
}
/* Reduce vertical spacing around button containers */
[data-testid="stButton"] {
    margin: 0 !important;
    padding: 0 !important;
}
[data-testid="stButton"] > div {
    margin: 0 !important;
    padding: 0 !important;
}
/* Shrink the column gaps around toggle buttons */
[data-testid="stHorizontalBlock"] {
    gap: 0.25rem !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover {
    border-color: #c97b00;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within {
    border-color: #c97b00;
    box-shadow: 0 0 0 2px #c97b0022;
}
</style>
""", unsafe_allow_html=True)

tab_bracket, tab_probs, tab_about = st.tabs(["⊐―  Bracket", "▦  Round Probabilities", "📖  How It Works"])

with tab_bracket:
    import json as _json

    # ── Year selector ──
    _bcol_l, _bcol_c, _bcol_r = st.columns([1, 6, 1])
    with _bcol_r:
        bracket_year = st.selectbox(
            "Year", _bracket_years,
            index=0, key="bracket_year", label_visibility="collapsed"
        )

    # Reload bracket data for selected year
    _bracket_data = _load_bracket_year(bracket_year)
    _bracket_keys = sorted(_bracket_data.keys())

    # Load play-in metadata for toggle labels
    _playin_meta_bracket = None
    if os.path.exists("playin_meta.csv"):
        _pm = pd.read_csv("playin_meta.csv")
        _pm["Season"] = _pm["Season"].astype(int)
        _pm_yr = _pm[_pm["Season"] == bracket_year]
        if not _pm_yr.empty:
            _playin_meta_bracket = _pm_yr[["Region", "SeedNum", "TeamA", "TeamB"]].to_dict(orient="records")

    # Determine which bracket to display
    if len(_bracket_keys) > 1 and _playin_meta_bracket:
        # Build play-in toggle HTML matching the round probs tab style
        _playin_choices = {}
        for idx, game in enumerate(_playin_meta_bracket):
            skey = f"bracket_playin_{bracket_year}_{idx}"
            if skey not in st.session_state:
                st.session_state[skey] = game["TeamA"]
            _playin_choices[idx] = st.session_state[skey]

        def _make_playin_callback(skey, team):
            def _cb():
                st.session_state[skey] = team
            return _cb

        # Render toggles in a single centered row: [spacer | label btnA btnB | label btnA btnB | spacer]
        n_games = len(_playin_meta_bracket)
        # Each game needs: label + 2 buttons = 3 units. Add spacers on sides.
        col_spec = [4] + [0.7, 1, 1] * n_games + [4]
        _tcols = st.columns(col_spec, gap="small")
        for idx, game in enumerate(_playin_meta_bracket):
            skey = f"bracket_playin_{bracket_year}_{idx}"
            current = st.session_state[skey]
            base = 1 + idx * 3  # offset into _tcols
            with _tcols[base]:
                st.markdown(
                    f'<div style="font-family:\'DM Sans\',sans-serif;font-size:0.60rem;'
                    f'font-weight:600;color:#888;letter-spacing:0.05em;text-transform:uppercase;'
                    f'text-align:right;padding-top:4px;white-space:nowrap;">{game["SeedNum"]}-seed:</div>',
                    unsafe_allow_html=True,
                )
            with _tcols[base + 1]:
                st.button(
                    game["TeamA"],
                    key=f"btn_{skey}_a",
                    on_click=_make_playin_callback(skey, game["TeamA"]),
                    type="primary" if current == game["TeamA"] else "secondary",
                    use_container_width=True,
                )
            with _tcols[base + 2]:
                st.button(
                    game["TeamB"],
                    key=f"btn_{skey}_b",
                    on_click=_make_playin_callback(skey, game["TeamB"]),
                    type="primary" if current == game["TeamB"] else "secondary",
                    use_container_width=True,
                )

        # Build the key to match
        _chosen_list = [_playin_choices.get(i, g["TeamA"]) for i, g in enumerate(_playin_meta_bracket)]
        _active_key = _json.dumps(_chosen_list, separators=(",", ":"))

        if _active_key in _bracket_data:
            bracket, shap_cache, results_cache = _bracket_data[_active_key]
        else:
            bracket, shap_cache, results_cache = _bracket_data[_bracket_keys[0]]
    else:
        bracket, shap_cache, results_cache = _bracket_data[_bracket_keys[0]]

    # Region layout for this year: (TL, BL, TR, BR)
    _layout = _get_layout(bracket_year)
    regions       = set(bracket["Region_A"].dropna().unique()) | set(bracket["Region_B"].dropna().unique())
    left_regions  = [r for r in [_layout[0], _layout[1]] if r in regions]   # TL, BL
    right_regions = [r for r in [_layout[2], _layout[3]] if r in regions]   # TR, BR

    # Title bar (rendered as HTML for consistent styling)
    with _bcol_c:
        st.markdown(
            f'<div style="font-family:\'DM Sans\',sans-serif;font-size:1.85rem;'
            f'font-weight:700;color:#c97b00;text-align:center;padding-top:6px;">'
            f'{bracket_year} Bracket</div>',
            unsafe_allow_html=True,
        )

    hdr_left  = make_header_cells(REGION_ROUNDS, rtl=False)
    hdr_right = make_header_cells(REGION_ROUNDS, rtl=True)

    headers_html = (
        f'<div class="round-headers-row">'
        f'<div class="round-headers-left" style="padding-right:{SIDE_PAD}px;">{hdr_left}</div>'
        f'<div class="round-headers-right" style="padding-left:{SIDE_PAD}px;">{hdr_right}</div>'
        f'</div>'
    )

    html = headers_html
    html += '<div class="bracket-wrapper">'
    html += f'<div class="side-half" style="padding-right:{SIDE_PAD}px;">'
    for r in left_regions:
        html += region_html(r, rtl=False)
    html += '</div>'
    html += champ_html(_layout)
    html += f'<div class="side-half" style="padding-left:{SIDE_PAD}px;">'
    for r in right_regions:
        html += region_html(r, rtl=True)
    html += '</div>'
    html += '</div>'

    st.markdown(html, unsafe_allow_html=True)

with tab_probs:
    import streamlit.components.v1 as components
    import json

    # Try adv_all.csv first (combined), fall back to adv_2025.csv
    if os.path.exists("adv_all.csv"):
        adv_all = pd.read_csv("adv_all.csv")
        # Handle old CSVs that saved the index as a column
        if "Unnamed: 0" in adv_all.columns:
            if "TeamName" not in adv_all.columns:
                adv_all = adv_all.rename(columns={"Unnamed: 0": "TeamName"})
            else:
                adv_all = adv_all.drop(columns=["Unnamed: 0"])
        adv_all["Season"] = adv_all["Season"].astype(int)
        # Ensure PlayinKey column exists (empty string for legacy files)
        if "PlayinKey" not in adv_all.columns:
            adv_all["PlayinKey"] = ""
        adv_all["PlayinKey"] = adv_all["PlayinKey"].fillna("")
        available_years = sorted(adv_all["Season"].unique(), reverse=True)
    elif os.path.exists("adv_2025.csv"):
        adv_all = pd.read_csv("adv_2025.csv")
        adv_all["Season"] = 2025
        adv_all["PlayinKey"] = ""
        available_years = [2025]
    else:
        st.error("Could not find adv_all.csv or adv_2025.csv.")
        adv_all = None
        available_years = []

    # Load play-in metadata (tells us which games have unresolved play-in choices)
    playin_meta = None
    if os.path.exists("playin_meta.csv"):
        playin_meta = pd.read_csv("playin_meta.csv")
        playin_meta["Season"] = playin_meta["Season"].astype(int)

    if adv_all is not None:
        round_cols   = ["Round of 32", "Sweet 16", "Elite 8", "Final Four", "Championship", "Champion"]
        display_cols = ["TeamName", "SeedNum"] + round_cols
        col_labels = {
            "Team": "Team", "Seed": "Seed",
            "Round of 32": "R32", "Sweet 16": "Sweet 16", "Elite 8": "Elite Eight",
            "Final Four": "Final Four", "Championship": "Championship", "Champion": "Champion",
        }
        all_cols = ["Team", "Seed"] + round_cols

        # Build per-year data, keyed by (year, playin_key_str)
        # For years with no play-in choices, there's one entry with key ""
        # For years with choices, there's one entry per combo
        all_years_data = {}        # { year: { playin_key_str: [rows] } }
        playin_meta_js = {}        # { year: [ {region, seed, teamA, teamB}, ... ] }

        for yr in available_years:
            yr_data = adv_all[adv_all["Season"] == yr]
            keys_in_year = yr_data["PlayinKey"].unique().tolist()

            year_dict = {}
            for pk in keys_in_year:
                subset = yr_data[yr_data["PlayinKey"] == pk][display_cols].copy()
                subset = subset.rename(columns={"TeamName": "Team", "SeedNum": "Seed"})
                subset = subset.astype(object).where(subset.notna(), None)
                year_dict[pk] = subset.to_dict(orient="records")
            all_years_data[int(yr)] = year_dict

            # Play-in metadata for this year
            if playin_meta is not None and not playin_meta.empty:
                pm_yr = playin_meta[playin_meta["Season"] == yr]
                if not pm_yr.empty:
                    playin_meta_js[int(yr)] = pm_yr[["Region", "SeedNum", "TeamA", "TeamB"]].to_dict(orient="records")

        all_data_json    = json.dumps(all_years_data)
        playin_meta_json = json.dumps(playin_meta_js)
        cols_json        = json.dumps(all_cols)
        labels_json      = json.dumps(col_labels)
        years_json       = json.dumps([int(y) for y in available_years])
        init_year_json   = json.dumps(int(available_years[0]))

        n_rows = max(
            len(rows)
            for year_dict in all_years_data.values()
            for rows in year_dict.values()
        )
        est_height = n_rows * 33 + 160  # extra room for play-in selectors

        table_component = f"""
<!DOCTYPE html>
<html>
<head>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700;800&display=swap" rel="stylesheet">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: transparent; font-family: 'DM Sans', sans-serif; padding: 8px 0; }}

  /* ── Top bar ── */
  #top-bar {{ display: flex; justify-content: center; margin-bottom: 8px; }}
  #top-bar-inner {{
    display: flex; align-items: center; justify-content: space-between;
    width: 1060px; max-width: 100%;
  }}
  #title {{
    font-size: 1.85rem; font-weight: 700; color: #c97b00;
    flex: 1; text-align: center;
  }}
  #year-select {{
    appearance: none; -webkit-appearance: none;
    border: 1px solid #ddd9d2; border-radius: 6px;
    background: #faf8f4 url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%23999'/%3E%3C/svg%3E") no-repeat right 8px center;
    font-family: 'DM Sans', sans-serif; font-size: 0.85rem; font-weight: 600; color: #333;
    padding: 5px 28px 5px 10px; cursor: pointer; outline: none; transition: border-color 0.15s;
  }}
  #year-select:hover {{ border-color: #c97b00; }}
  #year-select:focus {{ border-color: #c97b00; box-shadow: 0 0 0 2px #c97b0022; }}

  /* ── Play-in selectors ── */
  #playin-bar {{
    display: none; justify-content: center; margin-bottom: 10px;
  }}
  #playin-bar.visible {{ display: flex; }}
  #playin-inner {{
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
    justify-content: center;
  }}
  .playin-group {{
    display: flex; align-items: center; gap: 6px;
    font-size: 0.78rem; color: #555;
  }}
  .playin-label {{
    font-weight: 600; color: #888; font-size: 0.68rem;
    letter-spacing: 0.05em; text-transform: uppercase;
  }}
  .playin-btn {{
    padding: 4px 10px; border: 1px solid #ddd9d2; border-radius: 5px;
    background: #faf8f4; font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem; font-weight: 500; color: #555;
    cursor: pointer; transition: all 0.15s;
  }}
  .playin-btn:hover {{ border-color: #c97b00; color: #c97b00; }}
  .playin-btn.active {{
    background: #c97b00; color: #fff; border-color: #c97b00; font-weight: 700;
  }}

  /* ── Table ── */
  #outer {{ display: flex; justify-content: center; }}
  #wrap  {{ border-radius: 8px; border: 1px solid #ddd9d2;
            box-shadow: 0 2px 10px rgba(0,0,0,0.07);
            overflow: hidden; display: inline-block; max-width: 100%; }}

  table  {{ border-collapse: collapse; background: #fff; table-layout: fixed; }}
  thead  {{ background: #faf8f4; }}
  th     {{ padding: 9px 16px; font-size: 0.68rem; font-weight: 800;
            letter-spacing: 0.07em; text-transform: uppercase;
            border-bottom: 2px solid #ddd9d2; white-space: nowrap;
            cursor: pointer; user-select: none; color: #888;
            transition: color 0.15s; overflow: hidden; text-overflow: ellipsis; }}
  th:hover  {{ color: #555; }}
  th.active {{ color: #c97b00; }}
  td {{ padding: 6px 16px; font-size: 0.82rem; white-space: nowrap;
        overflow: hidden; text-overflow: ellipsis; }}
  tbody tr:nth-child(even) {{ background: #faf8f4; }}
  tbody tr:hover {{ filter: brightness(0.96); }}
  .arrow {{ font-size: 0.65rem; margin-left: 3px; opacity: 0.8; }}

  col.col-team  {{ width: 160px; }}
  col.col-seed  {{ width: 52px; }}
  col.col-round {{ width: 108px; }}

  /* Mobile: horizontal scroll for the table */
  @media (max-width: 768px) {{
    #outer {{ overflow-x: auto; -webkit-overflow-scrolling: touch; display: block; }}
    #wrap  {{ min-width: 920px; }}
    table  {{ min-width: 920px; }}
  }}
</style>
</head>
<body>

<div id="top-bar">
  <div id="top-bar-inner">
    <div style="width:80px;"></div>
    <div id="title">{available_years[0]} Tournament</div>
    <select id="year-select"></select>
  </div>
</div>

<div id="playin-bar">
  <div id="playin-inner"></div>
</div>

<div id="outer"><div id="wrap"><table id="tbl">
  <colgroup id="colgroup"></colgroup>
  <thead id="thead"></thead>
  <tbody id="tbody"></tbody>
</table></div></div>

<script>
// ALL_DATA[year] = {{ playin_key_str: [rows], ... }}
// For years with no play-in choices, the only key is ""
const ALL_DATA    = {all_data_json};
const PLAYIN_META = {playin_meta_json};
const ALL_COLS    = {cols_json};
const LABELS      = {labels_json};
const YEARS       = {years_json};
const TEXT_COLS   = new Set(["Team", "Seed"]);

let curYear    = {init_year_json};
let sortCol    = "Final Four";
let sortAsc    = false;

// Current play-in selections: {{ game_index: chosen_team_name }}
let playinChoices = {{}};

// Get the currently active playin key string from choices
function getActiveKey() {{
  const yearData = ALL_DATA[curYear];
  const keys = Object.keys(yearData);
  if (keys.length <= 1) return keys[0] || "";

  // Build the key from current choices
  const meta = PLAYIN_META[curYear] || [];
  if (meta.length === 0) return keys[0] || "";

  const chosen = meta.map((g, i) => playinChoices[i] || g.TeamA);
  const keyStr = JSON.stringify(chosen);

  // Find matching key
  if (yearData[keyStr] !== undefined) return keyStr;

  // Fallback: first key
  return keys[0];
}}

function getActiveRows() {{
  const key = getActiveKey();
  return ALL_DATA[curYear][key] || [];
}}

// Populate year dropdown
const sel = document.getElementById("year-select");
YEARS.forEach(y => {{
  const opt = document.createElement("option");
  opt.value = y;
  opt.textContent = y;
  if (y === curYear) opt.selected = true;
  sel.appendChild(opt);
}});

sel.addEventListener("change", () => {{
  curYear = parseInt(sel.value);
  playinChoices = {{}};
  document.getElementById("title").textContent = curYear + " Tournament";
  renderPlayinBar();
  render();
}});

function renderPlayinBar() {{
  const bar   = document.getElementById("playin-bar");
  const inner = document.getElementById("playin-inner");
  const meta  = PLAYIN_META[curYear];
  const keys  = Object.keys(ALL_DATA[curYear]);

  if (!meta || meta.length === 0 || keys.length <= 1) {{
    bar.classList.remove("visible");
    inner.innerHTML = "";
    return;
  }}

  bar.classList.add("visible");
  inner.innerHTML = "";

  meta.forEach((game, idx) => {{
    const grp = document.createElement("div");
    grp.className = "playin-group";

    const label = document.createElement("span");
    label.className = "playin-label";
    label.textContent = game.SeedNum + "-seed:";
    grp.appendChild(label);

    [game.TeamA, game.TeamB].forEach(team => {{
      const btn = document.createElement("button");
      btn.className = "playin-btn";
      btn.textContent = team;
      const chosen = playinChoices[idx] || game.TeamA;
      if (chosen === team) btn.classList.add("active");
      btn.addEventListener("click", () => {{
        playinChoices[idx] = team;
        renderPlayinBar();
        render();
      }});
      grp.appendChild(btn);
    }});

    inner.appendChild(grp);
  }});
}}

function pctStyle(v) {{
  if (v === null || v === undefined || isNaN(v)) return "";
  const r = Math.round(255 - (255-201)*v);
  const g = Math.round(253 - (253-123)*v);
  const b = Math.round(245 - (245-0)*v);
  const text = v > 0.5 ? "#fff" : "#333";
  return `background:rgb(${{r}},${{g}},${{b}});color:${{text}};text-align:center;`;
}}

function fmtPct(v) {{
  if (v === null || v === undefined || isNaN(+v)) return "—";
  return (v * 100).toFixed(1) + "%";
}}

function render() {{
  const rows = getActiveRows();

  const colgroup = document.getElementById("colgroup");
  colgroup.innerHTML = "";
  ALL_COLS.forEach(col => {{
    const c = document.createElement("col");
    if (col === "Team")       c.className = "col-team";
    else if (col === "Seed")  c.className = "col-seed";
    else                      c.className = "col-round";
    colgroup.appendChild(c);
  }});

  const thead = document.getElementById("thead");
  thead.innerHTML = "";
  const tr = document.createElement("tr");
  ALL_COLS.forEach(col => {{
    const th = document.createElement("th");
    const isActive = col === sortCol;
    th.style.textAlign = TEXT_COLS.has(col) ? "left" : "center";
    if (isActive) th.classList.add("active");
    const arrow = isActive ? `<span class="arrow">${{sortAsc ? "↑" : "↓"}}</span>` : "";
    th.innerHTML = LABELS[col] + arrow;
    th.addEventListener("click", () => {{
      const wasActive = (col === sortCol);
      if (wasActive) {{
        sortAsc = !sortAsc;
      }} else {{
        sortCol = col;
        sortAsc = TEXT_COLS.has(col);
      }}
      render();
    }});
    tr.appendChild(th);
  }});
  thead.appendChild(tr);

  const sorted = [...rows].sort((a, b) => {{
    const va = a[sortCol], vb = b[sortCol];
    if (va === null || va === undefined) return 1;
    if (vb === null || vb === undefined) return -1;
    const cmp = typeof va === "string" ? va.localeCompare(vb) : va - vb;
    return sortAsc ? cmp : -cmp;
  }});

  const tbody = document.getElementById("tbody");
  tbody.innerHTML = "";
  sorted.forEach(row => {{
    const tr = document.createElement("tr");
    ALL_COLS.forEach(col => {{
      const td = document.createElement("td");
      const val = row[col];
      if (col === "Team") {{
        td.style.fontWeight = "500";
        td.textContent = val;
      }} else if (col === "Seed") {{
        td.style.cssText = "text-align:center;color:#c97b00;font-weight:700;";
        td.textContent = val;
      }} else {{
        td.style.cssText = pctStyle(val);
        td.textContent = fmtPct(val);
      }}
      tr.appendChild(td);
    }});
    tbody.appendChild(tr);
  }});
}}

renderPlayinBar();
render();
</script>
</body>
</html>"""

        components.html(table_component, height=est_height, scrolling=False)
with tab_about:
    _col_spacer_l, _col_about, _col_spacer_r = st.columns([1, 2, 1])
    with _col_about:

        def _img_small(url):
            """2/3 width image, centered, with spacing below."""
            st.markdown(
                f'<div style="display:flex;justify-content:center;margin-bottom:1.2rem;">'
                f'<img src="{url}" style="width:67%;height:auto;"/>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("""
## Project Overview
This repository contains code to estimate win probabilities for NCAA men's tournament games from 2017–2025 and to turn those probabilities into a bracket strategy for upcoming tournaments. The workflow covers data collection, merging and cleaning, feature selection with grouped penalties, model training and evaluation, and bracket construction rules that balance model predictions with historical seeding outcomes.

This project exists in my relentless pursuit of reaching my high of the 2021 Tournament again. That year, I took home first place in my Dad's work bracket pool, with 150+ entries. It was glorious. I return to the bracket-making starting blocks, this time armed with the power of data and machine learning.

*\\*This project was done purely for my enjoyment and hopeful success in March Madness bracket groups, and is not associated with any class, employer, or other formal commitment.*

---

## Data Collection and Sources
Tournament results and core identifiers come from the official March Machine Learning Mania Kaggle competition (detailed results, seeds, and team metadata). Additional team strength metrics are pulled from several external sources covering efficiency, resume quality, and advanced scouting stats for recent seasons.

**Key sources:**
- **[Kaggle NCAA tourney data (MM Mania)](https://www.kaggle.com/competitions/march-machine-learning-mania-2025/data)**: play-by-play–level box score stats and tournament seeds
- **[Kaggle CBB dataset](https://www.kaggle.com/datasets/andrewsundberg/college-basketball-dataset)**: regular-season box score–based advanced metrics and adjusted efficiencies
- **[KenPom / Barttorvik](https://www.kaggle.com/datasets/nishaanamin/march-madness-data?select=KenPom+Barttorvik.csv)**: tempo-free efficiency, tempo, height, experience, talent ratings, and other advanced team-level metrics
- **[Resumes](https://www.kaggle.com/datasets/nishaanamin/march-madness-data?select=Resumes.csv)**: "resume" statistics summarizing quality wins and losses by quadrant and other selection committee-style descriptors
- **[KPI rankings](https://faktorsports.com/)**: resume-based rating and SOS metrics

---

## Building the Modeling Dataset
The modeling dataset is constructed at the game level, with each row corresponding to a single NCAA tournament game. The pipeline starts from the detailed tournament results and augments them with seeds and team names, restricted to the main bracket rounds for seasons where external sources are available.

**Key steps:**
- **Tournament results**: Load tournament results and drop unneeded box score columns, keeping scoring, team IDs, and day numbers. Map DayNum to human-readable rounds (Play-in, Round 1–6) using a custom function that accounts for the 2021 schedule quirk, double checking round counts per season.
- **Seeds and naming**: Strip regional prefixes from seeds to create numeric seeds (1–16) and merge them onto winning and losing teams. Normalize ordering by renaming the alphabetically first team as ATeam and the other as BTeam, then realign scores, seeds, and IDs so the "A" side is not always the winner.
- **Filtering seasons**: Restrict to seasons where all metrics are available, ultimately focusing the modeling window on 2017–2025.

The raw tournament results are cleaned and reshaped into a game-level table with team IDs, seeds, scores, and rounds, but no team statistics attached yet.
""", unsafe_allow_html=True)

        _img_small("https://github.com/user-attachments/assets/e45bf5b8-3e5a-4717-b70a-abd60b71be31")

        st.markdown("""
### Merging External Team Features
For each season, external datasets are reshaped into team-season–level tables and merged twice onto the game dataset: once for ATeam and once for BTeam. This creates a symmetric design where every numeric attribute exists in A and B versions.

Examples of merged features:
- From **CBB**: offensive and defensive efficiencies, pace, shooting splits, offensive/defensive rebounding, turnover rates, and ranking-based stats
- From **KenPom / Barttorvik**: adjusted offense/defense, tempo, average height, effective height, experience, talent, free-throw rates, and efficiency-based SOS measures
- From **Resumes**: quad-level win/loss counts, quality-win indices, and other committee-style resume indicators
- From **KPI**: KPI rating, SOS value, and SOS ranking for each team-season

The external sources are stacked and standardized into a unified team-season table so that any team in any year has a full set of candidate metrics.
""", unsafe_allow_html=True)

        _img_small("https://github.com/user-attachments/assets/1c1809e0-b03a-42fb-94c6-c6d6ad12ad9e")

        st.markdown("""
Because sources cover different year ranges, each file is filtered to the overlapping seasons and then concatenated. The team-season table is left-joined onto the game-level results using year and team ID, attaching the appropriate stats to each side of every matchup.
""", unsafe_allow_html=True)

        _img_small("https://github.com/user-attachments/assets/c0fcf85e-cb8f-4159-80c7-d1afb7835c01")

        st.markdown("""
The final merged DataFrame includes 100+ columns per game, with consistent A/B feature pairs that maintain symmetry between teams. The following represents a conceptual mapping of the final training data set, where each row is a single game with aligned features for both teams, ready to feed into the machine learning pipeline.
""", unsafe_allow_html=True)

        _img_small("https://github.com/user-attachments/assets/fca30f66-cb61-4656-be35-5b39afe43ffa")

        st.markdown("""
---

## Grouped Feature Selection
The main target variable is a binary flag AWon indicating whether ATeam won the game. To ensure fair modeling, the design matrix is built so that each underlying stat appears as a pair: one column for ATeam and one for BTeam.

Feature selection is performed using a **group-lasso–style approach**:
- Start from the curated set of base team features from sources listed above
- Run repeated fits of a logistic regression with L1-type penalty to select stable feature groups, selecting optimal alpha value across resamples
- Enforce symmetry by requiring that **if a stat is selected for A, the corresponding B stat is also included**; this prevents the model from exploiting arbitrary naming of the two teams
- The regularization strength selected here serves as a starting point; it is further varied as a hyperparameter during each individual model's tuning stage

The alpha vs. cross‑validated log loss and feature count plot is used to choose the regularization strength at the point where log loss is minimized, balancing predictive performance with model sparsity.
""", unsafe_allow_html=True)

        _img_small("https://github.com/user-attachments/assets/79f7e6f1-5c61-417f-b30b-b1faee9c219e")

        st.markdown("""
---

## Model Training and Hyperparameter Tuning

Five complementary models are trained using the selected feature set:

1. **LASSO Logistic Regression**
    - Binary logistic regression with L1 penalty to encourage sparsity with the grouped feature selection
2. **Elastic Net Logistic Regression**
    - Logistic regression with elastic net penalty (convex combination of L1 and L2) implemented via saga solver
    - Cross-validated grid search over values of inverse-regularization strength C and L1-ratio, with a diagnostic plot of validation log loss to select hyperparameters
3. **Gradient Boosting Classifier**
    - Tree-based model fit on the selected features
    - Optuna optimizes `n_estimators`, `learning_rate`, `max_depth`, `min_samples_leaf`, and `subsample` using log loss on held-out validation sets
4. **Neural Network**
    - Small fully connected network built with TensorFlow/Keras
    - Optuna tunes the `number of neurons per layer`, `learning rate`, `L2 regularization`, and `batch size`, using early stopping on validation loss
5. **Mixture of Experts (MoE)**
    - An ensemble of logistic regression "experts," each trained on a random subset of features, with a logistic gating function that learns which experts to trust for a given matchup
    - Optuna tunes the key hyperparameters: `alpha` (L1 regularization applied during grouped feature selection before experts are created), `n_experts` (number of individual expert models), `n_features` (how many randomly selected features each expert sees), `C_expert` (inverse L2 regularization strength within each expert's logistic regression), and `C_meta` (inverse L2 regularization strength for the gating function that combines expert outputs)

---

## Model Evaluation
Because the dataset is **relatively small** (roughly a few hundred games across tournaments), a single train/test split can give noisy estimates of performance. Each model is trained and evaluated across 100+ random stratified splits, reporting mean and standard deviation of test log loss.

The model performance plot compares out-of-sample log loss across each model type, as well as the baseline (defined as the predicted probability for every game being equal to the underlying prevalence in the data). The Mixture of Experts model is selected for this task since it achieves competitive log loss while maintaining strong generalization across splits.
""", unsafe_allow_html=True)

        _img_small("https://github.com/user-attachments/assets/634c745c-a553-4e7d-bd74-26408e633c3c")

        st.markdown("""
The calibration plot compares predicted win probabilities to actual outcomes for all five models, illustrating how well each model's probability estimates line up with observed frequencies across the probability range. No particular model shows any significant deviation from the baseline, indicating that all models produce reasonably well-calibrated probabilities.
""", unsafe_allow_html=True)

        _img_small("https://github.com/user-attachments/assets/aa1e5482-bebb-4e10-bcc4-186f41d27b7b")

        st.markdown("""
---

## Round Probabilities and Monte Carlo Simulation
To move from individual game predictions to full tournament projections, the pipeline constructs a **64×64 probability matrix** estimating the win probability for every possible team-vs-team matchup in the bracket. This matrix is built using the trained model's predictions, giving a complete picture of how any two teams in the field would fare head-to-head.

From there, a **Monte Carlo simulation** is run 10,000 times: each simulation plays out the entire tournament bracket by sampling game outcomes according to the 64×64 probability matrix. Across all 10,000 iterations, the pipeline records each team's probability of advancing to each subsequent round from the Round of 32 all the way through the championship.

---

## Seed-Based Baseline and Historical Upsets
Before trusting ML to drive bracket picks, the project establishes a **seeding-only baseline**:
- Fit a simple logistic model using only the log ratio of seeds, log(BSeed/ASeed), to predict ATeam win probability
- Use this simple model to **estimate typical win probabilities for every 1–16 vs. 1–16 pairing** and visualize them as a 16×16 probability matrix

Historical upset rates by seed matchup provide a reference for how aggressive ML-driven upsets should be.
""", unsafe_allow_html=True)

        _img_small("https://github.com/user-attachments/assets/8d0dd33f-2109-47d1-aa1f-5130e1da1974")

        st.markdown("""
---

## Bracket Construction Strategy
At the end of the day, the goal is to build a bracket that doesn't just rubber-stamp the higher seed in every game. Where's the fun in that? We want a bracket that's willing to go out on a limb and call upsets when the data says they're worth the risk.

The bracket logic revolves around a **lift score** that compares two perspectives on each game:
- The log‑odds from the full machine‑learning model
- The log‑odds implied by the simple, seed‑based baseline

When this lift is **negative**, it signals that the model thinks the underdog is more dangerous than the seeding alone would suggest, flagging a potential upset. For each early round, historical data is used to translate typical upset rates into a round‑specific lift cutoff:
- Historical games are ranked by lift (from most underdog‑friendly to most favorite‑friendly)
- The cutoff is chosen so that the fraction of games above that cutoff matches how often underdogs have actually won in that round
- In a future tournament, whenever a matchup's lift for the worse seed is below the relevant cutoff, the bracket intentionally picks the upset, **even if the favorite still has the higher raw win probability**
- This strategy naturally results in a distribution of first-round victories that aligns with history. 13-16 seeds are rarely, but not never, selected to win while 9-12 seeds are selected to win at a relatively common rate

This procedure is repeated separately for every round through the Elite Eight, giving each stage its own lift threshold. Because decisions are driven by these numeric thresholds rather than a fixed quota of upsets:
- The **number of predicted upsets varies** from year to year
- Over many tournaments, the average upset rate naturally lines up with history, but **any given bracket can lean more chaotic or more chalky** depending on how strongly the model disagrees with the seed baseline

In the Final Four and title game, the bracket simply takes the team with the higher modeled win probability.

### Historical Rate Adjustment
After implementing the initial bracket strategy, the results were **a bit too upset-happy,** especially in later rounds. The root cause was that the lift thresholds are calibrated against real historical games, where the teams that advance to later rounds include a fair amount of randomness. But in the hypothetical bracket, the **"analytically strong underdogs" from earlier rounds are the ones advancing**, which means later-round matchups are more likely to feature teams that are genuinely worthy of another upset pick. This creates a compounding effect where the bracket keeps picking upsets deeper into the tournament at a higher rate than history would support.

To correct for this, an **additional conservative factor** of `(`<code style="display:inline-flex;flex-direction:column;align-items:center;line-height:1;vertical-align:middle;font-size:0.85em;"><span style="border-bottom:1px solid #333;padding:0 2px;">2</span><span style="padding:0 2px;">3</span></code>`) · std(lift thresholds)` is added to the lift thresholds for Rounds 2–4 (Round of 32 through Elite Eight). This nudges the bar for calling an upset slightly higher in later rounds, dampening the compounding effect.

---

## Explaining Predictions with SHAP
To make the model's predictions interpretable at the matchup level, **SHAP (SHapley Additive exPlanations)** values are computed for every game. In the Bracket tab, hovering over any matchup surfaces a SHAP waterfall plot that breaks down exactly **why** the Mixture of Experts model arrived at a given win probability.
""", unsafe_allow_html=True)

        _img_small("https://github.com/user-attachments/assets/2ad785e7-2c27-476b-bfda-93c9d12bc45d")

        st.markdown("""
Take the North Carolina vs. Ole Miss example above. The model gives UNC a **62.8%** chance to win:
- **Blue bars** push the prediction toward the blue team (North Carolina); **red bars** push it toward the red team (Mississippi)
- **ADJOE (adjusted offensive efficiency) for UNC** is the biggest driver, shifting the prediction **11 percentage points** in their favor
- **ADJDE (adjusted defensive efficiency) for UNC** pushes **8 points back** toward Ole Miss, signaling that the Tar Heels are giving up buckets at a high rate
- **TALENT for UNC** contributes another **5 points** in their favor, an indicator of overall player talent level

### Prediction Storytelling
If you listen to college basketball analysts break down tournament matchups, their reasoning tends to be very specific and narrative-driven:
- *"The underdog will want to push the pace, force turnovers, and get transition buckets, which isn't the favorite's style of play and will make them uncomfortable"*
- *"I don't think they have the size and physicality to match up in the paint."*

These are stories built on particular box score traits, such as tempo, turnover rate, rebounding, and height. The SHAP plots tell a different story:
- The features that **dominate predictions** across nearly every matchup are **catch-all adjusted efficiency metrics**: adjusted offensive and defensive efficiency, overall efficiency margins, power ratings like BARTHAG, and resume-quality indicators like KPI and WAB
- The specific box score stats that analysts love to build narratives around don't often crack the top of the waterfall
- When I experimented with **removing the catch-all efficiency metrics** to force the model onto those more granular, "storytelling-friendly" features, the **predictions became noticeably less reliable**

Why? It likely comes down to what these composite metrics actually represent:
- A stat like adjusted offensive efficiency is already integrating a team's shooting, turnover rate, offensive rebounding, free throw rate, and the quality of defenses they've faced into a single tempo- and opponent-adjusted number
- Asking the model to re-derive that same signal from the raw components, especially with a small dataset, introduces noise without adding new information
- The composites are more predictive precisely because they're more stable and less susceptible to small-sample weirdness in any one stat

In short, the model's best predictions come from knowing ***how good*** a team is overall, not from dissecting ***how*** they're good. The narrative might be less colorful than what you'd hear on a studio show, but the probabilities are sharper for it.

---

## Thank You
If you've made it this far, I appreciate you taking an interest in the process behind all of this. I hope you enjoy the project, whether you're here to geek out over the modeling, steal some bracket strategy, or just see if the machine can beat your gut.

If you have any follow-up questions or just want to talk March Madness, feel free to reach out: **ethandavenport@utexas.edu**

*View the full source code on [GitHub](https://github.com/ethandavenport/March-Madness).*
""", unsafe_allow_html=True)