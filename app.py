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
}
# Default layout if year not in map
_DEFAULT_LAYOUT = ("W", "X", "Y", "Z")


def _get_layout(year):
    """Return (TL, BL, TR, BR) region layout for a given year."""
    return _REGION_LAYOUTS.get(year, _DEFAULT_LAYOUT)


@st.cache_data
def _load_bracket_year(year):
    """Load bracket_YYYY.csv for a single year and build caches.
    Returns (bracket_df, shap_cache, results_cache)."""
    path = _bracket_files[year]
    df = pd.read_csv(path)
    df["Round"] = pd.Categorical(df["Round"], categories=ROUND_ORDER, ordered=True)
    df = df.sort_values("Round")

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

    return df, sc, rc


# Initialise with the most recent year (will be overridden in bracket tab)
bracket, shap_cache, results_cache = _load_bracket_year(_bracket_years[0])

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
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(2.5rem, 6vw, 5rem);
    letter-spacing: 0.08em;
    background: linear-gradient(135deg, #c97b00 0%, #e8a000 50%, #c97b00 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.1em 0;
    line-height: 1;
}
.subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    color: #888;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
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
    gap: 12px;
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
    margin-bottom: 3px;
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
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("<h1>March Madness</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Model Predictions · Mixture of Experts</p>', unsafe_allow_html=True)

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
[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover {
    border-color: #c97b00;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within {
    border-color: #c97b00;
    box-shadow: 0 0 0 2px #c97b0022;
}
</style>
""", unsafe_allow_html=True)

tab_bracket, tab_probs = st.tabs(["⛶  Bracket", "▦  Round Probabilities"])

with tab_bracket:
    # ── Year selector ──
    # Use Streamlit columns to place title + dropdown in a single row
    _bcol_l, _bcol_c, _bcol_r = st.columns([1, 6, 1])
    with _bcol_r:
        bracket_year = st.selectbox(
            "Year", _bracket_years,
            index=0, key="bracket_year", label_visibility="collapsed"
        )

    # Reload bracket data for selected year
    bracket, shap_cache, results_cache = _load_bracket_year(bracket_year)

    # Region layout for this year: (TL, BL, TR, BR)
    _layout = _get_layout(bracket_year)
    regions       = set(bracket["Region_A"].dropna().unique()) | set(bracket["Region_B"].dropna().unique())
    left_regions  = [r for r in [_layout[0], _layout[1]] if r in regions]   # TL, BL
    right_regions = [r for r in [_layout[2], _layout[3]] if r in regions]   # TR, BR

    # Title bar (rendered as HTML for consistent styling)
    with _bcol_c:
        st.markdown(
            f'<div style="font-family:\'DM Sans\',sans-serif;font-size:1.25rem;'
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
        adv_all["Season"] = adv_all["Season"].astype(int)
        available_years = sorted(adv_all["Season"].unique(), reverse=True)
    elif os.path.exists("adv_2025.csv"):
        adv_all = pd.read_csv("adv_2025.csv")
        adv_all["Season"] = 2025
        available_years = [2025]
    else:
        st.error("Could not find adv_all.csv or adv_2025.csv.")
        adv_all = None
        available_years = []

    if adv_all is not None:
        round_cols   = ["Round of 32", "Sweet 16", "Elite 8", "Final Four", "Championship", "Champion"]
        display_cols = ["TeamName", "SeedNum"] + round_cols
        col_labels = {
            "Team": "Team", "Seed": "Seed",
            "Round of 32": "R32", "Sweet 16": "Sweet 16", "Elite 8": "Elite Eight",
            "Final Four": "Final Four", "Championship": "Championship", "Champion": "Champion",
        }
        all_cols = ["Team", "Seed"] + round_cols

        # Serialize ALL years' data into JS — year switching handled entirely in JS,
        # no Streamlit rerun needed, no page reload, tab state preserved.
        all_years_data = {}
        for yr in available_years:
            adv_yr = adv_all[adv_all["Season"] == yr][display_cols].copy()
            adv_yr = adv_yr.rename(columns={"TeamName": "Team", "SeedNum": "Seed"})
            adv_yr = adv_yr.astype(object).where(adv_yr.notna(), None)
            all_years_data[int(yr)] = adv_yr.to_dict(orient="records")

        all_data_json = json.dumps(all_years_data)
        cols_json     = json.dumps(all_cols)
        labels_json   = json.dumps(col_labels)
        years_json    = json.dumps([int(y) for y in available_years])
        init_year_json = json.dumps(int(available_years[0]))

        n_rows     = max(len(v) for v in all_years_data.values())
        est_height = n_rows * 33 + 120

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
    font-size: 1.05rem; font-weight: 700; color: #c97b00;
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

<div id="outer"><div id="wrap"><table id="tbl">
  <colgroup id="colgroup"></colgroup>
  <thead id="thead"></thead>
  <tbody id="tbody"></tbody>
</table></div></div>

<script>
const ALL_DATA  = {all_data_json};
const ALL_COLS  = {cols_json};
const LABELS    = {labels_json};
const YEARS     = {years_json};
const TEXT_COLS = new Set(["Team", "Seed"]);

let curYear = {init_year_json};
let rows    = ALL_DATA[curYear];
let sortCol = "Final Four";
let sortAsc = false;

// Populate year dropdown
const sel = document.getElementById("year-select");
YEARS.forEach(y => {{
  const opt = document.createElement("option");
  opt.value = y;
  opt.textContent = y;
  if (y === curYear) opt.selected = true;
  sel.appendChild(opt);
}});

// Year change: just swap data and re-render, no reload
sel.addEventListener("change", () => {{
  curYear = parseInt(sel.value);
  rows    = ALL_DATA[curYear];
  document.getElementById("title").textContent = curYear + " Tournament";
  render();
}});

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

render();
</script>
</body>
</html>"""

        components.html(table_component, height=est_height, scrolling=False)