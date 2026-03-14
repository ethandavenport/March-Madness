import streamlit as st
import pandas as pd
import base64
import os

st.set_page_config(page_title="2025 March Madness", layout="wide")

bracket = pd.read_csv("bracket_2025.csv")

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

bracket["Round"] = pd.Categorical(bracket["Round"], categories=ROUND_ORDER, ordered=True)
bracket = bracket.sort_values("Round")

# ── SHAP cache — read directly from the SHAPPlot column in bracket_2025.csv ──
if "SHAPPlot" in bracket.columns:
    shap_cache = (
        bracket[["MatchID", "SHAPPlot"]]
        .dropna(subset=["SHAPPlot"])
        .set_index("MatchID")["SHAPPlot"]
        .to_dict()
    )
else:
    shap_cache = {}

# ── Actual results cache — keyed by MatchID ────────────────────────────────
has_results = "ActualA" in bracket.columns
if has_results:
    results_cache = (
        bracket[["MatchID","ATeamID","BTeamID",
                  "ActualA","ActualASeed","ActualATid",
                  "ActualB","ActualBSeed","ActualBTid"]]
        .set_index("MatchID")
        .to_dict(orient="index")
    )
else:
    results_cache = {}

# ── Helpers ────────────────────────────────────────────────────────────────────

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
            top_tid  = str(row["ATeamID"])
            bot_tid  = str(row["BTeamID"])
            sort_key = pa
        else:
            top_name, top_seed = row["BTeamName"], int(row["Seed_B"])
            bot_name, bot_seed = row["ATeamName"], int(row["Seed_A"])
            top_tid  = str(row["BTeamID"])
            bot_tid  = str(row["ATeamID"])
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
    width: 560px;
    z-index: 2;
    display: flex;
    align-items: center;
    justify-content: center;
}
.champ-inner {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 8px;
    width: 100%;
    padding: 0 4px;
}
.champ-ff-col  { flex: 1; min-width: 0; }
.champ-ncg-col { flex: 1; min-width: 0; }
.champ-game {
    background: #ffffff;
    border: 1px solid #c97b0055;
    border-radius: 8px;
    overflow: hidden;
    width: 100%;
    box-shadow: 0 0 18px rgba(201,123,0,0.09);
}
.champion-banner {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 0.76rem;
    letter-spacing: 0.14em;
    color: #c97b00;
    text-align: center;
    padding: 5px 8px;
    background: #fff8ec;
    border-top: 1px solid #c97b0033;
}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("<h1>2025 March Madness</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Model Predictions · Mixture of Experts</p>', unsafe_allow_html=True)

# ── Game card renderers ────────────────────────────────────────────────────────

def team_row_html(name, seed, model_p, seed_p, state="neutral"):
    """state: 'correct', 'wrong', or 'neutral'"""
    mc = prob_color(model_p)
    sc = prob_color(seed_p)
    mp = f"{model_p*100:.0f}%" if not pd.isna(model_p) else "—"
    sp = f"{seed_p*100:.0f}%"  if not pd.isna(seed_p)  else "—"
    name_class = f"team-name {state}" if state != "neutral" else "team-name"
    return (
        f'<div class="team">'
        f'<span class="seed">{seed}</span>'
        f'<span class="{name_class}">{name}</span>'
        f'<span class="pct" style="color:{mc};">{mp}</span>'
        f'<span class="pct" style="color:{sc};">{sp}</span>'
        f'</div>'
    )

def game_card_parts(top_name, top_seed, bot_name, bot_seed, fp, sp,
                    match_id=None, tooltip_side="right",
                    actual_top=None, actual_top_seed=None, actual_top_tid=None,
                    actual_bot=None, actual_bot_seed=None, actual_bot_tid=None,
                    top_tid=None, bot_tid=None):
    """
    actual_top/actual_bot: the two actual teams from df for this slot.
    actual_top = df's A team, actual_bot = df's B team (not top/bot ordered).
    We check whether each predicted team (top_tid, bot_tid) appears in
    {actual_top_tid, actual_bot_tid}. If yes → green. If no → strikethrough
    and show the actual team that should be there in red.
    """
    if top_seed <= bot_seed:
        model_top, model_bot = fp, 1 - fp
        seed_top_p, seed_bot_p = sp, 1 - sp
    else:
        model_top, model_bot = 1 - fp, fp
        seed_top_p, seed_bot_p = 1 - sp, sp
    hdr = '<div class="prob-header"><span></span><span></span><span>Model</span><span>Seed</span></div>'

    actual_tids = {t for t in [actual_top_tid, actual_bot_tid] if t is not None}

    def _state(tid):
        if not actual_tids:
            return "neutral"
        return "correct" if str(tid) in actual_tids else "wrong"

    def _missing_actual(tid):
        """Return the actual team that replaced this predicted team."""
        # The wrong predicted team should be replaced by whichever actual team
        # is not accounted for by the other predicted team.
        other_predicted = bot_tid if str(tid) == str(top_tid) else top_tid
        # If the other predicted team IS one of the actual teams, the missing
        # actual is the other one. If neither predicted team is in the actual
        # game, show whichever actual team has the closer seed.
        if actual_top_tid and str(other_predicted) == str(actual_top_tid):
            return actual_bot, actual_bot_seed
        if actual_bot_tid and str(other_predicted) == str(actual_bot_tid):
            return actual_top, actual_top_seed
        # Neither predicted team is correct — show the actual team on same side
        # (top predicted → show actual_top, bot predicted → show actual_bot)
        if str(tid) == str(top_tid):
            return actual_top, actual_top_seed
        return actual_bot, actual_bot_seed

    state_top = _state(top_tid)
    state_bot = _state(bot_tid)

    above_html = below_html = ""
    if state_top == "wrong":
        actual_name, actual_seed_val = _missing_actual(top_tid)
        if actual_name:
            seed_str = f"{int(actual_seed_val)} " if actual_seed_val is not None else ""
            above_html = f'<div class="actual-winner above">{seed_str}{actual_name}</div>'
    if state_bot == "wrong":
        actual_name, actual_seed_val = _missing_actual(bot_tid)
        if actual_name:
            seed_str = f"{int(actual_seed_val)} " if actual_seed_val is not None else ""
            below_html = f'<div class="actual-winner below">{seed_str}{actual_name}</div>'

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
        f'{team_row_html(top_name, top_seed, model_top, seed_top_p, state=state_top)}'
        f'{team_row_html(bot_name, bot_seed, model_bot, seed_bot_p, state=state_bot)}'
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
        at_n = _clean(rc.get("ActualA"))
        at_s = _clean(rc.get("ActualASeed"))
        at_t = _clean(rc.get("ActualATid"))
        ab_n = _clean(rc.get("ActualB"))
        ab_s = _clean(rc.get("ActualBSeed"))
        ab_t = _clean(rc.get("ActualBTid"))

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

def champ_html():
    ff_games   = bracket[bracket["Round"] == "Final Four"].reset_index(drop=True)
    champ_game = bracket[bracket["Round"] == "Championship"].reset_index(drop=True)

    left_set  = {"W", "X"}
    right_set = {"Y", "Z"}
    top_left_region  = "W"   # W is top region on left side → goes on top in left FF
    top_right_region = "Y"   # Y is top region on right side → goes on top in right FF

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

    def _get_actuals(mid):
        """Return (at_n,at_s,at_t, ab_n,ab_s,ab_t) from results_cache."""
        if not mid or mid not in results_cache:
            return None,None,None, None,None,None
        rc = results_cache[mid]
        def _clean(v):
            return None if (v is None or (isinstance(v, float) and pd.isna(v))) else v
        return (
            _clean(rc.get("ActualA")),    _clean(rc.get("ActualASeed")), _clean(rc.get("ActualATid")),
            _clean(rc.get("ActualB")),    _clean(rc.get("ActualBSeed")), _clean(rc.get("ActualBTid")),
        )

    def ff_card_html(row, top_region):
        if row is None:
            return ""
        fp = row.get("FProb", float("nan"))
        sp = row.get("SProb", float("nan"))
        sa, sb = int(row["Seed_A"]), int(row["Seed_B"])
        ra = str(row.get("Region_A", ""))
        rb = str(row.get("Region_B", ""))
        mid = str(row["MatchID"]) if "MatchID" in row and not pd.isna(row["MatchID"]) else None
        if ra == top_region:
            top_n, top_s, bot_n, bot_s = row["ATeamName"], sa, row["BTeamName"], sb
            top_id, bot_id = str(row["ATeamID"]), str(row["BTeamID"])
        elif rb == top_region:
            top_n, top_s, bot_n, bot_s = row["BTeamName"], sb, row["ATeamName"], sa
            top_id, bot_id = str(row["BTeamID"]), str(row["ATeamID"])
        elif sa <= sb:
            top_n, top_s, bot_n, bot_s = row["ATeamName"], sa, row["BTeamName"], sb
            top_id, bot_id = str(row["ATeamID"]), str(row["BTeamID"])
        else:
            top_n, top_s, bot_n, bot_s = row["BTeamName"], sb, row["ATeamName"], sa
            top_id, bot_id = str(row["BTeamID"]), str(row["ATeamID"])

        at_n,at_s,at_t, ab_n,ab_s,ab_t = _get_actuals(mid)
        return game_card_parts(top_n, top_s, bot_n, bot_s, fp, sp, mid,
                               actual_top=at_n, actual_top_seed=at_s, actual_top_tid=at_t,
                               actual_bot=ab_n, actual_bot_seed=ab_s, actual_bot_tid=ab_t,
                               top_tid=top_id, bot_tid=bot_id)

    html = '<div class="champ-col"><div class="champ-inner">'
    html += f'<div class="champ-ff-col">{ff_card_html(ff_left, top_left_region)}</div>'

    # Championship
    html += '<div class="champ-ncg-col">'
    if not champ_game.empty:
        row  = champ_game.iloc[0]
        fp   = row.get("FProb", float("nan"))
        sp   = row.get("SProb", float("nan"))
        sa, sb = int(row["Seed_A"]), int(row["Seed_B"])
        ra = str(row.get("Region_A", ""))
        rb = str(row.get("Region_B", ""))
        mid = str(row["MatchID"]) if "MatchID" in row and not pd.isna(row["MatchID"]) else None
        if ra in left_set:
            tn, ts, bn, bs = row["ATeamName"], sa, row["BTeamName"], sb
            t_id, b_id = str(row["ATeamID"]), str(row["BTeamID"])
        elif rb in left_set:
            tn, ts, bn, bs = row["BTeamName"], sb, row["ATeamName"], sa
            t_id, b_id = str(row["BTeamID"]), str(row["ATeamID"])
        elif sa <= sb:
            tn, ts, bn, bs = row["ATeamName"], sa, row["BTeamName"], sb
            t_id, b_id = str(row["ATeamID"]), str(row["BTeamID"])
        else:
            tn, ts, bn, bs = row["BTeamName"], sb, row["ATeamName"], sa
            t_id, b_id = str(row["BTeamID"]), str(row["ATeamID"])

        at_n,at_s,at_t, ab_n,ab_s,ab_t = _get_actuals(mid)

        card = game_card_parts(tn, ts, bn, bs, fp, sp, mid,
                               actual_top=at_n, actual_top_seed=at_s, actual_top_tid=at_t,
                               actual_bot=ab_n, actual_bot_seed=ab_s, actual_bot_tid=ab_t,
                               top_tid=t_id, bot_tid=b_id)

        # Insert champion banner before closing </div>
        winner   = row["Selected"]
        win_seed = get_winner_seed(row)
        banner   = f'<div class="champion-banner">🏆 {win_seed} {winner}</div>'
        card     = card[:-len("</div>")] + banner + "</div>"
        card     = card.replace('<div class="game">', '<div class="champ-game game">', 1)
        html    += card
    html += '</div>'

    html += f'<div class="champ-ff-col">{ff_card_html(ff_right, top_right_region)}</div>'
    html += '</div></div>'
    return html

# ── Assemble ───────────────────────────────────────────────────────────────────
regions       = set(bracket["Region_A"].dropna().unique()) | set(bracket["Region_B"].dropna().unique())
left_regions  = [r for r in ["W", "X"] if r in regions]
right_regions = [r for r in ["Y", "Z"] if r in regions]

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

tab_bracket, tab_probs = st.tabs(["🏀 Bracket", "📊 Round Probabilities"])

with tab_bracket:
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
    html += champ_html()
    html += f'<div class="side-half" style="padding-left:{SIDE_PAD}px;">'
    for r in right_regions:
        html += region_html(r, rtl=True)
    html += '</div>'
    html += '</div>'

    st.markdown(html, unsafe_allow_html=True)

with tab_probs:
    adv_path = "adv_2025.csv"
    if not os.path.exists(adv_path):
        st.error(f"Could not find {adv_path}. Make sure it's in the same directory as app.py.")
    else:
        adv = pd.read_csv(adv_path)

        round_cols   = ["Round of 32", "Sweet 16", "Elite 8", "Final Four", "Championship", "Champion"]
        display_cols = ["TeamName", "SeedNum"] + round_cols
        adv_display  = adv[display_cols].copy().rename(columns={"TeamName": "Team", "SeedNum": "Seed"})

        col_labels = {
            "Team": "Team", "Seed": "Seed",
            "Round of 32": "R32", "Sweet 16": "S16", "Elite 8": "E8",
            "Final Four": "FF", "Championship": "Championship", "Champion": "Champion",
        }
        all_cols = ["Team", "Seed"] + round_cols

        # Sort controls
        sc1, sc2, _ = st.columns([2, 1.5, 6])
        with sc1:
            sort_col = st.selectbox(
                "Sort by",
                options=all_cols,
                format_func=lambda c: col_labels[c],
                index=all_cols.index("Champion"),
                key="adv_sort_col",
            )
        with sc2:
            default_asc = sort_col in ("Team", "Seed")
            sort_asc = st.selectbox(
                "Order",
                options=[True, False],
                format_func=lambda x: "Ascending ↑" if x else "Descending ↓",
                index=0 if default_asc else 1,
                key="adv_sort_asc",
            )

        adv_sorted = adv_display.sort_values(sort_col, ascending=sort_asc)

        def pct_to_style(val):
            if pd.isna(val):
                return "background:transparent;"
            v = float(val)
            r = int(255 - (255 - 201) * v)
            g = int(253 - (253 - 123) * v)
            b = int(245 - (245 -   0) * v)
            text = "#fff" if v > 0.5 else "#333"
            return f"background:rgb({r},{g},{b});color:{text};"

        def fmt_pct(val):
            return "—" if pd.isna(val) else f"{float(val)*100:.1f}%"

        # Build header
        header_cells = ""
        for col in all_cols:
            label  = col_labels[col]
            active = col == sort_col
            arrow  = ("↓" if not sort_asc else "↑") if active else ""
            align  = "left" if col in ("Team", "Seed") else "center"
            color  = "color:#c97b00;" if active else ""
            header_cells += (
                f'<th style="padding:9px 14px;text-align:{align};font-family:\'DM Sans\',sans-serif;'
                f'font-size:0.70rem;font-weight:800;letter-spacing:0.07em;text-transform:uppercase;'
                f'border-bottom:2px solid #ddd9d2;white-space:nowrap;{color}">'
                f'{label}{" " + arrow if arrow else ""}</th>'
            )

        # Build rows
        rows_html = ""
        for i, (_, row) in enumerate(adv_sorted.iterrows()):
            row_bg = "#fff" if i % 2 == 0 else "#faf8f4"
            cells  = ""
            for col in all_cols:
                val = row[col]
                if col == "Team":
                    cells += (
                        f'<td style="padding:7px 14px;font-family:\'DM Sans\',sans-serif;'
                        f'font-size:0.83rem;font-weight:500;white-space:nowrap;">{val}</td>'
                    )
                elif col == "Seed":
                    cells += (
                        f'<td style="padding:7px 14px;text-align:center;font-family:\'DM Sans\',sans-serif;'
                        f'font-size:0.83rem;color:#c97b00;font-weight:700;">{int(val)}</td>'
                    )
                else:
                    cs = pct_to_style(val)
                    cells += (
                        f'<td style="padding:7px 14px;text-align:center;font-family:\'DM Sans\',sans-serif;'
                        f'font-size:0.83rem;{cs}">{fmt_pct(val)}</td>'
                    )
            rows_html += f'<tr style="background:{row_bg};">{cells}</tr>'

        table_html = f"""
<style>
#adv-wrap {{ overflow-x:auto; border-radius:8px; border:1px solid #ddd9d2;
             box-shadow:0 2px 10px rgba(0,0,0,0.07); margin-top:8px; }}
#adv-tbl  {{ border-collapse:collapse; width:100%; background:#fff; }}
#adv-tbl thead {{ background:#faf8f4; }}
#adv-tbl tbody tr:hover {{ filter:brightness(0.96); }}
</style>
<div id="adv-wrap">
  <table id="adv-tbl">
    <thead><tr>{header_cells}</tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>"""

        st.markdown(table_html, unsafe_allow_html=True)