import streamlit as st
import pandas as pd

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
            sort_key = pa
        else:
            top_name, top_seed = row["BTeamName"], int(row["Seed_B"])
            bot_name, bot_seed = row["ATeamName"], int(row["Seed_A"])
            sort_key = pb
        result.append({
            "row": row,
            "top_name": top_name, "top_seed": top_seed,
            "bot_name": bot_name, "bot_seed": bot_seed,
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
.block-container { padding: 2rem 1rem 4rem 1rem; max-width: 100%; }

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
/* Header cells mirror the column padding so labels sit over their boxes */
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
    border-bottom: 2px solid #d8d4cc;
}
/* champ placeholder — invisible, just holds space so headers don't stretch */
.round-header-champ-spacer {
    width: 700px;
    flex-shrink: 0;
    margin-left: -200px;
    margin-right: -200px;
}

/* ── Main bracket wrapper ── */
.bracket-wrapper {
    display: flex;
    align-items: stretch;
    width: 100%;
    /* intentionally no overflow:hidden — champ col bleeds over sides */
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

/* Round columns flex freely; first col (R64) has no left pad, others have pad for connectors */
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
    overflow: hidden;
    margin-bottom: 3px;
    flex-shrink: 0;
    transition: border-color 0.15s, box-shadow 0.15s;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}
.game:hover { border-color: #c97b0077; box-shadow: 0 2px 8px rgba(201,123,0,0.10); }

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
    width: 700px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-left: -200px;
    margin-right: -200px;
    z-index: 2;
    position: relative;
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

def team_row_html(name, seed, model_p, seed_p):
    mc = prob_color(model_p)
    sc = prob_color(seed_p)
    mp = f"{model_p*100:.0f}%" if not pd.isna(model_p) else "—"
    sp = f"{seed_p*100:.0f}%"  if not pd.isna(seed_p)  else "—"
    return (
        f'<div class="team">'
        f'<span class="seed">{seed}</span>'
        f'<span class="team-name">{name}</span>'
        f'<span class="pct" style="color:{mc};">{mp}</span>'
        f'<span class="pct" style="color:{sc};">{sp}</span>'
        f'</div>'
    )

def game_card_parts(top_name, top_seed, bot_name, bot_seed, fp, sp):
    """
    fp = P(favorite wins), where favorite = lower seed number.
    """
    if top_seed <= bot_seed:
        model_top, model_bot = fp, 1 - fp
        seed_top_p, seed_bot_p = sp, 1 - sp
    else:
        model_top, model_bot = 1 - fp, fp
        seed_top_p, seed_bot_p = 1 - sp, sp
    hdr = '<div class="prob-header"><span></span><span></span><span>Model</span><span>Seed</span></div>'
    return (
        f'<div class="game">{hdr}'
        f'{team_row_html(top_name, top_seed, model_top, seed_top_p)}'
        f'{team_row_html(bot_name, bot_seed, model_bot, seed_bot_p)}'
        f'</div>'
    )

def game_card(gd):
    row = gd["row"]
    fp  = row.get("FProb", float("nan"))
    sp  = row.get("SProb", float("nan"))
    return game_card_parts(gd["top_name"], gd["top_seed"], gd["bot_name"], gd["bot_seed"], fp, sp)

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
            for idx, gd in enumerate(games):
                col_html += game_card(gd)
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

    def ff_card_html(row, top_region):
        """Render FF card with the team from top_region on top."""
        if row is None:
            return ""
        fp = row.get("FProb", float("nan"))
        sp = row.get("SProb", float("nan"))
        sa, sb = int(row["Seed_A"]), int(row["Seed_B"])
        ra = str(row.get("Region_A", ""))
        rb = str(row.get("Region_B", ""))
        # Put team from top_region on top; if neither matches, fall back to lower seed
        if ra == top_region:
            top_n, top_s, bot_n, bot_s = row["ATeamName"], sa, row["BTeamName"], sb
        elif rb == top_region:
            top_n, top_s, bot_n, bot_s = row["BTeamName"], sb, row["ATeamName"], sa
        elif sa <= sb:
            top_n, top_s, bot_n, bot_s = row["ATeamName"], sa, row["BTeamName"], sb
        else:
            top_n, top_s, bot_n, bot_s = row["BTeamName"], sb, row["ATeamName"], sa
        return game_card_parts(top_n, top_s, bot_n, bot_s, fp, sp)

    html = '<div class="champ-col"><div class="champ-inner">'
    html += f'<div class="champ-ff-col">{ff_card_html(ff_left, top_left_region)}</div>'

    # Championship: left-side winner on top, right-side winner on bottom
    html += '<div class="champ-ncg-col">'
    if not champ_game.empty:
        row  = champ_game.iloc[0]
        fp   = row.get("FProb", float("nan"))
        sp   = row.get("SProb", float("nan"))
        sa, sb = int(row["Seed_A"]), int(row["Seed_B"])
        ra = str(row.get("Region_A", ""))
        rb = str(row.get("Region_B", ""))
        # Left-side team (W or X region) goes on top
        if ra in left_set:
            tn, ts, bn, bs = row["ATeamName"], sa, row["BTeamName"], sb
        elif rb in left_set:
            tn, ts, bn, bs = row["BTeamName"], sb, row["ATeamName"], sa
        elif sa <= sb:
            tn, ts, bn, bs = row["ATeamName"], sa, row["BTeamName"], sb
        else:
            tn, ts, bn, bs = row["BTeamName"], sb, row["ATeamName"], sa

        html += '<div class="champ-game">'
        html += '<div class="prob-header"><span></span><span></span><span>Model</span><span>Seed</span></div>'
        if ts <= bs:
            html += team_row_html(tn, ts, fp, sp)
            html += team_row_html(bn, bs, 1 - fp, 1 - sp)
        else:
            html += team_row_html(tn, ts, 1 - fp, 1 - sp)
            html += team_row_html(bn, bs, fp, sp)
        winner   = row["Selected"]
        win_seed = get_winner_seed(row)
        html += f'<div class="champion-banner">🏆 {win_seed} {winner}</div>'
        html += '</div>'
    html += '</div>'

    html += f'<div class="champ-ff-col">{ff_card_html(ff_right, top_right_region)}</div>'
    html += '</div></div>'
    return html

# ── Assemble ───────────────────────────────────────────────────────────────────
regions       = set(bracket["Region_A"].dropna().unique()) | set(bracket["Region_B"].dropna().unique())
left_regions  = [r for r in ["W", "X"] if r in regions]
right_regions = [r for r in ["Y", "Z"] if r in regions]

# Round header cells — padding mirrors the column padding so labels align with boxes.
# Column 0 (R64): no padding offset.
# Columns 1-3 (R32, S16, E8): each has COL_PAD on its inner (right for left-half, left for right-half) side.
def make_header_cells(rounds, rtl=False):
    cells = []
    for i, rnd in enumerate(rounds):
        if i == 0:
            pad = ""
        else:
            # Inner padding matches the column's connector padding
            side = "right" if rtl else "left"
            pad = f"padding-{side}:{COL_PAD}px;"
        cells.append(f'<div class="round-header-cell" style="{pad}">{ROUND_SHORT[rnd]}</div>')
    return "".join(cells)

hdr_left  = make_header_cells(REGION_ROUNDS, rtl=False)
hdr_right = make_header_cells(list(reversed(REGION_ROUNDS)), rtl=True)

headers_html = (
    f'<div class="round-headers-row">'
    f'<div class="round-headers-left">{hdr_left}</div>'
    f'<div class="round-header-champ-spacer"></div>'
    f'<div class="round-headers-right">{hdr_right}</div>'
    f'</div>'
)

html = headers_html
html += '<div class="bracket-wrapper">'
html += '<div class="side-half">'
for r in left_regions:
    html += region_html(r, rtl=False)
html += '</div>'
html += champ_html()
html += '<div class="side-half">'
for r in right_regions:
    html += region_html(r, rtl=True)
html += '</div>'
html += '</div>'

st.markdown(html, unsafe_allow_html=True)