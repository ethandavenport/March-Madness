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
    """Red-green color for a probability in [0,1]. 0%=red, 50%=neutral, 100%=green."""
    if pd.isna(p):
        return "#555555"
    r = int(220 * (1 - p))
    g = int(220 * p)
    return f"rgb({r},{g},60)"

def games_for(region, round_name):
    mask = (
        ((bracket["Region_A"] == region) | (bracket["Region_B"] == region)) &
        (bracket["Round"] == round_name)
    )
    games = bracket[mask].reset_index(drop=True)
    return sort_games(games, round_name)

def get_winner_seed(row):
    return int(row["Seed_A"]) if row["Selected"] == row["ATeamName"] else int(row["Seed_B"])

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"] { background: #f5f3ef; color: #1a1a1a; }
[data-testid="stAppViewContainer"] { padding: 0; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding: 2rem 2rem 4rem 2rem; max-width: 100%; }

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
    margin-bottom: 2rem;
}

/* ── Round headers row (single strip above all columns) ── */
.round-headers-row {
    display: flex;
    gap: 0;
    margin-bottom: 4px;
}
.round-headers-left, .round-headers-right {
    display: flex;
    flex: 1;
    min-width: 0;
}
.round-headers-right { flex-direction: row-reverse; }
.round-header-cell {
    flex: 1;
    min-width: 140px;
    max-width: 200px;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 0.85rem;
    letter-spacing: 0.12em;
    color: #888;
    text-align: center;
    padding-bottom: 6px;
    border-bottom: 2px solid #d0ccc4;
    margin: 0 3px;
}
.round-header-champ {
    min-width: 340px;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 0.85rem;
    letter-spacing: 0.12em;
    color: #888;
    text-align: center;
    padding-bottom: 6px;
    border-bottom: 2px solid #d0ccc4;
    display: flex;
    gap: 0;
    justify-content: center;
}
.round-header-champ span {
    flex: 1;
    text-align: center;
    padding: 0 4px;
}

.bracket-wrapper {
    display: flex;
    gap: 0;
    align-items: flex-start;
    width: 100%;
    overflow-x: auto;
    padding-bottom: 1rem;
}

.region-block { flex: 1; min-width: 0; }
.rounds-row { display: flex; gap: 0; align-items: stretch; }
.rounds-row.rtl { flex-direction: row-reverse; }

/* Each round column: position:relative so we can draw SVG connector lines */
.round-col {
    display: flex;
    flex-direction: column;
    gap: 0;
    flex: 1;
    min-width: 140px;
    max-width: 200px;
    position: relative;
    padding: 0 3px;
}
.game-spacer { flex-shrink: 0; }

.game {
    background: #ffffff;
    border: 1px solid #ddd9d2;
    border-radius: 7px;
    overflow: hidden;
    margin-bottom: 3px;
    flex-shrink: 0;
    transition: border-color 0.2s, box-shadow 0.2s;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.game:hover { border-color: #c97b0088; box-shadow: 0 2px 8px rgba(201,123,0,0.12); }

/* Each team row: seed | name | model% | seed% */
.team {
    display: grid;
    grid-template-columns: 18px 1fr auto auto;
    align-items: center;
    padding: 6px 8px;
    gap: 6px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem;
    color: #333;
    border-bottom: 1px solid #eee9e2;
    min-height: 30px;
    white-space: nowrap;
    overflow: hidden;
}
.team:last-child { border-bottom: none; }

.seed {
    font-size: 0.65rem;
    color: #c97b00;
    font-weight: 700;
    text-align: right;
}
.team-name { overflow: hidden; text-overflow: ellipsis; font-weight: 500; }
.pct {
    font-size: 0.65rem;
    font-weight: 700;
    padding: 2px 5px;
    border-radius: 4px;
    background: #f5f0e8;
    min-width: 34px;
    text-align: center;
}

/* prob header row inside game — two labeled columns */
.prob-header {
    display: grid;
    grid-template-columns: 18px 1fr auto auto;
    gap: 6px;
    padding: 3px 8px 1px 8px;
    font-size: 0.58rem;
    color: #999;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    background: #faf8f4;
    border-bottom: 1px solid #eee9e2;
}

/* Championship centre */
.champ-col {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-width: 340px;
    padding: 0 8px;
    gap: 0;
}
.champ-inner {
    display: flex;
    flex-direction: row;
    align-items: flex-start;
    justify-content: center;
    width: 100%;
    gap: 8px;
}
.champ-ff-col {
    display: flex;
    flex-direction: column;
    flex: 1;
    min-width: 140px;
    max-width: 160px;
}
.champ-label {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 0.80rem;
    letter-spacing: 0.12em;
    color: #999;
    text-align: center;
    margin-bottom: 4px;
}
.champ-ncg-col {
    display: flex;
    flex-direction: column;
    align-items: center;
    flex: 1;
    min-width: 150px;
    max-width: 170px;
}
.champ-game {
    background: #ffffff;
    border: 1px solid #c97b0066;
    border-radius: 8px;
    overflow: hidden;
    width: 100%;
    box-shadow: 0 0 20px rgba(201,123,0,0.10);
}
.champion-banner {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 0.80rem;
    letter-spacing: 0.15em;
    color: #c97b00;
    text-align: center;
    padding: 5px 8px;
    background: #fff8ec;
    border-top: 1px solid #c97b0033;
}

/* SVG connector lines */
.connector-svg {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
    overflow: visible;
}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("<h1>2025 March Madness</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Model Predictions · Mixture of Experts</p>', unsafe_allow_html=True)

# ── Game card ──────────────────────────────────────────────────────────────────

# Canonical bracket seed order for Round 1 (top to bottom)
R1_SEED_ORDER = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]

def sort_games(games, rnd):
    """Sort games in bracket order based on the seeds involved."""
    if rnd == "Round 1":
        def sort_key(row):
            top_seed = min(int(row["Seed_A"]), int(row["Seed_B"]))
            try:
                return R1_SEED_ORDER.index(top_seed)
            except ValueError:
                return 99
        return games.sort_values(by=games.columns.tolist(), key=lambda _: games.apply(sort_key, axis=1))
    else:
        # Higher rounds: sort by the top seed (favorite) to preserve alignment
        def sort_key2(row):
            return min(int(row["Seed_A"]), int(row["Seed_B"]))
        return games.iloc[games.apply(sort_key2, axis=1).argsort().values]

def team_row_html(name, seed, model_p, seed_p):
    """One team row: seed | name | model prob | seed prob, both color-coded."""
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

def game_card(row):
    name_a = row["ATeamName"]
    name_b = row["BTeamName"]
    seed_a = int(row["Seed_A"])
    seed_b = int(row["Seed_B"])

    fp   = row.get("FProb", float("nan"))
    sp   = row.get("SProb", float("nan"))

    # Model probability from each team's perspective
    # FProb = P(favorite wins); favorite = lower seed
    if seed_a <= seed_b:
        model_a, model_b = fp, 1 - fp
        seed_a_p, seed_b_p = sp, 1 - sp
    else:
        model_a, model_b = 1 - fp, fp
        seed_a_p, seed_b_p = 1 - sp, sp

    header = (
        '<div class="prob-header">'
        '<span></span><span></span>'
        '<span>Model%</span><span>Seed%</span>'
        '</div>'
    )

    return (
        f'<div class="game">'
        f'{header}'
        f'{team_row_html(name_a, seed_a, model_a, seed_a_p)}'
        f'{team_row_html(name_b, seed_b, model_b, seed_b_p)}'
        f'</div>'
    )

# ── Region block ───────────────────────────────────────────────────────────────
REGION_ROUNDS = ["Round 1", "Round 2", "Round 3 (Sweet Sixteen)", "Round 4 (Elite Eight)"]
GAME_H = 84   # px: header(18) + 2 teams(30ea) + margins
GAP    = 3

def region_html(region, rtl=False):
    direction = "rtl" if rtl else ""
    html = f'<div class="region-block">'
    html += f'<div class="rounds-row {direction}">'

    for rnd_idx, rnd in enumerate(REGION_ROUNDS):
        games = games_for(region, rnd)

        n = len(games)
        r1_total = 8 * GAME_H + 7 * GAP

        # Build game positions for connector SVG
        game_positions = []  # (top_y, center_y) per game

        col_html = '<div class="round-col" style="position:relative;">'

        if n == 0:
            col_html += '</div>'
            html += col_html
            continue

        if n < 8:
            slot_h  = r1_total / n
            spacer  = int((slot_h - GAME_H) / 2)
            between = int(slot_h - GAME_H)

            col_html += f'<div class="game-spacer" style="height:{spacer}px"></div>'
            cur_y = spacer
            for idx, (_, row) in enumerate(games.iterrows()):
                game_positions.append(cur_y + GAME_H / 2)
                col_html += game_card(row)
                cur_y += GAME_H + 3  # margin-bottom 3px
                if idx < len(games) - 1:
                    col_html += f'<div class="game-spacer" style="height:{between}px"></div>'
                    cur_y += between
        else:
            cur_y = 0
            for _, row in games.iterrows():
                game_positions.append(cur_y + GAME_H / 2)
                col_html += game_card(row)
                cur_y += GAME_H + 3

        # SVG bracket connectors: draw lines from this column's games to next round
        # Only draw on non-last rounds and when this column feeds pairs into the next
        if rnd_idx > 0 and n > 0:
            total_h = r1_total
            svg = f'<svg class="connector-svg" viewBox="0 0 6 {total_h}" preserveAspectRatio="none" style="position:absolute;top:0;{"right:100%" if not rtl else "left:100%"};width:6px;height:100%;overflow:visible;">'
            # For each pair of games in the PREVIOUS round, draw a bracket line to this game
            # We draw: vertical line connecting two parent game centers, then horizontal line to this game
            for i, cy in enumerate(game_positions):
                if rnd_idx == 1:
                    p1 = (i * 2) * (r1_total / 8) + GAME_H / 2
                    p2 = (i * 2 + 1) * (r1_total / 8) + GAME_H / 2
                else:
                    prev_n = n * 2
                    prev_slot = r1_total / prev_n
                    p1 = int((prev_slot - GAME_H) / 2) + GAME_H / 2 + i * 2 * prev_slot
                    p2 = int((prev_slot - GAME_H) / 2) + GAME_H / 2 + (i * 2 + 1) * prev_slot

                # vertical line between the two parents' midpoints
                x_side = 0 if not rtl else 6
                svg += f'<line x1="{x_side}" y1="{p1:.1f}" x2="{x_side}" y2="{p2:.1f}" stroke="#ccc8c0" stroke-width="1.5"/>'
                # horizontal line from vertical midpoint to game
                mid_y = (p1 + p2) / 2
                x_game = 6 if not rtl else 0
                svg += f'<line x1="{x_side}" y1="{mid_y:.1f}" x2="{x_game}" y2="{cy:.1f}" stroke="#ccc8c0" stroke-width="1.5"/>'
            svg += '</svg>'
            col_html += svg

        col_html += '</div>'
        html += col_html

    html += '</div></div>'
    return html

# ── Championship centre ────────────────────────────────────────────────────────

def champ_html():
    ff_games   = bracket[bracket["Round"] == "Final Four"].reset_index(drop=True)
    champ_game = bracket[bracket["Round"] == "Championship"].reset_index(drop=True)

    # Split FF games: left game (W region winner) vs right game (Y region winner)
    # Typically 2 FF games; first feeds from left side, second from right side
    ff_left  = ff_games.iloc[[0]] if len(ff_games) > 0 else ff_games
    ff_right = ff_games.iloc[[1]] if len(ff_games) > 1 else ff_games.iloc[:0]

    html = '<div class="champ-col">'
    html += '<div class="champ-inner">'

    # Left FF game
    html += '<div class="champ-ff-col">'
    for _, row in ff_left.iterrows():
        html += game_card(row)
    html += '</div>'

    # Center: Championship
    html += '<div class="champ-ncg-col">'
    if not champ_game.empty:
        row  = champ_game.iloc[0]
        fp   = row.get("FProb", float("nan"))
        sp   = row.get("SProb", float("nan"))
        s_a  = int(row["Seed_A"])
        s_b  = int(row["Seed_B"])
        if s_a <= s_b:
            model_a, model_b = fp, 1 - fp
            seed_a_p, seed_b_p = sp, 1 - sp
        else:
            model_a, model_b = 1 - fp, fp
            seed_a_p, seed_b_p = 1 - sp, sp

        html += '<div class="champ-game">'
        html += '<div class="prob-header"><span></span><span></span><span>Model</span><span>Seed%</span></div>'
        html += team_row_html(row["ATeamName"], s_a, model_a, seed_a_p)
        html += team_row_html(row["BTeamName"], s_b, model_b, seed_b_p)

        winner   = row["Selected"]
        win_seed = get_winner_seed(row)
        html += f'<div class="champion-banner">🏆 {win_seed} {winner}</div>'
        html += '</div>'
    html += '</div>'

    # Right FF game
    html += '<div class="champ-ff-col">'
    for _, row in ff_right.iterrows():
        html += game_card(row)
    html += '</div>'

    html += '</div>'  # champ-inner
    html += '</div>'  # champ-col
    return html

# ── Assemble ───────────────────────────────────────────────────────────────────
regions = set(bracket["Region_A"].dropna().unique()) | set(bracket["Region_B"].dropna().unique())
left_regions  = [r for r in ["W", "X"] if r in regions]
right_regions = [r for r in ["Y", "Z"] if r in regions]

# Single round-header strip
hdr_left_cells  = "".join(f'<div class="round-header-cell">{ROUND_SHORT[r]}</div>' for r in REGION_ROUNDS)
hdr_right_cells = "".join(f'<div class="round-header-cell">{ROUND_SHORT[r]}</div>' for r in reversed(REGION_ROUNDS))
hdr_champ = (
    '<div class="round-header-champ">'
    '<span>FF</span><span>Championship</span><span>FF</span>'
    '</div>'
)
headers_html = (
    f'<div class="round-headers-row">'
    f'<div class="round-headers-left">{hdr_left_cells}</div>'
    f'{hdr_champ}'
    f'<div class="round-headers-right">{hdr_right_cells}</div>'
    f'</div>'
)

html = headers_html
html += '<div class="bracket-wrapper">'

html += '<div style="display:flex;flex-direction:column;gap:12px;flex:1;min-width:0;">'
for r in left_regions:
    html += region_html(r, rtl=False)
html += '</div>'

html += champ_html()

html += '<div style="display:flex;flex-direction:column;gap:12px;flex:1;min-width:0;">'
for r in right_regions:
    html += region_html(r, rtl=True)
html += '</div>'

html += '</div>'

st.markdown(html, unsafe_allow_html=True)