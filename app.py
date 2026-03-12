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
    min-width: 160px;
    max-width: 220px;
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
    align-items: stretch;
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
    min-width: 160px;
    max-width: 220px;
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
    align-self: stretch;
    min-width: 380px;
    padding: 0 8px;
    gap: 0;
}
.champ-inner {
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: center;
    width: 100%;
    gap: 8px;
}
.champ-ff-col {
    display: flex;
    flex-direction: column;
    flex: 1;
    min-width: 160px;
    max-width: 180px;
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
    min-width: 160px;
    max-width: 180px;
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

# Canonical R1 slot order: each entry is the lower seed of that matchup (top to bottom)
# Slots: 0=1v16, 1=8v9, 2=5v12, 3=4v13, 4=6v11, 5=3v14, 6=7v10, 7=2v15
R1_TOP_SEEDS = [1, 8, 5, 4, 6, 3, 7, 2]

def get_top_seed(row):
    """Lower-numbered seed in a matchup (the 'favorite')."""
    return min(int(row["Seed_A"]), int(row["Seed_B"]))

def sort_games(games, rnd, region=None):
    """
    Sort games into correct bracket slot order.
    R1: order by R1_TOP_SEEDS position.
    R2+: order by which R1 slot the winning team came from.
      - R2 slot i = winner of R1 slots 2i and 2i+1
      - R3 slot i = winner of R2 slots 2i and 2i+1, etc.
    We approximate by sorting on the top seed, but using the R1_TOP_SEEDS
    ordering as the canonical slot map.
    """
    if len(games) == 0:
        return games

    if rnd == "Round 1":
        def r1_key(row):
            ts = get_top_seed(row)
            try:
                return R1_TOP_SEEDS.index(ts)
            except ValueError:
                return 99
        indices = sorted(range(len(games)), key=lambda i: r1_key(games.iloc[i]))
        return games.iloc[indices].reset_index(drop=True)

    # For R2+, the bracket slot is determined by which R1 slot pair fed into it.
    # The top seed of the game tells us which "branch" of the bracket it's in:
    # R2: winners of (1v16 & 8v9) → slot 0; (5v12 & 4v13) → slot 1; etc.
    # This maps to groups of R1_TOP_SEEDS: group 0=[1,8], group 1=[5,4], group 2=[6,3], group 3=[7,2]
    # For R3: group 0=[1,8,5,4], group 1=[6,3,7,2]
    # For R4: group 0=[1,8,5,4,6,3,7,2]
    rnd_to_group_size = {
        "Round 2": 2,
        "Round 3 (Sweet Sixteen)": 4,
        "Round 4 (Elite Eight)": 8,
        "Final Four": 8,
        "Championship": 8,
    }
    grp_sz = rnd_to_group_size.get(rnd, 2)

    def slot_key(row):
        ts = get_top_seed(row)
        # Find which R1 slot this seed would have won from
        # by checking which group of R1_TOP_SEEDS contains it or its likely predecessor
        for slot_idx, r1_seed in enumerate(R1_TOP_SEEDS):
            group_start = (slot_idx // grp_sz) * grp_sz
            group_seeds = R1_TOP_SEEDS[group_start:group_start + grp_sz]
            if ts in group_seeds:
                return group_start
        # fallback: sort by seed value
        return ts * 10

    indices = sorted(range(len(games)), key=lambda i: slot_key(games.iloc[i]))
    return games.iloc[indices].reset_index(drop=True)

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
    seed_a = int(row["Seed_A"])
    seed_b = int(row["Seed_B"])

    fp   = row.get("FProb", float("nan"))
    sp   = row.get("SProb", float("nan"))

    # Always display lower seed (favorite) on top
    if seed_a <= seed_b:
        name_top, seed_top, name_bot, seed_bot = row["ATeamName"], seed_a, row["BTeamName"], seed_b
        model_top, model_bot = fp, 1 - fp
        seed_top_p, seed_bot_p = sp, 1 - sp
    else:
        name_top, seed_top, name_bot, seed_bot = row["BTeamName"], seed_b, row["ATeamName"], seed_a
        model_top, model_bot = fp, 1 - fp
        seed_top_p, seed_bot_p = sp, 1 - sp

    header = (
        '<div class="prob-header">'
        '<span></span><span></span>'
        '<span>Model</span><span>Seed</span>'
        '</div>'
    )

    return (
        f'<div class="game">'
        f'{header}'
        f'{team_row_html(name_top, seed_top, model_top, seed_top_p)}'
        f'{team_row_html(name_bot, seed_bot, model_bot, seed_bot_p)}'
        f'</div>'
    )

# ── Region block ───────────────────────────────────────────────────────────────
REGION_ROUNDS = ["Round 1", "Round 2", "Round 3 (Sweet Sixteen)", "Round 4 (Elite Eight)"]
GAME_H  = 84   # px per game card (header + 2 rows + borders)
GAME_MB = 3    # margin-bottom on each game div
GAP_PX  = 3    # padding on each side of round-col (total 6px per col)

def compute_positions(n):
    """
    Return list of (top_y, center_y) for n games laid out in a column
    whose total height = R1_TOTAL. Games are evenly spaced in slots.
    """
    r1_total = 8 * GAME_H + 7 * GAME_MB
    if n == 8:
        positions = []
        y = 0
        for _ in range(n):
            positions.append((y, y + GAME_H / 2))
            y += GAME_H + GAME_MB
        return positions
    else:
        slot_h  = r1_total / n
        spacer  = (slot_h - GAME_H) / 2
        positions = []
        for i in range(n):
            top_y = i * slot_h + spacer
            positions.append((top_y, top_y + GAME_H / 2))
        return positions

def connector_svg_html(prev_positions, curr_positions, rtl, total_h):
    """
    Build an SVG that draws bracket lines connecting pairs of prev_positions
    to each curr_position. The SVG is 8px wide and total_h tall.
    For LTR columns: SVG sits on the LEFT edge of the current column.
    For RTL columns: SVG sits on the RIGHT edge of the current column.
    Lines go: vertical bar connecting two parent centers, then horizontal to child center.
    """
    w = 8
    lines = []
    for i, (_, child_cy) in enumerate(curr_positions):
        p1_cy = prev_positions[i * 2][1]
        p2_cy = prev_positions[i * 2 + 1][1]
        mid_y = (p1_cy + p2_cy) / 2

        if not rtl:
            # SVG is to the LEFT of current col, lines go left→right
            x_vert = 0   # vertical bar on left edge
            x_game = w   # horizontal reaches right edge (into current col)
        else:
            # SVG is to the RIGHT of current col, lines go right→left
            x_vert = w   # vertical bar on right edge
            x_game = 0   # horizontal reaches left edge (into current col)

        lines.append(f'<line x1="{x_vert}" y1="{p1_cy:.1f}" x2="{x_vert}" y2="{p2_cy:.1f}" stroke="#ccc8c0" stroke-width="1.5"/>')
        lines.append(f'<line x1="{x_vert}" y1="{mid_y:.1f}" x2="{x_game}" y2="{child_cy:.1f}" stroke="#ccc8c0" stroke-width="1.5"/>')

    side = "left:0" if not rtl else "right:0"
    return (
        f'<svg style="position:absolute;top:0;{side};width:{w}px;height:{total_h:.0f}px;'
        f'pointer-events:none;overflow:visible;" viewBox="0 0 {w} {total_h:.0f}" preserveAspectRatio="none">'
        + "".join(lines) + "</svg>"
    )

def region_html(region, rtl=False):
    direction = "rtl" if rtl else ""
    r1_total = 8 * GAME_H + 7 * GAME_MB
    total_h  = r1_total

    all_positions = []  # one list of (top_y, center_y) per round

    html = f'<div class="region-block"><div class="rounds-row {direction}">'

    for rnd_idx, rnd in enumerate(REGION_ROUNDS):
        games    = games_for(region, rnd)
        n        = len(games)
        positions = compute_positions(n) if n > 0 else []
        all_positions.append(positions)

        col_style = f'position:relative;height:{total_h}px;'
        col_html  = f'<div class="round-col" style="{col_style}">'

        if n == 0:
            col_html += '</div>'
            html += col_html
            continue

        slot_h  = total_h / n
        spacer  = int((slot_h - GAME_H) / 2)
        between = int(slot_h - GAME_H)

        col_html += f'<div class="game-spacer" style="height:{spacer}px"></div>'
        for idx, (_, row) in enumerate(games.iterrows()):
            col_html += game_card(row)
            if idx < n - 1:
                col_html += f'<div class="game-spacer" style="height:{between}px"></div>'

        # Attach connector SVG if this column has a previous round to connect from
        if rnd_idx > 0 and len(all_positions[rnd_idx - 1]) == n * 2:
            col_html += connector_svg_html(
                all_positions[rnd_idx - 1], positions, rtl, total_h
            )

        col_html += '</div>'
        html += col_html

    html += '</div></div>'
    return html

# ── Championship centre ────────────────────────────────────────────────────────

def champ_html():
    ff_games   = bracket[bracket["Round"] == "Final Four"].reset_index(drop=True)
    champ_game = bracket[bracket["Round"] == "Championship"].reset_index(drop=True)

# ── Championship centre ────────────────────────────────────────────────────────

def champ_html():
    ff_games   = bracket[bracket["Round"] == "Final Four"].reset_index(drop=True)
    champ_game = bracket[bracket["Round"] == "Championship"].reset_index(drop=True)

    # Identify which FF game involves left regions (W/X) vs right regions (Y/Z)
    left_region_set  = {"W", "X"}
    right_region_set = {"Y", "Z"}

    ff_left_game  = None  # W vs X semifinal → left of championship
    ff_right_game = None  # Y vs Z semifinal → right of championship

    for _, row in ff_games.iterrows():
        regions_in_game = {str(row.get("Region_A", "")), str(row.get("Region_B", ""))}
        if regions_in_game & left_region_set:
            ff_left_game = row
        else:
            ff_right_game = row

    # Fallback order
    if ff_left_game is None and len(ff_games) > 0:
        ff_left_game = ff_games.iloc[0]
    if ff_right_game is None and len(ff_games) > 1:
        ff_right_game = ff_games.iloc[1]

    html = '<div class="champ-col"><div class="champ-inner">'

    # Left FF (W vs X)
    html += '<div class="champ-ff-col">'
    if ff_left_game is not None:
        html += game_card(ff_left_game)
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
            model_top, model_bot = fp, 1 - fp
            seed_top_p, seed_bot_p = sp, 1 - sp
            name_top, seed_top = row["ATeamName"], s_a
            name_bot, seed_bot = row["BTeamName"], s_b
        else:
            model_top, model_bot = fp, 1 - fp
            seed_top_p, seed_bot_p = sp, 1 - sp
            name_top, seed_top = row["BTeamName"], s_b
            name_bot, seed_bot = row["ATeamName"], s_a

        html += '<div class="champ-game">'
        html += '<div class="prob-header"><span></span><span></span><span>Model</span><span>Seed</span></div>'
        html += team_row_html(name_top, seed_top, model_top, seed_top_p)
        html += team_row_html(name_bot, seed_bot, model_bot, seed_bot_p)

        winner   = row["Selected"]
        win_seed = get_winner_seed(row)
        html += f'<div class="champion-banner">🏆 {win_seed} {winner}</div>'
        html += '</div>'
    html += '</div>'

    # Right FF (Y vs Z)
    html += '<div class="champ-ff-col">'
    if ff_right_game is not None:
        html += game_card(ff_right_game)
    html += '</div>'

    html += '</div></div>'
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