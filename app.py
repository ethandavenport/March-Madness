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
    return bracket[mask].reset_index(drop=True)

def get_winner_seed(row):
    return int(row["Seed_A"]) if row["Selected"] == row["ATeamName"] else int(row["Seed_B"])

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] { background: #0a0a0f; color: #e8e0d0; }
[data-testid="stAppViewContainer"] { padding: 0; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding: 2rem 2rem 4rem 2rem; max-width: 100%; }

h1 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(2.5rem, 6vw, 5rem);
    letter-spacing: 0.08em;
    background: linear-gradient(135deg, #f5a623 0%, #f5d623 50%, #f5a623 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.1em 0;
    line-height: 1;
}
.subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    color: #666;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
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
.region-label {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.1rem;
    letter-spacing: 0.12em;
    color: #f5a623;
    text-align: center;
    margin-bottom: 0.75rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #2a2a3a;
}
.rounds-row { display: flex; gap: 6px; align-items: stretch; }
.rounds-row.rtl { flex-direction: row-reverse; }

.round-col {
    display: flex;
    flex-direction: column;
    gap: 0;
    flex: 1;
    min-width: 130px;
    max-width: 180px;
}
.round-header {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    color: #444;
    text-align: center;
    margin-bottom: 6px;
    height: 16px;
}
.game-spacer { flex-shrink: 0; }

.game {
    background: #13131e;
    border: 1px solid #1e1e30;
    border-radius: 6px;
    overflow: hidden;
    margin-bottom: 4px;
    flex-shrink: 0;
    transition: border-color 0.2s;
}
.game:hover { border-color: #f5a62344; }

/* Each team row: seed | name | model% | seed% */
.team {
    display: grid;
    grid-template-columns: 16px 1fr auto auto;
    align-items: center;
    padding: 5px 7px;
    gap: 5px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.70rem;
    color: #aaa;
    border-bottom: 1px solid #1a1a28;
    min-height: 28px;
    white-space: nowrap;
    overflow: hidden;
}
.team:last-child { border-bottom: none; }

.seed {
    font-size: 0.60rem;
    color: #f5a623;
    font-weight: 600;
    text-align: right;
}
.team-name { overflow: hidden; text-overflow: ellipsis; }
.pct {
    font-size: 0.60rem;
    font-weight: 600;
    padding: 1px 4px;
    border-radius: 3px;
    background: #0a0a0f44;
    min-width: 28px;
    text-align: center;
}
.pct-label {
    font-size: 0.50rem;
    color: #444;
    text-align: center;
    min-width: 28px;
}

/* prob header row inside game */
.prob-header {
    display: grid;
    grid-template-columns: 16px 1fr auto auto;
    gap: 5px;
    padding: 2px 7px 0 7px;
    font-size: 0.50rem;
    color: #333;
    font-family: 'DM Sans', sans-serif;
    letter-spacing: 0.05em;
}

/* Championship centre */
.champ-col {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-width: 150px;
    padding: 0 10px;
    gap: 8px;
}
.champ-label {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 0.85rem;
    letter-spacing: 0.12em;
    color: #444;
}
.champ-game {
    background: #13131e;
    border: 1px solid #f5a62355;
    border-radius: 8px;
    overflow: hidden;
    width: 100%;
    box-shadow: 0 0 30px #f5a62318;
}
.champion-banner {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    color: #f5a623;
    text-align: center;
    padding: 5px;
    background: #f5a62310;
    border-top: 1px solid #f5a62333;
}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("<h1>2025 March Madness</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Model Predictions · Mixture of Experts</p>', unsafe_allow_html=True)

# ── Game card ──────────────────────────────────────────────────────────────────

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
        '<span>Model</span><span>Seed</span>'
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
GAME_H = 76   # px: header(14) + 2 teams(28each) + margins
GAP    = 4

def region_html(region, rtl=False):
    direction = "rtl" if rtl else ""
    html = f'<div class="region-block">'
    html += f'<div class="region-label">{region} Region</div>'
    html += f'<div class="rounds-row {direction}">'

    for rnd in REGION_ROUNDS:
        games = games_for(region, rnd)
        short = ROUND_SHORT[rnd]
        html += f'<div class="round-col"><div class="round-header">{short}</div>'

        n = len(games)
        if n == 0:
            html += '</div>'
            continue

        r1_total = 8 * GAME_H + 7 * GAP
        if n < 8:
            slot_h   = r1_total / n
            spacer   = int((slot_h - GAME_H) / 2)
            between  = int(slot_h - GAME_H)
            html += f'<div class="game-spacer" style="height:{spacer}px"></div>'

        for idx, (_, row) in enumerate(games.iterrows()):
            html += game_card(row)
            if n < 8 and idx < len(games) - 1:
                html += f'<div class="game-spacer" style="height:{between}px"></div>'

        html += '</div>'

    html += '</div></div>'
    return html

# ── Championship centre ────────────────────────────────────────────────────────

def champ_html():
    ff_games   = bracket[bracket["Round"] == "Final Four"].reset_index(drop=True)
    champ_game = bracket[bracket["Round"] == "Championship"].reset_index(drop=True)

    html = '<div class="champ-col">'
    html += '<div class="champ-label">Final Four</div>'
    for _, row in ff_games.iterrows():
        html += game_card(row)

    if not champ_game.empty:
        row   = champ_game.iloc[0]
        fp    = row.get("FProb", float("nan"))
        sp    = row.get("SProb", float("nan"))
        s_a   = int(row["Seed_A"])
        s_b   = int(row["Seed_B"])
        if s_a <= s_b:
            model_a, model_b = fp, 1 - fp
            seed_a_p, seed_b_p = sp, 1 - sp
        else:
            model_a, model_b = 1 - fp, fp
            seed_a_p, seed_b_p = 1 - sp, sp

        html += '<div style="height:10px"></div>'
        html += '<div class="champ-label">Championship</div>'
        html += '<div class="champ-game">'
        html += '<div class="prob-header"><span></span><span></span><span>Model</span><span>Seed</span></div>'
        html += team_row_html(row["ATeamName"], s_a, model_a, seed_a_p)
        html += team_row_html(row["BTeamName"], s_b, model_b, seed_b_p)

        winner     = row["Selected"]
        win_seed   = get_winner_seed(row)
        html += f'<div class="champion-banner">🏆 {win_seed} {winner}</div>'
        html += '</div>'

    html += '</div>'
    return html

# ── Assemble ───────────────────────────────────────────────────────────────────
# W plays X in Final Four → left side: W + X
# Y plays Z in Final Four → right side: Y + Z
regions = set(bracket["Region_A"].dropna().unique()) | set(bracket["Region_B"].dropna().unique())
left_regions  = [r for r in ["W", "X"] if r in regions]
right_regions = [r for r in ["Y", "Z"] if r in regions]

html = '<div class="bracket-wrapper">'

html += '<div style="display:flex;flex-direction:column;gap:16px;flex:1;min-width:0;">'
for r in left_regions:
    html += region_html(r, rtl=False)
html += '</div>'

html += champ_html()

html += '<div style="display:flex;flex-direction:column;gap:16px;flex:1;min-width:0;">'
for r in right_regions:
    html += region_html(r, rtl=True)
html += '</div>'

html += '</div>'

st.markdown(html, unsafe_allow_html=True)

