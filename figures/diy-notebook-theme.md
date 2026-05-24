---
name: diy-notebook-theme
description: "Locked figure theme for the DIY EMNLP paper — 'notebook theme'. All plotting parameters, colors, sizes, and layout preferences stored here. Reference this when creating any new figure."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 80807548-0e52-456e-81fb-693ba4c7a4c9
---

# DIY Paper Figure Theme — "Notebook Theme"

Locked on 2026-05-24 after iterating with the user. Use this for ALL paper figures going forward.

## Source code
`src/7_visualizations/plots/plot_pareto_frontier_and_average_rank.py` — function `plot_combined_average_rank()` is the reference implementation.

## Style origin
Based on `latex/references/generate_artifacts.py` (`set_style()` + `style_paper_axis()`), adapted with user preferences.

## Global rcParams
```python
import seaborn as sns
# Not using sns theme directly — manual rcParams only.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Cantarell", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"]
plt.rcParams["hatch.linewidth"] = 0.8
```

## Panel / Axes styling (style_paper_axis equivalent)
```python
PANEL_BG = "#f5f2e8"          # warm cream — slightly deeper than generate_artifacts' #faf9f4
GRID_COLOR = "#d7d9d4"
SPINE_COLOR = "#b3bac1"
INK = "#000000"               # all text is pure black

# Apply to each axes:
ax.set_facecolor(PANEL_BG)
ax.set_axisbelow(True)
ax.grid(axis="both", linestyle="-", linewidth=1.2, color=GRID_COLOR, alpha=0.82)
# Square grid cells: x-tick spacing = 1 (matches y spacing of 1 per row)
ax.xaxis.set_major_locator(MultipleLocator(1))
# Spines: only left + bottom visible
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)
for side in ["left", "bottom"]:
    ax.spines[side].set_color(SPINE_COLOR)
    ax.spines[side].set_linewidth(0.8)
```

## Color palette — muted, printable, no rainbow
```python
PALETTE = {
    "base":             "#C9CDD3",
    "baseline":         "#DCE7F0",   # pale sky — low profile
    "diy_show":         "#B8A9D4",   # muted lavender
    "diy_teach":        "#D4A76A",   # muted tan/amber (renamed to DIY-Train)
    "diy_teach_show":   "#7fb5a8",   # muted teal
    "diy_revise":       "#2f6f9f",   # navy
    "diy_teach_revise": "#d86565",   # muted rose
    "pending":          "#ECEEF2",
}
```

## Hatches per DIY method
```python
DIY_HATCH = {
    "diy_show":         "//",
    "diy_teach":        "\\\\",
    "diy_teach_show":   "xx",
    "diy_revise":       "..",
    "diy_teach_revise": "++",
}
```

## Bar styling
```python
height = 0.78
edgecolor = "#000000"
linewidth = 2.0
alpha = 0.96 (for pending bars only: 0.52)
```

## Font sizes (all black, normal weight unless noted)
```python
panel_title = 20.5       # model name badge
y_tick_labels = 19.5
x_tick_labels = 19
value_annotations = 18.5
legend = 19.5
```

## DIY method y-label treatment
- **Font weight:** bold
- **Font color:** black (#000000)
- **Background chip:** rounded box, same color as bar, alpha=0.30
```python
ytick.set_fontweight("bold")
ytick.set_color("#000000")
ytick.set_bbox(dict(
    boxstyle="round,pad=0.15,rounding_size=0.3",
    facecolor=bar_color,
    edgecolor=bar_color,
    alpha=0.30,
))
```
Baseline labels: normal weight, black, no chip.

## Model name badge
```python
ax.text(
    0.5, 1.02, model_label,
    transform=ax.transAxes,
    ha="center", va="bottom",
    fontsize=20.5, fontweight="normal", color="#000000",
    bbox=dict(
        boxstyle="round,pad=0.30",
        facecolor=PANEL_BG,
        edgecolor="#000000",
        linewidth=2.1,
        alpha=1.0,
    ),
)
```

## Legend
- Position: below the figure, `bbox_to_anchor=(0.5, -0.12)`
- `loc="lower center"`
- `frameon=True, fancybox=True, framealpha=1.0`
- `edgecolor="#000000"`, frame linewidth=2.1
- Background: same as PANEL_BG (`#f5f2e8`)
- Font: 19.5 pt, normal weight, black

## Figure dimensions
```python
fig_width = 25.0  # inches
fig_height = max(6.2, 0.28 * n_rows + 2.4)  # for 16 rows ≈ 6.9"
```

## Layout
```python
fig.subplots_adjust(left=0.085, right=0.985, bottom=0.08, top=0.95, wspace=0.45)
```

## Save settings
```python
fig.savefig(path, bbox_inches="tight", pad_inches=0.3, dpi=600, facecolor="white")
```
For the colm_style variant (full cream bg): `facecolor="#f5f2e8"`.

## Method naming
Use **Train** not Teach:
- DIY-Show, DIY-Train, DIY-Train-Show, DIY-Revise, DIY-Train-Revise
- Baselines: plain names (BBA, CAL, FairSteer, BiasEdit, LFTF, DPO, PEFT, DebiasLLMs, DebiasNLG, RSB, SelfDebias)

## Key user preferences (do NOT violate)
- Per-panel sorting (best rank at top in each model panel)
- No zebra stripes
- Square gridlines (x and y equally spaced)
- Thick gridlines (1.2 pt)
- All text black, normal weight (except DIY labels: bold)
- Colored highlight chips behind DIY labels only
- Legend below with cream background
- No title or axis label text like "Average rank (lower is better)" — removed per user request
- Hatches ON for DIY bars, none for baselines

**Why:** User iterated ~40 rounds to lock this aesthetic. Do not deviate without explicit instruction.

**How to apply:** Import these parameters into any new figure script. The colm_style variant uses `facecolor=PANEL_BG` for the full figure canvas; the main variant uses white outer + cream panels.

Related: [[diy-method-naming]] (Train not Teach), [[diy-repo-sync]] (push both repos after figure changes)
