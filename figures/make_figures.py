"""Generate doc-ready figures for HW2 §2.2/§2.3 from verified EDA numbers.
All values are taken from the executed notebook (notebooks/eda.ipynb) and a
cross-checked awk recount of the rating columns. Figures are sized for a Word
page (content width ~6.5in) with readable fonts."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from pathlib import Path

OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.titlesize': 13, 'axes.titleweight': 'bold', 'axes.labelsize': 11,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})

VERT = ['Books', 'Movies_and_TV', 'CDs_and_Vinyl', 'Video_Games', 'Toys_and_Games']
LBL = {'Books': 'Books', 'Movies_and_TV': 'Movies & TV', 'CDs_and_Vinyl': 'CDs & Vinyl',
       'Video_Games': 'Video Games', 'Toys_and_Games': 'Toys & Games'}
NAVY = '#1f3a5f'

# ---- rating counts (awk recount, reconciled to notebook totals) ----
rating_counts = {  # rating 1..5
    'Books':          [1299569, 1068843, 2032688, 4582684, 20155541],
    'Movies_and_TV':  [1395560,  759530, 1248470, 2580157, 11174802],
    'CDs_and_Vinyl':  [ 185013,  138843,  280338,  657119,  3510758],
    'Video_Games':    [ 579987,  246344,  335256,  608231,  2785682],
    'Toys_and_Games': [1608590,  758252, 1090727, 1775628, 10819243],
}

# ---- metadata population rate (% of items with non-empty value) ----
pop_fields = ['title', 'description', 'features', 'categories', 'main_category', 'store', 'details', 'price']
pop = {  # order matches pop_fields
    'Books':          [100.0, 41.4, 88.3, 88.1, 99.8, 96.1, 93.3, 86.1],
    'Movies_and_TV':  [ 58.0, 46.5,  5.0, 57.9, 95.1, 52.5, 98.6, 35.3],
    'CDs_and_Vinyl':  [100.0, 64.9,  0.2,100.0, 99.9, 99.9, 96.6, 78.7],
    'Video_Games':    [100.0, 62.3, 71.2, 90.8, 92.0, 96.8, 97.5, 45.2],
    'Toys_and_Games': [100.0, 65.7, 80.1, 90.0, 97.5, 99.4, 97.7, 48.1],
}

# ---- 0-core vs 5-core retention ----
retention = {  # % interactions, % users, % items retained at 5-core
    'Books':          [32.6,  7.5, 11.1],
    'Movies_and_TV':  [43.4, 10.1, 26.5],
    'CDs_and_Vinyl':  [32.5,  7.1, 12.7],
    'Video_Games':    [17.9,  3.4, 18.7],
    'Toys_and_Games': [24.1,  5.3, 18.2],
}

# ---- 5-core shared-user overlap (diagonal = users in vertical) ----
users_5c = {'Books': 776370, 'Movies_and_TV': 657203, 'CDs_and_Vinyl': 123876,
            'Video_Games': 94762, 'Toys_and_Games': 432264}
pair_shared = {
    ('Books', 'Movies_and_TV'): 127188, ('Books', 'Toys_and_Games'): 87225,
    ('Movies_and_TV', 'Toys_and_Games'): 52794, ('Movies_and_TV', 'CDs_and_Vinyl'): 38769,
    ('Books', 'CDs_and_Vinyl'): 32027, ('Movies_and_TV', 'Video_Games'): 19615,
    ('Toys_and_Games', 'Video_Games'): 17202, ('Books', 'Video_Games'): 14478,
    ('Toys_and_Games', 'CDs_and_Vinyl'): 7750, ('CDs_and_Vinyl', 'Video_Games'): 4399,
}
pair_rich = {  # avg key-field richness for the pair
    ('Books', 'Movies_and_TV'): 0.653, ('Books', 'Toys_and_Games'): 0.809,
    ('Movies_and_TV', 'Toys_and_Games'): 0.697, ('Movies_and_TV', 'CDs_and_Vinyl'): 0.712,
    ('Books', 'CDs_and_Vinyl'): 0.824, ('Movies_and_TV', 'Video_Games'): 0.693,
    ('Toys_and_Games', 'Video_Games'): 0.848, ('Books', 'Video_Games'): 0.804,
    ('Toys_and_Games', 'CDs_and_Vinyl'): 0.868, ('CDs_and_Vinyl', 'Video_Games'): 0.863,
}

def kfmt(n):
    return f'{n/1000:.0f}k' if n >= 1000 else str(n)

# ===== Figure 1: rating distribution (100% stacked horizontal) =====
def fig_rating():
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    colors = ['#b2182b', '#ef8a62', '#f7d486', '#a6d96a', '#1a9850']  # 1..5 red->green
    order = VERT[::-1]
    y = np.arange(len(order))
    left = np.zeros(len(order))
    fracs = {v: np.array(rating_counts[v]) / sum(rating_counts[v]) * 100 for v in order}
    for r in range(5):
        widths = [fracs[v][r] for v in order]
        ax.barh(y, widths, left=left, color=colors[r], edgecolor='white', linewidth=0.6,
                label=f'{r+1}★')
        left += widths
    # annotate >=4 share at right edge
    for i, v in enumerate(order):
        pos = fracs[v][3] + fracs[v][4]
        ax.text(101, i, f'{pos:.0f}% ≥4★', va='center', ha='left', fontsize=9.5,
                color=NAVY, fontweight='bold')
    ax.set_yticks(y); ax.set_yticklabels([LBL[v] for v in order])
    ax.set_xlim(0, 100); ax.set_xlabel('share of interactions (%)')
    ax.set_title('Rating distribution per vertical', loc='left')
    ax.legend(ncol=5, loc='lower center', bbox_to_anchor=(0.5, -0.32),
              frameon=False, handlelength=1.1, columnspacing=1.2)
    ax.spines['left'].set_visible(False); ax.tick_params(left=False)
    fig.savefig(OUT / 'fig1_rating_dist.png'); plt.close(fig)

# ===== Figure 2: metadata population heatmap =====
def fig_population():
    M = np.array([pop[v] for v in VERT])
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    im = ax.imshow(M, cmap='YlGn', vmin=0, vmax=100, aspect='auto')
    ax.set_xticks(range(len(pop_fields)))
    ax.set_xticklabels(pop_fields, rotation=35, ha='right')
    ax.set_yticks(range(len(VERT))); ax.set_yticklabels([LBL[v] for v in VERT])
    for i in range(len(VERT)):
        for j in range(len(pop_fields)):
            val = M[i, j]
            ax.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=9,
                    color='white' if val > 60 else '#333333', fontweight='bold')
    ax.set_title('Metadata field population rate (% of items non-empty)', loc='left', fontsize=12.5, pad=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label('% populated', fontsize=9)
    fig.savefig(OUT / 'fig2_meta_population.png'); plt.close(fig)

# ===== Figure 3: 5-core retention =====
def fig_retention():
    metrics = ['interactions', 'users', 'items']
    cols = ['#2c7fb8', '#7fcdbb', '#c7e9b4']
    x = np.arange(len(VERT)); w = 0.26
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    for k, (m, c) in enumerate(zip(metrics, cols)):
        vals = [retention[v][k] for v in VERT]
        bars = ax.bar(x + (k-1)*w, vals, w, label=m, color=c, edgecolor='white')
        for b, val in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, val+0.6, f'{val:.0f}', ha='center',
                    va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([LBL[v] for v in VERT], rotation=12, ha='right')
    ax.set_ylabel('% retained at 5-core'); ax.set_ylim(0, 50)
    ax.set_title('Share of each vertical surviving the 5-core filter', loc='left')
    ax.legend(frameon=False, ncol=3, loc='upper right')
    fig.savefig(OUT / 'fig3_core_retention.png'); plt.close(fig)

# ===== Figure 4: 5-core pairwise overlap heatmap (lower triangle) =====
def fig_overlap():
    n = len(VERT)
    M = np.full((n, n), np.nan)
    for i, a in enumerate(VERT):
        for j, b in enumerate(VERT):
            if i > j:
                key = (a, b) if (a, b) in pair_shared else (b, a)
                M[i, j] = pair_shared[key]
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    cmap = plt.cm.YlOrRd.copy(); cmap.set_bad('#f5f5f5')
    im = ax.imshow(M, cmap=cmap, norm=mcolors.LogNorm(vmin=4000, vmax=130000), aspect='auto')
    ax.set_xticks(range(n)); ax.set_xticklabels([LBL[v] for v in VERT], rotation=35, ha='right')
    ax.set_yticks(range(n)); ax.set_yticklabels([LBL[v] for v in VERT])
    for i in range(n):
        for j in range(n):
            if not np.isnan(M[i, j]):
                v = M[i, j]
                ax.text(j, i, kfmt(int(v)), ha='center', va='center', fontsize=10,
                        color='white' if v > 40000 else '#333333', fontweight='bold')
    # highlight the two candidate pairs
    cand = {('Movies_and_TV', 'Books'): '#1565c0', ('Toys_and_Games', 'Books'): '#2e7d32'}
    for (a, b), col in cand.items():
        i, j = VERT.index(a), VERT.index(b)
        ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor=col, lw=3))
    # in-plot legend, placed in the empty upper-right triangle (right-aligned)
    ax.scatter([4.35], [0.15], marker='s', s=150, facecolor='none', edgecolor='#1565c0', linewidth=3, clip_on=False)
    ax.text(4.18, 0.15, 'Books → Movies & TV', va='center', ha='right', fontsize=10, color='#1565c0', fontweight='bold')
    ax.scatter([4.35], [0.80], marker='s', s=150, facecolor='none', edgecolor='#2e7d32', linewidth=3, clip_on=False)
    ax.text(4.18, 0.80, 'Books → Toys & Games', va='center', ha='right', fontsize=10, color='#2e7d32', fontweight='bold')
    ax.set_title('Shared users between verticals (5-core)', loc='left')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('shared users (log scale)', fontsize=9)
    fig.savefig(OUT / 'fig4_overlap_5core.png'); plt.close(fig)

# ===== Figure 5: overlap vs richness tradeoff =====
def fig_tradeoff():
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for (a, b), shared in pair_shared.items():
        rich = pair_rich[(a, b)]
        clears = shared >= 10000
        is_cand = {a, b} == {'Books', 'Movies_and_TV'} or {a, b} == {'Books', 'Toys_and_Games'}
        color = '#1a9850' if clears else '#d73027'
        ax.scatter(shared, rich, s=170 if is_cand else 90,
                   color=color, edgecolor=NAVY if is_cand else 'white',
                   linewidth=2 if is_cand else 0.8, zorder=3, alpha=0.95)
        # label candidate pairs to the upper-left so they stay clear of the edge/legend
        if is_cand:
            ax.annotate(f'{LBL[a]} → {LBL[b]}' if a == 'Books' else f'{LBL[b]} → {LBL[a]}',
                        (shared, rich), xytext=(-12, 10), textcoords='offset points',
                        ha='right', fontsize=9.5, fontweight='bold', color=NAVY)
    ax.axvline(10000, ls='--', color='#888888', lw=1.3)
    ax.text(9300, 0.70, '~10k floor', rotation=90, va='center', ha='right', fontsize=9, color='#888888')
    ax.set_xscale('log')
    ax.set_xlabel('shared users after 5-core  (log scale)')
    ax.set_ylabel('avg item-text richness (title/desc/categories)')
    ax.set_title('Candidate source→target pairs: overlap vs catalog richness', loc='left')
    legend = [Patch(facecolor='#1a9850', label='clears ~10k floor'),
              Patch(facecolor='#d73027', label='below floor')]
    ax.legend(handles=legend, frameon=False, loc='lower left')
    ax.grid(True, alpha=0.25)
    fig.savefig(OUT / 'fig5_pair_tradeoff.png'); plt.close(fig)

fig_rating(); fig_population(); fig_retention(); fig_overlap(); fig_tradeoff()
print('wrote figures to', OUT)
for p in sorted(OUT.glob('*.png')):
    print(' ', p.name, f'{p.stat().st_size/1024:.0f} KB')
