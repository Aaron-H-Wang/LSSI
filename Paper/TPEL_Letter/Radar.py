import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

# ---- Font: Times New Roman, 10 pt (uniform across the whole figure) ----
# On your machine "Times New Roman" is found directly. Liberation Serif is a
# metric-compatible fallback so the figure renders identically where TNR is absent.
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
plt.rcParams['font.size'] = 10
plt.rcParams['mathtext.fontset'] = 'stix'  # serif-style math to match TNR
plt.rcParams['axes.unicode_minus'] = False

# ---- Total figure width = 0.45 x US Letter width (8.5 in) ----
LETTER_W = 8.5
FIG_W = 0.45 * LETTER_W          # 3.825 in  (entire figure incl. legend)
FIG_H = FIG_W * 0.82             # height auto-tuned to content

# ---- Data ----
categories = ['Accuracy', 'Model\nComplexity', 'Physical\nInterpretability',
              'Design\nSimplicity', 'Implementative\nFeasibility', 'Overall\nBenefit']
methods = {
    'SE':            [2.5, 5.0, 5.0, 5.0, 5.0, 4.5],
    'RF':            [3.5, 1.0, 1.2, 2.2, 2.5, 2.1],
    'FNN':           [4.7, 2.0, 1.0, 3.0, 3.0, 3.5],
    'Proposed SSI':  [4.6, 4.7, 4.6, 4.5, 4.8, 4.6],
    'Proposed LSSI': [4.8, 4.8, 5.0, 4.8, 5.0, 4.9],
}

# A clean, print-friendly palette (colorblind-aware)
colors = {
    'SE':            '#D1495B',  # warm red
    'RF':            '#EDAE49',  # amber
    'FNN':           '#66A182',  # green
    'Proposed SSI':  '#B39EB5',  # Purple
    'Proposed LSSI': '#2E5EAA',  # blue (proposed / highlight)
}

N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # close the loop

fig = plt.figure(figsize=(FIG_W, FIG_H))
ax = fig.add_axes([0.14, 0.14, 0.46, 0.72], polar=True)

# ---- Grid / spine styling ----
ax.set_theta_offset(np.pi / 2)     # first axis at top
ax.set_theta_direction(-1)         # clockwise
ax.set_ylim(0, 5)
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_yticklabels(['1', '2', '3', '4', '5'], color='black', fontsize=10)
ax.set_rlabel_position(360 / N / 2 * 6)  # place radial labels in the gap between two axes

# Outermost circle: solid black line (the polar spine)
ax.spines['polar'].set_color('black')
ax.spines['polar'].set_linewidth(0.6)
# 隐藏内置外圈 spine
# ax.spines['polar'].set_visible(False)

# Inner concentric circles + radial spokes: black dashed
ax.yaxis.grid(True, color='black', linewidth=0.6, linestyle=(0, (1, 1)))
ax.xaxis.grid(True, color='black', linewidth=0.6, linestyle=(0, (1, 1)))

# 网格置于数据之下
ax.set_axisbelow(True)

# ---- Category labels ----
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.tick_params(axis='x', pad=6)

ticklabels = ax.get_xticklabels()
ticklabels[0].set_verticalalignment('top')       # Accuracy（顶部）：锚点设为 top，文字下沉贴近圆圈
ticklabels[3].set_verticalalignment('bottom')    # Design Simplicity（底部）：锚点设为 bottom，文字上移贴近圆圈
ticklabels[3].set_y(ticklabels[3].get_position()[1] - 0.06)  # 底部略微下移，避免第一行压线
ticklabels[0].set_y(ticklabels[0].get_position()[1] + 0.06)   # Accuracy 更贴近外圈

def nudge(label, dx_pts=0.0, dy_pts=0.0):
    offset = mtransforms.offset_copy(label.get_transform(), fig=fig,
                                     x=dx_pts, y=dy_pts, units='points')
    label.set_transform(offset)

nudge(ticklabels[1], dx_pts=10)    # Model Complexity 右移
nudge(ticklabels[2], dx_pts=10)    # Physical Interp. 右移
nudge(ticklabels[4], dx_pts=-16)   # Implementative Feas. 左移

# ---- Plot each method (highlight PASSI last so it sits on top) ----
draw_order = ['SE', 'RF', 'FNN', 'Proposed SSI', 'Proposed LSSI']
for name in draw_order:
    vals = methods[name] + methods[name][:1]
    lw = 1.4 if name == 'Proposed LSSI' else 1.4
    fa = 0.03 if name == 'Proposed LSSI' else 0.03
    ax.plot(angles, vals, color=colors[name], linewidth=lw,
            label=name, marker='o', markersize=3, zorder=5)
    # ax.fill(angles, vals, color=colors[name], alpha=fa, zorder=2)

# ---- Legend: vertical, to the right of the radar ----
# leg = ax.legend(loc='center left', bbox_to_anchor=(1.18, 0.5),
#                 ncol=1, frameon=False, fontsize=10,
#                 handlelength=1.6, borderpad=0.6, labelspacing=0.8)
leg = fig.legend(loc='center left', bbox_to_anchor=(0.72, 0.5),
                 ncol=1, frameon=False, fontsize=8,
                 handlelength=1.3, borderpad=0.3, labelspacing=0.4)

leg.get_frame().set_edgecolor('#cccccc')
leg.get_frame().set_linewidth(0.8)

plt.savefig('Figure3.png', dpi=600, bbox_inches='tight')
plt.show()