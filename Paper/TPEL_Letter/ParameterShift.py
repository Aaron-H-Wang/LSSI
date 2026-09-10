"""
Learnable symbolic library in power-law exponent space (alpha, beta) — 0..4 范围版。
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

# Computer Modern 字体
plt.rcParams["mathtext.fontset"] = "cm"      # 数学符号保持 CM，不动
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]

# ---------- 颜色 ----------
C_LAT_FILL  = "#b0b0b0"
C_LAT_EDGE  = "#999999"
C_TGT_FILL  = "#f7c6d0"   # 粉底
C_TGT_EDGE  = "#cc2b3d"   # 红边
C_SEED_EDGE = "#1f6fe0"
C_SEED_FILL = "#e6eeff"
C_GLIDE     = "#1f6fe0"
C_STACK     = "#f2a3a3"
C_PROJ      = "#d24a4a"
C_RED70     = "#d24a4a"
C_GRID      = "#000000"
C_AX        = "#000000"

# 轴范围 0..4
GRID_X = [0, 1, 2, 3, 4]
GRID_Y = [0, 1, 2, 3, 4]

TARGETS = [(1.20, 2.68), (2.2, 2.0), (1.19, 3.29)]
SEEDS   = [(1, 2), (2, 1), (2, 3)]


def draw_star(ax, x, y, s=0.14):   # 星星缩小一号 0.16 -> 0.14
    outer, inner = s, s * 0.5
    pts = []
    for k in range(5):
        a_out = np.pi / 2 + k * 2 * np.pi / 5
        a_in = a_out + np.pi / 5
        pts.append((x + outer * np.cos(a_out), y + outer * np.sin(a_out)))
        pts.append((x + inner * np.cos(a_in), y + inner * np.sin(a_in)))
    ax.add_patch(plt.Polygon(pts, closed=True, facecolor=C_TGT_FILL,
                             edgecolor=C_TGT_EDGE, linewidth=1.0, zorder=6))


# 整张图宽度 = Letter 的 1/3 ≈ 2.83 in
fig_w = 8.5 / 3
fig = plt.figure(figsize=(fig_w, fig_w * 1), dpi=600)
ax = fig.add_axes([0.04, 0.04, 0.92, 0.92])
ax.set_xlim(-0.7, 4.7)
ax.set_ylim(-0.7, 5.4)
ax.set_aspect('equal')
ax.axis('off')

# ===== 网格（黑色虚线，范围 0..4）=====
for x in GRID_X:
    ax.plot([x, x], [0, 4], color=C_GRID, lw=0.6, ls="--", dashes=(4, 3), zorder=1, alpha=0.55)
for y in GRID_Y:
    ax.plot([0, 4], [y, y], color=C_GRID, lw=0.6, ls="--", dashes=(4, 3), zorder=1, alpha=0.55)

# ===== 坐标轴（带箭头）=====
ax.add_patch(FancyArrowPatch((-0.15, 0), (4.55, 0),
             arrowstyle='-|>', mutation_scale=14, lw=1.3, color=C_AX, zorder=5))
ax.add_patch(FancyArrowPatch((0, -0.15), (0, 4.7),
             arrowstyle='-|>', mutation_scale=14, lw=1.3, color=C_AX, zorder=5))
# 轴变量标签：alpha 横轴中间下方，beta 纵轴中间左边
ax.text(2.0, -0.45, r"$\alpha$", fontsize=10, va="top", ha="center")
ax.text(-0.5, 2.0, r"$\beta$", fontsize=10, va="center", ha="right")
# 刻度数字
for x in [1, 2, 3, 4]:
    ax.text(x, -0.16, str(x), fontsize=9, va="top", ha="center")
for y in [1, 2, 3, 4]:
    ax.text(-0.16, y, str(y), fontsize=9, va="center", ha="right")
ax.text(-0.16, -0.16, "0", fontsize=9, va="top", ha="right")

# ===== 格点（缩小一号 0.08 -> 0.07）=====
for x in GRID_X:
    for y in GRID_Y:
        ax.add_patch(plt.Circle((x, y), 0.07, facecolor=C_LAT_FILL,
                                edgecolor=C_LAT_EDGE, lw=0.8, zorder=3))

# ===== a 特有元素 =====
ax.add_patch(Rectangle((1, 2), 1, 1, facecolor="#fbe7e7", edgecolor="none", zorder=0))
ax.add_patch(Rectangle((1, 2), 1, 1, fill=False, edgecolor="#e08a8a",
                       linestyle="--", lw=1.0, zorder=2))
a1 = TARGETS[0]
for c in [(1, 2), (2, 2), (1, 3), (2, 3)]:
    ax.add_patch(FancyArrowPatch((c[0], c[1]), a1,
                 arrowstyle='-|>', mutation_scale=8, lw=1.0, color=C_STACK, zorder=4))
ax.text(3, 2.7, "$\\geq 4$ fixed terms \n$\\Rightarrow$ residual error",
        color=C_RED70, fontsize=8, ha="center", va="center")

# ===== b 元素 =====
for (sx, sy) in SEEDS:    # 种子圆缩小一号 0.12 -> 0.10
    ax.add_patch(plt.Circle((sx, sy), 0.09, facecolor=C_SEED_FILL,
                            edgecolor=C_SEED_EDGE, lw=1.1, zorder=4))
for (tx, ty) in TARGETS:
    draw_star(ax, tx, ty)
bends = [0.20, -0.12, 0.14]
for (s, t, bd) in zip(SEEDS, TARGETS, bends):
    ax.add_patch(FancyArrowPatch((s[0], s[1]), (t[0], t[1]),
                 connectionstyle=f"arc3,rad={bd}",
                 arrowstyle='-|>', mutation_scale=12, lw=1.4, color=C_GLIDE, zorder=5))
for (tx, ty), lab in zip(TARGETS, [r"$\theta_1$", r"$\theta_2$", r"$\theta_4$"]):
    ax.text(tx + 0.12, ty + 0.12, lab, fontsize=10, ha="left", va="bottom")
# 连续读出虚线投影（读到 0 轴）
b1 = TARGETS[0]
ax.plot([b1[0], b1[0]], [b1[1], 0], color=C_PROJ, ls="--", lw=0.9, zorder=2)
ax.plot([b1[0], 0], [b1[1], b1[1]], color=C_PROJ, ls="--", lw=0.9, zorder=2)
ax.text(1.40, -0.12, "$1.20$", color=C_RED70, fontsize=8, ha="center", va="top")
ax.text(-0.12, 2.68, "$2.68$", color=C_RED70, fontsize=8, ha="right", va="center")

# ===== 图例（右上角，Optimized Parameter / Initial Seed）=====
lx = 2.55
rows = 4.55
dy = 0.3
draw_star(ax, lx, rows, s=0.10)   # 图例星星缩小一号 0.12 -> 0.10
ax.text(lx + 0.22, rows, "Optimized", fontsize=8, va="center", ha="left")
ax.add_patch(plt.Circle((lx, rows - dy), 0.09, facecolor=C_SEED_FILL,   # 图例种子 0.12 -> 0.10
                        edgecolor=C_SEED_EDGE, lw=1.1, zorder=4))
ax.text(lx + 0.22, rows - dy, "Initial", fontsize=8, va="center", ha="left")

plt.savefig("learnable_lib.png", dpi=600, facecolor="white")
print("done")