import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

BLUE   = "#2a78d6"   # slot1 - continuous, no restart
AQUA   = "#1baf7a"   # slot3 - staged, with restart
ORANGE = "#eb6834"   # slot2 - projection (speculative)
INK    = "#0b0b0b"
INK2   = "#52514e"
GRID   = "#e4e3de"
SURF   = "#fcfcfb"

# --- Path A: plain155k, continuous single schedule, no restart -------------
xA = [4999,9999,14999,19999,24999,29999,34999,39999,44999,49999,54999,59999,
      64999,69999,74999,79999,84999,89999,94999,99999,104999,109999,114999,
      119999,124999,129999,134999,139999,144999,149999,155000]
yA = [8.06,19.81,29.94,35.5,41.03,41.75,42.73,50.3,50.13,50.23,54.13,55.62,
      53.81,53.27,55.8,56.09,55.61,53.44,57.45,57.55,57.25,58.98,52.93,58.5,
      63.53,64.5,64.11,64.81,64.39,64.52,64.48,64.48]  # pad last for align (155000 twice ok, will fix)
yA = [8.06,19.81,29.94,35.5,41.03,41.75,42.73,50.3,50.13,50.23,54.13,55.62,
      53.81,53.27,55.8,56.09,55.61,53.44,57.45,57.55,57.25,58.98,52.93,58.5,
      63.53,64.5,64.11,64.81,64.39,64.52,64.48]

# --- Path B part 1: plain135k, 0-135000 -------------------------------------
xB1 = [4999,9999,14999,19999,24999,29999,34999,39999,44999,49999,54999,59999,
       64999,69999,74999,79999,84999,89999,94999,99999,104999,109999,114999,
       119999,124999,129999,135000]
yB1 = [8.78,19.96,27.31,33.03,38.95,41.41,47.79,46.03,50.42,52.09,51.1,55.0,
       52.69,56.42,56.05,57.8,56.71,58.5,58.69,57.73,60.89,63.62,63.61,64.1,
       63.71,63.84,64.14]

# --- Path B part 2: plain_ext20k, LR restart, offset +135000 ---------------
xB2 = [135000, 139999, 144999, 149999, 155000]
yB2 = [64.14, 62.52, 63.18, 64.48, 64.73]  # starts at B1 endpoint -> dip -> recover -> new best

# --- Projection: +15k / +25k beyond 155000 (speculative, NOT data) ---------
xP = [155000, 170000, 180000]
yP_mid  = [64.73, 64.95, 65.05]
yP_low  = [64.73, 64.45, 64.35]
yP_high = [64.73, 65.35, 65.55]

fig, ax = plt.subplots(figsize=(10, 6.2), dpi=200)
fig.patch.set_facecolor(SURF)
ax.set_facecolor(SURF)

ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(GRID)

# Path A
ax.plot(xA, yA, color=BLUE, linewidth=2, zorder=3, label="plain155k — 155k kontinu, tanpa restart")

# Path B (135k) + restart segment
ax.plot(xB1, yB1, color=AQUA, linewidth=2, zorder=3, label="plain135k → plain_ext20k — 135k + restart LR 20k")
ax.plot(xB2, yB2, color=AQUA, linewidth=2, zorder=3, linestyle=(0, (1, 0)))

# Restart marker
ax.scatter([135000], [64.14], s=34, color=AQUA, zorder=5, edgecolor=SURF, linewidth=1.2)
ax.annotate("restart LR\n(warmup baru)", xy=(135000, 64.14), xytext=(120000, 68.5),
            fontsize=9, color=INK2, ha="center",
            arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
ax.annotate("dip pasca-restart", xy=(139999, 62.52), xytext=(113000, 52),
            fontsize=9, color=INK2, ha="center",
            arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))

# LR decay markers on Path A
for it, lbl in [(124000, "decay #1"), (144667, "decay #2")]:
    ax.axvline(it, color=BLUE, linewidth=0.8, linestyle=(0, (2, 3)), alpha=0.5, zorder=1)

# Projection
ax.fill_between(xP, yP_low, yP_high, color=ORANGE, alpha=0.12, zorder=2, linewidth=0)
ax.plot(xP, yP_mid, color=ORANGE, linewidth=2, linestyle=(0, (4, 3)), zorder=3,
        label="proyeksi +15k/+25k (spekulatif, bukan data aktual)")
ax.scatter([170000, 180000], [64.95, 65.05], s=28, color=ORANGE, zorder=5,
           edgecolor=SURF, linewidth=1.2)

# End-point labels (direct labels, selective) -- staggered to avoid collision
ax.annotate(f"plain155k: {yA[-1]:.2f}%", xy=(155000, yA[-1]), xytext=(156500, 60.5),
            fontsize=9.5, color=BLUE, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=BLUE, linewidth=0.7, alpha=0.6))
ax.annotate(f"plain_ext20k: {yB2[-1]:.2f}%", xy=(155000, yB2[-1]), xytext=(156500, 68.3),
            fontsize=9.5, color=AQUA, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=AQUA, linewidth=0.7, alpha=0.6))
ax.annotate("+15k: ~64,95% (±0,4pp)", xy=(170000, 64.95), xytext=(169000, 56.5),
            fontsize=8.5, color=ORANGE, ha="center",
            arrowprops=dict(arrowstyle="-", color=ORANGE, linewidth=0.7, alpha=0.6))
ax.annotate("+25k: ~65,05% (±0,5pp)", xy=(180000, 65.05), xytext=(178000, 53),
            fontsize=8.5, color=ORANGE, ha="center",
            arrowprops=dict(arrowstyle="-", color=ORANGE, linewidth=0.7, alpha=0.6))

# Noise-floor reference band (multi-seed std ~0.3-0.5pp, documented in project)
ax.axhspan(64.14, 64.9, color=INK2, alpha=0.05, zorder=0)
ax.text(2000, 64.5, "noise band multi-seed (±0,3–0,5pp)", fontsize=8, color=INK2,
        style="italic", va="center")

ax.set_xlim(0, 183000)
ax.set_ylim(0, 73)
ax.set_xticks([0, 25000, 50000, 75000, 100000, 125000, 135000, 155000, 170000, 180000])
ax.set_xticklabels(["0", "25k", "50k", "75k", "100k", "125k", "135k", "155k", "170k", "180k"],
                    fontsize=9, color=INK2)
ax.tick_params(axis="y", labelsize=9, colors=INK2)
ax.set_xlabel("Iterasi training", fontsize=10, color=INK)
ax.set_ylabel("Val segm AP50 (%)", fontsize=10, color=INK)
ax.set_title("Plateau & efek LR restart — 155k kontinu vs 135k+restart20k, dan proyeksi lanjutan",
             fontsize=12.5, color=INK, fontweight="bold", pad=14)

leg = ax.legend(loc="lower right", fontsize=9, frameon=False, labelcolor=INK)

plt.tight_layout()
out = "/tmp/claude-17769/-fs04-scratch2-pr65/2b9a6768-77e1-48bd-b156-244ec7d7a909/scratchpad/lr_restart_comparison.png"
plt.savefig(out, facecolor=SURF, bbox_inches="tight")
print("saved", out)
