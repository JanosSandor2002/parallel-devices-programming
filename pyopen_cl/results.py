"""
plot_results.py
===============
Benchmark eredmények vizualizálása – GPU vs CPU (sum of squares).

Futtatás:
    python sum_of_squares_gpu.py   # először ez generálja a benchmark_results.json-t
    python plot_results.py         # aztán ez a grafikon szkript

Képek kimenete: benchmark_plots.png (és képernyőn is megjelenik)
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Adatok betöltése
JSON_FILE = os.path.join(os.path.dirname(__file__), "results.json")

if not os.path.exists(JSON_FILE):
    print(f"[HIBA] Nem található: {JSON_FILE}")
    print("  Előbb futtasd le: python sum_of_squares_gpu.py")
    sys.exit(1)

with open(JSON_FILE) as fh:
    r = json.load(fh)

N               = r["N"]
cpu_ms          = r["cpu_time_ms"]
gpu_k_ms        = r["gpu_kernel_ms"]
gpu_p_ms        = r["gpu_pipeline_ms"]
speedup_kernel  = r["speedup_kernel"]
speedup_pipeline = r["speedup_pipeline"]
cpu_result      = r["cpu_result"]
gpu_result      = r["gpu_result"]

# Stílus
plt.style.use("dark_background")

C_CPU     = "#F5A623"
C_KERNEL  = "#4FC3F7"
C_PIPE    = "#29B6F6"  
C_ACCENT  = "#A5D6A7"  
C_BG      = "#0F1117"
C_GRID    = "#1E2130"
C_TEXT    = "#E0E0E0"

FONT_TITLE  = {"fontsize": 13, "fontweight": "bold", "color": C_TEXT}
FONT_LABEL  = {"fontsize": 10, "color": C_TEXT}
FONT_TICK   = {"labelsize": 9, "colors": C_TEXT}

def style_ax(ax, title=""):
    ax.set_facecolor(C_BG)
    ax.tick_params(**FONT_TICK)
    ax.spines[:].set_color(C_GRID)
    ax.grid(axis="y", color=C_GRID, linewidth=0.7, linestyle="--")
    if title:
        ax.set_title(title, **FONT_TITLE, pad=10)

# Ábra felépítése  (2 × 2 grid + 1 széles alul)
fig = plt.figure(figsize=(14, 10), facecolor=C_BG)
fig.suptitle(
    f"Sum of Squares – GPU vs CPU Benchmark  (N = {N:,})",
    fontsize=16, fontweight="bold", color=C_TEXT, y=0.98
)

gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
              left=0.07, right=0.97, top=0.91, bottom=0.08)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1])
ax6 = fig.add_subplot(gs[1, 2])

# ── 1. Futási idők (lineáris skála) ──────────────────────────
labels = ["CPU\n(NumPy)", "GPU Kernel\n(OpenCL)", "GPU Pipeline\n(+átvitel)"]
values = [cpu_ms, gpu_k_ms, gpu_p_ms]
colors = [C_CPU, C_KERNEL, C_PIPE]

bars = ax1.bar(labels, values, color=colors, width=0.55, edgecolor=C_GRID, linewidth=0.8)
for bar, val in zip(bars, values):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + cpu_ms * 0.01,
             f"{val:.1f} ms", ha="center", va="bottom", fontsize=9, color=C_TEXT)

ax1.set_ylabel("Idő (ms)", **FONT_LABEL)
style_ax(ax1, "Futási idők")

# ── 2. Gyorsítás (speedup) ───────────────────────────────────
sp_labels = ["GPU kernel\nvs CPU", "GPU pipeline\nvs CPU"]
sp_values = [speedup_kernel, speedup_pipeline]
sp_colors = [C_KERNEL, C_PIPE]

bars2 = ax2.bar(sp_labels, sp_values, color=sp_colors, width=0.45, edgecolor=C_GRID)
ax2.axhline(1.0, color=C_CPU, linestyle="--", linewidth=1.2, label="1× (egyenértékű)")
for bar, val in zip(bars2, sp_values):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
             f"{val:.1f}×", ha="center", va="bottom", fontsize=10,
             fontweight="bold", color=C_ACCENT)

ax2.set_ylabel("Gyorsítás (×)", **FONT_LABEL)
ax2.legend(fontsize=8, labelcolor=C_TEXT, facecolor=C_BG, edgecolor=C_GRID)
style_ax(ax2, "Gyorsítás (CPU / GPU)")

# ── 3. Eredmény összehasonlítás ──────────────────────────────
rel_err = abs(cpu_result - gpu_result) / abs(cpu_result)
res_labels = ["CPU eredmény", "GPU eredmény"]
res_values = [cpu_result, gpu_result]

bars3 = ax3.bar(res_labels, res_values, color=[C_CPU, C_KERNEL],
                width=0.4, edgecolor=C_GRID)
for bar, val in zip(bars3, res_values):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.5,
             f"{val:,.0f}", ha="center", va="center", fontsize=9,
             color="white", fontweight="bold")

ax3.set_ylabel("Négyzetösszeg értéke", **FONT_LABEL)
ax3.text(0.5, -0.18, f"Relatív eltérés: {rel_err:.2e}", transform=ax3.transAxes,
         ha="center", fontsize=9, color=C_ACCENT)
style_ax(ax3, "Numerikus eredmény")

# ── 4. Log-skálás időbár ─────────────────────────────────────
ax4.bar(labels, values, color=colors, width=0.55, edgecolor=C_GRID, linewidth=0.8)
ax4.set_yscale("log")
ax4.set_ylabel("Idő (ms, log skála)", **FONT_LABEL)
for i, (lbl, val) in enumerate(zip(labels, values)):
    ax4.text(i, val * 1.15, f"{val:.2f}", ha="center", va="bottom",
             fontsize=8, color=C_TEXT)
style_ax(ax4, "Futási idők (log skála)")

# ── 5. GPU pipeline bontás (stacked) ────────────────────────
# kernel idő  +  átvitel idő (pipeline - kernel)
transfer_ms = gpu_p_ms - gpu_k_ms

ax5.bar(["GPU Pipeline"], [gpu_k_ms], color=C_KERNEL,
        label=f"Kernel ({gpu_k_ms:.1f} ms)", edgecolor=C_GRID)
ax5.bar(["GPU Pipeline"], [transfer_ms], bottom=[gpu_k_ms], color="#EF5350",
        label=f"Host↔Device ({transfer_ms:.1f} ms)", edgecolor=C_GRID)
ax5.axhline(cpu_ms, color=C_CPU, linestyle="--", linewidth=1.5,
            label=f"CPU ({cpu_ms:.1f} ms)")

ax5.set_ylabel("Idő (ms)", **FONT_LABEL)
ax5.legend(fontsize=8, labelcolor=C_TEXT, facecolor=C_BG, edgecolor=C_GRID)
style_ax(ax5, "GPU Pipeline bontás")

# ── 6. Sávszélesség és hatékonyság ──────────────────────────
# Elméleti memória-sávszélesség becslés:
# Input olvasás + output írás = 2 × N × 4 byte
bytes_transferred = 2 * N * 4  # byte
bw_gpu = bytes_transferred / (gpu_k_ms / 1000) / 1e9   # GB/s
bw_cpu = bytes_transferred / (cpu_ms / 1000) / 1e9

bw_labels = ["CPU\nsávszélesség", "GPU Kernel\nsávszélesség"]
bw_vals   = [bw_cpu, bw_gpu]
bw_colors = [C_CPU, C_KERNEL]

bars6 = ax6.bar(bw_labels, bw_vals, color=bw_colors, width=0.4, edgecolor=C_GRID)
for bar, val in zip(bars6, bw_vals):
    ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(bw_vals) * 0.01,
             f"{val:.1f} GB/s", ha="center", va="bottom", fontsize=9, color=C_TEXT)

ax6.set_ylabel("Effektív sávszélesség (GB/s)", **FONT_LABEL)
style_ax(ax6, "Memória-sávszélesség becslés")

# ─────────────────────────────────────────────
# Mentés és megjelenítés
# ─────────────────────────────────────────────
out_png = os.path.join(os.path.dirname(__file__), "plots.png")
fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=C_BG)
print(f"[OK] Grafikon mentve → {out_png}")

plt.show()