#!/usr/bin/env python3
"""Plot validation curves for all 5 CPT (Continual Pre-Training) models."""

# --- Project root resolution (auto-generated) ---
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')
_sys.path.insert(0, _PROJECT_ROOT)
_os.chdir(_PROJECT_ROOT)
# --- End project root resolution ---

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ── Config ──────────────────────────────────────────────────────────────────
MODELS = {
    "130m": _os.path.join(_PROJECT_ROOT, "runs/biomamba_cpt_singledoc_mamba2-130m"),
    "370m": _os.path.join(_PROJECT_ROOT, "runs/biomamba_cpt_singledoc_mamba2-370m"),
    "780m": _os.path.join(_PROJECT_ROOT, "runs/biomamba_cpt_singledoc_mamba2-780m"),
    "1.3b": _os.path.join(_PROJECT_ROOT, "runs/biomamba_cpt_singledoc_mamba2-1.3b"),
    "2.7b": _os.path.join(_PROJECT_ROOT, "runs/biomamba_cpt_singledoc_mamba2-2.7b"),
}

COLORS = {
    "130m": "#4C72B0",
    "370m": "#DD8452",
    "780m": "#55A868",
    "1.3b": "#C44E52",
    "2.7b": "#8172B3",
}

MARKERS = {
    "130m": "o",
    "370m": "s",
    "780m": "D",
    "1.3b": "^",
    "2.7b": "v",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Bitstream Vera Serif"],
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ── Load data ───────────────────────────────────────────────────────────────
def load_tb_scalars(logdir, tag, last_run_only=False):
    """Load a scalar tag from a TensorBoard logdir."""
    ea = EventAccumulator(logdir)
    ea.Reload()
    events = ea.Scalars(tag)

    if last_run_only and len(events) > 0:
        sorted_events = sorted(events, key=lambda e: e.wall_time)
        last_start = 0
        for i in range(1, len(sorted_events)):
            gap = sorted_events[i].wall_time - sorted_events[i - 1].wall_time
            if gap > 3600 and sorted_events[i].step <= 100:
                last_start = i
        last_run = sorted_events[last_start:]
        step_map = {}
        for e in last_run:
            step_map[e.step] = e.value
        steps_sorted = sorted(step_map.keys())
        return np.array(steps_sorted), np.array([step_map[s] for s in steps_sorted])

    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


MULTI_RUN_MODELS = {"130m", "2.7b"}

data = {}
for name, path in MODELS.items():
    data[name] = {}
    need_last_run = name in MULTI_RUN_MODELS
    for tag in ["eval/loss", "eval/ppl"]:
        try:
            steps, values = load_tb_scalars(path, tag, last_run_only=need_last_run)
            data[name][tag] = (steps, values)
        except Exception as e:
            print(f"Warning: {name} missing {tag}: {e}")


# ── Figure 1: Validation Loss ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for name in MODELS:
    if "eval/loss" not in data[name]:
        continue
    steps, values = data[name]["eval/loss"]
    ax.plot(
        steps, values,
        color=COLORS[name], linewidth=2.2,
        marker=MARKERS[name], markersize=5, markevery=max(1, len(steps) // 10),
        label=f"BioMamba-{name}",
    )
    # Annotate best value — shift left if near right edge
    best_idx = np.argmin(values)
    if best_idx >= len(steps) - 2:
        offset = (-45, 4)
    else:
        offset = (8, 4)
    ax.annotate(
        f"{values[best_idx]:.3f}",
        xy=(steps[best_idx], values[best_idx]),
        xytext=offset, textcoords="offset points",
        fontsize=9, color=COLORS[name], fontweight="bold",
    )

ax.set_xlabel("Training Steps")
ax.set_ylabel("Validation Loss")
ax.set_title("Continual Pre-Training: Validation Loss")
ax.legend(bbox_to_anchor=(0.99, 0.63), loc="right", framealpha=0.95, edgecolor="0.8")
ax.set_ylim(1.6, 2.25)
ax.grid(True, alpha=0.3, linestyle="--")
ax.set_xlim(left=0)
fig.tight_layout()
fig.savefig(_os.path.join(_PROJECT_ROOT, "plot_cpt_eval_loss.png"))
fig.savefig(_os.path.join(_PROJECT_ROOT, "plot_cpt_eval_loss.pdf"), format="pdf", dpi=300)
print("Saved: plot_cpt_eval_loss.png / .pdf")
plt.close(fig)


# ── Figure 2: Validation Perplexity ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for name in MODELS:
    if "eval/ppl" not in data[name]:
        continue
    steps, values = data[name]["eval/ppl"]
    ax.plot(
        steps, values,
        color=COLORS[name], linewidth=2.2,
        marker=MARKERS[name], markersize=5, markevery=max(1, len(steps) // 10),
        label=f"BioMamba-{name}",
    )
    best_idx = np.argmin(values)
    if best_idx >= len(steps) - 2:
        offset = (-40, 4)
    else:
        offset = (8, 4)
    ax.annotate(
        f"{values[best_idx]:.2f}",
        xy=(steps[best_idx], values[best_idx]),
        xytext=offset, textcoords="offset points",
        fontsize=9, color=COLORS[name], fontweight="bold",
    )

ax.set_xlabel("Training Steps")
ax.set_ylabel("Perplexity")
ax.set_title("Continual Pre-Training: Validation Perplexity")
ax.legend(bbox_to_anchor=(0.99, 0.63),loc="right", framealpha=0.95, edgecolor="0.8")
ax.grid(True, alpha=0.3, linestyle="--")
ax.set_xlim(left=0)
fig.tight_layout()
fig.savefig(_os.path.join(_PROJECT_ROOT, "plot_cpt_eval_ppl.png"))
fig.savefig(_os.path.join(_PROJECT_ROOT, "plot_cpt_eval_ppl.pdf"), format="pdf", dpi=300)
print("Saved: plot_cpt_eval_ppl.png / .pdf")
plt.close(fig)


# ── Summary Table ───────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print(f"{'Model':<10} {'Steps':>8} {'Best Val Loss':>15} {'Best Val PPL':>14}")
print("-" * 55)
for name in MODELS:
    e_steps, e_vals = data[name].get("eval/loss", (np.array([0]), np.array([0])))
    _, ppl_vals = data[name].get("eval/ppl", (np.array([0]), np.array([0])))
    print(f"{name:<10} {int(e_steps[-1]):>8} {e_vals.min():>15.4f} {ppl_vals.min():>14.2f}")
print("=" * 55)
