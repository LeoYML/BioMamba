#!/usr/bin/env python3
"""Real-time training monitor. Reads TensorBoard logs and refreshes every N seconds."""
# --- Project root resolution (auto-generated) ---
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')
_sys.path.insert(0, _PROJECT_ROOT)
_os.chdir(_PROJECT_ROOT)
# --- End project root resolution ---

import sys
import time
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def monitor(log_dir, total_steps=None, interval=30):
    print(f"Monitoring: {log_dir}")
    print(f"Refresh every {interval}s. Ctrl+C to stop.\n")

    last_step = -1
    while True:
        try:
            ea = EventAccumulator(log_dir)
            ea.Reload()

            os.system("clear" if os.name == "posix" else "cls")

            # Train loss
            train_events = ea.Scalars("train/loss") if "train/loss" in ea.Tags().get("scalars", []) else []
            eval_events = ea.Scalars("eval/loss") if "eval/loss" in ea.Tags().get("scalars", []) else []
            ppl_events = ea.Scalars("eval/ppl") if "eval/ppl" in ea.Tags().get("scalars", []) else []

            cur_step = train_events[-1].step if train_events else 0
            progress = f"{cur_step}/{total_steps} ({100*cur_step/total_steps:.1f}%)" if total_steps else str(cur_step)

            print(f"{'='*60}")
            print(f"  Step: {progress}")
            print(f"{'='*60}")

            # Recent train loss
            print(f"\n  train/loss (recent 10):")
            for e in train_events[-10:]:
                print(f"    step {e.step:>5d}: {e.value:.4f}")

            # Eval
            if eval_events:
                print(f"\n  eval/loss:")
                for i, e in enumerate(eval_events):
                    ppl = ppl_events[i].value if i < len(ppl_events) else float("nan")
                    print(f"    step {e.step:>5d}: loss={e.value:.4f}  ppl={ppl:.2f}")

            # Trend
            if len(train_events) >= 2:
                recent = [e.value for e in train_events[-5:]]
                earlier = [e.value for e in train_events[-10:-5]] if len(train_events) >= 10 else [e.value for e in train_events[:len(train_events)//2]]
                if recent and earlier:
                    trend = sum(recent)/len(recent) - sum(earlier)/len(earlier)
                    arrow = "↓ improving" if trend < -0.01 else ("↑ worsening" if trend > 0.01 else "→ stable")
                    print(f"\n  Trend: avg_recent={sum(recent)/len(recent):.4f} vs avg_earlier={sum(earlier)/len(earlier):.4f} {arrow}")

            print(f"\n  Last update: {time.strftime('%H:%M:%S')}")
            print(f"  (Ctrl+C to stop)")

            last_step = cur_step
            time.sleep(interval)

        except KeyboardInterrupt:
            print("\nMonitor stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(interval)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir", default="./runs/biomamba_cpt_singledoc_mamba2-780m")
    p.add_argument("--total_steps", type=int, default=5967)
    p.add_argument("--interval", type=int, default=30, help="Refresh interval in seconds")
    args = p.parse_args()
    monitor(args.log_dir, args.total_steps, args.interval)
