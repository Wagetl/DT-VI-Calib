"""Regenerate the threshold comparison from the preserved CSVs and raw logs."""

import csv
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "time_delay_2026-11-12"


def read(path):
    with path.open() as file:
        return list(csv.DictReader(file))


old_runs, runs = read(OLD / "runs.csv"), read(HERE / "runs.csv")
old_trace, trace = read(OLD / "convergence.csv"), read(HERE / "convergence.csv")
assert len(runs) == 6 and len(trace) == 53
comparison = []
fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
step_fig, step_axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
for ax, step_ax, delay in zip(axes.flat, step_axes.flat, (0, 10, 20, 30, 40, 50)):
    repeat = "0" if delay == 0 else "1"
    before = next(r for r in old_runs if r["injected_ms"] == str(delay) and r["repeat"] == repeat)
    after = next(r for r in runs if r["injected_ms"] == str(delay))
    for rows, label, marker in ((old_trace, "1e-4", "o"), (trace, "1e-8 accepted", "+")):
        points = [r for r in rows if r["injected_ms"] == str(delay)
                  and r["repeat"] == repeat and r["status"] != "rejected"]
        ax.plot([int(r["iteration"]) for r in points],
                [float(r["estimate_ms"]) for r in points], marker + "-", label=label)
    rejected = [r for r in trace if r["injected_ms"] == str(delay) and r["status"] == "rejected"]
    ax.scatter([int(r["iteration"]) for r in rejected], [float(r["estimate_ms"]) for r in rejected],
               marker="x", color="red", label="1e-8 rejected trial", zorder=5)
    ax.set(title=f"Injected +{delay} ms", xlabel="Outer iteration", ylabel="Reported offset (ms)")
    raw = (HERE / "logs" / f"delay_{delay:02d}ms_run_{repeat}.log").read_text()
    attempts = re.findall(r"\[(ACCEPTED|REJECTED)\].*max_inc: ([-+0-9.eE]+)", raw)
    assert len(attempts) == sum(r["status"] != "initial" for r in trace if r["injected_ms"] == str(delay))
    for status, color, marker in (("ACCEPTED", "tab:blue", "o"), ("REJECTED", "red", "x")):
        step_ax.scatter([i for i, a in enumerate(attempts, 1) if a[0] == status],
                        [float(a[1]) for a in attempts if a[0] == status],
                        color=color, marker=marker, label=status.lower(), zorder=5)
    step_ax.axhline(1e-4, color="gray", linestyle="--", label="1e-4")
    step_ax.axhline(1e-8, color="black", linestyle=":", label="1e-8")
    step_ax.set(title=f"Injected +{delay} ms", xlabel="LM trial (including retries)",
                ylabel="Maximum state increment", yscale="log")
    for panel in (ax, step_ax):
        panel.xaxis.set_major_locator(MaxNLocator(integer=True))
        panel.grid(alpha=0.2)
        panel.legend(fontsize=7)
    comparison.append(dict(injected_ms=delay, estimate_1e4_ms=before["estimate_ms"],
                           last_accepted_1e8_ms=after["estimate_ms"],
                           delta_ms=float(after["estimate_ms"]) - float(before["estimate_ms"]),
                           iterations_1e4=before["iterations"], iterations_1e8=after["iterations"],
                           accepted_trials_1e8=sum(a[0] == "ACCEPTED" for a in attempts),
                           rejected_trials_1e8=after["rejected_trials"],
                           final_max_inc=attempts[-1][1], validated=after["validated"]))
for figure, name in ((fig, "threshold_comparison"), (step_fig, "step_size")):
    figure.savefig(HERE / f"{name}.png", dpi=180)
    figure.savefig(HERE / f"{name}.pdf")
    plt.close(figure)
with (HERE / "comparison.csv").open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(comparison[0]))
    writer.writeheader()
    writer.writerows(comparison)
