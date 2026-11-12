#!/usr/bin/env python3
"""Run the downloaded TUM VI bag at 0 ms, then repeat offsets 10..50 ms."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
NUMBER = r"[-+0-9.eE]+"


def parse_log(raw):
    text = re.sub(r"\x1b\[[0-9;]*m", "", raw)
    rows = [dict(trial=0, iteration=0, status="initial", estimate_ms=0.0,
                 objective=None, optimizer_elapsed_ms=0.0)]
    candidate = None
    elapsed = setup = 0.0
    pending = []
    iteration = 0
    for line in text.splitlines():
        match = re.search(rf"\[LINEARIZE\] Error: ({NUMBER})", line)
        if match:
            if rows[0]["objective"] is None:
                rows[0]["objective"] = float(match[1])
            iteration += 1
        match = re.search(rf"setup_time ({NUMBER}) ms", line)
        if match:
            setup = float(match[1])
        match = re.search(rf"cam_time_offset ({NUMBER}) s\.", line)
        if match:
            candidate = float(match[1]) * 1000
        match = re.search(rf"\[(ACCEPTED|REJECTED)\].* Error: ({NUMBER})", line)
        if match:
            if candidate is None:
                raise ValueError("Optimization step is missing its offset estimate")
            row = dict(trial=len(rows), iteration=iteration, status=match[1].lower(),
                       estimate_ms=candidate, objective=float(match[2]),
                       optimizer_elapsed_ms=None)
            rows.append(row)
            pending.append(row)
            candidate = None
        match = re.search(rf"iter_time ({NUMBER}) ms", line)
        if match:
            elapsed += setup + float(match[1])
            for row in pending:
                row["optimizer_elapsed_ms"] = elapsed
            pending.clear()
    total = re.search(rf"total opt_time ({NUMBER}) ms", text)
    reprojection = re.findall(rf"mean reprojection ({NUMBER})", text)
    if ("Optimization Converged !!" not in text or len(rows) < 2 or not total
            or not reprojection or pending or candidate is not None):
        raise ValueError("Incomplete or non-converged calibration log")
    if any(not math.isfinite(row[key]) for row in rows
           for key in ("estimate_ms", "objective", "optimizer_elapsed_ms")):
        raise ValueError("Non-finite convergence trace")
    if re.search(r"\b(?:nan|inf)\b|err:|timestamp err", text, re.IGNORECASE):
        raise ValueError("Calibration log contains a numerical or timestamp error")
    return rows, float(total[1]), float(reprojection[-1])


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def plot_results(output, results, traces):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for ax, delay in zip(axes.flat, (0, 10, 20, 30, 40, 50)):
        for repeat in sorted({row["repeat"] for row in results}):
            points = [row for row in traces if row["injected_ms"] == delay
                      and row["repeat"] == repeat and row["status"] != "rejected"]
            if points:
                ax.plot([row["iteration"] for row in points],
                        [row["estimate_ms"] for row in points], ".-",
                        label="Baseline" if repeat == 0 else f"Run {repeat}")
        ax.axhline(delay, color="black", linestyle="--", linewidth=0.8,
                   label="Injected shift")
        ax.set(title=f"Injected +{delay} ms", xlabel="Optimizer iteration",
               ylabel="Estimated offset (ms)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.savefig(output / "convergence.png", dpi=180)
    fig.savefig(output / "convergence.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for repeat in sorted({row["repeat"] for row in results} - {0}):
        points = [row for row in results if row["repeat"] == repeat]
        x = [row["injected_ms"] for row in points]
        axes[0].plot(x, [row["estimate_ms"] for row in points], ".-", label=f"Run {repeat}")
        axes[1].plot(x, [row["baseline_corrected_error_ms"] for row in points], ".-",
                     label=f"Run {repeat}")
    axes[0].plot([10, 50], [10, 50], "k--", linewidth=0.8, label="Ideal")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set(ylabel="Estimated offset (ms)", title="Final estimates")
    axes[1].set(ylabel="Estimate - baseline - injection (ms)", title="Recovered added delay error")
    for ax in axes:
        ax.set_xlabel("Injected IMU timestamp shift (ms)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.savefig(output / "estimation.png", dpi=180)
    fig.savefig(output / "estimation.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New output directory; existing directories are refused")
    parser.add_argument("--repeats", type=int, default=3, help="Runs per injected offset (default: 3)")
    parser.add_argument("--record-rejected-trials", action="store_true",
                        help="Record diagnostic runs with rejected trials; mark their estimates unvalidated")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    output = args.output.resolve()
    import matplotlib  # Check the plotting dependency before starting experiments.
    dataset = ROOT / "build/datasets/dataset-calib-imu1_512_16.bag"
    inputs = [dataset, ROOT / "data/aprilgrid_6x6.json"]
    cache_names = ["calibration.json", "calib-cam-imu_detected_corners.cereal",
                   "calib-cam-imu_init_poses.cereal"]
    inputs += [ROOT / "build/results" / name for name in cache_names]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    shutil.copy2(inputs[2], output / "camera_calibration.json")
    manifest = dict(
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        input_files={str(path.relative_to(ROOT)): dict(bytes=path.stat().st_size,
                                                      sha256=sha256(path)) for path in inputs},
        source_sha256={name: sha256(ROOT / name) for name in (
            "src/calibrate_imu.cpp", "src/calibration/cam_imu_calib.cpp",
            "include/basalt/optimization/spline_optimize.h",
            "scripts/time_delay_experiments.py")},
        matplotlib_version=matplotlib.__version__, runs=[])
    manifest["container"] = subprocess.check_output(
        ["docker", "inspect", "dt-vi-calib-calib-1", "--format", "{{.Image}}"], text=True).strip()
    results, traces = [], []
    with tempfile.TemporaryDirectory(prefix="time-delay-", dir=ROOT / "build") as work:
        work = Path(work)
        for name in cache_names:
            shutil.copy2(ROOT / "build/results" / name, work / name)
        for repeat, delay in [(0, 0)] + [(r, d) for r in range(1, args.repeats + 1) for d in (10, 20, 30, 40, 50)]:
            stem = f"delay_{delay:02d}ms_run_{repeat}"
            command = ["docker", "compose", "exec", "-T", "calib", "build/docker/basalt_calibrate_imu",
                       "--dataset-path", "/workspace/" + str(dataset.relative_to(ROOT)),
                       "--dataset-type", "bag", "--aprilgrid", "/workspace/data/aprilgrid_6x6.json",
                       "--result-path", "/workspace/" + str(work.relative_to(ROOT)) + "/",
                       "--gyro-noise-std", "0.000282", "--accel-noise-std", "0.016",
                       "--skip-images", "1", "--time-offset-ms", str(delay)]
            start = time.monotonic()
            log = output / "logs" / f"{stem}.log"
            with log.open("w") as file:
                subprocess.run(command, cwd=ROOT, stdout=file, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, check=True, timeout=180)
            wall_s = time.monotonic() - start
            text = log.read_text()
            if f"[TIME_OFFSET_EXPERIMENT] injected_ms={delay}\n" not in text:
                raise ValueError("Injected offset was not confirmed by the executable")
            rows, opt_ms, reprojection = parse_log(text)
            rejected = sum(row["status"] == "rejected" for row in rows)
            # shortcut: upstream rollback is incomplete; rejected trials are diagnostic unless rollback is repaired.
            if rejected and not args.record_rejected_trials:
                raise ValueError(f"{stem}: rejected LM trial; inspect upstream rollback before evaluating")
            estimate = next(row["estimate_ms"] for row in reversed(rows) if row["status"] == "accepted")
            if repeat == 0:
                baseline = estimate
            result = dict(repeat=repeat, injected_ms=delay, estimate_ms=estimate,
                          raw_error_ms=estimate - delay,
                          recovered_added_delay_ms=estimate - baseline,
                          baseline_corrected_error_ms=estimate - baseline - delay,
                          iterations=rows[-1]["iteration"], rejected_trials=rejected,
                          final_trial_status=rows[-1]["status"], validated=not rejected,
                          optimizer_ms=opt_ms, wall_s=wall_s, mean_reprojection_px=reprojection)
            results.append(result)
            traces.extend(dict(repeat=repeat, injected_ms=delay, **row) for row in rows)
            write_csv(output / "runs.csv", results)
            write_csv(output / "convergence.csv", traces)
            manifest["runs"].append(dict(log=f"logs/{stem}.log", command=command,
                                         exit_code=0, wall_s=wall_s))
            (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"{stem}: estimate={estimate:.6f} ms, iterations={result['iterations']}, "
                  f"error={result['raw_error_ms']:+.6f} ms", flush=True)
    summary = []
    for delay in (10, 20, 30, 40, 50):
        group = [row for row in results if row["injected_ms"] == delay]
        values = [row["estimate_ms"] for row in group]
        summary.append(dict(injected_ms=delay,
                            **{f"run{i}_ms": value for i, value in enumerate(values, 1)},
                            mean_ms=statistics.mean(values),
                            sample_std_ms=statistics.stdev(values) if len(values) > 1 else None,
                            validated=all(row["validated"] for row in group),
                            mean_raw_error_ms=statistics.mean(values) - delay,
                            mean_baseline_corrected_error_ms=statistics.mean(values) - baseline - delay))
    write_csv(output / "summary.csv", summary)
    plot_results(output, results, traces)


if __name__ == "__main__":
    main()
