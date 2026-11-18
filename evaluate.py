#!/usr/bin/env python3
"""Validate the forecaster in main.py (local checks only; not run by the grader).

Two checks, both reproducible from the kit's own data alone:

1. FORMAT + DETERMINISM: run main.py on the kit's practice inputs twice
   into two directories and verify the outputs are byte-identical, each a
   valid forecast (exactly one "hours" key, 23 rows for hours 2-24, five
   finite quantiles per row, each > 0, <= 1e9 and non-decreasing), and
   report the wall time per case.

2. LEAVE-ONE-OUT (LOO) over the comparables: every comparable launch is
   forecast as if it were the target -- its own hourly series up to its
   release-day checkpoint as the history, the other comparables as the
   comparables -- and scored against its actual hours 2-24 with the
   competition's pinball loss on the log scale. The mean LOO loss is the
   honest measure of how the forecaster generalises to an unseen launch
   (the final case), unlike the practice score, which every strong entry
   aces. The house baseline is scored the same way for comparison.

Usage:
    python3 evaluate.py <kit-input-dir>     # e.g. ../gta6-...-kit/input
"""
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main as forecaster

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]


def parse_ts(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def pinball(answer, rows):
    """Mean pinball loss on the log scale; also typical miss and coverage."""
    total, count, miss, covered = 0.0, 0, 0.0, 0
    for hour, actual in answer.items():
        actual = max(actual, 1)
        log_actual = math.log(actual)
        quantiles = rows[hour]
        for tau, q in zip(LEVELS, quantiles):
            d = log_actual - math.log(max(q, 1e-9))
            total += 2 * max(tau * d, (tau - 1) * d)
            count += 1
        miss += abs(log_actual - math.log(max(quantiles[2], 1e-9)))
        covered += 1 if quantiles[0] <= actual <= quantiles[4] else 0
    n = len(answer)
    return total / count, 100 * (math.exp(miss / n) - 1), covered / n


def validate_output(path):
    """Check one .out file against the competition's format rules."""
    size = os.path.getsize(path)
    assert size <= 65536, f"{path}: {size} bytes, over the 65,536 cap"
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert list(payload.keys()) == ["hours"], f"{path}: keys {payload.keys()}"
    rows = payload["hours"]
    assert len(rows) == 23, f"{path}: {len(rows)} rows, want 23"
    assert sorted(r["hour"] for r in rows) == list(range(2, 25)), f"{path}: hours"
    for row in rows:
        assert set(row.keys()) == {"hour", "q"}, f"{path}: row keys {row.keys()}"
        q = row["q"]
        assert len(q) == 5, f"{path}: hour {row['hour']} has {len(q)} quantiles"
        assert all(math.isfinite(v) and 0 < v <= 1e9 for v in q), f"{path}: {q}"
        assert all(q[i] <= q[i + 1] for i in range(4)), f"{path}: not sorted {q}"
    return {r["hour"]: r["q"] for r in rows}


def check_format_and_determinism(input_dir):
    print("== format + determinism ==")
    started = time.time()
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        env = dict(os.environ, MAKEFUN_INPUT_DIR=input_dir)
        for out_dir in (d1, d2):
            env["MAKEFUN_OUTPUT_DIR"] = out_dir
            subprocess.run([sys.executable, "main.py"], env=env,
                           cwd=os.path.dirname(os.path.abspath(__file__)),
                           check=True, capture_output=True)
        names = sorted(n for n in os.listdir(d1) if n.endswith(".out"))
        assert names, "no outputs written"
        for name in names:
            with open(os.path.join(d1, name), "rb") as a, \
                    open(os.path.join(d2, name), "rb") as b:
                assert a.read() == b.read(), f"{name}: runs differ"
            validate_output(os.path.join(d1, name))
            print(f"  {name}: format OK, byte-identical across runs")
    elapsed = time.time() - started
    print(f"  {len(names)} cases x 2 runs in {elapsed:.2f}s "
          f"({elapsed / (2 * len(names)):.2f}s per case; cap is 8s)")


def loo_validation(case_path):
    print("== leave-one-out over comparables ==")
    with open(case_path, "r", encoding="utf-8") as f:
        case = json.load(f)
    comps = case["comparables"]
    losses, base_losses, coverages = [], [], []
    for target in comps:
        hourly = {}
        for point in target["hourly"]:
            if point.get("viewers"):
                hourly[parse_ts(point["from"])] = point["viewers"]
        day_start = parse_ts(target["day_start"])
        day = [hourly.get(day_start + timedelta(hours=h)) for h in range(24)]
        if day[0] is None or any(v is None for v in day):
            continue
        pseudo = {
            "day_start": target["day_start"],
            "known": [{"viewers": day[0]}],
            "history_hourly": [
                {"from": (day_start + timedelta(hours=off)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"),
                 "viewers": hourly[day_start + timedelta(hours=off)]}
                for off in range(-48, 1)
                if (day_start + timedelta(hours=off)) in hourly
            ],
            "comparables": [c for c in comps if c is not target],
        }
        answer = {h + 1: day[h] for h in range(1, 24)}
        rows = forecaster.forecast_case(pseudo)
        loss, _, cov = pinball(answer, rows)
        base_loss, _, _ = pinball(
            answer, forecaster.baseline_forecast(max(day[0], 1)))
        losses.append(loss)
        base_losses.append(base_loss)
        coverages.append(cov)
        print(f"  {target['name'][:34]:34s} loss={loss:.4f} "
              f"cov90={cov:.2f} (baseline {base_loss:.4f})")
    print(f"  LOO n={len(losses)} mean loss={statistics.fmean(losses):.6f} "
          f"(house baseline {statistics.fmean(base_losses):.6f}), "
          f"mean 90% coverage={statistics.fmean(coverages):.2f}")


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    input_dir = argv[1]
    check_format_and_determinism(input_dir)
    loo_validation(os.path.join(input_dir, "p00.in"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
