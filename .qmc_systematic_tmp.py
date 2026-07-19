from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh


ROOT = Path(__file__).resolve().parent
EXE = ROOT / "cpp/build/release/examples/qmc_interacting_demo"
OUT = Path("/tmp/qmc_systematic_20260719")
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = (10101, 20202, 30303, 40404)


def compositions(total: int, parts: int):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in compositions(total - first, parts - 1):
            yield (first, *rest)


def exact_diagonalization(N: int, L: int, beta: float, t: float, U: float):
    basis = list(compositions(N, L))
    index = {state: i for i, state in enumerate(basis)}
    H = np.zeros((len(basis), len(basis)), dtype=np.float64)
    pairs = np.zeros(len(basis), dtype=np.float64)
    for col, state_tuple in enumerate(basis):
        state = list(state_tuple)
        pairs[col] = sum(n * (n - 1) / 2 for n in state)
        H[col, col] = U * pairs[col]
        for site in range(L):
            neighbor = (site + 1) % L
            if state[neighbor] > 0:
                final = state.copy()
                final[neighbor] -= 1
                final[site] += 1
                H[index[tuple(final)], col] += -t * math.sqrt(
                    (state[site] + 1) * state[neighbor]
                )
            if state[site] > 0:
                final = state.copy()
                final[site] -= 1
                final[neighbor] += 1
                H[index[tuple(final)], col] += -t * math.sqrt(
                    (state[neighbor] + 1) * state[site]
                )
    evals, evecs = eigh(H)
    weights = np.exp(-beta * (evals - evals[0]))
    Z = float(weights.sum())
    pair_eigen = (np.abs(evecs) ** 2).T @ pairs
    energy = float(weights @ evals / Z)
    pair_count = float(weights @ pair_eigen / Z)
    return {
        "total_energy": energy,
        "interaction_energy": U * pair_count,
        "kinetic_energy": energy - U * pair_count,
        "double_occupancy_per_site": pair_count / L,
        "hilbert_dimension": len(basis),
    }


def autocorrelation_metrics(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    n = x.size
    centered = x - x.mean()
    variance = float(np.mean(centered * centered))
    if variance == 0.0:
        return {"tau": float("inf"), "ess": 0.0, "se": 0.0}
    padded = 1 << (2 * n - 1).bit_length()
    transform = np.fft.rfft(centered, padded)
    autocov = np.fft.irfft(transform * transform.conjugate(), padded)[:n]
    autocov /= np.arange(n, 0, -1)
    rho = autocov / autocov[0]
    positive_sum = 0.0
    previous_pair = float("inf")
    for lag in range(1, n - 1, 2):
        pair = float(rho[lag] + rho[lag + 1])
        if pair <= 0.0:
            break
        pair = min(pair, previous_pair)
        positive_sum += pair
        previous_pair = pair
    tau = max(1.0, 1.0 + 2.0 * positive_sum)
    return {
        "tau": tau,
        "ess": n / tau,
        "se": math.sqrt(float(np.var(x, ddof=1)) * tau / n),
    }


def split_rhat(chains: list[np.ndarray]):
    half = min(len(x) for x in chains) // 2
    split = [x[:half] for x in chains] + [x[-half:] for x in chains]
    means = np.array([x.mean() for x in split])
    variances = np.array([x.var(ddof=1) for x in split])
    W = float(variances.mean())
    B = half * float(means.var(ddof=1))
    if W == 0.0:
        return 1.0 if B == 0.0 else float("inf")
    var_hat = (half - 1) / half * W + B / half
    return math.sqrt(max(0.0, var_hat / W))


def parse_stdout(stdout: str, key: str):
    match = re.search(rf"^{key} acceptance = ([0-9.eE+-]+|n/a)$", stdout, re.MULTILINE)
    return float(match.group(1)) if match and match.group(1) != "n/a" else None


def parse_stitch_topology_rate(stdout: str):
    match = re.search(
        r"^stitch topology changes/attempt = ([0-9.eE+-]+|n/a)$",
        stdout,
        re.MULTILINE,
    )
    return float(match.group(1)) if match and match.group(1) != "n/a" else None


def default_stitch_fraction(case: dict):
    """Use the feature-branch ground-state calibration at unit filling."""
    beta = float(case["beta"])
    hopping = float(case.get("t", 1.0))
    interaction = float(case["U"])
    volume = int(case["L"]) ** int(case.get("d", 1))
    if hopping <= 0.0 or int(case["N"]) != volume:
        return 0.75
    duration = 1.0 / hopping if interaction / hopping <= 5.0 else 5.0 / interaction
    return min(1.0, duration / beta)


def run_chain(case: dict, seed: int, samples: int, burn: int):
    path = OUT / f"{case['name']}_seed{seed}.dat"
    cmd = [
        str(EXE), "-n", str(case["N"]), "-l", str(case["L"]),
        "-d", str(case.get("d", 1)), "-b", str(case["beta"]),
        "-t", str(case.get("t", 1.0)), "-u", str(case["U"]),
        "--samples", str(samples), "--burn-in", str(burn),
        "--thin", str(case.get("thin", 1)), "--seed", str(seed),
        "--segment-updates", str(case.get("segment_updates", case["N"])),
        "--segment-fraction", str(case.get("segment_fraction", 0.35)),
        "--cycle-updates", str(case.get("cycle_updates", 1)),
        "--global-updates", str(case.get("global_updates", 0)),
        "--stitch-updates", str(case.get("stitch_updates", max(1, case["N"]))),
        "--stitch-fraction",
        str(case.get("stitch_fraction", default_stitch_fraction(case))),
        "--stitch-locality-radius", str(case.get("stitch_locality_radius", 1)),
        "--stitch-global-partner-probability",
        str(case.get("stitch_global_partner_probability", 0.05)),
        "-o", str(path),
    ]
    start = time.perf_counter()
    result = subprocess.run(cmd, text=True, capture_output=True)
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        return {
            "seed": seed, "returncode": result.returncode, "elapsed": elapsed,
            "stdout": result.stdout, "stderr": result.stderr,
        }
    values = np.loadtxt(path)
    columns = {
        "total_energy": values[:, 1],
        "kinetic_energy": values[:, 2],
        "interaction_energy": values[:, 3],
        "double_occupancy_per_site": values[:, 4],
        "winding_squared": values[:, 5],
        "event_count": values[:, 6],
    }
    return {
        "seed": seed, "returncode": 0, "elapsed": elapsed,
        "acceptance": {
            k: parse_stdout(result.stdout, k)
            for k in ("segment", "cycle", "stitch", "time_shift", "global")
        },
        "stitch_topology_rate": parse_stitch_topology_rate(result.stdout),
        "columns": columns,
    }


def summarize_case(case: dict, samples: int, burn: int):
    print(f"RUN {case['name']}", flush=True)
    runs = [run_chain(case, seed, samples, burn) for seed in SEEDS[:case.get("chains", 4)]]
    failures = [x for x in runs if x["returncode"] != 0]
    if failures:
        summary = {"case": case, "samples": samples, "burn": burn, "failures": failures}
        print(f"FAIL {case['name']} {failures[0]['stderr'].strip()}", flush=True)
        return summary
    exact = None
    if case.get("ed", False):
        exact = exact_diagonalization(case["N"], case["L"], case["beta"], case.get("t", 1.0), case["U"])
    result = {
        "case": case, "samples": samples, "burn": burn,
        "elapsed_total": sum(x["elapsed"] for x in runs),
        "acceptance": {
            move: (
                float(np.mean([x["acceptance"][move] for x in runs if x["acceptance"][move] is not None]))
                if any(x["acceptance"][move] is not None for x in runs)
                else None
            )
            for move in ("segment", "cycle", "stitch", "time_shift", "global")
        },
        "stitch_topology_rate": (
            float(np.mean([x["stitch_topology_rate"] for x in runs]))
            if all(x["stitch_topology_rate"] is not None for x in runs)
            else None
        ),
        "exact": exact,
        "metrics": {},
    }
    for column in runs[0]["columns"]:
        chains = [x["columns"][column] for x in runs]
        ac = [autocorrelation_metrics(x) for x in chains]
        chain_means = np.array([x.mean() for x in chains])
        within_se = math.sqrt(sum(x["se"] ** 2 for x in ac)) / len(ac)
        between_se = float(chain_means.std(ddof=1) / math.sqrt(len(chains)))
        combined_se = max(within_se, between_se)
        metric = {
            "mean": float(chain_means.mean()),
            "se": combined_se,
            "within_se": within_se,
            "between_se": between_se,
            "tau_max": max(x["tau"] for x in ac),
            "tau_median": float(np.median([x["tau"] for x in ac])),
            "ess_total": sum(x["ess"] for x in ac),
            "rhat": split_rhat(chains),
        }
        if column == "winding_squared":
            metric["transitions_total"] = sum(int(np.count_nonzero(np.diff(x))) for x in chains)
            metric["nonzero_fraction"] = float(np.mean(np.concatenate(chains) != 0.0))
            metric["chain_means"] = chain_means.tolist()
        if exact is not None and column in exact:
            metric["exact"] = exact[column]
            metric["delta"] = metric["mean"] - exact[column]
            metric["z"] = metric["delta"] / combined_se if combined_se > 0 else float("inf")
        result["metrics"][column] = metric
    e = result["metrics"]["total_energy"]
    topology = result["stitch_topology_rate"]
    topology_text = "n/a" if topology is None else f"{topology:.4g}"
    print(
        f"OK {case['name']} E={e['mean']:.6g} se={e['se']:.3g} "
        f"tau={e['tau_max']:.2f} Rhat={e['rhat']:.4f} "
        f"acc={result['acceptance']} "
        f"topo={topology_text}", flush=True
    )
    return result


def small_cases():
    cases = []
    for U in (0.0, 1.2, 4.0, 8.0, 16.0, 32.0):
        cases.append(dict(name=f"ed_n2_l3_b0p8_u{U:g}", N=2, L=3, beta=0.8, U=U, ed=True))
    for U in (0.0, 2.0, 4.0, 8.0, 12.0, 16.0, 32.0):
        cases.append(dict(name=f"ed_n4_l4_b2_u{U:g}", N=4, L=4, beta=2.0, U=U, ed=True))
    for U in (4.0, 16.0):
        cases.append(dict(name=f"ed_n3_l4_b1p5_u{U:g}", N=3, L=4, beta=1.5, U=U, ed=True))
    return cases


def large_cases():
    cases = []
    for N in (6, 8, 10, 12, 14, 16):
        cases.append(dict(name=f"size_1d_n{N}_b2_u16", N=N, L=N, beta=2.0, U=16.0))
    for U in (0.0, 2.0, 4.0, 8.0, 16.0, 32.0):
        cases.append(dict(name=f"large_1d_n8_b2_u{U:g}", N=8, L=8, beta=2.0, U=U))
    for U in (2.0, 8.0, 16.0):
        cases.append(dict(name=f"large_2d_n16_l4_b1_u{U:g}", N=16, L=4, d=2, beta=1.0, U=U))
    cases.extend([
        dict(name="dilute_1d_n8_l16_b2_u8", N=8, L=16, beta=2.0, U=8.0),
        dict(name="dilute_1d_n8_l16_b2_u16", N=8, L=16, beta=2.0, U=16.0),
    ])
    return cases


def tuning_cases():
    cases = []
    base = dict(N=4, L=4, beta=2.0, U=16.0, ed=True)
    for fraction in (0.1, 0.2, 0.35, 0.6, 0.75, 1.0):
        cases.append(base | dict(name=f"tune_stitch_fraction_{fraction:g}", stitch_fraction=fraction))
    cases += [
        base | dict(
            name="tune_global_only",
            segment_updates=0,
            cycle_updates=0,
            stitch_updates=0,
            global_updates=1,
        ),
        base | dict(name="tune_stitch_only", segment_updates=0, cycle_updates=0),
        base | dict(name="tune_stitch_plus_global", global_updates=1),
        base | dict(name="tune_more_stitches", stitch_updates=8),
        base | dict(name="tune_stitch_plus_segments", segment_updates=8),
    ]
    return cases


def regime_map_cases():
    cases = []
    for beta in (0.5, 1.0, 1.5, 2.0, 3.0):
        cases.append(dict(name=f"beta_n4_l4_b{beta:g}_u16", N=4, L=4, beta=beta, U=16.0, ed=True))
    for N in (2, 3, 4):
        cases.append(dict(name=f"density_n{N}_l4_b2_u16", N=N, L=4, beta=2.0, U=16.0, ed=True))
    for N in (5, 6):
        for U in (4.0, 8.0, 16.0, 32.0):
            cases.append(dict(name=f"ed_n{N}_l{N}_b2_u{U:g}", N=N, L=N, beta=2.0, U=U, ed=True))
    return cases


def long_cases():
    base = dict(N=4, L=4, beta=2.0, U=16.0, ed=True)
    return [
        base | dict(name="long_default_u16"),
        base | dict(name="long_stitch4_u16", stitch_updates=4),
        base | dict(name="long_stitch16_u16", stitch_updates=16),
        base | dict(name="long_stitch4_global1_u16", stitch_updates=4, global_updates=1),
        (base | dict(U=32.0, name="long_stitch16_u32", stitch_updates=16)),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("small", "large", "tuning", "map", "long"))
    parser.add_argument("--samples", type=int)
    parser.add_argument("--burn", type=int)
    args = parser.parse_args()
    if args.phase == "small":
        cases, samples, burn = small_cases(), args.samples or 30_000, args.burn or 5_000
    elif args.phase == "large":
        cases, samples, burn = large_cases(), args.samples or 12_000, args.burn or 3_000
    elif args.phase == "tuning":
        cases, samples, burn = tuning_cases(), args.samples or 30_000, args.burn or 5_000
    elif args.phase == "map":
        cases, samples, burn = regime_map_cases(), args.samples or 30_000, args.burn or 5_000
    else:
        cases, samples, burn = long_cases(), args.samples or 100_000, args.burn or 20_000
    output = [summarize_case(case, samples, burn) for case in cases]
    target = OUT / f"summary_{args.phase}.json"
    target.write_text(json.dumps(output, indent=2, allow_nan=True))
    print(f"WROTE {target}")


if __name__ == "__main__":
    main()


##  ### ED comparison
##
##  Four chains were run with 30,000 retained samples and 5,000 burn-in sweeps each. Errors account for autocorrelation and cross-chain variation.
##
##   (N,L,\beta,U)    ED energy          QMC energy    max (\tau_E)    (\hat R_E)    global acc.    Assessment
##  ━━━━━━━━━━━━━━━  ━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━  ━━━━━━━━━━━━  ━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━
##   2,3,0.8,8          -2.3563    -2.3485 ± 0.0125             2.6         1.000          39.8%    Good
##  ───────────────  ───────────  ──────────────────  ──────────────  ────────────  ─────────────  ──────────────────
##   2,3,0.8,32         -1.8002    -1.8034 ± 0.0133             4.3         1.000          23.8%    Good
##  ───────────────  ───────────  ──────────────────  ──────────────  ────────────  ─────────────  ──────────────────
##   4,4,2,4            -3.9736    -3.9738 ± 0.0186              11         1.000          8.28%    Good
##  ───────────────  ───────────  ──────────────────  ──────────────  ────────────  ─────────────  ──────────────────
##   4,4,2,8            -2.1610    -2.1400 ± 0.0500              79         1.002         0.321%    Correct but slow
##  ───────────────  ───────────  ──────────────────  ──────────────  ────────────  ─────────────  ──────────────────
##   4,4,2,16           -1.0339    -1.3399 ± 0.0761             761         1.033        0.0057%    Biased/unmixed
##  ───────────────  ───────────  ──────────────────  ──────────────  ────────────  ─────────────  ──────────────────
##   4,4,2,32           -0.5047    -1.3365 ± 0.3248             528         1.163        0.0014%    Failed
##  ───────────────  ───────────  ──────────────────  ──────────────  ────────────  ─────────────  ──────────────────
##   5,5,2,8            -2.5720    -2.6297 ± 0.0505             165         1.001         0.075%    Borderline
##  ───────────────  ───────────  ──────────────────  ──────────────  ────────────  ─────────────  ──────────────────
##   5,5,2,16           -1.2569    -1.9055 ± 0.3346             792         1.131        0.0021%    Failed
##  ───────────────  ───────────  ──────────────────  ──────────────  ────────────  ─────────────  ──────────────────
##   6,6,2,8            -3.0171    -2.9764 ± 0.1403             111         1.017        0.0107%    Borderline
##  ───────────────  ───────────  ──────────────────  ──────────────  ────────────  ─────────────  ──────────────────
##   6,6,2,16           -1.4978    -2.0963 ± 0.2850             147         1.072        0.0029%    Failed
##
##  Double occupancy exposes the same problem more strongly. For N=L=4, β=2, QMC gives 0.02351 versus ED 0.01705 at U=16, and 0.01508 versus 0.00401 at U=32.
##
##  The regime boundary is controlled by more than (U):
##
##  - At N=L=4, U=16, results remain consistent with ED for β≤1, become marginal at β=1.5, and are significantly biased at β=2–3.
##  - At L=4, β=2, U=16, half filling (N=2) is healthy: (\tau_E≈4.8), global acceptance 16.4%. Three-quarter filling is much slower, while unit filling fails.
##  - Segment acceptance can remain 45–65% in failed runs, so it is not a sufficient health indicator. Segment updates preserve cycle topology; the only topology-changing global ideal
##    proposal is effectively dead.
##
##  ### Larger systems
##
##  For 1D N=L=8, β=2, using four 12,000-sample chains:
##
##   (U)    max (\tau_E)    (\hat R_E)    global acc.    sampled winding changes
##  ━━━━━  ━━━━━━━━━━━━━━  ━━━━━━━━━━━━  ━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━
##     0             1.0         1.000           100%                     19,582
##  ─────  ──────────────  ────────────  ─────────────  ─────────────────────────
##     2              10         1.001          7.25%                      2,406
##  ─────  ──────────────  ────────────  ─────────────  ─────────────────────────
##     4              44         1.004         0.585%                        256
##  ─────  ──────────────  ────────────  ─────────────  ─────────────────────────
##     8              64         1.012        0.0117%                          3
##  ─────  ──────────────  ────────────  ─────────────  ─────────────────────────
##    16             107         1.017        0.0050%                          0
##  ─────  ──────────────  ────────────  ─────────────  ─────────────────────────
##    32             356         1.090        0.0033%                          0
##
##  The 2D 4×4, N=16, β=1 case is healthy at U=2, but at U=8–16 has (\hat R_E≈1.03–1.04), near-zero global acceptance, and major winding-sector disagreement between chains.
##
##  Dilution helps considerably: N=8, L=16, β=2 retains segment/cycle acceptances of roughly 79–85%/9–15% at U=8–16. Energy mixing is better, though global topology mixing remains weak.
##
##  ### Long-run and tuning result
##
##  For N=L=4, β=2, U=16, 100,000 samples per chain:
##
##  - Default schedule: (-1.5595±0.0367) versus ED (-1.0339), a 14σ discrepancy despite a misleadingly good (\hat R=1.003).
##  - Sixteen global proposals per sweep: (-1.0199±0.0579), consistent with ED, but (\tau_E≈1,239), only about 11 effective energy samples/second, and ~55 seconds for four tiny-system
##    chains.
##
##  - At U=32, the same aggressive schedule remains unmixed: (\hat R=1.21), (\tau_E≈14,700), zero winding changes.
##
##  Thus brute-force global proposals can recover U=16 for very small systems, but this is not efficient and does not scale. Changing only segment length or adding segment updates did not
##  fix the underlying topology bottleneck.
##
##  ### Separate numerical failure
##
##  There is also a process-abort boundary unrelated to interaction sampling. Global-only U=0 tests with 20 seeds and 500 samples each produced:
##
##  - β=2, N=L≤13: 0/20 aborts.
##  - β=2, N=L=14–16: 20/20 aborts.
##  - β≤1, N=L≤16: 0/20 aborts.
##
##  The executable dies with GSL special-function underflow. The code expects Bessel underflow to be returned and converted to zero, but the default GSL error handler aborts first.
