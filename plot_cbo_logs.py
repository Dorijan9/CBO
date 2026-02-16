#!/usr/bin/env python3
"""
Plot CBO logs (LLM prior vs uniform) from CSV files produced by cbo_pathway_bandit.py.

Produces:
  1) Best posterior mean vs iteration
  2) Probability of selecting the optimal pathway vs iteration (simulate mode only)

Notes:
- Your CSV logs contain updates only for the chosen pathway each iteration.
  This script reconstructs full posteriors by replaying the sequence of chosen actions & outcomes.
- For plot (2), we need to know which pathway is "optimal" in the synthetic ground truth.
  Provide the same simulation parameters (seed, sim_base_rate, sim_signal, sim_noise) used during the run.
  If you averaged multiple seeds, pass multiple CSVs and matching seeds.

Example:
  python3 plot_cbo_logs.py \
    --prior_json prior.json \
    --phenotype phenotype_1 \
    --llm_csv logs/p1_llm_seed42.csv \
    --uniform_csv logs/p1_uniform_seed42.csv \
    --seed 42 \
    --iters 50 \
    --prior_strength 10 \
    --sim_base_rate 0.55 --sim_signal 0.50 --sim_noise 0.05 \
    --out_dir plots
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class Pathway:
    pathway_id: str
    name: str
    confidence: float


@dataclass
class Posterior:
    alpha: float
    beta: float

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def var(self) -> float:
        a, b = self.alpha, self.beta
        denom = (a + b) ** 2 * (a + b + 1.0)
        return (a * b) / denom if denom > 0 else 0.0


# ----------------------------
# Helpers
# ----------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def load_prior(prior_path: str, phenotype_key: str) -> List[Pathway]:
    with open(prior_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if phenotype_key not in data:
        raise KeyError(f"Phenotype key '{phenotype_key}' not found in {prior_path}.")
    pathways = []
    for item in data[phenotype_key]:
        c = clamp(float(item["confidence"]), 0.0, 1.0)
        pathways.append(
            Pathway(
                pathway_id=str(item["pathway_id"]),
                name=str(item.get("name", "")),
                confidence=c,
            )
        )
    if not pathways:
        raise ValueError(f"No pathways found for phenotype '{phenotype_key}'.")
    return pathways


def load_adjacency(adjacency_path: Optional[str]) -> Dict[str, List[str]]:
    if not adjacency_path:
        return {}
    with open(adjacency_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    adj: Dict[str, List[str]] = {}
    for k, v in data.items():
        if isinstance(v, list):
            adj[str(k)] = [str(x) for x in v]
    return adj


def init_posteriors(
    pathways: List[Pathway],
    prior_strength: float,
    baseline_uniform: bool,
    eps: float = 1e-3,
) -> Dict[str, Posterior]:
    post: Dict[str, Posterior] = {}
    for p in pathways:
        if baseline_uniform:
            a0, b0 = 1.0, 1.0
        else:
            c = clamp(p.confidence, eps, 1.0 - eps)
            a0 = 1.0 + prior_strength * c
            b0 = 1.0 + prior_strength * (1.0 - c)
        post[p.pathway_id] = Posterior(alpha=a0, beta=b0)
    return post


def bayes_update(
    posteriors: Dict[str, Posterior],
    chosen_id: str,
    y: int,
    adjacency: Dict[str, List[str]],
    coupling_lambda: float,
) -> None:
    y = 1 if y else 0
    posteriors[chosen_id].alpha += y
    posteriors[chosen_id].beta += (1 - y)

    if coupling_lambda <= 0.0:
        return

    for nid in adjacency.get(chosen_id, []):
        if nid not in posteriors:
            continue
        posteriors[nid].alpha += coupling_lambda * y
        posteriors[nid].beta += coupling_lambda * (1 - y)


def read_log(csv_path: str) -> List[Tuple[int, str, int]]:
    """
    Returns list of (t, chosen_pathway_id, outcome_y)
    """
    rows: List[Tuple[int, str, int]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            t = int(r["t"])
            pid = str(r["chosen_pathway_id"])
            y = int(r["outcome_y"])
            rows.append((t, pid, y))
    rows.sort(key=lambda x: x[0])
    return rows


def build_synthetic_ground_truth(
    pathways: List[Pathway],
    seed: int,
    signal_from_confidence: float,
    base_rate: float,
    noise: float,
) -> Dict[str, float]:
    rng = random.Random(seed)
    gt: Dict[str, float] = {}
    for p in pathways:
        prob = base_rate + signal_from_confidence * (p.confidence - 0.5)
        prob += rng.uniform(-noise, noise)
        gt[p.pathway_id] = clamp(prob, 0.01, 0.99)
    return gt


def replay_run(
    pathways: List[Pathway],
    log_rows: List[Tuple[int, str, int]],
    adjacency: Dict[str, List[str]],
    *,
    baseline_uniform: bool,
    prior_strength: float,
    coupling_lambda: float,
    iters: int,
) -> Tuple[List[float], List[str]]:
    """
    Reconstruct full posteriors by replaying actions/outcomes.
    Returns:
      best_posterior_mean_per_t (len iters)
      chosen_id_per_t (len iters)
    """
    post = init_posteriors(pathways, prior_strength, baseline_uniform)
    best_means: List[float] = []
    chosen_ids: List[str] = []

    # Map t -> (pid, y)
    by_t: Dict[int, Tuple[str, int]] = {t: (pid, y) for t, pid, y in log_rows}

    for t in range(1, iters + 1):
        if t not in by_t:
            raise ValueError(
                f"{t=} missing in {t=}..{iters} for this log. "
                f"Did you pass the correct --iters?"
            )
        pid, y = by_t[t]
        chosen_ids.append(pid)

        bayes_update(post, pid, y, adjacency, coupling_lambda)

        best_mean = max(pb.mean() for pb in post.values())
        best_means.append(best_mean)

    return best_means, chosen_ids


def average_curves(curves: List[List[float]]) -> List[float]:
    if not curves:
        raise ValueError("No curves provided to average.")
    L = len(curves[0])
    for c in curves:
        if len(c) != L:
            raise ValueError("All curves must have the same length to average.")
    return [sum(c[t] for c in curves) / len(curves) for t in range(L)]


def average_indicators(indicators: List[List[int]]) -> List[float]:
    # same as average curves but for 0/1 lists
    curves = [[float(x) for x in ind] for ind in indicators]
    return average_curves(curves)


# ----------------------------
# Plotting
# ----------------------------

def plot_best_posterior_mean(
    llm_curve: List[float],
    uniform_curve: List[float],
    out_path: str,
) -> None:
    x = list(range(1, len(llm_curve) + 1))
    plt.figure()
    plt.plot(x, llm_curve, label="LLM+KEGG prior")
    plt.plot(x, uniform_curve, label="Uniform prior")
    plt.xlabel("Iteration")
    plt.ylabel("Best posterior mean (max over pathways)")
    plt.title("Best posterior mean vs iteration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_prob_select_optimal(
    llm_prob: List[float],
    uniform_prob: List[float],
    out_path: str,
) -> None:
    x = list(range(1, len(llm_prob) + 1))
    plt.figure()
    plt.plot(x, llm_prob, label="LLM+KEGG prior")
    plt.plot(x, uniform_prob, label="Uniform prior")
    plt.xlabel("Iteration")
    plt.ylabel("P(select optimal pathway)")
    plt.title("Probability of selecting optimal pathway vs iteration")
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ----------------------------
# Main
# ----------------------------

def split_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior_json", type=str, required=True)
    parser.add_argument("--phenotype", type=str, required=True)

    parser.add_argument("--adjacency_json", type=str, default=None)
    parser.add_argument("--coupling_lambda", type=float, default=0.0)

    parser.add_argument("--iters", type=int, default=50)

    # You can pass multiple CSVs (comma-separated) to average across seeds.
    parser.add_argument("--llm_csv", type=str, required=True,
                        help="One or more CSV logs for LLM prior, comma-separated")
    parser.add_argument("--uniform_csv", type=str, required=True,
                        help="One or more CSV logs for uniform prior, comma-separated")

    # Priors used for replay (must match your run settings)
    parser.add_argument("--prior_strength", type=float, default=10.0)

    # Synthetic ground truth params (for probability-of-optimal plot)
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed used to generate synthetic ground truth (only for plot 2)")
    parser.add_argument("--sim_base_rate", type=float, default=0.55)
    parser.add_argument("--sim_signal", type=float, default=0.50)
    parser.add_argument("--sim_noise", type=float, default=0.05)

    parser.add_argument("--out_dir", type=str, default="plots")

    args = parser.parse_args()

    pathways = load_prior(args.prior_json, args.phenotype)
    adjacency = load_adjacency(args.adjacency_json)

    os.makedirs(args.out_dir, exist_ok=True)

    llm_csvs = split_csv_list(args.llm_csv)
    uni_csvs = split_csv_list(args.uniform_csv)

    # --- Replay to build curves ---
    llm_best_curves: List[List[float]] = []
    llm_chosen_ids_runs: List[List[str]] = []

    for path in llm_csvs:
        rows = read_log(path)
        best_curve, chosen_ids = replay_run(
            pathways,
            rows,
            adjacency,
            baseline_uniform=False,
            prior_strength=args.prior_strength,
            coupling_lambda=args.coupling_lambda,
            iters=args.iters,
        )
        llm_best_curves.append(best_curve)
        llm_chosen_ids_runs.append(chosen_ids)

    uni_best_curves: List[List[float]] = []
    uni_chosen_ids_runs: List[List[str]] = []

    for path in uni_csvs:
        rows = read_log(path)
        best_curve, chosen_ids = replay_run(
            pathways,
            rows,
            adjacency,
            baseline_uniform=True,
            prior_strength=args.prior_strength,  # ignored for uniform, but fine
            coupling_lambda=args.coupling_lambda,
            iters=args.iters,
        )
        uni_best_curves.append(best_curve)
        uni_chosen_ids_runs.append(chosen_ids)

    llm_best_avg = average_curves(llm_best_curves)
    uni_best_avg = average_curves(uni_best_curves)

    out1 = os.path.join(args.out_dir, "best_posterior_mean.png")
    plot_best_posterior_mean(llm_best_avg, uni_best_avg, out1)
    print(f"Saved: {out1}")

    # --- Plot probability of selecting the optimal pathway (simulate-mode only) ---
    gt = build_synthetic_ground_truth(
        pathways,
        seed=args.seed,
        signal_from_confidence=args.sim_signal,
        base_rate=args.sim_base_rate,
        noise=args.sim_noise,
    )
    optimal_id = max(gt.items(), key=lambda kv: kv[1])[0]
    print(f"Optimal pathway under synthetic ground truth: {optimal_id}  p={gt[optimal_id]:.3f}")

    llm_indicators: List[List[int]] = []
    for chosen_ids in llm_chosen_ids_runs:
        llm_indicators.append([1 if pid == optimal_id else 0 for pid in chosen_ids])

    uni_indicators: List[List[int]] = []
    for chosen_ids in uni_chosen_ids_runs:
        uni_indicators.append([1 if pid == optimal_id else 0 for pid in chosen_ids])

    llm_prob = average_indicators(llm_indicators)
    uni_prob = average_indicators(uni_indicators)

    out2 = os.path.join(args.out_dir, "prob_select_optimal.png")
    plot_prob_select_optimal(llm_prob, uni_prob, out2)
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
