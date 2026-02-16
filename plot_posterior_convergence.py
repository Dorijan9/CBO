#!/usr/bin/env python3
"""
Plot posterior distribution "convergence" for pathways over time.

Given a single CBO log CSV (from cbo_pathway_bandit.py), this script:
1) Replays Bayesian updates to reconstruct posteriors at every iteration.
2) Selects top-K pathways by final posterior mean.
3) Plots their posterior distributions at selected iterations.

You can choose:
- --dist beta     : plot exact Beta PDFs (correct for Bernoulli/Beta model)
- --dist normal   : plot Gaussian approximations using mean/variance of Beta

Outputs a PNG.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from math import gamma, sqrt, pi, exp


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
    paths = []
    for item in data[phenotype_key]:
        c = clamp(float(item["confidence"]), 0.0, 1.0)
        paths.append(Pathway(str(item["pathway_id"]), str(item.get("name", "")), c))
    return paths


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
        post[p.pathway_id] = Posterior(a0, b0)
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


def replay_posteriors_over_time(
    pathways: List[Pathway],
    log_rows: List[Tuple[int, str, int]],
    adjacency: Dict[str, List[str]],
    *,
    baseline_uniform: bool,
    prior_strength: float,
    coupling_lambda: float,
    iters: int,
) -> List[Dict[str, Posterior]]:
    """
    Returns a list of posterior snapshots after each iteration t (1..iters).
    Each entry is a dict: pathway_id -> Posterior(alpha, beta).
    """
    post = init_posteriors(pathways, prior_strength, baseline_uniform)
    by_t = {t: (pid, y) for t, pid, y in log_rows}

    snapshots: List[Dict[str, Posterior]] = []
    for t in range(1, iters + 1):
        if t not in by_t:
            raise ValueError(f"Missing iteration {t} in log; check --iters.")
        pid, y = by_t[t]
        bayes_update(post, pid, y, adjacency, coupling_lambda)

        # deep-ish copy (Posterior is simple)
        snap = {k: Posterior(v.alpha, v.beta) for k, v in post.items()}
        snapshots.append(snap)

    return snapshots


# ----------------------------
# PDFs
# ----------------------------

def beta_pdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    # Beta(a,b) pdf = x^{a-1} (1-x)^{b-1} / B(a,b)
    # B(a,b) = Gamma(a)Gamma(b)/Gamma(a+b)
    B = gamma(a) * gamma(b) / gamma(a + b)
    return (x ** (a - 1)) * ((1 - x) ** (b - 1)) / B


def normal_pdf(x: np.ndarray, mu: float, var: float) -> np.ndarray:
    var = max(var, 1e-9)
    return (1.0 / sqrt(2 * pi * var)) * np.exp(-0.5 * ((x - mu) ** 2) / var)


# ----------------------------
# Plot
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior_json", type=str, required=True)
    parser.add_argument("--phenotype", type=str, required=True)
    parser.add_argument("--log_csv", type=str, required=True)

    parser.add_argument("--adjacency_json", type=str, default=None)
    parser.add_argument("--coupling_lambda", type=float, default=0.0)

    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--baseline_uniform", action="store_true")
    parser.add_argument("--prior_strength", type=float, default=10.0)

    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--checkpoints", type=str, default="1,10,25,50",
                        help="Comma-separated iterations to plot (must be <= iters)")
    parser.add_argument("--dist", choices=["beta", "normal"], default="beta")

    parser.add_argument("--out", type=str, default="posterior_convergence.png")
    args = parser.parse_args()

    pathways = load_prior(args.prior_json, args.phenotype)
    id_to_name = {p.pathway_id: p.name for p in pathways}
    adjacency = load_adjacency(args.adjacency_json)

    log_rows = read_log(args.log_csv)
    snaps = replay_posteriors_over_time(
        pathways,
        log_rows,
        adjacency,
        baseline_uniform=args.baseline_uniform,
        prior_strength=args.prior_strength,
        coupling_lambda=args.coupling_lambda,
        iters=args.iters,
    )

    checkpoints = [int(x.strip()) for x in args.checkpoints.split(",") if x.strip()]
    for t in checkpoints:
        if t < 1 or t > args.iters:
            raise ValueError(f"Checkpoint {t} is out of range 1..{args.iters}")

    final = snaps[-1]
    ranked = sorted(final.items(), key=lambda kv: kv[1].mean(), reverse=True)
    top_ids = [pid for pid, _ in ranked[: args.top_k]]

    x = np.linspace(1e-4, 1 - 1e-4, 800)

    plt.figure()
    for pid in top_ids:
        for t in checkpoints:
            pb = snaps[t - 1][pid]
            if args.dist == "beta":
                y = beta_pdf(x, pb.alpha, pb.beta)
                label = f"{pid} @ t={t} (Beta)"
            else:
                mu, var = pb.mean(), pb.var()
                y = normal_pdf(x, mu, var)
                label = f"{pid} @ t={t} (Normal approx)"

            plt.plot(x, y, label=label)

    title_prior = "Uniform prior" if args.baseline_uniform else "LLM+KEGG prior"
    plt.title(f"Posterior distribution convergence ({title_prior})\nTop-{args.top_k} pathways by final mean")
    plt.xlabel(r"Pathway success probability $\theta$")
    plt.ylabel("Density")
    plt.legend(fontsize=8)
    plt.tight_layout()

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.out, dpi=200)
    plt.close()

    print(f"Saved: {args.out}")
    print("Top pathways (by final posterior mean):")
    for pid in top_ids:
        pb = final[pid]
        print(f"  {pid}  mean={pb.mean():.3f}  var={pb.var():.4f}  name={id_to_name.get(pid,'')}")


if __name__ == "__main__":
    main()
