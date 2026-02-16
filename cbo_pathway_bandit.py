#!/usr/bin/env python3
"""
CBO-style Bayesian optimisation over a structured intervention space (KEGG pathways)
with an LLM-informed prior over phenotype->pathway graph edges.

- Prior: Beta(1 + s*c, 1 + s*(1-c)) where c is LLM confidence in [0,1]
- Acquisition: Thompson sampling over Beta posteriors
- Likelihood: Bernoulli outcome y in {0,1} (success/improvement)
- Update: alpha += y, beta += (1-y)
- Optional coupling: propagate a fraction of the update to KEGG-neighbour pathways

Modes:
  --mode simulate : uses a synthetic ground truth mapping pathway->success probability
  --mode real     : calls evaluate_intervention() stub (you plug in your real evaluation)

Outputs:
  - CSV log with per-iteration chosen pathway, sampled theta, outcome, posterior stats
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class Pathway:
    pathway_id: str
    name: str
    confidence: float  # LLM confidence in [0,1]


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
    # normalise values to lists of strings
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


def thompson_select(
    posteriors: Dict[str, Posterior],
    rng: random.Random,
) -> Tuple[str, float]:
    # sample theta_i ~ Beta(alpha_i, beta_i), pick argmax
    best_id = None
    best_sample = -1.0
    for pid, pb in posteriors.items():
        sample = rng.betavariate(pb.alpha, pb.beta)
        if sample > best_sample:
            best_sample = sample
            best_id = pid
    assert best_id is not None
    return best_id, best_sample


# ----------------------------
# Environment / evaluation
# ----------------------------

def build_synthetic_ground_truth(
    pathways: List[Pathway],
    rng: random.Random,
    signal_from_confidence: float,
    base_rate: float,
    noise: float,
) -> Dict[str, float]:
    """
    Creates a synthetic success probability per pathway.

    If signal_from_confidence > 0, higher LLM confidence tends to mean higher success probability
    (good for demonstrating that LLM priors help).
    """
    gt: Dict[str, float] = {}
    for p in pathways:
        # map confidence to a success prob with some noise
        prob = base_rate + signal_from_confidence * (p.confidence - 0.5)
        prob += rng.uniform(-noise, noise)
        gt[p.pathway_id] = clamp(prob, 0.01, 0.99)
    return gt


def evaluate_intervention_simulate(
    pathway_id: str,
    gt_probs: Dict[str, float],
    rng: random.Random,
) -> int:
    p = gt_probs[pathway_id]
    return 1 if rng.random() < p else 0


def evaluate_intervention_real(pathway_id: str) -> int:
    """
    TODO: Replace this with your real evaluation function.

    It should return:
      y = 1 if intervention improved the phenotype (or exceeded threshold),
      y = 0 otherwise.

    If your outcome is continuous (score), you can binarise using a threshold,
    or upgrade the model to Gaussian likelihood.
    """
    raise NotImplementedError(
        "Implement evaluate_intervention_real(pathway_id) with your own scoring."
    )


# ----------------------------
# Update rules (with optional coupling)
# ----------------------------

def bayes_update(
    posteriors: Dict[str, Posterior],
    chosen_id: str,
    y: int,
    adjacency: Dict[str, List[str]],
    coupling_lambda: float,
) -> None:
    y = 1 if y else 0

    # main update
    posteriors[chosen_id].alpha += y
    posteriors[chosen_id].beta += (1 - y)

    if coupling_lambda <= 0.0:
        return

    # propagate a soft fractional update to neighbours
    neigh = adjacency.get(chosen_id, [])
    for nid in neigh:
        if nid not in posteriors:
            continue
        posteriors[nid].alpha += coupling_lambda * y
        posteriors[nid].beta += coupling_lambda * (1 - y)


# ----------------------------
# Main loop
# ----------------------------

def run(
    pathways: List[Pathway],
    posteriors: Dict[str, Posterior],
    adjacency: Dict[str, List[str]],
    *,
    iters: int,
    seed: int,
    mode: str,
    coupling_lambda: float,
    log_csv_path: str,
    gt_probs: Optional[Dict[str, float]] = None,
) -> None:
    rng = random.Random(seed)

    id_to_name = {p.pathway_id: p.name for p in pathways}

    os.makedirs(os.path.dirname(log_csv_path) or ".", exist_ok=True)

    with open(log_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "t",
            "chosen_pathway_id",
            "chosen_pathway_name",
            "thompson_sample",
            "outcome_y",
            "posterior_alpha",
            "posterior_beta",
            "posterior_mean",
            "posterior_var",
        ])

        for t in range(1, iters + 1):
            chosen_id, sample = thompson_select(posteriors, rng)

            if mode == "simulate":
                assert gt_probs is not None
                y = evaluate_intervention_simulate(chosen_id, gt_probs, rng)
            elif mode == "real":
                y = evaluate_intervention_real(chosen_id)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            bayes_update(posteriors, chosen_id, y, adjacency, coupling_lambda)

            pb = posteriors[chosen_id]
            mean = pb.mean()
            var = pb.var()

            writer.writerow([
                t,
                chosen_id,
                id_to_name.get(chosen_id, ""),
                f"{sample:.6f}",
                int(y),
                f"{pb.alpha:.6f}",
                f"{pb.beta:.6f}",
                f"{mean:.6f}",
                f"{var:.6f}",
            ])

    # print a short summary
    print(f"Saved log to: {log_csv_path}")

    ranked = sorted(posteriors.items(), key=lambda kv: kv[1].mean(), reverse=True)
    print("\nTop pathways by posterior mean:")
    for pid, pb in ranked[:5]:
        print(f"  {pid:>10s}  mean={pb.mean():.3f}  var={pb.var():.4f}  name={id_to_name.get(pid,'')}")

    # Best at the END of the run
    best_id, best_pb = ranked[0]
    print(f"\nBest (final, by posterior mean): {best_id}  mean={best_pb.mean():.3f}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior_json", type=str, required=True, help="Path to prior.json")
    parser.add_argument("--phenotype", type=str, required=True, help="Phenotype key in prior.json (e.g. phenotype_1)")
    parser.add_argument("--adjacency_json", type=str, default=None, help="Optional adjacency.json for pathway coupling")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--baseline_uniform", action="store_true",
                        help="Use uniform Beta(1,1) priors instead of LLM-informed priors")

    parser.add_argument("--prior_strength", type=float, default=10.0,
                        help="s in Beta(1+s*c, 1+s*(1-c)) for LLM priors")

    parser.add_argument("--coupling_lambda", type=float, default=0.0,
                        help="Neighbour update strength (0 disables coupling)")

    parser.add_argument("--mode", choices=["simulate", "real"], default="simulate")

    # simulation knobs
    parser.add_argument("--sim_base_rate", type=float, default=0.55)
    parser.add_argument("--sim_signal", type=float, default=0.50,
                        help="How much LLM confidence correlates with success prob (0 = no advantage)")
    parser.add_argument("--sim_noise", type=float, default=0.05)

    parser.add_argument("--log_csv", type=str, default="cbo_log.csv")
    args = parser.parse_args()

    pathways = load_prior(args.prior_json, args.phenotype)
    adjacency = load_adjacency(args.adjacency_json)
    posteriors = init_posteriors(
        pathways,
        prior_strength=args.prior_strength,
        baseline_uniform=args.baseline_uniform,
    )

    gt_probs = None
    if args.mode == "simulate":
        rng = random.Random(args.seed)
        gt_probs = build_synthetic_ground_truth(
            pathways,
            rng=rng,
            signal_from_confidence=args.sim_signal,
            base_rate=args.sim_base_rate,
            noise=args.sim_noise,
        )
        # print gt top for sanity
        gt_ranked = sorted(gt_probs.items(), key=lambda kv: kv[1], reverse=True)
        print("Synthetic ground truth (top 5 success probabilities):")
        id_to_name = {p.pathway_id: p.name for p in pathways}
        for pid, pr in gt_ranked[:5]:
            print(f"  {pid:>10s}  p={pr:.3f}  name={id_to_name.get(pid,'')}")

    run(
        pathways,
        posteriors,
        adjacency,
        iters=args.iters,
        seed=args.seed,
        mode=args.mode,
        coupling_lambda=args.coupling_lambda,
        log_csv_path=args.log_csv,
        gt_probs=gt_probs,
    )


if __name__ == "__main__":
    main()
