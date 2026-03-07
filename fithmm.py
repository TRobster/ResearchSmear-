#!/usr/bin/env python3
"""
Gaussian HMM Batch Pipeline — dynamax + JAX on A100 GPU
========================================================
Fits a 2D Gaussian HMM to (nose_x, nose_y) across all sessions in:
  <base_dir>/{7010,7011,7012,7013}/m{1..20}/events.csv

Designed for Talapas HPC (University of Oregon) with A100 GPUs.

Usage:
  python fit_hmm.py --data-dir /path/to/events --output-dir /path/to/results
  python fit_hmm.py --data-dir /path/to/events --n-states 4 6 8 --n-restarts 10
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── JAX configuration (MUST come before importing jax) ──────────────
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")

# ── Monkey-patch numpy for tensorflow-probability compatibility ─────
# tfp 0.25 uses np.reshape(..., newshape=...) which numpy 2.x renamed
# to just "shape". This patch avoids touching any installed packages.
import numpy as _np
_original_reshape = _np.reshape
def _patched_reshape(a, *args, **kwargs):
    if 'newshape' in kwargs:
        kwargs['shape'] = kwargs.pop('newshape')
    return _original_reshape(a, *args, **kwargs)
_np.reshape = _patched_reshape
# ────────────────────────────────────────────────────────────────────

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pandas as pd

# Verify GPU before heavy imports
print(f"JAX {jax.__version__} | backend: {jax.default_backend()}")
print(f"Devices: {jax.devices()}")
if jax.default_backend() != "gpu":
    print("WARNING: No GPU detected — falling back to CPU. This will be slow.")

from dynamax.hidden_markov_model import GaussianHMM


# ── Data Loading ────────────────────────────────────────────────────

ANIMAL_IDS = [7010, 7011, 7012, 7013]
SESSIONS   = [f"m{i}" for i in range(1, 21)]
OBS_COLS   = ["nose_x", "nose_y"]


def load_single_session(csv_path: Path) -> np.ndarray | None:
    """Load and validate a single events.csv, returning (T, 2) float32 array."""
    try:
        df = pd.read_csv(csv_path, usecols=OBS_COLS)
    except Exception as e:
        print(f"  SKIP {csv_path}: {e}")
        return None

    data = df[OBS_COLS].values.astype(np.float32)

    # Drop rows with NaN/Inf (tracking dropouts)
    mask = np.isfinite(data).all(axis=1)
    n_bad = (~mask).sum()
    if n_bad > 0:
        print(f"  {csv_path.name}: dropped {n_bad}/{len(data)} bad frames")
        data = data[mask]

    if len(data) < 100:
        print(f"  SKIP {csv_path}: only {len(data)} valid frames")
        return None

    return data


def load_all_sessions(base_dir: Path) -> dict:
    """
    Load all sessions from the directory tree.

    Returns
    -------
    sessions : dict
        {(animal_id, session_name): np.ndarray of shape (T, 2)}
    """
    sessions = {}
    for aid in ANIMAL_IDS:
        for sess in SESSIONS:
            csv_path = base_dir / str(aid) / sess / "events.csv"
            if not csv_path.exists():
                print(f"  MISSING {csv_path}")
                continue
            data = load_single_session(csv_path)
            if data is not None:
                sessions[(aid, sess)] = data
                print(f"  Loaded {aid}/{sess}: {data.shape[0]:,} frames")
    return sessions


# ── Preprocessing ───────────────────────────────────────────────────

def zscore_per_session(sessions: dict) -> tuple[dict, dict]:
    """
    Z-score each session independently to remove camera/arena offsets.

    Returns (z-scored sessions dict, stats dict for inverse transform).
    """
    z_sessions = {}
    stats = {}
    for key, data in sessions.items():
        mu = data.mean(axis=0)
        sigma = data.std(axis=0)
        sigma = np.where(sigma < 1e-8, 1.0, sigma)  # guard zero-variance
        z_sessions[key] = ((data - mu) / sigma).astype(np.float32)
        stats[key] = {"mean": mu.tolist(), "std": sigma.tolist()}
    return z_sessions, stats


def pad_to_batch(sessions: dict) -> tuple[jnp.ndarray, list, np.ndarray]:
    """
    Pad all sessions to equal length and stack into a (B, T_max, 2) array.

    Padding with zeros is fine — dynamax supports masking via sequence lengths,
    or we can ignore padded timesteps during analysis.

    Returns
    -------
    batch : jnp.ndarray, shape (B, T_max, D)
    keys  : list of (animal_id, session) tuples in batch order
    lengths : np.ndarray of actual sequence lengths, shape (B,)
    """
    keys = list(sessions.keys())
    arrays = [sessions[k] for k in keys]
    lengths = np.array([len(a) for a in arrays])
    T_max = int(lengths.max())
    D = arrays[0].shape[1]

    # Pad each session to T_max
    padded = np.zeros((len(arrays), T_max, D), dtype=np.float32)
    for i, arr in enumerate(arrays):
        padded[i, :len(arr)] = arr

    print(f"\nBatch shape: {padded.shape}  (B={len(keys)}, T_max={T_max:,}, D={D})")
    print(f"Sequence lengths: min={lengths.min():,}, max={lengths.max():,}, "
          f"mean={lengths.mean():,.0f}")
    mem_gb = padded.nbytes / 1e9
    print(f"Batch memory: {mem_gb:.3f} GB")

    return jnp.array(padded), keys, lengths


# ── Model Fitting ───────────────────────────────────────────────────

def fit_single_model(
    batch: jnp.ndarray,
    K: int,
    seed: int,
    n_iters: int = 200,
    tol: float = 1e-4,
) -> tuple:
    """
    Fit a K-state Gaussian HMM with full covariance using EM.

    Returns (params, log_likelihoods_per_iter).
    """
    key = jr.PRNGKey(seed)
    model = GaussianHMM(
        num_states=K,
        emission_dim=batch.shape[-1],
    )

    # Initialize with k-means for stable starting point
    params, props = model.initialize(
        key=key,
        method="kmeans",
        emissions=batch,
    )

    # Fit with EM
    params, log_lls = model.fit_em(
        params, props,
        emissions=batch,
        num_iters=n_iters,
    )

    return model, params, log_lls


def compute_bic(log_ll: float, K: int, D: int, N_total: int) -> float:
    """
    BIC = -2 * LL + n_params * log(N_total)

    Parameters for K-state, D-dim full-covariance Gaussian HMM:
      - Initial probs:  K - 1
      - Transitions:    K * (K - 1)
      - Means:          K * D
      - Covariances:    K * D * (D + 1) / 2
    """
    n_params = (
        (K - 1)                      # initial state probs
        + K * (K - 1)                # transition matrix
        + K * D                      # emission means
        + K * D * (D + 1) // 2       # emission covariances (symmetric)
    )
    return -2 * log_ll + n_params * np.log(N_total)


def fit_with_restarts(
    batch: jnp.ndarray,
    K: int,
    n_restarts: int,
    n_iters: int,
    lengths: np.ndarray,
) -> dict:
    """
    Fit K-state model with multiple random restarts, return best by final LL.
    """
    print(f"\n{'='*60}")
    print(f"Fitting K={K} states  |  {n_restarts} restarts  |  {n_iters} EM iters")
    print(f"{'='*60}")

    best_ll = -np.inf
    best_result = None
    all_final_lls = []

    for r in range(n_restarts):
        seed = K * 1000 + r  # deterministic but unique per (K, restart)
        t0 = time.time()

        try:
            model, params, log_lls = fit_single_model(batch, K, seed, n_iters)
            final_ll = float(log_lls[-1])
            elapsed = time.time() - t0
            all_final_lls.append(final_ll)

            # Check convergence
            if len(log_lls) >= 2:
                delta = float(log_lls[-1] - log_lls[-2])
            else:
                delta = float("inf")

            status = "CONVERGED" if abs(delta) < 1e-4 else f"delta={delta:.2e}"
            print(f"  restart {r+1:2d}/{n_restarts}: "
                  f"LL={final_ll:+.2f}  {status}  ({elapsed:.1f}s)")

            if final_ll > best_ll:
                best_ll = final_ll
                best_result = (model, params, log_lls)

        except Exception as e:
            print(f"  restart {r+1:2d}/{n_restarts}: FAILED — {e}")
            all_final_lls.append(float("nan"))

    if best_result is None:
        return {"K": K, "error": "all restarts failed"}

    model, params, log_lls = best_result
    N_total = int(lengths.sum())  # total observed timesteps (not padding)
    D = batch.shape[-1]
    bic = compute_bic(best_ll, K, D, N_total)

    print(f"\n  Best LL: {best_ll:+.2f}  |  BIC: {bic:.2f}")
    print(f"  LL spread across restarts: {np.nanstd(all_final_lls):.2f}")

    return {
        "K": K,
        "best_ll": best_ll,
        "bic": bic,
        "all_final_lls": all_final_lls,
        "model": model,
        "params": params,
        "log_lls_history": [float(x) for x in log_lls],
    }


# ── Decoding ────────────────────────────────────────────────────────

def decode_states(model, params, batch: jnp.ndarray) -> np.ndarray:
    """Viterbi decode each session, return (B, T_max) state assignments."""
    # vmap Viterbi across batch dimension
    viterbi_fn = jax.vmap(lambda e: model.most_likely_states(params, e))
    states = viterbi_fn(batch)
    return np.array(states)


# ── Output ──────────────────────────────────────────────────────────

def save_results(
    output_dir: Path,
    best_result: dict,
    all_results: list[dict],
    batch: jnp.ndarray,
    keys: list,
    lengths: np.ndarray,
    z_stats: dict,
):
    """Save decoded states, parameters, and model selection summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    K = best_result["K"]
    model = best_result["model"]
    params = best_result["params"]

    # ── 1. Decode states for best model ──
    print(f"\nDecoding states with best model (K={K})...")
    states = decode_states(model, params, batch)

    # Save per-session state sequences (unpadded)
    states_dir = output_dir / f"decoded_states_K{K}"
    states_dir.mkdir(exist_ok=True)
    for i, key in enumerate(keys):
        aid, sess = key
        T = lengths[i]
        seq = states[i, :T]
        out_path = states_dir / f"{aid}_{sess}_states.csv"
        pd.DataFrame({"state": seq}).to_csv(out_path, index=False)

    print(f"  Saved {len(keys)} state sequence files → {states_dir}/")

    # ── 2. Save model parameters ──
    params_out = {
        "K": K,
        "emission_dim": 2,
        "obs_columns": OBS_COLS,
    }

    # Extract numpy arrays from params
    # dynamax params structure: params.emissions.means, params.emissions.covs, etc.
    try:
        params_out["initial_probs"] = np.array(
            jax.nn.softmax(params.initial.probs)
        ).tolist()
        params_out["transition_matrix"] = np.array(
            jax.nn.softmax(params.transitions.transition_matrix, axis=-1)
        ).tolist()
        params_out["emission_means"] = np.array(
            params.emissions.means
        ).tolist()
        params_out["emission_covariances"] = np.array(
            params.emissions.covs
        ).tolist()
    except AttributeError:
        # Fallback: save raw param tree as nested dict
        params_out["raw_params"] = jax.tree.map(
            lambda x: np.array(x).tolist(), params
        )

    with open(output_dir / f"params_K{K}.json", "w") as f:
        json.dump(params_out, f, indent=2)

    # ── 3. Save z-score stats for inverse transform ──
    z_stats_serializable = {
        f"{k[0]}_{k[1]}": v for k, v in z_stats.items()
    }
    with open(output_dir / "zscore_stats.json", "w") as f:
        json.dump(z_stats_serializable, f, indent=2)

    # ── 4. Model selection summary ──
    summary = []
    for res in all_results:
        if "error" in res:
            summary.append({"K": res["K"], "error": res["error"]})
        else:
            summary.append({
                "K": res["K"],
                "best_ll": res["best_ll"],
                "bic": res["bic"],
                "ll_std_across_restarts": float(np.nanstd(res["all_final_lls"])),
                "n_converged": sum(
                    1 for ll in res["all_final_lls"] if np.isfinite(ll)
                ),
            })

    with open(output_dir / "model_selection.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Model selection summary:")
    print(f"  {'K':>3}  {'Best LL':>14}  {'BIC':>14}  {'LL σ':>10}")
    print(f"  {'─'*3}  {'─'*14}  {'─'*14}  {'─'*10}")
    for s in summary:
        if "error" not in s:
            print(f"  {s['K']:3d}  {s['best_ll']:+14.2f}  {s['bic']:14.2f}  "
                  f"{s['ll_std_across_restarts']:10.2f}")

    # ── 5. Save EM convergence curves ──
    for res in all_results:
        if "error" not in res:
            np.save(
                output_dir / f"em_curve_K{res['K']}.npy",
                np.array(res["log_lls_history"]),
            )

    print(f"\nAll results saved → {output_dir}/")


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fit Gaussian HMM to nose tracking data on GPU"
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Base directory containing {7010..7013}/m{1..20}/events.csv"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory for output files"
    )
    parser.add_argument(
        "--n-states", type=int, nargs="+", default=[3, 4, 5, 6, 7, 8],
        help="K values to sweep (default: 3 4 5 6 7 8)"
    )
    parser.add_argument(
        "--n-restarts", type=int, default=10,
        help="Random restarts per K (default: 10)"
    )
    parser.add_argument(
        "--n-em-iters", type=int, default=200,
        help="Max EM iterations per fit (default: 200)"
    )
    parser.add_argument(
        "--downsample", type=int, default=1,
        help="Downsample factor (default: 1 = no downsampling)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Gaussian HMM Batch Pipeline — dynamax + JAX")
    print("=" * 60)
    print(f"Data dir:    {args.data_dir}")
    print(f"Output dir:  {args.output_dir}")
    print(f"K values:    {args.n_states}")
    print(f"Restarts:    {args.n_restarts}")
    print(f"EM iters:    {args.n_em_iters}")
    print(f"Downsample:  {args.downsample}x")
    print()

    # ── Load ──
    print("Loading sessions...")
    sessions = load_all_sessions(args.data_dir)
    if not sessions:
        print("ERROR: No sessions loaded. Check --data-dir path.")
        sys.exit(1)
    print(f"\nLoaded {len(sessions)} sessions across "
          f"{len(set(k[0] for k in sessions))} animals")

    # ── Downsample ──
    if args.downsample > 1:
        print(f"\nDownsampling {args.downsample}x...")
        sessions = {
            k: v[::args.downsample] for k, v in sessions.items()
        }
        print(f"  New frame counts: "
              f"{min(len(v) for v in sessions.values()):,}–"
              f"{max(len(v) for v in sessions.values()):,}")

    # ── Z-score ──
    print("\nZ-scoring per session...")
    z_sessions, z_stats = zscore_per_session(sessions)

    # ── Batch ──
    batch, keys, lengths = pad_to_batch(z_sessions)

    # ── Fit across K values ──
    all_results = []
    for K in args.n_states:
        result = fit_with_restarts(
            batch, K, args.n_restarts, args.n_em_iters, lengths
        )
        all_results.append(result)

    # ── Select best K by BIC ──
    valid = [r for r in all_results if "error" not in r]
    if not valid:
        print("\nERROR: All models failed.")
        sys.exit(1)

    best = min(valid, key=lambda r: r["bic"])
    print(f"\n{'='*60}")
    print(f"BEST MODEL: K={best['K']}  (BIC={best['bic']:.2f})")
    print(f"{'='*60}")

    # ── Save ──
    save_results(
        args.output_dir, best, all_results,
        batch, keys, lengths, z_stats
    )


if __name__ == "__main__":
    main()
