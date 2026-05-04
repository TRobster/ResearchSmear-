"""
mouse_hmm.py  –  Gaussian HMM on nose-tracking data
Usage:  python mouse_hmm.py events.csv [--states 3] [--downsample 6]
"""

import sys
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# 1.  Load 

def load_nose(path):
    """Pull nose x/y out of the events CSV and drop any NaN rows."""
    df = pd.read_csv(path)[["nose_x", "nose_y"]].dropna().reset_index(drop=True)
    return df["nose_x"].values, df["nose_y"].values


# 2.  Kinematics from Data

def kinematics(x, y, dt):
    """
    Compute three translation-invariant features from the nose position vector.

    r(t) = [x(t), y(t)]

    We never use raw coordinates as observations because they depend on where
    the mouse happens to be standing – the HMM would end up encoding arena
    position rather than behavioral mode.

    Features returned (one row per frame):
      speed  = |v(t)|   = magnitude of the first derivative of r(t)
      omega  = dθ/dt    = angular velocity of the heading direction
      accel  = |a(t)|   = magnitude of the second derivative of r(t)
    """
    # velocity vector v(t) = dr/dt
    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    speed = np.sqrt(vx**2 + vy**2)

    # heading angle θ(t) = atan2(vy, vx), then angular velocity ω = dθ/dt
    theta = np.unwrap(np.arctan2(vy, vx))   # unwrap removes ±2π discontinuities
    omega = np.gradient(theta, dt)

    # acceleration a(t) = dv/dt
    ax = np.gradient(vx, dt)
    ay = np.gradient(vy, dt)
    accel = np.sqrt(ax**2 + ay**2)

    return speed, omega, accel


# 3.  Fit 

def fit_hmm(features, n_states, n_iter=200):
    """
    Fit a Gaussian-emission HMM via Baum-Welch (EM).

    Each hidden state Z_t emits an observation vector o_t = [speed, ω, accel]
    drawn from a multivariate Gaussian N(μ_k, Σ_k).  The transition matrix A
    (what we ultimately want) is also estimated during training.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=0,
    )
    model.fit(X)
    return model, scaler


#4.  Label states by mean speed (makes them interpretable) 

def label_states(model, scaler, n_states):
    """
    Sort the raw model states from slowest to fastest mean speed so we get a
    consistent Resting → Exploring → Locomotion ordering regardless of the
    random initialisation.
    """
    # un-standardise the speed dimension (index 0 after column_stack)
    raw_speed = model.means_[:, 0] * scaler.scale_[0] + scaler.mean_[0]
    order = np.argsort(raw_speed)          # slowest to fastest

    if n_states == 3:
        names = ["Resting", "Exploring", "Locomotion"]
    else:
        names = [f"State {i}" for i in range(n_states)]

    return order, names


# ── 5.  Transition matrix ─────────────────────────────────────────────────────

def reorder_transmat(A_raw, order):
    """
    Permute the transition matrix so rows/cols match our speed-sorted labels.
    A[i,j] = P(next state = j | current state = i)
    """
    return A_raw[np.ix_(order, order)]


def print_transmat(A, names):
    col_w = 14
    header = f"{'':>{col_w}}" + "".join(f"{n:>{col_w}}" for n in names)
    print("\n" + header)
    print("-" * len(header))
    for i, row_name in enumerate(names):
        row = f"{row_name:>{col_w}}" + "".join(f"{A[i,j]:>{col_w}.4f}" for j in range(len(names)))
        print(row)
    print()


# ── 6.  Decode with Viterbi ───────────────────────────────────────────────────

def decode(model, scaler, features, order, n_states):
    """
    Viterbi algorithm: find the single most probable state sequence given the
    observed kinematics.  Then remap model indices to our sorted label indices.
    """
    X = scaler.transform(features)
    raw_states = model.predict(X)

    remap = {original: sorted_idx for sorted_idx, original in enumerate(order)}
    return np.array([remap[s] for s in raw_states])


# ── 7.  Plots ─────────────────────────────────────────────────────────────────

def plot(x, y, states, A, names, n_states, out="hmm_results.png"):
    colors = plt.cm.Set1(np.linspace(0, 0.8, n_states))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Gaussian HMM  –  Nose Tracking", fontsize=13)

    # --- trajectory coloured by state ---
    ax = axes[0]
    for s in range(n_states):
        mask = states == s
        ax.scatter(x[mask], y[mask], c=[colors[s]], s=1, alpha=0.4, label=names[s])
    ax.set_title("Trajectory by State")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.invert_yaxis()
    ax.legend(markerscale=8, fontsize=8)

    # --- transition matrix heatmap ---
    ax = axes[1]
    im = ax.imshow(A, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n_states))
    ax.set_yticks(range(n_states))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticklabels(names)
    ax.set_title("Transition Matrix  A[i→j]")
    ax.set_xlabel("Next state")
    ax.set_ylabel("Current state")
    plt.colorbar(im, ax=ax, fraction=0.046)
    for i in range(n_states):
        for j in range(n_states):
            ax.text(j, i, f"{A[i,j]:.2f}", ha="center", va="center",
                    fontsize=10, color="white" if A[i, j] > 0.6 else "black")

    # --- state sequence over first 3000 frames ---
    ax = axes[2]
    n_show = min(3000, len(states))
    ax.plot(states[:n_show], lw=0.6, color="steelblue")
    ax.set_yticks(range(n_states))
    ax.set_yticklabels(names)
    ax.set_title(f"State Sequence (first {n_show} frames)")
    ax.set_xlabel("Frame")

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Plot saved → {out}")
    plt.show()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fit a Gaussian HMM to mouse nose-tracking data."
    )
    parser.add_argument("csv", help="Path to events.csv")
    parser.add_argument("--states",     type=int,   default=3,    help="Number of hidden states (default: 3)")
    parser.add_argument("--fps",        type=float, default=30.0, help="Camera frame rate in Hz (default: 30)")
    parser.add_argument("--downsample", type=int,   default=6,    help="Take every Nth frame (default: 6)")
    args = parser.parse_args()

    # 1. load
    print(f"\nLoading {args.csv} ...")
    x, y = load_nose(args.csv)
    print(f"  {len(x):,} frames loaded")

    # 2. downsample
    #    At 30 Hz consecutive frames are nearly identical — the HMM ends up
    #    doing spatial binning instead of temporal state discovery.
    #    Skipping every 6th frame → ~5 Hz gives meaningful inter-frame jumps.
    if args.downsample > 1:
        x, y = x[::args.downsample], y[::args.downsample]
        fps = args.fps / args.downsample
        print(f"  Downsampled ×{args.downsample} → {len(x):,} frames @ {fps:.1f} Hz")
    else:
        fps = args.fps

    dt = 1.0 / fps

    # 3. kinematics
    print("Computing kinematics (speed, angular velocity, acceleration) ...")
    speed, omega, accel = kinematics(x, y, dt)
    features = np.column_stack([speed, omega, accel])

    # 4. fit
    print(f"Fitting {args.states}-state Gaussian HMM (Baum-Welch) ...")
    model, scaler = fit_hmm(features, args.states)
    print(f"  Log-likelihood: {model.score(scaler.transform(features)):.2f}")

    # 5. transition matrix
    order, names = label_states(model, scaler, args.states)
    A = reorder_transmat(model.transmat_, order)
    print("\n=== Transition Matrix A  (rows = current state, cols = next state) ===")
    print_transmat(A, names)

    # 6. decode
    print("Decoding state sequence (Viterbi) ...")
    states = decode(model, scaler, features, order, args.states)

    # 7. occupancy summary
    print("State occupancy:")
    for i, name in enumerate(names):
        pct = 100 * np.mean(states == i)
        dur = np.mean([
            sum(1 for _ in g)
            for k, g in __import__("itertools").groupby(states)
            if k == i
        ]) / fps
        print(f"  {name:<14s}  {pct:5.1f}%   mean bout ≈ {dur:.2f} s")

    # 8. plot
    plot(x, y, states, A, names, args.states)


if __name__ == "__main__":
    main()
