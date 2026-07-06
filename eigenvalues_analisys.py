# Imports
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.rnn_utils import forward_rnn
from utils.rnn_utils import rnn_params
from utils.rnn_utils import std_noise_func


# Parameters that can be tuned from the terminal
parser = argparse.ArgumentParser()

parser.add_argument("--trials_noise", type=int, default=10, help="Number of independent pairs of noise realizations.")
parser.add_argument("--trials_esn", type=int, default=10, help="Number of different ESN realizations.")
parser.add_argument("--noise_max", type=float, default=105.0, help="Upper limit of the noise scan in percent; this value is not included.")
parser.add_argument("--noise_step", type=float, default=5.0, help="Step of the noise scan in percent.")
parser.add_argument("--time_len", type=int, default=3000, help="Number of input samples used to generate the reservoir states.")
parser.add_argument("--N", type=int, default=200, help="Number of reservoir neurons.")
parser.add_argument("--spectral_radius", type=float, default=1.6, help="Reservoir spectral radius.")
parser.add_argument("--scaling", type=float, default=0.9, help="Input scaling.")
parser.add_argument("--seed", type=int, default=20, help="Base random seed.")
parser.add_argument("--magnitude_mode", type=str, choices=["sum", "mean"], default="sum", help="Magnitude computed as the sum or mean of the eigenvalue magnitudes in each group.")
parser.add_argument("--corr", type=lambda x: str(x).lower() in ["true", "1", "yes", "y"], default=False, help="True for correlated noise and False for uncorrelated noise.")

args = parser.parse_args()


###############################################################################
# ESN parameters
###############################################################################

spectral_radius = args.spectral_radius
scaling = args.scaling
bias_scaling = 0.4
alpha = 0.75
sparsity = None

N = args.N
time_len = args.time_len
corr = args.corr
magnitude_mode = args.magnitude_mode

trials_noise = args.trials_noise
trials_esn = args.trials_esn
trials = trials_noise * trials_esn

# CTC is fixed to two independent noisy realizations.
m = 2

if args.noise_step <= 0:
    raise ValueError("'noise_step' must be larger than zero.")

if args.noise_max <= 0:
    raise ValueError("'noise_max' must be larger than zero.")

if trials_noise < 1 or trials_esn < 1:
    raise ValueError("'trials_noise' and 'trials_esn' must be at least 1.")


###############################################################################
# Input signal: Rössler x
###############################################################################

data1 = pd.read_csv("Rossler_data/xRossler.txt", sep="\t", header=None, index_col=None)
data1 = data1.values[:time_len]
data1 = data1.reshape(-1, 1)

ut_train1 = data1[:-1]

input_size = ut_train1.shape[-1]
output_size = 1


###############################################################################
# Noise scan and storage
###############################################################################

noise_values = np.arange(0.0, args.noise_max, args.noise_step, dtype=float)

np.random.seed(args.seed)

# Independent realizations of the internal reservoir weights.
seed_esn = np.random.randint(0, 2_000_000, size=trials_esn)

# Independent noise realizations for every noise level, reservoir realization
# and noise trial. The final dimension contains the two independent noisy runs
# required to compute CTC with m = 2.
seed_pairs = np.random.randint(0, 2_000_000, size=(len(noise_values), trials_esn, trials_noise, m))

# One row per noise level and one column per ESN/noise-pair trial.
eigenvalue_kept = np.empty((len(noise_values), trials), dtype=float)
eigenvalue_removed = np.empty((len(noise_values), trials), dtype=float)


###############################################################################
# Eigenvalue magnitude analysis for CTC with m = 2
###############################################################################

for noise_idx, noise_level in enumerate(noise_values):
    for esn_idx in range(trials_esn):
        params = rnn_params(N, input_size, output_size, scaling, spectral_radius, alpha, bias_scaling, sparsity, seed=seed_esn[esn_idx])

        # Clean states are used to define the noise standard deviation.
        X_id = forward_rnn(params, ut_train1, 42, x_init=None, autonomous=False, conceptor=None)
        std_noise = std_noise_func(X_id, noise_level)

        for trial_idx in range(trials_noise):
            trial_index = esn_idx * trials_noise + trial_idx

            # Two independent noisy realizations, as required for CTC with m = 2.
            X_noi1 = forward_rnn(params, ut_train1, seed_pairs[noise_idx, esn_idx, trial_idx, 0], None, False, None, std_noise, corr=corr)
            X_noi2 = forward_rnn(params, ut_train1, seed_pairs[noise_idx, esn_idx, trial_idx, 1], None, False, None, std_noise, corr=corr)

            # Cross-trial correlation matrix.
            R_cross = X_noi1.T @ X_noi2 / X_noi1.shape[0]

            # The same symmetrization used before the PSD projection in CTC.
            R_symmetric = 0.5 * (R_cross + R_cross.T)

            # Eigenvalues before removing the negative part.
            eigenvalues = np.linalg.eigvalsh(R_symmetric)

            kept_values = eigenvalues[eigenvalues >= 0.0]
            removed_values = eigenvalues[eigenvalues < 0.0]

            if magnitude_mode == "sum":
                eigenvalue_kept[noise_idx, trial_index] = np.sum(np.abs(kept_values))
                eigenvalue_removed[noise_idx, trial_index] = np.sum(np.abs(removed_values))
            else:
                eigenvalue_kept[noise_idx, trial_index] = np.mean(np.abs(kept_values)) if kept_values.size > 0 else 0.0
                eigenvalue_removed[noise_idx, trial_index] = np.mean(np.abs(removed_values)) if removed_values.size > 0 else 0.0


###############################################################################
# Mean and standard deviation across reservoir-weight and noise realizations
###############################################################################

# Every column corresponds to one combination:
# reservoir realization x noise realization.
mean_kept = np.mean(eigenvalue_kept, axis=1)
std_kept = np.std(eigenvalue_kept, axis=1)

mean_removed = np.mean(eigenvalue_removed, axis=1)
std_removed = np.std(eigenvalue_removed, axis=1)


###############################################################################
# Save numerical results
###############################################################################

Path("plots").mkdir(parents=True, exist_ok=True)

results = pd.DataFrame({"noise_percent": noise_values, "mean_kept_magnitude": mean_kept, "std_kept_magnitude": std_kept, "mean_removed_magnitude": mean_removed, "std_removed_magnitude": std_removed})

c = "correlated" if corr else "uncorrelated"
results.to_csv(f"plots/CTC_eigenvalue_magnitude_m2_N{N}_trials{trials}_noisestep{args.noise_step:g}_maxnoise{args.noise_max:g}_{magnitude_mode}_{c}.csv", index=False)


###############################################################################
# Plot style
###############################################################################

plt.rcParams.update({"figure.figsize": (8, 4), "axes.labelsize": 20, "axes.titlesize": 20, "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 18, "lines.linewidth": 2, "lines.markersize": 9, "grid.alpha": 0.6, "grid.linestyle": "--", "axes.linewidth": 1.8, "axes.edgecolor": "black"})


###############################################################################
# Plot: kept and removed eigenvalue magnitudes
###############################################################################

plt.figure(figsize=(8, 4), dpi=300)

plt.errorbar(noise_values, mean_kept, yerr=std_kept, fmt="o", color="#009E73", alpha=0.85, ecolor="black", elinewidth=2, capsize=6, label="Kept eigenvalues")
plt.errorbar(noise_values, mean_removed, yerr=std_removed, fmt="s", color="#8E44AD", alpha=0.85, ecolor="black", elinewidth=2, capsize=6, label="Removed eigenvalues")

plt.xlabel("% Noise")
plt.ylabel("Eigenvalue magnitude")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

base_name = f"plots/CTC_eigenvalue_magnitude_m2_N{N}_trials{trials}_noisestep{args.noise_step:g}_maxnoise{args.noise_max:g}_{magnitude_mode}_{c}"

plt.savefig(f"{base_name}.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{base_name}.pdf", dpi=300, bbox_inches="tight")

plt.show()
plt.close()
