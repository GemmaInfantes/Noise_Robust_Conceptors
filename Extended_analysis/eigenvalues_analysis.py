# Imports
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


###############################################################################
# Project paths
###############################################################################

# Folder containing this script: Extended_analysis
SCRIPT_DIR = Path(__file__).resolve().parent

# Main project folder: Noise_Robust_Conceptors
PROJECT_DIR = SCRIPT_DIR.parent

# Input-data folder: Noise_Robust_Conceptors/Rossler_data
DATA_DIR = PROJECT_DIR / "Rossler_data"

# Output folder: Noise_Robust_Conceptors/Extended_analysis/plots_analysis
PLOTS_DIR = SCRIPT_DIR / "plots_analysis"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Allow imports from the sibling folder utils
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


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

data1 = pd.read_csv(DATA_DIR / "xRossler.txt", sep="\t", header=None, index_col=None)
data1 = data1.values[:time_len].reshape(-1, 1)

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
# and noise trial. The last dimension contains the two noisy runs used for CTC.
seed_pairs = np.random.randint(0, 2_000_000, size=(len(noise_values), trials_esn, trials_noise, m))

# Magnitude: sum of the absolute eigenvalues in each group.
magnitude_kept = np.empty((len(noise_values), trials), dtype=float)
magnitude_removed = np.empty((len(noise_values), trials), dtype=float)
magnitude_total = np.empty((len(noise_values), trials), dtype=float)

# Number of eigenvalues in each group.
count_kept = np.empty((len(noise_values), trials), dtype=float)
count_removed = np.empty((len(noise_values), trials), dtype=float)
count_total = np.empty((len(noise_values), trials), dtype=float)

# Magnitude divided by the number of eigenvalues in each group.
magnitude_per_kept = np.empty((len(noise_values), trials), dtype=float)
magnitude_per_removed = np.empty((len(noise_values), trials), dtype=float)
magnitude_per_total = np.empty((len(noise_values), trials), dtype=float)


###############################################################################
# Eigenvalue analysis for CTC with m = 2
###############################################################################

for noise_idx, noise_level in enumerate(noise_values):
    for esn_idx in range(trials_esn):
        params = rnn_params(N, input_size, output_size, scaling, spectral_radius, alpha, bias_scaling, sparsity, seed=seed_esn[esn_idx])

        # Clean states are used to define the noise standard deviation.
        X_id = forward_rnn(params, ut_train1, 42, x_init=None, autonomous=False, conceptor=None)
        std_noise = std_noise_func(X_id, noise_level)

        for trial_idx in range(trials_noise):
            trial_index = esn_idx * trials_noise + trial_idx

            # Two independent noisy realizations for CTC with m = 2.
            X_noi1 = forward_rnn(params, ut_train1, seed_pairs[noise_idx, esn_idx, trial_idx, 0], None, False, None, std_noise, corr=corr)
            X_noi2 = forward_rnn(params, ut_train1, seed_pairs[noise_idx, esn_idx, trial_idx, 1], None, False, None, std_noise, corr=corr)

            # Cross-trial correlation matrix.
            R_cross = X_noi1.T @ X_noi2 / X_noi1.shape[0]

            # Symmetrization before the positive-semidefinite projection.
            R_symmetric = 0.5 * (R_cross + R_cross.T)

            # Eigenvalues before setting the negative values to zero.
            eigenvalues = np.linalg.eigvalsh(R_symmetric)

            kept_values = eigenvalues[eigenvalues >= 0.0]
            removed_values = eigenvalues[eigenvalues < 0.0]

            ###################################################################
            # 1. Sum of absolute eigenvalue magnitudes
            ###################################################################

            kept_magnitude = np.sum(np.abs(kept_values))
            removed_magnitude = np.sum(np.abs(removed_values))
            total_magnitude = kept_magnitude + removed_magnitude

            magnitude_kept[noise_idx, trial_index] = kept_magnitude
            magnitude_removed[noise_idx, trial_index] = removed_magnitude
            magnitude_total[noise_idx, trial_index] = total_magnitude

            ###################################################################
            # 2. Number of kept, removed and total eigenvalues
            ###################################################################

            kept_number = kept_values.size
            removed_number = removed_values.size
            total_number = eigenvalues.size

            count_kept[noise_idx, trial_index] = kept_number
            count_removed[noise_idx, trial_index] = removed_number
            count_total[noise_idx, trial_index] = total_number

            ###################################################################
            # 3. Magnitude per eigenvalue
            ###################################################################

            # A value of zero is assigned when a group contains no eigenvalues.
            magnitude_per_kept[noise_idx, trial_index] = kept_magnitude / kept_number if kept_number > 0 else 0.0
            magnitude_per_removed[noise_idx, trial_index] = removed_magnitude / removed_number if removed_number > 0 else 0.0
            magnitude_per_total[noise_idx, trial_index] = total_magnitude / total_number


###############################################################################
# Means and standard deviations
###############################################################################

def mean_and_std(values):
    return np.mean(values, axis=1), np.std(values, axis=1)


mean_magnitude_kept, std_magnitude_kept = mean_and_std(magnitude_kept)
mean_magnitude_removed, std_magnitude_removed = mean_and_std(magnitude_removed)
mean_magnitude_total, std_magnitude_total = mean_and_std(magnitude_total)

mean_count_kept, std_count_kept = mean_and_std(count_kept)
mean_count_removed, std_count_removed = mean_and_std(count_removed)
mean_count_total, std_count_total = mean_and_std(count_total)

mean_magnitude_per_kept, std_magnitude_per_kept = mean_and_std(magnitude_per_kept)
mean_magnitude_per_removed, std_magnitude_per_removed = mean_and_std(magnitude_per_removed)
mean_magnitude_per_total, std_magnitude_per_total = mean_and_std(magnitude_per_total)


###############################################################################
# Save numerical results
###############################################################################

results = pd.DataFrame({
    "noise_percent": noise_values,
    "mean_kept_magnitude": mean_magnitude_kept,
    "std_kept_magnitude": std_magnitude_kept,
    "mean_removed_magnitude": mean_magnitude_removed,
    "std_removed_magnitude": std_magnitude_removed,
    "mean_total_magnitude": mean_magnitude_total,
    "std_total_magnitude": std_magnitude_total,
    "mean_kept_count": mean_count_kept,
    "std_kept_count": std_count_kept,
    "mean_removed_count": mean_count_removed,
    "std_removed_count": std_count_removed,
    "mean_total_count": mean_count_total,
    "std_total_count": std_count_total,
    "mean_magnitude_per_kept": mean_magnitude_per_kept,
    "std_magnitude_per_kept": std_magnitude_per_kept,
    "mean_magnitude_per_removed": mean_magnitude_per_removed,
    "std_magnitude_per_removed": std_magnitude_per_removed,
    "mean_magnitude_per_total": mean_magnitude_per_total,
    "std_magnitude_per_total": std_magnitude_per_total,
})

c = "correlated" if corr else "uncorrelated"
base_tag = f"m2_N{N}_trials{trials}_noisestep{args.noise_step:g}_maxnoise{args.noise_max:g}_{c}"
results.to_csv(PLOTS_DIR / f"CTC_eigenvalue_complete_analysis_{base_tag}.csv", index=False)


###############################################################################
# Plot style
###############################################################################

plt.rcParams.update({
    "figure.figsize": (8, 4),
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 15,
    "lines.linewidth": 2,
    "lines.markersize": 8,
    "grid.alpha": 0.6,
    "grid.linestyle": "--",
    "axes.linewidth": 1.8,
    "axes.edgecolor": "black",
})


###############################################################################
# Figure 1: Sum of absolute eigenvalue magnitudes
###############################################################################

plt.figure(figsize=(8, 4), dpi=300)

plt.errorbar(noise_values, mean_magnitude_kept, yerr=std_magnitude_kept, fmt="-o", color="#009E73", alpha=0.85, ecolor="black", elinewidth=1.6, capsize=5, label="Kept")
plt.errorbar(noise_values, mean_magnitude_removed, yerr=std_magnitude_removed, fmt="-s", color="#8E44AD", alpha=0.85, ecolor="black", elinewidth=1.6, capsize=5, label="Removed")
plt.errorbar(noise_values, mean_magnitude_total, yerr=std_magnitude_total, fmt="-^", color="#E67E22", alpha=0.9, ecolor="black", elinewidth=1.6, capsize=5, label="Total")

plt.xlabel("% Noise")
plt.ylabel("Eigenvalues magnitude",size=16.4)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

magnitude_name = f"CTC_eigenvalue_magnitude_sum_{base_tag}"
plt.savefig(PLOTS_DIR / f"{magnitude_name}.png", dpi=300, bbox_inches="tight")
plt.savefig(PLOTS_DIR / f"{magnitude_name}.pdf", dpi=300, bbox_inches="tight")

plt.show()
plt.close()


###############################################################################
# Figure 2: Number of kept and removed eigenvalues
###############################################################################

plt.figure(figsize=(8, 4), dpi=300)

plt.errorbar(noise_values, mean_count_kept, yerr=std_count_kept, fmt="-o", color="#009E73", alpha=0.85, ecolor="black", elinewidth=1.6, capsize=5, label="Kept")
plt.errorbar(noise_values, mean_count_removed, yerr=std_count_removed, fmt="-s", color="#8E44AD", alpha=0.85, ecolor="black", elinewidth=1.6, capsize=5, label="Removed")
plt.errorbar(noise_values, mean_count_total, yerr=std_count_total, fmt="-^", color="#E67E22", alpha=0.9, ecolor="black", elinewidth=1.6, capsize=5, label="Total")

plt.xlabel("% Noise")
plt.ylabel("Number of eigenvalues")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

count_name = f"CTC_eigenvalue_count_{base_tag}"
plt.savefig(PLOTS_DIR / f"{count_name}.png", dpi=300, bbox_inches="tight")
plt.savefig(PLOTS_DIR / f"{count_name}.pdf", dpi=300, bbox_inches="tight")

plt.show()
plt.close()


###############################################################################
# Figure 3: Absolute magnitude per eigenvalue
###############################################################################

plt.figure(figsize=(8, 4), dpi=300)

plt.errorbar(noise_values, mean_magnitude_per_kept, yerr=std_magnitude_per_kept, fmt="-o", color="#009E73", alpha=0.85, ecolor="black", elinewidth=1.6, capsize=5, label="Kept")
plt.errorbar(noise_values, mean_magnitude_per_removed, yerr=std_magnitude_per_removed, fmt="-s", color="#8E44AD", alpha=0.85, ecolor="black", elinewidth=1.6, capsize=5, label="Removed")
plt.errorbar(noise_values, mean_magnitude_per_total, yerr=std_magnitude_per_total, fmt="-^", color="#E67E22", alpha=0.9, ecolor="black", elinewidth=1.6, capsize=5, label="All")

plt.xlabel("% Noise")
plt.ylabel("Mean absolute eigenvalue", size=17)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

average_name = f"CTC_eigenvalue_mean_magnitude_{base_tag}"
plt.savefig(PLOTS_DIR / f"{average_name}.png", dpi=300, bbox_inches="tight")
plt.savefig(PLOTS_DIR / f"{average_name}.pdf", dpi=300, bbox_inches="tight")

plt.show()
plt.close()
