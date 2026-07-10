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


from utils.rnn_utils import compute_conceptor
from utils.rnn_utils import denoising_CTC_m
from utils.rnn_utils import forward_rnn
from utils.rnn_utils import ridge
from utils.rnn_utils import rnn_params
from utils.rnn_utils import std_noise_func
from utils.utils import NRMSE
from utils.utils import xcorr_PCA


###############################################################################
# Parameters
###############################################################################

parser = argparse.ArgumentParser()

parser.add_argument("--trials_noise", type=int, default=10, help="Number of independent noise realizations.")
parser.add_argument("--trials_esn", type=int, default=10, help="Number of independent ESN realizations.")
parser.add_argument("--noise", type=float, default=50.0, help="Fixed noise level in percent.")
parser.add_argument("--threshold_max", type=float, default=100.0, help="Maximum threshold as a percentage of the largest singular value or eigenvalue.")
parser.add_argument("--threshold_step", type=float, default=5.0, help="Step of the threshold scan in percent.")
parser.add_argument("--a", type=float, default=5.0, help="Conceptor aperture.")
parser.add_argument("--m", type=int, default=2, help="Number of noisy realizations used to compute CTC.")
parser.add_argument("--p", type=int, default=3, help="p-step-ahead prediction used to train the readout.")
parser.add_argument("--steps", type=int, default=3000, help="Maximum number of states used to compute SSI and NRMSE.")
parser.add_argument("--time_len", type=int, default=3000, help="Number of input samples.")
parser.add_argument("--N", type=int, default=200, help="Number of reservoir neurons.")
parser.add_argument("--spectral_radius", type=float, default=1.6, help="Reservoir spectral radius.")
parser.add_argument("--scaling", type=float, default=0.9, help="Input scaling.")
parser.add_argument("--reg", type=float, default=1.0, help="Ridge-regression regularization.")
parser.add_argument("--washout", type=int, default=20, help="Number of initial states discarded.")
parser.add_argument("--seed", type=int, default=20, help="Base random seed.")
parser.add_argument("--corr", type=lambda x: str(x).lower() in ["true", "1", "yes", "y"], default=False, help="True for correlated noise and False for uncorrelated noise.")

args = parser.parse_args()


###############################################################################
# Validation and ESN parameters
###############################################################################

if args.trials_noise < 1 or args.trials_esn < 1:
    raise ValueError("'trials_noise' and 'trials_esn' must be at least 1.")

if args.threshold_step <= 0:
    raise ValueError("'threshold_step' must be larger than zero.")

if not 0 <= args.threshold_max <= 100:
    raise ValueError("'threshold_max' must be between 0 and 100.")

if args.m < 2:
    raise ValueError("'m' must be at least 2 for the CTC computation.")

if args.p < 1:
    raise ValueError("'p' must be at least 1.")

spectral_radius = args.spectral_radius
scaling = args.scaling
bias_scaling = 0.4
alpha = 0.75
sparsity = None

N = args.N
a = args.a
m = args.m
p = args.p
step = p
reg = args.reg
washout = args.washout
time_len = args.time_len
steps = args.steps
corr = args.corr
noise_level = args.noise

trials_noise = args.trials_noise
trials_esn = args.trials_esn
trials = trials_noise * trials_esn


###############################################################################
# Input signal: Rössler x
###############################################################################

data1 = pd.read_csv(DATA_DIR / "xRossler.txt", sep="\t", header=None, index_col=None)
data1 = data1.values[:time_len].reshape(-1, 1)

ut_train1 = data1[:-p]
yt_train1 = data1[p:]

input_size = ut_train1.shape[-1]
output_size = yt_train1.shape[-1]
yt_train_effective = yt_train1[washout:]

eval_stop = min(washout + steps, len(yt_train1))
effective_steps = eval_stop - washout
y_target = np.asarray(yt_train1[washout:eval_stop]).ravel()


###############################################################################
# Thresholding functions
###############################################################################

def threshold_conceptor_singular_values(C, threshold_fraction):
    """
    Threshold the singular values of an already computed noisy conceptor.

    threshold_fraction = 0.5 means that singular values smaller than
    50% of the largest singular value are set to zero.
    """
    U, singular_values, _ = np.linalg.svd(C, full_matrices=False, hermitian=True)

    if singular_values.size == 0 or singular_values[0] <= 0:
        return np.zeros_like(C)

    absolute_threshold = threshold_fraction * singular_values[0]
    singular_values_thresholded = singular_values.copy()
    singular_values_thresholded[singular_values_thresholded < absolute_threshold] = 0.0

    C_thresholded = U @ np.diag(singular_values_thresholded) @ U.T
    # return 0.5 * (C_thresholded + C_thresholded.T)
    return C_thresholded

def conceptor_from_thresholded_correlation(X, aperture, threshold_fraction):
    """
    Threshold the eigenvalues of the state-correlation matrix and then compute
    the conceptor with the standard matrix expression.

    threshold_fraction = 0.5 means that eigenvalues smaller than
    50% of the largest eigenvalue are set to zero.
    """
    # Original noisy state-correlation matrix.
    R = X.T @ X / X.shape[0]
    R = 0.5 * (R + R.T)

    # Eigenvalue decomposition of the symmetric correlation matrix.
    eigenvalues, eigenvectors = np.linalg.eigh(R)

    # Remove possible small negative values caused by numerical precision.
    eigenvalues = np.maximum(eigenvalues, 0.0)

    largest_eigenvalue = eigenvalues[-1] if eigenvalues.size else 0.0

    if largest_eigenvalue <= 0:
        return np.zeros_like(R)

    # Relative threshold with respect to the largest eigenvalue.
    absolute_threshold = threshold_fraction * largest_eigenvalue

    eigenvalues_thresholded = eigenvalues.copy()
    eigenvalues_thresholded[eigenvalues_thresholded < absolute_threshold] = 0.0

    # Reconstruct the thresholded correlation matrix:
    # R_th = U diag(lambda_th) U.T
    R_thresholded = eigenvectors @ np.diag(eigenvalues_thresholded) @ eigenvectors.T
    R_thresholded = 0.5 * (R_thresholded + R_thresholded.T)

    # Standard conceptor expression:
    # C_th = R_th (R_th + aperture^(-2) I)^(-1)

    C_thresholded = np.dot(R_thresholded, np.linalg.inv(R_thresholded + aperture ** (-2) * np.eye(R_thresholded.shape[0])))
    return C_thresholded 


def evaluate_conceptor(params, conceptor, current_seed, std_noise, X_id):
    """Apply a conceptor, train the readout, and return SSI and NRMSE."""
    X_filtered = forward_rnn(params, ut_train1, current_seed, None, False, conceptor, std_noise, corr=corr)

    params_trained, _ = ridge(reg, X_filtered[washout:], yt_train_effective, step, params)

    ssi = xcorr_PCA(X_id, X_filtered, washout, effective_steps)

    y_prediction = np.asarray(
        X_filtered[washout:eval_stop] @ params_trained["wout"].T
        + params_trained["bias_out"]
    ).ravel()

    nrmse = NRMSE(y_target, y_prediction)

    return ssi, nrmse


###############################################################################
# Threshold scan and storage
###############################################################################

threshold_percentages = np.arange(
    0.0,
    args.threshold_max + 0.5 * args.threshold_step,
    args.threshold_step,
    dtype=float,
)

threshold_fractions = threshold_percentages / 100.0

np.random.seed(args.seed)

# Same seed generation used in the previous noise-scan code:
# first the noise seeds and then the ESN seeds, both in [0, 2000).
seed_noise = np.random.randint(0, 2000, size=trials_noise)
seed_esn = np.random.randint(0, 2000, size=trials_esn)

# Thresholding of singular values of C_noisy.
ssi_singular = np.empty((len(threshold_percentages), trials), dtype=float)
nrmse_singular = np.empty((len(threshold_percentages), trials), dtype=float)

# Thresholding of eigenvalues of R_noisy before computing the conceptor.
ssi_correlation = np.empty((len(threshold_percentages), trials), dtype=float)
nrmse_correlation = np.empty((len(threshold_percentages), trials), dtype=float)

# C_noisy reference: one value per ESN/noise trial.
ssi_c_noisy = np.empty(trials, dtype=float)
nrmse_c_noisy = np.empty(trials, dtype=float)

# CTC reference: one value per ESN/noise trial.
ssi_ctc = np.empty(trials, dtype=float)
nrmse_ctc = np.empty(trials, dtype=float)


###############################################################################
# Simulations at fixed noise
###############################################################################

for esn_idx in range(trials_esn):
    params = rnn_params(
        N,
        input_size,
        output_size,
        scaling,
        spectral_radius,
        alpha,
        bias_scaling,
        sparsity,
        seed=seed_esn[esn_idx],
    )

    # Clean states define the noise amplitude and provide the SSI reference.
    X_id = forward_rnn(
        params,
        ut_train1,
        42,
        x_init=None,
        autonomous=False,
        conceptor=None,
    )

    std_noise = std_noise_func(X_id, noise_level)

    # CTC reference does not depend on the threshold percentage.
    C_ctc = denoising_CTC_m(params, ut_train1, std_noise, a, m, corr=corr)

    for noise_idx in range(trials_noise):
        current_seed = seed_noise[noise_idx]
        trial_index = esn_idx * trials_noise + noise_idx

        # Noisy states used to construct C_noisy and R_noisy.
        X_noisy = forward_rnn(
            params,
            ut_train1,
            current_seed,
            None,
            False,
            None,
            std_noise,
            corr=corr,
        )

        C_noisy = compute_conceptor(X_noisy, a)

        # Evaluate the standard noisy conceptor once for this trial.
        ssi_c_noisy[trial_index], nrmse_c_noisy[trial_index] = evaluate_conceptor(
            params,
            C_noisy,
            current_seed,
            std_noise,
            X_id,
        )

        # Evaluate CTC once for this trial.
        ssi_ctc[trial_index], nrmse_ctc[trial_index] = evaluate_conceptor(
            params,
            C_ctc,
            current_seed,
            std_noise,
            X_id,
        )

        for threshold_idx, threshold_fraction in enumerate(threshold_fractions):
            ###################################################################
            # 1. Threshold singular values of C_noisy
            ###################################################################

            C_singular_thresholded = threshold_conceptor_singular_values(
                C_noisy,
                threshold_fraction,
            )

            (
                ssi_singular[threshold_idx, trial_index],
                nrmse_singular[threshold_idx, trial_index],
            ) = evaluate_conceptor(
                params,
                C_singular_thresholded,
                current_seed,
                std_noise,
                X_id,
            )

            ###################################################################
            # 2. Threshold eigenvalues of R_noisy before computing C
            ###################################################################

            C_correlation_thresholded = conceptor_from_thresholded_correlation(
                X_noisy,
                a,
                threshold_fraction,
            )

            (
                ssi_correlation[threshold_idx, trial_index],
                nrmse_correlation[threshold_idx, trial_index],
            ) = evaluate_conceptor(
                params,
                C_correlation_thresholded,
                current_seed,
                std_noise,
                X_id,
            )


###############################################################################
# Means and standard deviations
###############################################################################

def mean_and_std(values):
    return np.mean(values, axis=1), np.std(values, axis=1)


mean_ssi_singular, std_ssi_singular = mean_and_std(ssi_singular)
mean_nrmse_singular, std_nrmse_singular = mean_and_std(nrmse_singular)

mean_ssi_correlation, std_ssi_correlation = mean_and_std(ssi_correlation)
mean_nrmse_correlation, std_nrmse_correlation = mean_and_std(nrmse_correlation)

mean_ssi_c_noisy = float(np.mean(ssi_c_noisy))
std_ssi_c_noisy = float(np.std(ssi_c_noisy))

mean_nrmse_c_noisy = float(np.mean(nrmse_c_noisy))
std_nrmse_c_noisy = float(np.std(nrmse_c_noisy))

mean_ssi_ctc = float(np.mean(ssi_ctc))
std_ssi_ctc = float(np.std(ssi_ctc))

mean_nrmse_ctc = float(np.mean(nrmse_ctc))
std_nrmse_ctc = float(np.std(nrmse_ctc))


###############################################################################
# Save numerical results
###############################################################################

c = "correlated" if corr else "uncorrelated"

base_tag = (
    f"noise{noise_level:g}_N{N}_m{m}_trials{trials}"
    f"_threshold0-{args.threshold_max:g}_step{args.threshold_step:g}"
    f"_a{a:g}_p{p}_{c}"
)

results = pd.DataFrame({
    "threshold_percent": threshold_percentages,
    "mean_ssi_conceptor_singular_threshold": mean_ssi_singular,
    "std_ssi_conceptor_singular_threshold": std_ssi_singular,
    "mean_ssi_correlation_eigenvalue_threshold": mean_ssi_correlation,
    "std_ssi_correlation_eigenvalue_threshold": std_ssi_correlation,
    "mean_nrmse_conceptor_singular_threshold": mean_nrmse_singular,
    "std_nrmse_conceptor_singular_threshold": std_nrmse_singular,
    "mean_nrmse_correlation_eigenvalue_threshold": mean_nrmse_correlation,
    "std_nrmse_correlation_eigenvalue_threshold": std_nrmse_correlation,
    "mean_ssi_c_noisy": mean_ssi_c_noisy,
    "std_ssi_c_noisy": std_ssi_c_noisy,
    "mean_nrmse_c_noisy": mean_nrmse_c_noisy,
    "std_nrmse_c_noisy": std_nrmse_c_noisy,
    "mean_ssi_ctc": mean_ssi_ctc,
    "std_ssi_ctc": std_ssi_ctc,
    "mean_nrmse_ctc": mean_nrmse_ctc,
    "std_nrmse_ctc": std_nrmse_ctc,
})

results.to_csv(
    PLOTS_DIR / f"thresholding_analysis_{base_tag}.csv",
    index=False,
)


###############################################################################
# Plot style
###############################################################################

plt.rcParams.update({
    "figure.figsize": (8, 4),
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 14,
    "lines.linewidth": 2,
    "lines.markersize": 8,
    "grid.alpha": 0.6,
    "grid.linestyle": "--",
    "axes.linewidth": 1.8,
    "axes.edgecolor": "black",
})

# Show at most approximately 11 labels on the x axis.
# For example, with thresholds from 0 to 100 in steps of 5,
# the plotted points remain every 5%, but the labels are shown every 10%.
max_x_ticks = 11
tick_stride = max(
    1,
    int(np.ceil(len(threshold_percentages) / max_x_ticks)),
)

visible_threshold_ticks = threshold_percentages[::tick_stride]

# Always include the final threshold value.
if not np.isclose(
    visible_threshold_ticks[-1],
    threshold_percentages[-1],
):
    visible_threshold_ticks = np.append(
        visible_threshold_ticks,
        threshold_percentages[-1],
    )


###############################################################################
# Figure 1: SSI versus threshold
###############################################################################

plt.figure(figsize=(8, 4), dpi=300)

plt.errorbar(
    threshold_percentages,
    mean_ssi_singular,
    yerr=std_ssi_singular,
    fmt="-o",
    color="#009E73",
    alpha=0.9,
    ecolor="black",
    elinewidth=1.5,
    capsize=4,
    label=r"Thresholding singular values of $C_{noisy}$",
)

plt.errorbar(
    threshold_percentages,
    mean_ssi_correlation,
    yerr=std_ssi_correlation,
    fmt="-s",
    color="#8E44AD",
    alpha=0.9,
    ecolor="black",
    elinewidth=1.5,
    capsize=4,
    label=r"Thresholding eigenvalues of $R_{noisy}$",
)

plt.axhspan(
    mean_ssi_c_noisy - std_ssi_c_noisy,
    mean_ssi_c_noisy + std_ssi_c_noisy,
    color="#6BAED6",
    alpha=0.08,
)

plt.axhline(
    mean_ssi_c_noisy,
    color="#6BAED6",
    linestyle="--",
    linewidth=2.2,
    label=r"$C_{noisy}$ reference",
)

plt.axhspan(
    mean_ssi_ctc - std_ssi_ctc,
    mean_ssi_ctc + std_ssi_ctc,
    color="#1F4E79",
    alpha=0.08,
)

plt.axhline(
    mean_ssi_ctc,
    color="#1F4E79",
    linestyle="--",
    linewidth=2.2,
    label=r"$C_{ctc}$ reference",
)

plt.xlabel("Threshold (% of maximum value)")
plt.ylabel("PCA Subspace Similarity", size=18)
plt.xlim(0, args.threshold_max)
plt.xticks(visible_threshold_ticks)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()

ssi_name = f"SSI_thresholding_comparison_{base_tag}"

plt.savefig(PLOTS_DIR / f"{ssi_name}.png", dpi=300, bbox_inches="tight")
plt.savefig(PLOTS_DIR / f"{ssi_name}.pdf", dpi=300, bbox_inches="tight")

plt.show()
plt.close()


###############################################################################
# Figure 2: NRMSE versus threshold
###############################################################################

plt.figure(figsize=(8, 4), dpi=300)

plt.errorbar(
    threshold_percentages,
    mean_nrmse_singular,
    yerr=std_nrmse_singular,
    fmt="-o",
    color="#009E73",
    alpha=0.9,
    ecolor="black",
    elinewidth=1.5,
    capsize=4,
    label=r"Thresholding SV of $C_{noisy}$",
)

plt.errorbar(
    threshold_percentages,
    mean_nrmse_correlation,
    yerr=std_nrmse_correlation,
    fmt="-s",
    color="#8E44AD",
    alpha=0.9,
    ecolor="black",
    elinewidth=1.5,
    capsize=4,
    label=r"Thresholding EV of $R_{noisy}$",
)

plt.axhspan(
    mean_nrmse_c_noisy - std_nrmse_c_noisy,
    mean_nrmse_c_noisy + std_nrmse_c_noisy,
    color="#6BAED6",
    alpha=0.08,
)

plt.axhline(
    mean_nrmse_c_noisy,
    color="#6BAED6",
    linestyle="--",
    linewidth=2.2,
    label=r"$C_{noisy}$ reference",
)

plt.axhspan(
    mean_nrmse_ctc - std_nrmse_ctc,
    mean_nrmse_ctc + std_nrmse_ctc,
    color="#1F4E79",
    alpha=0.08,
)

plt.axhline(
    mean_nrmse_ctc,
    color="#1F4E79",
    linestyle="--",
    linewidth=2.2,
    label=r"$C_{ctc}$ reference",
)

plt.xlabel("Threshold (% of maximum value)")
plt.ylabel("NRMSE")
plt.xlim(0, args.threshold_max)
plt.xticks(visible_threshold_ticks)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()

nrmse_name = f"NRMSE_thresholding_comparison_{base_tag}"

plt.savefig(PLOTS_DIR / f"{nrmse_name}.png", dpi=300, bbox_inches="tight")
plt.savefig(PLOTS_DIR / f"{nrmse_name}.pdf", dpi=300, bbox_inches="tight")

plt.show()
plt.close()
