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


# Parameters that can be tuned from the terminal
parser = argparse.ArgumentParser()

# The default values give 10 x 10 = 100 ESN/noise combinations per noise level.
parser.add_argument("--trials_noise", type=int, default=2, help="Number of independent noise realizations.")
parser.add_argument("--trials_esn", type=int, default=2, help="Number of independent internal-weight realizations.")
parser.add_argument("--noise_max", type=float, default=105.0, help="Upper limit of the noise scan in percent; this value is not included.")
parser.add_argument("--noise_step", type=float, default=5.0, help="Step of the noise scan in percent.")
parser.add_argument("--a", type=float, default=5.0, help="Conceptor aperture.")
parser.add_argument("--m", type=int, default=2, help="Number of noisy realizations used to compute CTC.")
parser.add_argument("--p", type=int, default=3, help="p-step-ahead prediction used to train the readout.")
parser.add_argument("--time_len", type=int, default=3000, help="Number of samples used to train the reservoir.")
parser.add_argument("--N", type=int, default=200, help="Number of reservoir neurons.")
parser.add_argument("--spectral_radius", type=float, default=1.6, help="Reservoir spectral radius.")
parser.add_argument("--scaling", type=float, default=0.9, help="Input scaling.")
parser.add_argument("--reg", type=float, default=1.0, help="Ridge-regression regularization.")
parser.add_argument("--washout", type=int, default=20, help="Number of initial states discarded before readout training.")
parser.add_argument("--seed", type=int, default=20, help="Base random seed.")
parser.add_argument("--spectrum_noise_1", type=float, default=25.0, help="First noise level shown in the conceptor singular-value spectrum.")
parser.add_argument("--spectrum_noise_2", type=float, default=50.0, help="Second noise level shown in the conceptor singular-value spectrum.")
parser.add_argument("--corr", type=lambda x: str(x).lower() in ["true", "1", "yes", "y"], default=False, help="True for correlated noise and False for uncorrelated noise.")

args = parser.parse_args()


###############################################################################
# Validation and ESN parameters
###############################################################################

if args.trials_noise < 1 or args.trials_esn < 1:
    raise ValueError("'trials_noise' and 'trials_esn' must be at least 1.")

if args.noise_step <= 0:
    raise ValueError("'noise_step' must be larger than zero.")

if args.noise_max <= 0:
    raise ValueError("'noise_max' must be larger than zero.")

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
corr = args.corr

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


###############################################################################
# Noise scan and seeds
###############################################################################

noise_values = np.arange(0.0, args.noise_max, args.noise_step, dtype=float)


def find_noise_index(target_noise):
    """Return the index of an exact noise level included in the scan."""
    matching_indices = np.where(np.isclose(noise_values, target_noise))[0]
    if matching_indices.size == 0:
        raise ValueError(
            f"The requested spectrum noise level {target_noise:g}% is not included "
            "in noise_values. Adjust --noise_step, --noise_max, "
            "--spectrum_noise_1 or --spectrum_noise_2."
        )
    return int(matching_indices[0])


spectrum_noise_1 = args.spectrum_noise_1
spectrum_noise_2 = args.spectrum_noise_2
spectrum_noise_idx_1 = find_noise_index(spectrum_noise_1)
spectrum_noise_idx_2 = find_noise_index(spectrum_noise_2)

np.random.seed(args.seed)

# Independent realizations of the reservoir internal weights.
seed_esn = np.random.randint(0, 2_000_000, size=trials_esn)

# Independent evaluation-noise realizations. The same seeds are used at every
# noise percentage so that all methods are compared under matched trials.
seed_noise = np.random.randint(0, 2_000_000, size=trials_noise)


###############################################################################
# Storage
###############################################################################

# Mean of |Wout|^2 over all readout coefficients in each trial.
wout_sq_no_c = np.empty((len(noise_values), trials), dtype=float)
wout_sq_c_noisy = np.empty((len(noise_values), trials), dtype=float)
wout_sq_c_ctc = np.empty((len(noise_values), trials), dtype=float)

# Mean of |b_out|^2. With one output this is simply |b_out|^2.
bout_sq_no_c = np.empty((len(noise_values), trials), dtype=float)
bout_sq_c_noisy = np.empty((len(noise_values), trials), dtype=float)
bout_sq_c_ctc = np.empty((len(noise_values), trials), dtype=float)

# Standard deviation of the Wout coefficients within each trained model.
wout_std_no_c = np.empty((len(noise_values), trials), dtype=float)
wout_std_c_noisy = np.empty((len(noise_values), trials), dtype=float)
wout_std_c_ctc = np.empty((len(noise_values), trials), dtype=float)

# Singular-value spectra.
# C_ideal and C_ctc are stored once per ESN realization.
singular_values_c_ideal = np.empty((trials_esn, N), dtype=float)
singular_values_c_ctc_noise_1 = np.empty((trials_esn, N), dtype=float)
singular_values_c_ctc_noise_2 = np.empty((trials_esn, N), dtype=float)

# C_noisy depends on both the ESN and noise realizations.
singular_values_c_noisy_noise_1 = np.empty((trials, N), dtype=float)
singular_values_c_noisy_noise_2 = np.empty((trials, N), dtype=float)


def mean_squared_magnitude(values):
    """Return the mean squared absolute value of an array or scalar."""
    array = np.asarray(values)
    return float(np.mean(np.abs(array) ** 2))


def extract_readout_power(trained_params):
    """Extract mean |Wout|^2 and mean |b_out|^2."""
    return mean_squared_magnitude(trained_params["wout"]), mean_squared_magnitude(trained_params["bias_out"])


def extract_wout_std(trained_params):
    """Return the standard deviation of the Wout coefficients."""
    return float(np.std(np.asarray(trained_params["wout"])))


def sorted_singular_values(matrix):
    """Return singular values sorted from largest to smallest."""
    return np.linalg.svd(
        np.asarray(matrix),
        compute_uv=False,
        hermitian=True,
    )


###############################################################################
# Training scan
###############################################################################

for noise_idx, noise_level in enumerate(noise_values):
    for esn_idx in range(trials_esn):
        params = rnn_params(N, input_size, output_size, scaling, spectral_radius, alpha, bias_scaling, sparsity, seed=seed_esn[esn_idx])

        # Clean states define the noise standard deviation for this ESN.
        X_id = forward_rnn(params, ut_train1, 42, x_init=None, autonomous=False, conceptor=None)
        std_noise = std_noise_func(X_id, noise_level)

        # Store the ideal conceptor spectrum once per ESN realization.
        if noise_idx == 0:
            C_ideal = compute_conceptor(X_id, a)
            singular_values_c_ideal[esn_idx, :] = sorted_singular_values(C_ideal)

        # CTC depends on ESN, noise, aperture and m, not on the evaluation seed.
        C_ctc = denoising_CTC_m(params, ut_train1, std_noise, a, m, corr=corr)

        if noise_idx == spectrum_noise_idx_1:
            singular_values_c_ctc_noise_1[esn_idx, :] = sorted_singular_values(C_ctc)

        if noise_idx == spectrum_noise_idx_2:
            singular_values_c_ctc_noise_2[esn_idx, :] = sorted_singular_values(C_ctc)

        for trial_noise_idx in range(trials_noise):
            current_seed = seed_noise[trial_noise_idx]
            trial_index = esn_idx * trials_noise + trial_noise_idx

            ###################################################################
            # 1. Training without conceptor
            ###################################################################

            X_no_c = forward_rnn(params, ut_train1, current_seed, None, False, None, std_noise, corr=corr)
            params_trained_no_c, _ = ridge(reg, X_no_c[washout:], yt_train_effective, step, params)
            wout_sq_no_c[noise_idx, trial_index], bout_sq_no_c[noise_idx, trial_index] = extract_readout_power(params_trained_no_c)
            wout_std_no_c[noise_idx, trial_index] = extract_wout_std(params_trained_no_c)

            ###################################################################
            # 2. Training with C_noisy
            ###################################################################

            C_noisy = compute_conceptor(X_no_c, a)

            if noise_idx == spectrum_noise_idx_1:
                singular_values_c_noisy_noise_1[trial_index, :] = sorted_singular_values(C_noisy)

            if noise_idx == spectrum_noise_idx_2:
                singular_values_c_noisy_noise_2[trial_index, :] = sorted_singular_values(C_noisy)

            X_c_noisy = forward_rnn(params, ut_train1, current_seed, None, False, C_noisy, std_noise, corr=corr)
            params_trained_c_noisy, _ = ridge(reg, X_c_noisy[washout:], yt_train_effective, step, params)
            wout_sq_c_noisy[noise_idx, trial_index], bout_sq_c_noisy[noise_idx, trial_index] = extract_readout_power(params_trained_c_noisy)
            wout_std_c_noisy[noise_idx, trial_index] = extract_wout_std(params_trained_c_noisy)

            ###################################################################
            # 3. Training with CTC
            ###################################################################

            X_c_ctc = forward_rnn(params, ut_train1, current_seed, None, False, C_ctc, std_noise, corr=corr)
            params_trained_c_ctc, _ = ridge(reg, X_c_ctc[washout:], yt_train_effective, step, params)
            wout_sq_c_ctc[noise_idx, trial_index], bout_sq_c_ctc[noise_idx, trial_index] = extract_readout_power(params_trained_c_ctc)
            wout_std_c_ctc[noise_idx, trial_index] = extract_wout_std(params_trained_c_ctc)


###############################################################################
# Mean and standard deviation over ESN/noise combinations
###############################################################################

def mean_and_std(values):
    return np.mean(values, axis=1), np.std(values, axis=1)


mean_wout_no_c, std_wout_no_c = mean_and_std(wout_sq_no_c)
mean_wout_c_noisy, std_wout_c_noisy = mean_and_std(wout_sq_c_noisy)
mean_wout_c_ctc, std_wout_c_ctc = mean_and_std(wout_sq_c_ctc)

mean_bout_no_c, std_bout_no_c = mean_and_std(bout_sq_no_c)
mean_bout_c_noisy, std_bout_c_noisy = mean_and_std(bout_sq_c_noisy)
mean_bout_c_ctc, std_bout_c_ctc = mean_and_std(bout_sq_c_ctc)

mean_wout_std_no_c, std_wout_std_no_c = mean_and_std(wout_std_no_c)
mean_wout_std_c_noisy, std_wout_std_c_noisy = mean_and_std(wout_std_c_noisy)
mean_wout_std_c_ctc, std_wout_std_c_ctc = mean_and_std(wout_std_c_ctc)


def spectrum_mean_and_std(values):
    """Mean and standard deviation at every singular-value index."""
    return np.mean(values, axis=0), np.std(values, axis=0)


mean_sv_c_ideal, std_sv_c_ideal = spectrum_mean_and_std(singular_values_c_ideal)
mean_sv_c_noisy_noise_1, std_sv_c_noisy_noise_1 = spectrum_mean_and_std(singular_values_c_noisy_noise_1)
mean_sv_c_ctc_noise_1, std_sv_c_ctc_noise_1 = spectrum_mean_and_std(singular_values_c_ctc_noise_1)
mean_sv_c_noisy_noise_2, std_sv_c_noisy_noise_2 = spectrum_mean_and_std(singular_values_c_noisy_noise_2)
mean_sv_c_ctc_noise_2, std_sv_c_ctc_noise_2 = spectrum_mean_and_std(singular_values_c_ctc_noise_2)


###############################################################################
# Save numerical results
###############################################################################

# Defensive creation in case the folder was deleted or not synchronized.
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


results = pd.DataFrame({
    "noise_percent": noise_values,
    "mean_wout_abs_squared_no_c": mean_wout_no_c,
    "std_wout_abs_squared_no_c": std_wout_no_c,
    "mean_wout_abs_squared_c_noisy": mean_wout_c_noisy,
    "std_wout_abs_squared_c_noisy": std_wout_c_noisy,
    "mean_wout_abs_squared_c_ctc": mean_wout_c_ctc,
    "std_wout_abs_squared_c_ctc": std_wout_c_ctc,
    "mean_bout_abs_squared_no_c": mean_bout_no_c,
    "std_bout_abs_squared_no_c": std_bout_no_c,
    "mean_bout_abs_squared_c_noisy": mean_bout_c_noisy,
    "std_bout_abs_squared_c_noisy": std_bout_c_noisy,
    "mean_bout_abs_squared_c_ctc": mean_bout_c_ctc,
    "std_bout_abs_squared_c_ctc": std_bout_c_ctc,
    "mean_wout_std_no_c": mean_wout_std_no_c,
    "std_wout_std_no_c": std_wout_std_no_c,
    "mean_wout_std_c_noisy": mean_wout_std_c_noisy,
    "std_wout_std_c_noisy": std_wout_std_c_noisy,
    "mean_wout_std_c_ctc": mean_wout_std_c_ctc,
    "std_wout_std_c_ctc": std_wout_std_c_ctc,
})

c = "correlated" if corr else "uncorrelated"
# Keep output names short enough for Windows paths.
# The full configuration is still saved inside the CSV contents.
base_tag = (
    f"N{N}_m{m}_T{trials}"
    f"_dn{args.noise_step:g}_nmax{args.noise_max:g}"
    f"_a{a:g}_p{p}_{c}"
)

results.to_csv(PLOTS_DIR / f"readout_scan_{base_tag}.csv", index=False)

singular_value_indices = np.arange(1, N + 1)

singular_spectrum_results = pd.DataFrame({
    "singular_value_index": singular_value_indices,
    "mean_c_ideal_noise_0": mean_sv_c_ideal,
    "std_c_ideal_noise_0": std_sv_c_ideal,
    f"mean_c_noisy_noise_{spectrum_noise_1:g}": mean_sv_c_noisy_noise_1,
    f"std_c_noisy_noise_{spectrum_noise_1:g}": std_sv_c_noisy_noise_1,
    f"mean_c_ctc_noise_{spectrum_noise_1:g}": mean_sv_c_ctc_noise_1,
    f"std_c_ctc_noise_{spectrum_noise_1:g}": std_sv_c_ctc_noise_1,
    f"mean_c_noisy_noise_{spectrum_noise_2:g}": mean_sv_c_noisy_noise_2,
    f"std_c_noisy_noise_{spectrum_noise_2:g}": std_sv_c_noisy_noise_2,
    f"mean_c_ctc_noise_{spectrum_noise_2:g}": mean_sv_c_ctc_noise_2,
    f"std_c_ctc_noise_{spectrum_noise_2:g}": std_sv_c_ctc_noise_2,
})

singular_spectrum_results.to_csv(
    PLOTS_DIR / f"conceptor_sv_{base_tag}.csv",
    index=False,
)


###############################################################################
# Plot style
###############################################################################

plt.rcParams.update({"figure.figsize": (8, 4), "axes.labelsize": 20, "axes.titlesize": 20, "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 16, "lines.linewidth": 2, "lines.markersize": 8, "grid.alpha": 0.6, "grid.linestyle": "--", "axes.linewidth": 1.8, "axes.edgecolor": "black"})


###############################################################################
# Figure 1: Mean squared magnitude of Wout
###############################################################################

plt.figure(figsize=(8, 4), dpi=300)

plt.errorbar(noise_values, mean_wout_no_c, yerr=std_wout_no_c, fmt="-s", color="#B22222", alpha=0.85, ecolor="black", elinewidth=1.6, capsize=5, label="Without C")
plt.errorbar(noise_values, mean_wout_c_noisy, yerr=std_wout_c_noisy, fmt="-^", color="#6BAED6", alpha=0.85, ecolor="black", elinewidth=1.6, capsize=5, label=r"With $C_{noisy}$")
plt.errorbar(noise_values, mean_wout_c_ctc, yerr=std_wout_c_ctc, fmt="-o", color="#1F4E79", alpha=0.85, ecolor="black", elinewidth=1.6, capsize=5, label=r"With $C_{ctc}$")

plt.xlabel("% Noise")
plt.ylabel(r"$\langle ||W_{out}||^2 \rangle$")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

wout_name = f"Wout_power_{base_tag}"
plt.savefig(PLOTS_DIR / f"{wout_name}.png", dpi=300, bbox_inches="tight")
plt.savefig(PLOTS_DIR / f"{wout_name}.pdf", dpi=300, bbox_inches="tight")

plt.show()
plt.close()


###############################################################################
# Figure 2: Mean squared magnitude of b_out
###############################################################################

plt.figure(figsize=(8, 4), dpi=300)

plt.errorbar(noise_values, mean_bout_no_c, yerr=std_bout_no_c, fmt="-s", color="#B22222", alpha=0.85, ecolor="black", elinewidth=1.6, capsize=5, label="Without C")
plt.errorbar(noise_values, mean_bout_c_noisy, yerr=std_bout_c_noisy, fmt="-^", color="#6BAED6", alpha=0.85, ecolor="black", elinewidth=1.6, capsize=5, label=r"With $C_{noisy}$")
plt.errorbar(noise_values, mean_bout_c_ctc, yerr=std_bout_c_ctc, fmt="-o", color="#1F4E79", alpha=0.85, ecolor="black", elinewidth=1.6, capsize=5, label=r"With $C_{ctc}$")

plt.xlabel("% Noise")
plt.ylabel(r"$||b_{out}||^2$")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

bout_name = f"bout_power_{base_tag}"
plt.savefig(PLOTS_DIR / f"{bout_name}.png", dpi=300, bbox_inches="tight")
plt.savefig(PLOTS_DIR / f"{bout_name}.pdf", dpi=300, bbox_inches="tight")

plt.show()
plt.close()

###############################################################################
# Figure 3: Standard deviation of Wout coefficients
###############################################################################

plt.figure(figsize=(8, 4), dpi=300)

plt.errorbar(noise_values, mean_wout_std_no_c, yerr=std_wout_std_no_c, fmt="-s", color="#B22222", alpha=0.85, ecolor="black", elinewidth=1.6, capsize=5, label="Without C")
plt.errorbar(noise_values, mean_wout_std_c_noisy, yerr=std_wout_std_c_noisy, fmt="-^", color="#6BAED6", alpha=0.85, ecolor="black", elinewidth=1.6, capsize=5, label=r"With $C_{noisy}$")
plt.errorbar(noise_values, mean_wout_std_c_ctc, yerr=std_wout_std_c_ctc, fmt="-o", color="#1F4E79", alpha=0.85, ecolor="black", elinewidth=1.6, capsize=5, label=r"With $C_{ctc}$")

plt.xlabel("% Noise")
plt.ylabel(r"$\mathrm{std}(W_{out})$")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

wout_std_name = f"Wout_std_{base_tag}"
plt.savefig(PLOTS_DIR / f"{wout_std_name}.png", dpi=300, bbox_inches="tight")
plt.savefig(PLOTS_DIR / f"{wout_std_name}.pdf", dpi=300, bbox_inches="tight")

plt.show()
plt.close()

###############################################################################
# Figure 4: Mean singular-value spectra of the conceptors
###############################################################################

plt.figure(figsize=(8, 4), dpi=300)

# Solid lines: first selected noise level. Dashed lines: second noise level.
plt.plot(
    singular_value_indices,
    mean_sv_c_ideal,
    color="#D4A017",
    linestyle="-",
    linewidth=2.5,
    label=r"$C$, 0% noise",
)
ideal_lower = np.maximum(mean_sv_c_ideal - std_sv_c_ideal, 0.0)
ideal_upper = mean_sv_c_ideal + std_sv_c_ideal

plt.fill_between(singular_value_indices, ideal_lower, ideal_upper, color="#D4A017", alpha=0.2, linewidth=0)
plt.plot(singular_value_indices, ideal_lower, color="#D4A017", linestyle="-", linewidth=1.2, alpha=0.55)
plt.plot(singular_value_indices, ideal_upper, color="#D4A017", linestyle="-", linewidth=1.2, alpha=0.55)

plt.plot(
    singular_value_indices,
    mean_sv_c_noisy_noise_1,
    color="#6BAED6",
    linestyle="-",
    linewidth=2.2,
    label=rf"$C_{{noisy}}$, {spectrum_noise_1:g}% noise",
)
noisy_1_lower = np.maximum(mean_sv_c_noisy_noise_1 - std_sv_c_noisy_noise_1, 0.0)
noisy_1_upper = mean_sv_c_noisy_noise_1 + std_sv_c_noisy_noise_1

plt.fill_between(singular_value_indices, noisy_1_lower, noisy_1_upper, color="#6BAED6", alpha=0.10, linewidth=0)
plt.plot(singular_value_indices, noisy_1_lower, color="#6BAED6", linestyle="-", linewidth=1.1, alpha=0.50)
plt.plot(singular_value_indices, noisy_1_upper, color="#6BAED6", linestyle="-", linewidth=1.1, alpha=0.50)

plt.plot(
    singular_value_indices,
    mean_sv_c_ctc_noise_1,
    color="#1F4E79",
    linestyle="-",
    linewidth=2.2,
    label=rf"$C_{{ctc}}$, {spectrum_noise_1:g}% noise",
)
ctc_1_lower = np.maximum(mean_sv_c_ctc_noise_1 - std_sv_c_ctc_noise_1, 0.0)
ctc_1_upper = mean_sv_c_ctc_noise_1 + std_sv_c_ctc_noise_1

plt.fill_between(singular_value_indices, ctc_1_lower, ctc_1_upper, color="#1F4E79", alpha=0.10, linewidth=0)
plt.plot(singular_value_indices, ctc_1_lower, color="#1F4E79", linestyle="-", linewidth=1.1, alpha=0.50)
plt.plot(singular_value_indices, ctc_1_upper, color="#1F4E79", linestyle="-", linewidth=1.1, alpha=0.50)

plt.plot(
    singular_value_indices,
    mean_sv_c_noisy_noise_2,
    color="#6BAED6",
    linestyle="--",
    linewidth=2.2,
    label=rf"$C_{{noisy}}$, {spectrum_noise_2:g}% noise",
)
noisy_2_lower = np.maximum(mean_sv_c_noisy_noise_2 - std_sv_c_noisy_noise_2, 0.0)
noisy_2_upper = mean_sv_c_noisy_noise_2 + std_sv_c_noisy_noise_2

plt.fill_between(singular_value_indices, noisy_2_lower, noisy_2_upper, color="#6BAED6", alpha=0.07, linewidth=0)
plt.plot(singular_value_indices, noisy_2_lower, color="#6BAED6", linestyle="--", linewidth=1.1, alpha=0.50)
plt.plot(singular_value_indices, noisy_2_upper, color="#6BAED6", linestyle="--", linewidth=1.1, alpha=0.50)

plt.plot(
    singular_value_indices,
    mean_sv_c_ctc_noise_2,
    color="#1F4E79",
    linestyle="--",
    linewidth=2.2,
    label=rf"$C_{{ctc}}$, {spectrum_noise_2:g}% noise",
)
ctc_2_lower = np.maximum(mean_sv_c_ctc_noise_2 - std_sv_c_ctc_noise_2, 0.0)
ctc_2_upper = mean_sv_c_ctc_noise_2 + std_sv_c_ctc_noise_2

plt.fill_between(singular_value_indices, ctc_2_lower, ctc_2_upper, color="#1F4E79", alpha=0.07, linewidth=0)
plt.plot(singular_value_indices, ctc_2_lower, color="#1F4E79", linestyle="--", linewidth=1.1, alpha=0.50)
plt.plot(singular_value_indices, ctc_2_upper, color="#1F4E79", linestyle="--", linewidth=1.1, alpha=0.50)

plt.xlabel("Singular value index")
plt.ylabel("Singular value")
plt.xlim(1, N)
plt.ylim(bottom=0)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()

singular_spectrum_name = (
    f"conceptor_sv_"
    f"noise{spectrum_noise_1:g}_{spectrum_noise_2:g}_{base_tag}"
)

plt.savefig(
    PLOTS_DIR / f"{singular_spectrum_name}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    PLOTS_DIR / f"{singular_spectrum_name}.pdf",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close()