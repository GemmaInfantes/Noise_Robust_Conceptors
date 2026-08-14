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


from utils.rnn_utils import denoising_CTC_m
from utils.rnn_utils import forward_rnn
from utils.rnn_utils import ridge
from utils.rnn_utils import rnn_params
from utils.rnn_utils import std_noise_func
from utils.utils import NRMSE
from utils.utils import xcorr_PCA


###############################################################################
# Arguments
###############################################################################

parser = argparse.ArgumentParser()

parser.add_argument("--trials_noise", type=int, default=10)
parser.add_argument("--trials_esn", type=int, default=10)
parser.add_argument("--noise", type=float, default=50.0)
parser.add_argument("--p", type=int, default=3)
parser.add_argument("--a_in", type=float, default=1.0)
parser.add_argument("--a_fin", type=float, default=25.0)
parser.add_argument("--a_step", type=float, default=1.0)
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--time_len", type=int, default=3000)
parser.add_argument("--N", type=int, default=200)
parser.add_argument("--spectral_radius", type=float, default=1.6)
parser.add_argument("--scaling", type=float, default=0.9)
parser.add_argument("--m", type=int, default=2)
parser.add_argument("--seed", type=int, default=20)
parser.add_argument(
    "--corr",
    type=lambda x: str(x).lower() in ["true", "1", "yes", "y"],
    default=False,
)

args = parser.parse_args()


###############################################################################
# ESN parameters
###############################################################################

spectral_radius = args.spectral_radius
scaling = args.scaling
bias_scaling = 0.4
alpha = 0.75

a_in = args.a_in
a_fin = args.a_fin
a_step = args.a_step

N = args.N
washout = 20
reg = 1
time_len = args.time_len
steps = args.steps
m = args.m
sparsity = None
corr = args.corr
noise_level = args.noise
p = args.p

trials_noise = args.trials_noise
trials_esn = args.trials_esn
trials = trials_noise * trials_esn


###############################################################################
# Validation
###############################################################################

if trials_noise < 1 or trials_esn < 1:
    raise ValueError("'trials_noise' and 'trials_esn' must be at least 1.")

if a_step <= 0:
    raise ValueError("'a_step' must be larger than zero.")

if a_fin < a_in:
    raise ValueError("'a_fin' must be greater than or equal to 'a_in'.")

if m < 2:
    raise ValueError("'m' must be at least 2 for the CTC computation.")

if p < 1:
    raise ValueError("'p' must be at least 1.")

if time_len <= max(p, 10) + washout:
    raise ValueError("'time_len' is too short for the selected p values and washout.")


###############################################################################
# Input signal
###############################################################################

data1 = pd.read_csv(DATA_DIR / "xRossler.txt", sep="\t", header=None, index_col=None)

data1 = data1.values[:time_len].reshape(-1, 1)


###############################################################################
# Aperture values and matched seeds
###############################################################################

aa = np.arange(float(a_in), float(a_fin) + 0.5 * float(a_step), float(a_step), dtype=float)

np.random.seed(args.seed)

# Same ESN/noise seeds are used for the selected p, p=5 and p=10.
seed_noise = np.random.randint(0, 2000, size=trials_noise)
seed_esn = np.random.randint(0, 2000, size=trials_esn)


###############################################################################
# Aperture-scan function
###############################################################################

def run_aperture_scan(p_value, calculate_ssi=True):
    """
    Run the aperture scan for one p-step-ahead prediction horizon.

    Parameters
    ----------
    p_value : int
        Prediction step used to shift the target and train the readout.
    calculate_ssi : bool
        When True, also calculate PCA subspace similarity.

    Returns
    -------
    dict
        Mean and standard deviation of NRMSE and, optionally, SSI.
    """
    step_value = int(p_value)

    ut_train = data1[:-step_value]
    yt_train = data1[step_value:]

    input_size = ut_train.shape[-1]
    output_size = yt_train.shape[-1]

    yt_train_effective = yt_train[washout:]

    eval_stop = min(washout + steps, len(yt_train))
    y_target = np.asarray(yt_train[washout:eval_stop]).ravel()
    nrmse_values = np.empty((len(aa), trials), dtype=float)

    if calculate_ssi:
        ssi_values = np.empty((len(aa), trials), dtype=float)
    else:
        ssi_values = None

    for a_idx, aperture in enumerate(aa):
        aperture = float(aperture)

        for esn_idx in range(trials_esn):
            params = rnn_params(N, input_size, output_size, scaling, spectral_radius, alpha, bias_scaling, sparsity, seed=seed_esn[esn_idx])
            X_id = forward_rnn(params, ut_train, 42, x_init=None, autonomous=False, conceptor=None)
            std_noise = std_noise_func(X_id, noise_level)
            C_ctc = denoising_CTC_m(params, ut_train, std_noise, aperture, m, corr=corr)

            for noise_idx in range(trials_noise):
                current_seed = seed_noise[noise_idx]
                trial_index = esn_idx * trials_noise + noise_idx

                X_noi_C_ctc = forward_rnn(params, ut_train, current_seed, None, False, C_ctc, std_noise, corr=corr)

                X_effective = X_noi_C_ctc[washout:]

                params_trained_CTC, _ = ridge(reg, X_effective, yt_train_effective, step_value, params)

                if calculate_ssi:
                    effective_ssi_steps = min(steps, len(X_id) - washout, len(X_noi_C_ctc) - washout)
                    ssi_values[a_idx, trial_index] = xcorr_PCA(X_id, X_noi_C_ctc, washout, effective_ssi_steps)

                y_C_ctc = np.asarray(X_noi_C_ctc[washout:eval_stop] @ params_trained_CTC["wout"].T + params_trained_CTC["bias_out"]).ravel()

                nrmse_values[a_idx, trial_index] = NRMSE(y_target, y_C_ctc)

    results = {
        "mean_nrmse": np.mean(nrmse_values, axis=1),
        "std_nrmse": np.std(nrmse_values, axis=1),
    }

    if calculate_ssi:
        results["mean_ssi"] = np.mean(ssi_values, axis=1)
        results["std_ssi"] = np.std(ssi_values, axis=1)

    return results


###############################################################################
# Simulations
###############################################################################

# Original scan for the p selected from the terminal.
results_selected_p = run_aperture_scan(p, calculate_ssi=True)

# Additional NRMSE scans for the joint p=5 and p=10 comparison.
results_p5 = run_aperture_scan(5, calculate_ssi=False)

results_p10 = run_aperture_scan(10, calculate_ssi=False)


###############################################################################
# Extract results
###############################################################################

mean_ssi_C_ctc = results_selected_p["mean_ssi"]
std_ssi_C_ctc = results_selected_p["std_ssi"]

mean_nrmse_C_ctc = results_selected_p["mean_nrmse"]
std_nrmse_C_ctc = results_selected_p["std_nrmse"]

mean_nrmse_p5 = results_p5["mean_nrmse"]
std_nrmse_p5 = results_p5["std_nrmse"]

mean_nrmse_p10 = results_p10["mean_nrmse"]
std_nrmse_p10 = results_p10["std_nrmse"]


###############################################################################
# Save numerical results
###############################################################################

c = "correlated" if corr else "uncorrelated"
noise_tag = f"{noise_level:g}"

selected_p_results = pd.DataFrame({
    "aperture": aa,
    "mean_ssi": mean_ssi_C_ctc,
    "std_ssi": std_ssi_C_ctc,
    f"mean_nrmse_p{p}": mean_nrmse_C_ctc,
    f"std_nrmse_p{p}": std_nrmse_C_ctc,
})

selected_p_results.to_csv(PLOTS_DIR / f"ap_scan_n{noise_tag}_p{p}_N{N}_m{m}_T{trials}_{c}.csv", index=False)

p5_p10_results = pd.DataFrame({
    "aperture": aa,
    "mean_nrmse_p5": mean_nrmse_p5,
    "std_nrmse_p5": std_nrmse_p5,
    "mean_nrmse_p10": mean_nrmse_p10,
    "std_nrmse_p10": std_nrmse_p10,
})

p5_p10_results.to_csv(PLOTS_DIR / f"ap_scan_n{noise_tag}_p5_p10_N{N}_m{m}_T{trials}_{c}.csv", index=False)


###############################################################################
# Ensure output folder exists
###############################################################################

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


###############################################################################
# Plot style
###############################################################################

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "lines.linewidth": 2,
    "lines.markersize": 9,
    "grid.alpha": 0.6,
    "grid.linestyle": "--",
    "axes.linewidth": 1.8,
    "axes.edgecolor": "black",
})


###############################################################################
# Figure 1: SSI for the selected p
###############################################################################

plt.figure(figsize=(8, 4), dpi=300)

plt.plot(
    aa,
    mean_ssi_C_ctc,
    "-o",
    color="#1F4E79",
    alpha=0.95,
    label=rf"$C_{{ctc}}$, $p={p}$",
)

ssi_lower = mean_ssi_C_ctc - std_ssi_C_ctc
ssi_upper = mean_ssi_C_ctc + std_ssi_C_ctc

plt.fill_between(
    aa,
    ssi_lower,
    ssi_upper,
    color="#1F4E79",
    alpha=0.18,
    linewidth=0,
)

plt.plot(aa, ssi_lower, color="#1F4E79", alpha=0.45, linewidth=1.3)
plt.plot(aa, ssi_upper, color="#1F4E79", alpha=0.45, linewidth=1.3)

plt.xlabel("Aperture")
plt.ylabel("PCA Subspace Similarity", size=19)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

ssi_name = (
    f"SSI_ap_n{noise_tag}_p{p}_N{N}_m{m}_T{trials}_"
    f"a{a_in:g}-{a_fin:g}_da{a_step:g}_{c}"
)

plt.savefig(
    PLOTS_DIR / f"{ssi_name}.png",
    dpi=300,
    bbox_inches="tight",
)

plt.savefig(
    PLOTS_DIR / f"{ssi_name}.pdf",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close()


###############################################################################
# Figure 2: NRMSE for the selected p
###############################################################################

plt.figure(figsize=(8, 4), dpi=300)

plt.plot(
    aa,
    mean_nrmse_C_ctc,
    "-o",
    color="#1F4E79",
    alpha=0.95,
    label=rf"$C_{{ctc}}$, $p={p}$",
)

nrmse_lower = np.maximum(mean_nrmse_C_ctc - std_nrmse_C_ctc, 0.0)

nrmse_upper = mean_nrmse_C_ctc + std_nrmse_C_ctc

plt.fill_between(
    aa,
    nrmse_lower,
    nrmse_upper,
    color="#1F4E79",
    alpha=0.18,
    linewidth=0,
)

plt.plot(aa, nrmse_lower, color="#1F4E79", alpha=0.45, linewidth=1.3)
plt.plot(aa, nrmse_upper, color="#1F4E79", alpha=0.45, linewidth=1.3)

plt.xlabel("Aperture")
plt.ylabel("NRMSE")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

nrmse_name = (
    f"NRMSE_ap_n{noise_tag}_p{p}_N{N}_m{m}_T{trials}_"
    f"a{a_in:g}-{a_fin:g}_da{a_step:g}_{c}"
)

plt.savefig(
    PLOTS_DIR / f"{nrmse_name}.png",
    dpi=300,
    bbox_inches="tight",
)

plt.savefig(
    PLOTS_DIR / f"{nrmse_name}.pdf",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close()


###############################################################################
# Figure 3: Joint NRMSE comparison for p=5 and p=10
###############################################################################

plt.figure(figsize=(8, 4), dpi=300)

# p = 5
plt.plot(
    aa,
    mean_nrmse_p5,
    "-o",
    color="#1F4E79",
    alpha=0.95,
    label=r"$p=5$",
)

plt.fill_between(
    aa,
    np.maximum(
        mean_nrmse_p5 - std_nrmse_p5,
        0.0,
    ),
    mean_nrmse_p5 + std_nrmse_p5,
    color="#1F4E79",
    alpha=0.16,
    linewidth=0,
)

# p = 10
plt.plot(
    aa,
    mean_nrmse_p10,
    "-s",
    color="#009E9A",
    alpha=0.95,
    label=r"$p=10$",
)

plt.fill_between(
    aa,
    np.maximum(
        mean_nrmse_p10 - std_nrmse_p10,
        0.0,
    ),
    mean_nrmse_p10 + std_nrmse_p10,
    color="#009E9A",
    alpha=0.16,
    linewidth=0,
)

plt.xlabel("Aperture")
plt.ylabel("NRMSE")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

nrmse_p5_p10_name = (
    f"NRMSE_ap_n{noise_tag}_p5_p10_N{N}_m{m}_T{trials}_"
    f"a{a_in:g}-{a_fin:g}_da{a_step:g}_{c}"
)

plt.savefig(
    PLOTS_DIR / f"{nrmse_p5_p10_name}.png",
    dpi=300,
    bbox_inches="tight",
)

plt.savefig(
    PLOTS_DIR / f"{nrmse_p5_p10_name}.pdf",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close()
