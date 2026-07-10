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


# Parameters that can be tuned from the terminal
parser = argparse.ArgumentParser()

parser.add_argument("--trials_noise", type=int, default=10, help="Number of different noise realizations.")
parser.add_argument("--trials_esn", type=int, default=10, help="Number of different ESN realizations.")
parser.add_argument("--noise", type=float, default=50.0, help="Noise level used for the individual m-scan figures (%%).")
parser.add_argument("--p", type=int, default=1, help="p-step-ahead prediction.")
parser.add_argument("--m_in", type=int, default=1, help="Initial scan value. For m=1, C_noisy is used; for m>=2, CTC is used.")
parser.add_argument("--m_fin", type=int, default=10, help="Final number of realizations used to compute CTC.")
parser.add_argument("--m_step", type=int, default=1, help="Step between consecutive m values.")
parser.add_argument("--steps", type=int, default=3000, help="Number of time steps used to compute the SSI and NRMSE.")
parser.add_argument("--time_len", type=int, default=3000, help="Number of time steps used to train the reservoir.")
parser.add_argument("--N", type=int, default=200, help="Number of reservoir neurons.")
parser.add_argument("--spectral_radius", type=float, default=1.6, help="Reservoir spectral radius.")
parser.add_argument("--scaling", type=float, default=0.9, help="Input scaling.")
parser.add_argument("--seed", type=int, default=20, help="Seed used to generate the ESN and noise seeds.")
parser.add_argument("--corr", type=lambda x: str(x).lower() in ["true", "1", "yes", "y"], default=False, help="True for correlated noise and False for uncorrelated noise.")
parser.add_argument("--a", type=float, default=5.0, help="Conceptor aperture.")

args = parser.parse_args()


###############################################################################
# ESN parameters
###############################################################################

spectral_radius = args.spectral_radius
scaling = args.scaling
bias_scaling = 0.4
alpha = 0.75

a = args.a
m_in = args.m_in
m_fin = args.m_fin
m_step = args.m_step

N = args.N
washout = 20
reg = 1
time_len = args.time_len
steps = args.steps
sparsity = None
corr = args.corr
p = args.p
step = p

if m_in < 1:
    raise ValueError("'m_in' must be at least 1.")

if m_fin < m_in:
    raise ValueError("'m_fin' must be greater than or equal to 'm_in'.")

if m_step < 1:
    raise ValueError("'m_step' must be at least 1.")

if p < 1:
    raise ValueError("'p' must be at least 1.")


###############################################################################
# Input signal: Rössler x
###############################################################################

data1 = pd.read_csv(DATA_DIR / "xRossler.txt", sep="\t", header=None, index_col=None)
data1 = data1.values[:time_len]
data1 = data1.reshape(-1, 1)

dt = 0.3
t_r = np.linspace(0, (len(data1) - 1) * dt, len(data1))

ut_train1 = data1[:-p]
yt_train1 = data1[p:]

input_size = ut_train1.shape[-1]
output_size = yt_train1.shape[-1]
yt_train_effective = yt_train1[washout:]

eval_stop = min(washout + steps, len(yt_train1))
y_target = np.asarray(yt_train1[washout:eval_stop]).ravel()


###############################################################################
# Trials and m scan
###############################################################################

trials_noise = args.trials_noise
trials_esn = args.trials_esn
trials = trials_noise * trials_esn

np.random.seed(args.seed)
seed_noise = np.random.randint(0, 2000, size=trials_noise)
seed_esn = np.random.randint(0, 2000, size=trials_esn)

mm = np.arange(m_in, m_fin + 1, m_step, dtype=int)


###############################################################################
# Evaluate one fixed noise level
###############################################################################

def run_m_scan(noise_level):
    """Compute Without-C references and CTC results for one noise percentage."""

    # Without C does not depend on m: one value per ESN/noise trial.
    ssi_no_c = np.empty(trials, dtype=float)
    nrmse_no_c = np.empty(trials, dtype=float)

    # The conceptor curve contains C_noisy at m=1 and CTC at m>=2.
    ssi_conceptor = np.empty((len(mm), trials), dtype=float)
    nrmse_conceptor = np.empty((len(mm), trials), dtype=float)

    for esn_idx in range(trials_esn):
        params = rnn_params(N, input_size, output_size, scaling, spectral_radius, alpha, bias_scaling, sparsity, seed=seed_esn[esn_idx])

        # Clean reference states and noise amplitude.
        X_id = forward_rnn(params, ut_train1, 42, x_init=None, autonomous=False, conceptor=None)
        std_noise = std_noise_func(X_id, noise_level)

        #######################################################################
        # Without-C reference
        #######################################################################

        # Store each noisy trajectory because the m=1 point uses C_noisy,
        # computed directly from that same noise realization.
        X_noisy_trials = [None] * trials_noise

        for noise_idx in range(trials_noise):
            current_seed = seed_noise[noise_idx]
            trial_index = esn_idx * trials_noise + noise_idx

            X_noi = forward_rnn(params, ut_train1, current_seed, None, False, None, std_noise, corr=corr)
            X_noisy_trials[noise_idx] = X_noi

            X_effective = X_noi[washout:]
            params_trained_no_c, _ = ridge(reg, X_effective, yt_train_effective, step, params)

            ssi_no_c[trial_index] = xcorr_PCA(X_id, X_noi, washout, steps)

            y_no_c = np.asarray(X_noi[washout:eval_stop] @ params_trained_no_c["wout"].T + params_trained_no_c["bias_out"]).ravel()
            nrmse_no_c[trial_index] = NRMSE(y_target, y_no_c)

        #######################################################################
        # Conceptor scan: C_noisy for m=1 and CTC for m>=2
        #######################################################################

        for m_idx, m in enumerate(mm):
            # CTC depends on the number of independent realizations and is only
            # defined here for m >= 2. For m=1, a standard noisy conceptor is
            # computed separately from every noisy trial.
            C_ctc_m = None

            if m >= 2:
                C_ctc_m = denoising_CTC_m(params, ut_train1, std_noise, a, int(m), corr=corr)

            for noise_idx in range(trials_noise):
                current_seed = seed_noise[noise_idx]
                trial_index = esn_idx * trials_noise + noise_idx

                if m == 1:
                    C_current = compute_conceptor(X_noisy_trials[noise_idx], a)
                else:
                    C_current = C_ctc_m

                X_noi_C = forward_rnn(params, ut_train1, current_seed, None, False, C_current, std_noise, corr=corr)

                X_effective = X_noi_C[washout:]
                params_trained_C, _ = ridge(reg, X_effective, yt_train_effective, step, params)

                ssi_conceptor[m_idx, trial_index] = xcorr_PCA(X_id, X_noi_C, washout, steps)

                y_conceptor = np.asarray(X_noi_C[washout:eval_stop] @ params_trained_C["wout"].T + params_trained_C["bias_out"]).ravel()
                nrmse_conceptor[m_idx, trial_index] = NRMSE(y_target, y_conceptor)

    return {
        "noise": float(noise_level),
        "mean_ssi_no_c": float(np.mean(ssi_no_c)),
        "std_ssi_no_c": float(np.std(ssi_no_c)),
        "mean_ssi_conceptor": np.mean(ssi_conceptor, axis=1),
        "std_ssi_conceptor": np.std(ssi_conceptor, axis=1),
        "mean_nrmse_no_c": float(np.mean(nrmse_no_c)),
        "std_nrmse_no_c": float(np.std(nrmse_no_c)),
        "mean_nrmse_conceptor": np.mean(nrmse_conceptor, axis=1),
        "std_nrmse_conceptor": np.std(nrmse_conceptor, axis=1),
    }


###############################################################################
# Run individual noise level and 50%-100% comparison
###############################################################################

comparison_noise_levels = (50.0, 100.0)
required_noise_levels = [float(args.noise), *comparison_noise_levels]

# Avoid repeating a complete simulation when --noise is already 50 or 100.
results_cache = {}

for level in required_noise_levels:
    if level not in results_cache:
        results_cache[level] = run_m_scan(level)

single_result = results_cache[float(args.noise)]
result_50 = results_cache[50.0]
result_100 = results_cache[100.0]


###############################################################################
# Plot style
###############################################################################

plt.rcParams.update({"figure.figsize": (10, 6), "axes.labelsize": 20, "axes.titlesize": 20, "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 16, "lines.linewidth": 2, "lines.markersize": 9, "grid.alpha": 0.6, "grid.linestyle": "--", "axes.linewidth": 1.8, "axes.edgecolor": "black"})

c = "correlated" if corr else "uncorrelated"
noise_tag = f"{args.noise:g}"
a_tag = f"{a:g}"

# Show markers at every evaluated m value, but only a reduced number of
# equally spaced labels on the x axis.
max_x_ticks = 6
tick_step = max(1, int(np.ceil(len(mm) / max_x_ticks)))
tick_indices = np.arange(0, len(mm), tick_step, dtype=int)
visible_m_ticks = mm[tick_indices]


###############################################################################
# Individual figure 1: SSI versus m at --noise
###############################################################################

plt.figure(figsize=(8, 4), dpi=300)

plt.axhline(single_result["mean_ssi_no_c"], color="#B22222", linestyle="--", linewidth=2.2, alpha=0.9, label="Without C")
plt.errorbar(mm, single_result["mean_ssi_conceptor"], yerr=single_result["std_ssi_conceptor"], fmt="-o", color="#1F4E79", alpha=0.85, ecolor="black", elinewidth=1.8, capsize=5, label=r"$C_{ctc}$")

plt.xlabel(r"Number of realizations $m$")
plt.ylabel("PCA Subspace Similarity", size=19)
plt.xticks(visible_m_ticks)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

individual_ssi_name = f"PCAxcorr_Cnoisy_m1_CTC_mscan_noise{noise_tag}_N{N}_trials{trials}_steps{steps}_a{a_tag}_m{m_in}-{m_fin}_mstep{m_step}_traintime{time_len}_{c}"
plt.savefig(PLOTS_DIR / f"{individual_ssi_name}.png", dpi=300, bbox_inches="tight")
plt.savefig(PLOTS_DIR / f"{individual_ssi_name}.pdf", dpi=300, bbox_inches="tight")

plt.show()
plt.close()


###############################################################################
# Individual figure 2: NRMSE versus m at --noise
###############################################################################

plt.figure(figsize=(8, 4), dpi=300)

plt.axhline(single_result["mean_nrmse_no_c"], color="#B22222", linestyle="--", linewidth=2.2, alpha=0.9, label="Without C")
plt.errorbar(mm, single_result["mean_nrmse_conceptor"], yerr=single_result["std_nrmse_conceptor"], fmt="-o", color="#1F4E79", alpha=0.85, ecolor="black", elinewidth=1.8, capsize=5, label=r"$C_{ctc}$")

plt.xlabel(r"Number of realizations $m$")
plt.ylabel("NRMSE")
plt.xticks(visible_m_ticks)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

individual_nrmse_name = f"NRMSE_Cnoisy_m1_CTC_mscan_noise{noise_tag}_p{p}_N{N}_trials{trials}_steps{steps}_a{a_tag}_m{m_in}-{m_fin}_mstep{m_step}_traintime{time_len}_{c}"
plt.savefig(PLOTS_DIR / f"{individual_nrmse_name}.png", dpi=300, bbox_inches="tight")
plt.savefig(PLOTS_DIR / f"{individual_nrmse_name}.pdf", dpi=300, bbox_inches="tight")

plt.show()
plt.close()


###############################################################################
# Comparison figure 1: SSI at 50% and 100% noise
###############################################################################

plt.figure(figsize=(8, 4), dpi=300)

# Same color identifies the same noise level; dashed means Without C.
plt.axhline(result_50["mean_ssi_no_c"], color="#1F4E79", linestyle="--", linewidth=2.0, alpha=0.8, label=r"No $C$, 50%")
plt.axhline(result_100["mean_ssi_no_c"], color="#D55E00", linestyle="--", linewidth=2.0, alpha=0.8, label=r"No $C$, 100%")

plt.errorbar(mm, result_50["mean_ssi_conceptor"], yerr=result_50["std_ssi_conceptor"], fmt="-o", color="#1F4E79", alpha=0.9, ecolor="#1F4E79", elinewidth=1.6, capsize=4, label=r"$C_{ctc}$, 50%")
plt.errorbar(mm, result_100["mean_ssi_conceptor"], yerr=result_100["std_ssi_conceptor"], fmt="-s", color="#D55E00", alpha=0.9, ecolor="#D55E00", elinewidth=1.6, capsize=4, label=r"$C_{ctc}$, 100%")

plt.xlabel(r"Number of realizations $m$")
plt.ylabel("PCA Subspace Similarity", size=19)
plt.xticks(visible_m_ticks)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(ncol=2, fontsize=14, frameon=False, columnspacing=1.0, handletextpad=0.5)
plt.tight_layout()

comparison_ssi_name = f"PCAxcorr_Cnoisy_m1_CTC_mscan_noise50-100_comparison_N{N}_trials{trials}_steps{steps}_a{a_tag}_m{m_in}-{m_fin}_mstep{m_step}_traintime{time_len}_{c}"
plt.savefig(PLOTS_DIR / f"{comparison_ssi_name}.png", dpi=300, bbox_inches="tight")
plt.savefig(PLOTS_DIR / f"{comparison_ssi_name}.pdf", dpi=300, bbox_inches="tight")

plt.show()
plt.close()


###############################################################################
# Comparison figure 2: NRMSE at 50% and 100% noise
###############################################################################

plt.figure(figsize=(8, 4), dpi=300)

plt.axhline(result_50["mean_nrmse_no_c"], color="#1F4E79", linestyle="--", linewidth=2.0, alpha=0.8, label=r"No $C$, 50%")
plt.axhline(result_100["mean_nrmse_no_c"], color="#D55E00", linestyle="--", linewidth=2.0, alpha=0.8, label=r"No $C$, 100%")

plt.errorbar(mm, result_50["mean_nrmse_conceptor"], yerr=result_50["std_nrmse_conceptor"], fmt="-o", color="#1F4E79", alpha=0.9, ecolor="#1F4E79", elinewidth=1.6, capsize=4, label=r"$C_{ctc}$, 50%")
plt.errorbar(mm, result_100["mean_nrmse_conceptor"], yerr=result_100["std_nrmse_conceptor"], fmt="-s", color="#D55E00", alpha=0.9, ecolor="#D55E00", elinewidth=1.6, capsize=4, label=r"$C_{ctc}$, 100%")

plt.xlabel(r"Number of realizations $m$")
plt.ylabel("NRMSE")
plt.xticks(visible_m_ticks)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(ncol=2, fontsize=14, frameon=False, columnspacing=1.0, handletextpad=0.5)
plt.tight_layout()

comparison_nrmse_name = f"NRMSE_Cnoisy_m1_CTC_mscan_noise50-100_comparison_p{p}_N{N}_trials{trials}_steps{steps}_a{a_tag}_m{m_in}-{m_fin}_mstep{m_step}_traintime{time_len}_{c}"
plt.savefig(PLOTS_DIR / f"{comparison_nrmse_name}.png", dpi=300, bbox_inches="tight")
plt.savefig(PLOTS_DIR / f"{comparison_nrmse_name}.pdf", dpi=300, bbox_inches="tight")

plt.show()
plt.close()
