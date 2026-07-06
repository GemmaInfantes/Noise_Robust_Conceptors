# Imports
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from utils.rnn_utils import denoising_CTC_m
from utils.rnn_utils import forward_rnn
from utils.rnn_utils import forward_rnn_comb
from utils.rnn_utils import ridge
from utils.rnn_utils import rnn_params
from utils.rnn_utils import std_noise_func
from utils.utils import prediction_horizon


# Parameters that can be tuned from the terminal
parser = argparse.ArgumentParser()

parser.add_argument("--trials_noise", type=int, default=10, help="Number of different noise realizations.")
parser.add_argument("--trials_esn", type=int, default=10, help="Number of different ESN realizations.")
parser.add_argument("--noise", type=float, default=50.0, help="Fixed noise level used in the sensitivity analysis (%%).")
parser.add_argument("--a", type=float, default=5.0, help="Conceptor aperture.")
parser.add_argument("--steps_ol", type=int, default=100, help="Number of teacher-forced open-loop steps before autonomous prediction.")
parser.add_argument("--time_len", type=int, default=3000, help="Number of time steps used to train and evaluate the reservoir.")
parser.add_argument("--error_th", type=float, default=0.5, help="Reference error threshold used to determine the prediction horizon.")
parser.add_argument("--steps_th", type=int, default=20, help="Reference number of consecutive steps above the error threshold.")
parser.add_argument("--window", type=int, default=50, help="Reference window used to compute the prediction-horizon error.")
parser.add_argument("--N", type=int, default=200, help="Number of reservoir neurons.")
parser.add_argument("--spectral_radius", type=float, default=1.6, help="Reservoir spectral radius.")
parser.add_argument("--scaling", type=float, default=0.9, help="Input scaling.")
parser.add_argument("--m", type=int, default=2, help="Number of realizations used to compute the CTC conceptor.")
parser.add_argument("--seed", type=int, default=20, help="Seed used to generate the ESN and noise seeds.")
parser.add_argument("--corr", type=lambda x: str(x).lower() in ["true", "1", "yes", "y"], default=False, help="True for correlated noise and False for uncorrelated noise.")
parser.add_argument("--mode", type=str, choices=["error", "window", "steps"], default="error", help="Prediction-horizon parameter used in the sensitivity analysis.")
parser.add_argument("--var", type=float, default=0.7, help="Lower multiplicative factor. The upper factor is computed as 2-var.")

args = parser.parse_args()


###############################################################################
# ESN parameters
###############################################################################

spectral_radius = args.spectral_radius
scaling = args.scaling
bias_scaling = 0.4
alpha = 0.75

a = args.a
N = args.N
washout = 20
reg = 1
step = 1

time_len = args.time_len
steps_ol = args.steps_ol
m = args.m
sparsity = None
corr = args.corr
noise_level = args.noise

error_th = args.error_th
steps_th = args.steps_th
window = args.window
mode = args.mode
var = args.var

if not 0 < var < 2:
    raise ValueError("'var' must be between 0 and 2.")

if steps_ol <= 0:
    raise ValueError("'steps_ol' must be larger than zero.")

if steps_ol >= time_len - step:
    raise ValueError("'steps_ol' must be smaller than the available time series after shifting.")


###############################################################################
# Input signal: Rössler x
###############################################################################

data1 = pd.read_csv("Rossler_data/xRossler.txt", sep="\t", header=None, index_col=None)
data1 = data1.values[:time_len]
data1 = data1.reshape(-1, 1)

dt = 0.3
lyap = 0.0714 * dt

ut_train1 = data1[:-step]
yt_train1 = data1[step:]

input_size = ut_train1.shape[-1]
output_size = yt_train1.shape[-1]


###############################################################################
# Values used in the sensitivity analysis
###############################################################################

variation_factors = np.array([var, 1.0, 2.0 - var], dtype=float)

if mode == "error":
    reference_value = float(error_th)
    studied_values = reference_value * variation_factors
    x_parameter_label = "error threshold"
elif mode == "window":
    reference_value = int(window)
    studied_values = np.rint(reference_value * variation_factors).astype(int)
    studied_values = np.maximum(studied_values, 1)
    x_parameter_label = "window"
else:
    reference_value = int(steps_th)
    studied_values = np.rint(reference_value * variation_factors).astype(int)
    studied_values = np.maximum(studied_values, 1)
    x_parameter_label = "steps threshold"

variation_percentages = 100.0 * (variation_factors - 1.0)

tick_labels = []
for percentage, value in zip(variation_percentages, studied_values):
    percentage_label = "0%" if np.isclose(percentage, 0.0) else f"{percentage:+.0f}%"
    value_label = f"{value:g}" if mode == "error" else f"{int(value)}"
    tick_labels.append(f"{percentage_label}\n({value_label})")


###############################################################################
# Storage
###############################################################################

trials_noise = args.trials_noise
trials_esn = args.trials_esn
trials = trials_noise * trials_esn

np.random.seed(args.seed)
seed_noise = np.random.randint(0, 2000, size=trials_noise)
seed_esn = np.random.randint(0, 2000, size=trials_esn)

# One row per sensitivity value and one column per ESN/noise realization.
ph_noi = np.full((len(studied_values), trials), np.nan, dtype=float)
ph_noi_C_ctc_m = np.full((len(studied_values), trials), np.nan, dtype=float)


###############################################################################
# Reservoir simulations
###############################################################################

for esn_idx in range(trials_esn):
    params = rnn_params(N, input_size, output_size, scaling, spectral_radius, alpha, bias_scaling, sparsity, seed=seed_esn[esn_idx])

    # Clean states are used only to define the noise amplitude.
    X_id = forward_rnn(params, ut_train1, 42, x_init=None, autonomous=False, conceptor=None)
    std_noise = std_noise_func(X_id, noise_level)

    # CTC does not depend on the prediction-horizon sensitivity parameter.
    C_ctc_m = denoising_CTC_m(params, ut_train1, std_noise, a, m, corr=corr)

    for noise_idx in range(trials_noise):
        current_seed = seed_noise[noise_idx]
        trial_index = esn_idx * trials_noise + noise_idx

        #######################################################################
        # Open loop and readout training: without conceptor
        #######################################################################

        X_noi_ol = forward_rnn(params, ut_train1, current_seed, None, False, None, std_noise, corr=corr)
        X_effective = X_noi_ol[washout:]
        yt_train_effective = yt_train1[washout:]
        params_trained_noi, _ = ridge(reg, X_effective, yt_train_effective, step, params)

        #######################################################################
        # Open loop and readout training: CTC conceptor
        #######################################################################

        X_noi_C_ctc_m_ol = forward_rnn(params, ut_train1, current_seed, None, False, C_ctc_m, std_noise, corr=corr)
        X_effective = X_noi_C_ctc_m_ol[washout:]
        params_trained_CTC_m, _ = ridge(reg, X_effective, yt_train_effective, step, params)

        #######################################################################
        # Autonomous mode
        #######################################################################

        X_noi = forward_rnn_comb(params_trained_noi, ut_train1, current_seed, steps_ol, None, None, std_noise, corr=corr)
        X_noi_C_ctc_m = forward_rnn_comb(params_trained_CTC_m, ut_train1, current_seed, steps_ol, None, C_ctc_m, std_noise, corr=corr)

        y = np.asarray(yt_train1[steps_ol:]).ravel()
        y_noi = np.asarray(X_noi[steps_ol:] @ params_trained_noi["wout"].T + params_trained_noi["bias_out"]).ravel()
        y_noi_C_ctc_m = np.asarray(X_noi_C_ctc_m[steps_ol:] @ params_trained_CTC_m["wout"].T + params_trained_CTC_m["bias_out"]).ravel()

        eval_len = min(len(y), len(y_noi), len(y_noi_C_ctc_m))
        y = y[:eval_len]
        y_noi = y_noi[:eval_len]
        y_noi_C_ctc_m = y_noi_C_ctc_m[:eval_len]

        #######################################################################
        # Prediction horizon for the three sensitivity values
        #######################################################################

        for sensitivity_idx, studied_value in enumerate(studied_values):
            if mode == "error":
                current_error_th = float(studied_value)
                current_window = window
                current_steps_th = steps_th
            elif mode == "window":
                current_error_th = error_th
                current_window = int(studied_value)
                current_steps_th = steps_th
            else:
                current_error_th = error_th
                current_window = window
                current_steps_th = int(studied_value)

            _, ph_noi[sensitivity_idx, trial_index] = prediction_horizon(y, y_noi, current_window, None, 0, current_error_th, current_steps_th, lyap)
            _, ph_noi_C_ctc_m[sensitivity_idx, trial_index] = prediction_horizon(y, y_noi_C_ctc_m, current_window, None, 0, current_error_th, current_steps_th, lyap)


###############################################################################
# Grouped boxplots: Without C and CTC
###############################################################################

plt.rcParams.update({"figure.figsize": (8, 4.5), "axes.labelsize": 20, "axes.titlesize": 20, "xtick.labelsize": 16, "ytick.labelsize": 18, "legend.fontsize": 17, "axes.linewidth": 1.8, "axes.edgecolor": "black"})

color_without_c = "#B22222"
color_ctc = "#1F4E79"

centers = np.arange(1, len(studied_values) + 1, dtype=float)
offset = 0.18
box_width = 0.30

data_without_c = [ph_noi[idx, np.isfinite(ph_noi[idx])] for idx in range(len(studied_values))]
data_ctc = [ph_noi_C_ctc_m[idx, np.isfinite(ph_noi_C_ctc_m[idx])] for idx in range(len(studied_values))]

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

box_without_c = ax.boxplot(data_without_c, positions=centers - offset, widths=box_width, patch_artist=True, showmeans=False, boxprops={"linewidth": 1.8, "edgecolor": "black"}, whiskerprops={"linewidth": 1.6, "color": "black"}, capprops={"linewidth": 1.6, "color": "black"}, medianprops={"linewidth": 2.0, "color": "black"}, meanprops={"marker": "s", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 6}, flierprops={"marker": "o", "markerfacecolor": color_without_c, "markeredgecolor": "black", "markersize": 4, "alpha": 0.6})

box_ctc = ax.boxplot(data_ctc, positions=centers + offset, widths=box_width, patch_artist=True, showmeans=False, boxprops={"linewidth": 1.8, "edgecolor": "black"}, whiskerprops={"linewidth": 1.6, "color": "black"}, capprops={"linewidth": 1.6, "color": "black"}, medianprops={"linewidth": 2.0, "color": "black"}, meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 6}, flierprops={"marker": "o", "markerfacecolor": color_ctc, "markeredgecolor": "black", "markersize": 4, "alpha": 0.6})

for box in box_without_c["boxes"]:
    box.set_facecolor(color_without_c)
    box.set_alpha(0.8)

for box in box_ctc["boxes"]:
    box.set_facecolor(color_ctc)
    box.set_alpha(0.8)

ax.set_xticks(centers)
ax.set_xticklabels(tick_labels)
ax.set_xlabel(f"Variation in {x_parameter_label} (%)")
ax.set_ylabel("Prediction horizon")
ax.set_xlim(0.45, len(studied_values) + 0.55)
ax.grid(True, axis="y", linestyle="--", alpha=0.6)
ax.legend(handles=[Patch(facecolor=color_without_c, edgecolor="black", alpha=0.8, label="Without C"), Patch(facecolor=color_ctc, edgecolor="black", alpha=0.8, label=r"With $C_{ctc}$")])
fig.tight_layout()

Path("plots").mkdir(parents=True, exist_ok=True)

c = "correlated" if corr else "uncorrelated"
noise_tag = f"{noise_level:g}"
var_tag = f"{var:g}"
base_filename = f"plots/PH_sensitivity_{mode}_noise{noise_tag}_var{var_tag}_N{N}_m{m}_trials{trials}_a{a:g}_ol{steps_ol}_traintime{time_len}_{c}"

fig.savefig(f"{base_filename}.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{base_filename}.pdf", dpi=300, bbox_inches="tight")

plt.show()
plt.close()
