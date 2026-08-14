# Imports
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


###############################################################################
# Project paths
###############################################################################

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "Rossler_data"
PLOTS_DIR = SCRIPT_DIR / "plots_analysis"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from utils.rnn_utils import denoising_CTC_m
from utils.rnn_utils import forward_rnn
from utils.rnn_utils import forward_rnn_comb
from utils.rnn_utils import ridge
from utils.rnn_utils import rnn_params
from utils.rnn_utils import std_noise_func
from utils.utils import prediction_horizon


###############################################################################
# Parameters
###############################################################################

parser = argparse.ArgumentParser()

parser.add_argument("--trials_noise", type=int, default=10, help="Number of different noise realizations.")
parser.add_argument("--trials_esn", type=int, default=10, help="Number of different ESN realizations.")
parser.add_argument("--noise", type=float, default=30.0, help="Fixed noise level used in the sensitivity analysis (%%).")
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
var = args.var

trials_noise = args.trials_noise
trials_esn = args.trials_esn
trials = trials_noise * trials_esn


###############################################################################
# Validation
###############################################################################

if not 0 < var < 2:
    raise ValueError("'var' must be between 0 and 2.")

if trials_noise < 1 or trials_esn < 1:
    raise ValueError("'trials_noise' and 'trials_esn' must be at least 1.")

if m < 2:
    raise ValueError("'m' must be at least 2.")

if steps_ol <= 0:
    raise ValueError("'steps_ol' must be larger than zero.")

if steps_ol >= time_len - step:
    raise ValueError("'steps_ol' must be smaller than the available time series after shifting.")


###############################################################################
# Input signal: Rössler x
###############################################################################

data1 = pd.read_csv(DATA_DIR / "xRossler.txt", sep="\t", header=None, index_col=None)
data1 = data1.values[:time_len].reshape(-1, 1)

dt = 0.3
lyap = 0.0714 * dt

ut_train1 = data1[:-step]
yt_train1 = data1[step:]

input_size = ut_train1.shape[-1]
output_size = yt_train1.shape[-1]


###############################################################################
# Sensitivity values
###############################################################################

variation_factors = np.array([var, 1.0, 2.0 - var], dtype=float)
variation_percentages = 100.0 * (variation_factors - 1.0)


def build_studied_values(mode):
    """
    Builds the tested values, tick labels, and title for one sensitivity parameter.

    Args:
    - mode (str): Parameter to vary; must be ``error``, ``window``, or ``steps``.

    Returns:
    - studied_values (ndarray): Parameter values generated from the configured variation factors.
    - tick_labels (list of str): Plot labels showing relative and absolute parameter values.
    - parameter_label (str): Human-readable parameter name used as the panel title.
    """

    if mode == "error":
        reference_value = float(error_th)
        studied_values = reference_value * variation_factors
        parameter_label = "Error threshold"

    elif mode == "window":
        reference_value = int(window)
        studied_values = np.rint(reference_value * variation_factors).astype(int)
        studied_values = np.maximum(studied_values, 1)
        parameter_label = "Window"

    elif mode == "steps":
        reference_value = int(steps_th)
        studied_values = np.rint(reference_value * variation_factors).astype(int)
        studied_values = np.maximum(studied_values, 1)
        parameter_label = "Steps threshold"

    else:
        raise ValueError(f"Unknown sensitivity mode: {mode}")

    tick_labels = []

    for percentage, value in zip(variation_percentages, studied_values):
        percentage_label = "0%" if np.isclose(percentage, 0.0) else f"{percentage:+.0f}%"
        value_label = f"{float(value):g}" if mode == "error" else f"{int(value)}"
        tick_labels.append(f"{percentage_label}\n({value_label})")

    return studied_values, tick_labels, parameter_label


modes = ["error", "window", "steps"]
sensitivity_config = {mode: build_studied_values(mode) for mode in modes}


###############################################################################
# Storage
###############################################################################

np.random.seed(args.seed)

seed_noise = np.random.randint(0, 2000, size=trials_noise)
seed_esn = np.random.randint(0, 2000, size=trials_esn)

ph_noi = {mode: np.full((3, trials), np.nan, dtype=float) for mode in modes}
ph_ctc = {mode: np.full((3, trials), np.nan, dtype=float) for mode in modes}


###############################################################################
# Reservoir simulations
###############################################################################

for esn_idx in range(trials_esn):
    params = rnn_params(N, input_size, output_size, scaling, spectral_radius, alpha, bias_scaling, sparsity, seed=seed_esn[esn_idx])

    X_id = forward_rnn(params, ut_train1, 42, x_init=None, autonomous=False, conceptor=None)
    std_noise = std_noise_func(X_id, noise_level)

    C_ctc_m = denoising_CTC_m(params, ut_train1, std_noise, a, m, corr=corr)

    for noise_idx in range(trials_noise):
        current_seed = seed_noise[noise_idx]
        trial_index = esn_idx * trials_noise + noise_idx

        #######################################################################
        # Open loop and readout training
        #######################################################################

        X_noi_ol = forward_rnn(params, ut_train1, current_seed, None, False, None, std_noise, corr=corr)
        X_effective = X_noi_ol[washout:]
        yt_train_effective = yt_train1[washout:]
        params_trained_noi, _ = ridge(reg, X_effective, yt_train_effective, step, params)

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
        # Prediction-horizon sensitivity
        #######################################################################

        for mode in modes:
            studied_values, _, _ = sensitivity_config[mode]

            for sensitivity_idx, studied_value in enumerate(studied_values):
                if mode == "error":
                    current_error_th = float(studied_value)
                    current_window = int(window)
                    current_steps_th = int(steps_th)

                elif mode == "window":
                    current_error_th = float(error_th)
                    current_window = int(studied_value)
                    current_steps_th = int(steps_th)

                else:
                    current_error_th = float(error_th)
                    current_window = int(window)
                    current_steps_th = int(studied_value)

                _, ph_noi[mode][sensitivity_idx, trial_index] = prediction_horizon(y, y_noi, current_window, None, 0, current_error_th, current_steps_th, lyap)
                _, ph_ctc[mode][sensitivity_idx, trial_index] = prediction_horizon(y, y_noi_C_ctc_m, current_window, None, 0, current_error_th, current_steps_th, lyap)


###############################################################################
# Figure style
###############################################################################

plt.rcParams.update({"figure.figsize": (15, 4.8), "axes.labelsize": 18, "axes.titlesize": 19, "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 17, "axes.linewidth": 1.6, "axes.edgecolor": "black"})

color_without_c = "#B22222"
color_ctc = "#1F4E79"

centers = np.arange(1, 4, dtype=float)
offset = 0.18
box_width = 0.30


###############################################################################
# Helper for grouped boxplots
###############################################################################

def add_grouped_boxplot(ax, data_without_c, data_ctc, tick_labels, title):
    """
    Adds a grouped Without-C/CTC boxplot to one Matplotlib panel.

    Args:
    - ax (matplotlib.axes.Axes): Axis on which to draw the boxplots.
    - data_without_c (sequence): Without-C samples for each parameter value.
    - data_ctc (sequence): CTC samples for each parameter value.
    - tick_labels (sequence of str): Labels displayed at the parameter positions.
    - title (str): Panel title.

    Returns:
    - None
    """

    box_without_c = ax.boxplot(data_without_c, positions=centers - offset, widths=box_width, patch_artist=True, showmeans=False, boxprops={"linewidth": 1.4, "edgecolor": "black"}, whiskerprops={"linewidth": 1.3, "color": "black"}, capprops={"linewidth": 1.3, "color": "black"}, medianprops={"linewidth": 1.7, "color": "black"}, flierprops={"marker": "o", "markerfacecolor": color_without_c, "markeredgecolor": "black", "markersize": 3, "alpha": 0.55})

    box_ctc = ax.boxplot(data_ctc, positions=centers + offset, widths=box_width, patch_artist=True, showmeans=False, boxprops={"linewidth": 1.4, "edgecolor": "black"}, whiskerprops={"linewidth": 1.3, "color": "black"}, capprops={"linewidth": 1.3, "color": "black"}, medianprops={"linewidth": 1.7, "color": "black"}, flierprops={"marker": "o", "markerfacecolor": color_ctc, "markeredgecolor": "black", "markersize": 3, "alpha": 0.55})

    for box in box_without_c["boxes"]:
        box.set_facecolor(color_without_c)
        box.set_alpha(0.8)

    for box in box_ctc["boxes"]:
        box.set_facecolor(color_ctc)
        box.set_alpha(0.8)

    ax.set_xticks(centers)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Parameter variation")
    ax.set_title(title)
    ax.set_xlim(0.45, 3.55)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)


###############################################################################
# Combined 1x3 figure
###############################################################################

fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300, sharey=True)

for panel_idx, mode in enumerate(modes):
    _, tick_labels, panel_title = sensitivity_config[mode]

    data_without_c = [ph_noi[mode][idx, np.isfinite(ph_noi[mode][idx])] for idx in range(3)]
    data_ctc = [ph_ctc[mode][idx, np.isfinite(ph_ctc[mode][idx])] for idx in range(3)]

    add_grouped_boxplot(axs[panel_idx], data_without_c, data_ctc, tick_labels, panel_title)

axs[0].set_ylabel("Prediction horizon")

legend_handles = [
    Patch(facecolor=color_without_c, edgecolor="black", alpha=0.8, label="Without C"),
    Patch(facecolor=color_ctc, edgecolor="black", alpha=0.8, label=r"With $C_{ctc}$"),
]

fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06))
fig.tight_layout(rect=[0, 0, 1, 0.89], w_pad=2.0)


###############################################################################
# Save figure
###############################################################################

c = "correlated" if corr else "uncorrelated"
noise_tag = f"{noise_level:g}"
var_tag = f"{var:g}"

base_filename = f"PH_sens_1x3_n{noise_tag}_v{var_tag}_N{N}_m{m}_T{trials}_a{a:g}_{c}"

fig.savefig(PLOTS_DIR / f"{base_filename}.png", dpi=300, bbox_inches="tight")
fig.savefig(PLOTS_DIR / f"{base_filename}.pdf", dpi=300, bbox_inches="tight")

plt.show()
plt.close()
