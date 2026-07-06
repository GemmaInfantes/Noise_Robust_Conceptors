# Imports
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils.rnn_utils import rnn_params
from utils.rnn_utils import forward_rnn
from utils.rnn_utils import ridge
from utils.rnn_utils import compute_conceptor
from utils.rnn_utils import std_noise_func
from utils.rnn_utils import denoising_CTC_m
from utils.rnn_utils import compute_conceptor_avg
from utils.utils import xcorr_PCA
from utils.utils import NRMSE


# Parameters that can be tuned from the terminal
parser = argparse.ArgumentParser()

parser.add_argument("--trials_noise", type=int, default=10, help="Number of different noise realizations.")
parser.add_argument("--trials_esn", type=int, default=10, help="Number of different ESN realizations.")
parser.add_argument("--noise", type=float, default=50.0, help="Fixed noise level used for the complete aperture scan (%%).")
parser.add_argument("--p", type=int, default=3, help="p-step ahead prediction")
parser.add_argument("--a_in", type=float, default=1.0, help="Initial conceptor aperture.")
parser.add_argument("--a_fin", type=float, default=25.0, help="Final conceptor aperture.")
parser.add_argument("--a_step", type=float, default=1.0, help="Step between consecutive aperture values.")
parser.add_argument("--steps", type=int, default=3000, help="Number of time steps used to compute the SSI.")
parser.add_argument("--steps_ol", type=int, default=30, help="Number of open-loop time steps.")
parser.add_argument("--time_len", type=int, default=3000, help="Number of time steps used to train the reservoir.")
parser.add_argument("--N", type=int, default=200, help="Number of reservoir neurons.")
parser.add_argument("--spectral_radius", type=float, default=1.6, help="Reservoir spectral radius.")
parser.add_argument("--scaling", type=float, default=0.9, help="Input scaling.")
parser.add_argument("--m", type=int, default=2, help="Number of realizations used to compute the CTC conceptor.")
parser.add_argument("--corr", type=lambda x: str(x).lower() in ["true", "1", "yes", "y"], default=False, help="True for correlated noise and False for uncorrelated noise.")

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
# step = 1
time_len = args.time_len
steps_ol = args.steps_ol
steps = args.steps
m = args.m
sparsity = None
corr = args.corr
noise_level = args.noise
p = args.p
step = p
if a_step <= 0:
    raise ValueError("'a_step' must be larger than zero.")

if a_fin < a_in:
    raise ValueError("'a_fin' must be greater than or equal to 'a_in'.")


###############################################################################
# Input signal: Rössler x
###############################################################################

data1 = pd.read_csv("Rossler_data/xRossler.txt", sep="\t", header=None, index_col=None)

data1 = data1.values[:time_len]
data1 = data1.reshape(-1, 1)

dt = 0.3
t_r = np.linspace(0, (len(data1) - 1) * dt, len(data1))

ut_train1 = data1[:-p]
yt_train1 = data1[p:]

input_size = ut_train1.shape[-1]
output_size = yt_train1.shape[-1]


###############################################################################
# Aperture scan at one fixed noise level
###############################################################################

seed1 = 20
trials_noise = args.trials_noise
trials_esn = args.trials_esn
trials = trials_noise * trials_esn

np.random.seed(seed1)
seed_noise = np.random.randint(0, 2000, size=trials_noise)
seed_esn = np.random.randint(0, 2000, size=trials_esn)

aa = np.arange(float(a_in), float(a_fin) + 0.5 * float(a_step), float(a_step), dtype=float)

# One row per aperture and one column per ESN/noise trial.
ssi_noi = np.empty((len(aa), trials), dtype=float)
ssi_noi_C_noi = np.empty((len(aa), trials), dtype=float)
ssi_noi_C_ctc_m = np.empty((len(aa), trials), dtype=float)
ssi_noi_C_id = np.empty((len(aa), trials), dtype=float)
ssi_noi_C_avg = np.empty((len(aa), trials), dtype=float)

# NRMSE of the p-step-ahead prediction:
# one row per aperture and one column per ESN/noise trial.
nrmse_noi = np.empty((len(aa), trials), dtype=float)
nrmse_noi_C_noi = np.empty((len(aa), trials), dtype=float)
nrmse_noi_C_ctc_m = np.empty((len(aa), trials), dtype=float)
nrmse_noi_C_id = np.empty((len(aa), trials), dtype=float)
nrmse_noi_C_avg = np.empty((len(aa), trials), dtype=float)


for a_idx, a in enumerate(aa):
    a = float(a)

    for esn_idx in range(trials_esn):
        params = rnn_params(N, input_size, output_size, scaling, spectral_radius, alpha, bias_scaling, sparsity, seed=seed_esn[esn_idx])

        # Clean reservoir states used as the reference subspace.
        X_id = forward_rnn(params, ut_train1, 42, x_init=None, autonomous=False, conceptor=None)

        # Ideal conceptor: computed only from noise-free reservoir states.
        C_id = compute_conceptor(X_id, a)

        # The same fixed noise percentage is used for every aperture.
        std_noise = std_noise_func(X_id, noise_level)

        # These conceptors depend on the ESN, aperture and fixed noise level,
        # but not on the evaluation noise seed.
        C_ctc_m = denoising_CTC_m(params, ut_train1, std_noise, a, m, corr=corr)

        C_avg = compute_conceptor_avg(params, ut_train1, std_noise, a, corr=corr)

        for noise_idx in range(trials_noise):
            current_seed = seed_noise[noise_idx]

            ###################################################################
            # No conceptor
            ###################################################################

            X_noi = forward_rnn(params, ut_train1, current_seed, None, False, None, std_noise, corr=corr)

            X_effective = X_noi[washout:]
            yt_train_effective = yt_train1[washout:]
            params_trained_noi, _ = ridge(reg, X_effective, yt_train_effective, step, params)

            ###################################################################
            # Noisy conceptor
            ###################################################################

            C_noi = compute_conceptor(X_noi, a)

            X_noi_C_noi = forward_rnn(params, ut_train1, current_seed, None, False, C_noi, std_noise, corr=corr)

            X_effective = X_noi_C_noi[washout:]
            params_trained_C_noi, _ = ridge(reg, X_effective, yt_train_effective, step, params)

            ###################################################################
            # CTC conceptor
            ###################################################################

            X_noi_C_ctc_m = forward_rnn(params, ut_train1, current_seed, None, False, C_ctc_m, std_noise, corr=corr)

            X_effective = X_noi_C_ctc_m[washout:]
            params_trained_CTC_m, _ = ridge(reg, X_effective, yt_train_effective, step, params)


            ###################################################################
            # Ideal conceptor
            ###################################################################

            # C_id is computed from X_id without noise, but is applied here
            # to the reservoir affected by the fixed noise level.
            X_noi_C_id = forward_rnn(params, ut_train1, current_seed, None, False, C_id, std_noise, corr=corr)

            X_effective = X_noi_C_id[washout:]
            params_trained_C_id, _ = ridge(reg, X_effective, yt_train_effective, step, params)

            ###################################################################
            # Average conceptor
            ###################################################################

            X_noi_C_avg = forward_rnn(params, ut_train1, current_seed, None, False, C_avg, std_noise, corr=corr)

            X_effective = X_noi_C_avg[washout:]
            params_trained_C_avg, _ = ridge(reg, X_effective, yt_train_effective, step, params)

            ###################################################################
            # PCA subspace similarity
            ###################################################################

            trial_index = esn_idx * trials_noise + noise_idx

            ssi_noi[a_idx, trial_index] = xcorr_PCA(X_id, X_noi, washout, steps)
            ssi_noi_C_noi[a_idx, trial_index] = xcorr_PCA(X_id, X_noi_C_noi, washout, steps)
            ssi_noi_C_ctc_m[a_idx, trial_index] = xcorr_PCA(X_id, X_noi_C_ctc_m, washout, steps)
            ssi_noi_C_id[a_idx, trial_index] = xcorr_PCA(X_id, X_noi_C_id, washout, steps)
            ssi_noi_C_avg[a_idx, trial_index] = xcorr_PCA(X_id, X_noi_C_avg, washout, steps)
            ###################################################################
            # NRMSE of the p-step-ahead prediction
            ###################################################################

            # All readouts were trained against yt_train1, which is already
            # shifted p samples with respect to ut_train1.
            eval_stop = min(washout + steps, len(yt_train1))

            y_target = np.asarray(yt_train1[washout:eval_stop]).ravel()

            y_noi = np.asarray(X_noi[washout:eval_stop] @ params_trained_noi["wout"].T + params_trained_noi["bias_out"]).ravel()

            y_noi_C_noi = np.asarray(X_noi_C_noi[washout:eval_stop] @ params_trained_C_noi["wout"].T + params_trained_C_noi["bias_out"]).ravel()

            y_noi_C_ctc_m = np.asarray(X_noi_C_ctc_m[washout:eval_stop] @ params_trained_CTC_m["wout"].T + params_trained_CTC_m["bias_out"]).ravel()

            y_noi_C_id = np.asarray(X_noi_C_id[washout:eval_stop] @ params_trained_C_id["wout"].T + params_trained_C_id["bias_out"]).ravel()

            y_noi_C_avg = np.asarray(X_noi_C_avg[washout:eval_stop] @ params_trained_C_avg["wout"].T + params_trained_C_avg["bias_out"]).ravel()

            nrmse_noi[a_idx, trial_index] = NRMSE(y_target, y_noi)
            nrmse_noi_C_noi[a_idx, trial_index] = NRMSE(y_target, y_noi_C_noi)
            nrmse_noi_C_ctc_m[a_idx, trial_index] = NRMSE(y_target, y_noi_C_ctc_m)
            nrmse_noi_C_id[a_idx, trial_index] = NRMSE(y_target, y_noi_C_id)
            nrmse_noi_C_avg[a_idx, trial_index] = NRMSE(y_target, y_noi_C_avg)


###############################################################################
# Mean and standard deviation across trials at noise = 50% by default
###############################################################################

mssi_noi = np.mean(ssi_noi, axis=1)
mssi_noi_C_noi = np.mean(ssi_noi_C_noi, axis=1)
mssi_noi_C_ctc_m = np.mean(ssi_noi_C_ctc_m, axis=1)
mssi_noi_C_id = np.mean(ssi_noi_C_id, axis=1)
mssi_noi_C_avg = np.mean(ssi_noi_C_avg, axis=1)

sssi_noi = np.std(ssi_noi, axis=1)
sssi_noi_C_noi = np.std(ssi_noi_C_noi, axis=1)
sssi_noi_C_ctc_m = np.std(ssi_noi_C_ctc_m, axis=1)
sssi_noi_C_id = np.std(ssi_noi_C_id, axis=1)
sssi_noi_C_avg = np.std(ssi_noi_C_avg, axis=1)

# NRMSE mean across ESN and noise realizations for every aperture.
mean_nrmse_noi = np.mean(nrmse_noi, axis=1)
mean_nrmse_C_noi = np.mean(nrmse_noi_C_noi, axis=1)
mean_nrmse_C_ctc_m = np.mean(nrmse_noi_C_ctc_m, axis=1)
mean_nrmse_C_id = np.mean(nrmse_noi_C_id, axis=1)
mean_nrmse_C_avg = np.mean(nrmse_noi_C_avg, axis=1)

# NRMSE standard deviation across the same trials.
std_nrmse_noi = np.std(nrmse_noi, axis=1)
std_nrmse_C_noi = np.std(nrmse_noi_C_noi, axis=1)
std_nrmse_C_ctc_m = np.std(nrmse_noi_C_ctc_m, axis=1)
std_nrmse_C_id = np.std(nrmse_noi_C_id, axis=1)
std_nrmse_C_avg = np.std(nrmse_noi_C_avg, axis=1)


###############################################################################
# Plot style
###############################################################################

plt.rcParams.update({"figure.figsize": (10, 6), "axes.labelsize": 20, "axes.titlesize": 20, "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 18, "lines.linewidth": 2, "lines.markersize": 10, "grid.alpha": 0.6, "grid.linestyle": "--", "axes.linewidth": 1.8, "axes.edgecolor": "black"})

c = "correlated" if corr else "uncorrelated"
noise_tag = f"{noise_level:g}"


###############################################################################
# Figure 1: Without C, C_noisy, C_ctc and C_ideal
###############################################################################

plt.figure(figsize=(8, 4), dpi=300)

plt.errorbar(aa, mssi_noi, yerr=sssi_noi, fmt="s", color="#B22222", alpha=0.8, ecolor="black", elinewidth=2, capsize=6, label="Without C")

plt.errorbar(aa, mssi_noi_C_noi, yerr=sssi_noi_C_noi, fmt="^", color="#6BAED6", alpha=0.8, ecolor="black", elinewidth=2, capsize=6, label=r"With $C_{noisy}$")

plt.errorbar(aa, mssi_noi_C_ctc_m, yerr=sssi_noi_C_ctc_m, fmt="o", color="#1F4E79", alpha=0.8, ecolor="black", elinewidth=2, capsize=6, label=r"With $C_{ctc}$")

plt.errorbar(aa, mssi_noi_C_id, yerr=sssi_noi_C_id, fmt="p", color="#D4A017", alpha=0.9, markersize=9, ecolor="black", elinewidth=2, capsize=6, label=r"With $C_{ideal}$")

plt.xlabel("Aperture")
plt.ylabel("PCA Subspace Similarity")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

plt.savefig(f"plots/PCAxcorr_CTCm_ideal_aperture_scan_noise{noise_tag}" f"_N{N}_m{m}_trials{trials}_steps{steps}" f"_a{a_in}-{a_fin}_astep{a_step}" f"_traintime{time_len}_{c}.png", dpi=300, bbox_inches="tight")

plt.savefig(f"plots/PCAxcorr_CTCm_ideal_aperture_scan_noise{noise_tag}" f"_N{N}_m{m}_trials{trials}_steps{steps}" f"_a{a_in}-{a_fin}_astep{a_step}" f"_traintime{time_len}_{c}.pdf", dpi=300, bbox_inches="tight")

plt.show()
plt.close()


###############################################################################
# Figure 2: Including C_avg and C_ideal
###############################################################################

plt.figure(figsize=(8, 4), dpi=300)

plt.errorbar(aa, mssi_noi, yerr=sssi_noi, fmt="s", color="#B22222", alpha=0.8, ecolor="black", elinewidth=2, capsize=6, label="Without C")

plt.errorbar(aa, mssi_noi_C_noi, yerr=sssi_noi_C_noi, fmt="^", color="#6BAED6", alpha=0.8, ecolor="black", elinewidth=2, capsize=6, label=r"With $C_{noisy}$")

plt.errorbar(aa, mssi_noi_C_ctc_m, yerr=sssi_noi_C_ctc_m, fmt="o", color="#1F4E79", alpha=0.8, ecolor="black", elinewidth=2, capsize=6, label=r"With $C_{ctc}$")

plt.errorbar(aa, mssi_noi_C_id, yerr=sssi_noi_C_id, fmt="p", color="#D4A017", alpha=0.9, markersize=9, ecolor="black", elinewidth=2, capsize=6, label=r"With $C_{ideal}$")

plt.errorbar(aa, mssi_noi_C_avg, yerr=sssi_noi_C_avg, fmt="D", color="#009E9A", alpha=0.8, markersize=8, ecolor="black", elinewidth=2, capsize=6, label=r"With $C_{avg}$")

plt.xlabel("Aperture")
plt.ylabel("PCA Subspace Similarity")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

plt.savefig(f"plots/PCAxcorr_Cavg_ideal_aperture_scan_noise{noise_tag}" f"_N{N}_m{m}_trials{trials}_steps{steps}" f"_a{a_in}-{a_fin}_astep{a_step}" f"_traintime{time_len}_{c}.png", dpi=300, bbox_inches="tight")

plt.savefig(f"plots/PCAxcorr_Cavg_ideal_aperture_scan_noise{noise_tag}" f"_N{N}_m{m}_trials{trials}_steps{steps}" f"_a{a_in}-{a_fin}_astep{a_step}" f"_traintime{time_len}_{c}.pdf", dpi=300, bbox_inches="tight")

plt.show()
plt.close()
###############################################################################
# Figure 3: NRMSE aperture scan
# Without C, C_noisy, C_ctc and C_ideal
###############################################################################

plt.figure(figsize=(8, 4), dpi=300)

plt.errorbar(aa, mean_nrmse_noi, yerr=std_nrmse_noi, fmt="s", color="#B22222", alpha=0.8, ecolor="black", elinewidth=2, capsize=6, label="Without C")

plt.errorbar(aa, mean_nrmse_C_noi, yerr=std_nrmse_C_noi, fmt="^", color="#6BAED6", alpha=0.8, ecolor="black", elinewidth=2, capsize=6, label=r"With $C_{noisy}$")

plt.errorbar(aa, mean_nrmse_C_ctc_m, yerr=std_nrmse_C_ctc_m, fmt="o", color="#1F4E79", alpha=0.8, ecolor="black", elinewidth=2, capsize=6, label=r"With $C_{ctc}$")

plt.errorbar(aa, mean_nrmse_C_id, yerr=std_nrmse_C_id, fmt="p", color="#D4A017", alpha=0.9, markersize=9, ecolor="black", elinewidth=2, capsize=6, label=r"With $C_{ideal}$")

plt.xlabel("Aperture")
plt.ylabel("NRMSE")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

plt.savefig(f"plots/NRMSE_CTCm_ideal_aperture_scan_noise{noise_tag}" f"_p{p}_N{N}_m{m}_trials{trials}_steps{steps}" f"_a{a_in}-{a_fin}_astep{a_step}" f"_traintime{time_len}_{c}.png", dpi=300, bbox_inches="tight")

plt.savefig(f"plots/NRMSE_CTCm_ideal_aperture_scan_noise{noise_tag}" f"_p{p}_N{N}_m{m}_trials{trials}_steps{steps}" f"_a{a_in}-{a_fin}_astep{a_step}" f"_traintime{time_len}_{c}.pdf", dpi=300, bbox_inches="tight")

plt.show()
plt.close()


###############################################################################
# Figure 4: NRMSE aperture scan including C_avg
###############################################################################

plt.figure(figsize=(8, 4), dpi=300)

plt.errorbar(aa, mean_nrmse_noi, yerr=std_nrmse_noi, fmt="s", color="#B22222", alpha=0.8, ecolor="black", elinewidth=2, capsize=6, label="Without C")

plt.errorbar(aa, mean_nrmse_C_noi, yerr=std_nrmse_C_noi, fmt="^", color="#6BAED6", alpha=0.8, ecolor="black", elinewidth=2, capsize=6, label=r"With $C_{noisy}$")

plt.errorbar(aa, mean_nrmse_C_ctc_m, yerr=std_nrmse_C_ctc_m, fmt="o", color="#1F4E79", alpha=0.8, ecolor="black", elinewidth=2, capsize=6, label=r"With $C_{ctc}$")

plt.errorbar(aa, mean_nrmse_C_id, yerr=std_nrmse_C_id, fmt="p", color="#D4A017", alpha=0.9, markersize=9, ecolor="black", elinewidth=2, capsize=6, label=r"With $C_{ideal}$")

plt.errorbar(aa, mean_nrmse_C_avg, yerr=std_nrmse_C_avg, fmt="D", color="#009E9A", alpha=0.8, markersize=8, ecolor="black", elinewidth=2, capsize=6, label=r"With $C_{avg}$")

plt.xlabel("Aperture")
plt.ylabel("NRMSE")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

plt.savefig(f"plots/NRMSE_Cavg_ideal_aperture_scan_noise{noise_tag}" f"_p{p}_N{N}_m{m}_trials{trials}_steps{steps}" f"_a{a_in}-{a_fin}_astep{a_step}" f"_traintime{time_len}_{c}.png", dpi=300, bbox_inches="tight")

plt.savefig(f"plots/NRMSE_Cavg_ideal_aperture_scan_noise{noise_tag}" f"_p{p}_N{N}_m{m}_trials{trials}_steps{steps}" f"_a{a_in}-{a_fin}_astep{a_step}" f"_traintime{time_len}_{c}.pdf", dpi=300, bbox_inches="tight")

plt.show()
plt.close()
