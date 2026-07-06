# Imports
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.rnn_utils import denoising_CTC_m
from utils.rnn_utils import forward_rnn
from utils.rnn_utils import ridge
from utils.rnn_utils import rnn_params
from utils.rnn_utils import std_noise_func
from utils.utils import NRMSE
from utils.utils import xcorr_PCA


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
parser.add_argument("--corr", type=lambda x: str(x).lower() in ["true", "1", "yes", "y"], default=False)

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
step = p

if a_step <= 0:
    raise ValueError("'a_step' must be larger than zero.")

if a_fin < a_in:
    raise ValueError("'a_fin' must be greater than or equal to 'a_in'.")

if m < 2:
    raise ValueError("'m' must be at least 2 for the CTC computation.")

if p < 1:
    raise ValueError("'p' must be at least 1.")


###############################################################################
# Input signal
###############################################################################

data1 = pd.read_csv("Rossler_data/xRossler.txt", sep="\t", header=None, index_col=None)
data1 = data1.values[:time_len].reshape(-1, 1)

ut_train1 = data1[:-p]
yt_train1 = data1[p:]

input_size = ut_train1.shape[-1]
output_size = yt_train1.shape[-1]
yt_train_effective = yt_train1[washout:]

eval_stop = min(washout + steps, len(yt_train1))
y_target = np.asarray(yt_train1[washout:eval_stop]).ravel()


###############################################################################
# Aperture scan
###############################################################################

trials_noise = args.trials_noise
trials_esn = args.trials_esn
trials = trials_noise * trials_esn

np.random.seed(args.seed)
seed_noise = np.random.randint(0, 2000, size=trials_noise)
seed_esn = np.random.randint(0, 2000, size=trials_esn)

aa = np.arange(float(a_in), float(a_fin) + 0.5 * float(a_step), float(a_step), dtype=float)

ssi_C_ctc = np.empty((len(aa), trials), dtype=float)
nrmse_C_ctc = np.empty((len(aa), trials), dtype=float)


for a_idx, a in enumerate(aa):
    a = float(a)

    for esn_idx in range(trials_esn):
        params = rnn_params(N, input_size, output_size, scaling, spectral_radius, alpha, bias_scaling, sparsity, seed=seed_esn[esn_idx])

        X_id = forward_rnn(params, ut_train1, 42, x_init=None, autonomous=False, conceptor=None)
        std_noise = std_noise_func(X_id, noise_level)

        C_ctc = denoising_CTC_m(params, ut_train1, std_noise, a, m, corr=corr)

        for noise_idx in range(trials_noise):
            current_seed = seed_noise[noise_idx]
            trial_index = esn_idx * trials_noise + noise_idx

            X_noi_C_ctc = forward_rnn(params, ut_train1, current_seed, None, False, C_ctc, std_noise, corr=corr)

            X_effective = X_noi_C_ctc[washout:]
            params_trained_CTC, _ = ridge(reg, X_effective, yt_train_effective, step, params)

            ssi_C_ctc[a_idx, trial_index] = xcorr_PCA(X_id, X_noi_C_ctc, washout, steps)

            y_C_ctc = np.asarray(X_noi_C_ctc[washout:eval_stop] @ params_trained_CTC["wout"].T + params_trained_CTC["bias_out"]).ravel()
            nrmse_C_ctc[a_idx, trial_index] = NRMSE(y_target, y_C_ctc)


###############################################################################
# Mean and standard deviation
###############################################################################

mean_ssi_C_ctc = np.mean(ssi_C_ctc, axis=1)
std_ssi_C_ctc = np.std(ssi_C_ctc, axis=1)

mean_nrmse_C_ctc = np.mean(nrmse_C_ctc, axis=1)
std_nrmse_C_ctc = np.std(nrmse_C_ctc, axis=1)


###############################################################################
# Plot style
###############################################################################

plt.rcParams.update({"figure.figsize": (10, 6), "axes.labelsize": 20, "axes.titlesize": 20, "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 18, "lines.linewidth": 2, "lines.markersize": 9, "grid.alpha": 0.6, "grid.linestyle": "--", "axes.linewidth": 1.8, "axes.edgecolor": "black"})

Path("plots").mkdir(parents=True, exist_ok=True)

c = "correlated" if corr else "uncorrelated"
noise_tag = f"{noise_level:g}"


###############################################################################
# SSI figure
###############################################################################

plt.figure(figsize=(8, 4), dpi=300)

plt.errorbar(aa, mean_ssi_C_ctc, yerr=std_ssi_C_ctc, fmt="-o", color="#1F4E79", alpha=0.85, ecolor="black", elinewidth=1.8, capsize=5, label=r"$C_{ctc}$")

plt.xlabel("Aperture")
plt.ylabel("PCA Subspace Similarity")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

ssi_name = f"plots/PCAxcorr_CTC_only_aperture_scan_noise{noise_tag}_N{N}_m{m}_trials{trials}_steps{steps}_a{a_in}-{a_fin}_astep{a_step}_traintime{time_len}_{c}"
plt.savefig(f"{ssi_name}.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{ssi_name}.pdf", dpi=300, bbox_inches="tight")

plt.show()
plt.close()


###############################################################################
# NRMSE figure
###############################################################################

plt.figure(figsize=(8, 4), dpi=300)

plt.errorbar(aa, mean_nrmse_C_ctc, yerr=std_nrmse_C_ctc, fmt="-o", color="#1F4E79", alpha=0.85, ecolor="black", elinewidth=1.8, capsize=5, label=r"$C_{ctc}$")

plt.xlabel("Aperture")
plt.ylabel("NRMSE")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

nrmse_name = f"plots/NRMSE_CTC_only_aperture_scan_noise{noise_tag}_p{p}_N{N}_m{m}_trials{trials}_steps{steps}_a{a_in}-{a_fin}_astep{a_step}_traintime{time_len}_{c}"
plt.savefig(f"{nrmse_name}.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{nrmse_name}.pdf", dpi=300, bbox_inches="tight")

plt.show()
plt.close()
