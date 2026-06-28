#Imports
import numpy as np
import matplotlib.pyplot as plt
import argparse
from utils.rnn_utils import rnn_params
from utils.rnn_utils import forward_rnn
from utils.rnn_utils import forward_rnn_comb
from utils.rnn_utils import ridge
from utils.rnn_utils import compute_conceptor
from utils.rnn_utils import std_noise_func
from utils.rnn_utils import denoising_CTC
from utils.utils import PCA_3D

#Parameters that we can tune from the terminal
parser = argparse.ArgumentParser()
parser.add_argument("--noise_std", type=float, default=20) #% of deviation compared to x deviation in noise N(0,noise/100*std(x))
parser.add_argument("--seed", type=float, default=7) #for the noise

args = parser.parse_args()


#########################################################################

##Random inicialization of the ESN

########################################################################

spectral_radius=1.6 #spectral radius of W
scaling=0.9 #0.9 #input scaling
bias_scaling=0.4 # tanh bias
alpha=0.75 #Leakage 
a=5 #Aperture. 
N=100 #Network size 
# nu=2.5e-5 #Learning Rate 
# beta=0.9 #Control gain 
washout=20 # steps we wait until the network is stable, in order to show the results
reg=1 #regularization parameter in the Ride regression 
step=1 # number of steps that the model will predict
sparsity=None
noise_std=args.noise_std # %of noise
seed=args.seed #seed for the noise
steps_ol=30

#Input Signal
def sin(T,t, A):
    """Sinus wave with a given period T, and a time series"""
    s=A*np.sin((2*np.pi*t)/T)
    return s

#Temporal Series
points=40 #points in one periode
A=1 #Sinus Amplitude
T1=20 #period
#sample step
dt=T1/points
t_final=300
t=np.arange(0, t_final,dt)

#obtaining the input and some parameters for params
data1=sin(T1,t,A)
# data1=np.zeros((len(t)))
data1=data1.reshape(-1, 1)
# create shifted input and output, the output is the target
ut_train1 = data1[:-step]               # shape (N-step, 1)
yt_train1 = data1[step:]                # shape (N-step, 1)
#get dimensions
input_size=ut_train1.shape[-1]
output_size=yt_train1.shape[-1]

#build a dictionary with the parameters that will form the ESN, so we have them organized
#initalize parameters
params=rnn_params(
    N,
    input_size,
    output_size,
    scaling,
    spectral_radius,
    alpha,
    bias_scaling,
    sparsity,
    seed=1235
)


z=600 #for PCA visualization


##################################################################################

#Running the open loop without noise 

####################################################################################

#obtain matrix X1 (time, N) of internal states for all time points
X1_id=forward_rnn(params, ut_train1,seed, x_init=None,autonomous=False,conceptor=None)


X_effective = X1_id[washout:]
yt_train_effective = yt_train1[washout:]
#showing training X
#training Wout with Xi 
params_trained_id, mse = ridge(reg, X_effective, yt_train_effective,step,params) #this gives us the results for the trainning dataset


_,PCA1_id,PCA2_id,PCA3_id=PCA_3D(X1_id[washout:])


#computing the standard deviation for the noise
std_noise=std_noise_func(X1_id,noise_std)

##################################################################################

#Running the closed loop ideal (no noise)

####################################################################################

X_id=forward_rnn_comb(params_trained_id, ut_train1,seed,steps_ol, x_init=None,conceptor=None)

##################################################################################

#Running the open loop with noise without C and computing Wout

####################################################################################

#obtain matrix X1 (time, N) of internal states for all time points
X_noi=forward_rnn(params, ut_train1,seed, None,False,None,std_noise)
#Compute model conceptors noisy
C_noi=compute_conceptor(X_noi, a)

X_effective = X_noi[washout:]
yt_train_effective = yt_train1[washout:]

#training Wout with Xi 
params_trained_noi, mse = ridge(reg, X_effective, yt_train_effective,step,params) #this gives us the results for the trainning dataset

_,PCA1_noi,PCA2_noi,PCA3_noi=PCA_3D(X_noi[washout:z])



##################################################################################

#Running the Autonomous with noise without C 

####################################################################################
#obtain matrix X1 (time, N) of internal states for all time points
X1_noi=forward_rnn_comb(params_trained_noi, ut_train1,seed,steps_ol, None,None,std_noise)



##################################################################################

#Running the open loop with noise with noisy C and comouting Wout

####################################################################################

#obtain matrix X1 (time, N) of internal states for all time points
X_noi_C_noi=forward_rnn(params, ut_train1,seed, None,False,C_noi,std_noise)


X_effective = X_noi_C_noi[washout:]
yt_train_effective = yt_train1[washout:]

#training Wout with Xi 
params_trained_C, mse = ridge(reg, X_effective, yt_train_effective,step,params) #this gives us the results for the trainning dataset

_,PCA1_C_noi,PCA2_C_noi,PCA3_C_noi=PCA_3D(X_noi_C_noi[washout:z])


##################################################################################

#Running the Autonomous with noisy C 

####################################################################################
X1_noi_C_noi=forward_rnn_comb(params_trained_C, ut_train1,seed,steps_ol, None,C_noi,std_noise)

##################################################################################

#Running the open loop with noise with CTC C anc computing Wout

####################################################################################

C_ctc=denoising_CTC(params, ut_train1, std_noise, a)
#obtain matrix X1 (time, N) of internal states for all time points
X_noi_C_ctc=forward_rnn(params, ut_train1,seed, None,False,C_ctc,std_noise)

X_effective = X_noi_C_ctc[washout:]
yt_train_effective = yt_train1[washout:]

#training Wout with Xi 
params_trained_CTC, mse = ridge(reg, X_effective, yt_train_effective,step,params) #this gives us the results for the trainning dataset
_,PCA1_C_ctc,PCA2_C_ctc,PCA3_C_ctc=PCA_3D(X_noi_C_ctc[washout:z])

##################################################################################

#Running the Autonomous with C CTC

####################################################################################
X1_noi_C_ctc=forward_rnn_comb(params_trained_CTC, ut_train1,seed,steps_ol, None,C_ctc,std_noise)



#computing the outputs
#obtaining the outputs for autonomous mode
Y_target=yt_train1 #real data
Y1_id = X_id @ params_trained_id['wout'].T + params_trained_id['bias_out'] #autonomous with noise 
Y1_noi = X1_noi @ params_trained_noi['wout'].T + params_trained_noi['bias_out'] #autonomous with noise 
Y1_noi_C_noi = X1_noi_C_noi @ params_trained_C['wout'].T + params_trained_C['bias_out'] #autonomous with noise with noisy C
Y1_noi_C_ctc = X1_noi_C_ctc @ params_trained_CTC['wout'].T + params_trained_CTC['bias_out'] #autonomous with noise with ctc C
#outputs for open loop



# limits for the plot
x = 0
steps_in = steps_ol + x
steps = 200 + steps_ol + x

# ALIGN outputs
yt = Y_target[steps_in:steps]
y_ctc = Y1_noi_C_ctc[steps_in:steps]
y_noi = Y1_noi[steps_in:steps]
y_id = Y1_id[steps_in:steps]
y_Cnoi = Y1_noi_C_noi[steps_in:steps]

# Real time axis (k)
k = np.arange(0, 4000)
k = k[steps_in-steps_ol:steps-steps_ol]


# ==========================================================
# GLOBAL STYLE (Paper Optimized)
# ==========================================================

plt.rcParams.update({
    'font.size': 26,
    'axes.labelsize': 28,
    'axes.titlesize': 28,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 26,
    'lines.linewidth': 2.5,
    "axes.linewidth": 2.0,
    "axes.edgecolor": "black",
})

# ==========================================================
# PROFESSIONAL COLOR PALETTE
# ==========================================================

colors = {
    'ideal': '#2E8B57',        # green (no noise)
    'without_C': '#B22222',    # red (baseline)
    'C_noisy': '#6BAED6',      # light blue
    'C_ctc': '#1F4E79',        # dark blue
}

# ==========================================================
# OUTPUT COMPARISON
# ==========================================================

y_min = min(y_id.min(), y_noi.min(), y_Cnoi.min(), y_ctc.min(), yt.min())
y_max = max(y_id.max(), y_noi.max(), y_Cnoi.max(), y_ctc.max(), yt.max())

fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

for ax in axs:
    ax.set_axisbelow(True)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim([y_min-0.1, y_max+0.1])

# --- Ideal case ---
line_ideal, = axs[0].plot(
    k, y_id, color=colors['ideal'],
    linewidth=5, solid_capstyle='round', alpha=1.0
)
target_line, = axs[0].plot(
    k, yt, linestyle='--', color='black',
    linewidth=3, alpha=0.9
)

# --- Noisy cases ---
line_withoutC, = axs[1].plot(
    k, y_noi, color=colors['without_C'],
    linewidth=5, solid_capstyle='round', alpha=1.0
)
axs[1].plot(k, yt, linestyle='--', color='black', linewidth=3, alpha=0.9)

line_C_noisy, = axs[2].plot(
    k, y_Cnoi, color=colors['C_noisy'],
    linewidth=5, solid_capstyle='round', alpha=1.0
)
axs[2].plot(k, yt, linestyle='--', color='black', linewidth=3, alpha=0.9)

line_C_ctc, = axs[3].plot(
    k, y_ctc, color=colors['C_ctc'],
    linewidth=5, solid_capstyle='round', alpha=1.0
)
axs[3].plot(k, yt, linestyle='--', color='black', linewidth=3, alpha=0.9)

# Labels
fig.text(0.04, 0.5, 'Output (y(k))',
         va='center', rotation='vertical', fontsize=28)

axs[3].set_xlabel("Time steps (k)", fontsize=28)

# Section titles
fig.text(0.49, 0.91, "No internal noise",
         ha='center', fontsize=28, fontweight='bold')

fig.text(0.49, 0.69, "Internal noise",
         ha='center', fontsize=28, fontweight='bold')

# Legend
fig.legend(
    handles=[line_withoutC, line_C_noisy, line_C_ctc, target_line],
    labels=['Without C', r'With $C_{noisy}$', r'With $C_{ctc}$', 'Target'],
    loc='center right',
    frameon=True,
    framealpha=0.9
)

# X limits
axs[3].set_xlim(steps_in-steps_ol, steps-1-steps_ol)

plt.tight_layout(rect=[0.05, 0, 0.88, 0.92])
# plt.savefig("plots/Figure5a-d.png", dpi=300, bbox_inches='tight')
plt.savefig("plots/Figure5a-d.pdf", dpi=300, bbox_inches='tight')
plt.show()

# ==========================================================
# PCA COMPARISON
# ==========================================================

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_alpha(0)

s = 40  # slightly larger markers for paper

data = [
    (PCA1_id, PCA2_id, colors['ideal']),
    (PCA1_noi, PCA2_noi, colors['without_C']),
    (PCA1_C_noi, PCA2_C_noi, colors['C_noisy']),
    (PCA1_C_ctc, PCA2_C_ctc, colors['C_ctc'])
]

for ax, (PCA1, PCA2, color) in zip(axs.flat, data):
    ax.set_axisbelow(True)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.scatter(PCA1, PCA2, color=color, s=s, alpha=1.0, zorder=4.5)
    ax.set_xlabel("PC1", fontsize=28)
    ax.set_ylabel("PC2", fontsize=28)
    ax.set_facecolor('none')

# Unified axes
pc1_min = min(PCA1_id.min(), PCA1_noi.min(), PCA1_C_noi.min(), PCA1_C_ctc.min())
pc1_max = max(PCA1_id.max(), PCA1_noi.max(), PCA1_C_noi.max(), PCA1_C_ctc.max())
pc2_min = min(PCA2_id.min(), PCA2_noi.min(), PCA2_C_noi.min(), PCA2_C_ctc.min())
pc2_max = max(PCA2_id.max(), PCA2_noi.max(), PCA2_C_noi.max(), PCA2_C_ctc.max())

for ax in axs.flat:
    ax.set_xlim([pc1_min - 0.5, pc1_max + 0.5])
    ax.set_ylim([pc2_min - 0.5, pc2_max + 0.5])

plt.tight_layout()
# plt.savefig("plots/Figure5e-h.png", dpi=300, bbox_inches='tight', transparent=True)
plt.savefig("plots/Figure5e-h.pdf", dpi=300, bbox_inches='tight', transparent=True)
plt.show()
