#Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from utils.rnn_utils import rnn_params
from utils.rnn_utils import forward_rnn
from utils.rnn_utils import ridge
from utils.rnn_utils import compute_conceptor
from utils.rnn_utils import std_noise_func
from utils.rnn_utils import denoising_CTC
from utils.rnn_utils import compute_conceptor_avg
from utils.utils import PCA_3D

#Parameters that we can tune from the terminal
parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=float, default=42) #for the noise 

args = parser.parse_args()


#########################################################################

##Random inicialization of the ESN

########################################################################

spectral_radius=1.6 #spectral radius of W
scaling=0.9 #0.9 #input scaling
bias_scaling=0.4 # tanh bias
alpha=0.75 #Leakage 
a=5 #Aperture. 
N=200 #Network size 
# nu=2.5e-5 #Learning Rate 
# beta=0.9 #Control gain 
washout=20 # steps we wait until the network is stable, in order to show the results
reg=1 #regularization parameter in the Ride regression 
step=1 # number of steps that the model will predict
sparsity=None
seed=args.seed #seed for the noise
corr=False #default False. True: Correlated noise, False: Uncorrelated noise 

#Input Signal

time_len=3000
#Input Signal Rössler x
data1 = pd.read_csv("Rossler_data/xRossler.txt", sep="\t",header=None,index_col=None)
data1=data1.values[:time_len]
data1=data1.reshape(-1, 1)

# #obtaining the input and some parameters for params
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




#noise scan
noise_std=[10,30,50,70,90]
# noise_std=[0,0,0,0,0]

# storing the PCAs
PCA1_noi=np.empty((len(PCA1_id),len(noise_std)),dtype=float)
PCA2_noi=np.empty((len(PCA1_id),len(noise_std)),dtype=float)
PCA3_noi=np.empty((len(PCA1_id),len(noise_std)),dtype=float)

PCA1_C_noi=np.empty((len(PCA1_id),len(noise_std)),dtype=float)
PCA2_C_noi=np.empty((len(PCA1_id),len(noise_std)),dtype=float)
PCA3_C_noi=np.empty((len(PCA1_id),len(noise_std)),dtype=float)

PCA1_C_ctc=np.empty((len(PCA1_id),len(noise_std)),dtype=float)
PCA2_C_ctc=np.empty((len(PCA1_id),len(noise_std)),dtype=float)
PCA3_C_ctc=np.empty((len(PCA1_id),len(noise_std)),dtype=float)

PCA1_C_avg=np.empty((len(PCA1_id),len(noise_std)),dtype=float)
PCA2_C_avg=np.empty((len(PCA1_id),len(noise_std)),dtype=float)
PCA3_C_avg=np.empty((len(PCA1_id),len(noise_std)),dtype=float)

for i in range(len(noise_std)):
    #computing the standard deviation for the noise
    std_noise=std_noise_func(X1_id,noise_std[i])
    

    ##################################################################################
    
    #Running the open loop with noise without C and computing Wout
    
    ####################################################################################
    
    #obtain matrix X1 (time, N) of internal states for all time points
    X_noi=forward_rnn(params, ut_train1,seed, None,False,None,std_noise,corr=corr)
    #Compute model conceptors noisy
    C_noi=compute_conceptor(X_noi, a)
    
    X_effective = X_noi[washout:]
    yt_train_effective = yt_train1[washout:]
    
    #training Wout with Xi 
    params_trained_noi, mse = ridge(reg, X_effective, yt_train_effective,step,params) #this gives us the results for the trainning dataset
    
    _,PCA1_noi[:,i],PCA2_noi[:,i],PCA3_noi[:,i]=PCA_3D(X_noi[washout:])
    
    

    
    ##################################################################################
    
    #Running the open loop with noise with noisy C and computing Wout
    
    ####################################################################################
    
    #obtain matrix X1 (time, N) of internal states for all time points
    X_noi_C_noi=forward_rnn(params, ut_train1,seed, None,False,C_noi,std_noise,corr=corr)
    
    
    X_effective = X_noi_C_noi[washout:]
    yt_train_effective = yt_train1[washout:]
    
    #training Wout with Xi 
    params_trained_C, mse = ridge(reg, X_effective, yt_train_effective,step,params) #this gives us the results for the trainning dataset
    
    _,PCA1_C_noi[:,i],PCA2_C_noi[:,i],PCA3_C_noi[:,i]=PCA_3D(X_noi_C_noi[washout:])
    
    

    ##################################################################################
    
    #Running the open loop with noise with CTC C anc computing Wout
    
    ####################################################################################
    
    C_ctc=denoising_CTC(params, ut_train1, std_noise, a,corr=corr)
    #obtain matrix X1 (time, N) of internal states for all time points
    X_noi_C_ctc=forward_rnn(params, ut_train1,seed, None,False,C_ctc,std_noise,corr=corr)
    
    X_effective = X_noi_C_ctc[washout:]
    yt_train_effective = yt_train1[washout:]
    
    #training Wout with Xi 
    params_trained_CTC, mse = ridge(reg, X_effective, yt_train_effective,step,params) #this gives us the results for the trainning dataset
    _,PCA1_C_ctc[:,i],PCA2_C_ctc[:,i],PCA3_C_ctc[:,i]=PCA_3D(X_noi_C_ctc[washout:])

    ##################################################################################
    
    #Running the open loop with noise with C avg anc computing Wout
    
    ####################################################################################
    
    C_avg=compute_conceptor_avg(params, ut_train1, std_noise, a,corr=corr)
    #obtain matrix X1 (time, N) of internal states for all time points
    X_noi_C_avg=forward_rnn(params, ut_train1,seed, None,False,C_avg,std_noise,corr=corr)
    
    X_effective = X_noi_C_avg[washout:]
    yt_train_effective = yt_train1[washout:]
    
    #training Wout with Xi 
    params_trained_C_avg, mse = ridge(reg, X_effective, yt_train_effective,step,params) #this gives us the results for the trainning dataset
    _,PCA1_C_avg[:,i],PCA2_C_avg[:,i],PCA3_C_avg[:,i]=PCA_3D(X_noi_C_avg[washout:])
     
 
    
 
if corr:
    c = "correlated"
else:
    c = "uncorrelated"    

#limits for the plot
steps_in=washout
steps=500+washout

colors = {
    'ideal': '#2E8B57',        # green
    'without_C': '#B22222',    # red
    'C_noisy': '#6BAED6',      # blue
    'C_ctc': '#1F4E79',        # blue
    'C_avg': '#009E9A',        # blue
}
# colors = {
#     'ideal': 'gray',        # green
#     'without_C': 'gray',    # red
#     'C_noisy': 'gray',      # blue
#     'C_ctc': 'gray',        # blue
#     'C_avg': 'gray',        # blue
# }



##################################

# PCA vs Noise (PC1 vs PC2)

####################################

# Noise-free PCA trajectory used only as a visual background reference.
# It is deliberately excluded from all legends.
clean_reference_color = "gray"
clean_reference_alpha = 0.5
clean_reference_size = 20

n_noise = len(noise_std)

fig, axs = plt.subplots(
    3, n_noise,
    figsize=(3.5 * n_noise, 9),
    sharex=True,
    sharey=True
)

# White background
fig.patch.set_alpha(1)

# Plot each PCA
for i, noise in enumerate(noise_std):

    # Noise-free trajectory behind each noisy PCA.
    # It has no label and therefore does not appear in the legend.
    for r in range(3):
        axs[r, i].scatter(
            PCA1_id[steps_in:steps],
            PCA2_id[steps_in:steps],
            s=clean_reference_size,
            color=clean_reference_color,
            alpha=clean_reference_alpha,
            edgecolors="none",
            zorder=1,
        )

    # Row 0: Without C
    axs[0, i].scatter(
        PCA1_noi[steps_in:steps, i],
        PCA2_noi[steps_in:steps, i],
        s=20,
        # color=colors['without_C'],
        color=colors['without_C'],
        alpha=1,
        zorder=2,
        label="Without C" if i == 0 else ""
    )

    # Row 1: With noisy C
    axs[1, i].scatter(
        PCA1_C_noi[steps_in:steps, i],
        PCA2_C_noi[steps_in:steps, i],
        s=20,
        color=colors['C_noisy'],
        alpha=1,
        zorder=2,
        label=r"With $C_{noisy}$" if i == 0 else ""
    )

    # Row 2: With CTC C
    axs[2, i].scatter(
        PCA1_C_ctc[steps_in:steps, i],
        PCA2_C_ctc[steps_in:steps, i],
        s=20,
        color=colors['C_ctc'],
        alpha=1,
        zorder=2,
        label=r"With $C_{ctc}$" if i == 0 else ""
    )
    


    for r in range(3):
        ax = axs[r, i]
        ax.set_facecolor('white')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

# Unified axis limits
pc1_min = min(PCA1_id.min(), PCA1_noi.min(), PCA1_C_noi.min(), PCA1_C_ctc.min())
pc1_max = max(PCA1_id.max(), PCA1_noi.max(), PCA1_C_noi.max(), PCA1_C_ctc.max())
pc2_min = min(PCA2_id.min(), PCA2_noi.min(), PCA2_C_noi.min(), PCA2_C_ctc.min())
pc2_max = max(PCA2_id.max(), PCA2_noi.max(), PCA2_C_noi.max(), PCA2_C_ctc.max())

for ax in axs.flat:
    ax.set_xlim(pc1_min - 0.5, pc1_max + 0.5)
    ax.set_ylim(pc2_min - 0.5, pc2_max + 0.5)

# X-axis labels for each column (bottom row)
for i, noise in enumerate(noise_std):
    axs[2, i].set_xlabel(f"{noise}", fontsize=30)

# X-axis title with extra space (y-coordinate lower)
fig.text(0.5, 0.015, "Noise level (%)", ha='center', fontsize=30)

# Legend with larger markers and more separation
handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['without_C'], markersize=15, label="Without C"),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['C_noisy'], markersize=15, label=r"With $C_{noisy}$"),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['C_ctc'], markersize=15, label=r"With $C_{ctc}$")
]

# Move legend further up from the plot
fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False, fontsize=30, bbox_to_anchor=(0.5, 1.02), markerscale=1.5)

plt.tight_layout(pad=0.1, rect=[0, 0.05, 1, 0.95])
# plt.savefig(
#     "plots/PCA_vs_noise.pdf",
#     dpi=300,
#     bbox_inches="tight"
# )
plt.savefig(
    f"plots/PCA_vs_noise_{c}.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()



point_alpha=1

##################################
# PCA vs Noise (PC1 vs PC2) C_avg
##################################

n_noise = len(noise_std)

fig, axs = plt.subplots(
    4,
    n_noise,
    figsize=(3.5 * n_noise, 13),
    sharex=True,
    sharey=True,
    squeeze=False
)

# White, opaque background
fig.patch.set_facecolor("white")
fig.patch.set_alpha(1)

# Plot each PCA
for i, noise in enumerate(noise_std):

    # Noise-free trajectory behind each noisy PCA.
    # It has no label and therefore does not appear in the legend.
    for r in range(4):
        axs[r, i].scatter(
            PCA1_id[steps_in:steps],
            PCA2_id[steps_in:steps],
            s=clean_reference_size,
            color=clean_reference_color,
            alpha=clean_reference_alpha,
            edgecolors="none",
            zorder=1,
        )

    # Row 0: Without C
    axs[0, i].scatter(
        PCA1_noi[steps_in:steps, i],
        PCA2_noi[steps_in:steps, i],
        s=20,
        color=colors["without_C"],
        alpha=point_alpha,
        zorder=2
    )

    # Row 1: With noisy C
    axs[1, i].scatter(
        PCA1_C_noi[steps_in:steps, i],
        PCA2_C_noi[steps_in:steps, i],
        s=20,
        color=colors["C_noisy"],
        alpha=point_alpha,
        zorder=2
    )

    # Row 2: With CTC C
    axs[3, i].scatter(
        PCA1_C_ctc[steps_in:steps, i],
        PCA2_C_ctc[steps_in:steps, i],
        s=20,
        color=colors["C_ctc"],
        alpha=point_alpha,
        zorder=2
    )

    # Row 3: With average C
    axs[2, i].scatter(
        PCA1_C_avg[steps_in:steps, i],
        PCA2_C_avg[steps_in:steps, i],
        s=20,
        color=colors["C_avg"],
        alpha=point_alpha,
        zorder=2
    )

    # Common format for all rows
    for r in range(4):
        ax = axs[r, i]

        ax.set_facecolor("white")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)


# Unified axis limits
pc1_min = min(
    PCA1_id.min(),
    PCA1_noi.min(),
    PCA1_C_noi.min(),
    PCA1_C_ctc.min(),
    PCA1_C_avg.min()
)

pc1_max = max(
    PCA1_id.max(),
    PCA1_noi.max(),
    PCA1_C_noi.max(),
    PCA1_C_ctc.max(),
    PCA1_C_avg.max()
)

pc2_min = min(
    PCA2_id.min(),
    PCA2_noi.min(),
    PCA2_C_noi.min(),
    PCA2_C_ctc.min(),
    PCA2_C_avg.min()
)

pc2_max = max(
    PCA2_id.max(),
    PCA2_noi.max(),
    PCA2_C_noi.max(),
    PCA2_C_ctc.max(),
    PCA2_C_avg.max()
)

pc1_margin = 0.05 * (pc1_max - pc1_min)
pc2_margin = 0.05 * (pc2_max - pc2_min)

for ax in axs.flat:
    ax.set_xlim(
        pc1_min - pc1_margin,
        pc1_max + pc1_margin
    )

    ax.set_ylim(
        pc2_min - pc2_margin,
        pc2_max + pc2_margin
    )


# Noise labels below the fourth row
for i, noise in enumerate(noise_std):
    axs[3, i].set_xlabel(
        f"{noise}",
        fontsize=30,
        labelpad=10
    )


# General x-axis title
fig.text(
    0.5,
    0.015,
    "Noise level (%)",
    ha="center",
    fontsize=30
)


# Legend
handles = [
    plt.Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markerfacecolor=colors["without_C"],
        markeredgecolor="none",
        markersize=15,
        label="Without C"
    ),
    plt.Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markerfacecolor=colors["C_noisy"],
        markeredgecolor="none",
        markersize=15,
        label=r"With $C_{noisy}$"
    ),
    plt.Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markerfacecolor=colors["C_ctc"],
        markeredgecolor="none",
        markersize=15,
        label=r"With $C_{ctc}$"
    ),
    plt.Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markerfacecolor=colors["C_avg"],
        markeredgecolor="none",
        markersize=15,
        label=r"With $C_{avg}$"
    )
]

fig.legend(
    handles=handles,
    loc="upper center",
    ncol=4,
    frameon=False,
    fontsize=30,
    bbox_to_anchor=(0.5, 1.01),
    columnspacing=1.5,
    handletextpad=0.4
)


plt.tight_layout(
    pad=0.1,
    rect=[0, 0.055, 1, 0.94]
)



plt.savefig(
    f"plots/PCA_vs_noise_Cavg_{c}_new.png",
    dpi=300,
    bbox_inches="tight",
    transparent=False,
    facecolor="white"
)

plt.show()
plt.close()
