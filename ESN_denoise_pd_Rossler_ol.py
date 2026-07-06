#Imports
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from utils.rnn_utils import rnn_params
from utils.rnn_utils import forward_rnn
from utils.rnn_utils import ridge
from utils.rnn_utils import compute_conceptor
from utils.rnn_utils import std_noise_func
from utils.rnn_utils import denoising_CTC_m
from utils.rnn_utils import compute_conceptor_avg
from utils.utils import NRMSE

#Parameters that you can tune from terminal
parser = argparse.ArgumentParser()
parser.add_argument("--trials_noise", type=int, default=10) #number of trails, for different seeds
parser.add_argument("--trials_esn", type=int, default=10) #number of trails, for different seeds
parser.add_argument("--aperture", type=float, default=5) #conceptor aperture
parser.add_argument("--a_new", type=float, default=5) # aperture for the clean conceptors
parser.add_argument("--steps", type=int, default=3000) #number of time steps used tu compute the NRMSE and the correlation
parser.add_argument("--time_len", type=int, default=3000) #number of time steps used to trine the reservoir
parser.add_argument("--N", type=int, default=200) #number of neurons
parser.add_argument("--spectral_radius", type=float, default=1.6) #spectral radius
parser.add_argument("--scaling", type=float, default=0.9) #scaling
parser.add_argument("--seed", type=int, default=12) #seed for the noise
parser.add_argument("--noise", type=int, default=50) #% of noise
parser.add_argument("--p", type=int, default=25) #max steps to predict
parser.add_argument("--p_steps", type=int, default=1) # steps to predict
parser.add_argument("--m", type=int, default=2) # number of realizations of CTC
parser.add_argument(
    "--corr",
    type=lambda x: str(x).lower() in ['true', '1', 'yes', 'y'],
    default=False,
) #True: correlated noise, False: uncorrelated noise

args = parser.parse_args()


# max steps should be larger than p, at list 1.1*p_max
if not args.steps > 1.2 * args.p:
    raise ValueError(
        f"Inconsistent configuration: 'steps' must be at least 1.2 times larger than p_max.\n"
        f"Received: steps = {args.steps}, p_max = {args.p}.\n"
        f"Please increase 'steps' or reduce 'p_max'."
    )


#########################################################################

##Random inicialization of the ESN

########################################################################

spectral_radius=1.6 #spectral radius
scaling=0.9 #input scaling
bias_scaling=0.4 #scaling inside tanh
alpha=0.75 #Leakage
a=args.aperture #Aperture.  
N=args.N # Network size 256
nu=2.5e-5 #Learning Rate
beta=0.9 #Control gain
washout=20 # steps we wait until the network is stable, in order to show the results
reg=1 #regularization parameter in the Ride regression
m=args.m
time_len=args.time_len #number of points that we want to us eto train the Reservoir, as our data has 4000 points in total
steps=args.steps
sparsity=None
a_new=args.a_new
corr=args.corr

#Input Signal Rössler x
data1 = pd.read_csv("Rossler_data/xRossler.txt", sep="\t",header=None,index_col=None)
data1=data1.values[:time_len]
data1=data1.reshape(-1, 1)

#rossler was solved with a time step of 0.1 and then resempled with a time step og 0.3 so:
t_r = np.linspace(0, (len(data1)-1)*0.3, len(data1))
# plt.plot(data1[:time_len],'.-')
dt=0.3

########################################################################

#parameters for the sacn

######################################################################
seed1=20 #seed for the seed generator :)
trials_noise=args.trials_noise #number of trials for the noise
trials_esn=args.trials_esn #number of trials for the esn
np.random.seed(seed1) #if we want this seed to be alwas de same  
seed_noise=np.random.randint(0, 2000, size=trials_noise) #seed for the random noise
seed_esn=np.random.randint(0, 2000, size=trials_esn) #seed for the ESN

# number of steps that the model will predict
p1=np.arange(-args.p,args.p+args.p_steps,args.p_steps) 
# p1=[0]

#storing the error 
trials_noise=args.trials_noise #number of trials for the noise
trials_esn=args.trials_esn #number of trials for the esn
trials=trials_noise*trials_esn
nrmse_noi=np.empty((len(p1),trials),dtype=float)
nrmse_C_noi=np.empty((len(p1),trials),dtype=float)
nrmse_noi_C_ctc=np.empty((len(p1),trials),dtype=float)
nrmse_C_id=np.empty((len(p1),trials),dtype=float)
nrmse_noi_C_ctc_m=np.empty((len(p1),trials),dtype=float)
nrmse_noi_C_avg=np.empty((len(p1),trials),dtype=float)

for idx, p in enumerate(p1):
    #create shifted input and output, the output is the target
    if p > 0:  # p steps ahead
        ut_train1 = data1[:-p]        # all but the last p
        yt_train1 = data1[p:]         # all but the first p
    elif p < 0:  # |p| steps behind
        ut_train1 = data1[-p:]        # skip the first |p|
        yt_train1 = data1[:p]         # take up to -|p|
    else:  # p == 0
        ut_train1 = data1
        yt_train1 = data1
    #get dimensions
    input_size=ut_train1.shape[-1]
    output_size=yt_train1.shape[-1]
    
     
    
    for i in range(trials_esn): #trials of different ESN realizations
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
           seed=seed_esn[i]
       )

       ##################################################################################

       #Running the open loop without noise for the noise std

       ####################################################################################

       #obtain matrix X1 (time, N) of internal states for all time points
        X_id=forward_rnn(params, ut_train1, 42,x_init=None,autonomous=False,conceptor=None)
        # Ideal conceptor computed from noise-free reservoir states
        C_id=compute_conceptor(X_id, a)
    
       #computing the standard deviation for the noise
        std_noise=std_noise_func(X_id,args.noise)
    
        
        for j in range(trials_noise): #trials of differents seeds
                
                
            ######################################################################################
                
            #Running in open loop with noise without conceptor
                
            #########################################################################################
                
                
            X_noi=forward_rnn(params, ut_train1, seed_noise[j],None,False,None,std_noise,corr=corr)
            # trained_model_new(X_noi[washout:],ut_train1,yt_train1,params_trained,washout,True,None,label)
            #noisy conceptor
            C_noi=compute_conceptor(X_noi, a)
            #getting the final Wout with the ridge regression (Wout=Ytarget*X.T*(X*X.T+beta*I)^-1)
            X_effective = X_noi[washout:]
            yt_train_effective = yt_train1[washout:]
            #showing training X
            #training Wout with Xi 
            params_trained_noi, mse = ridge(reg, X_effective, yt_train_effective,p,params) #this gives us the results for the trainning dataset
            
            
           
            
            
                
            ######################################################################################
                
            #Running in open loop with noise with noisy conceptor
                
            #########################################################################################
                
                
            X_noi_C_noi=forward_rnn(params, ut_train1, seed_noise[j],None,False,C_noi,std_noise,corr=corr)
                
            
            #getting the final Wout with the ridge regression (Wout=Ytarget*X.T*(X*X.T+beta*I)^-1)
            X_effective = X_noi_C_noi[washout:]
            yt_train_effective = yt_train1[washout:]
            #showing training X
            #training Wout with Xi 
            params_trained_C, mse = ridge(reg, X_effective, yt_train_effective,p,params) #this gives us the results for the trainning dataset
            

            ######################################################################################

            #Running in open loop with noise with ideal conceptor

            #########################################################################################

            # C_id is computed from X_id without noise, but it is applied
            # to the reservoir affected by the current noise realization.
            X_noi_C_id=forward_rnn(
                params,
                ut_train1,
                seed_noise[j],
                None,
                False,
                C_id,
                std_noise,
                corr=corr
            )

            X_effective = X_noi_C_id[washout:]
            yt_train_effective = yt_train1[washout:]

            params_trained_C_id, mse = ridge(
                reg,
                X_effective,
                yt_train_effective,
                p,
                params
            )
            
           
            ######################################################################################
                
            #Running in open loop with noise with CTC m conceptor
                
            #########################################################################################
                
            #first computing the CTC conceptor for each level of noise
                
            C_ctc_m=denoising_CTC_m(params, ut_train1, std_noise, a_new,m,corr=corr)
                
            X_noi_C_ctc_m=forward_rnn(params, ut_train1, seed_noise[j],None,False,C_ctc_m,std_noise,corr=corr)
                
            #getting the final Wout with the ridge regression (Wout=Ytarget*X.T*(X*X.T+beta*I)^-1)
            X_effective = X_noi_C_ctc_m[washout:]
            yt_train_effective = yt_train1[washout:]
            #showing training X
            #training Wout with Xi 
            params_trained_CTC_m, mse = ridge(reg, X_effective, yt_train_effective,p,params) #this gives us the results for the trainning dataset
            
            ######################################################################################
                
            #Running in open loop with noise with avg conceptor
                
            #########################################################################################
                
            #first computing the avg conceptor for each level of noise
                
            C_avg=compute_conceptor_avg(params, ut_train1, std_noise, a_new,corr=corr)
                
            X_noi_C_avg=forward_rnn(params, ut_train1, seed_noise[j],None,False,C_avg,std_noise,corr=corr)
                
            #getting the final Wout with the ridge regression (Wout=Ytarget*X.T*(X*X.T+beta*I)^-1)
            X_effective = X_noi_C_avg[washout:]
            yt_train_effective = yt_train1[washout:]
            #showing training X
            #training Wout with Xi 
            params_trained_C_avg, mse = ridge(reg, X_effective, yt_train_effective,p,params) #this gives us the results for the trainning dataset
            
                 
            
            Y_target=yt_train1 #real data
            Y_noi = X_noi[washout:] @ params_trained_noi['wout'].T + params_trained_noi['bias_out'] #open loop with noise 
            Y_noi_C_noi = X_noi_C_noi[washout:] @ params_trained_C['wout'].T + params_trained_C['bias_out'] #open loop with noise with noisy C
            Y_noi_C_id = X_noi_C_id[washout:] @ params_trained_C_id['wout'].T + params_trained_C_id['bias_out'] #open loop with noise with ideal C
            Y_noi_C_ctc_m = X_noi_C_ctc_m[washout:] @ params_trained_CTC_m['wout'].T + params_trained_CTC_m['bias_out'] #open loop with noise with ctc m C
            Y_noi_C_avg = X_noi_C_avg[washout:] @ params_trained_C_avg['wout'].T + params_trained_C_avg['bias_out'] #open loop with noise with avg C
            
            #transforming the array 
            y = np.asarray(Y_target[washout:]).ravel()   
                  
            y_noi = np.asarray(Y_noi).ravel()   
            y_noi_C_noi = np.asarray(Y_noi_C_noi).ravel()
            y_noi_C_id = np.asarray(Y_noi_C_id).ravel()
            y_noi_C_ctc_m = np.asarray(Y_noi_C_ctc_m).ravel()
            y_noi_C_avg = np.asarray(Y_noi_C_avg).ravel()
            
            ###########################################################################################
                   
            #NRMSE
                   
            ############################################################################################# 
            trial_index = i * trials_noise + j #to store the results correctly
            nrmse_noi[idx,trial_index]        = NRMSE(y, y_noi)
            nrmse_C_noi[idx,trial_index]  = NRMSE(y, y_noi_C_noi)
            nrmse_C_id[idx,trial_index] = NRMSE(y, y_noi_C_id)
            nrmse_noi_C_ctc_m[idx,trial_index]  = NRMSE(y, y_noi_C_ctc_m)
            nrmse_noi_C_avg[idx,trial_index]  = NRMSE(y, y_noi_C_avg)


if corr:
    c = "correlated"
else:
    c = "uncorrelated"   

# Style settings 
plt.rcParams.update({
    # Figure
    "figure.figsize": (10, 6),

    # Axis labels
    "axes.labelsize": 20,
    "axes.titlesize": 20,

    # Tick labels
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,

    # Legend
    "legend.fontsize": 18,

    # Lines & markers
    "lines.linewidth": 2,
    "lines.markersize": 6,

    # Grid
    "grid.alpha": 0.6,
    "grid.linestyle": "--",

    # Axis spines
    "axes.linewidth": 1.8,
    "axes.edgecolor": "black",
})


steps_in=0
steps_fin=1000


# Compute mean and standard deviation across trials (axis=1)
mean_noi = np.mean(nrmse_noi, axis=1)
std_noi = np.std(nrmse_noi, axis=1)

mean_C_noi = np.mean(nrmse_C_noi, axis=1)
std_C_noi = np.std(nrmse_C_noi, axis=1)

mean_C_id = np.mean(nrmse_C_id, axis=1)
std_C_id = np.std(nrmse_C_id, axis=1)

mean_C_ctc_m = np.mean(nrmse_noi_C_ctc_m, axis=1)
std_C_ctc_m = np.std(nrmse_noi_C_ctc_m, axis=1)

mean_C_avg = np.mean(nrmse_noi_C_avg, axis=1)
std_C_avg = np.std(nrmse_noi_C_avg, axis=1)



# ----------------------------- Plot NRMSE with CTC m --------------------------------------
plt.figure(figsize=(8,4), dpi=300)
ax = plt.gca()
ax.set_axisbelow(True) 
# Without C
plt.plot(
    p1, mean_noi,
    color='#B22222', alpha=0.9,
    marker='s', label="Without C"
)
plt.fill_between(
    p1,
    mean_noi - std_noi,
    mean_noi + std_noi,
    color='#B22222', alpha=0.25
)

# With C_noisy
plt.plot(
    p1, mean_C_noi,
    color='#6BAED6', alpha=0.9,
    marker='^', 
    label=r"With $C_{noisy}$"
)
plt.fill_between(
    p1,
    mean_C_noi - std_C_noi,
    mean_C_noi + std_C_noi,
    color='#6BAED6', alpha=0.25
)

# With C_ctc
plt.plot(
    p1, mean_C_ctc_m,
    color='#1F4E79', alpha=0.9,
    marker='o', 
    label=r"With $C_{ctc}$"
)
plt.fill_between(
    p1,
    mean_C_ctc_m - std_C_ctc_m,
    mean_C_ctc_m + std_C_ctc_m,
    color='#1F4E79', alpha=0.25
)

# With C_ideal
plt.plot(
    p1, mean_C_id,
    color='#D4A017', alpha=0.95,
    marker='p',
    label=r"With $C_{ideal}$"
)
plt.fill_between(
    p1,
    mean_C_id - std_C_id,
    mean_C_id + std_C_id,
    color='#D4A017', alpha=0.22
)

# Labels and style
plt.xlabel(r"$p$ (prediction steps)", size=20)
plt.ylabel("NRMSE", size=20)

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Save figure
plt.savefig(
    f"plots/NRMSE_CTCm_vs_ideal_vs_p_N{N}_m{m}_trials{trials}_noise{args.noise}_steps{steps}_a{a}_aNew{a_new}_pmax{args.p}_{c}.png",
    dpi=300, bbox_inches='tight'
)

# plt.savefig(
#     f"plots/NRMSE_CTCm_vs_p_N{N}_m{m}_trials{trials}_noise{args.noise}_steps{steps}_a{a}_aNew{a_new}_pmax{args.p}_new.pdf",
#     dpi=300, bbox_inches='tight'
# )

plt.show()

# ----------------------------- Plot NRMSE with CTC avg --------------------------------------
plt.figure(figsize=(8,4), dpi=300)
ax = plt.gca()
ax.set_axisbelow(True) 
# Without C
plt.plot(
    p1, mean_noi,
    color='#B22222', alpha=0.9,
    marker='s', label="Without C"
)
plt.fill_between(
    p1,
    mean_noi - std_noi,
    mean_noi + std_noi,
    color='#B22222', alpha=0.25
)

# With C_noisy
plt.plot(
    p1, mean_C_noi,
    color='#6BAED6', alpha=0.9,
    marker='^', 
    label=r"With $C_{noisy}$"
)
plt.fill_between(
    p1,
    mean_C_noi - std_C_noi,
    mean_C_noi + std_C_noi,
    color='#6BAED6', alpha=0.25
)

# With C_ctc
plt.plot(
    p1, mean_C_ctc_m,
    color='#1F4E79', alpha=0.9,
    marker='o', 
    label=r"With $C_{ctc}$"
)
plt.fill_between(
    p1,
    mean_C_ctc_m - std_C_ctc_m,
    mean_C_ctc_m + std_C_ctc_m,
    color='#1F4E79', alpha=0.25
)

# With C_ctc
plt.plot(
    p1, mean_C_avg,
    color='#009E9A', alpha=0.9,
    marker='D', 
    label=r"With $C_{avg}$"
)
plt.fill_between(
    p1,
    mean_C_avg - std_C_avg,
    mean_C_avg + std_C_avg,
    color='#009E9A', alpha=0.25
)



# Labels and style
plt.xlabel(r"$p$ (prediction steps)", size=20)
plt.ylabel("NRMSE", size=20)

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Save figure
plt.savefig(
    f"plots/NRMSE_Cavg_and_ideal_vs_p_N{N}_m{m}_trials{trials}_noise{args.noise}_steps{steps}_a{a}_aNew{a_new}_pmax{args.p}_{c}.png",
    dpi=300, bbox_inches='tight'
)

# plt.savefig(
#     f"plots/NRMSE_CTCm_vs_p_N{N}_m{m}_trials{trials}_noise{args.noise}_steps{steps}_a{a}_aNew{a_new}_pmax{args.p}_new.pdf",
#     dpi=300, bbox_inches='tight'
# )

plt.show()





























