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
from utils.utils import xcorr_PCA

#Parameters that you can tune from terminal
parser = argparse.ArgumentParser()
parser.add_argument("--trials_noise", type=int, default=10) #number of trails, for different seeds
parser.add_argument("--trials_esn", type=int, default=10) #number of trails, for different seeds
parser.add_argument("--noise_max", type=int, default=105) #max % of the deviation of the noise
parser.add_argument("--noise_steps", type=int, default=5) #steps of % of the deviation of the noise
parser.add_argument("--aperture", type=int, default=5) #conceptor aperture
parser.add_argument("--a_new", type=float, default=5) # aperture for the clean conceptors
parser.add_argument("--steps", type=int, default=3000) #number of time steps used tu compute the NRMSE and the correlation
parser.add_argument("--steps_ol", type=int, default=30) #number of time steps used for the open loop
parser.add_argument("--time_len", type=int, default=3000) #number of time steps used to trine the reservoir
parser.add_argument("--N", type=int, default=200) #number of neurons
parser.add_argument("--spectral_radius", type=float, default=1.6) #spectral radius
parser.add_argument("--scaling", type=float, default=0.9) #scaling
parser.add_argument("--m", type=int, default=3) #realization for the ctc
parser.add_argument(
    "--corr",
    type=lambda x: str(x).lower() in ['true', '1', 'yes', 'y'],
    default=False,
) #True: correlated noise, False: uncorrelated noise

args = parser.parse_args()


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
step=1 # number of steps that the model will predict
time_len=args.time_len #number of points that we want to us eto train the Reservoir, as our data has 4000 points in total
steps_ol=args.steps_ol #steps for the open loop
m=args.m
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
#create shifted input and output, the output is the target
ut_train1 = data1[:-step]               # shape (N-step, 1)
yt_train1 = data1[step:]                # shape (N-step, 1)
#get dimensions
input_size=ut_train1.shape[-1]
output_size=yt_train1.shape[-1]


########################################################################

#parameters for the noise scan

######################################################################
seed1=20 #seed for the seed generator :)
trials_noise=args.trials_noise #number of trials for the noise
trials_esn=args.trials_esn #number of trials for the esn
np.random.seed(seed1) #if we want this seed to be alwas de same  
seed_noise=np.random.randint(0, 2000, size=trials_noise) #seed for the random noise
seed_esn=np.random.randint(0, 2000, size=trials_esn) #seed for the ESN
# seed=[42]
steps=args.steps #time steps for the comparison
#noise scan
k=np.arange(0,args.noise_max,args.noise_steps)

#storing the error and correlation
trials=trials_noise*trials_esn
xcorrp_noi=np.empty((len(k),trials),dtype=float)
xcorrp_noi_C_noi=np.empty((len(k),trials),dtype=float)
xcorrp_noi_C_ctc_m=np.empty((len(k),trials),dtype=float)



#starting the loop
for i in range(len(k)): #scan in increasing noise
 
    for idx in range(trials_esn): #trials of different ESN realizations
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
            seed=seed_esn[idx]
        )

        ##################################################################################

        #Running the open loop without noise for noise std

        ####################################################################################

        #obtain matrix X1 (time, N) of internal states for all time points
        X_id=forward_rnn(params, ut_train1, 42,x_init=None,autonomous=False,conceptor=None)
        #Compute model conceptors

        #computing the standard deviation for the noise
        std_noise=std_noise_func(X_id,k[i])
        
        
        for j in range(trials_noise): #trials of different noise realizations
            
            
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
            params_trained_noi, mse = ridge(reg, X_effective, yt_train_effective,step,params) #this gives us the results for the trainning dataset
        
            
        
            
            ######################################################################################
            
            #Running in open loop with noise with noisy conceptor
            
            #########################################################################################
            
            
            X_noi_C_noi=forward_rnn(params, ut_train1, seed_noise[j],None,False,C_noi,std_noise,corr=corr)
            
        
            #getting the final Wout with the ridge regression (Wout=Ytarget*X.T*(X*X.T+beta*I)^-1)
            X_effective = X_noi_C_noi[washout:]
            yt_train_effective = yt_train1[washout:]
            #showing training X
            #training Wout with Xi 
            params_trained_C, mse = ridge(reg, X_effective, yt_train_effective,step,params) #this gives us the results for the trainning dataset
        
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
            params_trained_CTC_m, mse = ridge(reg, X_effective, yt_train_effective,step,params) #this gives us the results for the trainning dataset
            
            
            
            #obtaining the outputs
            Y_target=yt_train1 #real data
            Y_noi = X_noi[washout:washout+steps] @ params_trained_noi['wout'].T + params_trained_noi['bias_out'] #open loop with noise 
            Y_noi_C_noi = X_noi_C_noi[washout:washout+steps] @ params_trained_C['wout'].T + params_trained_C['bias_out'] #open loop with noise with noisy C
            Y_noi_C_ctc_m = X_noi_C_ctc_m[washout:washout+steps] @ params_trained_CTC_m['wout'].T + params_trained_CTC_m['bias_out'] #open loop with noise with ctc m C
            
            #transforming the array 
            y = np.asarray(Y_target[washout:washout+steps]).ravel()   
            
            y_noi = np.asarray(Y_noi).ravel()   
            y_noi_C_noi = np.asarray(Y_noi_C_noi).ravel()
            y_noi_C_ctc_m = np.asarray(Y_noi_C_ctc_m).ravel()
           
            
            trial_index = idx * trials_noise + j #to store the results correctly
            ###########################################################################################
           
           #PCA xcross
           
           ############################################################################################ 
            
            xcorrp_noi[i,trial_index]=xcorr_PCA(X_id,X_noi,washout,steps)
            xcorrp_noi_C_noi[i,trial_index]=xcorr_PCA(X_id,X_noi_C_noi,washout,steps)
            xcorrp_noi_C_ctc_m[i,trial_index]=xcorr_PCA(X_id,X_noi_C_ctc_m,washout,steps)
            
            
#Computing the mean  for each noise

mxcorrp_noi=np.mean(xcorrp_noi,axis=1)
mxcorrp_noi_C_noi=np.mean(xcorrp_noi_C_noi,axis=1)
mxcorrp_noi_C_ctc_m=np.mean(xcorrp_noi_C_ctc_m,axis=1)


#Now computing the standard deviation
std_xcorrp_noi=np.std(xcorrp_noi,axis=1)
std_xcorrp_noi_C_noi=np.std(xcorrp_noi_C_noi,axis=1)
std_xcorrp_noi_C_ctc_m=np.std(xcorrp_noi_C_ctc_m,axis=1)

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
    "lines.markersize": 10,

    # Grid
    "grid.alpha": 0.6,
    "grid.linestyle": "--",
    
    # ---- Axis spines ----
    "axes.linewidth": 1.8,      
    "axes.edgecolor": "black", 
})


#-----------------------------Plot xcorr PCA CTC m--------------------------------------
plt.figure(figsize=(8, 4), dpi=300)

plt.errorbar(
    k, mxcorrp_noi, yerr=std_xcorrp_noi,
    fmt='s', color='#B22222',alpha=0.8,
    ecolor="black", elinewidth=2, capsize=6,
    label="Without C"
)

plt.errorbar(
    k, mxcorrp_noi_C_noi, yerr=std_xcorrp_noi_C_noi,
    fmt='^', color='#6BAED6',alpha=0.8,
    ecolor="black", elinewidth=2, capsize=6,
    label=r"With $C_{noisy}$"
)

plt.errorbar(
    k, mxcorrp_noi_C_ctc_m, yerr=std_xcorrp_noi_C_ctc_m,
    fmt='o', color='#1F4E79',alpha=0.8,
    ecolor="black", elinewidth=2, capsize=6,
    label=r"With $C_{ctc}$"
)



plt.xlabel(" % Noise", size=20)
plt.ylabel("PCA Subspace Similarity",size=19)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# plt.savefig(
#            "plots/Figure6a.png",
#            dpi=300, bbox_inches='tight'
#        )

if corr is True:
    c='correlated'
else:
    c='uncorrelated'
plt.savefig(
    f"plots/PCAxcorr_CTCm_ol_N{N}_m{m}_trials{trials}_noisestep{args.noise_steps}_maxnoise{args.noise_max}_a{a}_steps{steps}_anew{a_new}_traintime{time_len}_new_{c}.png",
    dpi=300, bbox_inches='tight'
)

plt.savefig(
    f"plots/PCAxcorr_CTCm_ol_N{N}_m{m}_trials{trials}_noisestep{args.noise_steps}_maxnoise{args.noise_max}_a{a}_steps{steps}_anew{a_new}_traintime{time_len}_new_{c}.pdf",
    dpi=300, bbox_inches='tight'
)
plt.show()




























