#Imports
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import seaborn as sns

from utils.rnn_utils import rnn_params
from utils.rnn_utils import forward_rnn
from utils.rnn_utils import forward_rnn_drift
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
parser.add_argument("--b_max", type=float, default=110) #max %of b drift
parser.add_argument("--b_steps", type=float, default=10) # % steps of b drift
parser.add_argument("--aperture", type=int, default=5) #conceptor aperture
parser.add_argument("--a_new", type=float, default=5) # aperture for the clean conceptors
parser.add_argument("--steps", type=int, default=3000) #number of time steps used tu compute the NRMSE and the correlation
parser.add_argument("--time_len", type=int, default=3000) #number of time steps used to trine the reservoir
parser.add_argument("--N", type=int, default=200) #number of neurons
parser.add_argument("--noise_max", type=float, default=110) #max % of noise
parser.add_argument("--noise_step", type=float, default=10) #noise level step for the scan
parser.add_argument("--m", type=int, default=2) #m for the CTC
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
washout=20 # steps we wait until the network is stable, in order to show the results
reg=1 #regularization parameter in the Ride regression
step=1 # number of steps that the model will predict
time_len=args.time_len #number of points that we want to us eto train the Reservoir, as our data has 4000 points in total
sparsity=None
a_new=args.a_new
m=args.m
steps=args.steps
noise_step=args.noise_step
noise_max=args.noise_max


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

#parameters for the b scan

######################################################################
seed1=20 #seed for the seed generator :)
trials_noise=args.trials_noise #number of trials for the noise
trials_esn=args.trials_esn #number of trials for the esn
np.random.seed(seed1) #if we want this seed to be alwas de same  
seed_noise=np.random.randint(0, 2000, size=trials_noise) #seed for the random noise
seed_esn=np.random.randint(0, 2000, size=trials_esn) #seed for the ESN
# seed=[42]
steps=args.steps #time steps for the comparison
#b scan
b=np.arange(0,args.b_max,args.b_steps)
noise=np.arange(0,noise_max,noise_step)

#storing the error and correlation
trials=trials_noise*trials_esn
nrmse_d=np.empty((len(noise),trials),dtype=float)
nrmse_d_C_ctc=np.empty((len(noise),trials),dtype=float)
nrmse_d_C_avg=np.empty((len(noise),trials),dtype=float)

mnrmse_noi=np.empty((len(noise),len(b)),dtype=float)
mnrmse_noi_C_ctc=np.empty((len(noise),len(b)),dtype=float)
mnrmse_noi_C_avg=np.empty((len(noise),len(b)),dtype=float)


#starting the loop
for i in range(len(b)): #scan in increasing b deviation
    #deviation of b
    std_drift=bias_scaling*(b[i]/100)
    for k in range(len(noise)): #noise scan    
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
            
            #Running the open loop without noise and computing Wout
            
            ####################################################################################
            
            #obtain matrix X1 (time, N) of internal states for all time points
            X_id=forward_rnn(params, ut_train1, 42,x_init=None,autonomous=False,conceptor=None)
            
            #noise std
            std_noise=std_noise_func(X_id,noise[k])
            
                  
            for j in range(trials_noise): #trials of different noise realizations
                
                
                ######################################################################################
                        
                #Running in open loop withot drift with noise without drift TRAINING NO C
                        
                #########################################################################################
                        
                
                X_noi=forward_rnn(params, ut_train1, seed_noise[j],None,False,None,std_noise)
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
                        
                #Running in open loop withot drift with noise without drift TRAINING CTC 
                        
                #########################################################################################
                
                #first computing the CTC conceptor for each level of noise       
                C_ctc=denoising_CTC_m(params, ut_train1, std_noise, a_new,m)
                       
                
                X_noi_CTC=forward_rnn(params, ut_train1, seed_noise[j],None,False,C_ctc,std_noise)
                # trained_model_new(X_noi[washout:],ut_train1,yt_train1,params_trained,washout,True,None,label)
                
                
                
                #getting the final Wout with the ridge regression (Wout=Ytarget*X.T*(X*X.T+beta*I)^-1)
                X_effective = X_noi_CTC[washout:]
                yt_train_effective = yt_train1[washout:]
                #showing training X
                #training Wout with Xi 
                params_trained_noi_CTC, mse = ridge(reg, X_effective, yt_train_effective,step,params) #this gives us the results for the trainning dataset


                ######################################################################################

                #Running in open loop without drift with noise: TRAINING C_avg

                #########################################################################################

                C_avg=compute_conceptor_avg(params, ut_train1, std_noise, a_new)

                X_noi_C_avg=forward_rnn(params, ut_train1, seed_noise[j],None,False,C_avg,std_noise)

                X_effective = X_noi_C_avg[washout:]
                yt_train_effective = yt_train1[washout:]
                params_trained_noi_C_avg, mse = ridge(reg, X_effective, yt_train_effective,step,params)

                    
                
                
                
                ################################ OPEN LOOP ############################################
                
                
                        
                ######################################################################################
                        
                #Running in open loop with drift without conceptor
                        
                #########################################################################################
                        
                
                X_d=forward_rnn_drift(params, ut_train1, seed_noise[j],None,False,None,std_drift,std_noise)
                
                
                ######################################################################################
                        
                #Running in open loop with dift with CTC conceptor
                        
                #########################################################################################
                        
                
                X_d_C_ctc=forward_rnn_drift(params, ut_train1, seed_noise[j],None,False,C_ctc,std_drift,std_noise)


                ######################################################################################

                #Running in open loop with drift with C_avg conceptor

                #########################################################################################

                X_d_C_avg=forward_rnn_drift(params, ut_train1, seed_noise[j],None,False,C_avg,std_drift,std_noise)

                  
                
               
                
                steps_in=100 #first 100 steps without drift
                #obtaining the outputs
                
                #open loop
                Y_target=yt_train1 #real data
                Y_d = X_d[steps_in:steps_in+steps] @ params_trained_noi['wout'].T + params_trained_noi['bias_out'] #autonomous with noise 
                Y_d_C_ctc = X_d_C_ctc[steps_in:steps_in+steps] @ params_trained_noi_CTC['wout'].T + params_trained_noi_CTC['bias_out'] #open loop with noise and drift with CTC
                Y_d_C_avg = X_d_C_avg[steps_in:steps_in+steps] @ params_trained_noi_C_avg['wout'].T + params_trained_noi_C_avg['bias_out'] #open loop with noise and drift with C_avg
               
                
                
                        
                #transforming the array 
                y = np.asarray(Y_target[steps_in:steps_in+steps]).ravel()          
                y_d = np.asarray(Y_d).ravel()   
                y_d_C_ctc = np.asarray(Y_d_C_ctc).ravel()
                y_d_C_avg = np.asarray(Y_d_C_avg).ravel()
               
                trial_index = idx * trials_noise + j #to store the results correctly
                ###########################################################################################
               
               #NRMSE
               
               ############################################################################################ 
                # NRMSE
                nrmse_d[k,trial_index]        = NRMSE(y, y_d)
                nrmse_d_C_ctc[k,trial_index] = NRMSE(y, y_d_C_ctc)
                nrmse_d_C_avg[k,trial_index] = NRMSE(y, y_d_C_avg)

    
    # ------------------ Compute mean and std ------------------
    mnrmse_noi[:,i]       = np.mean(nrmse_d, axis=1)
    mnrmse_noi_C_ctc[:,i] = np.mean(nrmse_d_C_ctc, axis=1)
    mnrmse_noi_C_avg[:,i] = np.mean(nrmse_d_C_avg, axis=1)
   
    


##########################################################################################
# NRMSE Heatmaps
##########################################################################################



inff=0.1
supf=0.3

xticks_labels = [int(v) for v in b]
yticks_labels = [int(v) for v in noise]


#without C
plt.figure(figsize=(8,6))

ax = sns.heatmap(
    mnrmse_noi,
    annot=True,
    fmt=".2f", 
    cmap='magma_r',
    vmin=inff,
    vmax=supf,
    yticklabels=yticks_labels,
    xticklabels=xticks_labels,
    annot_kws={"size": 13},
    cbar_kws={"label": "NRMSE"}
)

ax.set_title(r"ESN without $C$", fontsize=20, pad=15)
ax.set_xlabel("% Drift", fontsize=20)
ax.set_ylabel("% Noise", fontsize=20, labelpad=15)

ax.set_xticklabels(xticks_labels, fontsize=16)
ax.tick_params(axis='y', labelsize=16)


cbar = ax.collections[0].colorbar
cbar.set_label("NRMSE", fontsize=20)
cbar.ax.tick_params(labelsize=16)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"plots/NRMSE_NoC_heatmap_N{N}_trials{trials}_noisestep{args.noise_step}_maxnoise{args.noise_max}_bstep{args.b_steps}_maxb{args.b_max}_a{a}_steps{steps}_anew{a_new}_traintime{time_len}_inf{inff}_sup{supf}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"plots/NRMSE_NoC_heatmap_N{N}_trials{trials}_noisestep{args.noise_step}_maxnoise{args.noise_max}_bstep{args.b_steps}_maxb{args.b_max}_a{a}_steps{steps}_anew{a_new}_traintime{time_len}_inf{inff}_sup{supf}.pdf", dpi=300, bbox_inches='tight')
plt.show()



#with CTC 
plt.figure(figsize=(8,6))

ax = sns.heatmap(
    mnrmse_noi_C_ctc,
    annot=True,
    fmt=".2f", 
    cmap='magma_r',
    vmin=inff,
    vmax=supf,
    yticklabels=yticks_labels,
    xticklabels=xticks_labels,
    annot_kws={"size": 13},
    cbar_kws={"label": "NRMSE"}
)

ax.set_title(r"ESN with $C_{ctc}$", fontsize=20, pad=15)
ax.set_xlabel("% Drift", fontsize=20)
ax.set_ylabel("% Noise", fontsize=20, labelpad=15)

ax.set_xticklabels(xticks_labels, fontsize=16)
ax.tick_params(axis='y', labelsize=16)


cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=16)
cbar.set_label("NRMSE", fontsize=20, labelpad=15)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"plots/NRMSE_CTC_heatmap_N{N}_m{m}_trials{trials}_noisestep{args.noise_step}_maxnoise{args.noise_max}_bstep{args.b_steps}_maxb{args.b_max}_a{a}_steps{steps}_anew{a_new}_traintime{time_len}_inf{inff}_sup{supf}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"plots/NRMSE_CTC_heatmap_N{N}_m{m}_trials{trials}_noisestep{args.noise_step}_maxnoise{args.noise_max}_bstep{args.b_steps}_maxb{args.b_max}_a{a}_steps{steps}_anew{a_new}_traintime{time_len}_inf{inff}_sup{supf}.pdf", dpi=300, bbox_inches='tight')
plt.show()

#with C_avg
plt.figure(figsize=(8,6))

ax = sns.heatmap(
    mnrmse_noi_C_avg,
    annot=True,
    fmt=".2f",
    cmap='magma_r',
    vmin=inff,
    vmax=supf,
    yticklabels=yticks_labels,
    xticklabels=xticks_labels,
    annot_kws={"size": 13},
    cbar_kws={"label": "NRMSE"}
)

ax.set_title(r"ESN with $C_{avg}$", fontsize=20, pad=15)
ax.set_xlabel("% Drift", fontsize=20)
ax.set_ylabel("% Noise", fontsize=20, labelpad=15)

ax.set_xticklabels(xticks_labels, fontsize=16)
ax.tick_params(axis='y', labelsize=16)

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=16)
cbar.set_label("NRMSE", fontsize=20, labelpad=15)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"plots/NRMSE_Cavg_heatmap_N{N}_trials{trials}_noisestep{args.noise_step}_maxnoise{args.noise_max}_bstep{args.b_steps}_maxb{args.b_max}_a{a}_steps{steps}_anew{a_new}_traintime{time_len}_inf{inff}_sup{supf}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"plots/NRMSE_Cavg_heatmap_N{N}_trials{trials}_noisestep{args.noise_step}_maxnoise{args.noise_max}_bstep{args.b_steps}_maxb{args.b_max}_a{a}_steps{steps}_anew{a_new}_traintime{time_len}_inf{inff}_sup{supf}.pdf", dpi=300, bbox_inches='tight')
plt.show()
