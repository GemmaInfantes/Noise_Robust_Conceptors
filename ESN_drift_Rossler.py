#Imports
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from utils.rnn_utils import rnn_params
from utils.rnn_utils import forward_rnn
from utils.rnn_utils import forward_rnn_drift
from utils.rnn_utils import ridge
from utils.rnn_utils import compute_conceptor
from utils.rnn_utils import std_noise_func
from utils.rnn_utils import denoising_CTC


#Parameters that you can tune from terminal
parser = argparse.ArgumentParser()
parser.add_argument("--trials", type=int, default=50) #number of trails, for different seeds
parser.add_argument("--b", type=float, default=50) # % of deviation of b
parser.add_argument("--aperture", type=int, default=5) #conceptor aperture
parser.add_argument("--a_new", type=float, default=5) # aperture for the clean conceptors
parser.add_argument("--steps", type=int, default=400) #number of time steps used tu compute the NRMSE and the correlation
parser.add_argument("--steps_ol", type=int, default=100) #number of time steps used for the open loop
parser.add_argument("--time_len", type=int, default=3000) #number of time steps used to trine the reservoir
parser.add_argument("--N", type=int, default=200) #number of neurons
parser.add_argument("--spectral_radius", type=float, default=1.6) #spectral radius
parser.add_argument("--scaling", type=float, default=0.9) #scaling
parser.add_argument("--seedn", type=float, default=393) #seed for the noise 474  
parser.add_argument("--noise", type=float, default=10) #level of noise

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
reg=1#regularization parameter in the Ride regression
step=1 # number of steps that the model will predict
time_len=args.time_len #number of points that we want to us eto train the Reservoir, as our data has 4000 points in total
steps_ol=args.steps_ol #steps for the open loop
b=args.b
sparsity=None
a_new=args.a_new
noise=args.noise
steps=args.steps
seedn=args.seedn

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
    seed=1607
)

##################################################################################

#Running the open loop without noise and computing Wout

####################################################################################

#obtain matrix X1 (time, N) of internal states for all time points
X_id=forward_rnn(params, ut_train1, 42,x_init=None,autonomous=False,conceptor=None)

#noise std
std_noise=std_noise_func(X_id,noise)

#deviation of b
std_drift=bias_scaling*(b/100)

       

######################################################################################
        
#Running in open loop withot drift with noise without drift TRAINING NO C
        
#########################################################################################
        

X_noi=forward_rnn(params, ut_train1, seedn,None,False,None,std_noise)
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
C_ctc=denoising_CTC(params, ut_train1, std_noise, a_new)
       

X_noi_CTC=forward_rnn(params, ut_train1, seedn,None,False,C_ctc,std_noise)
# trained_model_new(X_noi[washout:],ut_train1,yt_train1,params_trained,washout,True,None,label)



#getting the final Wout with the ridge regression (Wout=Ytarget*X.T*(X*X.T+beta*I)^-1)
X_effective = X_noi_CTC[washout:]
yt_train_effective = yt_train1[washout:]
#showing training X
#training Wout with Xi 
params_trained_noi_CTC, mse = ridge(reg, X_effective, yt_train_effective,step,params) #this gives us the results for the trainning dataset
    



################################ OPEN LOOP ############################################


        
######################################################################################
        
#Running in open loop with drift without conceptor
        
#########################################################################################
        

X_d=forward_rnn_drift(params, ut_train1, seedn,None,False,None,std_drift,std_noise)


######################################################################################
        
#Running in open loop with dift with CTC conceptor
        
#########################################################################################
        

X_d_C_ctc=forward_rnn_drift(params, ut_train1, seedn,None,False,C_ctc,std_drift,std_noise)
  







washout=100 #first 100 steps without drift
#obtaining the outputs

#open loop
Y_target=yt_train1 #real data
Y_d = X_d[washout:washout+steps] @ params_trained_noi['wout'].T + params_trained_noi['bias_out'] #autonomous with noise 
Y_d_C_ctc = X_d_C_ctc[washout:washout+steps] @ params_trained_noi_CTC['wout'].T + params_trained_noi_CTC['bias_out'] #autonomous with noise with ctc C


        
#transforming the array 
y = np.asarray(Y_target[washout:washout+steps]).ravel()          
y_d = np.asarray(Y_d).ravel()   
y_d_C_ctc = np.asarray(Y_d_C_ctc).ravel()


##############################################################################################
        
#Plotting the outputs
        
############################################################################################
#limits for the plots      
x=100   
steps_in=0+x
steps_fin=steps+x-250
 
        
# Time axis
        
k1 = np.arange(len(y))
        
        
# Global style (paper-ready)
        
plt.rcParams.update({
            'font.size': 18,
            'axes.labelsize': 20,
            'axes.titlesize': 20,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            'lines.linewidth': 2.8,
            "axes.linewidth": 1.6,
            "axes.edgecolor": "black",
        })
        
# Consistent y-limits
y_min = min(y.min(), y_d.min(), y_d_C_ctc.min())
y_max = max(y.max(), y_d.max(), y_d_C_ctc.max())
        

 
 
##############################################################################################
        
#OPEN LOOP
        
############################################################################################
        

# Target vs CTC case
       
fig1, ax1 = plt.subplots(figsize=(8, 4), dpi=300)
        
ax1.plot(k1[steps_in:steps_fin], y[steps_in:steps_fin], color='black', linestyle='--', label='Target')
ax1.plot(k1[steps_in:steps_fin], y_d_C_ctc[steps_in:steps_fin], color='#1F4E79', label=r'With $C_{ctc}$')
        
ax1.set_xlim(steps_in, steps_fin - 1) 
ax1.set_ylim(y_min-0.05, y_max+0.05)
ax1.set_xlabel("Time steps (k)")
ax1.set_ylabel("Output $y(k)$")
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.legend(loc='upper left',frameon=True, framealpha=0.9)       
plt.tight_layout()
# plt.savefig(
#            "plots/Figure12b.pdf",
#            dpi=300, bbox_inches='tight'
#        )

plt.savefig(
           "plots/Figure12b.png",
           dpi=300, bbox_inches='tight'
       )
plt.show()
        

      
 
# Target vs without C
       
fig2, ax2 = plt.subplots(figsize=(8, 4), dpi=300)
        
ax2.plot(k1[steps_in:steps_fin], y[steps_in:steps_fin], color='black', linestyle='--', label='Target')
ax2.plot(k1[steps_in:steps_fin], y_d[steps_in:steps_fin], color='#B22222', label='Without $C$')
        
ax2.set_xlim(steps_in, steps_fin - 1) 
ax2.set_ylim(y_min-0.05, y_max+0.05)
ax2.set_xlabel("Time steps (k)")
ax2.set_ylabel("Output $y(k)$")
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.legend(loc='upper left',frameon=True, framealpha=0.9)         
plt.tight_layout()

# plt.savefig(
#            "plots/Figure12a.pdf",
#            dpi=300, bbox_inches='tight'
#        )

plt.savefig(
           "plots/Figure12a.png",
           dpi=300, bbox_inches='tight'
       )
    
plt.show()

























