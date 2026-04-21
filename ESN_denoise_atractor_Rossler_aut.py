#Imports
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from utils.rnn_utils import rnn_params
from utils.rnn_utils import forward_rnn
from utils.rnn_utils import forward_rnn_comb
from utils.rnn_utils import ridge
from utils.rnn_utils import compute_conceptor
from utils.rnn_utils import std_noise_func
from utils.rnn_utils import denoising_CTC_m
from utils.utils import tau_autocorr
from utils.utils import embedding3D
from utils.utils import draw_data_box

#parameters that we can tune from terminal
parser = argparse.ArgumentParser()

parser.add_argument("--aperture", type=int, default=5) #conceptor aperture
parser.add_argument("--a_new", type=float, default=5) # aperture for the clen conceptors
parser.add_argument("--steps", type=int, default=3000) #number of time steps used tu compute 
parser.add_argument("--steps_ol", type=int, default=100) #number of time steps used for the open loop
parser.add_argument("--time_len", type=int, default=3000) #number of time steps used to trine the reservoir
parser.add_argument("--N", type=int, default=200) #number of neurons
parser.add_argument("--m", type=int, default=2) #realizations for ctc

args = parser.parse_args()


#########################################################################

##Random inicialization of the ESN

########################################################################

spectral_radius=1.6 #spectral radius of W
scaling=0.9 #input scaling
bias_scaling=0.4 #bias inside tanh
alpha=0.75 #Leakage
a=args.aperture #Aperture.  
N=args.N # Network size 256
washout=20 # steps we wait until the network is stable, in order to show the results
reg=1 #regularization parameter in the Ride regression
step=1 # number of steps that the model will predict
time_len=args.time_len #number of points that we want to us eto train the Reservoir, as our data has 4000 points in total
steps_ol=args.steps_ol #steps for the open loop
sparsity=None
a_new=args.a_new #aperture for the new conceptor CTC
m=args.m
steps=args.steps

#Input Signal Rössler x
data1 = pd.read_csv("Rossler_data/xRossler.txt", sep="\t",header=None,index_col=None)


#max lyaponov exponent of the x rossler time series in discrete time dt=0.3 
lyap = 0.0714*0.3

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




##############################################################################

#Showing the outputs and FFT for %noi noise

###############################################################################


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
    seed=992
)

##################################################################################

#Running the open loop without noise and computing Wout

####################################################################################

#obtain matrix X1 (time, N) of internal states for all time points
X_id=forward_rnn(params, ut_train1, 42,x_init=None,autonomous=False,conceptor=None)
#Compute model conceptors
C_id=compute_conceptor(X_id, a)




seed30=1379 #seed (noise,esn) (1379,992)  (1242,992)
noi=10
#standard deviation for the noise
std_noise=std_noise_func(X_id,noi)
C_id=compute_conceptor(X_id, a)  

######################################################################################
        
#Running in open loop with noise without conceptor
        
#########################################################################################
        

X_noi_ol=forward_rnn(params, ut_train1, seed30,None,False,None,std_noise)
# trained_model_new(X_noi[washout:],ut_train1,yt_train1,params_trained,washout,True,None,label)
#noisy conceptor
C_noi=compute_conceptor(X_noi_ol, a)
#getting the final Wout with the ridge regression (Wout=Ytarget*X.T*(X*X.T+beta*I)^-1)
X_effective = X_noi_ol[washout:]
yt_train_effective = yt_train1[washout:]
#showing training X
#training Wout with Xi 
params_trained_noi, mse = ridge(reg, X_effective, yt_train_effective,step,params) #this gives us the results for the trainning dataset


######################################################################################
        
#Running in open loop with noise without C noisy
        
#########################################################################################
        

X_noi_C_ol=forward_rnn(params, ut_train1, seed30,None,False,C_noi,std_noise)
# trained_model_new(X_noi[washout:],ut_train1,yt_train1,params_trained,washout,True,None,label)

X_effective = X_noi_C_ol[washout:]
yt_train_effective = yt_train1[washout:]
#showing training X
#training Wout with Xi 
params_trained_noi_C, mse = ridge(reg, X_effective, yt_train_effective,step,params) #this gives us the results for the trainning dataset
    
 


    
        
######################################################################################
        
#Running in open loop with noise with CTC conceptor
        
#########################################################################################
        
#first computing the CTC conceptor for each level of noise       
C_ctc=denoising_CTC_m(params, ut_train1, std_noise, a_new,m)
X_noi_C_ctc_ol=forward_rnn(params, ut_train1, seed30,None,False,C_ctc,std_noise)
# trained_model_new(X_noi_C_ctc[washout:],ut_train1,yt_train1,params_trained,washout,True,None,label)
#getting the final Wout with the ridge regression (Wout=Ytarget*X.T*(X*X.T+beta*I)^-1)
X_effective = X_noi_C_ctc_ol[washout:]
yt_train_effective = yt_train1[washout:]
#showing training X
#training Wout with Xi 
params_trained_CTC, mse = ridge(reg, X_effective, yt_train_effective,step,params) #this gives us the results for the trainning dataset
    


    
######################################################################################
        
#Running in autonomous mode with noise without conceptor
        
        
#########################################################################################
        
X_noi=forward_rnn_comb(params_trained_noi, ut_train1, seed30,steps_ol,None,None,std_noise)
# trained_model_new(X_noi[washout:],ut_train1,yt_train1,params_trained,washout,True,None,label)
        
        
        
        
######################################################################################
        
#Running in autonomous mode with noise with CTC conceptor
        
#########################################################################################
        
#first computing the CTC conceptor for each level of noise
        

X_noi_C_ctc=forward_rnn_comb(params_trained_CTC, ut_train1, seed30,steps_ol,None,C_ctc,std_noise)
        

######################################################################################
        
#Running in autonomous mode with noise with noisy conceptor
        
#########################################################################################
        
#first computing the CTC conceptor for each level of noise
        

X_noi_C_noi=forward_rnn_comb(params_trained_noi_C, ut_train1, seed30,steps_ol,None,C_noi,std_noise)
        




        
steps=3000    
#obtaining the outputs
Y_target=yt_train1[steps_ol:steps_ol+steps] #real data
Y_noi = X_noi[steps_ol:steps_ol+steps] @ params_trained_noi['wout'].T + params_trained_noi['bias_out'] #autonomous with noise 
Y_noi_C_ctc = X_noi_C_ctc[steps_ol:steps_ol+steps] @ params_trained_CTC['wout'].T + params_trained_CTC['bias_out'] #autonomous with noise with ctc C
Y_noi_C_noi = X_noi_C_noi[steps_ol:steps_ol+steps] @ params_trained_noi_C['wout'].T + params_trained_noi_C['bias_out'] #autonomous with noise with ctc C
 
        
# -------------------------------------------------------------------------
#   Attractor with embedding
# -----------------------------------------------------------------------------
# ------------------------------
# GLOBAL PAPER STYLE
# ------------------------------
plt.rcParams.update({
    'font.size': 26,
    'axes.labelsize': 28,
    'axes.titlesize': 28,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 26,
    'lines.linewidth': 3,
    "axes.linewidth": 2.0,
    "axes.edgecolor": "black"
})

# ------------------------------
# PROFESSIONAL COLOR PALETTE
# ------------------------------
colors = {
    'Target': '#2E8B57',       # green
    'Without C': '#B22222',    # red
    'C_noisy': '#6BAED6',      # light blue
    'C_ctc': '#1F4E79',        # dark blue
}

# ------------------------------
# Prepare signals (1D)
# ------------------------------
y_target  = np.asarray(Y_target[:steps,0]).ravel()
y_noC     = np.asarray(Y_noi[:steps,0]).ravel()
y_CTC     = np.asarray(Y_noi_C_ctc[:steps,0]).ravel()
y_C_noi   = np.asarray(Y_noi_C_noi[:steps,0]).ravel()  # C_noisy

# Compute tau using autocorrelation
tau, _ = tau_autocorr(y_target)

# ------------------------------
# Create 3D embeddings
# ------------------------------
embeddings = {
    "Target": embedding3D(y_target, tau),
    "Without C": embedding3D(y_noC, tau),
    'C_ctc': embedding3D(y_CTC, tau),
    'C_noisy': embedding3D(y_C_noi, tau)
}



# ------------------------------
# Plot each attractor separately
# ------------------------------
point_size = 5
alpha_val = 1
i=0

for label, (y0, y1, y2) in embeddings.items():

    fig = plt.figure(figsize=(12,11), dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    draw_data_box(ax, y0, y1, y2, face_alpha=0.05, edge_lw=3)

    ax.scatter(y0, y1, y2,
           s=point_size,
           color=colors[label],
           alpha=alpha_val,
           depthshade=True)  
    # Remove default axes, panes, ticks
    ax.set_axis_off()

    # View angle
    ax.view_init(elev=10, azim=10)
    ax.set_box_aspect([1.5,2,1.5])

    # ------------------------------
    # Axis labels on box corners
    # ------------------------------
    x_min, x_max = np.min(y0), np.max(y0)
    y_min, y_max = np.min(y1), np.max(y1)
    z_min, z_max = np.min(y2), np.max(y2)



    # Turn off grid
    plt.grid(False)
    plt.subplots_adjust(top=0.88, bottom=0.05, left=0.05, right=0.95)
    
    # plt.savefig(
    #            f"plots/Figure8_{i}.pdf",
    #            dpi=300, bbox_inches='tight'
    #        )
    
    plt.savefig(
               f"plots/Figure8_{i}.png",
               dpi=600, bbox_inches='tight'
           )
    plt.show()
    i=i+1