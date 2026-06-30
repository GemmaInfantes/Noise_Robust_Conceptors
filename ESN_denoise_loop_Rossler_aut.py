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
from utils.rnn_utils import compute_conceptor_avg
from utils.utils import xcorr_new
from utils.utils import smooth_spectrum
from utils.utils import prediction_horizon
from utils.utils import power_spectrum

#parameters that we can tune from terminal
parser = argparse.ArgumentParser()

parser.add_argument("--trials_noise", type=int, default=10) #number of trails, for different seeds
parser.add_argument("--trials_esn", type=int, default=10) #number of trails, for different seeds
parser.add_argument("--noise_max", type=int, default=105) #max % of the deviation of the noise
parser.add_argument("--noise_steps", type=int, default=10) #steps of % of the deviation of the noise
parser.add_argument("--aperture", type=int, default=5) #conceptor aperture
parser.add_argument("--a_new", type=float, default=5) # aperture for the clen conceptors
parser.add_argument("--steps", type=int, default=3000) #number of time steps used tu compute 
parser.add_argument("--steps_ol", type=int, default=100) #number of time steps used for the open loop
parser.add_argument("--time_len", type=int, default=3000) #number of time steps used to trine the reservoir
parser.add_argument("--error_th", type=float, default=0.5) # NRMSE threshold for the prediction horizont
parser.add_argument("--steps_th", type=int, default=20) # steps threshold for the prediction horizont
parser.add_argument("--seed", type=int, default=20) #seed for the seed generator :)
parser.add_argument("--N", type=int, default=200) #number of neurons
parser.add_argument("--window", type=int, default=50) #window fo ph
parser.add_argument("--m", type=int, default=2) #realizations for ctc
parser.add_argument("--f_max", type=float, default=0.175) #maximum for f
parser.add_argument("--f_min", type=float, default=0.025) #minimum for f
parser.add_argument(
    "--corr",
    type=lambda x: str(x).lower() in ['true', '1', 'yes', 'y'],
    default=False,
) #True: correlated noise, False: uncorrelated noise

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
steps_th=args.steps_th
error_th=args.error_th
sparsity=None
window=args.window
a_new=args.a_new #aperture for the new conceptor CTC
m=args.m
flog_min=args.f_min
flog_max=args.f_max
corr=args.corr

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

xcorro_noi1=np.empty((len(k),trials),dtype=float)
xcorro_noi_C_noi1=np.empty((len(k),trials),dtype=float)
xcorro_noi_C_ctc1_m=np.empty((len(k),trials),dtype=float)
xcorro_noi_C_avg=np.empty((len(k),trials),dtype=float)

ph_noi=np.empty((len(k),trials),dtype=float)
ph_noi_C_noi=np.empty((len(k),trials),dtype=float)
ph_noi_C_ctc_m=np.empty((len(k),trials),dtype=float)
ph_noi_C_avg=np.empty((len(k),trials),dtype=float)

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

        #Running the open loop without noise and computing Wout

        ####################################################################################

        #obtain matrix X1 (time, N) of internal states for all time points
        X_id=forward_rnn(params, ut_train1, 42,x_init=None,autonomous=False,conceptor=None)
        #Compute model conceptors
        C_id=compute_conceptor(X_id, a)
        #computing the standard deviation for the noise
        std_noise=std_noise_func(X_id,k[i])
        
        
        for j in range(trials_noise): #trials of different noise realizations
            ######################################################################################
            
            #Running in open loop with noise without conceptor
            
            #########################################################################################
            
            label="Open loop  with Noise, without C"
            X_noi_ol=forward_rnn(params, ut_train1, seed_noise[j],None,False,None,std_noise,corr=corr)
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
            
            #Running in open loop with noise with noisy conceptor
            
            #########################################################################################
            
            
            X_noi_C_noi_ol=forward_rnn(params, ut_train1, seed_noise[j],None,False,C_noi,std_noise,corr=corr)
            # trained_model_new(X_noi_C[washout:],ut_train1,yt_train1,params_trained,washout,True,None,label)
        
            #getting the final Wout with the ridge regression (Wout=Ytarget*X.T*(X*X.T+beta*I)^-1)
            X_effective = X_noi_C_noi_ol[washout:]
            yt_train_effective = yt_train1[washout:]
            #showing training X
            #training Wout with Xi 
            params_trained_C, mse = ridge(reg, X_effective, yt_train_effective,step,params) #this gives us the results for the trainning dataset
        
            
            ######################################################################################
            
            #Running in open loop with noise with CTC m conceptor
            
            #########################################################################################
            
            #first computing the CTC conceptor for each level of noise
            
            C_ctc_m=denoising_CTC_m(params, ut_train1, std_noise, a_new,m,corr=corr)
            
            X_noi_C_ctc_ol_m=forward_rnn(params, ut_train1, seed_noise[j],None,False,C_ctc_m,std_noise,corr=corr)
            # trained_model_new(X_noi_C_ctc[washout:],ut_train1,yt_train1,params_trained,washout,True,None,label)
            #getting the final Wout with the ridge regression (Wout=Ytarget*X.T*(X*X.T+beta*I)^-1)
            X_effective = X_noi_C_ctc_ol_m[washout:]
            yt_train_effective = yt_train1[washout:]
            #showing training X
            #training Wout with Xi 
            params_trained_CTC_m, mse = ridge(reg, X_effective, yt_train_effective,step,params) #this gives us the results for the trainning dataset

            
            ######################################################################################
            
            #Running in open loop with noise with avg conceptor
            
            #########################################################################################
            
            #first computing the CTC conceptor for each level of noise
            
            C_avg=compute_conceptor_avg(params, ut_train1, std_noise, a_new,corr=corr)
            
            X_noi_C_avg_ol=forward_rnn(params, ut_train1, seed_noise[j],None,False,C_avg,std_noise,corr=corr)
            # trained_model_new(X_noi_C_ctc[washout:],ut_train1,yt_train1,params_trained,washout,True,None,label)
            #getting the final Wout with the ridge regression (Wout=Ytarget*X.T*(X*X.T+beta*I)^-1)
            X_effective = X_noi_C_avg_ol[washout:]
            yt_train_effective = yt_train1[washout:]
            #showing training X
            #training Wout with Xi 
            params_trained_C_avg, mse = ridge(reg, X_effective, yt_train_effective,step,params) #this gives us the results for the trainning dataset
        
        
    
        
            ######################################################################################
            
            #Running in autonomous mode with noise without conceptor
            
            
            #########################################################################################
            
            
            X_noi=forward_rnn_comb(params_trained_noi, ut_train1, seed_noise[j],steps_ol,None,None,std_noise,corr=corr)
            # trained_model_new(X_noi[washout:],ut_train1,yt_train1,params_trained,washout,True,None,label)
            
            
            
            ######################################################################################
            
            #Running in autonomous mode with noise with noisy conceptor
            
            #########################################################################################
            
            
            X_noi_C_noi=forward_rnn_comb(params_trained_C, ut_train1, seed_noise[j],steps_ol,None,C_noi,std_noise,corr=corr)
           
            
           
            ######################################################################################
            
            #Running in autonomous mode with noise with CTC m conceptor
            
            #########################################################################################
            
            X_noi_C_ctc_m=forward_rnn_comb(params_trained_CTC_m, ut_train1, seed_noise[j],steps_ol,None,C_ctc_m,std_noise,corr=corr)
           
   
            ######################################################################################
            
            #Running in autonomous mode with noise with avg conceptor
            
            #########################################################################################
            
            X_noi_C_avg=forward_rnn_comb(params_trained_C_avg, ut_train1, seed_noise[j],steps_ol,None,C_avg,std_noise,corr=corr)
           
                     
            #obtaining the outputs
            Y_target=yt_train1[steps_ol:] #real data
            Y_noi = X_noi[steps_ol:] @ params_trained_noi['wout'].T + params_trained_noi['bias_out'] #autonomous with noise 
            Y_noi_C_noi = X_noi_C_noi[steps_ol:] @ params_trained_C['wout'].T + params_trained_C['bias_out'] #autonomous with noise with noisy C
            Y_noi_C_ctc_m = X_noi_C_ctc_m[steps_ol:] @ params_trained_CTC_m['wout'].T + params_trained_CTC_m['bias_out'] #autonomous with noise with ctc C
            Y_noi_C_avg = X_noi_C_avg[steps_ol:] @ params_trained_C_avg['wout'].T + params_trained_C_avg['bias_out'] #autonomous with noise with avg C
            
            #transforming the array 
            y = np.asarray(Y_target).ravel()   
            
            y_noi = np.asarray(Y_noi).ravel()   
            y_noi_C_noi = np.asarray(Y_noi_C_noi).ravel()
            y_noi_C_ctc_m = np.asarray(Y_noi_C_ctc_m).ravel()
            y_noi_C_avg = np.asarray(Y_noi_C_avg).ravel()
            
    
            trial_index = idx * trials_noise + j #to store the results correctly
            
            ###########################################################################################
            
            #FFT xcorr
            
            ############################################################################################
    
            xcorro_noi1[i,trial_index]=xcorr_new(y,y_noi,0,steps,dt)
            xcorro_noi_C_noi1[i,trial_index]=xcorr_new(y,y_noi_C_noi,0,steps,dt)
            xcorro_noi_C_ctc1_m[i,trial_index]=xcorr_new(y,y_noi_C_ctc_m,0,steps,dt)
            xcorro_noi_C_avg[i,trial_index]=xcorr_new(y,y_noi_C_avg,0,steps,dt)
            
            
            ###########################################################################################
            
            #Prediction horizont
            
            ############################################################################################
    
            _,ph_noi[i,trial_index]= prediction_horizon(y, y_noi,window,None,0,error_th,steps_th,lyap)
            _,ph_noi_C_noi[i,trial_index]= prediction_horizon(y, y_noi_C_noi,window,None,0,error_th,steps_th,lyap)
            _,ph_noi_C_ctc_m[i,trial_index]= prediction_horizon(y, y_noi_C_ctc_m,window,None,0,error_th,steps_th,lyap)
            _,ph_noi_C_avg[i,trial_index]= prediction_horizon(y, y_noi_C_avg,window,None,0,error_th,steps_th,lyap)
            
    
    
            
            
###################################################################################
# BOXPLOTS
###################################################################################

if corr:
    c = "correlated"
else:
    c = "uncorrelated"


plt.rcParams.update({
    "figure.figsize": (8, 4),

    # Axis labels
    "axes.labelsize": 20,
    "axes.titlesize": 20,

    # Tick labels
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,

    # Legend
    "legend.fontsize": 18,

    # Lines and markers
    "lines.linewidth": 2,
    "lines.markersize": 10,

    # Grid
    "grid.alpha": 0.4,
    "grid.linestyle": "--",

    # Axis spines
    "axes.linewidth": 1.8,
    "axes.edgecolor": "black",
})


# Common properties
positions = np.arange(len(k))*1.2
tick_step = 2

box_width = 0.18
group_width = 0.25

flierprops = dict(
    marker="o",
    markersize=3,
    alpha=1,
    markeredgecolor="black"
)


def add_boxplot(data, box_positions, color):
    """
    Add a boxplot using the common format.
    """
    return plt.boxplot(
        data,
        positions=box_positions,
        widths=box_width,
        patch_artist=True,
        showfliers=True,
        whis=1.5,
        flierprops=flierprops,
        boxprops=dict(
            facecolor=color,
            edgecolor="black",
            linewidth=1.5,
            alpha=1.0
        ),
        medianprops=dict(
            color="black",
            linewidth=1.5
        ),
        whiskerprops=dict(
            color="black",
            linewidth=1.5
        ),
        capprops=dict(
            color="black",
            linewidth=1.5
        )
    )


def format_boxplot(ylabel):
    """
    Apply the same axis format to every figure.
    """
    plt.xticks(
        positions[::tick_step],
        k[::tick_step]
    )

    plt.xlabel("% Noise", fontsize=20)
    plt.ylabel(ylabel, fontsize=20)

    plt.grid(
        True,
        linestyle="--",
        alpha=0.4
    )

    plt.tight_layout()


###################################################################################
# PREDICTION HORIZON DATA
###################################################################################

data_ph_noC = [
    ph_noi[i, :]
    for i in range(len(k))
]

data_ph_Cnoi = [
    ph_noi_C_noi[i, :]
    for i in range(len(k))
]

data_ph_Cctc = [
    ph_noi_C_ctc_m[i, :]
    for i in range(len(k))
]

data_ph_Cavg = [
    ph_noi_C_avg[i, :]
    for i in range(len(k))
]


###################################################################################
# 1. PREDICTION HORIZON: WITHOUT C, C_NOISY AND C_CTC
###################################################################################

plt.figure(figsize=(8, 4), dpi=300)

bp1 = add_boxplot(
    data_ph_noC,
    positions - group_width,
    "#B22222"
)

bp2 = add_boxplot(
    data_ph_Cnoi,
    positions,
    "#6BAED6"
)

bp3 = add_boxplot(
    data_ph_Cctc,
    positions + group_width,
    "#1F4E79"
)

format_boxplot("Prediction Horizon")

plt.legend(
    [
        bp1["boxes"][0],
        bp2["boxes"][0],
        bp3["boxes"][0]
    ],
    [
        "Without C",
        r"With $C_{noisy}$",
        r"With $C_{ctc}$"
    ],
    frameon=True
)

plt.savefig(
    f"plots/PH_CTCm_boxplot_trials{trials}_N{N}_m{m}"
    f"_noisestep{args.noise_steps}_maxnoise{args.noise_max}"
    f"_a{a}_anew{a_new}_window{window}_error{error_th}"
    f"_stepsth{steps_th}_traintime{time_len}_new_{c}.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    f"plots/PH_CTCm_boxplot_trials{trials}_N{N}_m{m}"
    f"_noisestep{args.noise_steps}_maxnoise{args.noise_max}"
    f"_a{a}_anew{a_new}_window{window}_error{error_th}"
    f"_stepsth{steps_th}_traintime{time_len}_new_{c}.pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


###################################################################################
# 2. PREDICTION HORIZON: INCLUDING C_AVG
###################################################################################

plt.figure(figsize=(9, 5), dpi=300)

# Four equally spaced boxes around each noise value
offsets_4 = np.array([
    -1.5,
    -0.5,
     0.5,
     1.5
]) * group_width

bp1 = add_boxplot(
    data_ph_noC,
    positions + offsets_4[0],
    "#B22222"
)

bp2 = add_boxplot(
    data_ph_Cnoi,
    positions + offsets_4[1],
    "#6BAED6"
)

bp3 = add_boxplot(
    data_ph_Cctc,
    positions + offsets_4[2],
    "#1F4E79"
)

bp4 = add_boxplot(
    data_ph_Cavg,
    positions + offsets_4[3],
    "#009E9A"
)

format_boxplot("Prediction Horizon")

plt.legend(
    [
        bp1["boxes"][0],
        bp2["boxes"][0],
        bp3["boxes"][0],
        bp4["boxes"][0]
    ],
    [
        "Without C",
        r"With $C_{noisy}$",
        r"With $C_{ctc}$",
        r"With $C_{avg}$"
    ],
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=4,
    frameon=False,
    # fontsize=13,
    handlelength=0.7,
    handleheight=0.6,
    handletextpad=0.35,
    columnspacing=0.9
)

plt.subplots_adjust(top=0.80)


plt.savefig(
    f"plots/PH_Cavg_boxplot_trials{trials}_N{N}_m{m}"
    f"_noisestep{args.noise_steps}_maxnoise{args.noise_max}"
    f"_a{a}_anew{a_new}_window{window}_error{error_th}"
    f"_stepsth{steps_th}_traintime{time_len}_new_{c}.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    f"plots/PH_Cavg_boxplot_trials{trials}_N{N}_m{m}"
    f"_noisestep{args.noise_steps}_maxnoise{args.noise_max}"
    f"_a{a}_anew{a_new}_window{window}_error{error_th}"
    f"_stepsth{steps_th}_traintime{time_len}_new_{c}.pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


###################################################################################
# CROSS-CORRELATION DATA
###################################################################################

data_xcorr_noC = [
    xcorro_noi1[i, :]
    for i in range(len(k))
]

data_xcorr_Cnoi = [
    xcorro_noi_C_noi1[i, :]
    for i in range(len(k))
]

data_xcorr_Cctc = [
    xcorro_noi_C_ctc1_m[i, :]
    for i in range(len(k))
]

data_xcorr_Cavg = [
    xcorro_noi_C_avg[i, :]
    for i in range(len(k))
]


###################################################################################
# 3. CROSS CORRELATION: WITHOUT C, C_NOISY AND C_CTC
###################################################################################

plt.figure(figsize=(8, 4), dpi=300)

bp1 = add_boxplot(
    data_xcorr_noC,
    positions - group_width,
    "#B22222"
)

bp2 = add_boxplot(
    data_xcorr_Cnoi,
    positions,
    "#6BAED6"
)

bp3 = add_boxplot(
    data_xcorr_Cctc,
    positions + group_width,
    "#1F4E79"
)

format_boxplot("Cross Correlation")

plt.legend(
    [
        bp1["boxes"][0],
        bp2["boxes"][0],
        bp3["boxes"][0]
    ],
    [
        "Without C",
        r"With $C_{noisy}$",
        r"With $C_{ctc}$"
    ],
    frameon=True
)

plt.savefig(
    f"plots/newFFTxcorr_CTCm_boxplot_trials{trials}_N{N}_m{m}"
    f"_noisestep{args.noise_steps}_maxnoise{args.noise_max}"
    f"_a{a}_steps{steps}_anew{a_new}_traintime{time_len}_{c}.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    f"plots/newFFTxcorr_CTCm_boxplot_trials{trials}_N{N}_m{m}"
    f"_noisestep{args.noise_steps}_maxnoise{args.noise_max}"
    f"_a{a}_steps{steps}_anew{a_new}_traintime{time_len}_{c}.pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


###################################################################################
# 4. CROSS CORRELATION: INCLUDING C_AVG
###################################################################################

# box_width=0.1

plt.figure(figsize=(9, 5), dpi=300)

bp1 = add_boxplot(
    data_xcorr_noC,
    positions + offsets_4[0],
    "#B22222"
)

bp2 = add_boxplot(
    data_xcorr_Cnoi,
    positions + offsets_4[1],
    "#6BAED6"
)

bp3 = add_boxplot(
    data_xcorr_Cctc,
    positions + offsets_4[2],
    "#1F4E79"
)

bp4 = add_boxplot(
    data_xcorr_Cavg,
    positions + offsets_4[3],
    "#009E9A"
)

format_boxplot("Cross Correlation")

plt.legend(
    [
        bp1["boxes"][0],
        bp2["boxes"][0],
        bp3["boxes"][0],
        bp4["boxes"][0]
    ],
    [
        "Without C",
        r"With $C_{noisy}$",
        r"With $C_{ctc}$",
        r"With $C_{avg}$"
    ],
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=4,
    frameon=False,
    # fontsize=13,
    handlelength=0.7,
    handleheight=0.6,
    handletextpad=0.35,
    columnspacing=0.9
)

plt.subplots_adjust(top=0.80)

plt.savefig(
    f"plots/newFFTxcorr_Cavg_boxplot_trials{trials}_N{N}_m{m}"
    f"_noisestep{args.noise_steps}_maxnoise{args.noise_max}"
    f"_a{a}_steps{steps}_anew{a_new}_traintime{time_len}_{c}.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    f"plots/newFFTxcorr_Cavg_boxplot_trials{trials}_N{N}_m{m}"
    f"_noisestep{args.noise_steps}_maxnoise{args.noise_max}"
    f"_a{a}_steps{steps}_anew{a_new}_traintime{time_len}_{c}.pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()



##############################################################################

#Showing the outputs and FFT for certain noise

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
    seed=1142
)

##################################################################################

#Running the open loop without noise and computing Wout

####################################################################################

#obtain matrix X1 (time, N) of internal states for all time points
X_id=forward_rnn(params, ut_train1, 42,x_init=None,autonomous=False,conceptor=None)
#Compute model conceptors
C_id=compute_conceptor(X_id, a)




seed30=271 #seed 
noi=10
#standard deviation for the noise
std_noise=std_noise_func(X_id,noi)
C_id=compute_conceptor(X_id, a)  

######################################################################################
        
#Running in open loop with noise without conceptor
        
#########################################################################################
        
label="Open loop  with Noise, without C"
X_noi_ol=forward_rnn(params, ut_train1, seed30,None,False,None,std_noise,corr=corr)
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
        
#Running in open loop with noise with CTC conceptor
        
#########################################################################################
        
#first computing the CTC conceptor for each level of noise       
C_ctc=denoising_CTC_m(params, ut_train1, std_noise, a_new,m,corr=corr)
label="Open loop  with Noise, with CTC C"
X_noi_C_ctc_ol=forward_rnn(params, ut_train1, seed30,None,False,C_ctc,std_noise,corr=corr)
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
        
X_noi=forward_rnn_comb(params_trained_noi, ut_train1, seed30,steps_ol,None,None,std_noise,corr=corr)
# trained_model_new(X_noi[washout:],ut_train1,yt_train1,params_trained,washout,True,None,label)
        
        
        
        
######################################################################################
        
#Running in autonomous mode with noise with CTC conceptor
        
#########################################################################################
        
#first computing the CTC conceptor for each level of noise
        

X_noi_C_ctc=forward_rnn_comb(params_trained_CTC, ut_train1, seed30,steps_ol,None,C_ctc,std_noise,corr=corr)
        

        

        
     
#obtaining the outputs
Y_target=yt_train1[steps_ol:] #real data
Y_noi = X_noi[steps_ol:] @ params_trained_noi['wout'].T + params_trained_noi['bias_out'] #autonomous with noise 
Y_noi_C_ctc = X_noi_C_ctc[steps_ol:] @ params_trained_CTC['wout'].T + params_trained_CTC['bias_out'] #autonomous with noise with ctc C

        
#transforming the array 
y = np.asarray(Y_target).ravel()   
y_noi = np.asarray(Y_noi).ravel()   
y_noi_C_ctc = np.asarray(Y_noi_C_ctc).ravel()

 
     
###########################################################################################
        
#Prediction horizont
        
############################################################################################

_,ph_noi30= prediction_horizon(y, y_noi,window,None,0,error_th,steps_th,lyap)
_,ph_noi_C_ctc30= prediction_horizon(y, y_noi_C_ctc,window,None,0,error_th,steps_th,lyap)
        
        
# -------------------------------
# Flatten arrays
# -------------------------------
y_target = np.asarray(Y_target).ravel()
y_noC = np.asarray(Y_noi).ravel()          # Without C
y_CTC = np.asarray(Y_noi_C_ctc).ravel()   # With CTC
   
# -------------------------------
# Parameters
# -------------------------------
steps_in = 0
steps_fin = 150
k1 = np.arange(len(y_target))  # Time axis
        
# Global style (paper-ready)
plt.rcParams.update({
            'font.size': 18,
            'axes.labelsize': 20,
            'axes.titlesize': 20,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            'lines.linewidth': 1.7,
            "axes.linewidth": 1.6,
            "axes.edgecolor": "black",
            'legend.fontsize': 18
        })
        
# Consistent y-limits for temporal plot
y_min = min(y_target.min(), y_noC.min(), y_CTC.min())
y_max = max(y_target.max(), y_noC.max(), y_CTC.max())
        
#prediction horizont lines
ph_th=ph_noi30
ph_th_ctc=ph_noi_C_ctc30


# -------------------------------
# Figure 1: Temporal output comparison
# -------------------------------
fig1, ax1 = plt.subplots(figsize=(8, 4), dpi=300)

# -------------------------------
# Prediction horizon shading
# -------------------------------

# Shade up to PH (WITHOUT C)
if steps_in <= ph_th <= steps_fin:
    ax1.axvspan(
        steps_in, ph_th,
        color='#B22222',
        alpha=0.18,
        zorder=1
    )
    ax1.axvline(
        ph_th,
        color='#B22222',
        linewidth=3.5,
        zorder=5
    )

# Shade up to PH (WITH CTC)
if steps_in <= ph_th_ctc <= steps_fin:
    ax1.axvspan(
        steps_in, ph_th_ctc,
        color='#1F4E79',
        alpha=0.18,
        zorder=1
    )
    ax1.axvline(
        ph_th_ctc,
        color='#1F4E79',
        linewidth=3.5,
        zorder=5
    )

# Plot lines
ax1.plot(k1[steps_in:steps_fin], y_target[steps_in:steps_fin], color='black', linestyle='--', label='Target')
ax1.plot(k1[steps_in:steps_fin], y_noC[steps_in:steps_fin], color='#B22222', label='Without C',linewidth=2)
ax1.plot(k1[steps_in:steps_fin], y_CTC[steps_in:steps_fin], color='#1F4E79', label=r'With $C_{ctc}$',linewidth=2)

# Axes limits and labels
ax1.set_xlim(steps_in, steps_fin - 1)
ax1.set_ylim(y_min - 0.05, y_max + 0.05)
ax1.set_xlabel("Time steps (k)")
ax1.set_ylabel("Output $y(k)$")

# Grid
ax1.grid(True, linestyle='--', alpha=0.4)

# Legend in the upper right corner
ax1.legend(loc='upper right', frameon=True, framealpha=0.9)

# Layout, save and show
plt.tight_layout()
plt.savefig("plots/Figure8a.png", dpi=300, bbox_inches='tight')  
# plt.savefig("plots/Figure8a.pdf", dpi=300, bbox_inches='tight')       
plt.show()




        
# -------------------------------
# Figure 2.1: logaritmic Frequency spectrum comparison
# -------------------------------
# Compute and smooth FFT
p = False
f_target, m_target = power_spectrum(y_target, dt)
f_noC, m_noC = power_spectrum(y_noC, dt)
f_CTC, m_CTC = power_spectrum(y_CTC, dt)
   
norm=5
m_target = smooth_spectrum(m_target, norm).squeeze()
m_noC = smooth_spectrum(m_noC, norm).squeeze()
m_CTC = smooth_spectrum(m_CTC, norm).squeeze()
f_target = f_target.squeeze()
f_noC = f_noC.squeeze()
f_CTC = f_CTC.squeeze()
        
# Consistent y-limits for FFT
fft_min = min(m_target[1:].min(), m_noC[1:].min(), m_CTC[1:].min())
fft_max = max(m_target[1:].max(), m_noC[1:].max(), m_CTC[1:].max())
        
fig2, ax2 = plt.subplots(figsize=(8, 4), dpi=300)
ax2.semilogy(f_target[1:], m_target[1:], '--', color='black', label='Target')
ax2.semilogy(f_noC[1:], m_noC[1:], '-', color='#B22222', label='Without C')
ax2.semilogy(f_CTC[1:], m_CTC[1:], '-', color='#1F4E79', label=r'With $C_{ctc}$')
ax2.set_xlabel("Frequency (1/k)")
ax2.set_ylabel("Power Spectrum")
ax2.set_ylim(fft_min*+10, fft_max*5)
ax2.set_xlim(0.01, 0.75)
# ax2.set_xlim(0, 0.5)
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.legend(frameon=True, framealpha=0.9)
plt.tight_layout()


plt.savefig(
           f"plots/Figure9a_{c}.png",
           dpi=300, bbox_inches='tight'
       )

# plt.savefig(
#            "plots/Figure9a.pdf",
#            dpi=300, bbox_inches='tight'
#        )
        
plt.show()








