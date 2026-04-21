#Imports
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from itertools import product, combinations
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LightSource
from utils.rnn_utils import rnn_params
from utils.rnn_utils import forward_rnn
from utils.rnn_utils import ridge
from utils.rnn_utils import compute_conceptor




#parameters that we can tune from terminal
parser = argparse.ArgumentParser()
parser.add_argument("--noise_std", type=float, default=0) #% of deviation compared to x deviation in noise N(0,noise/100*std(x))
parser.add_argument("--seed", type=float, default=300) #seed


args = parser.parse_args()


#########################################################################

##Random inicialization of the ESN

########################################################################

spectral_radius=1.6#spectral radius of W
scaling=1.6 #input scaling
bias_scaling=0.1 #bias inside tanh
alpha=0.45 #Leakage 
a=20 #Aperture. 
N=3 #Network size 
# nu=2.5e-5 #Learning Rate 
# beta=0.9 #Control gain 
washout=20 # steps we wait until the network is stable, in order to show the results
reg=1 #regularization parameter in the Ride regression 
step=1 # number of steps that the model will predict
sparsity=None
noise_std=args.noise_std # % noise
seed=args.seed #seed

#Rossler dataset
time_len=800
data1 = pd.read_csv("Rossler_data/xRossler.txt", sep="\t",header=None,index_col=None)
data1=data1.values[:time_len]
data1=data1.reshape(-1, 1)

#obtaining the input and some parameters for params
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
    seed=args.seed
)

##################################################################################

#Running the open loop without noise 

####################################################################################

#obtain matrix X1 (time, N) of internal states for all time points
X1_id=forward_rnn(params, ut_train1,seed, x_init=None,autonomous=False,conceptor=None)
#Compute the conceptor
C_id=compute_conceptor(X1_id[washout:], a)

X_effective = X1_id[washout:]
yt_train_effective = yt_train1[washout:]
#showing training X
#training Wout with Xi 
params_trained_id, mse = ridge(reg, X_effective, yt_train_effective,step,params) #this gives us the results for the trainning dataset



#computing the outputs
#obtaining the outputs for open loop
Y_target=yt_train1[washout:] #real data
Y11_id = X1_id[washout:] @ params_trained_id['wout'].T + params_trained_id['bias_out']


#limits for the plots
steps_in=100
steps=300
# ALIGN outputs
yt = Y_target[steps_in:steps]
y_id= Y11_id[steps_in:steps]


# Real time axis (k)
k = np.arange(steps_in, steps_in + len(y_id))

#computing the svd of C_id
U, S, Vt = np.linalg.svd(C_id)


###########################################################################3
  
#plotting the output for open loop without noise (the one related with C_id)

###############################################################################


y1_id= Y11_id[steps_in:steps]
y= Y_target[steps_in:steps]
# ===================== Matplotlib Style =====================
plt.rcParams.update({
    'font.size': 22,           # general font size
    'axes.labelsize': 22,      # axis label size
    'axes.titlesize': 22,      # title size
    'xtick.labelsize': 20,     # x-axis tick labels
    'ytick.labelsize': 20,     # y-axis tick labels
    'lines.linewidth': 3,      # default line width for all plots
    'axes.linewidth': 1.8,     # thickness of axis borders
    'axes.edgecolor': 'black', # color of axis borders
})

# ===================== Colors =====================
turquoise = np.array([128, 0, 32])/255  # same color as the conceptor

# ===================== Figure =====================
plt.figure(figsize=(8,4), dpi=300)

# Target: black dashed line
plt.plot(k, y, '--', color='black', linewidth=2.5, label='Target')

# Prediction: turquoise solid line
plt.plot(k, y1_id, '-', color=turquoise, linewidth=3, label='Prediction')

# Legend
plt.legend(frameon=True, fontsize=20)

# Axis labels and grid
plt.xlabel('Time steps (k)')
plt.ylabel('Output y(k)')
plt.grid(True, linestyle=':', linewidth=0.8, alpha=0.7)

# Tight layout
plt.tight_layout()
plt.savefig(
           "plots/Figure2c.pdf",
           dpi=300, bbox_inches='tight'
       )
plt.show()


##################################################################################

#Conceptor 3d for N=3

####################################################################################
# ===================== CONFIG =====================
ellipse_base_color = np.array([128, 0, 32])/255  #color
alpha_surface = 0.25  #transparency surface
alpha_points = 0.9    #outside points

# ===================== FIGURE =====================
fig = plt.figure(figsize=(4,4), dpi=300)
ax = fig.add_subplot(111, projection='3d')

# ===================== 1. Extract points =====================
X = X1_id[washout:, 0]
Y = X1_id[washout:, 1]
Z = X1_id[washout:, 2]
points = np.vstack((X,Y,Z))

# ===================== 2. Inside ellipsoid =====================
C_inv = np.linalg.inv(C_id)
transformed = C_inv @ points
dists = np.sum(transformed**2, axis=0)
alphas = np.where(dists <= 1, 0.2, alpha_points)

# ===================== 3. Colors for points =====================
colors = np.zeros((len(X),4))
colors[:,:3] = ellipse_base_color
colors[:,3] = alphas

ax.scatter(
    X, Y, Z,
    color=colors,
    s=30,
    depthshade=False,
    edgecolors='none',
    linewidths=0,
    rasterized=False
)

# ===================== 4. Ellipsoid surface =====================
phi = np.linspace(0, np.pi, 60)
theta = np.linspace(0, 2*np.pi, 60)
phi, theta = np.meshgrid(phi, theta)

x = np.sin(phi)*np.cos(theta)
y = np.sin(phi)*np.sin(theta)
z = np.cos(phi)
sphere = np.vstack((x.flatten(),y.flatten(),z.flatten()))
ellipsoid = C_id @ sphere

X_surf = ellipsoid[0].reshape(phi.shape)
Y_surf = ellipsoid[1].reshape(phi.shape)
Z_surf = ellipsoid[2].reshape(phi.shape)

# ===================== 5. Advanced shading with LightSource =====================
ls = LightSource(azdeg=315, altdeg=45)
intensity = ls.shade(np.ones_like(Z_surf), cmap=plt.cm.Blues, vert_exag=1, blend_mode='soft')
facecolors = np.ones_like(intensity[...,:3]) * ellipse_base_color
facecolors = facecolors * intensity[...,:3]

ax.plot_surface(
    X_surf, Y_surf, Z_surf,
    facecolors=facecolors,
    edgecolor='none',
    linewidth=0.3,
    alpha=alpha_surface
)

# ===================== 6. Limits, grid, box =====================
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
ax.set_box_aspect([1,1,1])
ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
ax.set_axis_off()  

# ===================== 7. Optional transparent bounding box =====================
def draw_transparent_box(ax, lim=(-1,1), face_color='white', edge_color='black', face_alpha=0.08, edge_lw=2.0):
    r = [lim[0], lim[1]]
    faces = [
        [(r[0],r[0],r[0]),(r[1],r[0],r[0]),(r[1],r[1],r[0]),(r[0],r[1],r[0])],
        [(r[0],r[0],r[1]),(r[1],r[0],r[1]),(r[1],r[1],r[1]),(r[0],r[1],r[1])],
        [(r[0],r[0],r[0]),(r[1],r[0],r[0]),(r[1],r[0],r[1]),(r[0],r[0],r[1])],
        [(r[0],r[1],r[0]),(r[1],r[1],r[0]),(r[1],r[1],r[1]),(r[0],r[1],r[1])],
        [(r[0],r[0],r[0]),(r[0],r[1],r[0]),(r[0],r[1],r[1]),(r[0],r[0],r[1])],
        [(r[1],r[0],r[0]),(r[1],r[1],r[0]),(r[1],r[1],r[1]),(r[1],r[0],r[1])]
    ]
    # Fondo semitransparente
    poly = Poly3DCollection(faces, facecolors=face_color, edgecolors='none', alpha=face_alpha, zorder=0)
    ax.add_collection3d(poly)

    # Líneas siempre encima
    for s,e in combinations(product(r,r,r),2):
        if sum(abs(s[i]-e[i]) for i in range(3)) == r[1]-r[0]:
            ax.plot3D([s[0],e[0]], [s[1],e[1]], [s[2],e[2]], color=edge_color, linewidth=edge_lw, zorder=10)

# Llamada al final, después de todo lo demás
draw_transparent_box(ax, lim=(-1,1))

# ===================== 8. View =====================
ax.view_init(elev=30, azim=50)
plt.tight_layout(pad=0.2)
plt.savefig(
           "plots/Figure2a_Rossler.png",
           dpi=300, bbox_inches='tight'
       )
plt.show()

