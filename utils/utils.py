import numpy as np
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from scipy.signal import correlate, correlation_lags
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from itertools import product, combinations

   


def NRMSE(y1,y2):
    """
    Normalized Root Mean Square Error
    
   
    - y1 (numpy.ndarray): Reference output 
    - y2 (numpy.ndarray):Predicted output
    
    Returns:
    - nrsme: Normalized Root Mean Square Error between y1 and y2

    """
    y1=np.ravel(y1)
    y2=np.ravel(y2)
    
    #fist computing mse
    mse=np.mean((y2 - y1)**2)
    
    #secondly rmse
    rmse=np.sqrt(mse)
    
    #computing the mean of y1
    # mean=np.mean(y1)
    std = np.std(y1)
    rang = np.max(y1) - np.min(y1)
    
    # Normalization
    if std != 0:
        nrmse = rmse / std
    elif rang != 0:
        nrmse = rmse / rang
    else:
        nrmse = np.inf  
    
    return nrmse




    
    
    
def FFT(y, dt, label, plot=True):
    """
    Compute the frequency spectrum of a signal, ignoring the DC component.

    Args:
    - y: array-like, input time series
    - dt: float, sampling interval
    - label: str, label for plot title
    - plot: bool, whether to show the plot

    Returns:
    - freqs_noDC: array, positive frequencies (Hz) excluding 0
    - magnitude_noDC: array, corresponding amplitudes
    """
  

    # Flatten y to 1D
    y = y.flatten()
    n = len(y)

    # Compute FFT
    fft_values = np.fft.fft(y)
    freqs = np.fft.fftfreq(n, d=dt)

    # Take only positive frequencies
    half_n = n // 2
    freqs = freqs[1:half_n]                  # skip DC (freq 0)
    magnitude = np.abs(fft_values[1:half_n]) * 2 / n  # normalize amplitude

    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(freqs, magnitude, color='b')
        plt.title(f"Frequency Spectrum: {label}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return freqs, magnitude


    
def PCA_3D(X):
    """
    Visualize the PCA of any matrix, and project other the conceptor in the PCA space
    Args:
    - X: (numpy.ndarray): Input matrix 
    
    
    Returns:
    - PCA components
    """
    # centering the X
    X_centered = X - np.mean(X, axis=0)
    # X_centered=X
    #Obtainning the PCA components
    pca = PCA(n_components=30)  # number of axis
    X_pca = pca.fit_transform(X_centered)
    # X_pca = X_pca / np.max(np.abs(X_pca), axis=0)
    #storing the PCA1
    PCA1=X_pca[:,0]
    PCA2=X_pca[:,1]
    PCA3=X_pca[:,2]
    
    
    
    return X_pca,PCA1, PCA2, PCA3




    
    
  
def xcorr_PCA(X,Y,washout,steps):
    """
    Computes a similarity score between two sets of internal states X and Y
    using PCA maximum cross-correlation.

    Args:
    
    - X : (array) Reference matrix of internal states (T timesteps, N_X dimensions)
    - Y : (array) Matrix of internal states to compare (T timesteps, N_Y dimensions)
    - washout : (int) Steps fo the washout
    - steps : (int) steps that we are going to use in the xcorr

    Returns:
    - xcorr : (float) Mean of the max xcorr between the 3 first PCAs
    """
    
    #taking the steps that we want to use
    X=X[washout:washout+steps]
    Y=Y[washout:washout+steps]
    
    # centering the X
    X_centered = X - np.mean(X, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)
    
    #Obtainning the PCA components
    pca_X = PCA(n_components=3).fit(X_centered)
    pca_Y = PCA(n_components=3).fit(Y_centered)
    X_pca = pca_X.transform(X_centered)
    Y_pca = pca_Y.transform(Y_centered)
    
    #storing the X PCAi
    XPCA1=X_pca[:,0]
    XPCA2=X_pca[:,1]
    XPCA3=X_pca[:,2]
    #storing the X PCAi
    YPCA1=Y_pca[:,0]
    YPCA2=Y_pca[:,1]
    YPCA3=Y_pca[:,2]
    
    #computing the xcorrelation between the 3 PCAs
    y=YPCA1
    x = safe_normalize(XPCA1)
    y = safe_normalize(y)
    corr_free1 = (correlate(x, y, mode='full').max())/len(x)

    y=YPCA2
    x = safe_normalize(XPCA2)
    y = safe_normalize(y)
    corr_free2 = (correlate(x, y, mode='full').max())/len(x)

    y=YPCA3
    x = safe_normalize(XPCA3)
    y = safe_normalize(y)
    corr_free3 = (correlate(x, y, mode='full').max())/len(x)
    
    #computing the mean correlation between the PCAs
    xcorr=(corr_free1+corr_free2+corr_free3)/3
    
    return xcorr
    


def xcorr(X,Y,washout,steps,f_min,f_max,dt=None):
    """
    Computes a similarity score between two sets of arrays X and Y
    .

    Args:
    
    - X : (array) Reference matrix  
    - Y : (array) Matrix to compare
    - washout : (int) Steps fo the washout
    - steps : (int) steps that we are going to use in the xcorr
    - dt : (float) dt for the FFT
    - f_min, f_max : (float) range of frequencies that we are going to use
    
    Returns:
    - xcorr : (float)  normalized max xcorr
    """
    
    #taking the steps that we want to use
    X=X[washout:washout+steps]
    Y=Y[washout:washout+steps]
    
    #for the frequancy spectra
    if dt is not None:
        fx, X = power_spectrum(X, dt)
        fy, Y = power_spectrum(Y, dt)
    
        epsilon = 1e-12
        X = np.log10(X + epsilon)
        Y = np.log10(Y + epsilon)
    
        # in the case that that vaule is not in f, take the closer one
        i_min = np.argmin(np.abs(fx - f_min))
        i_max = np.argmin(np.abs(fx - f_max))

        if i_min > i_max:
            i_min, i_max = i_max, i_min
        #only using this range of frequency
        X = X[i_min:i_max+1]
        Y = Y[i_min:i_max+1]
    
    #computing the xcorrelation
    x = safe_normalize(X).ravel()
    y = safe_normalize(Y).ravel()
    
    #cross correlation
    corr = correlate(x, y, mode='full')
    #  lag = 0
    zero_lag = len(x) - 1
    #cross correlation with lag=0
    xcorr = corr[zero_lag] / len(x)

    return xcorr


def xcorr_new(X,Y,washout,steps,dt=None):
    """
    Computes a similarity score between two sets of arrays X and Y
    .

    Args:
    
    - X : (array) Reference matrix  
    - Y : (array) Matrix to compare
    - washout : (int) Steps fo the washout
    - steps : (int) steps that we are going to use in the xcorr
    - dt : (float) dt for the FFT
    
    Returns:
    - xcorr : (float)  normalized max xcorr
    """
    
    #taking the steps that we want to use
    X=X[washout:washout+steps]
    Y=Y[washout:washout+steps]
    
    #for the frequancy spectra
    if dt is not None:
       _,X=power_spectrum(X,dt)
       _,Y=power_spectrum(Y,dt) 
          
          
    #computing the xcorrelation
    x = safe_normalize(X).ravel()
    y = safe_normalize(Y).ravel()
    xcorr = (correlate(x, y, mode='full').max())/len(x)

   
    return xcorr


def safe_normalize(v):
    std = np.std(v)
    if std == 0 or np.isnan(std):
        return np.zeros_like(v)   # o v - np.mean(v)
    return (v - np.mean(v)) / std




def power_spectrum(y, dt):
    """
    Computes the FFT and returns frequency in cycles per discrete step.
    """
    N = len(y)
    Y = np.fft.rfft(y - np.mean(y))
    power = np.abs(Y)**2 / N
    freqs = np.fft.rfftfreq(N, dt)   # convert to cycles per discrete step
    return freqs, power




def smooth_spectrum(power, window=5):
    """
    Smooths a power spectrum using a moving average of given window size.
    """
    kernel = np.ones(window) / window
    return np.convolve(power, kernel, mode="same")


    
# def prediction_horizon(y_true, y_pred, window ,steps=None, washout=0, threshold=0.1, consecutive=3,lyap=None):
#     """
#     Compute NRMSE for a window that moves one step at a time and stop when the error exceeds 
#     threshold for a number of consecutive steps.

#     Args:
#     - y_true (array): reference series
#     - y_pred (array): predicted series
#     - window (int): length of the window where error is computed
#     - steps (int): max number of steps to evaluate, if None, uses full series
#     - washout (int): initial steps to ignore
#     - threshold (float): NRMSE threshold to trigger break
#     - consecutive (int): number of consecutive steps above threshold to stop
#     - lyap (float):lyaponov exponent 

#     Returns:
#     - nrmse_series (list): NRMSE at each evaluated step
#     - horizon (int): number of steps before threshold exceeded consecutively
#     """
#     washout = int(washout)
    
#     y_true_eff = y_true[washout:]
#     y_pred_eff = y_pred[washout:]
    
#     if steps is None:
#         steps = len(y_true_eff)
#     else:
#         steps = min(steps, len(y_true_eff))
    
#     nrmse_series = []
#     above_count = 0
#     horizon = steps
    
#     for t in range(0, steps - window + 1): # we dont want teh avaluation to excead the len of y
#     #avoiding the last windoe possible to be smaller than the other ones
#         nrmse = NRMSE(y_true_eff[t:window+t], y_pred_eff[t:window+t])
#         nrmse_series.append(nrmse)
        
#         if nrmse > threshold:
#             above_count += 1
#             if above_count >= consecutive:
#                 horizon = t - consecutive + 1  # step before consecutive violations
#                 break
#         else:
#             above_count = 0  # reset counter if below threshold
#     if lyap is not None:
#         #horizont prediction with lyaponov units
#         # LT=1/lyap
#         horizon_normalized=horizon*lyap #the samas duinh horizon/LT (it says how many lyapunov time is the horizon)
        
#     else:    
#         #normalizing the horizon
#         # horizon_normalized=horizon/(steps-washout)
#         horizon_normalized=horizon
#     return nrmse_series, horizon_normalized 
    
def prediction_horizon(y_true, y_pred, window ,steps=None, washout=0, threshold=0.1, consecutive=3,lyap=None):
    """
    Compute NRMSE for a window that moves one step at a time and stop when the error exceeds 
    threshold for a number of consecutive steps.

    Args:
    - y_true (array): reference series
    - y_pred (array): predicted series
    - window (int): length of the window where error is computed
    - steps (int): max number of steps to evaluate, if None, uses full series
    - washout (int): initial steps to ignore
    - threshold (float): NRMSE threshold to trigger break
    - consecutive (int): number of consecutive steps above threshold to stop
    - lyap (float):lyaponov exponent 

    Returns:
    - nrmse_series (list): NRMSE at each evaluated step
    - horizon (int): number of steps before threshold exceeded consecutively
    """
    washout = int(washout)
    
    y_true_eff = y_true[washout:]
    y_pred_eff = y_pred[washout:]
    
    if steps is None:
        steps = len(y_true_eff)
    else:
        steps = min(steps, len(y_true_eff))
    
    nrmse_series = []
    above_count = 0
    horizon = steps
    
    for t in range(0, steps - window + 1): # we dont want teh avaluation to excead the len of y
    #avoiding the last windoe possible to be smaller than the other ones
        nrmse = NRMSE(y_true_eff[t:window+t], y_pred_eff[t:window+t])
        nrmse_series.append(nrmse)
        
        if nrmse > threshold:
            above_count += 1
            if above_count >= consecutive:
                horizon = t - consecutive + 1  # step before consecutive violations
                break
        else:
            above_count = 0  # reset counter if below threshold
    
    
    return nrmse_series,horizon




#tau for the embedding, using 1/e in the autocorrelation    
def tau_autocorr(signal):

    signal = (signal - np.mean(signal)) / np.std(signal)

    corr = correlate(signal, signal, mode='full')
    corr = corr[len(corr)//2:]      # positive lags 
    corr /= corr[0]                 # normalization

    #  first index that the correlation is lower than 1/e
    tau = np.where(corr < 1/np.e)[0][0]

    return tau, corr        
    
# Function to create 3D embedding
def embedding3D(signal, tau):
    y0 = signal[2*tau:]    # x(t)
    y1 = signal[tau:-tau]  # x(t - tau)
    y2 = signal[:-2*tau]   # x(t - 2*tau)
    return y0, y1, y2
    
def xcorr_emb(X, Y, washout, steps):
    """
    Computes a similarity score between two sets of internal states X and Y
    using PCA maximum cross-correlation (with centering).
    """
    # Slice for steps
    X = X[washout:washout+steps]
    Y = Y[washout:washout+steps]
    
    tau, _ = tau_autocorr(X)
    X0, X1, X2 = embedding3D(X, tau)
    Y0, Y1, Y2 = embedding3D(Y, tau)
    
    corr_vals = []
    
    for Xc, Yc in zip([X0, X1, X2], [Y0, Y1, Y2]):
        # Center
        x = Yc - np.mean(Yc)
        y = Xc - np.mean(Xc)
        # Normalize
        x = safe_normalize(x)
        y = safe_normalize(y)
        # Max cross-correlation
        corr_vals.append(correlate(x, y, mode='full').max() / len(x))
    
    # Mean of the three PCA components
    xcorr = np.mean(corr_vals)
    
    return xcorr





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

    
    for s,e in combinations(product(r,r,r),2):
        if sum(abs(s[i]-e[i]) for i in range(3)) == r[1]-r[0]:
            ax.plot3D([s[0],e[0]], [s[1],e[1]], [s[2],e[2]], color=edge_color, linewidth=edge_lw, zorder=10)




# ------------------------------
# Function to draw transparent box based on data limits
# ------------------------------
def draw_data_box(ax, x_vals, y_vals, z_vals,
                  face_color='white',
                  edge_color='black',
                  face_alpha=0.08,
                  edge_lw=2.0):

    x_min, x_max = np.min(x_vals), np.max(x_vals)
    y_min, y_max = np.min(y_vals), np.max(y_vals)
    z_min, z_max = np.min(z_vals), np.max(z_vals)

    w = 0.5
    r = [(x_min-w, x_max+w),
         (y_min-w, y_max+w),
         (z_min-w, z_max+w)]

    faces = [
        [(r[0][0],r[1][0],r[2][0]),(r[0][1],r[1][0],r[2][0]),(r[0][1],r[1][1],r[2][0]),(r[0][0],r[1][1],r[2][0])],
        [(r[0][0],r[1][0],r[2][1]),(r[0][1],r[1][0],r[2][1]),(r[0][1],r[1][1],r[2][1]),(r[0][0],r[1][1],r[2][1])],
        [(r[0][0],r[1][0],r[2][0]),(r[0][1],r[1][0],r[2][0]),(r[0][1],r[1][0],r[2][1]),(r[0][0],r[1][0],r[2][1])],
        [(r[0][0],r[1][1],r[2][0]),(r[0][1],r[1][1],r[2][0]),(r[0][1],r[1][1],r[2][1]),(r[0][0],r[1][1],r[2][1])],
        [(r[0][0],r[1][0],r[2][0]),(r[0][0],r[1][1],r[2][0]),(r[0][0],r[1][1],r[2][1]),(r[0][0],r[1][0],r[2][1])],
        [(r[0][1],r[1][0],r[2][0]),(r[0][1],r[1][1],r[2][0]),(r[0][1],r[1][1],r[2][1]),(r[0][1],r[1][0],r[2][1])]
    ]

   
    poly = Poly3DCollection(
        faces,
        facecolors=face_color,
        edgecolors='none',
        alpha=face_alpha
    )

    poly.set_zsort('min') 
    ax.add_collection3d(poly)

    # --- ARISTAS ---
    for s, e in combinations(product(*[(r[i][0], r[i][1]) for i in range(3)]), 2):
        if sum(abs(s[i]-e[i]) for i in range(3)) in [
            r[0][1]-r[0][0],
            r[1][1]-r[1][0],
            r[2][1]-r[2][0]
        ]:
            ax.plot3D(
                [s[0], e[0]],
                [s[1], e[1]],
                [s[2], e[2]],
                color=edge_color,
                linewidth=edge_lw
            )