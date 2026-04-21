import numpy as np
from scipy.sparse import random 
from sklearn.linear_model import Ridge
import copy


#container that randomly initializes W and Win
def rnn_params(
    rnn_size,
    input_size,
    output_size,
    input_scaling,
    spectral_radius,
    a_dt,
    bias_scaling,
    sparsity,
    seed=1235
    
):
    """
    Initializes the parameters for a simple RNN model.

    Args:
    - rnn_size (int): The number of hidden units in the RNN.
    - input_size (int): The number of input features.
    - output_size (int): The number of output features.
    - input_scaling (float): Scaling factor for the input weights.
    - spectral_radius (float): Desired spectral radius of the recurrent weight matrix.
    - a_dt (float): Time step size.
    - bias_scaling (float, optional): Scaling factor for the bias terms. Defaults to 0.4.
    - seed (int, optional): Seed for the random number generator. Defaults to 1235.
    - Sparsity (int or None): If sparsity is needed you can give a value, if not None
    
    Returns:
    - params (dict): A dictionary containing the initialized parameters.
    """

    prng = np.random.default_rng(seed)
    def normal_data(n):
        # return prng.normal(loc=0,scale=1.0,size=n)
        return prng.normal(size=n)
    #crea la matriz W
    def rnn_ini(shape, spectral_radius,sparsity): #matrix dimension and desired spectral radius
        if sparsity is not None and sparsity!=1:
            w=random(shape[0], shape[1], density=sparsity, data_rvs=normal_data, format='coo',random_state=prng.integers(1e9)) #if we want to give sparsity to the matrix
            w=w.toarray()
        else:
            w = prng.normal(size=shape)
            # w = prng.normal(loc=0,scale=1.0,size=shape) #internal weights, generates a matrix with Gaussian distribution (values range from -1 to 1)
        current_spectral_radius = max(abs(np.linalg.eig(w)[0])) # calcula el radio expectral de la matriz w aleatoria
        w *= spectral_radius / current_spectral_radius # adjusts W by scaling all its values so that its new spectral radius equals the specified spectral_radius
                                                        # this controls the recurrent dynamics (prevents exploding or vanishing activations in the RNN)
        return w
    
    params = dict(
        win=prng.normal(loc=0,scale=1.0,size=(rnn_size, input_size)) * input_scaling, #input weight matrix with range [-a, a]
        w=rnn_ini((rnn_size, rnn_size), spectral_radius,sparsity), #calls the function to create the matrix
        bias=prng.normal(size=(rnn_size,)) * bias_scaling, #the bias vector for hidden layers, with dimension Nx1 where N is the number of internal neurons
        wout=prng.normal(size=(output_size, rnn_size)), #output weight matrix 1xN, 1 output neurone
        bias_out=prng.normal(size=(output_size,)) * bias_scaling, #b fot output layer
        a_dt=a_dt * np.ones(rnn_size), 
        x_ini=0.1 * prng.normal(size=(rnn_size)), # random inital hidden state
    )

    return  params



def forward_rnn(params, ut,seed=42,x_init=None, autonomous=False,conceptor=None,std_Noise=None): 
    """
    Forward pass of a recurrent neural network (RNN) .

    Args:   
    - params (dict): dictionary containing the RNN parameters (weights and biases).
    - ut (ndarray): input to the RNN.
    - seed : for the noise generator
    - x_init (ndarray, optional): initial state of the RNN. Defaults to None.
    - autonomous (boolean): True or False if we want to use this mode or not
    - conceptor (array): The conceptor we want to use or None
    - std_Noise (float): None if we dont want to add noise or float if we want to ad a % of std_noise

    Returns:
    - X (matriz): hidden satate for all the time series
    
    
    use params_trained for every case that you use this function after training the model
    """
    #random number generator for the noise
    prng = np.random.default_rng(seed)
    
    # initial x
    if x_init is None:
        x = params["x_ini"]
    else:
        x = x_init
    x = np.ravel(x)      
    T=len(ut)
    N=params['w'].shape[0]
    # Creating the container for the state matrix
    X = np.zeros((T, N))
    if conceptor is None:
        conceptor = np.eye(x.shape[0]) 
    else:
        conceptor=conceptor
    # temporal loop
   
    for t_idx in range(T):#iterating through the time vector
          
        u_t = (
            ut[t_idx] if not autonomous else np.dot(params["wout"], x) + params["bias_out"]
            )
        #The part inside the tanh (Non Lineality)
        dentro = params["w"] @ x \
            + params["win"] @ u_t \
            + params["bias"]
            
        
        
        # Updating 'leaky tanh', element-wise multiplication
        x = ((1 - params["a_dt"]) * x \
             + params["a_dt"] * np.tanh(dentro))
            
        #noise 2.0    
        if std_Noise is not None: #introducing the noise in all the x
            # r=prng.normal(0,std_Noise)
            r=prng.normal(0,std_Noise,x.shape[0])
            x=x+r
            
        x=conceptor @ x
        x=np.ravel(x)
        # Storing the hidden state
        X[t_idx] = x
        
    return X




def forward_rnn_comb(params, ut,seed=42,steps_ol=30,x_init=None, conceptor=None,std_Noise=None): 
    """
    Forward pass of a recurrent neural network (RNN), autonomous+open loop combination .

    Args:   
    - params (dict): dictionary containing the RNN parameters (weights and biases).
    - ut (ndarray): input to the RNN.
    - seed : for the noise generator
    - steps_ol (int): number the steps for the open loop
    - x_init (ndarray, optional): initial state of the RNN. Defaults to None.
    - conceptor (array): The conceptor we want to use or None
    - std_Noise (float): None if we dont want to add noise or float if we want to ad a % of std_noise

    Returns:
    - X (matriz): hidden satate for all the time series
    
    
    use params_trained for every case that you use this function after training the model
    """
    #random number generator for the noise
    prng = np.random.default_rng(seed)
    
    # initial x
    if x_init is None:
        x = params["x_ini"]
    else:
        x = x_init
    x = np.ravel(x)      
    T=len(ut)
    N=params['w'].shape[0]
    # Creating the container for the state matrix
    X = np.zeros((T, N))
    if conceptor is None: 
        conceptor = np.eye(x.shape[0]) 
    else:
        conceptor=conceptor
    # temporal loop
   
    for t_idx in range(T):#iterating through the time vector
         
        if t_idx< (steps_ol+1):
            u_t=ut[t_idx]
        else:
            u_t=np.dot(params["wout"], x) + params["bias_out"]
        #The part inside the tanh (Non Lineality)
        dentro = params["w"] @ x \
            + params["win"] @ u_t \
            + params["bias"]
            
        
        
        # Updating 'leaky tanh', element-wise multiplication
        x = ((1 - params["a_dt"]) * x \
             + params["a_dt"] * np.tanh(dentro))
            
        #noise 2.0    
        if std_Noise is not None: #introducing the noise in all the x
            # r=prng.normal(0,std_Noise)
            r=prng.normal(0,std_Noise,x.shape[0])
            x=x+r
            
        x=conceptor @ x
        x=np.ravel(x)
        # Storing the hidden state
        X[t_idx] = x
        
    return X




def compute_conceptor(X, aperture,denoise_svd=False):
    """
    Computes the conceptor matrix for a given input matrix X and an aperture value.

    Arg:
    - X (numpy.ndarray): Input matrix of shape (n_samples, n_features). (t,N)
    - aperture (float): Aperture value used to compute the conceptor matrix.
   
    Returns:
    - C (ndarray) Conceptor matrix of shape (n_features, n_features). (N,N)
    """
    R = np.dot(X.T, X) / X.shape[0]
    if denoise_svd is False:
        C = np.dot(R, np.linalg.inv(R + aperture ** (-2) * np.eye(R.shape[0])))
        return C
    if denoise_svd is True:  
        U, S, _ = np.linalg.svd(R, full_matrices=False, hermitian=True)
        C = U * (S / (S + (aperture**(-2)))) @ U.T
        return C



#getting Wout with ridge regression of Scikit-Learn
def ridge(beta,X,Y_target,step,params):
    """
    Trainning the model and visualize the results for the input dataset
    
    Arg:
    - beta:(float) ridge coefficient
    - X (array): hidden state matrix (T,N)
    - Y_target(array): target outpot values (T,1)
    - step: (int) number of steps the model has tu predict
    - params (dict): the ESN params
    
    Returns:
    - params_trained (dict): same ESN params but with the trained Wout and bias_out
    - mse (float): mean squared error between the prediction and the real signal
    
    """
    #copy of params
    params_trained = copy.deepcopy(params)
    
    #Generating the model with beta
    ridge_model = Ridge(alpha=beta, fit_intercept=True) #fit_intercept=True if we want to train bias_out
    
    #Training the model with X and Y_target
    ridge_model.fit(X, Y_target)
    
    #getting Wout (1,N)
    W_out = ridge_model.coef_
    bias_out = ridge_model.intercept_
    
    #Predict Y with this Wout    
    # Y_pred = X @ W_out.T + bias_out
    
    #computing the mse
    # mse = np.mean((Y_pred[:-step] - Y_target[:-step])**2)
    mse=0
    #updating params
    params_trained['wout']=W_out
    params_trained['bias_out']=bias_out
    
    
    return params_trained, mse







#std_noise for a given internal states
def std_noise_func(X,noise_perc):
    
    """
    Computing the noise that we want to introduce into the autonomous mode internal state equation
    
    Args:
    - X (ndarray): internal state in open loop
    - noise_perc (float): the standar desviation of the noise is a % of the std of X
    
    
    Returns:
    - std_new (float): the standard desviation for the noise
    
    """
    if noise_perc is None:
        return None
    else:
        #Computing the standard desviation of X 
        # std_X=X.std(axis=0).mean()
        std_X=X.std()
        #New std for N(0,std_new)
        std_new=(noise_perc/100)*std_X
       
        return std_new
    
    
    
    

 





    
def denoising_CTC(params,ut_train1,std_noise,a_new):
    """
    
    Denoising ("cleaning") the conceptor using the Cross-Trial correlation (Input Forcing). For this we will need to trials

    Arg:
    - params (dict): dictionary containing the RNN parameters (weights and biases).
    - std_Noise (float): None if we dont want to add noise or float if we want to ad a % of std_noise
    - ut_train1 (ndarray): input to the RNN.
    - a_new (float): Aperture for the cleaned conceptor.
    
   
    Returns:
    - C_cct (ndarray): cleaned conceptor.
    
    """
    #Running the ESN twice with diffenrent noises
    X_noi1=forward_rnn(params, ut_train1,42, None,False,None,std_noise)
    X_noi2=forward_rnn(params, ut_train1,1234, None,False,None,std_noise)
    
    #Compute Cross-Correlation
    R_cross = np.dot(X_noi1.T, X_noi2) / X_noi1.shape[0]
    
    #Symmetrize and PSD 
    R_final=0.5*(R_cross+R_cross.T)
    lamdaf, Uf = np.linalg.eigh(R_final)
    # Uf, lamdaf, _ = np.linalg.svd(R_final, full_matrices=False, hermitian=True)
    lamdaf[lamdaf<0]=0
    # lamdaf=np.abs(lamdaf)
    
    #rebuilding R
    R_final=Uf @ np.diag(lamdaf) @ Uf.T
    
    
    #computing C
    a=a_new
    C_ctc= np.dot(R_final, np.linalg.inv(R_final + a ** (-2) * np.eye(R_final.shape[0])))
    
    return C_ctc






def denoising_CTC_m(params,ut_train1,std_noise,a_new,m):
    """
    
    Denoising ("cleaning") the conceptor using the Cross-Trial correlation (Input Forcing). For this we will need to trials

    Arg:
    - params (dict): dictionary containing the RNN parameters (weights and biases).
    - std_Noise (float): None if we dont want to add noise or float if we want to ad a % of std_noise
    - ut_train1 (ndarray): input to the RNN.
    - a_new (float): Aperture for the cleaned conceptor.
    - m (int): number of trials.
    
   
    Returns:
    - C_ctc (ndarray): cleaned conceptor.
    
    """
    
    #setting Xi and Xj
    Xi=np.empty((m),dtype=object)
    
    #parameters for the noise scan
    seed1=20 #seed for the seed generator :)
    np.random.seed(seed1) #if we want this seed to be alwas de same  
    seed=np.random.randint(0, 2000, size=m) #seed for the random degradation
    
    #Running the ESN twice with diffenrent noises
    for i in range(m):
        X=forward_rnn(params, ut_train1,seed[i], None,False,None,std_noise)
        Xi[i]=X
        
    N=Xi[0].shape[1]
    R_cross=np.zeros((N,N),dtype=float)
    
    #computing the cross correlation
    for idx1 in range(m):
        for idx2 in range(m):
            if idx1<idx2:
                R_cross += Xi[idx1].T @ Xi[idx2] / Xi[idx1].shape[0]
            
      
    R=R_cross*(2/(m*(m-1)))
    #Symmetrize and PSD 
    R_final=0.5*(R+R.T)
    lamdaf, Uf = np.linalg.eigh(R_final)
    # Uf, lamdaf, _ = np.linalg.svd(R_final, full_matrices=False, hermitian=True)
    lamdaf[lamdaf<0]=0
    # lamdaf=np.abs(lamdaf)
    
    #rebuilding R
    R_final=Uf @ np.diag(lamdaf) @ Uf.T
    
    
    #computing C
    a=a_new
    C_ctc= np.dot(R_final, np.linalg.inv(R_final + a ** (-2) * np.eye(R_final.shape[0])))
    
    return C_ctc







    

def forward_rnn_drift(params, ut,seed=42,x_init=None, autonomous=False,conceptor=None,std_drift=None,std_Noise=None): #autonomous mode False by default
    """
    Forward pass of a recurrent neural network (RNN) with uniform drift.

    Args:   
    - params (dict): dictionary containing the RNN parameters (weights and biases).
    - ut (ndarray): input to the RNN.
    - seed : for the noise generator
    - x_init (ndarray, optional): initial state of the RNN. Defaults to None.
    - autonomous (boolean): True or False if we want to use this mode or not
    - conceptor (array): The conceptor we want to use or None
    - d (float): 0 if we dont want to add drift or float if we want to ad a % of std_drift

    Returns:
    - X (matriz): hidden satate for all the time series
    
    
    use params_trained for every case that you use this function after training the model
    """
    #random number generator for the noise
    prng = np.random.default_rng(seed)
    
    # initial x
    if x_init is None:
        x = params["x_ini"]
    else:
        x = x_init
    x = np.ravel(x)      
    T=len(ut)
    N=params['w'].shape[0]
    # Creating the container for the state matrix
    X = np.zeros((T, N))
    if conceptor is None: 
        conceptor = np.eye(x.shape[0]) 
    else:
        conceptor=conceptor
    # temporal loop
   
    for t_idx in range(T):#iterating through the time vector
        
        u_t = (
            ut[t_idx] if not autonomous else np.dot(params["wout"], x) + params["bias_out"]
            )
        
        if t_idx<100:
             
            
            #The part inside the tanh (Non Lineality)
            dentro = params["w"] @ x \
                + params["win"] @ u_t \
                + params["bias"] 
                
        
        else:
            #The part inside the tanh (Non Lineality)
            dentro = params["w"] @ x \
                + params["win"] @ u_t \
                + (params["bias"] +std_drift)
               
            
        # Updating 'leaky tanh', element-wise multiplication
        x = ((1 - params["a_dt"]) * x \
             + params["a_dt"] * np.tanh(dentro)) 
            
        #noise    
        if std_Noise is not None: #introducing the noise in all the x
            # r=prng.normal(0,std_Noise)
            r=prng.normal(0,std_Noise,x.shape[0])
            x=x+r
            
        x=conceptor @ x
        x=np.ravel(x)
        # Storing the hidden state
        X[t_idx] = x
        
    return X

def forward_rnn_drift_new(params, ut,seed=42,x_init=None, autonomous=False,conceptor=None,std_drift=None,std_Noise=None): #autonomous mode False by default
    """
    Forward pass of a recurrent neural network (RNN) with random drift.

    Args:   
    - params (dict): dictionary containing the RNN parameters (weights and biases).
    - ut (ndarray): input to the RNN.
    - seed : for the noise generator
    - x_init (ndarray, optional): initial state of the RNN. Defaults to None.
    - autonomous (boolean): True or False if we want to use this mode or not
    - conceptor (array): The conceptor we want to use or None
    - d (float): 0 if we dont want to add drift or float if we want to ad a % of std_drift

    Returns:
    - X (matriz): hidden satate for all the time series
    
    
    use params_trained for every case that you use this function after training the model
    """
    #random number generator for the noise
    prng = np.random.default_rng(seed)
    prng1 = np.random.default_rng(seed+6)
    delta_b=prng1.normal(0,std_drift,params["bias"].shape[0])
    # initial x
    if x_init is None:
        x = params["x_ini"]
    else:
        x = x_init
    x = np.ravel(x)      
    T=len(ut)
    N=params['w'].shape[0]
    # Creating the container for the state matrix
    X = np.zeros((T, N))
    if conceptor is None: 
        conceptor = np.eye(x.shape[0]) 
    else:
        conceptor=conceptor
    # temporal loop
   
    for t_idx in range(T):#iterating through the time vector
        
        u_t = (
            ut[t_idx] if not autonomous else np.dot(params["wout"], x) + params["bias_out"]
            )
        
        if t_idx<100:
             
            
            #The part inside the tanh (Non Lineality)
            dentro = params["w"] @ x \
                + params["win"] @ u_t \
                + params["bias"] 
                
        
        else:
            #The part inside the tanh (Non Lineality)
            dentro = params["w"] @ x \
                + params["win"] @ u_t \
                + (params["bias"] + delta_b)
               
            
        # Updating 'leaky tanh', element-wise multiplication
        x = ((1 - params["a_dt"]) * x \
             + params["a_dt"] * np.tanh(dentro)) 
            
        #noise    
        if std_Noise is not None: #introducing the noise in all the x
            # r=prng.normal(0,std_Noise)
            r=prng.normal(0,std_Noise,x.shape[0])
            x=x+r
            
        x=conceptor @ x
        x=np.ravel(x)
        # Storing the hidden state
        X[t_idx] = x
        
    return X


    
    
    
