import numpy as np
import warnings
from numpy.lib.stride_tricks import sliding_window_view
import time
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import gridspec


def get_shapes(W, H):
    
    N = W.shape[0]
    K = W.shape[1]
    L = W.shape[2]
    
    if H.ndim == 2:
        S = 1
        T = H.shape[1]
    else:
        S = H.shape[0]
        T = H.shape[2]

    return S, N, K, L, T


def pad_axis(X, pad_before, pad_after, axis, **kwargs):
    """
    Pad only one axis of an array, regardless of dimensionality.

    Parameters
    ----------
    X : ndarray
    pad_before : Padding before along the axis
    pad_after : Padding after along the axis
    axis : Axis to pad
    kwargs : Passed to np.pad (e.g. constant_values=0)

    Returns
    -------
    padded : ndarray
    """
    pad_width = [(0, 0)] * X.ndim
    pad_width[axis] = (pad_before, pad_after)
    return np.pad(X, pad_width, **kwargs)


def build_lag_mat(X, L, mode='right', axis=-1):
    """
    Builds a time lag matrix

    H: (S, K, T) Loadings
    Returns:
        H_lag: (S, K, T, L) where H_lag[s, k, t, l] = H[s, k, t-l]
    """
    if mode == 'right':
        # pad X only on the left to move the lag matrix to the right, or backward in time
        X_pad = pad_axis(X, L-1, 0, axis)
    elif mode == 'left':
        X_pad = pad_axis(X, 0, L-1, axis)
    elif mode == 'center':
        c = L // 2
        X_pad = pad_axis(X, c, L-1-c, axis)

    # build a time delayed matrix of H
    X_lag = sliding_window_view(X_pad, L, axis=axis)

    # if shifting backward in time, reverse lag axis so l=0 is current time 
    if mode == 'right':
        X_lag = X_lag[:, :, :, ::-1]
    
    return X_lag


def convolve(X, kernel, axis=-1):
    """
    Perform efficient same convolution using delay matrices

    Parameters
    ----------
    X : Data matrix
    kernel : convolution kernel

    Returns
    -------
    None.

    """
    X_lag = build_lag_mat(X, len(kernel), mode='center', axis=axis)
    X_lag = np.moveaxis(X_lag, axis, -1)
    return np.einsum('...l,l->...', X_lag, kernel)
    

def reconstruct(W, H):
    """
    Reconstruct data matrix, X, by convolving the factor kernels in W with the factor loadings in H.
    
    W: (S, N, K, L) Factors
    H: (S, K, T) Loadings
    
    Returns:
        X_hat: (S, N, T)
    """
    
    S, N, K, L, T = get_shapes(W, H)
    
    one_sess = H.ndim == 2
    if one_sess:
        H = H[None, ...]

    H_lag = build_lag_mat(H, L, mode='right')

    # compute the reconstruction. W is [n, k, l], H_lag is [k, t, l].
    # sum the reconstructed signal over all factors and lags
    # this is equivalent to summing W[:,k,:] @ H_lag[k,:,:].T over all k factors
    X_hat = np.einsum('nkl,sktl->snt', W, H_lag)
    
    if one_sess:
        X_hat = X_hat[0,...]

    return X_hat


def reconstruct_old(W, H):

    _, N, K, L, T = get_shapes(W, H)

    H = np.hstack((np.zeros([K, L]), H, np.zeros([K, L])))
    T += 2 * L
    X_hat = np.zeros([N, T])

    for t in np.arange(L):
        X_hat += np.dot(W[:, :, t], np.roll(H, t, axis=1))

    return X_hat[:, L:-L]


def reconstruct_factors(W, H):
    """
    Reconstruct the signals for each factor by convolving the factor kernels in W with the factor loadings in H.
    
    W: (N, K, L) Factors
    H: (S, K, T) Loadings
    
    Returns:
        X_recon: (S, N, T, K) Reconstructed signals per factor

    """
    
    S, N, K, L, T = get_shapes(W, H)
    
    one_sess = H.ndim == 2
    if one_sess:
        H = H[None, ...]
    
    H_lag = build_lag_mat(H, L, mode='right')
    
    X_recon = np.einsum('nkl,sktl->sntk', W, H_lag)
    
    if one_sess:
        X_recon = X_recon[0, ...]
        
    return X_recon


def shift_factors(W, H, eps=1e-6):
    """
    Center factor kernels in time by aligning their center of mass to the middle of the kernel window.

    W: (N, K, L) Factors
    H: (K, T) Loadings

    Returns:
        W_shifted: (N, K, L) Shifted Factors
        H_shifted: (K, T) Shifted Loadings
    """
    warnings.simplefilter('ignore') #ignore warnings for nan-related errors

    S, N, K, L, T = get_shapes(W, H)
    
    if L <= 1:
        return W, H
    
    one_sess = H.ndim == 2
    if one_sess:
        H = H[None, ...]

    center = int(max(np.floor(L / 2), 1))

    # pad W with a small value so it can still be updated with the multiplicative update rule
    W_pad = pad_axis(W, L, L, axis=-1, constant_values=eps)
    # H is already padded
    H_shifted = H.copy()

    mass = np.sum(W, axis=0)
    denom = np.sum(mass, axis=1)
    cmass = np.where(denom > 0, 
                     np.floor(np.sum(mass * np.arange(1, L+1)[None,:], axis=1) / denom).astype(int),
                     center)
    
    for k in range(K):
        s = center - cmass[k]
        W_pad[:, k, :] = np.roll(W_pad[:, k, :], s, axis=1)
        H_shifted[:, k, :] = np.roll(H[:, k, :], -s, axis=1)
        
    if one_sess:
        H_shifted = H_shifted[0, ...]

    return W_pad[:, :, L:-L], H_shifted

def compute_percent_power(X, X_hat):
    
    axis = None if X.ndim == 2 else (1,2)
    tot_power = np.sum(X**2, axis=axis)
    return (tot_power - np.sum((X - X_hat)**2, axis=axis)) / tot_power

def compute_loadings_percent_power(X, W, H):
    """
    Compute the percent of the total power in X contributed by the reconstruction of each factor
    
    X: (N, T) Data matrix
    W: (N, K, L) Factors
    H: (K, T) Loadings
    
    Returns:
        loadings: (K,) Percent of total power contributed by the reconstruction of each factor

    """
    S, N, K, L, T = get_shapes(W, H)
    
    one_sess = X.ndim == 2
    if one_sess:
        X = X[None,...]
        H = H[None,...]

    loadings = np.zeros((S,K))
    tot_power = np.sum(X**2, axis=(1,2))
    X_recon = reconstruct_factors(W, H)

    for k in np.arange(K):
        X_hat_k = X_recon[...,k]
        loadings[:,k] = (tot_power - np.sum((X - X_hat_k)**2, axis=(1,2))) / tot_power

    loadings[loadings < 0] = 0
    
    if one_sess:
        loadings = loadings[0,:]
    
    return loadings


def compute_rmse(X, X_hat):
    return np.sqrt(np.mean((X - X_hat)**2))


def compute_recon_error(X, X_hat):
    return 0.5*np.sum((X - X_hat)**2)


def compute_penalty(X, X_hat, W, H, Lambda, lambda_L1W=0, lambda_L1H=0, lambda_OrthH=0, lambda_OrthW=0, smooth_kernel=None, offdiag=None):
    
    S, N, K, L, T = get_shapes(W, H)
    
    if smooth_kernel is None:
        smooth_kernel = np.ones((2*L)-1)
        
    if offdiag is None:
        offdiag = 1 - np.eye(K)
    
    penalty = 0
    if lambda_L1H > 0:
        penalty += lambda_L1H*np.sum(H)
        
    if lambda_L1W > 0:
        penalty += lambda_L1W*np.sum(W)

    if lambda_OrthH > 0:
        penalty += lambda_OrthH/2 * np.sum(np.einsum('skt,smt->skm', convolve(H, smooth_kernel), H) * offdiag[None,:,:])
        
    if lambda_OrthW > 0:
        W_flat = np.sum(W, axis=2)
        penalty += lambda_OrthW/2 * np.sum((W_flat.T @ W_flat) * offdiag)
        
    if Lambda > 0:
        X_lag = build_lag_mat(X, L, mode='left')
        WTX = np.einsum('nkl,sntl->skt', W, X_lag)
        penalty += Lambda * np.sum(np.einsum('skt,smt->skm', WTX, convolve(H, smooth_kernel)) * offdiag[None,:,:])
        
    return penalty
    

def plot(W, H, cmap='gray_r', factor_cmap='Spectral', exclude_empty=True):
    '''
    :param W: N (features) by K (factors) by L (per-factor timepoints) tensor of factors
    :param H: K (factors) by T (timepoints) matrix of factor loadings (i.e. factor timecourses)
    :param cmap: colormap used to draw heatmaps for the factors, factor loadings, and data reconstruction
    :param factor_cmap: colormap used to distinguish individual factors
    :return f: matplotlib figure handle
    '''

    S, N, K, L, T = get_shapes(W, H)
    
    data_recon = reconstruct(W, H)
    
    one_sess = H.ndim == 2
    if one_sess:
        H = H[None, ...]
        data_recon = data_recon[None, ...]
    
    # Remove empty factors
    if exclude_empty:
        W_sum = np.sum(W, axis=(0,2))
        non_zero = W_sum != 0
        K = np.sum(non_zero)
        W = W[:,non_zero,:]
        H = H[:,non_zero,:]
    
    figs = []
    
    for s in range(S):

        fig, axs = plt.subplots(2, 2, width_ratios=[1, 4], height_ratios=[1, 4], figsize=(10,8))
        axs[0,0].axis('off')
        ax_h = axs[0,1]
        ax_w = axs[1,0]
        ax_data = axs[1,1]
    
        # plot W, H, and data_recon
        sns.heatmap(np.hstack(list(map(np.squeeze, np.split(W, K, axis=1)))), cmap=cmap, ax=ax_w, cbar=False)
        sns.heatmap(H[s,...], cmap=cmap, ax=ax_h, cbar=False)
        sns.heatmap(data_recon[s,...], cmap=cmap, ax=ax_data, cbar=False)
    
        # add dividing bars for factors of W and H
        factor_colors = sns.color_palette(factor_cmap, K)
        for k in range(K):
            start_w = k * L
            ax_w.plot([start_w, start_w], [0, N - 1], '-', color=factor_colors[k])

            ax_h.plot([0, T - 1], [k, k], '-', color=factor_colors[k])
            
        figs.append(fig)
        
    if one_sess:
        figs = figs[0]

    return figs

