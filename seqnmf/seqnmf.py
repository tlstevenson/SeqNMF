import warnings
warnings.simplefilter('ignore') #ignore numpy incompatability warning (harmless)

import numpy as np
from scipy.signal import convolve2d as conv2
import helpers
import time

def loss_slope(loss_history):

    x = np.arange(len(loss_history))

    slope = np.polyfit(x, loss_history, 1)[0]

    return slope / max(loss_history[0], 1e-12)


def seqnmf(X, K=10, L=100, Lambda=0.001, W_init=None, H_init=None,
           max_iter=1000, tol=1e-5, eval_window=6, shift=True, sort_factors=True,
           lambda_L1W=0, lambda_L1H=0, lambda_OrthH=0, lambda_OrthW=0, M=None,
           use_W_update=True, W_fixed=False, calc_penalty=True, print_interval=20):
    '''
    :param X: an N (features) by T (timepoints) or S (sessions) by N by T data matrix to be factorized using seqNMF
    :param K: the (maximum) number of factors to search for; any unused factors will be set to all zeros
    :param L: the (maximum) number of timepoints to consider in each factor; any unused timepoints will be set to zeros
    :param Lambda: regularization parameter (default: 0.001)
    :param W_init: initial factors (if unspecified, use random initialization)
    :param H_init: initial per-timepoint factor loadings (if unspecified, initialize randomly)
    :param max_iter: maximum number of iterations/updates
    :param tol: stopping criteria. Updates will stop when the slope over the eval_window is negative and smaller than the tolerance
    :param eval_window: the number of trials to evaluate the slope over to determine if the model fitting has plateaued
    :param shift: whether to shift the factors in W into the middle of the window
    :param sort_factors: sort factors by explanatory power
    :param lambda_L1W: regularization parameter for W (default: 0)
    :param lambda_L1H: regularization parameter for H (default: 0)
    :param lambda_OrthH: regularization parameter for H (default: 0)
    :param lambda_OrthW: regularization parameter for W (default: 0)
    :param M: binary mask of the same size as X, used to ignore a subset of the data during training (default: use all data)
    :param use_W_update: set to True for more accurate results; set to False for faster results (default: True)
    :param W_fixed: if true, fix factors (W), e.g. for cross validation (default: False)
    :param calc_penalty: if true, calculate the full regularization penalty to compute the total loss (default: True)
    :param print_interval: the iteration interval over which to print out the training progress

    :return:
    :W: N (features) by K (factors) by L (per-factor timepoints) tensor of factors
    :H: K (factors) by T (timepoints) or S (sessions) by K by T tensor of factor loadings (i.e. factor timecourses)
    :power: the total power (across all factors) explained by the full reconstruction
    :loadings: the explanatory power of each individual factor
    :costs: a dictionary containing vectors containing the RMSE, reconstruction loss, penalty loss, and total loss for each iteration
    '''
    
    one_sess = X.ndim == 2
    if one_sess:
        X = X[None, ...]   # add session axis

    S, N, _ = X.shape
    # pad original data
    X = helpers.pad_axis(X, L, L, -1)
    T = X.shape[-1]

    if W_init is None:
        W_init = np.max(X) * np.random.rand(N, K, L)
    if H_init is None:
        H_init = np.max(X) * np.random.rand(S, K, T) / np.sqrt(T / 3)
    
    has_mask = not M is None
    if not has_mask:
        M = np.ones([S, N, T])
    elif one_sess:
        M = M[None, ...]

    assert np.all(X >= 0), 'All data values must be positive'

    W = W_init.copy()
    H = H_init.copy()

    X_hat = helpers.reconstruct(W, H)
    
    if has_mask:
        mask = M == 0
        X[mask] = X_hat[mask]
    else:
        # if no masking, the lagged X matrix can be built once and reused
        X_lag = helpers.build_lag_mat(X, L, mode='left')

    eps = np.max(X) * 1e-6
    last_time = False

    rmse = np.zeros(max_iter)
    error = np.zeros(max_iter)
    penalty = np.zeros(max_iter)
    loss = np.zeros(max_iter)
    
    start_t = time.perf_counter()
    loop_start_t = start_t
    
    # make matrices used in updates. 
    offdiag = 1 - np.eye(K)
    smooth_kernel = np.ones((2*L)-1)

    for i in np.arange(max_iter):

        if (i == max_iter - 1) or ((i >= eval_window) and (-tol < loss_slope(loss[i-eval_window:i]) <= 0)):
            last_time = True
            if i > 0:
                Lambda = 0

        # perform H updates
        
        # Lagged data matrices. Shape (N, T, L)
        if has_mask:
            # Make X_lag every loop if masking
            X_lag = helpers.build_lag_mat(X, L, mode='left')
        
        Xhat_lag = helpers.build_lag_mat(X_hat, L, mode='left')
    
        # compute WTX matrices that describe the correlation between the factors in W/W_hat and X
        WTX = np.einsum('nkl,sntl->skt', W, X_lag)
        WTX_hat = np.einsum('nkl,sntl->skt', W, Xhat_lag)
        
        # compute penalty terms and perform update to H
        dRdH = lambda_L1H
        
        # x-ortho penalty
        if Lambda > 0:
            dRdH += Lambda * np.einsum('skt,km->smt', helpers.convolve(WTX, smooth_kernel), offdiag)

        # smoothed orthogonality in H
        if lambda_OrthH > 0:
            dRdH += lambda_OrthH * np.einsum('skt,km->smt', helpers.convolve(H, smooth_kernel), offdiag)
            
        H *= WTX / (WTX_hat + dRdH + eps)

        if not W_fixed:
            
            # shift the factor weights into the center of the window
            if shift:
                W, H = helpers.shift_factors(W, H, eps=eps)
                
            # normalize H and W
            norms = np.sqrt(np.sum(H**2, axis=(0,2)))

            H /= norms[None, :, None] + eps
            W *= norms[None, :, None]
            
            # perform W updates
            
            X_hat = helpers.reconstruct(W, H)
            if has_mask:
                # replace data at masked elements with reconstruction, so masked datapoints do not effect fit
                X[mask] = X_hat[mask]

            H_lag = helpers.build_lag_mat(H, L, mode='right')
            # compute XHT matrices that describe the correlation between the factor loadings in H and X/X_Hat at each lag
            XHT = np.einsum('snt,sktl->nkl', X, H_lag)
            XhatHT = np.einsum('snt,sktl->nkl', X_hat, H_lag)
            
            # compute penalty terms and perform update to W
            dRdW = lambda_L1W
            
            # x-ortho penalty
            if Lambda > 0 and use_W_update:
                XS = helpers.convolve(X, smooth_kernel)
                XSH = np.einsum('snt,sktl->nkl', XS, H_lag)
                dRdW += Lambda * np.einsum('nkl,km->nml', XSH, offdiag)
            
            # smoothed orthogonality in W
            if lambda_OrthW > 0:
                W_flat = np.sum(W, axis=2)
                dWWdW = lambda_OrthW * (W_flat @ offdiag)
                dRdW += dWWdW[:, :, None]

            W *= XHT / (XhatHT + dRdW + eps)

        X_hat = helpers.reconstruct(W, H)
        if has_mask:
            X[mask] = X_hat[mask]
        rmse[i] = helpers.compute_rmse(X, X_hat)
        error[i] = helpers.compute_recon_error(X, X_hat)
        if calc_penalty:
            penalty[i] = helpers.compute_penalty(X, X_hat, W, H, Lambda, lambda_L1W=lambda_L1W, lambda_L1H=lambda_L1H, 
                                                 lambda_OrthH=lambda_OrthH, lambda_OrthW=lambda_OrthW, smooth_kernel=smooth_kernel, offdiag=offdiag)
            
        loss[i] = error[i] + penalty[i]
        
        if i % print_interval == print_interval - 1 and not last_time:
            stop_t = time.perf_counter()
            print_error = np.mean(error[i-print_interval+1:i+1])
            print_penalty = np.mean(penalty[i-print_interval+1:i+1])
            print_loss = np.mean(loss[i-print_interval+1:i+1])
            if calc_penalty:
                print('Step {}, error {:.1f}, penalty {:.1f}, loss {:.1f}, elapsed time: {:.1f}s, time per step: {:.3f}s'.format(
                      i+1, print_error, print_penalty, print_loss, stop_t-start_t, (stop_t-loop_start_t)/print_interval))
            else:
                print('Step {}, error {:.1f}, elapsed time: {:.1f}s, time per step: {:.3f}s'.format(i+1, print_error, stop_t-start_t, (stop_t-loop_start_t)/print_interval))
            
            loop_start_t = time.perf_counter()      

        if last_time:
            rmse = rmse[:i+1]
            error = error[:i+1]
            penalty = penalty[:i+1]
            loss = loss[:i+1]
            break

    X = X[..., L:-L]
    X_hat = X_hat[..., L:-L]
    H = H[..., L:-L]
    
    power = helpers.compute_percent_power(X, X_hat)
    loadings = helpers.compute_loadings_percent_power(X, W, H)

    if sort_factors:
        inds = np.flip(np.argsort(np.mean(loadings, axis=0)), 0)
        loadings = loadings[:,inds]

        W = W[:, inds, :]
        H = H[:, inds, :]
        
    if one_sess:
        H = H[0, ...]
        power = power[0]
        loadings = loadings[0,:]

    return W, H, power, loadings, {'rmse': rmse, 'error': error, 'penalty': penalty, 'loss': loss}


def seqnmf_old(X, K=10, L=100, Lambda=0.001, W_init=None, H_init=None,
           plot_it=False, max_iter=500, tol=1e-4, shift=True, sort_factors=True,
           lambda_L1W=0, lambda_L1H=0, lambda_OrthH=0, lambda_OrthW=0, M=None,
           use_W_update=True, W_fixed=False, print_interval=20, eval_window=6):
    '''
    :param X: an N (features) by T (timepoints) data matrix to be factorized using seqNMF
    :param K: the (maximum) number of factors to search for; any unused factors will be set to all zeros
    :param L: the (maximum) number of timepoints to consider in each factor; any unused timepoints will be set to zeros
    :param Lambda: regularization parameter (default: 0.001)
    :param W_init: initial factors (if unspecified, use random initialization)
    :param H_init: initial per-timepoint factor loadings (if unspecified, initialize randomly)
    :param plot_it: if True, display progress in each update using a plot (default: False)
    :param max_iter: maximum number of iterations/updates
    :param tol: if cost is within tol of the average of the previous 5 updates, the algorithm will terminate (default: tol = -inf)
    :param shift: allow timepoint shifts in H
    :param sort_factors: sort factors by time
    :param lambda_L1W: regularization parameter for W (default: 0)
    :param lambda_L1H: regularization parameter for H (default: 0)
    :param lambda_OrthH: regularization parameter for H (default: 0)
    :param lambda_OrthW: regularization parameter for W (default: 0)
    :param M: binary mask of the same size as X, used to ignore a subset of the data during training (default: use all data)
    :param use_W_update: set to True for more accurate results; set to False for faster results (default: True)
    :param W_fixed: if true, fix factors (W), e.g. for cross validation (default: False)

    :return:
    :W: N (features) by K (factors) by L (per-factor timepoints) tensor of factors
    :H: K (factors) by T (timepoints) matrix of factor loadings (i.e. factor timecourses)
    :cost: a vector of length (number-of-iterations + 1) containing the initial cost and cost after each update (i.e. the reconstruction error)
    :loadings: the per-factor loadings-- i.e. the explanatory power of each individual factor
    :power: the total power (across all factors) explained by the full reconstruction
    '''
    N = X.shape[0]
    # pad original data
    X = np.concatenate((np.zeros([N, L]), X, np.zeros([N, L])), axis=1)
    T = X.shape[1]

    if W_init is None:
        W_init = np.max(X) * np.random.rand(N, K, L)
    if H_init is None:
        H_init = np.max(X) * np.random.rand(K, T) / np.sqrt(T / 3)
    if M is None:
        M = np.ones([N, T])

    assert np.all(X >= 0), 'all data values must be positive!'

    W = W_init
    H = H_init

    X_hat = helpers.reconstruct(W, H)
    mask = M == 0
    X[mask] = X_hat[mask]

    smooth_kernel = np.ones([1, (2*L)-1])
    eps = np.max(X) * 1e-6
    last_time = False

    cost = np.zeros([max_iter+1, 1])
    cost[0] = np.sqrt(np.mean((X - X_hat)**2))
    
    start_t = time.perf_counter()
    loop_start_t = start_t

    for i in np.arange(max_iter):

        if (i == max_iter - 1) or ((eval_window > 6) and (cost[i + 1] + tol) > np.mean(cost[i - eval_window:i])):
            cost = cost[:(i + 2)]
            last_time = True
            if i > 0:
                Lambda = 0

        WTX = np.zeros([K, T])
        WTX_hat = np.zeros([K, T])
        for j in np.arange(L):
            X_shifted = np.roll(X, -j, axis=1)
            X_hat_shifted = np.roll(X_hat, -j, axis=1)

            WTX += np.dot(W[:, :, j].T, X_shifted)
            WTX_hat += np.dot(W[:, :, j].T, X_hat_shifted)

        if Lambda > 0:
            dRdH = np.dot(Lambda * (1 - np.eye(K)), conv2(WTX, smooth_kernel, 'same'))
        else:
            dRdH = 0

        if lambda_OrthH > 0:
            dHHdH = np.dot(lambda_OrthH * (1 - np.eye(K)), conv2(H, smooth_kernel, 'same'))
        else:
            dHHdH = 0

        dRdH += lambda_L1H + dHHdH

        H *= np.divide(WTX, WTX_hat + dRdH + eps)

        if shift:
            W, H = helpers.shift_factors(W, H)
            W += eps

        norms = np.sqrt(np.sum(np.power(H, 2), axis=1)).T
        H = np.dot(np.diag(np.divide(1., norms + eps)), H)
        for j in np.arange(L):
            W[:, :, j] = np.dot(W[:, :, j], np.diag(norms))

        if not W_fixed:
            X_hat = helpers.reconstruct_old(W, H)
            mask = M == 0
            X[mask] = X_hat[mask]

            if lambda_OrthW > 0:
                W_flat = np.sum(W, axis=2)
            if (Lambda > 0) and use_W_update:
                XS = conv2(X, smooth_kernel, 'same')

            for j in np.arange(L):
                H_shifted = np.roll(H, j, axis=1)
                XHT = np.dot(X, H_shifted.T)
                X_hat_HT = np.dot(X_hat, H_shifted.T)

                if (Lambda > 0) and use_W_update:
                    dRdW = Lambda * np.dot(np.dot(XS, H_shifted.T), (1. - np.eye(K)))
                else:
                    dRdW = 0

                if lambda_OrthW > 0:
                    dWWdW = np.dot(lambda_OrthW * W_flat, 1. - np.eye(K))
                else:
                    dWWdW = 0

                dRdW += lambda_L1W + dWWdW
                W[:, :, j] *= np.divide(XHT, X_hat_HT + dRdW + eps)

        X_hat = helpers.reconstruct_old(W, H)
        mask = M == 0
        X[mask] = X_hat[mask]
        cost[i + 1] = np.sqrt(np.mean(np.power(X - X_hat, 2)))

        if i % print_interval == print_interval - 1:
            print_cost = np.mean(cost[i-print_interval+1:i+1])
            print('Step {}, Cost {:.5f}, elapsed time: {:.1f}s, time per step: {:.3f}s'.format(i+1, print_cost, time.perf_counter()-start_t, (time.perf_counter()-loop_start_t)/print_interval))
            loop_start_t = time.perf_counter()      

        if last_time:
            break

    X = X[:, L:-L]
    X_hat = X_hat[:, L:-L]
    H = H[:, L:-L]

    power = np.divide(np.sum(np.power(X, 2)) - np.sum(np.power(X - X_hat, 2)), np.sum(np.power(X, 2)))

    loadings = helpers.compute_loadings_percent_power(X, W, H)

    if sort_factors:
        inds = np.flip(np.argsort(loadings), 0)
        loadings = loadings[inds]

        W = W[:, inds, :]
        H = H[inds, :]

    return W, H, cost, loadings, power

