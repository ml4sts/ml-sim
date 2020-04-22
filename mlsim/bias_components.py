import numpy as np
import pandas as pd


def demographic_independent(N,rho_a, rho_z):
    '''
    Bias where the labeling errors are correlated with the protected attribute

    Parameters
    -----------
    rho_a : float
        p(a = 1)
    rho_z : float
        p(z=1)
    beta : float
        error rate in y, p(y=z) = 1-beta
    N : int
         number of samples
    mu : matrix like, 2xD
        mu[0] is the mean for z=0, D= len(mu[0]) = number of features
    cov : 2x2
        covariance, shared across classes

    Returns
    --------
    df : DataFrame
        a data frame with N rows and columns: a,y,z, x0:xD

    '''
    p_a = [1-rho_a, rho_a]
    p_z = [1-rho_z, rho_z]

    a = np.random.choice([0,1], p=p_a, size=N)
    z = np.random.choice([0,1], p=p_z, size=N)

    return np.asarray(a).T,np.asarray(z).T


def target_disadvantaged_error(a,z,beta):
    '''
    for label bias. the disadvantaged group is wrong beta percent of the time.

    Parameters
    -----------
    a :
    z :
    beta : float

    '''
    beta = [0 beta]

    y = [np.random.choice([zi,1-zi],p=[1-beta[ai], beta[ai]]) for ai,zi in zip(a,z)]

    return np.asarray(y).T

def target_two_error_rates(a,z,beta):
    '''
    each group has error some amount of the time

    Parameters
    -----------
    a :
    z :
    beta : list-like floats
        error rate for advantaged and disadvantaged groups
    '''


    y = [np.random.choice([zi,1-zi],p=[1-beta[ai], beta[ai]]) for ai,zi in zip(a,z)]

    return np.asarray(y).T

def feature_shared_param(a,z,y,dist,theta):
    '''
    '''
    mu = theta[0]
    cov = theta [1]
    x = [dist(mu[z_i],cov) for z_i in z]

    return np.asarray(x)

def feature_two_params(a,z,y,dist,theta):
    '''
    '''
    mu = theta[0]
    cov = theta [1]
    x = [dist(mu[z_i],cov[z_i]) for z_i in z]

    return np.asarray(x)

def feature_pergroup_shared_param(a,z,y,dist,theta):
    '''
    for feature bias

    Parameters
    ----------
    theta :
    '''
    mu = theta[0]
    cov = theta [1]
    x = [dist(mu[z_i][a_i],cov) for z_i,a_i in zip(z,a)]

    return np.asarray(x)

def feature_pergroup_two_params(a,z,y,dist,theta):
    '''
    for feature bias

    Parameters
    ----------
    theta :
    '''
    mu = theta[0]
    cov = theta [1]
    x = [dist(mu[z_i][a_i],cov[z_i][a_i]) for z_i,a_i in zip(z,a)]

    return np.asarray(x)

def feature_proxy_measurment_quality(a,z,y,dist,theta):
    '''
    the measurement locations vary with the true target value z and the
    measurements spread vary with the meaured target value y, allowing for error
    to be present in both the features and the measurements. Also may vary with
    the protected attribute

    Parameters
    ----------
    theta :
    '''
    loc = theta[0]
    spread = theta [1]
    x = [dist(loc[z_i][a_i],spread[y_i][a_i]) for z_i,y_i,a_i in zip(z,y,a)]

    return np.asarray(x)

def feature_proxy(a,z,y,dist,theta):
    '''
    some features are related to the ground truth and some are realated to the
    proxy,

    Parameters
    ----------
    theta :
    '''
    loc = theta[0]
    spread = theta [1]
    x_signal = [dist(loc[z_i][a_i],spread[z_i][a_i]) for z_i,a_i in zip(z,a)]
    x_signal = np.asarray(x_signal)

    x_proxy = [dist(loc[y_i][a_i],spread[y_i][a_i]) for y_i,a_i in zip(y,a)]
    x_proxy = np.asarray(x_proxy)

    x = np.concatenate([x_signal,x_proxy])
    return x


def feature_noise_replace(a,z,y,x,dist,theta):
    '''
    for subspace bias

    keep the same number of total features, replace some with noise, keep
    d_shared in the middle valid for both groups; replace the first 1/2(ceiled)
    of the remaining with noise for the disadvantaged group and the last portion
    (floored) for the advantaged group
    '''
    d,N = x.shape
    d_shared = theta[0]

    d_pad_a = int(np.floor((d-d_shared)/2)) # noise dims per row
    d_pad_d = int(np.ceil((d-d_shared)/2))
    d_advantaged_end = d_shared + d_pad_d # total dims
    d_noise = max(d_pad_a,d_pad_d)

    # generate the noise
    x_n = dist([0]*d_noise,np.eye(d_noise),N)
    # functions for combining noise and true vectors
    x_a = {0: lambda x,n: np.concatenate((x[:d_advantaged_end],n[:d_pad_a])),
          1: lambda x,n: np.concatenate((n, x[d_pad_d:]))}
    x = [x_a[a](x_zi,x_ni) for a,x_zi,x_ni in zip(a,x_z,x_n)]
    x = np.asarray(x)

    return x

def feature_noise_shift(a,z,y,x,dist,theta):
    '''
    for subspace bias

    keep d_shared in the middle aligned for both groups, with d total
    informative features for each group by appending noise at the end fo the
    feature vector for the advantaged group and prepending noise and moving the
    first few features to the end for the disadvantaged group
    '''
    d,N = x.shape
    d_shared = theta[0]

    d_noise = d-d_shared # noise dims per row
    d_total = d + d_noise # total dims

    # generate the noise
    x_n = np.random.multivariate_normal([0]*d_noise,np.eye(d_noise),N)
    # functions for combining noise and true vectors
    x_a = {0: lambda x,n: np.concatenate((x[:d_noise],n)),
          1: lambda x,n: np.concatenate((n, x[d_shared-1:d],  x[:d_noise]))}
    x = [x_a[a](x_zi,x_ni) for a,x_zi,x_ni in zip(a,x_z,x_n)]
    x = np.asarray(x)

    return x


def feature_noise_groupwise(a,z,y,x,dist,theta):
    '''
    add a groupwise noise to the feature vectors than the other
    '''

    x = [x_i + dist(theta[a_i]) for x_i,a_i in zip(x,a)]

    return x
