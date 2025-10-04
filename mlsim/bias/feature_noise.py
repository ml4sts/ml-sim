import numpy as np
import pandas as pd
from collections import namedtuple
from collections.abc import Iterable
from .bias_components import Sampler

NoiseParams = namedtuple('NoiseParams',['noisefunc','theta'])


shape_spread_only_mvn = lambda x,cov: x + np.random.multivariate_normal([0]*len(x),cov*np.eye(len(x)))

class FeatureNoise(Sampler):
    '''
    Base class for adding noise to features
    '''
    ParamCreator = NoiseParams

    def __init__(self, dist=shape_spread_only_mvn, sig=1.0, N_a=2):
        '''
        '''
        if type(sig) ==float:
            # constant noise
            theta = [[[sig,sig]]*N_a]*2
        else:
            # diff noise for protected attributes
            theta = [[sigi,sigi] for sigi in sig]*2

        super().__init__((dist,theta))

    def sample(self,a,z,y,x):
        '''
        add noise to the features conditions on a,z,y
        add a groupwise noise to the feature vectors than the other
        '''

        x = [self.params.noisefunc(xi,self.params.theta[yi][ai][zi])
                                for xi,ai,zi,yi in zip(x,a,z,y)]

        return np.asarray(x)



class FeatureNoiseReplace(FeatureNoise):
    '''
    feature noise that replcaes some of the features with noise according to
    mean and covariance attributes
    '''
    def __init__(self,dist,mu = [0,0,0],cov = [[1,0,0],[0,1,0],[0,0,1]],d_shared=1):
        '''
        for subspace bias

        keep the same number of total features, replace some with noise, keep
        d_shared in the middle valid for both groups; replace the first 1/2(ceiled)
        of the remaining with noise for the disadvantaged group and the last portion
        (floored) for the advantaged group

        Parameters
        ----------
        mu : List
            noise mean, default [0, 0, 0]
        cov: list
            noise covariance matrix, default is identity in 3 dimensions
        d_shared: int =1
            number of shared features that are informative for both groups
        '''
        d = len(mu)
        

        d_adv_noise = int(np.floor((d-d_shared)/2)) # noise dims per row
        d_dis_noise = int(np.ceil((d-d_shared)/2))
        d_adv_signal = d_shared + d_dis_noise # total dims
        d_dis_signal = d_shared + d_adv_noise
        # d_noise = max(d_pad_a,d_pad_d)

        # create masks to 0 out features or noise as appropriate for adding
        adv_data_mask = np.asarray([1]*d_adv_signal + [0]*d_adv_noise)
        adv_noise_mask = np.asarray([1-d for d in adv_data_mask])
        dis_data_mask = np.asarray([0]*d_dis_noise + [1]*d_dis_signal)
        dis_noise_mask = np.asarray([1-d for d in dis_data_mask])

        theta_adv = (mu,cov,adv_data_mask,adv_noise_mask)
        theta_dis = (mu,cov,dis_data_mask,dis_noise_mask)
        theta_az = [[theta_adv,theta_adv],
                    [theta_dis,theta_dis]]

        noisefunc = lambda x,theta: self.noise_replace_func(x,*theta)
        super().__init__((dist,[theta_az,theta_az]))


    def noise_replace_func(self,x,mu,cov,data_mask,noise_mask):
        # generate the noise
        x = x*data_mask + self.params.distfunc(mu,cov)*noise_mask

        return  x

class FeatureNoiseShift(FeatureNoise):
    '''
    TODO make work
    '''

    def sample(a,z,y,x,dist,theta):
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
        x = [x_a[a](x_zi,x_ni) for a,x_zi,x_ni in zip(a,x,x_n)]
        x = np.asarray(x)

        return x



# --------------------------------------------
# need to be incorporated




def feature_proxy(a,z,y,distfunc,theta):
    '''
    some features are related to the ground truth and some are realated to the
    proxy,

    Parameters
    ----------
    theta :
    '''
    loc = theta[0]
    spread = theta [1]
    x_signal = [distfunc(loc[z_i][a_i],spread[z_i][a_i]) for z_i,a_i in zip(z,a)]
    x_signal = np.asarray(x_signal)

    x_proxy = [distfunc(loc[y_i][a_i],spread[y_i][a_i]) for y_i,a_i in zip(y,a)]
    x_proxy = np.asarray(x_proxy)

    x = np.concatenate([x_signal,x_proxy])
    return x
