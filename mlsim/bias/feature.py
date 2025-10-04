import numpy as np
import pandas as pd
from collections import namedtuple
from collections.abc import Iterable

from .bias_components import Sampler
FeatureParams = namedtuple('FeatureParams',['distfunc','theta'])


def mean_only_mvn(mu):
    '''
    multivariate normal with identity covariance

    Parameters
    ----------
    mu : list-like
        mean of the multivariate normal
    '''
    return np.random.multivariate_normal(mu,np.eye(len(mu)))

def mvn(mu,cov):
    '''
    multivariate normal with general covariance

    Parameters
    ----------
    mu : list-like
        mean of the multivariate normal
    cov : float or list-like
        if float, then interpreted as isotropic covariance, otherwise must be
        a square matrix of size len(mu)
    '''
    if type(cov) == float:
        cov = cov*np.eye(len(mu))
    return np.random.multivariate_normal(mu,cov)


# def cat_dist(mu):

class Feature(Sampler):
    '''
    base class for all feature samplers: P(X|A,Z,Y) by default creates two
    dimensional features with shared parameters across groups and good
    separability of classes

    Attributes
    ----------
    dist : function handle
        function to sample X|parameters where the paramters are dependend on
         Z,A,Y
    theta : list-like or list of tupples
        params of dist, one per value of z,a, y

    '''
    ParamCreator = FeatureParams
    def __init__(self,dist= mean_only_mvn,mu = [[5,2],[2,5]],
                            param_tuple = None,N_a =2):
        '''
        Parameters
        ----------
        dist : function handle
            function to sample X|parameters where the paramters are dependend on
             Z,A,Y through theta default mean only multivariate_normal
        mu : list like
            parameters for dist, for each value of z, inner list can be tuple or list depending
            on need of dist
        N_a : int
            number of values for A
        '''
        # same mean for both values of y and a
        if param_tuple:
            # used by subclasses
            super().__init__(param_tuple)
        else:
            # default params passed
            # mu has diffs for Z=0,1; repeat for all A for all Y
            N_y = len(mu) # |Y| = |Z| // z and y have sam enumber of values
            theta = [[mu]*N_a]*N_y
            super().__init__((dist,theta))

    def sample(self,a,z,y):
        '''
        sample P(X|A,Z,Y) using distribution and parameters initialized for
        each a,z,y. The vectors a,z,y must be the same shape

        Parameters
        ----------
        a : list-like length n
            demographic variables
        z : list like length n
            true target
        y : list-like length n
            proxy target


        Returns
        --------
        x : list like, length n
            featuers, same shape as a,z,y
        '''

        if type(self.params.theta[0][0][0]) == tuple:
            # if a tuple, then expand and pass 2 params
            
            x = [self.params.distfunc(*self.params.theta[yi][ai][zi])
                                        for ai,zi,yi in zip(a,z,y)]
        else:
            x = [self.params.distfunc(self.params.theta[yi][ai][zi])
                                    for ai,zi,yi in zip(a,z,y)]
        return np.asarray(x)


class FeatureSharedParam(Feature):
    '''
    feature sampler with two total parameters and one parameter shared across Z (eg shared spread)
    A and Y have no impact on X
    '''

    def __init__(self, loc, spread, dist=mvn,N_a=2):
        '''
        unique locations and shared spread for no impact of A or Y

        Parameters
        -----------
        dist : function handle
            function to sample X|parameters where the paramters are dependend on
             Z,A,Y
        loc : list-like length |Z|
            location parameter of dist, one per value of z
        spread : scalar
            shared spread parameter of dist
        '''

        theta_z = [(li,spread) for li in loc]
        theta = [[theta_z]*N_a]*len(loc)
        super().__init__(param_tuple=(dist,theta))

class FeatureTwoParams(Feature):
    '''
    feature sampler with two unique parameters per class
    '''

    def __init__(self, loc, spread, dist=mvn,N_a=2):
        '''
        unique locations and shared spread for z, no impact of a an y

        Parameters
        -----------
        dist : function handle
            function to sample X|parameters where the paramters are dependend on
             Z,A,Y
        loc : list-like length |Z|
            location parameter of dist, one per value of z
        spread : list-like length |Z|
            spread parameter of dist, one per value of z
        '''

        theta_z = [(li, si) for li, si in zip(loc, spread)]
        theta = [[theta_z]*N_a]*2
        super().__init__(param_tuple=(dist,theta))

class FeaturePerGroupTwoParam(Feature):
    '''
    feature sampler with two parameters that vary per group
    '''
    def __init__(self,dist,loc,spread):
        '''
        for feature bias where P(X|Z,Y, A=0) != P(X|Z,Y, A=1)

        Parameters
        -----------
        dist : function handle
            function to sample X|parameters where the paramters are dependend on
             Z,A,Y
        loc : list-like length |Z| of list-like length |A| 
            location parameter of dist, one per value of z,a; for multivariate feature spaces
            in genral the location paramters will each be a list of length = number of features
        spread : list-like length |Z| of lists length  |A|
            spread parameter of dist, one per value of z,a
        # '''
        # print(len(loc), len(spread))
        # print(len(loc[0]), len(spread[0]))
        theta_za = [[(lii,sii) for lii,sii in zip(li,si)] for li,si in zip(loc,spread)]
        # repeat so that features do not vary with Y
        theta = [theta_za,theta_za]
        # print(theta)
        super().__init__(param_tuple=(dist,theta))

class FeaturePerGroupSharedParamWithinGroup(Feature):
    '''
    '''
    def __init__(sel,dist,loc,spread):
        '''
        for feature bias where P(X|Z,Y, A=0) != P(X|Z,Y, A=1) but one
        parameter of dist is shared across groups, but unique per class

        Parameters
        -----------
        dist : function handle
            function to sample X|parameters where the paramters are dependend on
             Z,A,Y
        loc : list-like length |Z| of lists length 2
            location parameter of dist, one per value of z,a
        spread : list-like length |Z|
            spread parameter of dist, one per value of z
        '''
        theta_za = [[(laizi,covzi) for laizi,covzi in zip(laiz,spread)] for laiz in loc]
        # same for both values fo y
        theta = [theta_za,theta_za]
        super().__init__(param_tuple=(dist,theta))

class FeaturePerGroupSharedParamAcrossGroups(Feature):
    '''
    '''
    def __init__(sel,dist,loc,spread):
        '''
        for feature bias where P(X|Z,Y, A=0) != P(X|Z,Y, A=1) but one paramter
        is shared across groups and classes

        Parameters
        -----------
        dist : function handle
            function to sample X|parameters where the paramters are dependend on
             Z,A,Y
        loc : list-like length |Z| of lists length 2
            location parameter of dist, one per value of z,a
        spread : scalar
            spread parameter of dist
        '''
        theta_za = [[(laizi,spread) for laizi in laiz] for laiz in loc]
        # same for both values fo y
        theta = [theta_za,theta_za]
        super().__init__(param_tuple=(dist,theta))

class FeatureMeasurementQualityProxy(Feature):
    '''
    the measurement locations vary with the true target value z and the
    measurements spread vary with the meaured target value y, allowing for error
    to be present in both the features and the measurements. Also may vary with
    the protected attribute

    '''
    def __init__(self,dist,loc,spread):
        '''
        Parameters
        ----------
        loc : list-like
            one location parameter value per (true value, protected attribute) pair
        spread : list-like
            one spread parameter value per (proxy value, protected attribute) pair
        '''
        theta_yaz = [[[(lii,sii) for lii,sii in zip(li,si)]
                                for li in loc] for si in spread]

        super().__init__(param_tuple=(dist,theta_yaz))
