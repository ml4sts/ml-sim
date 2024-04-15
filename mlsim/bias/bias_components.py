import numpy as np
import pandas as pd
from collections import namedtuple
from collections.abc import Iterable

DemParams = namedtuple('DemParams',['Pa','Pz_a'])
TargetParams = namedtuple('TargetParams',['Py_az'])
FeatureParams = namedtuple('FeatureParams',['distfunc','theta'])
NoiseParams = namedtuple('NoiseParams',['noisefunc','theta'])

class Sampler():
    '''
    base class for all samplers
    '''
    def __init__(self,param_tuple):
        '''
        '''
        self.params = self.ParamCreator(*param_tuple)


    # def sample():
    #     '''
    #     '''
    #     return self.outputs

class Demographic(Sampler):
    '''
    base class for sampling demographics (a= protected attribute,z = true target
    value)
    '''
    ParamCreator = DemParams

    def __init__(self,rho_a=.5,rho_z=.5):
        '''
        P(A = 1) = rho_a
        P(Z=1) = rho_z

        default is independent sampling of a and z
        '''
        Pa = [1-rho_a, rho_a]
        self.A = [0, 1]

        Pz = [1-rho_z, rho_z]
        super().__init__((Pa,[Pz,Pz]))


    def sample(self,N):
        '''
        Sample P(A,Z) = P(Z|A)P(A)

        Parameters
        -----------
        N : integer
            number of samples to return

        Returns
        -------
        a_z_tuple : Tuple
            a tuple of lenght 2 with elements a and z as column np arrays each
            of length N
        '''
        a = np.random.choice(self.A, p= self.params.Pa, size=N)
        z = [np.random.choice([0,1], p= self.params.Pz_a[ai]) for ai in a]

        return np.asarray(a).T,np.asarray(z).T

    def get_rho_a(self):
        '''
        get  P(A=1)

        Parameters
        -----------

        Returns
        -------
        rho_a : float
            Probability of being in the disadvantaged group, A =1
        '''
        return self.params.Pa[1]

    def get_rho_z(self):
        '''
        return P(Z=1|A)

        Parameters
        -----------

        Returns
        -------
        rho_z : nparray of floats
            probability of the favorable outcome(z =1) for A=0 and A=1 in that
            order
        '''

        return np.asarray(self.params.Pz_a)[:,1]

class DemographicIndependent(Demographic):
    '''
    '''
    def __init__(self,rho_a=.2,rho_z=.1):
        '''
        P(A = 1) = rho_a
        P(Z=1) = rho_z

        default is independent sampling of a and z
        '''
        super().__init__(rho_a,rho_z)


class DemographicCorrelated(Demographic):
    '''
    '''

    def __init__(self,rho_a=.5,rho_z=[.5,.3]):
        '''
        P(A = 1) = rho_a or P(A) = rho_a
        P(Z=1|A=i) = rho_z[i]

        Parameters
        rho_a : scalar or vector of floats
            probablity of A = 1 or distribution of A
        rho_z : vector of 2 or len(rho_a)
            probability Z=1, for A = i
        '''
        if isinstance(rho_a, Iterable):
            Pa = rho_a
            self.A = list(range(len(rho_a)))
        else:
            Pa = [1-rho_a, rho_a]
            self.A = [0, 1]

        Pz_a = [[1-rho_zi, rho_zi] for rho_zi in rho_z]

        Sampler.__init__(self,(Pa,Pz_a))


class Target(Sampler):
    '''
    '''
    ParamCreator = TargetParams
    def __init__(self,beta=0.05,N_a=2):
        '''
        P(Y=Z|A,Z ) = P(Y=Z) = 1-beta
        make errors with prob beta

        beta =0, makes Y =Z
        '''
        pyeqz = [1-beta,beta]
        Py_az = [[pyeqz,pyeqz]]*N_a
        super().__init__((Py_az,))


    def sample(self,a,z):
        '''
        sample P(Y|A,Z) via P(Y=Z|A,Z)
        Parameters
        -----------
        a :
        z :
        beta : float

        '''
        y = [np.random.choice([zi,1-zi],p= self.params.Py_az[ai][zi])
                                            for ai,zi in zip(a,z)]

        return np.asarray(y).T


class TargetDisadvantagedError(Target):
    '''
    '''
    def __init__(self,beta=.1,N_a=2):
        '''
        make errors with prob beta (advantaged, A=(N_a-1))
        P(Y=Z|A=1,Z ) = P(Y=Z|A=1) = 1-beta
        P(Y=Z|A=0,Z ) = P(Y=Z|A=0) = 1

        '''
        pyeqz = [1-beta,beta]
        Py_az = [[pyeqz, pyeqz]]*(N_a-1) + [[1, 0], [1, 0]]
        Sampler.__init__(self,(Py_az,))

class TargetTwoError(Target):
    '''
    '''
    def __init__(self,beta=[0,.1]):
        '''
        make errors with prob beta
        P(Y=Z|A=1,Z ) = P(Y=Z|A=1) = 1-beta1
        P(Y=Z|A=0,Z ) = P(Y=Z|A=0) = 1-beta0

        '''
        pyz_a0 = [1-beta[0],beta[0]]
        pyz_a1 = [1-beta[1],beta[1]]
        Py_az = [[pyz_a0,pyz_a0],[pyz_a1,pyz_a1]]
        Sampler.__init__(self,(Py_az,))


class TargetAllAError(Target):
    '''
    '''

    def __init__(self, beta=[0, .1]):
        '''
        make errors with prob beta
        P(Y=Z|A=1,Z ) = P(Y=Z|A=1) = 1-beta1
        P(Y=Z|A=0,Z ) = P(Y=Z|A=0) = 1-beta0

        # '''
        # pyz_a0 = [1-beta[0], beta[0]]
        # pyz_a1 = [1-beta[1], beta[1]]
        Py_az =  [[1-betaai, betaai]*2 for betaai in beta]
        Sampler.__init__(self, (Py_az,))

class TargetFlipNegative(Target):
    '''
    '''
    def __init__(self,beta=[0,.1]):
        '''

        make errors with prob beta only for the Z=1 class
        P(Y=Z|A=1,Z =1 ) = 1-beta[1]
        P(Y=Z|A=0,Z = 1) = 1-beta[0]
        P(Y=Z|Z  =0) = 1

        '''
        # pyz1_a0 = [1-beta[0],beta[0]]
        # pyz1_a1 = [1-beta[1],beta[1]]
        no_error = [1,0] # if z=0, P(Y=z) =1
        Py_az = [[no_error, [1-betaai, betaai]] for betaai in beta]
        Sampler.__init__(self,(Py_az,))

class TargetFlipAllIndep(Target):
    '''
    '''
    def __init__(self,beta=[[.05,.1],[.05,.1]]):
        '''
        make errors with prob beta for all possible combinations of A,Z
        P(Y=Z|A=1,Z =1 ) = 1- beta[1][1]
        P(Y=Z|A=0,Z = 1) = 1- beta[0][1]
        P(Y=Z|A=1,Z =0 ) = 1- beta[1][0]
        P(Y=Z|A=0,Z = 0) = 1- beta[0][0]


        '''
        Py_az = [[[1-b,b] for b in be] for be in beta]
        Sampler.__init__(self,(Py_az,))

mean_only_mvn = lambda mu :np.random.multivariate_normal(mu,np.eye(len(mu)))

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
        params of dist, one per value of z

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
            parameters for dist, for each value of z
        '''
        # same mean for both values of y and a
        if param_tuple:
            # used by subclasses
            super().__init__(param_tuple)
        else:
            # default params passed
            # mu has diffs for Z=0,1; repeat for all A for all Y
            theta = [[mu]*N_a]*2
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

mvn = lambda mu,var :np.random.multivariate_normal(mu,var*np.eye(len(mu)))

class FeatureSharedParam(Feature):
    '''
    feature sampler with one parameter shared across Z (eg shared spread)
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
        theta = [[theta_z]*N_a]*2
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

# you can use the existing feature sampler tha takes in means and just pass a mu
# where your function takes in a distance and computes the mu
# you could even randomly sample one mu[0]  then add the distance to get mu[1]
# and pass that value to the Feature constructor
class FeatureOneParam(Feature):

    '''Feature sampler with one parameter that defines the distance between the means of the data (X) for A=0 and A=1 '''

    def __init__(self, distribution, distance_between_means,mu0):
        '''
        for label bias where P(Y = Z | A = 0) != P(Y = Z | A = 1)
        We will generate mu[0], by randomly sampling. Then we will add the distance_between_means to compute mu[1]. This way, the user only inputs the distance between the two means 

        Parameters 
        ----------
        distribution : function handle
            function to sample X 

        distance_between_means: float
            fixed distance between the mean of the data when A = 0 versus A = 1

        '''
        mu1 = mu0 + distance_between_means

        
        
        
        super().__init__(param_tuple= (distribution,))

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
        loc : list-like length |Z| of lists length |A|
            location parameter of dist, one per value of z,a
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
        d_shared = d_shared

        d_adv_noise = int(np.floor((d-d_shared)/2)) # noise dims per row
        d_dis_noise = int(np.ceil((d-d_shared)/2))
        d_adv_signal = d_shared + d_dis_noise # total dims
        d_dis_signal = d_shared + d_adv_noise
        d_noise = max(d_pad_a,d_pad_d)

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
