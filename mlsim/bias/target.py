import numpy as np
import pandas as pd
from collections import namedtuple
from collections.abc import Iterable
from .bias_components import Sampler


TargetParams = namedtuple('TargetParams',['Py_az'])
FeatureParams = namedtuple('FeatureParams',['distfunc','theta'])


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
