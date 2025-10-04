import numpy as np
import pandas as pd
from collections import namedtuple
from collections.abc import Iterable
from .bias_components import Sampler


DemParams = namedtuple('DemParams',['Pa','Pz_a'])

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
