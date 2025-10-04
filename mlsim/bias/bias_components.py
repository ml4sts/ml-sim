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

