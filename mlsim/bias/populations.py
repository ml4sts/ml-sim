import numpy as np
import pandas as pd
from aif360.datasets import  StructuredDataset
from .bias_components import Demographic, Target, Feature, FeatureNoise

default_params = {'dem':None,}

class Population():
    '''
    Object for describing a population so that sampling from the population
    and biased samples are possible from a sampler type and parameter dictionary
    '''
    def __init__(self, demographic_sampler= Demographic,
                target_sampler = Target,
                feature_sampler = Feature,
                feature_noise_sampler = FeatureNoise, parameter_dictionary = {}):
        '''
        initialize a population based on the way to sample from it. a population
        object has properties that define the samplers for the demographic
        variables (A,Z) the observed target (Y) and the features (X)

        Parameters:
        -----------
        demographic_sampler : Demographic
            a sampler that inherits from mlsim.Demographic
        target_sampler : Target,
        feature_sampler : Feature,
        feature_noise_sampler : FeatureNoise,
        parameter_dictionary : dictionary default empty
        '''


        required_keys = ['dem','target','feat','featnoise']

        #ensure required keys are set
        for key in required_keys:
            if not(key in parameter_dictionary.keys()):
                parameter_dictionary[key]= None

        dem_params = parameter_dictionary['dem']
        target_params = parameter_dictionary['target']
        feat_params = parameter_dictionary['feat']
        featnoise_params = parameter_dictionary['featnoise']

        # initialize objects for each, with parameters if provided
        if dem_params:
            self.demographic_sampler = demographic_sampler(*dem_params)
        else:
            self.demographic_sampler = demographic_sampler()

        if target_params:
            self.target_sampler = target_sampler(*target_params)
        else:
            self.target_sampler = target_sampler()

        if feat_params:
            self.feature_sampler = feature_sampler(*feat_params)
        else:
            self.feature_sampler = feature_sampler()

        if featnoise_params:
            self.feature_noise_sampler = feature_noise_sampler(*featnoise_params)
        else:
            self.feature_noise_sampler = feature_noise_sampler()


    def sample(self, N,return_as = 'DataFrame'):
        '''
        sample N members of the  population, according to its underlying
        distribution

        Parameters
        -----------
        N : int
            number of samples
        return_as : string, 'dataframe'
            type to return as, can be pandas 'DataFrame' or IBM AIF360
            'structuredDataset'
        '''
        a,z = self.demographic_sampler.sample(N)
        y = self.target_sampler.sample(a,z)
        x = self.feature_sampler.sample(a,z,y)
        x = self.feature_noise_sampler.sample(a,z,y,x)

        if return_as == 'DataFrame':
            df = self.make_DataFrame(a,z,y,x)

        elif return_as == 'structuredDataset':
            df = self.make_StructuredDataset(a,z,y,x)


        return df

    def sample_unfavorable_outcomes(self,N,rho_z_scale):
        '''
        sample so that the disadvantaged group (a=1) gets the favorable
        outcome (y=1) less often based on the rho_z_scale
        '''
        # get original demographic parameters
        rho_z0 = self.demographic_sampler.get_rho_z()
        rho_a = self.demographic_sampler.get_rho_a()
        # scale rho_a
        rho_z = [rho_z0[0],rho_z0[1]*rho_z_scale]
        # sameple the demongraphic vars with the new sampler
        self.unfavorable_dem = self.DemographicCorrelated(rho_a,rho_z)
        a,z = self.unfavorable_dem.sample(N)

        # sample the rest as usual
        y = self.target_sampler.sample(a,z)
        x = self.feature_sampler.sample(a,z,y)
        x = self.feature_noise_sampler.sample(a,z,y,x)

        return self.make_DataFrame(a,z,y,x)

    def make_DataFrame(self,a,z,y,x):
        '''
        combine into data frame with labels

        Parameters
        ----------
        a : list
        '''
        # concatenate the data and p
        azy = np.vstack([a,z,y]).T
        data = np.concatenate([azy,x],axis=1)
        labels =['a','z','y']
        _,D = x.shape
        labels.extend(['x'+str(i) for i in range(D)])

        return pd.DataFrame(data=data, columns = labels)

    def make_StructuredDataset(self,a,z,y,x):
        '''
        Converts a dataframe created by one of the above functions into a dataset usable in IBM 360 package

        Parameters
        -----------
        df : pandas dataframe
        label_names : optional, a list of strings describing each label
        protected_attribute_names : optional, a list of strings describing
        features corresponding to      protected attributes

        Returns
        --------
        aif360.datasets.StructuredDataset containing the data with y as the target and a as protected attribute. 

        '''
        df = self.make_DataFrame(a,z,y,x)
        return StructuredDataset(df, ['y'], ['a'])

    def get_parameter_description(self):
        '''
        Build a string output that describes this object

        Returns
        --------
        description : string
            values of each parameter value grouped by sampler
        '''
        description = ''


        description += 'Demographic Parameters\n'
        description += self.demographic_sampler.params.__str__()
        description += '\nTarget Parameters \n'
        description += self.target_sampler.params.__str__()
        description += '\nFeature Parameters \n'
        description += self.feature_sampler.params.__str__()
        description += '\nFeature Noise Parameters \n'
        description += self.feature_noise_sampler.params.__str__()

        return description


class PopulationInstantiated(Population):
    '''
    To instantiate with either default parameters or instantiated sampler objects
    '''
    def __init__(self, demographic_sampler= Demographic(),
                target_sampler = Target(),
                feature_sampler = Feature(),
                feature_noise_sampler = FeatureNoise()):
        '''
        initialize a population based on the way to sample from it

        Parameters:
        -----------
        population_sampler : function handle
            function to sample from the distribution
        '''

        self.demographic_sampler = demographic_sampler
        self.target_sampler = target_sampler
        self.feature_sampler = feature_sampler
        self.feature_noise_sampler = feature_noise_sampler


class PopulationProxyTargets(PopulationInstantiated):
    '''
    Population for proxy target overfitting bias.

    Generates a population whose features are split into two sets: one set is
    informative of the true target ``z`` and a second "proxy" set is
    informative of the observed, possibly biased, target ``y``. When ``y``
    differs from ``z`` more often for one protected group (for example using
    ``TargetTwoError``), a model trained on ``y`` can rely on the proxy
    features and be systematically wrong about ``z`` for that group, producing
    a per-group accuracy gap.

    Parameters
    ----------
    demographic_sampler : Demographic
        Sampler for the protected attribute ``a`` and the true target ``z``.
    target_sampler : Target
        Sampler for the observed target ``y``; use a sampler with per-group
        label error (e.g. ``TargetTwoError``) so that ``y != z`` more for
        one group.
    feature_sampler_true : Feature
        Sampler for features informative of the true target ``z``; sampled as
        ``sample(a, z, y)``.
    feature_sampler_proxy : Feature
        Sampler for features informative of the proxy target ``y``; sampled as
        ``sample(a, y, y)`` so its class locations follow the observed label
        instead of the truth.
    feature_noise_sampler : FeatureNoise
        Sampler that adds noise to the concatenated feature matrix, applied as
        ``sample(a, z, y, x)``.

    See Also
    --------
    PopulationInstantiated : parent class taking instantiated samplers.
    TargetTwoError : per-group label error so that ``y != z`` more for one group.

    Notes
    -----
    The proxy features are produced by passing ``y`` in the position the base
    samplers use for ``z``; this is the mechanism that makes them track the
    observed label rather than the truth.
    '''

    def __init__(self, demographic_sampler=Demographic(),
                target_sampler=Target(),
                feature_sampler_true=Feature(),
                feature_sampler_proxy=Feature(),
                feature_noise_sampler=FeatureNoise()):
        self.demographic_sampler = demographic_sampler
        self.target_sampler = target_sampler
        self.feature_sampler = feature_sampler_true
        self.feature_sampler_proxy = feature_sampler_proxy
        self.feature_noise_sampler = feature_noise_sampler

    def sample(self, N, return_as='DataFrame'):
        '''
        Sample ``N`` members of the population.

        The returned feature columns are the true-target features followed by
        the proxy-target features.

        Parameters
        ----------
        N : int
            Number of samples to draw.
        return_as : string, 'DataFrame'
            Type to return as, either pandas ``'DataFrame'`` or IBM AIF360
            ``'structuredDataset'``.

        Returns
        -------
        df : pandas.DataFrame or aif360.datasets.StructuredDataset
            Columns ``a``, ``z``, ``y`` followed by the feature columns
            ``x0 ... x(d_true + d_proxy - 1)``.
        '''
        a, z = self.demographic_sampler.sample(N)
        y = self.target_sampler.sample(a, z)

        x_true  = self.feature_sampler.sample(a, z, y)
        x_proxy = self.feature_sampler_proxy.sample(a, y, y)  # y in z slot = proxy trick
        x = np.concatenate([x_true, x_proxy], axis=1)
        x = self.feature_noise_sampler.sample(a, z, y, x)

        if return_as == 'DataFrame':
            df = self.make_DataFrame(a, z, y, x)
        elif return_as == 'structuredDataset':
            df = self.make_StructuredDataset(a, z, y, x)

        return df
