import numpy as np
import pandas as pd

def demographic_template(N):
    a = np.random.choice([0,1], size=N)
    z = np.random.choice([0,1], size=N)
    return np.asarray([a,z]).T

def target_template(a,z, beta):
    y = z
    return np.asarray(y).T

def feature_template(a,z,y,dist,theta):
    '''
    features sampled per true group only

    dist : function handle
    theta : params of dist, one per value of z
    '''
    x = [dist(theta[z_i]) for z_i in z]
    return np.asarray(x)

def feature_noise_template(a,z,y,x,dist,theta):
    '''
    no noise added
    '''
    return x


class Population():
    '''
    base class
    '''
    def __init__(self, demographic_sampler, target_sampler, feature_sampler,
                feature_noise_sampler, parameter_dictionary):
        '''
        initialize a population based on the way to sample from it

        Parameters:
        -----------
        population_sampler : function handle
            function to sample from the distribution
        '''
        self.parameters  = parameter_dictionary
        self.demographic_sampler = demographic_sampler
        self.target_sampler = target_sampler
        self.feature_sampler = feature_sampler
        self.feature_noise_sampler = feature_noise_sampler

    def sample_population(self, N):
        '''
        sample N members of the  population, according to its underlying
        distribution
        '''
        a,z

        # concatenate the data and p
        data = np.concatenate([labels_protected,x],axis=1)
        labels =['a','z','y']
        labels.extend(['x'+str(i) for i in range(D)])
        df = pd.DataFrame(data=data, columns = labels)

        return df

    def sample_unfavorable(self,N,skew):
        '''
        sample so that the disadvantaged group (a=1) gets the favorable
        outcome (y=1) less often based on the skew
        '''


    def




def sample_bias(df,frac,favor_factor,sample_name = 'train'):
    '''
    sample to favor a=1 by favor_ratio amount in percent of the dataset

    Parameters
    -----------
    df : DataFrame
    percent : float
        percentage of the data to include in the sample
    favor_ratio : float
        how much to overrepresent a=1.

    Returns
    -------
    df : DataFrame
        data frame with an added column
    '''
    N = len(df)

    weights_a = {0:1, 1:favor_factor}
    p_sample = lambda row: weights_a[row['a']]

    weights = df.apply(p_sample)

    sample_df = df.sample(frac =percent,weights)
