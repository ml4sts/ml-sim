import pandas as pd
import numpy as np
import aif360.datasets

# generators for biased data by models
def feature_bias(rho_a, rho_z, N, d, mu):
    '''
    Bias that occurs when different protected attributes have different means (mu)
    
    Parameters
    -----------
    rho_a : float
        p(a = 1)
    rho_z : float
        p(z = 1)
    N : int
         number of samples
    mu : matrix like, 2xD
        mu[0] is the mean for z=0, D= len(mu[0]) = number of features
    d : int
        total number of features

    Returns
    --------
    df : DataFrame
        a data frame with N rows and columns: a,y,z, x0:xD
    '''
    # portion of disadvantaged group
    p_a = [1-rho_a, rho_a]
    # portion of allocation of target variable
    p_z = [1-rho_z, rho_z]
    cov = np.eye(d)

    a = np.random.choice([0,1], p=p_a, size=N)
    z = np.random.choice([0,1], p=p_z, size=N)
    y = z
    x = [np.random.multivariate_normal(mu[a_i][z_i],cov) for a_i, z_i in zip(a,z)]

    x = np.asarray(x)
    # concatenate the data and p
    data = np.concatenate([np.asarray([a,z,y]).T,x],axis=1)

    labels =['a','z','y']
    labels.extend(['x'+str(i) for i in range(d)])
    df = pd.DataFrame(data=data, columns = labels)

    return df

def subspace_bias(rho_a, rho_z,  N, d, d_shared, mu):
    '''
    Bias that occurs when different features are informative for different protected classes (d not shared) 

    Parameters
    -----------
    rho_a : float
        p(a = 1)
    rho_z : float
        p(z=1)
    N : int
         number of samples
    mu : matrix like, 2xD
        mu[0] is the mean for a=0, mu[0][0] is the mean for a=0, z=0, D = len(mu[0][0) = number of features
    d : int
        total number of features
    d_shared : int
        number of shared features

    Returns
    --------
    df : DataFrame
        a data frame with N rows and columns: a,y,z, x0:xD
    '''
    p_a = [1-rho_a, rho_a]
    p_z = [1-rho_z, rho_z]

    cov = np.eye(d)

    d_noise = d-d_shared # noise dims per row
    d_total = d + d_noise # total dims

    a = np.random.choice([0,1], p=p_a, size=N)
    z = np.random.choice([0,1], p=p_z, size=N)
    y = z
    labels_protected = np.asarray([a,z,y]).T
    x_z = [np.random.multivariate_normal(mu[z_i],cov) for z_i in z]
    x_n = np.random.multivariate_normal([0]*d_noise,np.eye(d_noise),N)
        # functions for combining noise and true vectors
    x_a = {0: lambda x,n: np.concatenate((x,n)),
          1: lambda x,n: np.concatenate((n, x[d_shared-1:d],  x[:d_noise]))}
    x = [x_a[a](x_zi,x_ni) for a,x_zi,x_ni in zip(a,x_z,x_n)]
    x = np.asarray(x)
    # concatenate the data and p
    data = np.concatenate([labels_protected,x],axis=1)

    labels =['a','z','y']
    labels.extend(['x'+str(i) for i in range(d_total)])
    df = pd.DataFrame(data=data, columns = labels)

    return df

def label_bias(rho_a, rho_z, beta, N, mu, cov):
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
    x = [np.random.multivariate_normal(mu[z_i],cov) for z_i in z]
    y = [np.random.choice([zi,1-zi],p=[1-beta[ai], beta[ai]]) for ai,zi in zip(a,z)]
    # convert to numpy arrays and reshape
    labels_protected = np.asarray([a,z,y]).T
    x = np.asarray(x)
    # concatenate the data and p
    data = np.concatenate([labels_protected,x],axis=1)
    labels =['a','z','y']
    labels.extend(['x'+str(i) for i in range(len(mu[0]))])
    df = pd.DataFrame(data=data, columns = labels)

    return df

def convert_to_dataset(df, label_names, protected_attribute_names):
    '''
    Converts a dataframe created by one of the above functions into a dataset usable in IBM 360 package

    Parameters
    -----------
    df : pandas dataframe
    label_names : optional, a list of strings describing each label
    protected_attribute_names : optional, a list of strings describing features corresponding to      protected attributes

    Returns
    --------
    aif360.datasets.StructuredDataset

    '''
    return aif360.datasets.StructuredDataset(df, label_names, protected_attribute_names)
