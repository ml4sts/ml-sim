import numpy as np
import pandas as pd
import string

from .basic_gens import means_with_spread

def geometric_2d_gmm_sp(r_clusters,cluster_size,cluster_spread,p_sp_clusters,
                domain_range,k,N,p_clusters=None):
    """
    Sample from a gaussian mixture model with Simpson's Paradox and spread means
    return data in a data fram

    r_clusters : scalar [0,1]
        correlation coefficient of clusters
    cluster_size : 2 vector
        variance in each direction of each cluster
    cluster_spread : scalar [0,1]
        pearson correlation of means
    p_sp_clusters : scalar in [0,1]
        portion of clusters with SP
    p_clusters : vector in [0,1)^k, optional
        probabilty of membership of a sample in each cluster (controls relative
        size of clusters) default is [1.0/k]*k for uniform
    domain_range : [xmin, xmax, ymin, ymax]
        planned region for points to be in, means will be in middle 80%
    k : integer
        number of clusters
    N : scalar
        number of points
    """

    # if not defined, set uniform cluster probaiblity
    if p_clusters is None:
        p_clusters = [1.0/k]*k

    # sample the data
    x, z = data_only_geometric_2d_gmm(r_clusters,cluster_size,cluster_spread,
                                      p_sp_clusters,
                                      domain_range,k,N,p_clusters)

    # make a dataframe
    latent_df = pd.DataFrame(data=x,
                           columns = ['x1', 'x2'])

    # code cluster as color and add it a column to the dataframe
    latent_df['color'] = z


    return latent_df

def geometric_indep_views_gmm_sp(d,r_clusters,cluster_size,cluster_spread,p_sp_clusters,
                domain_range,k,N,p_clusters=None,numeric_categorical=False):
    """
    Sample from a gaussian mixture model with Simpson's Paradox and spread means
    return data in a data fram

    d : integer
        number of independent views, groups of 3 columns with sp
    r_clusters : scalar [0,1] or list of d
        correlation coefficient of clusters
    cluster_size : 2 vector or list of d
        variance in each direction of each cluster
    cluster_spread : scalar [0,1] list of d
        pearson correlation of means
    p_sp_clusters : scalar in [0,1] list of d
        portion of clusters with SP
    p_clusters : vector in [0,1)^k, optional or list of d vectors
        probabilty of membership of a sample in each cluster (controls relative
        size of clusters) default is [1.0/k]*k for uniform
    domain_range : [xmin, xmax, ymin, ymax] list of d
        planned region for points to be in, means will be in middle 80%
    k : integer or list of d
        number of clusters
    N : scalar
        number of points, shared across all views
    numeric_categorical=False
        use numerical (ordinal) values instead of letters
    """
    log_info = format_log(locals(),'geometric_indep_views_gmm_sp')

    # if not defined, set uniform cluster probaiblity
    if p_clusters is None:
        p_clusters = [1.0/k]*k


    # make inputs lists if not
    sclar_to_list = lambda x: [x]*d # if float, make d list by repeats

    if type(r_clusters) in [float, int]:
        r_clusters = sclar_to_list(r_clusters)

    if type(cluster_spread) in [float, int]:
        cluster_spread = sclar_to_list(cluster_spread)

    if type(p_sp_clusters) in [float, int]:
        p_sp_clusters = sclar_to_list(p_sp_clusters)

    if type(k) is int:
        k = sclar_to_list(k)

    if type(p_clusters[0]) in [float, int]:
        p_clusters = sclar_to_list(p_clusters)

    if type(cluster_size[0]) in [float, int]:
        cluster_size = sclar_to_list(cluster_size)

    if type(domain_range[0]) in [float, int]:
        domain_range = sclar_to_list(domain_range)


    # set x to none for logic below to add stuff
    x = None
    z = []

    for r,c_std,c_sp,p_sp, d_r,k,rho in zip(r_clusters,cluster_size,cluster_spread,p_sp_clusters,
                            domain_range,k,p_clusters):
        # sample the data
        x_tmp, z_tmp = data_only_geometric_2d_gmm(r,c_std,c_sp,p_sp, d_r,k,N,rho)
        # x.append(x_tmp)
        if x is None:
            x = x_tmp
        else:
            x = np.append(x,x_tmp,axis=1)
        z.append(z_tmp)

    col_names = ['x'+ str(i+1) for i in range(d*2)]

    # make a dataframe
    print(len(x))
    print(len(x[0]))

    latent_df = pd.DataFrame(data=x,
                           columns = col_names )

    #cluster naming will be name the columns: A, B, ...
    # valuses will be A1, A2, ..., Ak...
    z_names = list(string.ascii_uppercase[:d])
    # code cluster as and add it a column to the dataframe
    for z_i,name in zip(z,z_names):
        if numeric_categorical:
            latent_df[name] = [z_ii for z_ii in z_i]
        else:
            latent_df[name] = [name + str(z_ii) for z_ii in z_i]


    return latent_df

def data_only_geometric_2d_gmm(r_clusters,cluster_size,cluster_spread,p_sp_clusters,
                domain_range,k,N,p_clusters):
    """
    private, sampler only, returns raw variables, utily for sharing in other
    samplers
    Sample from a gaussian mixture model with Simpson's Paradox and spread means

    r_clusters : scalar [0,1]
        correlation coefficient of clusters
    cluster_size : 2 vector
        variance in each direction of each cluster
    cluster_spread : scalar [0,1]
        pearson correlation of means
    p_sp_clusters : scalar in [0,1]
        portion of clusters with SP
    p_clusters : vector in [0,1)^k, optional
        probabilty of membership of a sample in each cluster (controls relative
        size of clusters) default is [1.0/k]*k for uniform
    domain_range : [xmin, xmax, ymin, ymax]
        planned region for points to be in, means will be in middle 80%
    k : integer
        number of clusters
    N : scalar
        number of points
    """
    # define distribution for means, using the range provided
    mu_mu = [np.mean(domain_range[:2]),np.mean(domain_range[2:])]
    # first set correlation mat for means
    mu_sign = - np.sign(r_clusters)
    corr = [[1, mu_sign*cluster_spread],[mu_sign*cluster_spread,1]]
    # use a trimmed range to comput std
    mu_trim = .2
    mu_transform = np.repeat(np.diff(domain_range)[[0,2]]*(mu_trim),2)
    mu_transform[[1,3]] = mu_transform[[1,3]]*-1 # sign flip every other
    mu_domain = [d + m_t for d, m_t in zip(domain_range,mu_transform)]
    d = np.sqrt(np.diag(np.diff(mu_domain)[[0,2]]))
    # construct covariance from correlation
    mu_cov = np.dot(d,corr).dot(d)

    # sample means
    mu = means_with_spread(mu_mu,mu_cov,k)

    # create cluster covariances for SP and not SP
    cluster_std = np.diag(np.sqrt(cluster_size))
    cluster_corr_sp = np.asarray([[1,r_clusters],[r_clusters,1]]) # correlation with sp
    cluster_cov_sp = np.dot(cluster_std,cluster_corr_sp).dot(cluster_std) #cov with sp
    cluster_corr = np.asarray([[1,-r_clusters],[-r_clusters,1]]) #correlation without sp
    cluster_cov = np.dot(cluster_std,cluster_corr).dot(cluster_std) #cov wihtout sp
    cluster_covs = [cluster_corr_sp, cluster_corr]


    # sample the[0,1] k times to assign each cluster to SP or not
    c_sp = np.random.choice(2,k,p=[p_sp_clusters,1-p_sp_clusters])


    # sample from a GMM
    z = np.random.choice(k,N,p_clusters)
    # sample data using the cluster assignments z, means and cluster covariances
    x = np.asarray([np.random.multivariate_normal(mu[z_i],
                                        cluster_covs[c_sp[z_i]]) for z_i in z])

    return x,z
