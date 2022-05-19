"""
commonly used basic generators
"""
import numpy as np
import scipy.stats as stats



def means_with_spread(mu_mu,cov,k):
    """
    sample k means from a gaussian distribution and downsample with a SE kernel
    for a point process-like effect

    Parameters
    -----------
    mu_mu : 2vector
        center of means
    cov : 2x2
        covariance matrix for means
    k : scalar
        number of means to return

    Returns
    -------
    mu_sort : vector [2,k]
        means sorted by distance

    """
    # define a sampling function so that we can sample jointly instead of
    next_sample = lambda: np.random.multivariate_normal(mu_mu, cov)

    # we'll use a gaussian kernel around each to filter
    # only the closest point matters
    # scale here probably should be set to help provide guarantees
    dist = lambda mu_c,x: stats.norm.pdf(min(np.sum(np.square(mu_c -x),axis=1)))


    # keep the first one
    mu = [next_sample()]
    p_dist = [1]

    while len(mu) <= k:
        m = next_sample()
        p_keep = 1- dist(mu,m)
        if p_keep > .97:
            mu.append(m)
            p_dist.append(p_keep)

    mu = np.asarray(mu)

    # sort by distance
    mu_sort, p_sort = zip(*sorted(zip(mu,p_dist),
                key = lambda x: x[1], reverse =True))

    return mu_sort
