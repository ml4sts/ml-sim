import numpy as np

# CRP samplers
def sample_CRP(N, alpha,d =0):
    """
    sample from a Pitman-Yor process via the Chinese Restaraunt process, default
    value of d=0 samples from a Dirichlet process

    Parameters
    -----------
    N : scalar, integer
        number of samples to return
    alpha : scalar > -d
        concentration parameter
    d : [0,1)

    Returns
    --------
    z : list
        list of integers in [0,K] sampled from the CRP,
    """

    pi = [1]

    z = []

    for n in range(N):
        # sample from pi
        z.append(np.random.choice(len(pi),p=pi))
        K = max(z_py) +1
        # update counts

        counts,e = np.histogram(z_py,bins = np.arange(K+1)-.5)
        # append alpha and normalize to a distribution
    #     denoms = np.append()
        pi = np.append(counts - d,alpha + d*K)/(alpha + n +1)

    return z

# IBP Sampler

def p_row(p):
    """
    sample a binary vector where the ith element is 1 with probability p_i
    """
    return np.asarray([np.random.choice([1,0],p=[p_i, 1-p_i]) for p_i in p])

def sample_IBP(N, gamma):
    """
    sample from a Pitman-Yor process via the Chinese Restaraunt process

    Parameters
    -----------
    N : scalar, integer
        number of samples to return
    alpha : scalar > -d
        concentration parameter
    d : [0,1)

    Returns
    --------
    z : list of lists
        list of N binary lists of up to length K sampled from the IBP,
    """

    z = []

    z_tmp = np.ones(np.random.poisson(gamma))
    m = np.zeros(z_tmp.shape)
    z.append(z_tmp)

    for n in range(1,D):
        m += z_tmp
    #     print(m)
        p = m/(n+1)
    #     print(p)
        new = np.random.poisson(gamma/n)
        z_tmp = np.concatenate((p_row(p),np.ones(new)))
        m = np.concatenate((m,np.zeros(new)))
        z.append(z_tmp)

    return z
