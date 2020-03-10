# generators for biased data by models

# rho_a - float
# rho_z - float
# beta - float array size 2, error rate for each group
# N - int, number of samples
# d - int, number of features
# mu - array of arrays with dimensions [a][z]
# d_shared - int, number of shared features
def feature_bias2(rho_a, rho_z, beta, N, d, d_shared, mu):
    # portion of disadvantaged group  
    p_a = [1-rho_a, rho_a]
    # portion of allocation of target variable
    p_z = [1-rho_z, rho_z]
    cov = np.eye(d)
    
    a = np.random.choice([0,1], p=p_a, size=N)
    z = np.random.choice([0,1], p=p_z, size=N)
    x = [np.random.multivariate_normal(mu[a_i][z_i],cov) for a_i, z_i in zip(a,z)]

    x = np.asarray(x)
    
    y = z
    data = np.asarray([a,z,y]).T
    df = pd.DataFrame(data=data, columns = ['a','z','y'])

    var_list = []
    for i in range(d):
        var = 'x' + str(i)
        df[var] = x[:,i]
        var_list.append(var)
    
    df.head()
    
    return df

def feature_bias(rho_a, rho_z, beta, N, d, d_shared, mu):
    p_a = [1-rho_a, rho_a]
    p_z = [1-rho_z, rho_z]
    
    cov = np.eye(d)
    
    d_noise = d-d_shared # noise dims per row
    d_total = d + d_noise # total dims

    a = np.random.choice([0,1], p=p_a, size=N)
    z = np.random.choice([0,1], p=p_z, size=N)
    x_z = [np.random.multivariate_normal(mu[z_i],cov) for z_i in z]
    x_n = np.random.multivariate_normal([0]*d_noise,np.eye(d_noise),N)
        # functions for combining noise and true vectors
    x_a = {0: lambda x,n: np.concatenate((x,n)), 
          1: lambda x,n: np.concatenate((n, x[d_shared-1:d],  x[:d_noise]))}
    x = [x_a[a](x_zi,x_ni) for a,x_zi,x_ni in zip(a,x_z,x_n)]
    x = np.asarray(x)


    y = z
    data = np.asarray([a,z,y]).T
    df = pd.DataFrame(data=data, columns = ['a','z','y'])

    var_list = []
    for i in range(d_total):
        var = 'x' + str(i)
        df[var] = x[:,i]
        var_list.append(var)

    return df

def label_bias(rho_a, rho_z, beta, N, mu, cov):
    p_a = [1-rho_a, rho_a]
    p_z = [1-rho_z, rho_z]
    
    a = np.random.choice([0,1], p=p_a, size=N)
    z = np.random.choice([0,1], p=p_z, size=N)
    x = [np.random.multivariate_normal(mu[z_i],cov) for z_i in z]
    y = [np.random.choice([zi,1-zi],p=[1-beta[ai], beta[ai]]) for ai,zi in zip(a,z)]
    data = np.asarray([a,z,y]).T
    x = np.asarray(x)
    df = pd.DataFrame(data=data, columns = ['a','z','y'])
    
    df['x0'] = x[:,0]
    df['x1'] = x[:,1]
    
    return df