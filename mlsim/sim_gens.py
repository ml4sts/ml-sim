import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from mlsim import bias_generators

def feature_sim(rho_a, rho_z, N, d, mu, classifier):
    '''
    returns dataframe
    '''
    df1 = bias_generators.feature_bias(rho_a, rho_z, N, d, mu)
    df2 = bias_generators.feature_bias(rho_a, rho_z, N, d, mu)
    df3 = bias_generators.feature_bias(rho_a, rho_z, N, d, mu)
    
    y1 = df1['y']
    y2 = df2['y']
    y3 = df3['y']

    x_cols = ['x'+str(i) for i in range(d)]
    x1 = df1[x_cols]
    x2 = df2[x_cols]
    x3 = df3[x_cols]
    
    # first classifier predicts x2 based on training on df1
    clas1 = classifier()
    pred2 = clas1.fit(x1, y1).predict(x2)

    # second classifier predicts x3 based on training x2 and prediction from classifier 1
    clas2 = classifier()
    pred3a = clas2.fit(x2, pred2).predict(x3)

    # third classifier predicts x3 based on training x2 with true y values
    clas3 = classifier()
    pred3b = clas3.fit(x2, y2).predict(x3)

    df2['pred2'] = pred2
    df3['pred3a'] = pred3a
    df3['pred3b'] = pred3b
    #compare pred3a and pred3b
    df3['pred3a_acc'] = df3['y'] == df3['pred3a']
    df3['pred3b_acc'] = df3['y'] == df3['pred3b']
    df3['pred3a_acc'] = df3['pred3a_acc'].astype(int, copy = False)
    df3['pred3b_acc'] = df3['pred3b_acc'].astype(int, copy = False)
    
    return df3
    
def subspace_sim(rho_a, rho_z,  N, d, d_shared, mu, classifier):
    '''
    '''
    df1 = bias_generators.subspace_bias(rho_a, rho_z, N, d, d_shared, mu)
    df2 = bias_generators.subspace_bias(rho_a, rho_z, N, d, d_shared, mu)
    df3 = bias_generators.subspace_bias(rho_a, rho_z, N, d, d_shared, mu)
    
    y1 = df1['y']
    y2 = df2['y']
    y3 = df3['y']

    x_cols = ['x'+str(i) for i in range(d)]
    x1 = df1[x_cols]
    x2 = df2[x_cols]
    x3 = df3[x_cols]
    
    # first classifier predicts x2 based on training on df1
    clas1 = classifier()
    pred2 = clas1.fit(x1, y1).predict(x2)
    # second classifier predicts x3 based on training x2 and prediction from classifier 1
    clas2 = classifier()
    pred3a = clas2.fit(x2, pred2).predict(x3)
    # third classifier predicts x3 based on training x2 with true y values
    clas3 = classifier()
    pred3b = clas3.fit(x2, y2).predict(x3)
    
    # add columns to dataframe
    df2['pred2'] = pred2
    df3['pred3a'] = pred3a
    df3['pred3b'] = pred3b
    
    # compare pred3a and pred3b
    df3['pred3a_acc'] = df3['y'] == df3['pred3a']
    df3['pred3b_acc'] = df3['y'] == df3['pred3b']
    df3['pred3a_acc'] = df3['pred3a_acc'].astype(int, copy = False)
    df3['pred3b_acc'] = df3['pred3b_acc'].astype(int, copy = False)
    
    return df3
    
    
def label_sim(rho_a, rho_z, beta, N, d, mu, classifier):
    '''
    '''
    df1 = bias_generators.label_bias(rho_a, rho_z, beta, N, d, mu)
    df2 = bias_generators.label_bias(rho_a, rho_z, beta, N, d, mu)
    df3 = bias_generators.label_bias(rho_a, rho_z, beta, N, d, mu)
    
    y1 = df1['y']
    y2 = df2['y']
    y3 = df3['y']

    x_cols = ['x'+str(i) for i in range(len(mu[0]))]
    x1 = df1[x_cols]
    x2 = df2[x_cols]
    x3 = df3[x_cols]
    
    # first classifier predicts x2 based on training on df1
    clas1 = classifier()
    pred2 = clas1.fit(x1, y1).predict(x2)
    # second classifier predicts x3 based on training x2 and prediction from classifier 1
    clas2 = classifier()
    pred3a = clas2.fit(x2, pred2).predict(x3)
    # third classifier predicts x3 based on training x2 with true y values
    clas3 = classifier()
    pred3b = clas3.fit(x2, y2).predict(x3)
    
    # add columns to dataframe
    df2['pred2'] = pred2
    df3['pred3a'] = pred3a
    df3['pred3b'] = pred3b
    
    # compare pred3a and pred3b
    df3['pred3a_acc'] = df3['y'] == df3['pred3a']
    df3['pred3b_acc'] = df3['y'] == df3['pred3b']
    df3['pred3a_acc'] = df3['pred3a_acc'].astype(int, copy = False)
    df3['pred3b_acc'] = df3['pred3b_acc'].astype(int, copy = False)
    
    return df3