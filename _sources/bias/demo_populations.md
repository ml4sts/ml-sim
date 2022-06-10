---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Using Population objects to create biased data

```{code-cell} ipython3
import mlsim
import pandas as pd
import numpy as np
import seaborn as sns
from collections import namedtuple
```

Create an all default population

```{code-cell} ipython3
pop = mlsim.bias.Population()
```

To view the details on this population, we can use the `get_parameter_description` method.

```{code-cell} ipython3
print(pop.get_parameter_description())
```

The instantiation just assigns values to these parameters. In order to get data, we use the `sample` method.  

```{code-cell} ipython3
help(pop.sample)
```

```{code-cell} ipython3
pop_df1 = pop.sample(100)
pop_df1.head()
```

## Changing the type of bias

Now demo some with various biases to create examples

```{code-cell} ipython3
# create a correlated demographic sampler
label_bias_dem = mlsim.bias.DemographicCorrelated(rho_a=.2,rho_z=[.25,.15])

# instantiate a population with that
pop_label_bias = mlsim.bias.PopulationInstantiated(demographic_sampler=label_bias_dem)
```

```{code-cell} ipython3
pop_label_bias_df1 = pop_label_bias.sample(100)
pop_label_bias_df1.head()
```

New we'll create a feature bias where the classes are separable for one group and not for the other. 

```{code-cell} ipython3
feature_sample_dist = lambda mu,cov :np.random.multivariate_normal(mu,cov)
per_group_means = [[[1,2,3,4,3,3],[4,6,8,8,10,6]],[[3,2,3,4,4,3],[1,3,4,4,5,3]]]
D =6
shared_cov = [np.eye(D)*.75,.95*np.eye(D)]
feature_bias = mlsim.bias.FeaturePerGroupSharedParamWithinGroup(
            feature_sample_dist,per_group_means,shared_cov)
pop_feature_bias = mlsim.bias.PopulationInstantiated(feature_sampler=feature_bias)
```

```{code-cell} ipython3
pop_feature_bias_df1 = pop_feature_bias.sample(100)
pop_feature_bias_df1.head()
```

```{code-cell} ipython3
var_list = ['x'+ str(i) for i in range(D)]
g = sns.pairplot(pop_feature_bias_df1, vars= var_list, hue = 'z')
```

```{code-cell} ipython3
[sns.pairplot(dffbai, vars= var_list, hue = 'z') for ai,dffbai in pop_feature_bias_df1.groupby('a')]
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```
