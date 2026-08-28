# Default Samplers
ml-sim has built in distribution sampler functions for different sampling schenarios. The functions are often used as default distrubitions in sample and feature classes, but can be mixed and matched to meet your needs. 

## mvn(mu, var): multi-variate normal
A random normal sample given mean, mu, and variance, var.  
Usage: 
```
bias.Feature(dist = mvn, mu = [[4,2]])
```
returns a feature sampler with a mean of 4 and variance of 2.
## mean_only_mvn(mu): multi-variate normal
A random normal sample given mean, mu, only.

Usage: 
```
bias.Feature(dist = mean_only_mvn, mu = [[4]])
```
returns a feature sampler with a mean of 4

## shape_spread_only_mvn(x, cov): multi-variate normal
A random normal sample given size/mean, x, and covariance array, cov. The mean and size of the covariate matrix are both determined by the value x. This function is used as the default noise function.
Usage:
```
bias.Feature(dist = shape_spread_only_mvn, [[3, [.3, .6, .8]]])
```
returns a feature sampler with a mean of 3 and covarince matrix: [0.3,0.6,0.8]

## cat: categorical distribution
A random categorical distribution given the categories, and probablities of thoes categories. 

```
bias.Feature(dist=cat, mu=[(["a", "b", "c"], [0.2, 0.3, 0.5])])
```
returns a feature sampler with categories a, b, and c, and respective probablities, 0.2, 0.3, and 0.5.