# Population Sampling for Bias Modeling

The basic entity for the bias modeling is to create a population object.

```
import mlsim

pop = mlsim.bias.Population()
```

The default is not a completely iid and balanced population. All populations are
defined by the following variables: $A$, $Z$, $Y$, $X$. A population has a `sample` method
and attributes for each component sampler.


## Sampling bias

Populations also have samplers that
insert sampling, rather than population level biases. This allows for the creation of a population with one set of biases and to use the same object to draw additional datasets that have additionally biased sampls.  For example you may wish to have training data and audit datasets that have different disributions to demonstrate the impact of a biased sampling at one of those times. 
