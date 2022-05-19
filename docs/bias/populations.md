# Population Sampling for Bias Modeling

The basic entity for the bias modeling is to create a population object.

```
import mlsim
mlsim.Population()
```

The default is not a completely iid and balanced population. All populations are
defined by the following variables: $A$, $Z$, $Y$, $X$. A population has a `sample` method
and attributes for each component sampler. Populations also have samplers that
insert sampling, rather than population level biases.
