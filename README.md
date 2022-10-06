# ML-Sim:

Synthetic Data generators for ML  evaluation

## Getting Started




To use the package, download (or clone) and:

```
cd ml-sim/
pip install .
```

To use the package, after installed::

```
import mlsim
```



## Development

To work in a separate development environment use the `requirements.txt` to install dependencies


### To reinstall  package after changes

```
pip install --upgrade .
```

Or use
```
pip intall -e .
```
When updating the package and working in a notebook, the notebook's kernel will
need to be restarted to get the updates, if they're done outside of the notebook.

(only needed in development or after upgrade)



## Offline Documentation
To compile docs, jupyter book and some sphinx extensions are required, install
them with

```
pip install -r requirements.txt
```

then

```
jupyter-book build docs/
```

to build the documentation offline.
