# README #

[![Documentation Status](https://readthedocs.org/projects/tccon-qc-plots/badge/?version=latest)](https://tccon-qc-plots.readthedocs.io/en/latest/?badge=latest)

This code can be used to produce plots from TCCON netcdf files.

### Installation ###

Requires [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) 
or [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) to be available.

The script **install-conda.sh** will use `conda` and check if you have the **tccon-qc** environment installed. If not it will install it from the environment.yml file.
If you already have that environment, it will update from the same file.

To run the install script use:

```
./install-conda.sh
```

If you prefer `micromamba`, use the `install-micromamba.sh` script instead.

For other install options, see https://tccon-qc-plots.readthedocs.io/en/latest/pages/installation.html.

### Usage ###


#### How to run the code ####

Use **run-qc-plots** to make sure the code runs with the correct environment.

For usage info run:

```
./run-qc-plots --help
```

Full online documentation is on [readthedocs](https://tccon-qc-plots.readthedocs.io/en/latest/)

### Who do I talk to? ###

sebastien.roche@mail.utoronto.ca or josh.laughner@jpl.nasa.gov
