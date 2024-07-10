# README #

[![Documentation Status](https://readthedocs.org/projects/tccon-qc-plots/badge/?version=latest)](https://tccon-qc-plots.readthedocs.io/en/latest/?badge=latest)

This code can be used to produce plots from TCCON netcdf files.

### Installation ###

Requires anaconda3 to be installed and **conda** console commands available.

The script **install.sh** will check if you have the **tccon-qc** environment installed. If not it will install it from the environment.yml file.

To run the install script use:

> ./install.sh

For other install options, see https://tccondata.org/qcdocs/index.html. Login is the standard TCCON partner login.

### Usage ###


#### How to run the code ####

Use **qc_plots.sh** to make sure the code runs with the correct environment.

If you use **qc_plots.py** make sure that you activated the **tccon-qc** environment with

```
conda activate tccon-qc
```

For usage info run:

```
./qc_plots.sh --help
```

Full online documentation is on [readthedocs](https://tccon-qc-plots.readthedocs.io/en/latest/)

### Who do I talk to? ###

sebastien.roche@mail.utoronto.ca or josh.laughner@jpl.nasa.gov
