# README #

This code can be use to produce plots from TCCON netcdf files.

### Installation ###

Requires anaconda3 to be installed and **conda** console commands available.

The script **install.sh** will check if you have the **tccon-qc** environment installed. If not it will install it from the environment.yml file.

To run the install scriptr use:

> ./install.sh

### Usage ###

## variables.json ##

This is the only input file.

It is a json with the main keys being the x-axis variables and values being lists of y-axis variables

The variable must have exactly the same name as in the netcdf files

## How to run the code ##

Use qc_plots.sh to make sure the code runs with the correct environment.
If you use qc_plots.py make sure that you activated the tccon-qc environment with

> conda activate tccon-qc

For usage info run:

> ./qc_plots.sh --help

The **--ref** argument can be used to give a reference input file. If given, the data from that file will be plotted in the background in lightgray, and the data from the main file will be on top of those points.

### Who do I talk to? ###

sebastien.roche@mail.utoronto.ca