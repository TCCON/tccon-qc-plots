# README #

This code can be use to produce plots from TCCON netcdf files.

### Installation ###

Requires anaconda3 to be installed and **conda** console commands available.

The script **install.sh** will check if you have the **tccon-qc** environment installed. If not it will install it from the environment.yml file.

To run the install script use:

> ./install.sh

### Usage ###

#### inputs/variables.json ####

It is a json file with the main keys being the x-axis variables and values being lists of y-axis variables

The variables must have exactly the same names as in the netcdf files

The items in the list of variables can be a list of two variables, in that case the plot will be the first variable minus the second variable

Instead of the x-axis variable, a speciel key can be given to plot timeseries of statistics on resampled data

The format for this special key is *freq_stat* where freq is a pandas offset alias (https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases)

and *freq* is one of *mean*, *median*, or *std* for standard deviation. For example to make time series of 3-hourly mean, use *3H_mean*, for daily medians use *D_median*

#### How to run the code ####

Use **qc_plots.sh** to make sure the code runs with the correct environment.

If you use **qc_plots.py** make sure that you activated the **tccon-qc** environment with

> conda activate tccon-qc

For usage info run:

> ./qc_plots.sh --help

The **nc_file** argument is the full path to the input netCDF file

The optional **--flag0** argument can be used to only show data with flag=0 and limit the x-axis and y-axis ranges to the Vmin and Vmax of each variable

The optional **--ref** argument can be used to give a reference input file. If given, the data from that file will be plotted in the background in lightgray, and the data from the main file will be on top of those points.

The optional **--json** argument can be used to give the full path to a different json input file than the one already in the **inputs** folder

The optional **--cmap** argument can be used to change the colormap used in the plots, it must be one of the matplotlib named colormaps https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

### Who do I talk to? ###

sebastien.roche@mail.utoronto.ca