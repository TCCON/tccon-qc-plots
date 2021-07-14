# README #

This code can be used to produce plots from TCCON netcdf files.

### Installation ###

Requires anaconda3 to be installed and **conda** console commands available.

The script **install.sh** will check if you have the **tccon-qc** environment installed. If not it will install it from the environment.yml file.

To run the install script use:

> ./install.sh

### Usage ###

#### inputs/variables.json ####

It is a json file with the main keys being the x-axis variables and values being lists of y-axis variables

The variables must have exactly the same names as in the netcdf files

The items in the list of variables can be:

- a list of two variables e.g. **[a,b]**, in that case the plot will be the first variable minus the second variable
- a list containing a list of two variables e.g. **[[a,b]]**, in that case there will be two subplots with 1:3 size ratio, with variable **a** in the larger bottom plot

Instead of the x-axis variable, a special key can be given to plot timeseries of statistics on resampled data (also supports the **[a,b]**, and **[[a,b]]** notations)

The format for this special key is **freq_stat** where **freq** is a pandas [offset alias](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases) (also see [anchored offsets](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#anchored-offsets))

and **stat** is one of **mean**, **median**, or **std** for standard deviation. For example to make time series of 3-hourly mean, use **3H_mean**, for daily medians use **D_median**

A special key can also be given as the x-axis variable to plot rolling statistics, the format is **roll_stat** with stat defined like above.

Plots of rolling mean will also have +/- 1 rolling standard deviation.

#### inputs/limits.json ####

It is a json file with the main keys being variable names and values being lists of axis ranges (e.g. **"variable_name":[start,end]**)

If a variable is listed in this file with given limits, these will overwrite the default limits set by the code for that variable.

By default the **limits.json** file has one variable to give an example of the synthax: **"solzen":[0,90]**

This means that when solar zenith angle is plotted the solzen axis range will be from 0 to 90 instead of from the vmin and vmax values you set in the qc.dat file that it would use otherwise.

For difference plots you need to add **\_dif** to the variable name of one of the variables used for the difference (e.g. if you want to change the range limits for the tout-tmod differences to +- 4 degrees Celsius you can use either **"tout_dif":[-4,4]** or **"tmod_dif":[-4,4]**)

You can specify the full path to a custom file instead of the one in the **inputs** folder by using the **--json-limits** argument

Note the **limits.json** file is ignored if you run the code with the **--show-all** argument

#### How to run the code ####

Use **qc_plots.sh** to make sure the code runs with the correct environment.

If you use **qc_plots.py** make sure that you activated the **tccon-qc** environment with

> conda activate tccon-qc

For usage info run:

> ./qc_plots.sh --help

The **nc_in** argument is the full path to the input netCDF file or to an input folder that contains netCDF files

The optional **--flag0** argument can be used to only show data with flag=0 and limit the x-axis and y-axis ranges to the Vmin and Vmax of each variable

The optional **--ref** argument can be used to give a reference input file. If given, the data from that file will be plotted in the background in lightgray, and the data from the main file will be on top of those points.

The optional **--json** argument can be used to give the full path to a different json input file than the variables.json file already in the **inputs** folder

The optional **--json-limits** argument can be used to give the full path to a different json input file than the limits.json file already in the **inputs** folder

The optional **--cmap** argument can be used to change the colormap used in the plots, it must be one of the [matplotlib named colormaps](https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html)

The optional **--show-all** argument can be used so that plots don't make use of fixed axis ranges, when given all the data points will appear in the plots, even huge outliers.

The optional **--roll-gaps** argument can be used to set the Size of the rolling window in number of spectra, defaults to 500

The optional **--roll-gaps** argument can be used to set the minimum time interval for which the data will be split for rolling stats, defaults to "20000 days" () [supports all the pandas timdelta specifiers: https://pandas.pydata.org/pandas-docs/stable/user_guide/timedeltas.html]

The optional **--email** argument can be used to send the .pdf file of plots by email. The first argument is the sender email and the second argument is the recipient's email (comma separated for multiple recipients). Only tested for sender outlook accounts and gmail accounts that enabled less secured apps access.

### Who do I talk to? ###

sebastien.roche@mail.utoronto.ca
