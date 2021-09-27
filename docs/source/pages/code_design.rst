QC plots code design
====================

The :py:mod:`qc_plots2` module follows an object-oriented approach to the data and plots. Datasets are represented
by :py:class:`qc_plots2.TcconData` instances. These instances wrap a netCDF dataset and provide several methods to 
access the data - ``get_data`` and ``get_flag0_or_all_data``. The former will return a certain category of data 
(all, flag = 0, or flag > 0 data currently). Using these intermediate methods allows the instance to impose some
additional constraints on the data returned; for example, reference data is configured to limit what it returns 
to the same time period as the main (and context, if available) data. The second method provides the "default"
data, which is flag = 0 data if that instance knows how to find flag = 0 data, otherwise it is all data.

Plots are represented by subclasses of the :py:class:`~qc_plots.qc_plots2.AbstractPlot` class. This abstract 
class defines the framework of methods any plot kind must implement. Each plot kind allowed in the configuration 
file has a corresponding class in :py:mod:`~qc_plots.qc_plots2`. The figure below shows the inheritance graph 
for the various plot classes. When one plot needs only to add or adjust functionality of another, it can inherit 
from another concrete plot class. For example, several plot kinds inherit from :py:class:`~qc_plots.qc_plots2.TimeseriesPlot`,
as they are fundamentally time series, but with additional functionality.

.. inheritance-diagram:: qc_plots.qc_plots2.ResampledTimeseriesPlot qc_plots.qc_plots2.RollingDerivativePlot qc_plots.qc_plots2.RollingTimeseriesPlot qc_plots.qc_plots2.Timeseries2PanelPlot qc_plots.qc_plots2.TimeseriesDeltaPlot qc_plots.qc_plots2.HexbinPlot qc_plots.qc_plots2.TimingErrorMultipleSZAs qc_plots.qc_plots2.TimingErrorAMvsPM qc_plots.qc_plots2.FlagAnalysisPlot
   :top-classes: qc_plots.qc_plots2.AbstractPlot, qc_plots.qc_plots2.TimeseriesMixin
   :parts: 1
   :caption: Inheritance diagram for the various plot classes. Note that ``ABC`` is the abstract base class,
     not actually a plot class.


There is a very close correspondance between the ``__init__`` method of the plot classes and the plot configuration 
file. Except for the ``self``, ``other_plots``, ``default_style``, and ``limits`` keyword, all the arguments in a 
plot class's ``__init__`` method correspond to valid configuration options in the plot configuration file. Arguments
without a default value are required in the configuration file, others are optional. 

.. note::
   The :py:meth:`~qc_plots.qc_plots2.AbstractPlot.from_config_section` method creates plot class instances from 
   sections in the configuration file.


Plot instance methods
---------------------

The idea behind separating each plot type into its own class is that this keeps code for each plot class grouped
together, but encourages reuse of code by inheriting from parent classes. In this section, we'll go over the general
structure of a plot class and cover the main methods that you would likely customize one or more of in new plot 
classes. Below is a figure showing the relationship among these methods.

.. graphviz::
   :caption: Call structure for the key methods in a plot class.

   digraph {
       "make_plots" -> "setup_figure";
       "make_plots" -> "_plot";
       "make_plots" -> "get_save_name";
       "_plot" -> "get_plot_args";
       "get_plot_args" -> "get_plot_data";
       "get_plot_args" -> "get_plot_kws";
       "get_plot_args" -> "get_legend_kws";
   }

In any plot class, the ``make_plots`` method is the main entry point. This will be called by the main script with 
relevant command line values (like where to save the figure to) as well as a list of all available data, as 
:py:class:`~qc_plots.qc_plots2.TcconData` instances. It must ultimately save a figure as an image (usually PNG format)
and return a path to that image.

``make_plots`` usually first calls ``setup_figure``. This method is responsible for creating the figure and axes to 
plot into, as well as setting limits, axes labels, tick formatting, etc. It must return the handle to the figure and 
axes (or array of axes handles, if >1 axes needed).

``_plot`` handles the actual plotting. It is called with each :py:class:`~qc_plots.qc_plots2.TcconData` instance in turn;
if one or more of those instances should be skipped, ``make_plots`` should skip it. (For example, the 
:py:class:`~qc_plots.qc_plots2.FlagAnalysisPlot` only plots the main data, so ``make_plots`` handles skipping over the 
reference and context data, if they are present.)

Within ``_plot``, usually the first step is to call ``get_plot_args``, which returns a list of arguments for separate
plotting calls. Each entry in the list represents a different subset of the data to plot. For example, if plotting 
flag = 0 and flag > 0 data separately, then ``get_plot_args`` would return a two-element list. Each element of the 
list is *usually* a dictionary, often with keys "data", "kws", and "legend_kws". "data" points to the data that will 
be plotted, "kws" would have the style keywords to pass to the plotting function, and "legend_kws" would have keywords 
to use when creating the legend. 

.. note::
   Often "legend_kws" values are only used from the first element in a list of plot arguments. 

``get_plot_args`` relies on three other functions to get the different parts of the arguments. ``get_plot_data`` should,
given a :py:class:`~qc_plots.qc_plots2.TcconData` instance and a flag category (e.g. all, flag = 0, or flag > 0 data), 
return the subset of data corresponding to the given flag category to use in the plot for that. Any preprocessing of the 
data (like calculating rolling medians) usually happens in ``get_plot_data``. Likewise, ``get_plot_kws`` returns the style
keywords to use for the given flag category. This is also where fallback styles are usually set; for instance, 
:py:class:`~qc_plots.qc_plots2.ScatterPlot` will set the linestyle to "none" and marker to "." if those weren't set by 
any of the configured styles, since these are needed to make a scatter plot, rather than a line plot. Finally,
``get_legend_kws`` returns style keywords for the legend, merging legend keywords from the style with plot-specific keyword.

Back in ``_plot``, once it has received the list of plotting arguments, it loops through them and plots each subset 
of data into the axes provided to it. Once the plotting is complete, ``make_plots`` will need to save the plot as a PNG
and return the path to that image.

Getting style keywords
**********************

Getting the final set of style keywords for a given plot can be complicated, since it needs to merge default styles for 
a given plot type with data type-specific styles. When the plot configuration file is read, default styles are stored in 
each concrete plot instance. The plot instances each store styles for *all* plot kinds, in order to allow for styles to 
be cloned from one plot kind to another. Likewise, each :py:class:`~qc_plots.qc_plots2.TcconData` instance stores the 
dictionary of plot styles relevant for its data category (main, context, reference).

Recall from the :ref:`Styles` section of the configuration documentation that styles are organized by 
``data category -> plot kind -> data subset``. Generally the algorithm that ``get_plot_kws`` follows is:

#. Get the default styles for the current plot kind from the :py:class:`~qc_plots.qc_plots2.AbstractPlot` 
   subclass instance. 
#. If the default styles have a style for the current data subset, get that style. If not, but do have a 
   "clone" directive, get the styles for the cloned plot kind and check if those have the right data subset.
   If not, return an empty dict.
#. Get the data category-specific styles from the :py:class:`~qc_plots.qc_plots2.TcconData` instance. Follow 
   the same approach as for the default styles to get the subset-specific style.
#. Merge the default and data category-specific options, with the specific options overriding the default options 
   wherever there is a conflict. 

Implementations of ``get_plot_kws`` usually rely heavily on the helper ``_get_style`` method. This method handles getting 
the correct style for the data subset, including looking it up from the cloned plot type if needed. 


Getting legend keywords
***********************

Legend keywords can be specified differently for different instances of the same plot kind. To keep things simple, 
legend keywords defined in the style section of the configuration file, rather than the plots section, can only be 
defined in the "default" subsection, not any of the data category-specific subsections. When a plot instance is created,
any plot-specific legend keywords are stored on that instance. The ``get_legend_kws`` method then only has to retrieve the 
default style legend keywords and merge them with the instance-specific keywords. It also relies on the ``_get_style``
method to retrieve the default legend style, in order to allow legend keywords to be cloned just like the plot styles 
themselves. 