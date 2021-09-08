TCCON QC Configuration
======================

There are two configuration files that TCCON QC uses when making plots: the limits file, and the primary plots
file. The limits file configures what axis limits to use for different variables. The plots configuration file
controls which plots to make and how to style them.

Both files use `TOML format <https://toml.io/en/>`_. TOML has some similarities to the `INI format <https://en.wikipedia.org/wiki/INI_file>`_
often used for Linux configuration file, though TOML is more flexible.

Defaults for both files can be found in the ``inputs`` directory of the TCCON QC source code. You may edit these or
copy them and use the ``--config`` and ``--limits`` command line arguments to tell the QC plotting program where to
look for these.

Primary file
------------

The primary configuration file is broken down into two main parts: styles and plots.

Plots
*****

This section is the meat of the configuration, as it specifies which plots to make and in what order.
A very short example is::

    [[plots]]
    kind = "flag-analysis"

    [[plots]]
    kind = "timeseries"
    yvar = "dip"

    [[plots]]
    kind = "timeseries"
    yvar = "fvsi"

Each plot begins with ``[[plots]]``. Within each plot subsection, there are one or more key-value pairs such as
``kind = "timeseries"``. Every plot subsection *must* include the ``kind`` key, as this determines what type of plot
to make. Different plots have different sets of options, which will be covered in the :ref:`PlotTypes` section.

In this example, we have two types of plots: one "flag-analysis" and two "timeseries". The two timeseries plots
have different ``yvar`` values, so each will plot a time series of their specified variables.

Styles
******

The styles section allows you to specify details of how different data are plotted in the QC plots. This section is
the more complex, so let's look at an example right away::

    [style.default.scatter]
    all = {color = "black", marker = "o", markersize = 1}
    flag0 = {color = "black", marker = "o", markersize = 1}
    flagged = {color = "red", marker = "o", markersize = 1}

    [style.main.scatter]
    all = {color = "royalblue"}
    flag0 = {color = "royalblue"}
    flagged = {color = "red"}

    [style.ref.scatter]
    all = {color = "lightgray"}
    flag0 = {color = "lightgray"}

Each style subsection is defined by a single bracketed header with the format ``[style.<data type>.<plot kind>]``.
In the first subsection of the example, "default" is the data type and "scatter" the plot type. In the second
subsection, "main" is the data type and "scatter" is again the plot type. The four allowed data types are:

* ``main`` - styles defined for the "main" data type affect how the data in the file passed as a positional argument
  are plotted, that is the "main" focus of the plots.
* ``ref`` - styles defined for the "ref" data type affect reference data (i.e. the file passed to the ``--ref`` command
  line argument to be used as reference good quality data).
* ``context`` - styles defined for the "context" data type affect data in the file passed through the ``--context``
  command line argument, i.e. data from earlier in the record for the same site as the main data used to place the main
  data in the context of the overall record.
* ``default`` - style values defined for "default" provide a fallback for the other three.

The allows plot types are the allowed values for the ``kind`` option in the plots section, which are enumerated in
the :ref:`PlotTypes` section of this documentation.

Within each style subsection, how style options are organized depends on the specific plot type. Usually (but not
always), the keys within the subsection refer to specific subsets of data, and their values are dictionaries of
key-value pairs that affect the style used when plotting that subtype of data.

Let's walk through how the example shown above is interpreted by the QC plotting program. We will assume that we're
making a scatter plot, since that is the only plot type defined here. When the code goes to plot the main data, it
reads the ``[style.main.scatter]`` section. For a scatter plot, the default behavior is to plot good (``flag == 0``)
data and not good (``flag > 0``) data as two separate series. The style for ``flag == 0`` data is set by the
``flag0`` entry. To build the full style, the ``flag0`` options from *both* the ``[style.main.scatter]`` and
``[style.default.scatter]`` sections are combined, with ``main`` options taking precedence. In this example, the
``flag == 0`` style would be::

    {color = "royalblue", marker = "o", markersize = 1}

All three of these options were in the ``default`` section, but ``color`` was also defined in the ``main`` section, and
so the latter color takes precedence.  Likewise, the ``flag > 0`` style comes from the combination of the ``default``
and ``main`` section's ``flagged`` entries, and so is::

    {color = "red", marker = "o", markersize = 1}

(In this case, both sections specified the same color, so it didn't matter that ``main`` overrode the color value from
``default``.)

.. _StyleCloning:

Cloning styles
~~~~~~~~~~~~~~

Since many plot types are closely related, many plots offer the option to "clone" their style from another plot.
For example, in the default configuration::

    [style.default.scatter]
    all = {color = "black", marker = "o", markersize = 1}
    flag0 = {color = "black", marker = "o", markersize = 1}
    flagged = {color = "red", marker = "o", markersize = 1}

    [style.default.timeseries]
    clone = 'scatter'

By specifying ``clone = 'scatter'`` in the ``[style.default.timeseries]`` section, this means that all the styles
defined for ``[style.default.scatter]`` are replicated in ``[style.default.timeseries]``. In other words, the previous
example is identical to::

    [style.default.scatter]
    all = {color = "black", marker = "o", markersize = 1}
    flag0 = {color = "black", marker = "o", markersize = 1}
    flagged = {color = "red", marker = "o", markersize = 1}

    [style.default.timeseries]
    all = {color = "black", marker = "o", markersize = 1}
    flag0 = {color = "black", marker = "o", markersize = 1}
    flagged = {color = "red", marker = "o", markersize = 1}

The value that comes after the ``clone =`` key is the plot kind to clone from. You can only clone styles from the same
data type; that is, in this example, we could clone the default styles from scatter plots for the default styles in
timeseries plots, but we could *not* clone the **main** data styles from scatter plots for the **default** styles in
timeseries plots. Default to default, main to main, ref to ref, and context to context only.

.. note::
   Not all plot types support cloning styles. If they do not, this will be noted in :ref:`PlotTypes` below.


.. _PlotTypes:

Plot types
**********

The following table summarizes the available plots.

* The "Kind" column lists the string to give as the ``kind =`` value in the configuration file to create a plot of this type.
* "Required keys" lists other keys that must be present in that configuration section to create that kind of plot.
* "Optional keys" lists keys that may be provided to change the behavior of the given plot.
* "Style keywords" describes what keys may be passed in the style section for this plot type; using this is "MPL ``function`` kws", meaning any keywords for the Matplotlib function named can be given.
* "Cloning supported" indicates whether that plot type allows :ref:`style cloning <StyleCloning>`


.. csv-table::
   :file: plot_types.csv
   :widths: 20, 20, 20, 30, 10
   :header-rows: 1


Common optional keys
~~~~~~~~~~~~~~~~~~~~

All plot types accept the following as optional keys:

* ``key`` (default = ``None``): a string used to refer to this plot from another plot. If not given, this plot
  cannot be referenced from another plot.

.. warning::
   There is currently no check to protect against two plots having the same key. If you get odd results when
   trying to refer to another plot, make sure you don't have duplicated plot keys!

* ``width`` (default = ``20``): initial width of the plot in centimeters
* ``height`` (default = ``10``): initial height of the plot in centimeters

.. note::
   This does not guarantee the final page size will be 20 x 10 cm. Excess whitespace is trimmed from the plots
   and the final page size depends on the ``--size`` command line argument.

flag-analysis
~~~~~~~~~~~~~

A flag analysis plot shows bar graphs of the number of spectra and percent of spectra marked as bad by different
variables during the automatic QC process in TCCON post processing.

**Required keys**

None

**Optional keys**

* ``min_percent`` (default = ``1.0``): the minimum percent of spectra a variable must flag for it to be shown on the plot.

**Style**

A flag-analysis style subsection must have the ``all`` key, this is the only one used. Keywords can be any valid
keywords for :func:`matplotlib.pyplot.bar`. In addition, ``legend_fontsize`` (default 7) adjusts the size of the
text in the legend.

.. _PT_timing_error_am_pm:

timing-error-am-pm
~~~~~~~~~~~~~~~~~~

A plot that shows a time series of resampled values for a specific range of SZA values in the morning and afternoon.
This is an experimental plot type to try to detect timing errors from differences in the morning and afternoon values.

.. note::
   This plot uses all data from the main and context files unless the ``--flag0`` command line flag was given.
   ``flag == 0`` and ``flag > 0`` data is not plotted separately.

**Required keys**

* ``sza_range``: a 2-element list giving the range of SZA values (in degrees) to average the ``yvar`` in. Example: ``[70, 80]``

**Optional keys**

* ``yvar`` (default = ``"xluft"``): the variable from the netCDF file to plot on the y-axis.
* ``freq`` (default = ``"W"``): the temporal frequency to bin the data to. Any `Pandas frequency interval <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_ is supported
* ``op`` (default = ``"median"``): what operation to use in the binning, usually "median" or "mean", but any operation supported on a Pandas resampled data frame is supported.
* ``time_buffer_days`` (default = ``2``): number of days to buffer the edges of the plot by to ensure the first and last points do not end up on the plot edge.

**Style**

A style subsection for one of these plots may have any or all of the keys ``both``, ``am``, or ``pm``. These provide
style keywords that apply to the series for the morning data (``am``), afternoon data (``pm``) or both (``both``).
The keywords given can be any style keywords accepted by :func:`matplotlib.pyplot.plot`.

The `label` keyword is treated specially. In Matplotlib, this keyword is used to set the legend text for a given
data series. The QC plots will include a default label if you do not specify one. If you do specify one, it is
passed through a format call where three keyword values are available:

* ``data`` will be replaced with a short description of the data (site name and whether flag == 0, flag > 0, etc)
* ``ll`` and ``ul`` will be replace with the lower and upper SZA limits, respectively.

timing-error-szas
~~~~~~~~~~~~~~~~~

A plot that shows a time series of resampled values for multiple SZA ranges in the morning or afternoon.
This is an experimental plot type to detect timing errors from differences in the typical value at different
SZAs.

.. note::
   This plot uses all data from the main and context files unless the ``--flag0`` command line flag was given.
   ``flag == 0`` and ``flag > 0`` data is not plotted separately.

**Required keys**

* ``sza_ranges``: a list of 2-element lists specifying which SZA ranges to plot. Example: ``[[70,80], [40,50], [20,30]]``.
* ``am_or_pm``: one of the strings "am" or "pm", indicating that the plot should use morning ("am") or afternoon ("pm") data.

**Optional keys**

Identical to those for :ref:`timing-error-am-pm <PT_timing_error_am_pm>` plots.

**Style**

Because these plots have an arbitrary number of data series (one per SZA range) rather than specific data
categories, their style definitions follow a different pattern from other plots. Valid keywords are those accepted
by :func:`matplotlib.pyplot.plot`, but they are not grouped by data subset. These keywords are specified directly
within a ``[style.<data type>.timing-error-szas]`` section, as::

    [style.default.timing-error-szas]
    marker = "o"
    markersize = 1
    linestyle = "none"
    color = ["tab:blue", "tab:orange", "tab:green"]

The value for each key may be *either* a scalar value (as in ``marker``, ``markersize``, and ``linestyle`` above) *or*
a list of values (as with ``color``). If a scalar value is provided, that value is used for all data series representing
different data ranges. If a list is provided, then the plot cycles through the values for the different SZA ranges.

.. note::
   If the list has fewer values than there are SZA ranges, then the plot cycles back through the values as many
   times as needed. If you are getting identical styles for two data series, make sure your lists are long enough.

Similar to :ref:`timing-error-am-pm <PT_timing_error_am_pm>`, if a value for ``label`` is provided, then that string
is formatted with the ``data``, ``ll``, and ``ul`` keywords. If ``label`` is not provided, a default is used.
See above for their meanings. Like the other options in this plot's styles, ``label`` may be a single string or a
list of strings.

.. _PT_scatter:

scatter
~~~~~~~

A plot of one variable versus another.

**Required keys**

* ``xvar``: the name of the variable in the netCDF files to plot on the x-axis
* ``yvar``: the name of the variable in the netCDF files to plot on the y-axis

**Optional keys**

* ``match_axes_size`` (default = ``None``): if given, this must be a valid hex to a "hexbin" plot. The scatter plot's axes will be compressed to match the width of the hexbin, allowing for colorbars.

**Style**

A scatter plot's style subsection may have the keys ``all``, ``flag0``, or ``flagged``. These provide the style
keyword arguments for plotting all data, ``flag == 0`` data, and ``flag > 0`` data, respectively. Allowed keywords
are those for :func:`matplotlib.pyplot.plot`. If ``linestyle`` is not provided, it defaults to "none".

.. note::
   Do not use the ``ls`` shorthand for ``linestyle``, since ``linestyle`` is always set.

A default label is provided that include the site name and what subset of data (``flag == 0``, ``flag > 0``, etc) a
series refers to. If you provide a custom label, this string can be inserted by including ``{data}`` in your string.

hexbin
~~~~~~

A plot of one variable versus another similar to a scatter plot, except it plots a 2D histogram rather than individual
points.

.. note::
   This does not plot ``flag == 0`` and ``flag > 0`` data separately. If the ``--flag0`` command line flag is present,
   only ``flag == 0`` data is used, otherwise all data is used.

**Required keys**

* ``xvar``: the name of the variable in the netCDF files to plot on the x-axis
* ``yvar``: the name of the variable in the netCDF files to plot on the y-axis

**Optional keys**

* ``show_reference`` (default = ``false``): Set to ``true`` to plot the reference data (if provided) as a second
  2D histogram.
* ``show_context`` (default = ``false``): Set to ``true`` to plot the context data (if provided) as a second 2D
  histogram.

**Style**

A hexbin's style subsection may have the keys ``all`` and ``flag0``, used when plotting all data or ``flag == 0`` data,
respectively. This accepts all style keywords allowed by :func:`matplotlib.pyplot.hexbin`. Note that ``extent`` is
provided a reasonable default and usually does not need specified.

There are two special keywords in addition to the standard :func:`matplotlib.pyplot.hexbin` keywords:

* ``fit_style`` takes as value another dictionary of style keywords valid for :func:`matplotlib.pyplot.plot` to use
  when plotting the linear fit through the hexbin data. If ``label`` is included in these keywords, the first
  ``{}`` in it will be replaced with the linear fit information.
* ``legend_fontsize`` sets the fontsize of the legend. 7 pts is the default, and usually keeps the linear fit
  within the plot bounds.

timeseries
~~~~~~~~~~

A plot of a given variable vs. time.

**Required keys**

* ``yvar``: the variable from the netCDF file(s) to plot on the y-axis

**Optional keys**

* ``time_buffer_days`` (default = ``2``): number of days to buffer the edges of the plot by to ensure the first and last points do not end up on the plot edge.

**Style**

Style configuration is identical to that for :ref:`scatter plots <PT_scatter>`.

timeseries-2panel
~~~~~~~~~~~~~~~~~

A plot of two variables vs. time, with the second in a smaller upper panel. Typically used for a retrieved variable
and its error.

**Required keys**

* ``yvar``: the variable from the netCDF file(s) to plot on the y-axis for the main axes
* ``yerror_var``: that variable from the netCDF file(s) to plot on the y-axis for the smaller upper axes.

**Optional keys**

* ``time_buffer_days`` (default = ``2``): number of days to buffer the edges of the plot by to ensure the first and last points do not end up on the plot edge.

**Style**

Style configuration is identical to that for :ref:`scatter plots <PT_scatter>`. Both panels will use the same style for
the same data subset.

resampled-timeseries
~~~~~~~~~~~~~~~~~~~~

Similar to "timeseries" plots, except that the data is broken down into chunks of a specified length of time and
summarized as a mean/median/etc.

**Required keys**

* ``yvar``: the variable from the netCDF file(s) to plot on the y-axis
* ``freq``: the temporal frequency to bin the data to. Any `Pandas frequency interval <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_ is supported
* ``op``: what operation to use in the binning, usually "median" or "mean", but any operation supported on a Pandas resampled data frame is supported.

**Optional keys**

* ``time_buffer_days`` (default = ``2``): number of days to buffer the edges of the plot by to ensure the first and last points do not end up on the plot edge.

**Style**

Style configuration is identical to that for :ref:`scatter plots <PT_scatter>`.

rolling-timeseries
~~~~~~~~~~~~~~~~~~

Similar to "timeseries" plots, but in addition to plotting the raw data, running mean/median/etc. series are
overplotted.

**Required keys**

* ``yvar``: the variable from the netCDF file(s) to plot on the y-axis
* ``ops``: what operation(s) to use for the rolling, usually "median" or "mean", but any operation supported on a Pandas
  rolling data frame is supported. This can be either a string for a single operation, or a list of strings to plot
  multiple rolled series. A special case is the "quantile" operation, this must include the quantile value to calculate,
  e.g. "quantile0.75" to compute the quantile with ``q = 0.75``.

**Optional keys**

* ``gap`` (default = ``"20000 days"``): this specified a gap in time that the rolling operation will not cross. This can
  be any string recognized by `Pandas timedelta <https://pandas.pydata.org/pandas-docs/stable/user_guide/timedeltas.html>`_.
  If there is a gap in the data longer than this duration, the data on either side will have the rolling operation
  applied separately. The default of "20000 days" (~50 years) is set to effectively disable this behavior by default.
* ``rolling_window`` (default = ``500``): the number of points to use in the rolling window.
* ``uncertainty`` (default = ``false``): set this to ``true`` to plot uncertainty ranges for mean or median operations;
  means will use 1-sigma standard deviation and medians the upper and lower quartiles.
* ``data_category`` (default = ``None``): which subset of the ``yvar`` data to use, both when plotting the raw data and
  when computing the rolling operation(s). The default behavior is to use the normal subset for a given data type, or
  ``flag == 0`` data if the ``--flag0`` command line argument is set. Passing one of the strings "all", "flag0", or
  "flagged" will force the use of that subset (this may result in errors if one of the data files does not have the
  "flag" variable, which is required to figure out the latter two subsets).

**Style**

Style configuration is similar to that for :ref:`scatter plots <PT_scatter>`, in that the keys within a
``[style.<data type>.rolling-timeseries]`` section can be the data subsets (``all``, ``flag0``, ``flagged``),
each of which has a dictionary of style arguments as its value. However, the rolling operations can each
have their own style, as additional subsection keys (e.g. ``mean``, ``median``, etc.). Quantile operations
will prefer to use a style for the specific quantile being calculated (if one is available) but will fall
back on a provided generic ``quantile`` style if not.

.. note::
   The fallback to a generic ``quantile`` style is done on a per-data type basis. That is, if your "main"
   data type section has both a ``quantile`` and ``quantile0.75`` style and your "default" section has only
   a ``quantile`` section, then when using the "quantile0.75" operation, the final style will use the
   "main" section's ``quantile0.75`` style plus the default section's ``quantile`` style. The "main" section's
   ``quantile`` style is entirely ignored.

Like scatter plots, if you provide a ``label`` as one of the style keywords, it will be passed through a
``format`` call. The ``{data}`` substring will still be replaced by the description of the data (site name
+ data subset). In addition, the ``{op}`` substring will be replaced with the rolling operation.

.. note::
   If you use ``{op}`` in a label for regular data (e.g. ``all``, ``flag0``, ``flagged``),
   it will get replaced by the string "None".

If you provide styles for ``std`` and ``quantile``, those styles will be used if plotting uncertainty
ranges for mean and median operations, respectively.

If the final style (composed from data-specific + default styles) does not include a linestyle, then
the linestyle value is set to "none", as for scatter plots. Avoid using the "ls" shorthand for "linestyle"
since "linestyle" will always be set if absent.

Limits file
-----------

Basic format
************

The limits file is broken down into sections that specify limits for different kinds of plots. Default values for
each variable can also be specified. An example of a simple limits file is::

    [default]
    xluft = [0.975, 1.025]
    xch4 = [1.6, 2.0]
    xch4_error = [0, 0.05]

    [scatter]
    xluft = [0.996, 1.002]

Each section starts with a value in brackets. The ``[default]`` section in this example specifies the default limits
for three variables: xluft, xch4, and xch4_error. Note that each set of limits is given as a list, also in square
brackets.

.. note::
   Make sure the limits have the lower value first! The TCCON QC code makes no guarantees about how the plots
   will behave if the limits are reversed.

In this example, we have a second section, ``[scatter]`` which specifies limits for xluft. This means that any scatter
plots will use the tighter limits specified in this second section, while all other plots will use the looser limits
given in ``[default]``.

The allowed section names other than ``[default]`` are the same as the allowed values for the ``kind`` argument in the
primary configuration.

Wildcards
*********

The limits file also supports limited wildcards in the variable names, so that a limit can match for all variables
whose names follow a certain pattern. The allowed wildcards are:

* ``*`` - matches 0 or more characters (i.e. anything)
* ``?`` - matches any single character
* ``[seq]`` - matches any character in "seq"
* ``[!seq]`` - matches any character not in "seq"

Consider this example::

    [default]
    "*vsf_hcl*" = [0.7, 1.4]
    "vsf_*" = [0.9, 1.1]
    "*_fs" = [-2, 2]

The first entry will match *any* variable that includes the substring "vsf_hcl" anywhere, because the two ``*`` can
match anything (including nothing). The second entry will only match variables that begin with "vsf\_", while the third
will only match variables that end in "_fs".

.. note::
   In this example, the strings on the left side of the equals sign are quoted, when they weren't in the non-wildcard
   example. Whenever using special characters like ``*``, it's best to quote the string to ensure TOML interprets it
   as a string.

Precedence
**********

With wildcards, it is quite easy to have a variable match multiple entries in your limits file. TCCON QC uses three
rules to determine which limit to use:

#. A plot specific section takes precedence over the ``[default]`` section
#. Use the first entry in a section that matches the variable
#. If no entry matches that variable, use the ``vmin`` and ``vmax`` attributes for that variable from the netCDF file(s) being plotted.

