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

The primary configuration file is broken down into several main parts: variables, image postprocessing, styles, and plots.

.. _Variables:

Variables
*********

The ``[variables]`` section of the configuration allows you to define strings to replace with other values elsewhere
in the configuration. For example, if my configuration file included::

  [variables]
  static_ref_file = "/data/tccon/static/standard_network_data.nc"

then I could have a plots subsection like::

  [[plots]]
  kind = "timeseries-2panel+violin"
  name = "XCO2 timeseries"
  yvar = "xco2"
  yerror_var = "xco2_error"
  violin_data_file = "$static_ref_file"

This will see ``"$static_ref_file"`` in the ``[[plots]]`` section replaced with "/data/tccon/static/standard_network_data.nc".
In other words, QC plots interprets the plotting section as::

  [[plots]]
  kind = "timeseries-2panel+violin"
  name = "XCO2 timeseries"
  yvar = "xco2"
  yerror_var = "xco2_error"
  violin_data_file = "/data/tccon/static/standard_network_data.nc"


Substitutions obey the following rules:

* Variables must be references with a leading dollar sign. Variable names may contain letters, numbers, and/or underscores,
  but may not start with a number. 
* If the variable makes up the entire value, as in the above example, type is preserved. That is, if you have a configuration
  file such as::

    [variables]
    custom_width = 16
    custom_height = 8

    [[plots]]
    kind = "flag-analysis"
    width = "$custom_width"
    height = "$custom_height"

  then ``width`` and ``height`` would be integers. This is important as some of the inputs are expected to be actual numeric or
  boolean types.

* If the variable is inserted in the middle of a longer string, it is expanded using Bash-like rules: a variable begins with a 
  dollar sign and ends with the first non-alphanumeric or underscore character, or curly braces can be used to disambiguate where 
  the variable ends if it must abut an alphanumeric or underscore character. For example, you could do this::

    [variables]
    ref_path = "/data/tccon/static"
    ref_site = "oc"

    [[plots]]
    kind = "timeseries+violin"
    yvar = "xluft"
    violin_data_file = "$ref_path/${site}_static.nc"

  In this example, ``$ref_path`` does not need curly braces because the next character (``/``) cannot be part of a variable name.
  ``$site`` *does* need curly braces; it this were written as ``$site_static.nc``, it would look for a variable named ``site_static``.
  Any case where the value on the right hand side is not exactly one variable name, with no extra character, will always result in a 
  string. That is, something like::

    [variables]
    custom_width = 16

    [[plots]]
    ...
    width = "$width "  # note the extra space at the end!

  will cause ``width`` for this plot to be a string, not an integer.

.. note::
   When writing references to variables, they must be quoted as shown above. TOML syntax does not allow dollar signs in bare strings.

.. _ImagePostProc:

Image postprocessing
********************

This section contains options that pertain to final conversion of the individual figures into combined PDFs.
It begins with the ``[image_postprocessing]`` header. All options are optional (with one caveat regarding
``font_file``). Options are:

* ``disable_info`` (default = ``false``): set this to ``true`` to skip writing plot information (plot number, name,
  and source file) in the upper left corner of each plot. Since writing that information requires that ``font_file``
  points to a valid TrueType font file, setting ``disable_info = true`` is a workaround if you cannot find a TrueType 
  font on your computer.

* ``font_file`` (default = "LiberationSans-Regular.ttf"): TrueType font file to use when writing the plot 
  number, name, and input file in the top left of each page. This is done using Python's 
  `Pillow library <https://pillow.readthedocs.io/en/stable/>`_ for image manipulation. Pillow searches common 
  directories for TrueType files, so in many cases you only need give the file name, and not a full path. 

.. note::
   If :file:`LiberationSans-Regular.ttf` is not available on your system, you will need to change this option 
   to a valid TrueType font file, or set ``disable_info`` to ``true``. Otherwise the QC plots program will crash 
   when it reaches this part. On Linux, fonts can usually be found under :file:`/usr/share/fonts`.

* ``font_size`` (default = 30): size of the font used to write the plot number, name, etc. in the upper left 
  corner of each page. 

* ``bookmark_all`` (default = ``None``): this controls whether each page in the output PDF automatically receives
  a bookmark. Setting this to ``true`` or ``false`` will turn that behavior on or off, respectively.  However, the 
  default when this is not specified is to check whether any of the plots have a value for their individual ``bookmark``
  properties. If not, QC plots behaves as if ``bookmark_all`` is ``true`` (and so makes a bookmark for every plot);
  if so, then QC plots behaves as if ``bookmark_all`` is ``false`` (only making bookmarks for plots that have their
  individual ``bookmark`` properties set).

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

.. _Styles:

Styles
******

The styles section allows you to specify details of how different data are plotted in the QC plots. This section is
the more complex, so let's look at an example right away::

    [style.default.scatter]
    all = {color = "black", marker = "o", markersize = 1}
    flag0 = {color = "black", marker = "o", markersize = 1}
    flagged = {color = "red", marker = "o", markersize = 1}
    legend_kws = {ncol = 2}

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

All plot types are permitted to include ``legend_kws`` as a key within the "default" subsection, as you see in this
example. This can point to a dictionary of keywords to pass to the
`matplotlib legend function <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html>`_. Unlike the
other components of styles, the legend keywords can be overridden on individual plots using the ``legend_kws`` key
in a ``[[plots]]`` subsection of the TOML file.

.. warning::
   ``legend_kws`` is only read from the "default" subsection. If you put it in "main", "ref", or "context", it will be
   ignored.

.. note::
   The `legend documentation <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html>`_ makes a
   distinction between when ``legend`` is called on a figure vs. axes. Currently, all plot types in the TCCON QC
   program call ``legend`` on axes.

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
    clone = "scatter"

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

You can override specific keys within a subsection after cloning. For example::

    [style.default.timeseries]
    clone = "scatter"
    legend_kws = {ncol = 2}

would clone the ``all``, ``flag0``, and ``flagged`` values from ``[style.default.scatter]`` (from the first
example in this section) but use ``{ncol = 2}`` for the ``legend_kws`` value.

.. _PlotTypes:

Plot types
**********

The following table summarizes the available plots.

* The "Kind" column lists the string to give as the ``kind =`` value in the configuration file to create a plot of this type.
* "Required keys" lists other keys that must be present in that configuration section to create that kind of plot.
* "Optional keys" lists keys that may be provided to change the behavior of the given plot.
* "Style keywords" describes what keys may be passed in the style section for this plot type; using this is "MPL ``function`` kws", meaning any keywords for the Matplotlib function named can be given.
* "Cloning supported" indicates whether that plot type allows :ref:`style cloning <StyleCloning>`
* "Aux plots" lists auxiliary plots that can be added to that main style plot. 


.. csv-table::
   :file: plot_types.csv
   :widths: 20, 20, 20, 30, 10, 10
   :header-rows: 1


Common optional keys
~~~~~~~~~~~~~~~~~~~~

All plot types accept the following as optional keys:

* ``key`` (default = ``None``): a string used to refer to this plot from another plot. If not given, this plot
  cannot be referenced from another plot.

.. warning::
   There is currently no check to protect against two plots having the same key. If you get odd results when
   trying to refer to another plot, make sure you don't have duplicated plot keys!

* ``name`` (default = ``None``): a name to use for the plot alongside the plot number in the upper left corner
  of each page. If this is not given, then the filename used to save the intermediate plot images is inserted 
  instead. 
* ``bookmark`` (default = ``None``): controls whether and how this page gets bookmarked in the output PDF. Assigning
  a string as this property will use that name for the bookmark in the PDF (e.g. setting ``bookmark = "Flags"`` on 
  a plot will cause that page in the final PDF to have the bookmark "Flags"). Setting this to ``true`` will use 
  the value of ``name`` for the bookmark (either the value passed as ``name`` explicitly or the fallback file name).
  If the :ref:`ImagePostProc` key ``bookmark_all`` is ``true`, then all plots have a bookmark in the final PDF. In 
  that case, the value of ``bookmark`` is used if available, then QC plots falls back on ``name``.
* ``legend_kws`` (default = ``{}``): keyword to pass to the `legend <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html>`_
  call for this plot only. Will be merged with legend keywords defined in the default style for this plot type.
* ``extra_qc_lines`` (default = ``[]``): a list of dictionaries specifying extra horizontal or vertical lines to 
  plot as a guide for whether data is in family or not. An example::

    extra_qc_lines = [{value = 0.996, axis="y", linestyle = "--", color = "darkorange", label="Expected range"},
                      {value = 1.002, axis="y", linestyle = "--", color = "darkorange"}]

  Each dictionary *must* have the key ``value``, which gives the position of the line. The ``axis`` key is optional;
  it specifies on which axis the lines are positioned on ("y" = horizontal lines, "x" = vertical) and defaults to "y"
  if absent. Any other key-value pairs must be valid keyword arguments to :py:func:`~matplotlib.pyplot.axvline` or 
  :py:func:`~matplotlib.pyplot.axhline`.
* ``width`` (default = ``20``): initial width of the plot in centimeters
* ``height`` (default = ``10``): initial height of the plot in centimeters

.. note::
   This does not guarantee the final page size will be 20 x 10 cm. Excess whitespace is trimmed from the plots
   and the final page size depends on the ``--size`` command line argument.

If a plot has an auxiliary plot added, it may have additional required or optional keys beyond those described in 
this section (or the plot-specific sections below). See :ref:`AuxPlots` for information on which keys are added by 
which auxiliary plots.

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

nan-check
~~~~~~~~~

A plot that displays the number of percentage of data that is a NaN or fill value in each window. It uses the VSW
variables, and shows the larger percentage/number for the VSW column amount and error amount for each window.

**Required keys**

None

**Optional keys**

* ``vsw_windows`` (default = ``None``): which windows from the VSW variables to include (e.g. ``["co2_6220", "ch4_6002"]``). 
  Generally you will not use this input; use ``groups`` instead. Only use this if you need to limit to specific windows.
  When this is ``None`` (the default), all vsw variables are available.
* ``groups`` (default = ``None``): defines how to group the gases into axes. The default is to put all gases into one axes.
  Otherwise, this value must be a list of lists of gas names, e.g. ``[["co2", "ch4"], ["!h2o", "!co2", "!ch4"], ["h2o"]]``. 
  Each inner list corresponds to one axes; this example would plot CO2 and CH4 in the first, everything *except* CO2, CH4,
  and H2O in the second, and only H2O in the third. Prefixing a name with exclamation points (as in the second inner list
  of the example) will exclude that gas from the axes. Note that while it is allowed to mix excludes and includes 
  (e.g. ``["co2", "!h2o"]``), this is identical to only providing the includes (e.g. ``["co2"]``). 
* ``percentage`` (default = ``true``): whether to plot what percentage of the data in each window is a NaN or fill value 
  (``true``) or a number of spectra (``false``).
* ``window_font_size`` (default = 6): the font size to use for the label over each bar that indicates exactly what window 
  it represents. 
* ``sharey`` (default = ``false``): whether to force all axes in this plot to use the same y-limits.

**Style**

A nan-check style subsection must have the ``all`` key, this is the only one used. Keywords can be:

* ``width`` (default = 0.8): the width of each window group. Since this scales all groups together, it isn't generally useful.
* ``zero_color`` (default = "b"): the color to use for the bars for windows with no NaNs/fill values. May be any valid 
  Matplotlib color specification.
* ``color_map`` (default = "autumn_r"): the name of the color map to use to color bars with >0 NaNs/fill values. May be any 
  recognized Matplotlib color map name.

.. note::
   These options have not been tested, please report if they do not work.

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
* ``flag_cat_override`` (default = ``None``): whether to override the default flag category that the data is drawn from for the medians.
  If this option is not present, the all data (flag = 0 and flag > 0) is used unless the command line argument ``--flag0`` is given, in which
  case only flag = 0 data is used. If given, this option must be one of the strings "all", "flag0", or "flagged", and that category of data will
  always be used.

**Style**

A style subsection for one of these plots may have any or all of the keys ``both``, ``am``, or ``pm``. These provide
style keywords that apply to the series for the morning data (``am``), afternoon data (``pm``) or both (``both``).
The keywords given can be any style keywords accepted by :func:`matplotlib.pyplot.plot`.

The `label` keyword is treated specially. In Matplotlib, this keyword is used to set the legend text for a given
data series. The QC plots will include a default label if you do not specify one. If you do specify one, it is
passed through a format call where three keyword values are available:

* ``data`` will be replaced with a short description of the data (site name and whether flag == 0, flag > 0, etc)
* ``ll`` and ``ul`` will be replace with the lower and upper SZA limits, respectively.

delta-timing-error-am-pm
~~~~~~~~~~~~~~~~~~~~~~~~

This is the same as :ref:`timing-error-am-pm <PT_timing_error_am_pm>` except that the value plotted on the y-axis
is the difference (afternoon - morning) instead of plotting them separately. All the required and optional keys are
the same.

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
* ``show_out_of_range_data`` (default = ``true``): determines whether or not to plots points that would fall outside the plot limits at the edge. The 
  default behavior is to plot them; set this to ``false`` to turn that feature off.

.. note::
   The points outside the plot limits will use one of the triangle markers or the large diamond, depending on which limit or limit(s) they are outside.
   If you want to avoid confusing in-limit points for out-of-limit points, do not use any of the markers "v", "^", "<", ">", or "D" in your styles. 

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
* ``show_out_of_range_data`` (default = ``true``): determines whether or not to plots points that would fall outside the plot limits at the edge. The 
  default behavior is to plot them; set this to ``false`` to turn that feature off.

.. note::
   The points outside the plot limits will use one of the triangle markers or the large diamond, depending on which limit or limit(s) they are outside.
   If you want to avoid confusing in-limit points for out-of-limit points, do not use any of the markers "v", "^", "<", ">", or "D" in your styles. 

**Style**

Style configuration is identical to that for :ref:`scatter plots <PT_scatter>`.

delta-timeseries
~~~~~~~~~~~~~~~~

A plot of the difference of two variables vs. time.

**Required keys**

* ``yvar1`` and ``yvar2``: the two variables to difference. The quantity plotted on the *y*-axis will be ``yvar1 - yvar2``.

**Optional keys**

* ``time_buffer_days`` (default = ``2``): number of days to buffer the edges of the plot by to ensure the first and last points do not end up on the plot edge.
* ``show_out_of_range_data`` (default = ``true``): determines whether or not to plots points that would fall outside the plot limits at the edge. The 
  default behavior is to plot them; set this to ``false`` to turn that feature off.

.. note::
   The points outside the plot limits will use one of the triangle markers or the large diamond, depending on which limit or limit(s) they are outside.
   If you want to avoid confusing in-limit points for out-of-limit points, do not use any of the markers "v", "^", "<", ">", or "D" in your styles. 

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
* ``show_out_of_range_data`` (default = ``true``): determines whether or not to plots points that would fall outside the plot limits at the edge. The 
  default behavior is to plot them; set this to ``false`` to turn that feature off.

.. note::
   The points outside the plot limits will use one of the triangle markers or the large diamond, depending on which limit or limit(s) they are outside.
   If you want to avoid confusing in-limit points for out-of-limit points, do not use any of the markers "v", "^", "<", ">", or "D" in your styles. 

**Style**

Style configuration is identical to that for :ref:`scatter plots <PT_scatter>`. Both panels will use the same style for
the same data subset.


.. _PT_ts3panel:

timeseries-3panel
~~~~~~~~~~~~~~~~~

A plot of one variable vs. time with the y-axis split into three panels to allow different degrees of zoom on different
parts of the y-values' ranges. The middle panel will have its y-limits set to those specified in the :ref:`Limits`
file or the min/max values indicated in the netCDF file. The lower and upper panels will show data outside these limits
(less and greater than, respectively) out to either the maximum of the data or limits specified with the ``bottom_limit``
and ``top_limit`` keywords.

**Required keys**

* ``yvar``: the variable from the netCDF file(s) to plot

**Optional keys**

* ``time_buffer_days`` (default = ``2``): number of days to buffer the edges of the plot by to ensure the first and last points do not end up on the plot edge.
* ``show_out_of_range_data`` (default = ``true``): determines whether or not to plots points that would fall outside the plot limits at the edge. The
  default behavior is to plot them; set this to ``false`` to turn that feature off. For this kind of plot, data that exceeds
  the ``top_limit`` will be plotted on the top edge of the upper panel, and data that is less than the ``bottom_limit``
  will be plotted on the bottom edge of the lower panel. (If those keywords are not set, this should be moot, as the limits
  will adjust to include all data.)
* ``plot_height_ratios`` (default = ``[1.0, 1.0, 1.0]``) - a three number sequence giving the relative size of the top, middle, and bottom panels, respectively.
* ``height_space`` (default = ``0.01``) - fraction of vertical space reserved for the gap between plots.
* ``bottom_limit`` (default = ``None``) - providing a value for this keyword sets the lower limit of the bottom panel to that value.
* ``top_limit`` (default = ``None``) - providing a value for this keyword sets the upper limit of the top panel to that value.
* ``even_top_bottom`` (default = ``false``) - set this to ``true`` to automatically set ``bottom_limit`` and ``top_limit`` to be equal in
  magnitude but opposite in sign.

.. note::
   ``even_top_bottom`` cannot be set to ``true`` if either ``bottom_limit`` or ``top_limit`` are provided. Doing so will
   cause an error to be thrown.


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
* ``show_out_of_range_data`` (default = ``true``): determines whether or not to plots points that would fall outside the plot limits at the edge. The 
  default behavior is to plot them; set this to ``false`` to turn that feature off.

.. note::
   The points outside the plot limits will use one of the triangle markers or the large diamond, depending on which limit or limit(s) they are outside.
   If you want to avoid confusing in-limit points for out-of-limit points, do not use any of the markers "v", "^", "<", ">", or "D" in your styles. 

**Style**

Style configuration is identical to that for :ref:`scatter plots <PT_scatter>`.

.. _PT_RollingTimeseries:

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
* ``show_out_of_range_data`` (default = ``true``): determines whether or not to plots points that would fall outside the plot limits at the edge. The 
  default behavior is to plot them; set this to ``false`` to turn that feature off.

.. note::
   The points outside the plot limits will use one of the triangle markers or the large diamond, depending on which limit or limit(s) they are outside.
   If you want to avoid confusing in-limit points for out-of-limit points, do not use any of the markers "v", "^", "<", ">", or "D" in your styles. 

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

rolling-timeseries-3panel
~~~~~~~~~~~~~~~~~~~~~~~~~

A combination of the :ref:`three panel timeseries plot <PT_ts3panel>` and the :ref:`rolling timeseries plot <PT_RollingTimeseries>`.
This plots rolling means, medians, etc. in the three panel format of :ref:`PT_ts3panel`.

**Required keys**

Required keys are the same as :ref:`PT_RollingTimeseries`.

**Optional keys**

All keys accepted by :ref:`PT_RollingTimeseries` and :ref:`PT_ts3panel` are accepted by this plot.

.. _PT_RollingDeltaTimeseries:

delta-rolling-timeseries
~~~~~~~~~~~~~~~~~~~~~~~~

A rolling timeseries plot of the difference between two quantities in the netCDF file.

**Required keys**

* ``yvar1`` and ``yvar2``: the variables from the netCDF file(s) to difference. The quantity plotted on the *y*-axis is 
  ``yvar1 - yvar2``. 
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
* ``show_out_of_range_data`` (default = ``true``): determines whether or not to plots points that would fall outside the plot limits at the edge. The 
  default behavior is to plot them; set this to ``false`` to turn that feature off.

.. note::
   The points outside the plot limits will use one of the triangle markers or the large diamond, depending on which limit or limit(s) they are outside.
   If you want to avoid confusing in-limit points for out-of-limit points, do not use any of the markers "v", "^", "<", ">", or "D" in your styles. 

**Style**

Style is the same as for :ref:`PT_RollingTimeseries`.


rolling-derivative
~~~~~~~~~~~~~~~~~~

Rolling derivative plots compute a derivative of one variable vs. another across spectra in a rolling 
window. For example, if told to compute the first derivative of ``y`` with respect to ``x`` using a 
rolling window of 500 spectra, this will take spectra 1 through 500 and fit a slope of ``y`` versus 
``x`` in those 500 spectra, then do the same for spectra 2 through 501, and so on. 

**Required keys**

* ``yvar``: the variable in the numerator of the derivative (the dependent variable).
* ``dvar``: the varibale in the denominator of the derivative (the independent variable).

**Optional keys**

* ``derivative_order`` (default = ``1``): order of the derivative to calculate; ``1`` will compute a slope, ``2`` curvature, etc.
  Only ``1`` is implemented.
* ``gap`` (default = ``"20000 days"``): this specified a gap in time that the rolling operation will not cross. This can
  be any string recognized by `Pandas timedelta <https://pandas.pydata.org/pandas-docs/stable/user_guide/timedeltas.html>`_.
  If there is a gap in the data longer than this duration, the data on either side will have the rolling operation
  applied separately. The default of "20000 days" (~50 years) is set to effectively disable this behavior by default.
* ``rolling_window`` (default = ``500``): the number of points to use in the rolling window.
* ``data_category`` (default = ``None``): which subset of the data to use when computing the rolling derivative. 
  The default behavior is to use the normal subset for a given data type, or
  ``flag == 0`` data if the ``--flag0`` command line argument is set. Passing one of the strings "all", "flag0", or
  "flagged" will force the use of that subset (this may result in errors if one of the data files does not have the
  "flag" variable, which is required to figure out the latter two subsets).
* ``show_out_of_range_data`` (default = ``true``): determines whether or not to plots points that would fall outside the plot limits at the edge. The 
  default behavior is to plot them; set this to ``false`` to turn that feature off.

.. note::
   The points outside the plot limits will use one of the triangle markers or the large diamond, depending on which limit or limit(s) they are outside.
   If you want to avoid confusing in-limit points for out-of-limit points, do not use any of the markers "v", "^", "<", ">", or "D" in your styles. 
   
zmin-zobs-delta-rolling-timeseries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A :ref:`PT_RollingDeltaTimeseries` plot customized for the zmin - zobs difference. It includes the estimated corresponding
pressure difference on the right hand side of the axes as well as an annotation indicating the site altitude and bottom GEOS 
level altitude.

**Required keys**

* ``ops``: same as for :ref:`PT_RollingDeltaTimeseries`.

**Optional keys**

* ``annotation_font_size`` (default = ``6``): the font size for the site/GEOS altitude annotation.

The other optional keys are the same as for :ref:`PT_RollingDeltaTimeseries`. Note that when using the 
violin auxiliary plot for this, the ``violin_plot_pad`` keyword is given a default value of 1.0 instead
of 0.5 and the violin plot y-ticks are turned off by default. Both of these changes are to allow space 
for the estimated pressure difference.

.. _AuxPlots:

Auxiliary Plots
***************

Auxiliary plots are extra panels that can be added to a main plot to provide extra information. To add an 
auxiliary plot to a page, add ``+<auxkind>`` to the end of the main plot's ``kind`` values. For example, to 
add a violin plot to a timeseries plot, set the ``kind`` value to ``"timeseries+violin"``.  Internally, 
``"timeseries"`` and ``"timeseries+violin"`` are implemented as separate plot kinds. While this should be 
largely transparent to a user, it does have several implications to be aware of:

#. Not all combinations of main + auxiliary plots will be implemented. Which auxiliary plots are supported 
   with which main plots is listed above in :ref:`PlotTypes`.
#. Only one auxiliary plot can be combined with a main plot. (Allowing multiple auxiliary plots to be added 
   to a single plot would require a separate implementation for each possible combination, which isn't practical.
   Future work could refactor the approach to auxiliary plots to make this more viable.)
#. A main + auxiliary combination can have different styles and limits than the the main plot type alone; 
   to continue our example from the first paragraph, you could readily define a ``["timeseries+violin"]`` section 
   in the :ref:`Limits` or a ``[style.main."timeseries+violin"]`` :ref:`Styles` section to set limits or a style 
   customized for timeseries plots with a violin plot attached only (i.e. not for normal timeseries plots). However,
   the default behavior is for the main plot to use the limits and styles it would without the auxiliary plot.

.. note::
   If you do add a section for a main+auxiliary plot, you will need to quote the plot kind in the TOML file. 
   Note how in the examples in the last point above, such as ``[style.main."timeseries+violin"]`` the 
   "timeseries+violin" part is quoted. TOML files will not include plus signs in a string without it being quoted;
   if you did not quote this (i.e. ``[style.main.timeseries+violin]``), it would be interpreted as a section 
   named ``[style.main.timeseries]``. If you already have a section named that, you'll get a TOML error when 
   running QC plots.

Note that in the third point above, the styles referred to are those for the main plot. Styles for the auxiliary 
plots need to be defined separately in the configurations; this will be described with each plot kind below.

The following subsections describe the available auxiliary plot kinds, including the extra required or optional 
keys they add to their ``[[plots]]`` section in the configuration file and their style options.

Violin aux plots
~~~~~~~~~~~~~~~~

A violin auxiliary plot adds a small plot to the side of the main plot that shows the distribution of the 
*y*-variable of the main plot in some standard good-quality data. Note that this is separate from the 
normal reference file. 

**Required keys**

* ``violin_data_file``: a path to the netCDF file to use to create the violin plots.

**Optional keys**

* ``violin_plot_side`` (default = "right"): which side of the main plot axes to put the violin plot on.
  Can be "right", "left", "bottom", or "top" (though only "left" or "right" are recommended).
* ``violin_plot_size`` (default = "10%"): how big to make the violin plot horizontally (if the side is 
  "left" or "right") or vertically (if the side is "bottom" or "top"). To give as a percentage of 
  the original plot size (easiest), make this a string ending in the percent sign, as the default is. 
* ``violin_plot_pad`` (default = 0.5): space to reserve between the original axes and the new violin plot 
  axes. 
* ``violin_plot_hide_yticks`` (default = ``false``): set to ``true`` to hide the *y*-tick labels on the 
  violin plot axes.

**Style**

Style for the violin plots is read exclusively from the ``[style.extra.aux-violin]`` section. While this can 
have all the usual data subsets as keys (``flag0``, ``flagged``, ``all``), usually only the ``flag0`` style 
matters since violin plots use flag = 0 exclusively. This can accept any keywords that 
:py:func:`matplotlib.pyplot.violinplot` does except for ``dataset`` and ``positions`` (they are already used),
plus two additional keywords:

* ``fill_color``: color to make the violin density kernel.
* ``line_color``: color to make any lines (medians, extrema, etc.) on the plot.

An example style section is::

    [style.extra.aux-violin]
    flag0 = {showmedians = true, showextrema = false, fill_color = "silver", line_color = "dimgray"}


.. _Limits:

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


Email file
----------

The email configuration file allows you to specify how to send emails containing the plots. An example file is::

    [server]
    use_external_program = true

    [server.program]
    program = "mail"
    subject_flag = "-s"
    from_addr_flag = "-r"
    attachment_flag = "-a"
    body_arg = "stdin"

    [server.smtp]
    smtp_address = "smtp.gmail.com"
    smtp_port = 587

    [email]
    from = "me@self.com"
    to = "you@other.edu"
    body = "Plots automatically generated by `tccon_qc_plots` on {date} from {basename}."
    subject_from_site_id = true
    subject = "[#275]"

    [email.sites]
    ae = 226  # Ascension Island
    an = 224  # Anmyeondo
    bi = 213  # Bialystok
    br = 236  # Bremen
    bu = 220  # Burgos
    ci = 210  # Caltech/Pasadena
    db = 214  # Darwin
    df = 225  # Armstrong/Dryden/Edwards
    et = 227  # East Trout Lake
    eu = 222  # Eureka
    gm = 234  # Garmisch
    hf = 276  # Hefei
    hw = 274  # Harwell
    iz = 216  # Izana
    js = 233  # Saga
    ka = 217  # Karlsruhe
    ll = 219  # Lauder pre-2018 125HR
    lr = 221  # Lauder post-2018 125HR
    ni = 240  # Nicosia
    ny = 237  # Ny-Alesund
    oc = 223  # Lamont
    or = 212  # Orleans
    pa = 211  # Park Falls
    pr = 231  # Paris
    ra = 260  # Reunion
    rj = 228  # Rikubetsu
    so = 218  # Sodankyla
    sp = 237  # Alternate abbreviation for Ny-Alesund?
    tj = 229  # Tsukuba 120HR
    tk = 229  # Tsukuba 125HR
    wg = 215  # Wollongong
    xh = 271  # Xianghe
    zs = 235  # Zugspitze

This is a `TOML <https://toml.io/en/>`_ document. For details on the TOML syntax, see https://toml.io/en/. Now,
let's consider each section.

server section
**************

This section contains general options for what email server to use to send the email. It only has one option presently:

* ``use_external_program`` - a boolean value that determines whether emails are sent using a command line program like
  ``mail`` (true) or Python's own SMTP library (false).

server.program section
**********************

This section contains options specific to the case where emails are sent using a command line program. The required options
are:

* ``program`` - the name of the command line program to call.
* ``subject_flag`` - what command line flag to use to pass the subject of the email.
* ``from_addr_flag`` - what command line flag to use to pass the from address.
* ``attachment_flag`` - what command line flag to use to pass a path to a file to attach.
* ``body_arg`` - How to pass the body of the email. Currently the only acceptable value is "stdin", meaning that the program
  accepts the body through piping (e.g. ``echo "This is the body" | mail``) or input redirection (e.g. ``mail < body_file``).
  If you intend to use a program that does not accept the message body in this way, QC plots will need updated.

Note that when using an external program, QC plots assumes that it accepts the "to" email addresses as the sole positional argument.
If you wish to use an email program for which that is not true, QC plots will need upgraded.

server.smtp section
*******************

This section contains options specific to the case where emails are sent using Python's :py:mod:`smtp` module. Note that this functionality
has not been thoroughly tested, as it did not work with the SMTP server on tccondata.

* ``smtp_address`` - what address to connect to to send the email. Common values are "localhost" (use an SMTP server on this computer),
  "smtp.gmail.com" (to send from a Gmail account) and "smtp.outlook.com" (to send from an Outlook account). Note that these last two
  may require an account for which insecure sending is permitted.
* ``smtp_port`` - what port to connect to. A value of 0 will try to guess; gmail and output server both use 587.
* ``password`` - password to use to connect to the sending account.
* ``requires_auth`` - whether the sending account needs authentication (true or false). If true, then you will be prompted to enter your
  password interactively (so don't use this in automated scripts). If false, the sending account does not need authentication to connect.
  If ``password`` is present, this option is ignored. Otherwise, true is the default.

.. warning::
   If you put a login password in this configuration file, you should make sure that only trusted users can read it. On a Unix/Linux system, you should
   remove all access permissions for "other" set of users at the very least, and ideally this file would only be readable by the owner.

email section
*************

The section controls the content of the email, as well as where it is sent.

* ``from`` - the sending email address. This can be used to connect to a Gmail or Outlook server if ``use_external_program`` is false, in which
  case you will need to provide login authentication. If sending emails with an external (command line) program, this account does not need to
  be logged in to, it will just be set as the sender.
* ``to`` - the recipient email address. If sending to GGGBugs, this will be the same email that sends alerts about watched topics (that you can
  reply to to update the topic).
* ``body`` - the main body of the email. There are three substrings that will be substituted with useful values, if present:

    * "{date}" will be replaced with the current date, time, and timezone when the email is sent.
    * "{basename}" will be replaced with the name of the netCDF file given as input to the plotting program, without leading directories.
    * "{plot_url}" will be replaced with a URL at which the plots can be accessed. Note that if the plotting program is called without the
      ``--plot-url`` command line argument (or ``plot_url=None`` in the driver Python function) then this value will be ``None``. If this
      substring is *not* present in your email body, but the QC plotting program was told to provide a URL, then a short sentence giving the
      URL is appended automatically.

.. note::
   The body is formatted using Python's `string formatter <https://docs.python.org/3/library/string.html?highlight=strings#format-string-syntax>`_.
   This means that if you have curly brace in the body (other than in the allowed substrings listed above) it will try to replace those curly braces,
   and probably crash due to missing format arguments. Avoid putting curly braces in your body other than around the substrings mentioned above, 
   but if you *must* have a curly brace, write a double brace (``{{`` or ``}}``) to protect it from formatting.

* ``subject_from_site_id`` - a boolean indicating if the subject should be derived from the site ID, which is assumed to be the first two
  letters of the netCDF file name. If this is true, then the subject is determined from the email.sites section of this configuration file.
  If this is false, then the subject is set to the value of the ``subject`` setting in this section.
* ``subject`` - the subject for the email; only used if ``subject_from_site_id`` is false.

email.sites section
*******************

In this section, each key value (on the left site of the equals sign) is a site ID, and the value on the right is the topic number that site
has in GGG Bugs. When sending an email with ``subject_from_site_id = true``, the first two letters of the netCDF file name will be compared
against the keys in this section. If a match is found, the subject will be "[#N]", where *N* is the number from this section. This is the
format GGG Bug's redmine software uses to match up an incoming email to a topic. 

.. warning::
   If your netCDF file has a site ID not in this list when ``subject_from_site_id`` is true, you'll get an error and the email won't send.
