from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
from enum import Enum
from fnmatch import fnmatch
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from math import ceil
import netCDF4 as ncdf
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import Optional, Sequence, Tuple, Union
import warnings

import tomli

from . import utils
from .constants import DEFAULT_LIMITS, DEFAULT_IMG_DIR


# TODO:
#   - [x] Rolling timeseries
#       + Test that gap splitting works
#   - Do a 1:1 comparison with the old code
#   - Test reference file
#   - Test context file
#   - Test limits file works
#       + It did for the o2_7885_vsf_o2 resampled plots
#   - Test uncertainty and multiple ops for the rolling plots

class FlagCategory(Enum):
    ALL_DATA = 'all'
    FLAG0 = 'flag0'
    FLAGGED = 'flagged'


class DataCategory(Enum):
    PRIMARY = 'primary'
    REFERENCE = 'reference'
    CONTEXT = 'context'


class DataError(Exception):
    pass


class PlotClassError(Exception):
    pass


class TcconData:
    def __init__(self,
                 data_category: DataCategory,
                 nc_file,
                 styles: dict,
                 exclude_times=None,
                 include_times=None,
                 allowed_flag_categories=None):
        self.data_category = data_category
        self.nc_dset = ncdf.Dataset(nc_file)
        self.styles = styles

        # Directly converting the netCDF times to matplotlib times is ~700x
        # faster, in my testing (JLL). I've done my best to check that the
        # conversion will be accurate within the `nctime_to_mpltime` function
        # by verifying that at least one conversion is correct. 
        #
        self.datetimes = self.nctime_to_pytime(self.nc_dset['time'])
        # self.times = self.pytime_to_mpltime(self.datetimes)
        self.times = self.nctime_to_mpltime(self.nc_dset['time'])

        if exclude_times is not None:
            self.all_idx = (self.times < np.min(exclude_times)) | (self.times > np.max(exclude_times))
        elif include_times is not None:
            self.all_idx = (self.times >= np.min(include_times)) & (self.times <= np.max(include_times))
        else:
            self.all_idx = np.ones(self.times.shape, dtype=np.bool_)

        available_categories = {FlagCategory.ALL_DATA}

        if 'flag' in self.nc_dset.variables:
            self.flag0_idx = (self.nc_dset['flag'][:] == 0) & self.all_idx
            self.flagged_idx = (self.nc_dset['flag'][:] != 0) & self.all_idx
            available_categories.update({FlagCategory.FLAG0, FlagCategory.FLAGGED})
        else:
            self.flag0_idx = None
            self.flagged_idx = None

        # Limit which categories of data plots can access to the intersection of what the user
        # specified this dataset should provide and what is actually available
        if allowed_flag_categories is None:
            self._allowed_flag_categories = available_categories
        else:
            self._allowed_flag_categories = available_categories.intersection(allowed_flag_categories)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.nc_dset.close()

    def has_flag_category(self, flag_category: FlagCategory) -> bool:
        return flag_category in self._allowed_flag_categories

    def get_data(self, varname, flag_category: FlagCategory) -> np.ma.masked_array:
        if flag_category not in self._allowed_flag_categories:
            raise DataError('This TcconData instance does not provide data in flag category {}'.format(flag_category))

        if flag_category == FlagCategory.ALL_DATA:
            return self._get_all_data(varname)
        elif flag_category == FlagCategory.FLAG0:
            return self._get_flag0_data(varname)
        elif flag_category == FlagCategory.FLAGGED:
            return self._get_flagged_data(varname)

    @property
    def default_category(self):
        if FlagCategory.FLAG0 in self._allowed_flag_categories:
            return FlagCategory.FLAG0
        elif FlagCategory.ALL_DATA in self._allowed_flag_categories:
            return FlagCategory.ALL_DATA
        else:
            raise DataError('This TcconData instance provides neither flag0 nor all data')

    def get_flag0_or_all_data(self, varname) -> np.ma.masked_array:
        if FlagCategory.FLAG0 in self._allowed_flag_categories:
            return self._get_flag0_data(varname)
        elif FlagCategory.ALL_DATA in self._allowed_flag_categories:
            return self._get_all_data(varname)
        else:
            raise DataError('This TcconData instance provides neither flag0 nor all data')

    def _get_variable(self, varname) -> Union[np.ndarray, np.ma.masked_array]:
        if varname == 'time':
            return self.times
        elif varname == 'datetime':
            return self.datetimes
        else:
            return self.nc_dset[varname][tuple()]

    def _get_all_data(self, varname) -> np.ma.masked_array:
        data = self._get_variable(varname)
        return data[self.all_idx]

    def _get_flag0_data(self, varname) -> np.ma.masked_array:
        if self.flag0_idx is None:
            raise DataError('This TcconData instance wraps a dataset that does not include flags - data cannot be subset to flag0')
        data = self._get_variable(varname)
        return data[self.flag0_idx]

    def _get_flagged_data(self, varname) -> np.ma.masked_array:
        if self.flagged_idx is None:
            raise DataError('This TcconData instance wraps a dataset that does not include flags - data cannot be subset to flagged')

        data = self._get_variable(varname)
        return data[self.flagged_idx]

    def get_label(self, flag_category: FlagCategory) -> str:
        if flag_category not in self._allowed_flag_categories:
            raise DataError('This TcconData instance does not provide data in flag category {}'.format(flag_category))

        if flag_category == FlagCategory.FLAG0:
            return self._get_flag0_label()
        elif flag_category == FlagCategory.FLAGGED:
            return self._get_flagged_label()
        elif flag_category == FlagCategory.ALL_DATA:
            return self._get_alldata_label()

    def get_flag0_or_all_label(self) -> str:
        if FlagCategory.FLAG0 in self._allowed_flag_categories:
            return self._get_flag0_label()
        elif FlagCategory.ALL_DATA in self._allowed_flag_categories:
            return self._get_alldata_label()
        else:
            raise DataError('This TcconData instance provides neither flag0 nor all data')

    def _get_flag0_label(self) -> str:
        site_name = self.nc_dset.long_name
        if self.data_category == DataCategory.PRIMARY:
            return f'{site_name} flag==0'
        else:
            return f'{site_name} flag==0 ({self.data_category.value})'

    def _get_flagged_label(self) -> str:
        site_name = self.nc_dset.long_name
        if self.data_category == DataCategory.PRIMARY:
            return f'{site_name} flagged'
        else:
            return f'{site_name} flagged ({self.data_category.value})'

    def _get_alldata_label(self) -> str:
        site_name = self.nc_dset.long_name
        if self.data_category == DataCategory.PRIMARY:
            return f'{site_name} all data'
        else:
            return f'{site_name} all data ({self.data_category.value})'

    @staticmethod
    def nctime_to_pytime(nc_time_var: ncdf.Variable):
        cftimes = ncdf.num2date(nc_time_var[:], units=nc_time_var.units, calendar=nc_time_var.calendar)
        return np.array([datetime(*t.timetuple()[:6]) for t in cftimes])

    @staticmethod
    def pytime_to_mpltime(pytimes):
        return mdates.date2num(pytimes)

    @staticmethod
    def nctime_to_mpltime(nc_time_var: ncdf.Variable):
        # First, check that the matplotlib epoch is the numpy epoch
        mpl_epoch = np.datetime64(mdates.get_epoch()).astype('float')
        assert np.isclose(mpl_epoch, 0)

        # Second, check that converting to numpy.datetime64[s] gives the same answer as converting
        # to the python time first
        cftime = ncdf.num2date(nc_time_var[0], units=nc_time_var.units, calendar=nc_time_var.calendar)
        pytime = datetime(*cftime.timetuple()[:6])
        assert pytime == nc_time_var[0].astype('datetime64[s]').item()

        # If so we can do a very efficient conversion
        return mdates.date2num(nc_time_var[:].astype('datetime64[s]'))


class Limits:
    def __init__(self, limits_file):
        with open(limits_file) as f:
            self.limits = tomli.load(f)

    def get_limit(self, varname: str, data: Union[TcconData, Sequence[TcconData]], plot_kind: Optional[str] = None) -> Tuple[float, float, bool]:
        if isinstance(data, TcconData):
            data = [data]

        # Prefer to get the limit from the limits file, and in the file, prefer plot-specific sections
        plot_section = self.limits.get(plot_kind, dict())
        plot_limits = self._get_var_from_section(varname, plot_section)
        if plot_limits is not None:
            # If we found limits, the third return value is False to tell the caller to turn off axis autoscaling
            return plot_limits[0], plot_limits[1], False

        defaults_section = self.limits.get('default', dict())
        default_limits = self._get_var_from_section(varname, defaults_section)
        if default_limits is not None:
            # If we found limits, the third return value is False to tell the caller to turn off axis autoscaling
            return default_limits[0], default_limits[1], False

        # Finally try getting limits from the variable in the netCDF dataset
        # If those attributes aren't present, default to None (i.e. do not set limits)
        vmins = [getattr(d.nc_dset[varname], 'vmin', None) for d in data]
        vmins = [v for v in vmins if v is not None]
        vmaxes = [getattr(d.nc_dset[varname], 'vmax', None) for d in data]
        vmaxes = [v for v in vmaxes if v is not None]
        vmin = min(vmins) if len(vmins) > 0 else None
        vmax = max(vmaxes) if len(vmaxes) > 0 else None
        # In this branch whether or not the caller should turn of autoscaling depends on whether or not we found
        # both limits. If missing either, we should allow autoscaling so that the plot displays properly. Even
        # setting limits as `None` will turn off autoscaling by default.
        if (vmin is None) != (vmax is None):
            print(f'\nWARNING: found only one limit for variable {varname}; it will be ignored', file=sys.stderr)
        autoscale = vmin is None or vmax is None
        return vmin, vmax, autoscale

    @staticmethod
    def _get_var_from_section(varname: str, section: dict):
        for key in section.keys():
            if fnmatch(varname, key):
                return section[key]

        return None


class AbstractPlot(ABC):
    plot_kind = None

    def __init__(self, other_plots, default_style, limits: Limits, key=None, width=20, height=10,
                 legend_kws: Optional[dict] = None):
        self.key = key
        self._other_plots = other_plots
        self._default_style = default_style
        self._limits = limits
        self._width = width
        self._height = height
        self._legend_kws = dict() if legend_kws is None else legend_kws

    @classmethod
    def from_config_section(cls, section, other_plots, full_config, limits_file):
        kind = section.pop('kind')
        klass = cls._dispatch_plot_kind(kind)
        default_styles = full_config.get('style', dict()).get('default', dict())
        limits = Limits(limits_file)
        # other_plots must be a reference to the list of plots so that all plots know about each other,
        # as a reference, it updates so that the first plots know about the last plots
        return klass(other_plots=other_plots, default_style=default_styles, limits=limits, **section)

    @classmethod
    def _get_subclasses(cls):
        """Get both direct and indirect subclasses

        Credit to https://stackoverflow.com/a/5883218
        """
        subclasses = set()
        work = [cls]
        while work:
            parent = work.pop()
            for child in parent.__subclasses__():
                if child not in subclasses:
                    subclasses.add(child)
                    work.append(child)
        return subclasses

    @classmethod
    def _dispatch_plot_kind(cls, kind):
        """Return the correct plot subclass for the specified plot kind.

        Parameters
        ----------
        kind : str
            What kind of plot requested in the configuration
        """
        # Gather all subclasses - ensure no one forgot to update their action/how strings
        available_classes = dict()

        def add_class(klass):
            key = klass.plot_kind
            if key is None:
                return
            elif key in available_classes:
                conflicting_class = available_classes[key]
                raise TypeError(
                    f'{klass.__name__} has a plot kind ("{key}") already in use by {conflicting_class.__name__}. '
                    f'Please make sure to define a unique combination of the `action` and `how` class attributes '
                    f'for each {cls.__name__} subclass.')
            else:
                available_classes[key] = klass

        add_class(cls)
        for subcls in cls._get_subclasses():
            add_class(subcls)

        try:
            return available_classes[kind]
        except KeyError:
            raise PlotClassError(f'Plot kind "{kind}" not implemented') from None

    def get_plot_by_key(self, key):
        for plot in self._other_plots:
            if plot.key is not None and plot.key == key:
                return plot

        raise KeyError(f'No plot with key {key} found')

    def get_plot_args(self, data: TcconData, flag0_only: bool = False):
        # There will always be at least one thing to plot, whether it is all data or just flag0 data
        plot_args = [{
            'data': self.get_plot_data(data, FlagCategory.FLAG0),
            'kws': self.get_plot_kws(data, FlagCategory.FLAG0),
            'legend_kws': self.get_legend_kws()
        }]

        # Whether there is more data to plot depends on whether the user specified that we should only plot
        # flag0 data and whether this particular data cares
        if not flag0_only and data.has_flag_category(FlagCategory.FLAGGED):
            plot_args.append({
                'data': self.get_plot_data(data, FlagCategory.FLAGGED),
                'kws': self.get_plot_kws(data, FlagCategory.FLAGGED),
            })

        return plot_args

    def get_plot_kws(self, data: TcconData, flag_category: Optional[FlagCategory], _format_label: bool = True) -> dict:
        # Get the default style: style dictionaries are organized plot_kind -> flag category,
        # Allow both to be missing, and just provide an empty dictionary as the default
        style_fc = data.default_category if flag_category is None else flag_category
        default = self._get_style(self._default_style, self.plot_kind, style_fc)
        specific = self._get_style(data.styles, self.plot_kind, style_fc)

        # Override options in default with values in the data type-specific keywords
        kws = deepcopy(default)
        kws.update(specific)

        if _format_label:
            # Since labels need some extra logic to format, add it separately here if we're supposed to
            data_label = data.get_flag0_or_all_label() if flag_category is None else data.get_label(flag_category)
            label_spec = kws.get('label', '{data}')
            kws['label'] = label_spec.format(data=data_label)
        return kws

    def get_legend_kws(self):
        kws = deepcopy(self._get_style(self._default_style, self.plot_kind, 'legend_kws'))
        kws.update(self._legend_kws)
        kws.setdefault('fontsize', 7)
        return kws

    def _get_style(self, styles, plot_kind, sub_category: Union[FlagCategory, str]):
        plot_styles = styles.get(plot_kind, dict())
        sc = sub_category if isinstance(sub_category, str) else sub_category.value
        if 'clone' in plot_styles and sc not in plot_styles:
            return self._get_style(styles, plot_styles['clone'], sc)
        else:
            return plot_styles.get(sc, dict())

    def make_plot(self, data: Sequence[TcconData], flag0_only: bool = False, show_all: bool = False,
                  img_path: Path = DEFAULT_IMG_DIR, tight=True):
        fig, axs = self.setup_figure(data, show_all=show_all)
        for i, d in enumerate(data):
            self._plot(d, i, axs=axs, flag0_only=flag0_only)
        fig_path = img_path / self.get_save_name()
        if tight:
            # I tried using bbox_inches='tight' in the savefig call, but it causes the scatter plots/hexbins
            # to not line up, so we'll stick with tight_layout() and just suppress the warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                fig.tight_layout()
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)
        return fig_path
    
    def _make_or_check_fig_ax_args(self, fig, axs, nrow=1, ncol=1, ax_ndim=None, ax_size=None, ax_shape=None):
        if (fig is None) != (axs is None):
            raise TypeError('Must give both or neither of fig and axs')
        elif fig is None and axs is None:
            size = utils.cm2inch(self._width, self._height)
            fig, axs = plt.subplots(nrow, ncol, figsize=size)
        elif ax_ndim is not None and np.ndim(axs) != ax_ndim:
            raise TypeError('axs must be a {} dimensional array'.format(ax_ndim))
        elif ax_size is not None and np.size(axs) != ax_size:
            raise TypeError('axs must be an array with {} elements'.format(ax_size))
        elif ax_shape is not None and np.shape(axs) != ax_shape:
            raise TypeError('axs must be an array with shape {}'.format(ax_shape))
        
        return fig, axs

    def add_qc_lines(self, ax, axis: str, nc_var: ncdf.Variable):
        axline_fxn = ax.axvline if axis == 'x' else ax.axhline
        vmin = getattr(nc_var, 'vmin', None)
        if vmin is not None:
            axline_fxn(vmin, linestyle='--', color='black')
        vmax = getattr(nc_var, 'vmax', None)
        if vmax is not None:
            axline_fxn(vmax, linestyle='--', color='black')

    @staticmethod
    def _get_data_for_limits(data: Sequence[TcconData]) -> Sequence[TcconData]:
        """Get the subset of data instances to use when determining limits"""
        # Omit the reference data - we usually want the limits to focus on the
        # main data (+ context, if give)
        return [d for d in data if d.data_category != DataCategory.REFERENCE]

    @staticmethod
    def _get_main_data(data: Sequence[TcconData]) -> TcconData:
        """Get the data instance that represents the main focus of this plot"""

        # Try to find the PRIMARY data type; if that's missing, assume that the first
        # data in the list is the one to use
        main_data = [d for d in data if d.data_category == DataCategory.PRIMARY]
        if len(main_data) > 0:
            return main_data[0]
        else:
            return data[0]

    @staticmethod
    def _get_flag_category(flag_category, flag0_only: bool = False):
        # Which subset of data to use depends on several things:
        #   * If the plot options included one, then use that
        #   * If the plot options did not include one, but --flag0 was set, only use flag0 data
        #   * Otherwise, use the default for this data type
        if flag_category is None and not flag0_only:
            return None
        elif flag_category is None and flag0_only:
            return FlagCategory.FLAG0
        else:
            return flag_category

    def _get_plot_args_mono(self, data: TcconData, flag_category: Optional[FlagCategory], flag0_only: bool = False):
        flag_category = self._get_flag_category(flag_category, flag0_only)
        plot_args = [{
            'data': self.get_plot_data(data, flag_category),
            'kws': self.get_plot_kws(data, flag_category),
            'legend_kws': self.get_legend_kws()
        }]
        return plot_args

    @classmethod
    def get_label_string(cls, main_data, variable):
        units = cls.get_units(main_data, variable)
        if units:  # okay to just check for truthiness; this way empty strings avoid leaving empty parentheses
            return f'{variable} ({units})'
        else:
            return f'{variable}'

    @staticmethod
    def get_units(main_data, variable):
        return getattr(main_data.nc_dset[variable], 'units', '')

    @abstractmethod
    def setup_figure(self, data: Sequence[TcconData], show_all: bool = False, fig=None, axs=None):
        pass

    @abstractmethod
    def get_plot_data(self, data: TcconData, flagged_category: FlagCategory):
        pass

    @abstractmethod
    def get_save_name(self):
        pass

    @abstractmethod
    def _plot(self, data: TcconData, idata: int, axs=None, flag0_only: bool = False):
        pass


class TimeseriesMixin:
    def format_time_axis(self, ax, data: Sequence[TcconData], xvar='time', buffer_days=2, show_all: bool = False):
        first_time = min(np.min(d.get_data(xvar, FlagCategory.ALL_DATA)) for d in data)
        last_time = max(np.max(d.get_data(xvar, FlagCategory.ALL_DATA)) for d in data)
        self.format_time_axis_custom_times(ax, first_time, last_time, buffer_days=buffer_days, show_all=show_all)

    @staticmethod
    def format_time_axis_custom_times(ax, first_time, last_time, buffer_days=2, show_all: bool = False):
        if not show_all:
            # assumes that times are in Matplotlib date numbers, which are days since some epoch
            xmin = first_time - buffer_days
            xmax = last_time + buffer_days
            ax.set_xlim(xmin, xmax)

        # Also override the x-axis label
        ax.set_xlabel('Time')

        # And fix the ticks to look halfway decent
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    @staticmethod
    def resample_data(times, yvals, freq, op):
        df = pd.DataFrame({'y': yvals}, index=times)
        df = getattr(df.resample(freq), op)()  # this retrieves the method named `op` and calls it
        return df['y']

    @classmethod
    def roll_data(cls, xvals, yvals, npts, ops, gap, times=None):
        """
        Compute rolling statistics on data

        Inputs:
            - xvals: array of x values
            - yvals: array of y values
            - npts: number of points to use in the rolling window
            - ops: name of an operation to do on the rolling windows, or a list of such names
            - gap: a Pandas Timedelta string (https://pandas.pydata.org/pandas-docs/stable/user_guide/timedeltas.html)
              specifying the longest time difference between adjacent points that a rolling window can operate over.
              That is, if this is set to "7 days" and there is a data gap a 14 days, the data before and after that
              gap will have the rolling operation applied to each separately.
            - times: if `xvals` is not a time variable, then this input must be the times corresponding to the `xvals`
              and `yvals`. It is used to split on gaps.

        Outputs:
            - outputs: If `ops` was a single string (e.g. "median"), then a single dataframe with "x", "y", and (if
            `times` was given) "time" columns that has the result of the rolling operation. If `ops` was a list, then
            the output is a list of dataframes, each one with the result of the corresponding operation.
        """
        if isinstance(ops, str):
            ops = [ops]
            return_single = True
        else:
            return_single = False

        df = pd.DataFrame({'x': xvals, 'y': yvals})
        if times is not None:
            df['time'] = times
        grouped_df = cls.split_by_gaps(df, gap, 'x' if times is None else 'time')
        outputs = []
        for op in ops:
            results = []
            for n, group in grouped_df:
                if 'quantile' in op:
                    q = float(op.replace('quantile', ''))
                    result = group.rolling(npts, center=True, min_periods=1).quantile(q)
                else:
                    result = getattr(group.rolling(npts, center=True, min_periods=1), op)()

                # Times cannot be computed with rolling operations, therefore we need to copy the
                # time column back over
                if times is not None and 'time' not in result.columns:
                    result['time'] = group['time']
                elif 'x' not in result.columns:
                    result['x'] = group['x']
                results.append(result)
            outputs.append(pd.concat(results))

        if return_single:
            return outputs[0]
        else:
            return outputs

    @staticmethod
    def split_by_gaps(df, gap, time):
        """
        Split an input dataframe with a datetime variable into a groupby dataframe with each group having data without gaps larger than the "gap" variable

        Inputs:
            - df: pandas dataframe with a datetime
            - time: column name of the datetime variable
            - gap: minimum gap length, a string compatible with pandas Timedelta https://pandas.pydata.org/pandas-docs/stable/user_guide/timedeltas.html
        Outputs:
            - df_by_gaps: groupby object with each group having data without gaps larger than the "gap" variable
        """

        df['isgap'] = df[time].diff() > pd.Timedelta(gap)
        df['isgap'] = df['isgap'].apply(lambda x: 1 if x else 0).cumsum()
        df_by_gaps = df.groupby('isgap')

        return df_by_gaps


class FlagAnalysisPlot(AbstractPlot):
    plot_kind = 'flag-analysis'

    def __init__(self, other_plots, default_style, limits: Limits, key=None, min_percent=1.0, width=20, height=10,
                 legend_kws: Optional[dict] = None):
        super().__init__(other_plots=other_plots, default_style=default_style, key=key, limits=limits,
                         width=width, height=height, legend_kws=legend_kws)
        self.min_percent = min_percent

    def setup_figure(self, data: Sequence[TcconData], show_all: bool = False, fig=None, axs=None):
        fig, axs = self._make_or_check_fig_ax_args(fig, axs, nrow=2, ncol=1, ax_ndim=1, ax_size=2)

        main_data = self._get_main_data(data)
        nspec = main_data.nc_dset['time'].size
        site_name = main_data.nc_dset.long_name
        axs[0].set_title('{} total spectra from {}'.format(nspec, site_name))
        axs[0].set_ylabel('Count')
        axs[1].set_ylabel('Count (%)')

        for elem in axs:
            elem.axes.get_xaxis().set_visible(False)
            elem.grid()

        fig.suptitle('Summary of flags with count>1%')
        return fig, axs

    def get_plot_args(self, data: TcconData, flag0_only: bool = False):
        # The flag summary plot always uses all data
        plot_args = [{
            'data': self.get_plot_data(data, FlagCategory.ALL_DATA),
            'kws': self.get_plot_kws(data, FlagCategory.ALL_DATA),
            'legend_kws': self.get_legend_kws()
        }]

        return plot_args

    def get_plot_data(self, data: TcconData, flag_category: Optional[FlagCategory]) -> dict:
        flag_df = pd.DataFrame()
        flag_df_pcnt = pd.DataFrame()
        flag_set = sorted(set(data.nc_dset['flag'][:]))
        nspec = data.nc_dset['time'].size
        nflag_tot = 0

        # Need the leading newline because the plot progress message omits its trailing newline to
        # allow printing "DONE"
        print('\nSummary of flags:')
        print('  #  Parameter              N_flag      %')

        # For each unique flag, count the number and percent of spectra flagged by that variable
        # Print all of them to the screen
        for flag in sorted(flag_set):
            if flag == 0:
                continue
            where_flagged = data.nc_dset['flag'][:] == flag
            nflag = np.count_nonzero(where_flagged)
            nflag_pcnt = 100 * nflag / nspec
            nflag_tot += nflag
            flagged_var_name = list(data.nc_dset['flagged_var_name'][where_flagged])[0]
            print('{:>3}  {:<20} {:>6}   {:>8.3f}'.format(flag, flagged_var_name, nflag, nflag_pcnt))
            flag_df[flagged_var_name] = pd.Series([nflag])
            flag_df_pcnt[flagged_var_name] = pd.Series([nflag_pcnt])

        print('     {:<20} {:>6}   {:>8.3f}'.format('TOTAL', nflag_tot, 100 * nflag_tot / nspec))
        flag_df['Total'] = nflag_tot
        flag_df_pcnt['Total'] = 100 * nflag_tot / nspec

        # For the plot data, we only want to show flags that remove a significant percentage of
        # data. The exact percentage is configurable.
        drop_list = [var for var in flag_df_pcnt if flag_df_pcnt[var][0] < self.min_percent]
        flag_df = flag_df.drop(columns=drop_list)
        flag_df_pcnt = flag_df_pcnt.drop(columns=drop_list)

        return {'counts': flag_df, 'percentages': flag_df_pcnt}

    def get_save_name(self):
        return 'flags_summary.png'

    def make_plot(self, data: Sequence[TcconData], flag0_only: bool = False, show_all: bool = False,
                  img_path: Path = DEFAULT_IMG_DIR, tight=True):
        # The flag plot is only configured to show the current data
        data = [self._get_main_data(data)]
        return super().make_plot(data, flag0_only=flag0_only, show_all=show_all, img_path=img_path, tight=tight)

    def _plot(self, data: TcconData, idata: int, axs=None, flag0_only: bool = False):
        # There should only ever be one set of plot arguments, since we overrode the get_plot_args
        # method to ensure that
        plot_args = self.get_plot_args(data, flag0_only=False)
        assert len(plot_args) == 1
        plot_args = plot_args[0]

        # The counts for the number of spectra flagged go on top, the percent flagged below
        flag_df = plot_args['data']['counts']
        flag_df_pcnt = plot_args['data']['percentages']
        barplot = flag_df.plot(kind='bar', ax=axs[0], **plot_args['kws'])
        barplot_pcnt = flag_df_pcnt.plot(kind='bar', ax=axs[1], **plot_args['kws'])

        # Annotate the bars with the exact counts/percentages
        formats = ['{:.0f}', '{:.2f}']
        for i, curplot in enumerate([barplot, barplot_pcnt]):
            for p in curplot.patches:
                label = formats[i].format(p.get_height())
                x = p.get_x() * 1.005
                y = p.get_height() * 1.005
                curplot.annotate(label, (x, y))

        axs[0].legend(**plot_args['legend_kws'])
        # Plotting with Pandas automatically adds a legend, so we need to remove it from the second axes
        axs[1].get_legend().remove()
        
        
class TimingErrorAbstractPlot(AbstractPlot, TimeseriesMixin, ABC):
    plot_kind = None
    _title_prefix = ''

    def __init__(self, other_plots, default_style, sza_ranges: Sequence[Sequence[float]], limits: Limits,
                 yvar='xluft', freq='W', op='median', time_buffer_days=2, key=None, width=20, height=10,
                 legend_kws: Optional[dict] = None):
        if any(len(r) != 2 for r in sza_ranges):
            raise TypeError('Each SZA range must be a two-element sequence')
        if any(r[1] < r[0] for r in sza_ranges):
            raise TypeError('Each SZA range must have the smaller value first')

        super().__init__(other_plots=other_plots, default_style=default_style, limits=limits, key=key, 
                         width=width, height=height, legend_kws=legend_kws)
        self.sza_ranges = sza_ranges
        self.yvar = yvar
        self.freq = freq
        self.op = op
        self._time_buffer_days = time_buffer_days

    def make_plot(self, data: Sequence[TcconData], flag0_only: bool = False, show_all: bool = False,
                  img_path: Path = DEFAULT_IMG_DIR, tight=True):
        data = self._get_main_data(data)
        return super().make_plot([data], flag0_only=flag0_only, show_all=show_all, img_path=img_path, tight=tight)
        
    def setup_figure(self, data: Sequence[TcconData], show_all: bool = False, fig=None, axs=None):
        fig, ax = self._make_or_check_fig_ax_args(fig, axs)
        freq_op_str = utils.freq_op_str(self.freq, self.op)
        ax.set_title(f'{self._title_prefix} ({freq_op_str})')

        main_data = self._get_main_data(data)
        ax.set_ylabel(self.get_label_string(main_data, self.yvar))

        data_for_limits = self._get_data_for_limits(data)
        # We will override the x limits in the plotting function, since that knows the resampled frequency
        self.format_time_axis(ax, data_for_limits, show_all=show_all)
        if not show_all:
            ymin, ymax, yscale = self._limits.get_limit(self.yvar, data_for_limits, self.plot_kind)
            ax.set_ylim(ymin, ymax, auto=yscale)
        ax.grid(True)

        return fig, ax

    def get_plot_args(self, data: TcconData, flag0_only: bool = False):
        if flag0_only:
            return [{
                'data': self.get_plot_data(data, FlagCategory.FLAG0),
                'kws': self.get_plot_kws(data, FlagCategory.FLAG0),
                'legend_kws': self.get_legend_kws()
            }]
        else:
            return [{
                'data': self.get_plot_data(data, FlagCategory.ALL_DATA),
                'kws': self.get_plot_kws(data, FlagCategory.ALL_DATA),
                'legend_kws': self.get_legend_kws()
            }]

    def get_plot_data(self, data: TcconData, flag_category: FlagCategory):
        # NB: unlike other plots, we need actual datetime types for time to
        # permit resampling to work.
        if flag_category is None:
            data = {
                'time': data.get_flag0_or_all_data('datetime'),
                'solzen': data.get_flag0_or_all_data('solzen'),
                'azim': data.get_flag0_or_all_data('azim'),
                'y': data.get_flag0_or_all_data(self.yvar),
            }
        else:
            data = {
                'time': data.get_data('datetime', flag_category),
                'solzen': data.get_data('solzen', flag_category),
                'azim': data.get_data('azim', flag_category),
                'y': data.get_data(self.yvar, flag_category)
            }

        return pd.DataFrame(data)


class TimingErrorAMvsPM(TimingErrorAbstractPlot):
    plot_kind = 'timing-error-am-pm'
    _title_prefix = 'Timing check AM vs. PM'

    def __init__(self, other_plots, default_style, sza_range: Sequence[float], limits: Limits,
                 yvar='xluft', freq='W', op='median', time_buffer_days=2, key=None, width=20, height=10,
                 legend_kws: Optional[dict] = None):
        if len(sza_range) != 2:
            raise TypeError('SZA range must have two elements (min, max)')

        super().__init__(other_plots=other_plots, default_style=default_style, limits=limits, key=key,
                         width=width, height=height, legend_kws=legend_kws, sza_ranges=[sza_range], yvar=yvar,
                         freq=freq, op=op, time_buffer_days=time_buffer_days)

    def get_save_name(self):
        return 'timing_error_check_{y}_{freq}_{op}_AM_PM_sza_{ll}_to_{ul}_VS_time.png'.format(
            y=self.yvar, freq=self.freq, op=self.op, ll=self.sza_ranges[0][0], ul=self.sza_ranges[0][1]
        )

    def _resample_data(self, df):
        # Before we plot, we need to subset the data to the SZA range requested and to morning/evening
        sza_min, sza_max = self.sza_ranges[0]
        xx_sza = (df['solzen'] >= sza_min) & (df['solzen'] <= sza_max)
        xx_am = df['azim'] <= 180
        xx_pm = df['azim'] > 180

        # Also need to make an additional column that converts the Matplotlib date numbers
        # back into actual time stamps

        df_am = df.loc[xx_sza & xx_am, :].set_index('time').resample(self.freq)
        df_am = getattr(df_am, self.op)()  # this gets the method named `self.op` and calls it
        df_am.index.freq = None  # not sure if this needed, was added to try to deal with issues setting xlimits

        df_pm = df.loc[xx_sza & xx_pm, :].set_index('time').resample(self.freq)
        df_pm = getattr(df_pm, self.op)()
        df_pm.index.freq = None

        return df_am, df_pm

    def get_plot_kws(self, data: TcconData, flag_category: Optional[FlagCategory],  _format_label: bool = True) -> dict:
        # AM/PM plots have their styles organized differently since they plot either flag0 or all data but not both
        # The default style may be specified for "both" (AM and PM), or just AM/PM. The more specific styles override
        # the less specific ones
        default_both = self._get_style(self._default_style, self.plot_kind, 'both')
        style_am = deepcopy(default_both)
        style_am.update(self._get_style(self._default_style, self.plot_kind, 'am'))
        style_pm = deepcopy(default_both)
        style_pm.update(self._get_style(self._default_style, self.plot_kind, 'pm'))

        specific_both = self._get_style(data.styles, self.plot_kind, 'both')

        style_am.update(specific_both)
        style_am.update(self._get_style(data.styles, self.plot_kind, 'am'))

        style_pm.update(specific_both)
        style_pm.update(self._get_style(data.styles, self.plot_kind, 'pm'))

        # Since labels need some extra logic to format, add it separately here
        if _format_label:
            data_label = data.get_flag0_or_all_label() if flag_category is None else data.get_label(flag_category)
            sza_min, sza_max = self.sza_ranges[0]
            am_label_spec = style_am.get('label', r'{data} AM SZA $\in [{ll}, {ul}]$')
            style_am['label'] = am_label_spec.format(data=data_label, ll=sza_min, ul=sza_max)
            pm_label_spec = style_pm.get('label', r'{data} PM SZA $\in [{ll}, {ul}]$')
            style_pm['label'] = pm_label_spec.format(data=data_label, ll=sza_min, ul=sza_max)

        return {'am': style_am, 'pm': style_pm}

    def _plot(self, data: TcconData, idata: int, axs=None, flag0_only: bool = False):
        plot_args = self.get_plot_args(data, flag0_only=flag0_only)
        assert len(plot_args) == 1
        plot_args = plot_args[0]

        df = plot_args['data']
        morning_kws = plot_args['kws']['am']
        afternoon_kws = plot_args['kws']['pm']

        df_am, df_pm = self._resample_data(df)
        xmin = min(df_am.index.min(), df_pm.index.min()) - pd.Timedelta(days=self._time_buffer_days)
        xmax = max(df_am.index.max(), df_pm.index.max()) + pd.Timedelta(days=self._time_buffer_days)
        axs.set_xlim(xmin, xmax)

        # Using df_am['y'].plot() causes weird behavior where I couldn't set the xlimits to be what I wanted
        # Calling the matplotlib plotting methods seems to avoid that behavior
        if df_am.shape[0] > 0:
            axs.plot(df_am.index, df_am['y'], **morning_kws)
        if df_pm.shape[0] > 0:
            axs.plot(df_pm.index, df_pm['y'], **afternoon_kws)

        axs.legend(**plot_args['legend_kws'])


class TimingErrorMultipleSZAs(TimingErrorAbstractPlot):
    plot_kind = 'timing-error-szas'
    _title_prefix = 'Timing check multiple SZAs'

    def __init__(self, other_plots, default_style, sza_ranges: Sequence[Sequence[float]], limits: Limits,
                 am_or_pm, yvar='xluft', freq='W', op='median', time_buffer_days=2, key=None, width=20, height=10,
                 legend_kws: Optional[dict] = None):

        if am_or_pm.lower() not in {'am', 'pm'}:
            raise TypeError('Allows values for am_or_pm are "am" or "pm" only')

        super().__init__(other_plots=other_plots, default_style=default_style, limits=limits, key=key,
                         width=width, height=height, legend_kws=legend_kws, sza_ranges=sza_ranges, yvar=yvar,
                         freq=freq, op=op, time_buffer_days=time_buffer_days)
        self.am_or_pm = am_or_pm

    def get_save_name(self):
        szas = '_'.join('{}to{}'.format(*r) for r in self.sza_ranges)
        return 'timing_error_check_{y}_{freq}_{op}_AM_PM_sza_{szas}_VS_time.png'.format(
            y=self.yvar, freq=self.freq, op=self.op, szas=szas
        )

    def _resample_data(self, df):
        dfs_out = []
        for sza_min, sza_max in self.sza_ranges:
            # Before we plot, we need to subset the data to the SZA range requested and to morning/evening
            xx_sza = (df['solzen'] >= sza_min) & (df['solzen'] <= sza_max)
            xx_tod = df['azim'] <= 180 if self.am_or_pm == 'am' else df['azim'] > 180
            sub_df = df.loc[xx_sza & xx_tod].set_index('time').resample(self.freq)
            sub_df = getattr(sub_df, self.op)()   # this gets the method named `self.op` and calls it
            dfs_out.append(sub_df)

        return dfs_out

    def get_plot_kws(self, data: TcconData, flag_category: Optional[FlagCategory],  _format_label: bool = True) -> list:
        # This
        style = deepcopy(self._default_style.get(self.plot_kind, dict()))
        style.update(data.styles.get(self.plot_kind, dict()))

        # Now check that every style option is either a scalar value (which should be used for
        # every SZA range) or a list/tuple with sufficient length. If it is too short, repeat it
        nranges = len(self.sza_ranges)
        for k, v in style.items():
            if isinstance(v, (list, tuple)):
                nrepeats = ceil(len(v) / nranges)
                style[k] *= nrepeats
            else:
                style[k] = nranges * [v]

        # Next set up the labels, using the data's label plus the SZA range
        # If there was a label in the user specified options, it has already been
        # duplicated
        if _format_label:
            data_label = data.get_flag0_or_all_label() if flag_category is None else data.get_label(flag_category)
            label_spec = style.get('label', [r'{data} SZA $\in [{ll}, {ul}]$'] * nranges)
            labels = [ls.format(data=data_label, ll=r[0], ul=r[1]) for ls, r in zip(label_spec, self.sza_ranges)]
            style['label'] = labels

        # Finally reorganize from a dict of lists to a list of dicts
        styles_out = []
        for i in range(len(self.sza_ranges)):
            styles_out.append({k: v[i] for k, v in style.items()})

        return styles_out

    def _plot(self, data: TcconData, idata: int, axs=None, flag0_only: bool = False):
        plot_args = self.get_plot_args(data, flag0_only=flag0_only)
        assert len(plot_args) == 1
        plot_args = plot_args[0]

        dfs = self._resample_data(plot_args['data'])
        kws = plot_args['kws']

        xmin = min(df.index.min() for df in dfs) - pd.Timedelta(days=self._time_buffer_days)
        xmax = max(df.index.max() for df in dfs) + pd.Timedelta(days=self._time_buffer_days)
        axs.set_xlim(xmin, xmax)

        # Using df_am['y'].plot() causes weird behavior where I couldn't set the xlimits to be what I wanted
        # Calling the matplotlib plotting methods seems to avoid that behavior
        for df, kw in zip(dfs, kws):
            if df.shape[0] == 0:
                continue
            axs.plot(df.index, df['y'], **kw)

        axs.legend(**plot_args['legend_kws'])


class ScatterPlot(AbstractPlot):
    plot_kind = 'scatter'

    def __init__(self, other_plots, xvar, yvar, default_style, limits: Limits, key=None, width=20, height=10,
                 legend_kws: Optional[dict] = None,
                 add_fit: bool = True, fit_flag_category: Optional[FlagCategory] = None, match_axes_size=None):
        super().__init__(other_plots=other_plots, default_style=default_style, key=key, limits=limits,
                         width=width, height=height, legend_kws=legend_kws)
        self.xvar = xvar
        self.yvar = yvar
        self._do_add_fit = add_fit
        self._fit_flag_category = None if fit_flag_category is None else FlagCategory(fit_flag_category)
        self._match_axes_size = match_axes_size

    def get_plot_data(self, data: TcconData, flag_category: Optional[FlagCategory]) -> dict:
        if flag_category is None:
            x = data.get_flag0_or_all_data(self.xvar)
            y = data.get_flag0_or_all_data(self.yvar)
        else:
            x = data.get_data(self.xvar, flag_category)
            y = data.get_data(self.yvar, flag_category)
        return {'x': x, 'y': y}

    def get_plot_kws(self, data: TcconData, flag_category: Optional[FlagCategory], _format_label: bool = True) -> dict:
        kws = super().get_plot_kws(data, flag_category, _format_label=_format_label)
        # For scatter plots, we want there to be no connecting lines by default - don't make the user
        # provide that for every plot style in the config. Also set a default marker so that we avoid
        # mysterious empty plots
        kws.setdefault('linestyle', 'none')
        kws.setdefault('marker', '.')
        return kws

    def get_save_name(self):
        return f'{self.yvar}_VS_{self.xvar}_scatter.png'

    def setup_figure(self, data: Sequence[TcconData], show_all=False, fig=None, axs=None):
        # Find the main data to use for most things; if not present, default to the first data
        main_data = self._get_main_data(data)

        if (fig is None) != (axs is None):
            raise TypeError('Must give both or neither of fig and axs')
        elif fig is None and axs is None:
            size = utils.cm2inch(self._width, self._height)
            if self._match_axes_size is not None:
                plot_to_match = self.get_plot_by_key(self._match_axes_size)
                gs_kws = plot_to_match.get_gridspec_kws(data)
                ncol = len(gs_kws['width_ratios'])
                fig, axs = plt.subplots(1, ncol, figsize=size, gridspec_kw=gs_kws)
                for ax in axs[1:]:
                    ax.axis('off')
                ax = axs[0]
            else:
                fig, ax = plt.subplots(figsize=size)
        else:
            ax = axs
        ax.grid(True)

        ax.set_xlabel(self.get_label_string(main_data, self.xvar))
        ax.set_ylabel(self.get_label_string(main_data, self.yvar))

        if not show_all:
            # Reference data should not affect the limits
            data_for_limits = self._get_data_for_limits(data)
            xmin, xmax, xscale = self._limits.get_limit(self.xvar, data_for_limits, self.plot_kind)
            ax.set_xlim(xmin, xmax, auto=xscale)
            ymin, ymax, yscale = self._limits.get_limit(self.yvar, data_for_limits, self.plot_kind)
            ax.set_ylim(ymin, ymax, auto=yscale)

        ax.set_title(f'{self.yvar} vs. {self.xvar}')

        return fig, ax

    def _plot(self, data: TcconData, idata: int, axs=None, flag0_only: bool = False):
        plot_args = self.get_plot_args(data, flag0_only=flag0_only)
        for args in plot_args:
            # Ignore fit_style or legend_fontsize keywords
            args['kws'].pop('fit_style', None)
            args['kws'].pop('legend_fontsize', None)  # use a small default so the text fits on the plot

            axs.plot(args['data']['x'], args['data']['y'], **args['kws'])
        self.add_qc_lines(axs, 'x', data.nc_dset[self.xvar])
        self.add_qc_lines(axs, 'y', data.nc_dset[self.yvar])

        if self._do_add_fit:
            self._add_linfit(axs, data, flag0_only)

        # legend keywords are only ever in the first set of plot args
        axs.legend(**plot_args[0]['legend_kws'])

    def _add_linfit(self, ax, data: TcconData, flag0_only: bool = False):
        fit_flag_category = self._get_flag_category(self._fit_flag_category, flag0_only)
        fit_data = self.get_plot_data(data, fit_flag_category)
        fit_flag_kws = self.get_plot_kws(data, fit_flag_category)
        fit_style = fit_flag_kws.get('fit_style', dict())

        fit_label = fit_style.get('label', 'Fit to {data} data\n{fit}')
        data_label = data.get_flag0_or_all_label() if fit_flag_category is None else data.get_label(fit_flag_category)
        fit_label = utils.preformat_string(fit_label, data=data_label)
        fit_style['label'] = fit_label

        fit_style.setdefault('linestyle', ':')
        fit_style.setdefault('color', 'C1')

        utils.add_linfit(ax, fit_data['x'], fit_data['y'], **fit_style)


class HexbinPlot(ScatterPlot):
    plot_kind = 'hexbin'

    def __init__(self, other_plots, xvar, yvar, default_style, limits: Limits, key=None, width=20, height=10,
                 legend_kws: Optional[dict] = None,
                 hexbin_flag_category=None, fit_flag_category=None, show_reference=False, show_context=True):
        super().__init__(other_plots=other_plots, xvar=xvar, yvar=yvar, default_style=default_style, limits=limits,
                         legend_kws=legend_kws, fit_flag_category=fit_flag_category, key=key,
                         width=width, height=height)
        self._hexbin_flag_category = None if hexbin_flag_category is None else FlagCategory(hexbin_flag_category)
        self._show_ref = show_reference
        self._show_context = show_context
        self._caxes = []

    def setup_figure(self, data: Sequence[TcconData], show_all=False, fig=None, axs=None):
        size = utils.cm2inch(self._width, self._height)
        # Figure out how many colorbars we need.  This depends on what data is present and whether
        # the user told us to show it.
        n_cb = len(data)
        gridspec_kws = self.get_gridspec_kws(data, data_prelimited=True)
        fig, axs = plt.subplots(1, 1 + n_cb, gridspec_kw=gridspec_kws, figsize=size)
        self._caxes = axs[1:]
        return super().setup_figure(data, show_all=show_all, fig=fig, axs=axs[0])

    def get_save_name(self):
        return f'{self.yvar}_VS_{self.xvar}_hexbin.png'

    def make_plot(self, data: Sequence[TcconData], flag0_only: bool = False, show_all: bool = False,
                  img_path: Path = DEFAULT_IMG_DIR, tight=True):
        data_to_plot = self._get_data_to_plot(data)
        return super().make_plot(data=data_to_plot, flag0_only=flag0_only, show_all=show_all, img_path=img_path,
                                 tight=tight)

    def _plot(self, data: TcconData, idata: int, axs=None, flag0_only: bool = False):
        plot_args = self.get_plot_args(data, flag0_only=flag0_only)
        flag_cat = self._get_flag_category(self._hexbin_flag_category, flag0_only)
        clabel = data.get_flag0_or_all_label() if flag_cat is None else data.get_label(flag_cat)

        assert len(plot_args) == 1
        args = plot_args[0]

        # Best to compute extent now, when we have the actual data to plot
        args['kws'].setdefault('extent', self._compute_extent(data=data, **args['data']))

        # Also need to extract the "fit_style" if present, because that isn't passed to hexbin
        # Same for "legend_fontsize"
        args['kws'].pop('fit_style', None)

        h = axs.hexbin(args['data']['x'], args['data']['y'], **args['kws'])
        plt.colorbar(h, cax=self._caxes[idata], label=clabel)

        self._add_linfit(axs, data, flag0_only)

        self.add_qc_lines(axs, 'x', data.nc_dset[self.xvar])
        self.add_qc_lines(axs, 'y', data.nc_dset[self.yvar])
        axs.legend(**args['legend_kws'])

    def get_plot_args(self, data: TcconData, flag0_only: bool = False):
        # For hexbin, this needs overridden. We only ever have one set of plot arguments,
        # which either plots flag0 data or all data, depending on the user flags.
        return self._get_plot_args_mono(data, self._hexbin_flag_category, flag0_only=flag0_only)

    def get_plot_kws(self, data: TcconData, flag_category: Optional[FlagCategory], _format_label: bool = True) -> dict:
        kws = super().get_plot_kws(data, flag_category, _format_label=_format_label)
        # Remove linestyle, marker (from ScatterPlot), and label (from AbstractPlot). None are used for hexbin plots.
        kws.pop('linestyle')
        kws.pop('marker')
        kws.pop('label')

        return kws

    def get_gridspec_kws(self, data, data_prelimited=False):
        if data_prelimited:
            n_cb = len(data)
        else:
            n_cb = len(self._get_data_to_plot(data))
        return utils.colorbar_gridspec_kws(n_cb)

    def _get_data_to_plot(self, data: Sequence[TcconData]):
        data_to_plot = []
        for d in data:
            if d.data_category == DataCategory.REFERENCE and self._show_ref:
                data_to_plot.append(d)
            elif d.data_category == DataCategory.CONTEXT and self._show_context:
                data_to_plot.append(d)
            elif d.data_category not in {DataCategory.REFERENCE, DataCategory.CONTEXT}:
                data_to_plot.append(d)

        return data_to_plot

    def _compute_extent(self, x, y, data):
        def get_limits_inner(varname, xy):
            # autoscaling is irrelevant for the hexbin plots - if we don't have a priori
            # limits, we calculate them from the data
            limmin, limmax, _ = self._limits.get_limit(varname, data)
            lims = [limmin, limmax]
            if None in lims:
                if isinstance(xy, np.ma.masked_array):
                    xy = xy.filled(np.nan)
                lims = [np.nanmin(xy), np.nanmax(xy)]
            return lims

        xmin, xmax = get_limits_inner(self.xvar, x)
        ymin, ymax = get_limits_inner(self.yvar, y)
        return xmin, xmax, ymin, ymax


class TimeseriesPlot(ScatterPlot, TimeseriesMixin):
    plot_kind = 'timeseries'

    def __init__(self, other_plots, yvar, default_style, limits: Limits, width=20, height=10,
                 legend_kws: Optional[dict] = None, key=None, time_buffer_days=2):
        super().__init__(other_plots=other_plots, xvar='time', yvar=yvar, default_style=default_style, limits=limits,
                         key=key, width=width, height=height, legend_kws=legend_kws, add_fit=False)
        self._time_buffer_days = time_buffer_days

    def setup_figure(self, data: Sequence[TcconData], show_all=False, fig=None, axs=None):
        fig, ax = super(TimeseriesPlot, self).setup_figure(data=data, show_all=show_all, fig=fig, axs=axs)
        # self._format_time_axis(ax, data, show_all=show_all)
        self.format_time_axis(ax, self._get_data_for_limits(data), xvar=self.xvar, buffer_days=self._time_buffer_days,
                              show_all=show_all)
        ax.set_title(f'{self.yvar} time series')
        return fig, ax

    def get_save_name(self):
        return f'{self.yvar}_timeseries.png'

    def _plot(self, data: TcconData, idata: int, axs=None, flag0_only: bool = False):
        plot_args = self.get_plot_args(data, flag0_only=flag0_only)
        for args in plot_args:
            args['kws'].pop('fit_style', None)
            axs.plot(args['data']['x'], args['data']['y'], **args['kws'])
        self.add_qc_lines(axs, 'y', data.nc_dset[self.yvar])

        # legend kws are always in the first set of plot argument
        axs.legend(**plot_args[0]['legend_kws'])


class TimeseriesDeltaPlot(TimeseriesPlot):
    plot_kind = 'delta-timeseries'

    def __init__(self, other_plots, yvar1, yvar2, default_style, limits: Limits, width=20, height=10,
                 legend_kws: Optional[dict] = None, key=None, time_buffer_days=2):
        super().__init__(other_plots=other_plots, yvar=yvar1, default_style=default_style, limits=limits,
                         key=key, width=width, height=height, legend_kws=legend_kws, time_buffer_days=time_buffer_days)
        self.yvar2 = yvar2

    def setup_figure(self, data: Sequence[TcconData], show_all=False, fig=None, axs=None):
        fig, ax = super().setup_figure(data=data, show_all=show_all, fig=fig, axs=axs)
        main_data = self._get_main_data(data)
        yunits = self.get_units(main_data, self.yvar)
        yunits2 = self.get_units(main_data, self.yvar2)
        if yunits == yunits2:
            ylabel = f'{self.yvar} - {self.yvar2} ({yunits})'
        else:
            print(f'\nWARNING: differencing variables ({self.yvar}, {self.yvar2}) with different units ({yunits}, {yunits2})', file=sys.stderr)
            ylabel = f'{self.yvar} - {self.yvar2} ({yunits} - {yunits2})'
        ax.set_ylabel(ylabel)
        return fig, ax

    def get_save_name(self):
        return f'{self.yvar}_MINUS_{self.yvar2}_timeseries.png'

    def get_plot_data(self, data: TcconData, flag_category: Optional[FlagCategory]) -> dict:
        if flag_category is None:
            x = data.get_flag0_or_all_data(self.xvar)
            y1 = data.get_flag0_or_all_data(self.yvar)
            y2 = data.get_flag0_or_all_data(self.yvar2)
        else:
            x = data.get_data(self.xvar, flag_category)
            y1 = data.get_data(self.yvar, flag_category)
            y2 = data.get_data(self.yvar2, flag_category)
        return {'x': x, 'y': y1 - y2}


class Timeseries2PanelPlot(TimeseriesPlot):
    plot_kind = 'timeseries-2panel'

    def __init__(self, other_plots, yvar, yerror_var, default_style, limits: Limits, key=None, width=20, height=10,
                 legend_kws: Optional[dict] = None, time_buffer_days=2):
        super().__init__(other_plots=other_plots, yvar=yvar, default_style=default_style, limits=limits,
                         key=key, width=width, height=height, time_buffer_days=time_buffer_days, legend_kws=legend_kws)
        self.yerror_var = yerror_var

    def setup_figure(self, data: Sequence[TcconData], show_all=False):
        main_data = self._get_main_data(data)

        size = utils.cm2inch(self._width, self._height)
        fig, axs = plt.subplots(2, 1, figsize=size, sharex='all', gridspec_kw={'height_ratios': [1, 3]})
        axs = {'main': axs[1], 'error': axs[0]}
        for ax in axs.values():
            ax.grid(True)

        axs['main'].set_xlabel(self.get_label_string(main_data, self.xvar))
        axs['main'].set_ylabel(self.get_label_string(main_data, self.yvar))
        axs['error'].set_ylabel(self.get_label_string(main_data, self.yerror_var))

        if not show_all:
            # Reference data should not affect the limits
            data_for_limits = self._get_data_for_limits(data)
            ymin, ymax, yscale = self._limits.get_limit(self.yvar, data_for_limits, self.plot_kind)
            axs['main'].set_ylim(ymin, ymax, auto=yscale)
            ymin2, ymax2, yscale2 = self._limits.get_limit(self.yerror_var, data_for_limits, self.plot_kind)
            axs['error'].set_ylim(ymin2, ymax2, auto=yscale2)

        # For whatever reason, both axes need their format set for the ticks to behave properly
        for ax in axs.values():
            self.format_time_axis(ax, self._get_data_for_limits(data), xvar=self.xvar,
                                  buffer_days=self._time_buffer_days, show_all=show_all)

        # Remove the x-axis label from the top axes
        axs['error'].set_xlabel('')

        fig.suptitle(f'{self.yvar} and {self.yerror_var} time series')
        return fig, axs

    def get_save_name(self):
        return f'{self.yvar}_timeseries_two_panel.png'

    def get_plot_data(self, data: TcconData, flag_category: Optional[FlagCategory]) -> dict:
        if flag_category is None:
            x = data.get_flag0_or_all_data(self.xvar)
            y = data.get_flag0_or_all_data(self.yvar)
            yerr = data.get_flag0_or_all_data(self.yerror_var)
        else:
            x = data.get_data(self.xvar, flag_category)
            y = data.get_data(self.yvar, flag_category)
            yerr = data.get_data(self.yerror_var, flag_category)
        return {'x': x, 'y': y, 'yerr': yerr}

    def _plot(self, data: TcconData, idata: int, axs=None, flag0_only: bool = False):
        plot_args = self.get_plot_args(data, flag0_only=flag0_only)
        for args in plot_args:
            args['kws'].pop('fit_style', None)  # can't leave fit style in if present
            axs['main'].plot(args['data']['x'], args['data']['y'], **args['kws'])
            axs['error'].plot(args['data']['x'], args['data']['yerr'], **args['kws'])

        self.add_qc_lines(axs['main'], 'y', data.nc_dset[self.yvar])
        self.add_qc_lines(axs['error'], 'y', data.nc_dset[self.yerror_var])

        # Legend keywords are always in the first set of plot args
        axs['main'].legend(**plot_args[0]['legend_kws'])


class ResampledTimeseriesPlot(TimeseriesPlot):
    plot_kind = 'resampled-timeseries'

    def __init__(self, other_plots, yvar, freq, op, default_style, limits: Limits, key=None, width=20, height=10,
                 legend_kws: Optional[dict] = None, time_buffer_days=2):
        super().__init__(other_plots=other_plots, yvar=yvar, default_style=default_style, limits=limits,
                         key=key, width=width, height=height, legend_kws=legend_kws, time_buffer_days=time_buffer_days)
        self.freq = freq
        self.op = op

    def setup_figure(self, data: Sequence[TcconData], show_all=False, fig=None, axs=None):
        fig, axs = super().setup_figure(data, show_all, fig, axs)
        freq_op_str = utils.freq_op_str(self.freq, self.op)
        axs.set_title(f'{self.yvar} {freq_op_str} time series')
        return fig, axs

    def get_plot_data(self, data: TcconData, flag_category: Optional[FlagCategory]) -> dict:
        if flag_category is None:
            x = data.get_flag0_or_all_data('datetime')
            y = data.get_flag0_or_all_data(self.yvar)
        else:
            x = data.get_data('datetime', flag_category)
            y = data.get_data(self.yvar, flag_category)

        # Now we need to resample the data to the specified frequency before returning it
        y_series = self.resample_data(x, y, self.freq, self.op)
        return {'x': y_series.index, 'y': y_series.to_numpy()}

    def get_save_name(self):
        return f'{self.yvar}_{self.freq}_{self.op}_timeseries.png'


class RollingDerivativePlot(TimeseriesPlot):
    plot_kind = 'rolling-derivative'

    def __init__(self, other_plots, yvar, dvar, default_style, limits: Limits, key=None, width=20, height=10,
                 legend_kws: Optional[dict] = None, derivative_order=1,
                 gap='20000 days', rolling_window=500, flag_category=None, time_buffer_days=2):
        super().__init__(other_plots=other_plots, yvar=yvar, default_style=default_style, limits=limits,
                         legend_kws=legend_kws, key=key, width=width, height=height, time_buffer_days=time_buffer_days)

        self.dvar = dvar
        self.derivative_order = derivative_order
        self.gap = gap
        self.rolling_window = rolling_window
        self.flag_category = None if flag_category is None else FlagCategory(flag_category)

    def _derivative_str(self, latex=True) -> str:
        n = self.derivative_order
        n = f'^{n}' if n != 1 else ''
        s = f'$d{n}${self.yvar}/$d${self.dvar}${n}$'.replace('$$', '')  # double $$ occur is n is an empty string and cause rendering problems
        if latex:
            return s
        else:
            return s.replace('$', '').replace('^', '')

    def setup_figure(self, data: Sequence[TcconData], show_all=False, fig=None, axs=None):
        fig, axs = super().setup_figure(data, show_all, fig, axs)
        main_data = self._get_main_data(data)
        yunits = self.get_units(main_data, self.yvar)
        dunits = self.get_units(main_data, self.dvar)
        axs.set_ylabel(f'{self.yvar}/{self.dvar} slope ({yunits}/{dunits})')

        deriv_str = self._derivative_str()
        axs.set_title(f'Rolling {deriv_str} timeseries')
        return fig, axs

    def get_plot_args(self, data: TcconData, flag0_only: bool = False):
        def roll(x, y, t):
            df = pd.DataFrame({'x': x, 'y': y, 't': t})
            all_derivatives = []
            for _, grouped_df in self.split_by_gaps(df, self.gap, 't'):
                derivatives = np.full(grouped_df.shape[0], np.nan)

                i = -1
                for roll_df in grouped_df.rolling(self.rolling_window, center=True, min_periods=1):
                    i += 1
                    if self.derivative_order == 1:
                        fit, _, _ = utils.compute_linfit(roll_df['x'].to_numpy(), roll_df['y'].to_numpy())
                        derivatives[i] = fit[0]  # fit has slope then intercept
                    else:
                        raise NotImplementedError('Only derivatives of order 1 implemented')

                all_derivatives.append(pd.DataFrame({'x': grouped_df['t'], 'y': derivatives}))

            return pd.concat(all_derivatives)

        flag_category = self._get_flag_category(self.flag_category, flag0_only)
        raw_data_vals = self.get_plot_data(data, flag_category)
        rolled_derivatives = roll(raw_data_vals['x'], raw_data_vals['y'], raw_data_vals['t'])
        rolled_data = {'x': rolled_derivatives['x'].to_numpy(), 'y': rolled_derivatives['y'].to_numpy()}
        data_kws = self.get_plot_kws(data, flag_category)
        legend_kws = self.get_legend_kws()

        return [{'data': rolled_data, 'kws': data_kws, 'legend_kws': legend_kws}]

    def get_plot_data(self, data: TcconData, flag_category: Optional[FlagCategory]) -> dict:
        if flag_category is None:
            x = data.get_flag0_or_all_data(self.dvar)
            y = data.get_flag0_or_all_data(self.yvar)
            t = data.get_flag0_or_all_data('datetime')
        else:
            x = data.get_data(self.dvar, flag_category)
            y = data.get_data(self.yvar, flag_category)
            t = data.get_data('datetime', flag_category)
        return {'x': x, 'y': y, 't': t}

    def get_plot_kws(self, data: TcconData, flag_category: Optional[FlagCategory], _format_label: bool = True) -> dict:
        # Do not let the superclass method format the label because we need to do that here
        kws = super().get_plot_kws(data, flag_category, _format_label=False)
        if _format_label:
            data_label = data.get_flag0_or_all_label() if flag_category is None else data.get_label(flag_category)
            label_spec = kws.get('label', '{n} spectra {data}')
            kws['label'] = label_spec.format(data=data_label, n=self.rolling_window)
        return kws

    def _plot(self, data: TcconData, idata: int, axs=None, flag0_only: bool = False):
        # Needed overridden because we don't want QC limit lines

        plot_args = self.get_plot_args(data, flag0_only=flag0_only)
        for args in plot_args:
            args['kws'].pop('fit_style', None)
            axs.plot(args['data']['x'], args['data']['y'], **args['kws'])

        # legend kws are always in the first set of plot argument
        axs.legend(**plot_args[0]['legend_kws'])

    def get_save_name(self):
        deriv_str = self._derivative_str(latex=False).replace('/', '_')
        return f'{deriv_str}_rolling{self.rolling_window}_timeseries.png'


class RollingTimeseriesPlot(TimeseriesPlot):
    plot_kind = 'rolling-timeseries'

    def __init__(self, other_plots, yvar, ops, default_style, limits: Limits, key=None, width=20, height=10,
                 legend_kws: Optional[dict] = None,
                 gap='20000 days', rolling_window=500, uncertainty=False, flag_category=None, time_buffer_days=2):
        super().__init__(other_plots=other_plots, yvar=yvar, default_style=default_style, limits=limits,
                         legend_kws=legend_kws, key=key, width=width, height=height, time_buffer_days=time_buffer_days)
        self.ops = [ops] if isinstance(ops, str) else ops
        self.gap = gap
        self.rolling_window = rolling_window
        self.uncertainty = uncertainty
        self.flag_category = None if flag_category is None else FlagCategory(flag_category)

    def setup_figure(self, data: Sequence[TcconData], show_all=False, fig=None, axs=None):
        fig, axs = super().setup_figure(data, show_all, fig, axs)
        ops = ', '.join(self.ops)
        axs.set_title(f'{self.yvar} rolling {ops} timeseries')
        return fig, axs

    def get_plot_args(self, data: TcconData, flag0_only: bool = False):
        def roll(x, y, t, o):
            tmp = self.roll_data(x, y, times=t, npts=self.rolling_window, ops=o, gap=self.gap)
            return {'x': tmp['x'].to_numpy(), 'y': tmp['y'].to_numpy()}

        flag_category = self._get_flag_category(self.flag_category, flag0_only)
        raw_data_vals = self.get_plot_data(data, flag_category)
        raw_data_kws = self.get_plot_kws(data, flag_category)

        args = [{'data': raw_data_vals, 'kws': raw_data_kws, 'legend_kws': self.get_legend_kws()}]
        for op in self.ops:
            op_vals = roll(raw_data_vals['x'], raw_data_vals['y'], raw_data_vals['t'], op)
            op_kws = self.get_plot_kws(data, flag_category, op=op)
            args.append({'data': op_vals, 'kws': op_kws})

            if self.uncertainty and op == 'mean':
                dy = roll(raw_data_vals['x'], raw_data_vals['y'], raw_data_vals['t'], 'std')['y']
                unc_kws = self.get_plot_kws(data, flag_category, op='std')
                unc_kws['label'] = 'Uncertainty on mean'
                args.append({'data': {'x': op_vals['x'], 'y': op_vals['y'] + dy}, 'kws': deepcopy(unc_kws)})
                unc_kws['label'] = ''  # don't show the label for the second series
                args.append({'data': {'x': op_vals['x'], 'y': op_vals['y'] - dy}, 'kws': unc_kws})
            elif self.uncertainty and op == 'median':
                uq = roll(raw_data_vals['x'], raw_data_vals['y'], raw_data_vals['t'], 'quantile0.75')
                lq = roll(raw_data_vals['x'], raw_data_vals['y'], raw_data_vals['t'], 'quantile0.25')
                unc_kws = self.get_plot_kws(data, flag_category, 'quantile')
                unc_kws['label'] = 'Uncertainty on median (interquartile range)'
                args.append({'data': uq, 'kws': deepcopy(unc_kws)})
                unc_kws['label'] = ''
                args.append({'data': lq, 'kws': deepcopy(unc_kws)})

        return args
    
    def get_plot_data(self, data: TcconData, flag_category: Optional[FlagCategory]) -> dict:
        if flag_category is None:
            x = data.get_flag0_or_all_data(self.xvar)
            y = data.get_flag0_or_all_data(self.yvar)
            t = data.get_flag0_or_all_data('datetime')
        else:
            x = data.get_data(self.xvar, flag_category)
            y = data.get_data(self.yvar, flag_category)
            t = data.get_data('datetime', flag_category)
        return {'x': x, 'y': y, 't': t}


    def get_plot_kws(self, data: TcconData, flag_category: Optional[FlagCategory], op=None, _format_label: bool = True) -> dict:
        # Get the default style: style dictionaries are organized plot_kind -> flag category,
        # Allow both to be missing, and just provide an empty dictionary as the default
        if op is not None:
            style_fc = op
        elif flag_category is None:
            style_fc = data.default_category
        else:
            style_fc = flag_category

        default = self._get_style(self._default_style, self.plot_kind, style_fc)
        specific = self._get_style(data.styles, self.plot_kind, style_fc)
        if isinstance(style_fc, str) and 'quantile' in style_fc:
            # Quantile operations need to include what quantile to calculate, e.g. "quantile0.75",
            # but the user may have just specified a generic "quantile" style, so if we didn't
            # find styles for the specific quantile to be calculated, try for the generic one.
            default = default if len(default) > 0 else self._get_style(self._default_style, self.plot_kind, 'quantile')
            specific = specific if len(specific) > 0 else self._get_style(data.styles, self.plot_kind, 'quantile')

        # Override options in default with values in the data type-specific keywords
        kws = deepcopy(default)
        kws.update(specific)

        # No connecting lines by default
        kws.setdefault('linestyle', 'none')

        if _format_label:
            # Since labels need some extra logic to format, add it separately here
            data_label = data.get_flag0_or_all_label() if flag_category is None else data.get_label(flag_category)
            label_spec = kws.get('label', '{data}' if op is None else '{n} spectra {op} {data}')
            kws['label'] = label_spec.format(data=data_label, op=op, n=self.rolling_window)
        return kws

    def get_save_name(self):
        ops = '+'.join(self.ops)
        return f'{self.yvar}_rolling{self.rolling_window}_{ops}_timeseries.png'


def setup_plots(config, limits_file=DEFAULT_LIMITS, allow_missing=False):
    plots_config = config['plots']
    plots = []
    for iplot, section in enumerate(plots_config, start=1):
        # the plot kind is popped from the section dictionary to allow it to be passed as keyword args
        # so copy the kind here if needed for the error message
        kind = section.get('kind', '?')

        try:
            plots.append( AbstractPlot.from_config_section(section, plots, full_config=config, limits_file=limits_file) )
        except PlotClassError:
            if not allow_missing:
                raise
        except TypeError as err:
            msg = \
"""Error setting up plot #{i} (kind = "{kind}"):

    {err} 

Check the section for this plot in the variables TOML file for missing required settings or other errors.""".format(
    i=iplot, kind=kind, err=err
)
            raise TypeError(msg)

    return plots
