from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import timedelta, datetime
from enum import Enum
from fnmatch import fnmatch
import matplotlib.pyplot as plt
import netCDF4 as ncdf
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import tomli

from . import utils
from .constants import DEFAULT_LIMITS, DEFAULT_IMG_DIR


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
                 allowed_flag_categories=None):
        self.data_category = data_category
        self.nc_dset = ncdf.Dataset(nc_file)
        self.styles = styles
        self.times = self.nctime_to_pytime(self.nc_dset['time'])

        if exclude_times is not None:
            self.all_idx = (self.times < np.min(exclude_times)) | (self.times > np.max(exclude_times))
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
        return '{} flag==0'.format(site_name)

    def _get_flagged_label(self) -> str:
        site_name = self.nc_dset.long_name
        return '{} flagged'.format(site_name)

    def _get_alldata_label(self) -> str:
        site_name = self.nc_dset.long_name
        return '{} all data'.format(site_name)

    @staticmethod
    def nctime_to_pytime(nc_time_var: ncdf.Variable):
        cftimes = ncdf.num2date(nc_time_var[:], units=nc_time_var.units, calendar=nc_time_var.calendar)
        return np.array([datetime(*t.timetuple()[:6]) for t in cftimes])


class Limits:
    def __init__(self, limits_file):
        with open(limits_file) as f:
            self.limits = tomli.load(f)

    def get_limit(self, varname: str, data: Union[TcconData, Sequence[TcconData]], plot_kind: Optional[str] = None) -> Tuple[float, float]:
        if isinstance(data, TcconData):
            data = [data]

        # Prefer to get the limit from the limits file, and in the file, prefer plot-specific sections
        plot_section = self.limits.get(plot_kind, dict())
        plot_limits = self._get_var_from_section(varname, plot_section)
        if plot_limits is not None:
            return tuple(plot_limits)

        defaults_section = self.limits.get('default', dict())
        default_limits = self._get_var_from_section(varname, defaults_section)
        if default_limits is not None:
            return tuple(default_limits)

        # Finally try getting limits from the variable in the netCDF dataset
        # If those attributes aren't present, default to None (i.e. do not set limits)
        vmins = [getattr(d.nc_dset[varname], 'vmin', None) for d in data]
        vmins = [v for v in vmins if vmins is not None]
        vmaxes = [getattr(d.nc_dset[varname], 'vmax', None) for d in data]
        vmaxes = [v for v in vmaxes if vmaxes is not None]
        vmin = min(vmins) if len(vmins) > 0 else None
        vmax = max(vmaxes) if len(vmaxes) > 0 else None
        return vmin, vmax

    @staticmethod
    def _get_var_from_section(varname: str, section: dict):
        for key in section.keys():
            if fnmatch(varname, key):
                return section[key]

        return None


class AbstractPlot(ABC):
    plot_kind = ''

    def __init__(self, default_style, limits: Limits, width=20, height=10):
        self._default_style = default_style
        self._limits = limits
        self._width = width
        self._height = height

    @classmethod
    def from_config_section(cls, section, full_config, limits_file):
        kind = section.pop('kind')
        klass = cls._dispatch_plot_kind(kind)
        default_styles = full_config.get('style', dict()).get('default', dict())
        limits = Limits(limits_file)
        return klass(default_style=default_styles, limits=limits, **section)

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
            if key in available_classes:
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

    def get_plot_args(self, data: TcconData, flag0_only: bool = False):
        # There will always be at least one thing to plot, whether it is all data or just flag0 data
        plot_args = [{
            'data': self.get_plot_data(data, FlagCategory.FLAG0),
            'kws': self.get_plot_kws(data, FlagCategory.FLAG0)
        }]

        # Whether there is more data to plot depends on whether the user specified that we should only plot
        # flag0 data and whether this particular data cares
        if not flag0_only and data.has_flag_category(FlagCategory.FLAGGED):
            plot_args.append({
                'data': self.get_plot_data(data, FlagCategory.FLAGGED),
                'kws': self.get_plot_kws(data, FlagCategory.FLAGGED)
            })

        return plot_args

    def get_plot_kws(self, data: TcconData, flag_category: Optional[FlagCategory]) -> dict:
        # Get the default style: style dictionaries are organized plot_kind -> flag category,
        # Allow both to be missing, and just provide an empty dictionary as the default
        style_fc = data.default_category if flag_category is None else flag_category
        default = self._get_style(self._default_style, self.plot_kind, style_fc)
        specific = self._get_style(data.styles, self.plot_kind, style_fc)

        # Override options in default with values in the data type-specific keywords
        kws = deepcopy(default)
        kws.update(specific)

        # Since labels need some extra logic to format, add
        kws['label'] = data.get_flag0_or_all_label() if flag_category is None else data.get_label(flag_category)
        return kws

    def _get_style(self, styles, plot_kind, flag_category: FlagCategory):
        plot_styles = styles.get(plot_kind, dict())
        if 'clone' in plot_styles:
            return self._get_style(styles, plot_styles['clone'], flag_category)
        else:
            return plot_styles.get(flag_category.value, dict())

    def make_plot(self, data: Sequence[TcconData], flag0_only: bool = False, show_all: bool = False,
                  img_path: Path = DEFAULT_IMG_DIR, tight=True):
        fig, axs = self.setup_figure(data, show_all=show_all)
        for d in data:
            self._plot(d, axs=axs, flag0_only=flag0_only)
        fig_path = img_path / self.get_save_name()
        if tight:
            fig.tight_layout()
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)
        return fig_path

    def add_qc_lines(self, ax, axis: str, nc_var: ncdf.Variable):
        axline_fxn = ax.axvline if axis == 'x' else ax.axhline
        vmin = getattr(nc_var, 'vmin', None)
        if vmin:
            axline_fxn(vmin, linestyle='--', color='black')
        vmax = getattr(nc_var, 'vmax', None)
        if vmax:
            axline_fxn(vmax, linestyle='--', color='black')

    @abstractmethod
    def setup_figure(self, data: Sequence[TcconData], show_all: bool = False):
        pass

    @abstractmethod
    def get_plot_data(self, data: TcconData, flagged_category: FlagCategory):
        pass

    @abstractmethod
    def get_save_name(self):
        pass

    @abstractmethod
    def _plot(self, data: TcconData, axs=None, flag0_only: bool = False):
        pass


class ScatterPlot(AbstractPlot):
    plot_kind = 'scatter'

    def __init__(self, xvar, yvar, default_style, limits: Limits, width=20, height=10):
        super().__init__(default_style=default_style, limits=limits, width=width, height=height)
        self.xvar = xvar
        self.yvar = yvar

    def get_plot_data(self, data: TcconData, flag_category: Optional[FlagCategory]) -> dict:
        if flag_category is None:
            x = data.get_flag0_or_all_data(self.xvar)
            y = data.get_flag0_or_all_data(self.yvar)
        else:
            x = data.get_data(self.xvar, flag_category)
            y = data.get_data(self.yvar, flag_category)
        return {'x': x, 'y': y}

    def get_plot_kws(self, data: TcconData, flag_category: Optional[FlagCategory]) -> dict:
        kws = super().get_plot_kws(data, flag_category)
        # For scatter plots, we want there to be no connecting lines by default - don't make the user
        # provide that for every plot style in the config
        kws.setdefault('linestyle', 'none')
        return kws

    def get_save_name(self):
        return f'{self.yvar}_VS_{self.xvar}_scatter.png'

    def setup_figure(self, data: Sequence[TcconData], show_all=False):
        # Find the main data to use for most things; if not present, default to the first data
        main_data = [d for d in data if d.data_category == DataCategory.PRIMARY]
        if len(main_data) > 0:
            main_data = main_data[0]
        else:
            main_data = data[0]

        size = utils.cm2inch(self._width, self._height)
        fig, ax = plt.subplots(figsize=size)
        ax.grid(True)

        xunits = getattr(main_data.nc_dset[self.xvar], 'units', None)
        if xunits:
            ax.set_xlabel(f'{self.xvar} ({xunits})')
        else:
            ax.set_xlabel(f'{self.xvar}')

        yunits = getattr(main_data.nc_dset[self.yvar], 'units', None)
        if yunits:
            ax.set_ylabel(f'{self.yvar} ({yunits})')
        else:
            ax.set_ylabel(f'{self.yvar}')

        if not show_all:
            # Reference data should not affect the limits
            data_for_limits = [d for d in data if d.data_category != DataCategory.REFERENCE]
            ax.set_xlim(self._limits.get_limit(self.xvar, data_for_limits, self.plot_kind))
            ax.set_ylim(self._limits.get_limit(self.yvar, data_for_limits, self.plot_kind))

        return fig, ax

    def _plot(self, data: TcconData, axs=None, flag0_only: bool = False):
        plot_args = self.get_plot_args(data, flag0_only=flag0_only)
        for args in plot_args:
            axs.plot(args['data']['x'], args['data']['y'], **args['kws'])
        self.add_qc_lines(axs, 'x', data.nc_dset[self.xvar])
        self.add_qc_lines(axs, 'y', data.nc_dset[self.yvar])
        axs.legend()


class TimeseriesPlot(ScatterPlot):
    plot_kind = 'timeseries'

    def __init__(self, yvar, default_style, limits: Limits, width=20, height=10, time_buffer_days=2):
        super().__init__(xvar='time', yvar=yvar, default_style=default_style, limits=limits, width=width, height=height)
        self._time_buffer_days = time_buffer_days

    def setup_figure(self, data: Sequence[TcconData], show_all=False):
        fig, ax = super(TimeseriesPlot, self).setup_figure(data=data, show_all=show_all)
        # Override the default x axis limits to add some buffer
        if not show_all:
            data_for_limits = [d for d in data if d.data_category != DataCategory.REFERENCE]
            xmin = min(np.min(d.get_data(self.xvar, FlagCategory.ALL_DATA)) for d in data_for_limits) - timedelta(days=self._time_buffer_days)
            xmax = max(np.max(d.get_data(self.xvar, FlagCategory.ALL_DATA)) for d in data_for_limits) + timedelta(days=self._time_buffer_days)
            ax.set_xlim(xmin, xmax)

        # Also override the x-axis label
        ax.set_xlabel('Time')
        return fig, ax

    def get_save_name(self):
        return f'{self.yvar}_timeseries.png'

    def _plot(self, data: TcconData, axs=None, flag0_only: bool = False):
        plot_args = self.get_plot_args(data, flag0_only=flag0_only)
        for args in plot_args:
            axs.plot(args['data']['x'], args['data']['y'], **args['kws'])
        self.add_qc_lines(axs, 'y', data.nc_dset[self.yvar])
        axs.legend()


def setup_plots(config, limits_file=DEFAULT_LIMITS, allow_missing=True):
    plots_config = config['plots']
    plots = []
    for section in plots_config:
        try:
            plots.append( AbstractPlot.from_config_section(section, full_config=config, limits_file=limits_file) )
        except PlotClassError:
            if not allow_missing:
                raise

    return plots
