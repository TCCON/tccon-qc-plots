from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
import matplotlib.pyplot as plt
import netCDF4 as ncdf
import numpy as np
import pandas as pd


class FlagPlotMode(Enum):
    USER_PREFERENCE = 0
    ONLY_FLAG0 = 1
    ALWAYS_ALL = 2
    ALWAYS_BOTH = 3


class FlagCategory(Enum):
    ALL_DATA = 'all'
    FLAG0 = 'flag0'
    FLAGGED = 'flagged'


class TcconData:
    def __init__(self,
                 nc_file,
                 styles,
                 exclude_times=None,
                 flag_plot_mode=FlagPlotMode.USER_PREFERENCE):
        self.nc_dset = ncdf.Dataset(nc_file)
        self.styles = styles
        self.times = self.nctime_to_pytime(self.nc_dset['time'])
        self.flag_plot_mode = flag_plot_mode

        if exclude_times is not None:
            self.all_idx = (self.times < np.min(exclude_times)) | (self.times > np.max(exclude_times))
        else:
            self.all_idx = np.ones(self.times.shape, dtype=np.bool_)

        if 'flag' in self.nc_dset.variables:
            self.flag0_idx = (self.nc_dset['flag'][:] == 0) & self.all_idx
            self.flagged_idx = (self.nc_dset['flag'][:] != 0) & self.all_idx
        else:
            self.flag0_idx = None
            self.flagged_idx = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.nc_dset.close()

    def get_data(self, varname, flag_category: FlagCategory):
        if flag_category == FlagCategory.ALL_DATA or self.flag_plot_mode == FlagPlotMode.ALWAYS_ALL:
            return self.get_all_data(varname)
        elif flag_category == FlagCategory.FLAG0:
            return self.get_flag0_data(varname)
        elif flag_category == FlagCategory.FLAGGED:
            return self.get_flagged_data(varname)

    def get_all_data(self, varname):
        data = self.times if varname == 'time' else self.nc_dset[varname][:]
        return data[self.all_idx]

    def get_flag0_data(self, varname):
        data = self.times if varname == 'time' else self.nc_dset[varname][:]
        return data[self.flag0_idx]

    def get_flagged_data(self, varname):
        if self.flag_plot_mode == FlagPlotMode.ALWAYS_ALL:
            return None

        data = self.times if varname == 'time' else self.nc_dset[varname][:]
        return data[self.flagged_idx]

    def get_flag0_label(self):
        site_name = self.nc_dset.long_name
        if self.flag_plot_mode == FlagPlotMode.ALWAYS_ALL:
            return '{} all data'.format(site_name)
        else:
            return '{} flag==0'.format(site_name)

    def get_label(self, flag_category: FlagCategory):
        if flag_category == FlagCategory.FLAG0:
            return self.get_flag0_label()
        elif flag_category == FlagCategory.FLAGGED:
            return self.get_flagged_label()
        elif flag_category == FlagCategory.ALL_DATA:
            return self.get_alldata_label()

    def get_flagged_label(self):
        site_name = self.nc_dset.long_name
        if self.flag_plot_mode in {FlagPlotMode.USER_PREFERENCE, FlagPlotMode.ALWAYS_BOTH}:
            return '{} flagged'.format(site_name)
        else:
            return ''

    def get_alldata_label(self):
        site_name = self.nc_dset.long_name
        return '{} all data'.format(site_name)

    @staticmethod
    def nctime_to_pytime(nc_time_var):
        return ncdf.num2date(nc_time_var[:], units=nc_time_var.units, calendar=nc_time_var.calendar)


class AbstractPlot(ABC):
    def get_plot_args(self, data: TcconData, flag0_only: bool = False):
        # There will always be at least one thing to plot, whether it is all data or just flag0 data
        plot_args = [{
            'data': self.get_plot_data(data, FlagCategory.FLAG0),
            'kws': self.get_plot_kws(data, FlagCategory.FLAG0)
        }]

        # Whether there is more data to plot depends on whether the user specified that we should only plot
        # flag0 data and whether this particular data cares
        plot_from_user_choice = data.flag_plot_mode == FlagPlotMode.USER_PREFERENCE and not flag0_only
        plot_from_data_setting = data.flag_plot_mode == FlagPlotMode.ALWAYS_BOTH
        if plot_from_data_setting or plot_from_user_choice:
            plot_args.append({
                'data': self.get_plot_data(data, FlagCategory.FLAGGED),
                'kws': self.get_plot_kws(data, FlagCategory.FLAGGED)
            })

        return plot_args

    @abstractmethod
    def get_plot_kws(self, data: TcconData, flagged_category: FlagCategory):
        pass

    @abstractmethod
    def get_plot_data(self, data: TcconData, flagged_category: FlagCategory):
        pass


class ScatterPlot(AbstractPlot):
    def __init__(self, xvar, yvar, default_style):
        self.xvar = xvar
        self.yvar = yvar
        self._default_style = default_style

    def get_plot_kws(self, data: TcconData, flag_category: FlagCategory):
        default = self._default_style.get(flag_category, dict()).get('scatter', dict())
        specific = data.styles.get(flag_category, dict()).get('scatter', dict())
        style = deepcopy(default)
        style.update(specific)
        style['label'] = data.get_label(flag_category)
        style.setdefault('linestyle', 'none')
        return style

    def get_plot_data(self, data: TcconData, flag_category: FlagCategory):
        x = data.get_data(self.xvar, flag_category)
        y = data.get_data(self.yvar, flag_category)
        return {'x': x, 'y': y}

    def plot(self, data: TcconData, ax=None, flag0_only: bool = False):
        plot_args = self.get_plot_args(data, flag0_only=flag0_only)
        for args in plot_args:
            ax.plot(args['data']['x'], args['data']['y'], **args['kws'])

