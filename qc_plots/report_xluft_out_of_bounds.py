from argparse import ArgumentParser
from matplotlib import dates as mdates
import numpy as np
import pandas as pd
import sys

from . import qc_plots2


def driver(nc_files, xluft_lower_limit=0.995, xluft_upper_limit=1.003, min_out_of_bounds_period=pd.Timedelta(days=7), plot_file=None, include_all=False, time_period_file=None):
    time_periods = []
    time_period_strings = []
    all_dec_times = []
    all_xluft = []

    for nc_file in nc_files:
        with qc_plots2.TcconData(data_category=qc_plots2.DataCategory.PRIMARY, nc_file=nc_file, styles=dict()) as data:
            dummy_plot = qc_plots2.RollingTimeseriesPlot(
                other_plots=tuple(),
                yvar='xluft',
                ops='median',
                default_style=dict(),
                limits=None
            )

            args = dummy_plot.get_plot_args(data)
            assert len(args) == 2, "Expected raw and median Xluft to be returned"

        # The rolling values don't have time as objects because of how rolling ops work,
        # but the raw data times should be fine to use, assuming the rolling window is centered.
        times = args[0]['data']['t']
        decimal_time = args[1]['data']['x']
        xluft = args[1]['data']['y']
        
        xx_oob = (xluft < xluft_lower_limit) | (xluft > xluft_upper_limit)
        # Since xx_oob will be True for out-of-bounds points and False for in-bounds
        # points, a difference of +1 means data when out of bounds and -1 means it when
        # in bounds. We'll just need to manually handle the beginning and end of the record
        changes = np.diff(xx_oob.astype('int'))
        start = 1 if xx_oob[0] else -1
        changes[-1] = 1 if xx_oob[-1] else -1
        changes = np.concatenate([[start], changes])
        change_inds = np.flatnonzero(changes)

        # This handles the beginning of the record - if we start out of bounds, we'll look
        # for the time when it comes back in bounds
        time_start_oob = times[0] if start == 1 else None
        index_start_oob = 0 if start == 1 else None
        for time, direction, index in zip(times[change_inds], changes[change_inds], change_inds):
            if direction > 0 and time_start_oob is None:
                # Started a new out of bounds time period. Record when it began.
                time_start_oob = time
                index_start_oob = index
            elif direction > 0:
                # This should never happen, but this means we resumed an out of bounds
                # time period while a previous one was ongoing, so do nothing - just
                # continue.
                pass
            elif direction < 0 and time_start_oob is None:
                # This should also never happen, it means we resumed an in bounds time
                # period while one was already ongoing, so again we do nothing.
                pass
            else:
                # This means that we exited an out-of-bounds time period. Check that
                # it is sufficiently long and print it if so. 
                if (time - time_start_oob) >= min_out_of_bounds_period:
                    if time_period_file is None:
                        print(f'Xluft out of bounds: {time_start_oob.date()} to {time.date()}')
                    time_periods.append((True, decimal_time[index_start_oob], decimal_time[index]))
                    time_period_strings.append(f'- {time_start_oob.date()} to {time.date()}')
                elif include_all:
                    time_periods.append((False, decimal_time[index_start_oob], decimal_time[index]))
                    time_period_strings.append(f'- {time_start_oob.date()} to {time.date()}')
                time_start_oob = None

        # If the last change was into an out-of-bounds period, we need to record everything
        # from the last change up to the end of the record as out of bounds.
        if direction > 0:
            time = times[-1]
            if (time - time_start_oob) >= min_out_of_bounds_period:
                if time_period_file is None:
                    print(f'Xluft out of bounds: {time_start_oob.date()} to {time.date()}')
                time_periods.append((True, decimal_time[index_start_oob], decimal_time[index]))
                time_period_strings.append(f'- {time_start_oob.date()} to {time.date()}')
            elif include_all:
                time_periods.append((False, decimal_time[index_start_oob], decimal_time[index]))
                time_period_strings.append(f'- {time_start_oob.date()} to {time.date()}')

        all_dec_times.append(decimal_time)
        all_xluft.append(xluft)


    if time_period_file is not None:
        # Override the time periods calculated with those given in the input file.
        # This is just easier than refactoring above to only run if no time period file
        # is provided.
        time_periods, time_period_strings = read_time_period_file(time_period_file)


    if plot_file is not None:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        _, axs = plt.subplots(1,2,figsize=(16,6), gridspec_kw={'width_ratios': [2,1]})
        ax = axs[0]
        text_ax = axs[1]
        for decimal_time, xluft in zip(all_dec_times, all_xluft):
            ax.plot(decimal_time, xluft, ls='none', marker='.', color='k')
        ax.axhline(xluft_lower_limit, color='r', ls='--')
        ax.axhline(xluft_upper_limit, color='r', ls='--')
        for sig, t1, t2 in time_periods:
            ax.fill_betweenx([xluft_lower_limit-0.02, xluft_upper_limit+0.02], t1, t2, color='orange' if sig else 'gray', alpha=0.5)
        locator = mdates.MonthLocator(bymonth=[1,4,7,10])
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        text_ax.axis('off')
        if time_period_strings:
            text_ax.text(0, 1, 'Marked periods:\n  ' + '\n  '.join(time_period_strings), transform=text_ax.transAxes, ha='left', va='top')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
        plt.savefig(plot_file, bbox_inches='tight')


def read_time_period_file(time_period_file):
    time_periods = []
    time_period_strings = []

    with open(time_period_file) as f:
        for line in f:
            # ignore everything after a # for comments
            line = line.split('#')[0].strip()
            if len(line) == 0:
                continue

            start, stop = line.split()
            start = pd.to_datetime(start)
            stop = pd.to_datetime(stop)
            dec_start = mdates.date2num(start)
            dec_stop = mdates.date2num(stop)

            time_periods.append((True, dec_start, dec_stop))
            time_period_strings.append(f'- {start.date()} {stop.date()}')

    return time_periods, time_period_strings


def main():
    p = ArgumentParser(description='Print a report on out-of-bounds Xluft from a private TCCON file')
    p.add_argument('nc_files', nargs='+', help='The netCDF file(s) to check')
    p.add_argument('-p', '--plot-file', help='File to output a plot as, if not given, no plot is created.')
    p.add_argument('-a', '--include-all', action='store_true', help='Flag periods shorter than 7 days as well as longer ones')
    p.add_argument('-f', '--time-period-file', help='A file containing time periods to mark, with one period per line, as YYYYMMDD YYYYMMDD. If this argument is given, --plot-file must be given as well.')

    clargs = vars(p.parse_args())
    if clargs['time_period_file'] is not None and clargs['plot_file'] is None:
        print('ERROR: --time-period-file requires that --plot-file be given as well.', file=sys.stderr)
        sys.exit(1)
    driver(**clargs)


if __name__ == '__main__':
    main()
