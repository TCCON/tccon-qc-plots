from argparse import ArgumentParser
import tomli

from . import qc_plots2
from .constants import DEFAULT_CONFIG, DEFAULT_LIMITS


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('nc_in',help='full path to the netCDF file from which plots should be made, or a path to a directory (plots will be made from all *.nc files in that directory)')
    parser.add_argument('-r','--ref',default='',help='full path to another netCDF file to use as reference')
    parser.add_argument('-c', '--context', default='', help='full path to another netCDF file to use as context (i.e. full record for the site being plotted)')
    parser.add_argument('-d', '--output-dir', help='Directory to output the .pdf plots to, the default is "outputs" in the code repo')
    parser.add_argument('--flag0',action='store_true',help='only plot flag=0 data with axis ranges only within the vmin/vmax values of each variable')
    parser.add_argument('--cmap',default='PuBu',help='valid name of a matplotlib colormap https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html')
    parser.add_argument('--config', default=DEFAULT_CONFIG, help='full path to the input .toml file for variables to plot')
    parser.add_argument('--json-limits', default=DEFAULT_LIMITS, help='full path to the input json file for axis ranges')
    parser.add_argument('--show-all',action='store_true',help='if given, the axis ranges of the plots will automatically fit in all the data, even huge outliers')
    parser.add_argument('--roll-window',type=int,default=500,help='Size of the rolling window in number of spectra')
    parser.add_argument('--roll-gaps',default='20000 days',help='Minimum time interval for which the data will be split for rolling stats')
    parser.add_argument('--size',choices=('small','medium','large'),default='medium',help='Size of the figures, default is %(default)s')
    parser.add_argument('--quality',choices=('low','medium','high'),default='high',help='Quality of the figures, default is %(default)s')

    return vars(parser.parse_args())


def driver(nc_in, config, flag0=False, show_all=False, **kwargs):
    with open(config) as f:
        config = tomli.load(f)

    #import pdb; pdb.set_trace()
    primary_styles = config.get('style', dict()).get('main', dict())
    data = [qc_plots2.TcconData(qc_plots2.DataCategory.PRIMARY, nc_in, styles=primary_styles)]
    plots = qc_plots2.setup_plots(config)
    fig_paths = []
    n = len(plots)
    for i, plot in enumerate(plots, start=1):
        print(f'  - Plot {i}/{n}: {plot.get_save_name()}', end='')
        try:
            this_path = plot.make_plot(data, flag0_only=flag0, show_all=show_all)
        except IndexError as err:
            print(f' SKIPPED ({err})')
        else:
            print(' DONE')
            fig_paths.append(this_path)


def main():
    clargs = parse_args()
    driver(**clargs)


if __name__ == '__main__':
    main()
