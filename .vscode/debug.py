from qc_plots import __main__ as qcpm
from argparse import ArgumentParser
import os


def run(config_file, limits_file):
    # nc_file = 'ra20150801_20160206.private.nc'
    # nc_file = 'ra20200626_20200718.private.nc'
    # nc_file = 'bu20200531_20200630.private.nc'
    # nc_file = 'bu20200630_20200728.private.nc'
    nc_file = 'iz20150115_20151230.private.nc'
    # nc_file = 'iz20170620_20171227.private.nc'
    # nc_file = 'iz20190102_20191227.private.nc'
    # nc_file = 'xh20180614_20180630.private.nc'
    # nc_file = 'll20161231_20170629.private.nc'
    # nc_file = 'll20170702_20170930.private.nc'

    output_dir = '/opt/qaqc/tccon_qc_plots/outputs'
    qcpm.driver(
        nc_in=os.path.join('/data/tccon/1-preliminary', nc_file),
        config=config_file,
        limits=limits_file,
        output_dir=output_dir,
        use_tmp_img_dir=True,
    )


if __name__ == '__main__':
    p = ArgumentParser(description='Debug QC plots')
    p.add_argument('--config-file', default='/etc/tccon/qc-plot-config.toml')
    p.add_argument('--limits-file', default='/etc/tccon/qc-plot-limits.toml')
    clargs = vars(p.parse_args())
    run(**clargs)
