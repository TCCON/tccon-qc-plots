from __future__ import print_function

import os
import sys
import numpy as np
import pandas as pd
import netCDF4
import argparse
import json
from contextlib import ExitStack
from pylab import *
from PIL import Image
from datetime import datetime, timedelta
from pathlib import Path


def cm2inch(*tupl):
    """
    Converts reasonable units into terrible units.
    """
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def make_fig(args,nc,xvar,yvar,width=20,height=10):
    fig,ax = subplots()
    fig.set_size_inches(cm2inch(width,height))
    ax.grid()
    if 'units' in nc[yvar].ncattrs():
        ax.set_ylabel('{} ({})'.format(yvar,nc[yvar].units))
    else:
        ax.set_ylabel(yvar)
    if xvar!='time' and 'units' in nc[xvar].ncattrs():
        ax.set_xlabel('{} ({})'.format(xvar,nc[xvar].units))
    else:
        ax.set_xlabel(xvar)
    
    # if showing only flag=0 data, set the axis ranges to the vmin and vmax values of the variable.
    if args.flag0 and 'vmin' in nc[yvar].ncattrs():
        ax.set_ylim(nc[yvar].vmin,nc[yvar].vmax)
    if xvar!='time' and args.flag0 and 'vmin' in nc[yvar].ncattrs():
        ax.set_xlim(nc[xvar].vmin,nc[xvar].vmax)

    return fig,ax


def add_qc_lines(args,nc,ax,xvar,yvar):
    for elem in ['vmin','vmax']:
        if elem in nc[yvar].ncattrs():
            ax.axhline(y=nc[yvar].getncattr(elem),linestyle='dashed',color='black')
        if elem in nc[xvar].ncattrs():
            ax.axvline(x=nc[xvar].getncattr(elem),linestyle='dashed',color='black')


def savefig(fig,code_dir,xvar,yvar,plot_type='sc'):
    fig_name = '{}_VS_{}_{}.jpg'.format(yvar,xvar,plot_type)
    fig_path = os.path.join(code_dir.parent,'outputs',fig_name)
    fig.savefig(fig_path,bbox_inches='tight')
    return fig_path


def make_scatter_plots(args,nc,xvar,yvar,nc_time,flag0,flagged):
    fig,ax = make_fig(args,nc,xvar,yvar)
    if xvar == 'time':
        ax.set_xlim(nc_time[0]-timedelta(days=2),nc_time[-1]+timedelta(days=2))
        if args.ref:
            ax.plot(ref_time,ref[yvar][:],linewidth=0,marker='o',markersize=1,color='lightgray',label=ref.long_name)
        if not args.flag0:
            ax.plot(nc_time[flagged],nc[yvar][flagged],linewidth=0,marker='o',markersize=1,color='red',label='flagged')
        ax.plot(nc_time[flag0],nc[yvar][flag0],linewidth=0,marker='o',markersize=1,color='royalblue',label='flag=0')
    else:
        if not args.flag0:
            ax.plot(nc[xvar][flagged],nc[yvar][flagged],linewidth=0,marker='o',markersize=1,color='red',label='flagged')
        ax.plot(nc[xvar][flag0],nc[yvar][flag0],linewidth=0,marker='o',markersize=1,color='royalblue',label='flag=0')

    # if the variables have a vmin and/or vmax attribute, add lines to the plot
    if not args.flag0:
        add_qc_lines(args,nc,ax,xvar,yvar)

    ax.legend()

    return fig


def make_hexbin_plots(args,nc,xvar,yvar,flag0):
    fig,ax = make_fig(args,nc,xvar,yvar)
    if not args.flag0:
        hb = ax.hexbin(nc[xvar][:],nc[yvar][:],bins='log',mincnt=1,cmap=args.cmap)
        add_qc_lines(args,nc,ax,xvar,yvar)
    else:
        extent = (np.nanmin(nc['solzen'][flag0]), np.nanmax(nc['solzen'][flag0]), nc[yvar].vmin, nc[yvar].vmax)
        hb = ax.hexbin(nc[xvar][:],nc[yvar][:],bins='log',mincnt=1,extent=extent,cmap=args.cmap)
    cb = fig.colorbar(hb,ax=ax)

    return fig


def main():
    if 'tccon-qc' not in sys.executable:
        print('Running qc_plots.py with',sys.executable)
        print('The code should be run with the tccon-qc conda environment')
    code_dir = Path(os.path.dirname(__file__)).absolute()

    parser = argparse.ArgumentParser()
    parser.add_argument('nc_file',help='full path to the netCDF file from which plots should be made')
    parser.add_argument('-r','--ref',default='',help='full path to another netCDF file to use as reference')
    parser.add_argument('--flag0',action='store_true',help='only plot flag=0 data with axis ranges only within the vmin/vmax values of each variable')
    parser.add_argument('--cmap',default='PuBu',help='valid name of a matplotlib colormap https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html')
    parser.add_argument('--json',default=os.path.join(code_dir.parent,'inputs','variables.json'),help='full path to the input json file')
    args = parser.parse_args()

    if not os.path.exists(args.nc_file):
        sys.exit('Invalid path: {}'.format(args.nc_file))
    if args.ref and not os.path.exists(args.ref):
        sys.exit('Invalid path: {}'.format(args.ref))
    if not os.path.exists(args.json):
        sys.exit('Invalid path {}'.format(args.json))

    if args.cmap not in colormaps():
        sys.exit('{} is an Invalid --cmap value, must be one of {}'.format(args.cmap,colormaps()))

    with open(args.json,'r') as f:
        vardata = json.load(f)

    pdf_path = os.path.join(code_dir.parent,'outputs',os.path.basename(args.nc_file).replace('.nc','.pdf'))
    if args.flag0:
        pdf_path = os.path.join(code_dir.parent,'outputs',os.path.basename(args.nc_file).replace('.nc','flag0.pdf'))

    fig_path_list = []
    with ExitStack() as stack:
        nc = stack.enter_context(netCDF4.Dataset(args.nc_file,'r')) # input netcdf file
        if args.ref:
            ref = stack.enter_context(netCDF4.Dataset(args.ref,'r')) # input reference netcdf file

        flag0 = np.where(nc['flag'][:]==0)[0]
        flagged = np.where(nc['flag'][:]!=0)[0]

        fnum = 0
        for xvar in vardata.keys():
            if xvar not in nc.variables:
                print(xvar,'is not in the netCDF file')
                continue
            print('Making plots vs',xvar)
            if xvar == 'time':
                nc_time = np.array([datetime(*cftime.timetuple()[:6]) for cftime in netCDF4.num2date(nc['time'][:],units=nc['time'].units,calendar=nc['time'].calendar)])
                if args.ref:
                    ref_time = np.array([datetime(*cftime.timetuple()[:6]) for cftime in netCDF4.num2date(ref['time'][:],units=ref['time'].units,calendar=ref['time'].calendar)])
            else:
                nc_time = None
            for yvar in vardata[xvar]:
                if yvar not in nc.variables:
                    print('\t',yvar,'is not in the netCDF file')
                    continue
                print('\t',yvar)
                fig_path_list += [savefig(make_scatter_plots(args,nc,xvar,yvar,nc_time,flag0,flagged),code_dir,xvar,yvar,plot_type='sc')]
                close('all')

                if xvar=='solzen':
                    fig_path_list += [savefig(make_hexbin_plots(args,nc,xvar,yvar,flag0),code_dir,xvar,yvar,plot_type='hex')]
                    close('all')
            # end of for yvar loop
        # end of for xvar loop

        # concatenate all .jpg file into a .pdf file
        im_list = [stack.enter_context(Image.open(fig_path)) for fig_path in fig_path_list]
        im_list[0].save(pdf_path, "PDF" ,resolution=100.0, save_all=True, append_images=im_list[1:])

if __name__=="__main__":
    main()