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
    Converts reasonable units (cm) into terrible units (inches).
    """
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def make_fig(args,nc,xvar,yvar,width=20,height=10,kind='',freq=''):
    fig,ax = subplots()
    fig.set_size_inches(cm2inch(width,height))
    ax.grid()
    if kind:
        ax.set_title('{} of data resampled with frequency={}'.format(kind,freq))
     # if showing only flag=0 data, set the axis ranges to the vmin and vmax values of the variable.
    if xvar!='time' and args.flag0 and 'vmin' in nc[yvar].ncattrs():
        ax.set_xlim(nc[xvar].vmin,nc[xvar].vmax)
    if type(yvar)==list:
        ylab = '{} minus {}'.format(*yvar)
        yvar = yvar[0]
    else:
        ylab = yvar
        if args.flag0 and 'vmin' in nc[yvar].ncattrs() and kind!='std':
            ax.set_ylim(nc[yvar].vmin,nc[yvar].vmax)

    if 'units' in nc[yvar].ncattrs():
        ax.set_ylabel('{} ({})'.format(ylab,nc[yvar].units))
    else:
        ax.set_ylabel(ylab)    
    if xvar!='time' and 'units' in nc[xvar].ncattrs():
        ax.set_xlabel('{} ({})'.format(xvar,nc[xvar].units))
    else:
        ax.set_xlabel(xvar)

    return fig,ax


def add_qc_lines(args,nc,ax,xvar,yvar,kind=''):
    for elem in ['vmin','vmax']:
        if type(yvar)!=list and kind!='std' and (elem in nc[yvar].ncattrs()): # don't add the line for difference plots and standard deviation plots
            ax.axhline(y=nc[yvar].getncattr(elem),linestyle='dashed',color='black')
        if elem in nc[xvar].ncattrs():
            ax.axvline(x=nc[xvar].getncattr(elem),linestyle='dashed',color='black')


def savefig(fig,code_dir,xvar,yvar,plot_type='sc'):
    if type(yvar)==list:
        fig_name = '{}_minus{}_VS_{}_{}.png'.format(yvar[0],yvar[1],xvar,plot_type)
    else:
        fig_name = '{}_VS_{}_{}.png'.format(yvar,xvar,plot_type)
    fig_path = os.path.join(code_dir.parent,'outputs',fig_name)
    fig.savefig(fig_path,bbox_inches='tight')
    return fig_path


def make_scatter_plots(args,nc,ref,xvar,yvar,nc_time,ref_time,flag0,flagged,kind='',freq=''):
    if xvar not in nc.variables:
        xvar = 'time'
    fig,ax = make_fig(args,nc,xvar,yvar,kind=kind,freq=freq)
    if type(yvar)==list:
        ydata = nc[yvar[0]][:]-nc[yvar[1]][:] # used for the resampled plots without --flag0
        if args.ref:
            ref_data = ref[yvar[0]][:]-ref[yvar[1]][:]
    else:
        ydata = nc[yvar][:] # used for the resampled plots without --flag0
        if args.ref:
            ref_data = ref[yvar][:]

    if kind:
        if args.flag0:
            ydata = ydata[flag0]
            nc_time = nc_time[flag0]
        if kind=='mean':
            main_frame = pd.DataFrame().from_dict({'x':nc_time,'y':ydata}).set_index('x').resample(freq).mean()
        elif kind=='median':
            main_frame = pd.DataFrame().from_dict({'x':nc_time,'y':ydata}).set_index('x').resample(freq).median()
        elif kind=='std':
            main_frame = pd.DataFrame().from_dict({'x':nc_time,'y':ydata}).set_index('x').resample(freq).std(ddof=1)
        ydata = main_frame['y'].values
        nc_time = np.array([pd.Timestamp(x).to_pydatetime() for x in main_frame.index])

        if args.ref:
            if kind=='mean':
                ref_frame = pd.DataFrame().from_dict({'x':ref_time,'y':ref_data}).set_index('x').resample(freq).mean()
            elif kind=='median':
                ref_frame = pd.DataFrame().from_dict({'x':ref_time,'y':ref_data}).set_index('x').resample(freq).median()
            elif kind=='std':
                ref_frame = pd.DataFrame().from_dict({'x':ref_time,'y':ref_data}).set_index('x').resample(freq).std(ddof=1)
            ref_time = np.array([pd.Timestamp(x).to_pydatetime() for x in ref_frame.index])
            ref_data = ref_frame['y'].values

    if xvar == 'time':
        ax.set_xlim(nc_time[0]-timedelta(days=2),nc_time[-1]+timedelta(days=2))
        if args.ref:
            ax.plot(ref_time,ref_data,linewidth=0,marker='o',markersize=1,color='lightgray',label=ref.long_name)
        if kind:
            ax.plot(nc_time,ydata,linewidth=0,marker='o',markersize=1,color='royalblue',label=nc.long_name)
        else:
            if not args.flag0:
                ax.plot(nc_time[flagged],ydata[flagged],linewidth=0,marker='o',markersize=1,color='red',label='flagged')
            ax.plot(nc_time[flag0],ydata[flag0],linewidth=0,marker='o',markersize=1,color='royalblue',label='flag=0')
    else:
        if not args.flag0:
            ax.plot(nc[xvar][flagged],ydata[flagged],linewidth=0,marker='o',markersize=1,color='red',label='flagged')
        ax.plot(nc[xvar][flag0],ydata[flag0],linewidth=0,marker='o',markersize=1,color='royalblue',label='flag=0')

    # if the variables have a vmin and/or vmax attribute, add lines to the plot
    if not args.flag0:
        add_qc_lines(args,nc,ax,xvar,yvar,kind='')

    ax.legend()

    return fig    


def make_hexbin_plots(args,nc,xvar,yvar,flag0):
    fig,ax = make_fig(args,nc,xvar,yvar)
    if type(yvar)==list:
        if not args.flag0:
            ydata = nc[yvar[0]][:]-nc[yvar[1]][:]
        else:
            ydata = nc[yvar[0]][flag0]-nc[yvar[1]][flag0]
    else:
        if not args.flag0:
            ydata = nc[yvar][:]
        else:
            ydata = nc[yvar][flag0]

    if not args.flag0:
        hb = ax.hexbin(nc[xvar][:],ydata,bins='log',mincnt=1,cmap=args.cmap)
        add_qc_lines(args,nc,ax,xvar,yvar,kind='')
    else:
        if type(yvar)==list:
            yvar = yvar[0]
        if 'vmin' in nc[yvar].ncattrs():
            extent = (np.nanmin(nc['solzen'][flag0]), np.nanmax(nc['solzen'][flag0]), nc[yvar].vmin, nc[yvar].vmax)
        else:
            extent = (np.nanmin(nc['solzen'][flag0]), np.nanmax(nc['solzen'][flag0]), np.min(nc[yvar][flag0]), np.max(nc[yvar][flag0]))
        hb = ax.hexbin(nc[xvar][flag0],ydata,bins='log',mincnt=1,extent=extent,cmap=args.cmap)
    cb = fig.colorbar(hb,ax=ax)

    return fig


def flag_analysis(code_dir,nc):
    """
    Make a histogram to summarize flags
    """
    flag_df = pd.DataFrame()
    flag_df_pcnt = pd.DataFrame()
    flag_set = list(set(nc['flag'][:]))
    N = nc['time'].size
    print('Summary of flags:')
    print('  #  Parameter              N_flag      %')
    nflag_tot = 0
    for flag in sorted(flag_set):
        if flag==0:
            continue
        where_flagged = nc['flag'][:]==flag
        nflag = np.count_nonzero(where_flagged)
        nflag_pcnt = 100*nflag/N
        nflag_tot += nflag
        flagged_var_name = nc['flagged_var_name'][where_flagged][0]
        print('{:>3}  {:<20} {:>6}   {:>8.3f}'.format(flag,flagged_var_name,nflag,nflag_pcnt))
        flag_df[flagged_var_name] = pd.Series([nflag])
        flag_df_pcnt[flagged_var_name] = pd.Series([nflag_pcnt])
    
    print('     {:<20} {:>6}   {:>8.3f}'.format('TOTAL',nflag_tot,100*nflag_tot/N))
    flag_df['Total'] = nflag_tot
    flag_df_pcnt['Total'] = 100*nflag_tot/N

    drop_list = [var for var in flag_df_pcnt if flag_df_pcnt[var][0]>0.05]
    flag_df = flag_df.drop(columns=drop_list)
    flag_df_pcnt = flag_df_pcnt.drop(columns=drop_list)
    
    fig, ax = subplots(2,1)
    ax[0].set_ylabel('Count')
    ax[1].set_ylabel('Count (%)')
    ax[0].set_title('{} total spectra from {}'.format(N,nc.long_name))
    fig.suptitle('Summary of flags')
    barplot = flag_df.plot(kind='bar',ax=ax[0])
    barplot_pcnt = flag_df_pcnt.plot(kind='bar',ax=ax[1],legend=False)
    for elem in ax:
        elem.axes.get_xaxis().set_visible(False)
        elem.grid()
    ax[0].legend(prop=dict(size=8))

    prec = ['{:.0f}','{:.2f}']
    for i,curplot in enumerate([barplot,barplot_pcnt]):
        for p in curplot.patches:
            curplot.annotate(prec[i].format(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))

    fig_path = os.path.join(code_dir.parent,'outputs','flags_summary.png')
    fig.savefig(fig_path,bbox_inches='tight')

    return fig_path


def simple_plots(args,code_dir,nc,ref,nc_time,ref_time,vardata,xvar,flag0,flagged,kind='',freq=''):
    fig_path_list = []
    for yvar in vardata[xvar]:
        if type(yvar)==list:
            check = False
            for v in yvar:
                if v not in nc.variables:
                    print('\t',v,'is not in the netCDF file')
                    check = True
            if check:
                continue
            print("\t {} minus {}".format(*yvar))
            fig_path_list += [savefig(make_scatter_plots(args,nc,ref,xvar,yvar,nc_time,ref_time,flag0,flagged,kind=kind,freq=freq),code_dir,xvar,yvar,plot_type='sc')]
        else:
            if yvar not in nc.variables:
                print('\t',yvar,'is not in the netCDF file')
                continue
            if kind:
                print(yvar,freq,kind)
            else:
                print('\t',yvar)
            fig_path_list += [savefig(make_scatter_plots(args,nc,xvar,yvar,nc_time,flag0,flagged,kind=kind,freq=freq),code_dir,xvar,yvar,plot_type='sc')]
            close('all')

            if xvar=='solzen':
                fig_path_list += [savefig(make_hexbin_plots(args,nc,xvar,yvar,flag0),code_dir,xvar,yvar,plot_type='hex')]
                close('all')
    return fig_path_list


def default_plots(args,code_dir,nc,nc_time,flag0,flagged):
    """
    Make (70-80SZA AM - 70-80SZA PM) and (70-80SZA PM - 40-50SZA PM) plots for xluft
    """
    fig_path_list = []
    if 'xluft' not in nc.variables:
        return fig_path_list

    xluft = nc['xluft'][:]
    solzen = nc['solzen'][:]
    azim = nc['azim'][:]

    df = pd.DataFrame().from_dict({'x':nc_time,'y':xluft})

    all_ids = np.arange(nc['xluft'].size)

    AM = all_ids[azim<=180]
    PM = all_ids[azim>180]

    high_sza = all_ids[(70<=solzen) & (solzen<=80)]
    mid_sza = all_ids[(40<=solzen) & (solzen<=50)]
    low_sza = all_ids[(20<=solzen) & (solzen<=30)]

    if args.flag0:
        all_ids = flag0

    AM_high_sza = np.array(set(all_ids).intersection(set(AM),set(high_sza)))
    PM_high_sza = np.array(set(all_ids).intersection(set(PM),set(high_sza)))

    AM_mid_sza = np.array(set(all_ids).intersection(set(AM),set(mid_sza)))
    PM_mid_sza = np.array(set(all_ids).intersection(set(PM),set(mid_sza)))

    AM_low_sza = np.array(set(all_ids).intersection(set(AM),set(low_sza)))
    PM_low_sza = np.array(set(all_ids).intersection(set(PM),set(low_sza)))

    df_AM_high_sza = df.loc[AM_high_sza].set_index('x').resample('W').median()
    df_PM_high_sza = df.loc[PM_high_sza].set_index('x').resample('W').median()
    df_AM_mid_sza = df.loc[AM_mid_sza].set_index('x').resample('W').median()
    df_PM_mid_sza = df.loc[PM_mid_sza].set_index('x').resample('W').median()
    df_AM_low_sza = df.loc[AM_low_sza].set_index('x').resample('W').median()
    df_PM_low_sza = df.loc[PM_low_sza].set_index('x').resample('W').median()

    # 70-80 SZA AM and 70-80 SZA PM
    if df_AM_high_sza['y'].size!=0 and df_PM_high_sza['y'].size!=0:
        fig,ax = make_fig(args,nc,'time','xluft',width=20,height=10,kind='',freq='')
        ax.set_title('Weekly medians')
        ax.set_ylim(0.99,1.01)
        df_AM_high_sza['y'].plot(ax=ax,marker='o',markersize=1,linewidth=0,color='royalblue',label='70<=SZA<=80 AM')
        df_PM_high_sza['y'].plot(ax=ax,marker='o',markersize=1,linewidth=0,color='red',label='70<=SZA<=80 PM')
        fig_path = os.path.join(code_dir.parent,'outputs','xluft_high_sza_AM_high_sza_PM_vs_time.png')
        ax.grid()
        ax.legend()
        if not args.flag0:
            add_qc_lines(args,nc,ax,'time','xluft')
        fig.savefig(fig_path,bbox_inches='tight')
        close('all')
        fig_path_list += [fig_path]

    # 70-80 SZA PM and 40-50 SZA PM and 20-30 SZA PM
    if df_PM_high_sza['y'].size!=0 and df_PM_mid_sza['y'].size!=0:
        fig,ax = make_fig(args,nc,'time','xluft',width=20,height=10,kind='',freq='')
        ax.set_title('Weekly medians')
        ax.set_ylim(0.99,1.01)
        df_PM_high_sza['y'].plot(ax=ax,marker='o',markersize=1,linewidth=0,color='royalblue',label='70<=SZA<=80 PM')
        df_PM_mid_sza['y'].plot(ax=ax,marker='o',markersize=1,linewidth=0,color='red',label='40<=SZA<=50 PM')
        if df_PM_low_sza['y'].size!=0:
            df_PM_low_sza['y'].plot(ax=ax,marker='o',markersize=1,linewidth=0,color='green',label='20<=SZA<=30 PM')
        ax.grid()
        ax.legend()
        if not args.flag0:
            add_qc_lines(args,nc,ax,'time','xluft')
        fig_path = os.path.join(code_dir.parent,'outputs','xluft_sza_PM_vs_time.png')
        fig.savefig(fig_path,bbox_inches='tight')
        close('all')
        fig_path_list += [fig_path]

    # 70-80 SZA PM and 40-50 SZA PM and 20-30 SZA PM
    if df_AM_high_sza['y'].size!=0 and df_AM_mid_sza['y'].size!=0:
        fig,ax = make_fig(args,nc,'time','xluft',width=20,height=10,kind='',freq='')
        ax.set_title('Weekly medians')
        ax.set_ylim(0.99,1.01)
        df_AM_high_sza['y'].plot(ax=ax,marker='o',markersize=1,linewidth=0,color='royalblue',label='70<=SZA<=80 AM')
        df_AM_mid_sza['y'].plot(ax=ax,marker='o',markersize=1,linewidth=0,color='red',label='40<=SZA<=50 AM')
        if df_AM_low_sza['y'].size!=0:
            df_AM_low_sza['y'].plot(ax=ax,marker='o',markersize=1,linewidth=0,color='green',label='20<=SZA<=30 AM')
        if not args.flag0:
            add_qc_lines(args,nc,ax,'time','xluft')
        ax.grid()
        ax.legend()
        fig_path = os.path.join(code_dir.parent,'outputs','xluft_sza_AM_vs_time.png')
        fig.savefig(fig_path,bbox_inches='tight')
        close('all')
        fig_path_list += [fig_path]

    return fig_path_list

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
        nc_time = np.array([datetime(*cftime.timetuple()[:6]) for cftime in netCDF4.num2date(nc['time'][:],units=nc['time'].units,calendar=nc['time'].calendar)])
        if args.ref:
            ref = stack.enter_context(netCDF4.Dataset(args.ref,'r')) # input reference netcdf file
            ref_time = np.array([datetime(*cftime.timetuple()[:6]) for cftime in netCDF4.num2date(ref['time'][:],units=ref['time'].units,calendar=ref['time'].calendar)])
        else:
            ref = None
            ref_time = None

        all_ids = np.arange(nc['time'].size)
        flag0 = all_ids[nc['flag'][:]==0]
        flagged = all_ids[nc['flag'][:]!=0]

        if not args.flag0:
            fig_path_list += [flag_analysis(code_dir,nc)]

        print('Making default plots')
        fig_path_list += default_plots(args,code_dir,nc,nc_time,flag0,flagged)

        fnum = 0
        for xvar in vardata.keys():
            if xvar.endswith("median"):
                fig_path_list += simple_plots(args,code_dir,nc,ref,nc_time,ref_time,vardata,xvar,flag0,flagged,kind='median',freq=xvar.split('_')[0])
                continue
            elif xvar.endswith("mean"):
                fig_path_list += simple_plots(args,code_dir,nc,ref,nc_time,ref_time,vardata,xvar,flag0,flagged,kind='mean',freq=xvar.split('_')[0])
                continue
            elif xvar.endswith("std"):
                fig_path_list += simple_plots(args,code_dir,nc,ref,nc_time,ref_time,vardata,xvar,flag0,flagged,kind='std',freq=xvar.split('_')[0])
                continue
            elif xvar not in [v for v in nc.variables]:
                print(xvar,'is not in the netCDF file')
                continue
            print('Making plots vs',xvar)
            fig_path_list += simple_plots(args,code_dir,nc,ref,nc_time,ref_time,vardata,xvar,flag0,flagged)
            # end of for yvar loop
        # end of for xvar loop

        # concatenate all .png file into a .pdf file
        im_list = [stack.enter_context(Image.open(fig_path).convert('RGB')) for fig_path in fig_path_list]
        im_list[0].save(pdf_path, "PDF" ,resolution=100.0, save_all=True, append_images=im_list[1:])

if __name__=="__main__":
    main()