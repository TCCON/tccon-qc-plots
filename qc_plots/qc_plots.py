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
from PyPDF2 import PdfFileWriter, PdfFileReader
from scipy.optimize import curve_fit
from scipy.stats import pearsonr


def cm2inch(*tupl):
    """
    Converts reasonable units (cm) into terrible units (inches).
    """
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def get_limits(nc,var):

    limits = [None,None]

    if 'vsf_h2o' in var or 'vsf_hdo' in var or 'vsf_hf' in var:
        limits = [0.3,2]
    elif 'vsf_hcl' in var:
        limits = [0.7,1.4]
    elif 'vsf_' in var:
        limits = [0.9,1.1]
    elif var.endswith('_fs'):
        limits = [-2,2]
    elif var.endswith('_sg'):
        limits = [-8,8]
    elif var.endswith('_zo'):
        limits = [-0.006,0.006]
    elif var.endswith('_cfampocl'):
        limits = [0,0.005]
    elif var.endswith('_cfperiod'):
        limits = [-1,50]
    elif var == 'xluft':
        limits = [0.975,1.025]
    elif var == 'xch4':
        limits = [1.6,2.0]
    elif var == 'xco':
        limits = [0,200]
    elif var == 'xn2o':
        limits = [200,400]
    elif var == 'xhdo':
        limits = [0,10000]
    elif var == 'xh2o':
        limits = [0,10000]
    elif var == 'xch4_error':
        limits = [0,0.05]
    elif var == 'xco_error':
        limits = [0,10]
    elif var == 'xhdo_error':
        limits = [0,100]
    elif var == 'xh2o_error':
        limits = [0,100]
    elif var == 'xhf_error':
        limits = [0,50]
    elif var == 'xn2o_error':
        limits = [0,25]
    elif var == 'dip':
        limits = [-0.05,0.05]
    elif var == 'mvd':
        limits = [-5,250]
    elif var == 'lse':
        limits = [-0.1,0.1]
    elif var == 'lsu':
        limits = [0,0.06]
    elif 'vmin' in nc[var].ncattrs():
        limits = [nc[var].vmin,nc[var].vmax]  

    return limits


def make_fig(args,nc,xvar,yvar,width=20,height=10,kind='',freq=''):
    two_subplots = type(yvar)==list and type(yvar[0])==list
    if two_subplots:
        fig,axes = subplots(2,1,sharex=True,gridspec_kw={'height_ratios':[1,3]})
        axes[0].grid()
        axes[1].grid()
        yvar2 = yvar[0][1] # for the top plot
        yvar = yvar[0][0] # for the bottom plot
        ax = axes[1] # the bottom plot
        if 'units' in nc[yvar2].ncattrs():
            axes[0].set_ylabel('{} ({})'.format(yvar2,nc[yvar2].units))
        else:
            axes[0].set_ylabel(yvar2)
    else:
        fig,ax = subplots()
        ax.grid()
    fig.set_size_inches(cm2inch(width,height))
    
    if kind:
        fig.suptitle('{} of data resampled with frequency={}'.format(kind,freq))

    if not args.show_all:
        if kind!='std':
            ax.set_ylim(get_limits(nc,yvar))
        if xvar!='time':
            ax.set_xlim(get_limits(nc,xvar))
        if two_subplots:
            axes[0].set_ylim(get_limits(nc,yvar2))

    if type(yvar)==list:
        ylab = '{} minus {}'.format(*yvar)
        yvar = yvar[0]
    else:
        ylab = yvar

    if 'units' in nc[yvar].ncattrs():
        ax.set_ylabel('{} ({})'.format(ylab,nc[yvar].units))
    else:
        ax.set_ylabel(ylab)

    if xvar!='time' and 'units' in nc[xvar].ncattrs():
        ax.set_xlabel('{} ({})'.format(xvar,nc[xvar].units))
    else:
        ax.set_xlabel(xvar)

    if two_subplots:
        ax = axes
    return fig,ax


def add_qc_lines(args,nc,ax,xvar,yvar,kind=''):
    for elem in ['vmin','vmax']:
        if type(yvar)!=list and kind!='std' and (elem in nc[yvar].ncattrs()): # don't add the line for difference plots and standard deviation plots
            ax.axhline(y=nc[yvar].getncattr(elem),linestyle='dashed',color='black')
        if elem in nc[xvar].ncattrs():
            ax.axvline(x=nc[xvar].getncattr(elem),linestyle='dashed',color='black')


def savefig(fig,code_dir,xvar,yvar,plot_type='sc'):
    if type(yvar)==list and type(yvar[0])==list:
        fig_name = '{}_and_{}_VS_{}_{}.png'.format(yvar[0][0],yvar[0][1],xvar,plot_type)
    elif type(yvar)==list:
        fig_name = '{}_minus_{}_VS_{}_{}.png'.format(yvar[0],yvar[1],xvar,plot_type)
    else:
        fig_name = '{}_VS_{}_{}.png'.format(yvar,xvar,plot_type)
    fig_path = os.path.join(code_dir.parent,'outputs',fig_name)
    fig.savefig(fig_path,bbox_inches='tight',dpi=300)
    return fig_path


def lin_model(x,a,b):
    return a*x+b


def add_linfit(ax,x,y,yerr=None):
    """
    Add a line with the results of a linear fit to y = a*x + b
    Also shows the squared pearson correlation coefficient

    Inputs:
        - ax: matplotlib subplot
        - x: x axis data
        - y: y axis data
    """
    
    # linear fit using y = a*x + b
    if yerr:
        fit,cov = curve_fit(lin_model,x,y,p0=[1,0],sigma=yerr)
    else:
        fit,cov = curve_fit(lin_model,x,y,p0=[1,0])
    
    # pearson correlation coefficient
    R = pearsonr(x,y)[0]

    leg = 'y=({:.4f} $\pm$ {:.4f})*x + ({:.4f} $\pm$ {:.4f}); R²={:.3f}'.format(fit[0],np.sqrt(cov[0][0]),fit[1],np.sqrt(cov[1][1]),R**2) 

    # plot line fits
    ax.plot(x,lin_model(x,fit[0],fit[1]),linestyle='--',label=leg,color="C1")


def make_scatter_plots(args,nc,ref,xvar,yvar,nc_time,ref_time,flag0,flagged,ref_flag0,kind='',freq=''):
    if xvar not in nc.variables:
        xvar = 'time'

    fig,ax = make_fig(args,nc,xvar,yvar,kind=kind,freq=freq)

    two_subplots = type(yvar)==list and type(yvar[0])==list
    if two_subplots:
        ax2 = ax[0]
        ax = ax[1]
        yvar2 = yvar[0][1]
        yvar = yvar[0][0]
        ydata = nc[yvar][:]
        ydata2 = nc[yvar2][:]
        if args.ref and args.flag0 and 'flag' in ref.variables:
            ref_data = ref[yvar][ref_flag0]
            ref_data2 = ref[yvar2][ref_flag0]            
        elif args.ref:
            ref_data = ref[yvar][:]
            ref_data2 = ref[yvar2][:]
    elif type(yvar)==list:
        ydata = nc[yvar[0]][:]-nc[yvar[1]][:] # used for the resampled plots without --flag0
        if args.ref and args.flag0 and 'flag' in ref.variables:
            ref_data = ref[yvar[0]][ref_flag0]-ref[yvar[1]][ref_flag0]
        elif args.ref:
            ref_data = ref[yvar[0]][:]-ref[yvar[1]][:]
    else:
        ydata = nc[yvar][:] # used for the resampled plots without --flag0
        if args.ref and args.flag0 and 'flag' in ref.variables:
            ref_data = ref[yvar][ref_flag0]
        elif args.ref:
            ref_data = ref[yvar][:]

    if len(set(ydata.data)) in [1,0]:
        ax.text(0.5,0.5,'{} is constant'.format(yvar),transform=ax.transAxes,color='black')

    if kind:
        if args.flag0:
            ydata = ydata[flag0]
            nc_time = nc_time[flag0]
            if two_subplots:
                ydata2 = ydata2[flag0]
        if two_subplots:
            data_dict = {'x':nc_time,'y':ydata,'y2':ydata2}
        else:
            data_dict = {'x':nc_time,'y':ydata}

        if kind=='mean':
            main_frame = pd.DataFrame().from_dict(data_dict).set_index('x').resample(freq).mean()
        elif kind=='median':
            main_frame = pd.DataFrame().from_dict(data_dict).set_index('x').resample(freq).median()
        elif kind=='std':
            main_frame = pd.DataFrame().from_dict(data_dict).set_index('x').resample(freq).std(ddof=1)
        ydata = main_frame['y'].values
        nc_time = np.array([pd.Timestamp(x).to_pydatetime() for x in main_frame.index])
        if two_subplots:
            ydata2 = main_frame['y2'].values

        if args.ref:
            if two_subplots:
                ref_data_dict = {'x':ref_time,'y':ref_data,'y2':ref_data2}
            else:
                ref_data_dict = {'x':ref_time,'y':ref_data}            
            if kind=='mean':
                ref_frame = pd.DataFrame().from_dict(ref_data_dict).set_index('x').resample(freq).mean()
            elif kind=='median':
                ref_frame = pd.DataFrame().from_dict(ref_data_dict).set_index('x').resample(freq).median()
            elif kind=='std':
                ref_frame = pd.DataFrame().from_dict(ref_data_dict).set_index('x').resample(freq).std(ddof=1)
            ref_time = np.array([pd.Timestamp(x).to_pydatetime() for x in ref_frame.index])
            ref_data = ref_frame['y'].values
            if two_subplots:
                ref_data2 = ref_frame['y2'].values

    if xvar == 'time':
        ax.set_xlim(nc_time[0]-timedelta(days=2),nc_time[-1]+timedelta(days=2))
        if args.ref:
            ax.plot(ref_time,ref_data,linewidth=0,marker='o',markersize=1,color='lightgray',label=ref.long_name)
            if two_subplots:
                ax2.plot(ref_time,ref_data2,linewidth=0,marker='o',markersize=1,color='lightgray')
        if kind:
            ax.plot(nc_time,ydata,linewidth=0,marker='o',markersize=1,color='royalblue',label=nc.long_name)
            if two_subplots:
               ax2.plot(nc_time,ydata2,linewidth=0,marker='o',markersize=1,color='royalblue') 
        else:
            if not args.flag0:
                ax.plot(nc_time[flagged],ydata[flagged],linewidth=0,marker='o',markersize=1,color='red',label='{} flagged'.format(nc.long_name))
                if two_subplots:
                    ax2.plot(nc_time[flagged],ydata2[flagged],linewidth=0,marker='o',markersize=1,color='red')
            ax.plot(nc_time[flag0],ydata[flag0],linewidth=0,marker='o',markersize=1,color='royalblue',label='{} flag=0'.format(nc.long_name))
            if two_subplots:
                ax2.plot(nc_time[flag0],ydata2[flag0],linewidth=0,marker='o',markersize=1,color='royalblue')
    else:
        if not args.flag0:
            ax.plot(nc[xvar][flagged],ydata[flagged],linewidth=0,marker='o',markersize=1,color='red',label='{} flagged'.format(nc.long_name))
            if two_subplots:
                ax2.plot(nc[xvar][flagged],ydata2[flagged],linewidth=0,marker='o',markersize=1,color='red')
        x = nc[xvar][flag0]
        y = ydata[flag0]
        ax.plot(x,y,linewidth=0,marker='o',markersize=1,color='royalblue',label='{} flag=0'.format(nc.long_name))
        add_linfit(ax,x,y) # only add the linear fit to the flag=0 data, even when plotting all data
        if two_subplots:
            ax2.plot(x,ydata2[flag0],linewidth=0,marker='o',markersize=1,color='royalblue')

    # if the variables have a vmin and/or vmax attribute, add lines to the plot
    add_qc_lines(args,nc,ax,xvar,yvar,kind='')
    if two_subplots:
        add_qc_lines(args,nc,ax2,xvar,yvar2,kind='')

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

    if args.flag0:
        xdata = nc[xvar][flag0]
    else:
        xdata = nc[xvar][:]

    if type(yvar)==list:
        yvar = yvar[0]

    xmin, xmax = get_limits(nc,xvar)
    ymin, ymax = get_limits(nc,yvar)
    if None in [xmin,xmax]:
        xmin, xmax = [np.nanmin(xdata), np.nanmax(xdata)]
    if None in [ymin,ymax]:
         ymin, ymax = [np.nanmin(ydata), np.nanmax(ydata)]

    extent = (xmin, xmax, ymin, ymax)

    hb = ax.hexbin(xdata,ydata,bins='log',mincnt=1,extent=extent,cmap=args.cmap,linewidths=(0,))
    cb = fig.colorbar(hb,ax=ax)

    add_linfit(ax,nc[xvar][flag0],nc[yvar][flag0])
    ax.legend()

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

    drop_list = [var for var in flag_df_pcnt if flag_df_pcnt[var][0]<0.05]
    flag_df = flag_df.drop(columns=drop_list)
    flag_df_pcnt = flag_df_pcnt.drop(columns=drop_list)
    
    fig, ax = subplots(2,1)
    ax[0].set_ylabel('Count')
    ax[1].set_ylabel('Count (%)')
    ax[0].set_title('{} total spectra from {}'.format(N,nc.long_name))
    fig.suptitle('Summary of flags with count>0.05%')
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


def simple_plots(args,code_dir,nc,ref,nc_time,ref_time,vardata,xvar,flag0,flagged,ref_flag0,kind='',freq=''):
    fig_path_list = []
    for yvar in vardata[xvar]:
        if type(yvar)==list and type(yvar[0])==list: # two plots with 1:3 ratio
            check = False
            for v in yvar[0]:
                if v not in nc.variables:
                    print('\t',v,'is not in the netCDF file')
                    check = True
                elif np.count_nonzero(nc[v][:].mask)==nc[v].size:
                    print('\t',v,'has only masked values')
                    check = True
            if check:
                continue
            print("\t {} and {}".format(*yvar[0]))
            fig_path_list += [savefig(make_scatter_plots(args,nc,ref,xvar,yvar,nc_time,ref_time,flag0,flagged,ref_flag0,kind=kind,freq=freq),code_dir,xvar,yvar,plot_type='sc')]
        elif type(yvar)==list: # difference plot
            check = False
            for v in yvar:
                if v not in nc.variables:
                    print('\t',v,'is not in the netCDF file')
                    check = True
                elif np.count_nonzero(nc[v][:].mask)==nc[v].size:
                    print('\t',v,'has only masked values')
                    check = True
            if check:
                continue
            print("\t {} minus {}".format(*yvar))
            fig_path_list += [savefig(make_scatter_plots(args,nc,ref,xvar,yvar,nc_time,ref_time,flag0,flagged,ref_flag0,kind=kind,freq=freq),code_dir,xvar,yvar,plot_type='sc')]
        else:
            if yvar not in nc.variables:
                print('\t',yvar,'is not in the netCDF file')
                continue
            elif np.count_nonzero(nc[yvar][:].mask)==nc[yvar].size:
                print('\t',yvar,'has only masked values')
                continue
            if kind:
                print(yvar,freq,kind)
            else:
                print('\t',yvar)
            fig_path_list += [savefig(make_scatter_plots(args,nc,ref,xvar,yvar,nc_time,ref_time,flag0,flagged,ref_flag0,kind=kind,freq=freq),code_dir,xvar,yvar,plot_type='sc')]
            close('all')

            if xvar!='time' and not np.count_nonzero([i in xvar for i in ['median','mean','std']]):
                fig_path_list += [savefig(make_hexbin_plots(args,nc,xvar,yvar,flag0),code_dir,xvar,yvar,plot_type='hex')]
                close('all')
    return fig_path_list


def default_plots(args,code_dir,nc,nc_time,flag0,flagged):
    """
    Make (70-80 SZA AM - 70-80 SZA PM) and (70-80 SZA PM - 40-50 SZA PM) plots for xluft
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
        ax.set_ylim(0.975,1.025)
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
        ax.set_ylim(0.975,1.025)
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
        ax.set_ylim(0.975,1.025)
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
    parser.add_argument('--show-all',action='store_true',help='if given, the axis ranges of the plots will automatically fit in all the data, even huge outliers')
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
            all_ref_ids = np.arange(ref['time'].size)
            if args.flag0 and 'flag' in ref.variables:
                ref_flag0 = all_ref_ids[ref['flag'][:]==0]
                ref_time = ref_time[ref_flag0]
            else:
                ref_flag0 = all_ref_ids
        else:
            ref = None
            ref_time = None
            ref_flag0 = None

        all_ids = np.arange(nc['time'].size)
        if 'flag' in nc.variables:
            flag0 = all_ids[nc['flag'][:]==0]
            flagged = all_ids[nc['flag'][:]!=0]
        else:
            flag0 = all_ids
            flagged = []

        if not args.flag0:
            fig_path_list += [flag_analysis(code_dir,nc)]

        print('Making default plots')
        fig_path_list += default_plots(args,code_dir,nc,nc_time,flag0,flagged)

        fnum = 0
        for xvar in vardata.keys():
            if xvar.endswith("median"):
                fig_path_list += simple_plots(args,code_dir,nc,ref,nc_time,ref_time,vardata,xvar,flag0,flagged,ref_flag0,kind='median',freq=xvar.split('_')[0])
                continue
            elif xvar.endswith("mean"):
                fig_path_list += simple_plots(args,code_dir,nc,ref,nc_time,ref_time,vardata,xvar,flag0,flagged,ref_flag0,kind='mean',freq=xvar.split('_')[0])
                continue
            elif xvar.endswith("std"):
                fig_path_list += simple_plots(args,code_dir,nc,ref,nc_time,ref_time,vardata,xvar,flag0,flagged,ref_flag0,kind='std',freq=xvar.split('_')[0])
                continue
            elif xvar not in [v for v in nc.variables]:
                print(xvar,'is not in the netCDF file')
                continue
            print('Making plots vs',xvar)
            fig_path_list += simple_plots(args,code_dir,nc,ref,nc_time,ref_time,vardata,xvar,flag0,flagged,ref_flag0)
            # end of "for yvar" loop
        # end of "for xvar" loop

        # concatenate all .png file into a temporary .pdf file
        temp_pdf_path = pdf_path.replace('.pdf','temp.pdf')
        im_list = [stack.enter_context(Image.open(fig_path).convert('RGB')) for fig_path in fig_path_list]
        im_list[0].save(temp_pdf_path, "PDF" ,resolution=100.0, save_all=True, append_images=im_list[1:])

        # add bookmarks to the temporary pdf file and save the final file
        pdf_in = stack.enter_context(open(temp_pdf_path,'rb'))
        pdf_out = stack.enter_context(open(pdf_path,'wb'))

        input = PdfFileReader(pdf_in)
        output = PdfFileWriter()
        
        for i,fig_path in enumerate(fig_path_list):
            output.addPage(input.getPage(i))
            output.addBookmark(os.path.basename(fig_path).strip('.jpg'),i)

        output.write(pdf_out)

    os.remove(temp_pdf_path) # remove the temporary pdf file

if __name__=="__main__":
    main()