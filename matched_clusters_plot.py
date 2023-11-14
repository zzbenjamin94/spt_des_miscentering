import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from glob import glob
import numpy as np
import healpy as hp
import struct

import setup
from astropy.table import Table
#Coordinate
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.cosmology import WMAP9 as cosmo
from astropy.coordinates import SkyCoord
from astropy.io import fits

import pandas as pd
from tqdm import tqdm
import math

import pickle

#Plotting params
#Plotting parameters
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as mticker
from matplotlib.ticker import StrMethodFormatter, NullFormatter


mpl.rcParams['xtick.direction'], mpl.rcParams['ytick.direction'] = 'in', 'in'
mpl.rcParams['xtick.major.size'], mpl.rcParams['xtick.minor.size'] = 14, 8
mpl.rcParams['xtick.major.width'], mpl.rcParams['xtick.minor.width'] = 1.2, 0.8
mpl.rcParams['xtick.major.pad'], mpl.rcParams['xtick.minor.pad'] = 10, 10
mpl.rcParams['ytick.major.size'], mpl.rcParams['ytick.minor.size'] = 14, 8
mpl.rcParams['ytick.major.width'], mpl.rcParams['ytick.minor.width'] = 1.2, 0.8
mpl.rcParams['xtick.labelsize'], mpl.rcParams['ytick.labelsize'] = 20, 20

# Parameters used in this module
# One dictionary to store default values
# Another that user can view/change as necessary
Default_Params = {'default_cmap'   : plt.cm.coolwarm,
                  'title_fontsize' : 25,
                  'legend_fontsize': 22,
                  'xlabel_fontsize': 30,
                  'ylabel_fontsize': 30,
                  'scatter_factor' : 1.0}

Params = Default_Params.copy()

'''
Plot scaling relations. 
Modifications:
* Changed to log-log plots
* Changed the x/y tickmark locations and from scientific format to integer for prettier plot
* Changed the colormap and the location of the colormap
'''
def plot_scaling_relations(match_top_df, maskfrac=True, savefig=False, figname='foo'):
            #match_idx = self.match_top_df[~pd.isna(self.match_top_df['MEM_MATCH_ID'])].index.tolist()
            #nomatch_idx = self.match_top_df[pd.isna(self.match_top_df['MEM_MATCH_ID'])].index.tolist()
            #print('Num match', len(np.where(matched_idx)[0]))
            #print('Num no-match', len(np.where(nomatch_idx)[0]))

            #Need to change this. Make this work. 
        fig, axs = plt.subplots(1, 3, figsize=(16,5))
        #[a.grid(True) for a in axs]
        fig.subplots_adjust(wspace = 0.3, right=0.85)

        axs[0].scatter(match_top_df.loc[:, 'XI'], match_top_df.loc[:, 'LAMBDA_CHISQ'], c=match_top_df.loc[:, 'MASKFRAC'], cmap='RdBu')
        axs[1].scatter(match_top_df.loc[:, 'M500'].values, match_top_df.loc[:, 'LAMBDA_CHISQ'].values, c=match_top_df.loc[:, 'MASKFRAC'].values, cmap='RdBu')
        sc = axs[2].scatter(match_top_df.loc[:, 'Z_REDM'], match_top_df.loc[:, 'XI'], c=match_top_df.loc[:, 'MASKFRAC'], cmap='RdBu')


        #Make into log-log
        axs[0].set_yscale('log'); axs[0].set_xscale('log')
        axs[1].set_yscale('log'); axs[1].set_xscale('log')

        #axs[1].xaxis.set_major_formatter()
        richness_positions = [30,40,60,100]
        xi_positions = [5,7,10, 20,30]
        M500_positions = [1,2,5,10] 
        axs[0].yaxis.set_minor_locator(mticker.FixedLocator(richness_positions))
        axs[0].xaxis.set_minor_locator(mticker.FixedLocator(xi_positions))
        axs[1].yaxis.set_minor_locator(mticker.FixedLocator(richness_positions))
        axs[1].xaxis.set_minor_locator(mticker.FixedLocator(M500_positions))     

        axs[0].xaxis.set_major_formatter(FormatStrFormatter('%d'))
        axs[0].xaxis.set_minor_formatter(FormatStrFormatter('%d'))
        axs[0].yaxis.set_minor_formatter(FormatStrFormatter('%d'))
        axs[0].yaxis.set_major_formatter(FormatStrFormatter('%d'))
        axs[1].xaxis.set_minor_formatter(FormatStrFormatter('%d'))
        axs[1].xaxis.set_major_formatter(FormatStrFormatter('%d'))
        axs[1].yaxis.set_minor_formatter(FormatStrFormatter('%d')) 
        axs[1].yaxis.set_major_formatter(FormatStrFormatter('%d'))    

        #labels
        axs[0].set_xlabel(r'$\xi$', fontsize=16); axs[0].set_ylabel(r'$\lambda$', fontsize=16)
        axs[1].set_xlabel(r'$M_{500}~(\times 10^{14} M_{\odot})$ ', fontsize=16); axs[1].set_ylabel(r'$\lambda$', fontsize=16); 
        axs[2].set_xlabel(r'$Z_{\lambda}$', fontsize=16); axs[2].set_ylabel(r'$\xi$', fontsize=16)


        cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
        cbar = fig.colorbar(sc, cax=cbar_ax)
        cbar.set_label('maskfrac', y = 1.1, labelpad=-60, rotation=0, fontsize=16)
        
        
        if savefig:
            plt.savefig('matched_scaling_' + figname + '.png')
        
        plt.plot()

        return 
    
'''
Plots the properties of the matched clusters. 
*pairplot of distributions (Z, M, Xi, Lambda, D_sep, z_sep)
'''
def plot_matched_properties(match_top_df, field=False, savefig=False, figname='foo'):
        
        plt.figure(tight_layout=True)
        if field:
            col_compare = ['XI', 'LAMBDA_CHISQ', 'propD_sep', 'Delta_Z', 'MASKFRAC', 'FIELD']
            match_top_pairplot_df = match_top_df[col_compare]
            sns.pairplot(match_top_pairplot_df, corner=True, hue='FIELD', diag_kind='hist', \
                         diag_kws=dict(stat="probability", kde=True, alpha=0, linewidth=0, bins=10,common_norm=False))
            
        else: #No field parameter
            col_compare = ['XI', 'LAMBDA_CHISQ', 'propD_sep', 'Delta_Z', 'MASKFRAC']
            match_top_pairplot_df = match_top_df[col_compare]
            sns.pairplot(match_top_pairplot_df, corner=True, diag_kind='hist', \
                         diag_kws=dict(stat="probability", kde=True, alpha=0, linewidth=0, bins=10,common_norm=False))
            
        if savefig:
            plt.savefig('matched_pairplot_' + figname + '.png')
            
        plt.plot()
        
        return
    
'''
Plots the properties of SPT clusters with no match. Creates a pairplot with (Xi, M500 and Z_SPT) with SPT field specified.
'''
def plot_nomatch_properties(nomatch_df, field=False, savefig=False, figname='foo'):
        plt.figure(tight_layout=True)
        if field:
            col_compare = ['XI', 'M500', 'Z_SPT', 'FIELD']
            nomatch_pairplot_df = nomatch_df[col_compare]
            sns.pairplot(nomatch_pairplot_df, corner=True, hue='FIELD', diag_kind='hist', \
                         diag_kws=dict(stat="probability", kde=True, alpha=0, linewidth=0, bins=10,common_norm=False))
        
        else:
            col_compare = ['XI', 'M500', 'Z_SPT']
            nomatch_pairplot_df = nomatch_df[col_compare]
            sns.pairplot(nomatch_pairplot_df, corner=True, diag_kind='hist', \
                     diag_kws=dict(stat="probability", kde=True, alpha=0, linewidth=0, bins=10,common_norm=False))
            
        if savefig:
            plt.savefig('nomatch_pairplot_' + figname + '.png')
        
        plt.plot()
        return