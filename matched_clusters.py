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

#########################################################################


class matched_clusters(object):
    
    '''
    Builds matched_clusters class object containing the SPT and DES dataframes to match. 
    
    Parameters:
    spt_df: SPT dataframe
    des_df: RedMaPPer dataframe 
    z_lim: z_lim: np.array, tuple or list indicating lowest and highest Z range, e.g. [0.0, 1.0]
    '''
    
    #Input the respective dataframe files containing information for parameter estimation and matching. 
    def __init__(self, spt_df, des_df, z_lim = [0,2], verbose=True):
        assert isinstance(spt_df, pd.core.frame.DataFrame), "SPT file needs to be a dataframe"
        assert isinstance(des_df, pd.core.frame.DataFrame), "DES file needs to be a dataframe"
        
        if verbose:
            print("Number of clusters in SPT is:", len(spt_df))
            print("Number of clusters in DES is:", len(des_df))
            print("SPT columns: \n", list(spt_df.columns))
            print("DES columns: \n", list(des_df.columns))
                  
        self.spt_df = spt_df
        self.des_df = des_df
        
        #After applying this cut it messes up with the matched indexing. To fix after you come back. 
        self.__apply_z_cut(z_lim, verbose=True)
        
        ##A list of redMaPPer candidate and a rundown off associated matching properties within proximity matching space ranked order by richness
        self.match_candidates_df = self.spt_df.copy(deep=True) 
        
        ##Full combined properties of top redM candidate and SPT. 
        combined_column = self.spt_df.columns.tolist() + self.des_df.columns.tolist()
        self.match_top_df = pd.DataFrame(self.spt_df, index = self.spt_df.index, columns = combined_column)
        
        #SPT clusters with no matches
        self.nomatch_df = self.spt_df.copy(deep=True)
        
        #Record aperture information for finding the matches
        self.aperture = {"type": None, "size": None}
        
        self.nomatch_df
        
        return
    
    """
    Private function to reset the matched dataframes. Called internally by run_match before each matching. 
    """
    def __reset_matches(self):
        self.match_candidates_df = self.spt_df.copy(deep=True) 
        
        self.nomatch_df = self.spt_df.copy(deep=True) 
        
        combined_column = self.spt_df.columns.tolist() + self.des_df.columns.tolist()
        self.match_top_df = pd.DataFrame(self.spt_df, index = self.spt_df.index, columns = combined_column)
        
        self.aperture = {"type": None, "size": None}
    
        return
    
    '''
    Applies a redshift cut to the cluster dataframes.
    
    Parameter:
    z_lim: np.array, tuple or list with lowest and highest Z
    
    Returns:
    None
    '''
    def __apply_z_cut(self, z_lim, verbose=True):
        assert isinstance(z_lim, (np.ndarray, tuple, list)), "z_lim type must be of (np.ndarray, tuple or list)"
        assert len(z_lim) == 2, "Length of z_lim must be 2"
        
        z_low = z_lim[0]; z_high = z_lim[1]
        
        if verbose:
            print("Before z_cut number of SPT clusters is ", len(self.spt_df))
            print("Before z cut number of RM clusters is ", len(self.des_df))
        
        self.spt_df = self.spt_df[(self.spt_df.Z_SPT > z_low) & (self.spt_df.Z_SPT < z_high)]
        self.des_df = self.des_df[(self.des_df.Z_REDM > z_low) & (self.des_df.Z_REDM < z_high)]
        
        if verbose:
            print("After z cut number of SPT clusters is ", len(self.spt_df))
            print("After z cut number of RM clusters is ", len(self.des_df))
        
        return
        
    '''
    Chooses mutual matches between clusters by:
    1. First finding clusters within an aperture of given angular separation or distance separation if given a redshift. 
    2. If redshift information is given filter clusters within a \pm 0.05 Delta z 
    3. If there are multiple redM candidates rank order redM by richness and choose the one with highest values. Since SPT clusters are
    sparse we assume that the opposite case of multiple SPT clusters in a FOV to be unlikely. 
    4. Output the most likely candidate and a list of redM candidates that made it to step 3 rank ordering. 
    
    Adds the following columns to self.matched_df
    1. A tuple of redM candidates that made it to step 3
    2. distance sep
    3. (for 'distance aperture type') the redshift separation (RM_z - SPT_z) 
    ------------------------------------------------------------------
    Input:
    
    spt_params: list of column names in the spt_df corresponding to ['RA', 'DEC', 'Xi'] or ['RA', 'DEC', 'Xi', 'Z']. 
    des_params: list of column names in the des_df corresponding to ['RA', 'DEC', 'richness', 'Z'].
    sep_limit: separation limit for matching depending on the aperture type.
    delta_z_lim (optional): redshift separation limit between clusters in the form of delta_z = delta_z_lim/(1+z),
                            only applicable to aperture_type = distance
    aperture_type: 'ang_sep' for angular separation in arc minutes; 'distance' Proper distance in kpc/h assuming a WMAP9 cosmology.
    --------------------------------------------------------------------
    
    Returns:
    None
    '''    
    def run_match(self, spt_params, des_params, sep_limit, delta_z_lim=0.05, aperture_type = 'distance', verbose=True):
        assert aperture_type.lower() in ['distance', 'ang_sep'], "aperture_type input not recognized. "
        assert len(des_params) == 4, "des_params needs to be a list of strings in the format ['RA', 'DEC', 'richness', 'Z']"
        
        #Reset the dataframes match_top_df and top_candidates_df when running new match
        self.__reset_matches()
        
        self.aperture['type'] = aperture_type
        self.aperture['size'] = sep_limit
        
        #Extract DES parameters
        des_ra = self.des_df[des_params[0]].values; des_dec = self.des_df[des_params[1]].values
        des_richness = self.des_df[des_params[2]].values; des_z = self.des_df[des_params[3]].values 
          
        if aperture_type.lower() == 'distance':
            assert len(spt_params) == 4, "spt_params needs to be a list of strings in the format ['RA', 'DEC', 'Xi', 'Z']"
            
            #Extract SPT parameters
            spt_ra = self.spt_df[spt_params[0]].values; spt_dec = self.spt_df[spt_params[1]].values
            spt_xi = self.spt_df[spt_params[2]].values; spt_z = self.spt_df[spt_params[3]].values
            
            #Using find_nearest iterate through the array and find index of matched ones, also index of unmatched. 
            des_coord = SkyCoord(des_ra*u.deg, des_dec*u.deg,  frame='icrs')
            spt_coord = SkyCoord(spt_ra*u.deg, spt_dec*u.deg, frame='icrs')
            
            #RedM properties to extract for all candidates
            redM_ID_tot = []
            propD_sep_tot = []
            delta_z_tot = []
            richness_tot = []
            maskfrac_tot = []
            
            #RM properties to extract for the top candidate
            propD_sep_top = []
            delta_z_top = []
            
            if verbose: iterator = tqdm(range(len(spt_coord)))
            else: iterator = range(len(spt_coord))
            
            for i in iterator:
                sep = des_coord.separation(spt_coord[i]).value #In degrees
                propD_sep = (cosmo.kpc_proper_per_arcmin(spt_z[i])*sep*60).value 
                mask = propD_sep < sep_limit
                mask &= (np.abs(des_z - spt_z[i]) <= delta_z_lim/(1+spt_z[i]))
                           
                if any(mask):
                    #Find redM candidate attributes of current SPT cluster
                    redM_ID = self.des_df.loc[mask, 'MEM_MATCH_ID'].values
                    propD_sep = propD_sep[mask]
                    delta_z = np.array(des_z[mask] - spt_z[i])
                    richness = self.des_df.loc[mask, 'LAMBDA_CHISQ'].values
                    maskfrac = self.des_df.loc[mask, 'MASKFRAC'].values
                    
                    #Rank order by richness
                    rank_order = np.argsort(richness)[::-1] #Rank by richness in descending order
                    
                    redM_ID = redM_ID[rank_order]
                    propD_sep = propD_sep[rank_order]
                    delta_z = delta_z[rank_order]
                    richness = richness[rank_order]
                    maskfrac = maskfrac[rank_order]
                    
                    #Add to the _tot list
                    redM_ID_tot.append(redM_ID)
                    propD_sep_tot.append(propD_sep)
                    delta_z_tot.append(delta_z)  
                    richness_tot.append(richness)
                    maskfrac_tot.append(maskfrac)
                    
                    #Add to the _top list
                    propD_sep_top.append(propD_sep[0])
                    delta_z_top.append(delta_z[0])
                    
                    #ADD RM properties to the top_match_df
                    col_start_idx = len(self.spt_df.columns.tolist())
                    #Extract top match properties. 
                    top_match_ID = redM_ID[0] #top choice
                    match_redM = self.des_df[self.des_df['MEM_MATCH_ID'].eq(top_match_ID)] 
                    self.match_top_df.iloc[i,col_start_idx:] = match_redM.iloc[0,:]
                    
                else: #no matches
                    redM_ID_tot.append([float("NaN")])
                    propD_sep_tot.append([float("NaN")])
                    delta_z_tot.append([float("NaN")])
                    richness_tot.append([float("NaN")])
                    maskfrac_tot.append([float("NaN")])
                    
                    propD_sep_top.append([float("NaN")])
                    delta_z_top.append([float("NaN")])
                    
            #Include properties of the candidates and matching properties to match_candidates_df
            self.match_candidates_df = self.match_candidates_df.assign(MEM_MATCH_ID = redM_ID_tot)
            self.match_candidates_df = self.match_candidates_df.assign(propD_sep = propD_sep_tot)
            self.match_candidates_df = self.match_candidates_df.assign(Delta_Z = delta_z_tot)
            self.match_candidates_df = self.match_candidates_df.assign(LAMBDA_CHISQ = richness_tot)
            self.match_candidates_df = self.match_candidates_df.assign(MASKFRAC = maskfrac_tot)  
            
            #Add matching properties to the match_top_df.
            self.match_top_df = self.match_top_df.assign(propD_sep = propD_sep_top)
            self.match_top_df = self.match_top_df.assign(Delta_Z = delta_z_top)
            
            
        #To run and test this part. 
        elif aperture_type.lower() == 'ang_sep':
            assert len(spt_params) == 3, "spt_params needs to be a list of strings in the format ['RA', 'DEC', 'Xi']"
            
            #Extract SPT parameters
            spt_ra = self.spt_df[spt_params[0]].values; spt_dec = self.spt_df[spt_params[1]].values
            spt_xi = self.spt_df[spt_params[2]].values; 
            
            #Using find_nearest iterate through the array and find index of matched ones, also index of unmatched. 
            des_coord = SkyCoord(des_ra*u.deg, des_dec*u.deg,  frame='icrs')
            spt_coord = SkyCoord(spt_ra*u.deg, spt_dec*u.deg, frame='icrs')
            
            #RedM properties to extract for match_candidate_df
            redM_ID_tot = []
            ang_sep_tot = []
            richness_tot = []
            maskfrac_tot = []
            
            #RM properties to extract for match_top_df
            ang_sep_top = [] 
            
            if verbose: iterator = tqdm(range(len(spt_coord)))
            else: iterator = range(len(spt_coord))
            
            for i in iterator:
                ang_sep = des_coord.separation(spt_coord[i]).value #In degrees
                mask = ang_sep < sep_limit
                           
                if any(mask):
                    #Find redM candidate attributes of current SPT cluster
                    redM_ID = self.des_df.loc[mask, 'MEM_MATCH_ID'].values
                    ang_sep = ang_sep[mask]
                    richness = self.des_df.loc[mask, 'LAMBDA_CHISQ'].values
                    maskfrac = self.des_df.loc[mask, 'MASKFRAC'].values
                    
                    #Rank order by richness
                    rank_order = np.argsort(richness)[::-1] #Rank by richness in descending order
                    
                    redM_ID = redM_ID[rank_order]
                    ang_sep = ang_sep[rank_order]
                    richness = richness[rank_order]
                    maskfrac = maskfrac[rank_order]
                    
                    #Add to the _tot list
                    redM_ID_tot.append(redM_ID)
                    ang_sep_tot.append(ang_sep)
                    richness_tot.append(richness)
                    maskfrac_tot.append(maskfrac)
                    
                    #Add to the _top list
                    ang_sep_top.append(ang_sep[0])
                    
                    #Add RM properties to top_match_df
                    col_start_idx = len(self.spt_df.columns.tolist())
                    #Extract top match properties. 
                    top_match_ID = redM_ID[0] #top choice
                    match_redM = self.des_df[self.des_df['MEM_MATCH_ID'].eq(top_match_ID)] 
                    self.match_top_df.iloc[i,col_start_idx:] = match_redM.iloc[0,:]
                    
                else: #no matches
                    redM_ID_tot.append([float("NaN")])
                    ang_sep_tot.append([float("NaN")])
                    richness_tot.append([float("NaN")])
                    maskfrac_tot.append([float("NaN")])
                    
                    ang_sep_top.append([float("NaN")])
                    
            #Include properties of the candidates and matching properties.    
            self.match_candidates_df = self.match_candidates_df.assign(MEM_MATCH_ID = redM_ID_tot)
            self.match_candidates_df = self.match_candidates_df.assign(ANG_SEP = ang_sep_tot)
            self.match_candidates_df = self.match_candidates_df.assign(LAMBDA_CHISQ = richness_tot)
            self.match_candidates_df = self.match_candidates_df.assign(MASKFRAC = maskfrac_tot)
            
            #Add matching properties to the match_top_df.
            self.match_top_df = self.match_top_df.assign(ANG_SEP = ang_sep_top)
         
        
        #Filter data into matched and no matches
        match_idx = self.match_top_df[~pd.isna(self.match_top_df['MEM_MATCH_ID'])].index.tolist()
        nomatch_idx = self.match_top_df[pd.isna(self.match_top_df['MEM_MATCH_ID'])].index.tolist()
        
        self.match_candidates_df = self.match_candidates_df.loc[match_idx,:]
        self.match_top_df = self.match_top_df.loc[match_idx,:]
        self.nomatch_df = self.nomatch_df.loc[nomatch_idx,:]
            
        return

    '''
    Returns the dataframe of SPT candidates with matched properties. 
    '''
    def get_match_candidates_df(self):
        return self.match_candidates_df
    
    '''
    Returns the dataframe combining all columns of the SPT cluster with the top matched RM cluster
    '''
    def get_match_top_df(self):
        return self.match_top_df
    
    '''
    Returns the dataframe of SPT clusters with no matches
    '''
    def get_nomatch_df(self):
        return self.nomatch_df
    
    '''
    Gets the number of SPT clusters with a RM match. 
    '''
    def get_num_match(self):
        match_idx = self.match_top_df[~pd.isna(self.match_top_df['MEM_MATCH_ID'])].index.tolist()
        #nomatch_idx = self.match_top_df[pd.isna(self.match_top_df['MEM_MATCH_ID'])].index.tolist()
        return len(match_idx)
    

    '''
    Finds SPT clusters with more than one redMaPPer matched candidate found by the function match_run()

    Parameters:
    None

    Returns:
    A dictionary output containing the following keys:
    'multiple_candidate_df': Sub dataframe containing clusters with multiple matches
    'num_multiple': the number of clusters in the catalog with multiple candidates. 
    '''
    def get_multiple_candidates(self):
        #At least one match
        output = {}
        multiple_candidates_idx = self.match_candidates_df.apply(lambda x: len(x['MEM_MATCH_ID']) > 1, axis=1)
        multiple_candidate_df = self.match_candidates_df[multiple_candidates_idx]
        output['multiple_candidate_df'] = multiple_candidate_df
        output['num_multiple'] = len(np.where(multiple_candidates_idx)[0])
        return output
