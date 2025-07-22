################################################################################
# This module takes a directory of JWST calints files as input and builds a    #
# database of reference info for each file, which can then be read and         #
# filtered.                                                                    #
#                                                                              #
# Basically it will hold the target-specific info needed by the pipeline, as   #
# well as info needed to choose reference observations for a given science     #
# target.                                                                      #
#                                                                              #
# Written 2024-07-10 by Ellis Bogat                                            #
################################################################################

# TODO:
# - Create an add_obs() or update_refdb() function to add new files to the 
#   reference db without overwriting old data
# - Think about how to compare observations in the ref_db to science files 
#   not in the ref_db
# - Create a function to generate an OPD difference grid 
#   - We want the RMS OPD differences between two given dates
#   - How often to update this grid with new JWST observations? ~ 1/month?
#   - How finely to sample the OPD maps in time? Every 2 days?
#   - Estimate computation time as a function of # of dates to compare 

# imports
import os
import warnings
from glob import glob
import re
import mocapy
import pandas as pd
from astropy.io import fits
from astropy import units as u
from spaceKLIP import mast
import astropy
import stpsf
from tqdm import tqdm

from astroquery.simbad import Simbad
import numpy as np

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

sp_class_letters = ['O','B','A','F','G','K','M','T','Y']

sensitivitygrid_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'resources','sensitivity_loss_grids')

# Helper functions for logic with database series
def isnone(series):
    """Helper function to determine which elements of a series
    are NaN, None, or empty strings.

    Args:
        series (list-like): input data to check

    Returns:
        boolean array: True values where input array is NaN/None/''
    """

    try:
        return np.isnan(series)
    except:
        return (series == '') | (series == None) 


def not_isnone(series):
    
    return ~ isnone(series)

# Main functions
def load_refdb(fpath):
    """
    Reads the database of target- and observation-specific reference info for 
    each observation in the PSF library.

    Args:
        fpath (str or os.path): Path to the .csv file containing the reference
            database.

    Returns:
        pandas.DataFrame: database containing target- and observation-specific 
         info for each file.
    """

    refdb = pd.read_csv(fpath)
    refdb.set_index('TARGNAME',inplace=True)

    return refdb


def decode_simbad_sptype(input_sptypes):
    """Decodes the complicated SIMBAD spectral type string into simplified 
    spectral type components.

    Args:
        input_sptypes (str or list): SIMBAD spectral type string (or list of strings)

    Returns:
        tuple: list of spectral class letters, list of subclass numbers, 
            and list of luminosity class numerals for each input string.
    """

    # TODO:
        # - test this!
        # - figure out all the weird edge cases or find documentation 
        #   on SIMBAD spectypes

    if isinstance(input_sptypes,str):
        input_sptypes = [input_sptypes]

    
    sp_classes = []
    sp_subclasses = []
    sp_lclasses = []

    for simbad_spectype in input_sptypes:

        if not isinstance(simbad_spectype,str):
            warnings.warn('Spectral type decoder encountered a non-string spectral type. Skipping.')
            sp_classes.append('')
            sp_subclasses.append(np.nan)
            sp_lclasses.append('')
            continue

        m = re.search(r'([OBAFGKMTY])(\d\.*\d*)[-+/]*(\d\.*\d*)*[-+/]*(I*V*I*)[-/]*(I*V*I*)',simbad_spectype)
        
        if m is None:
            sp_classes.append('')
            sp_subclasses.append(np.nan)
            sp_lclasses.append('')
        else:
            res = m.group(1,2,3,4,5)
            
            sp_classes.append(res[0])
            sp_subclasses.append(res[1])
            if 'V' in res[3:] or res[3:] == ('',''):
                sp_lclasses.append('V')

            elif res[4] != '':
                sp_lclasses.append(res[4])

            else:
                sp_lclasses.append(res[3])

    return (sp_classes,sp_subclasses,sp_lclasses)


def adjust_spttype(spt_tup):
    # TODO: 
    # - Tests:
        # - input hotter than O0
        # - input colder than Y9
        # - subclass not int                
        # - known transitions: A10 -> F0, M-3 -> F7

    spt_class, spt_subclass = spt_tup

    spt_subclass = int(spt_subclass)

    if not spt_class in sp_class_letters:
        raise Exception(f'Invalid spectral class letter: {spt_class}')

    i_class = sp_class_letters.index(spt_class)

    while spt_subclass > 9:

        if spt_class == sp_class_letters[-1]:
            return (spt_class,9)

        i_class += 1
        spt_class = sp_class_letters[i_class]
        spt_subclass -= 10

    while spt_subclass < 0:

        if spt_class == sp_class_letters[0]:
            return (spt_class,0)
        
        i_class -= 1
        spt_class = sp_class_letters[i_class]
        spt_subclass += 10

    return (spt_class,spt_subclass)


def update_db_sptypes(refdb):
    """
    Separate the spectral type column into a column each for the
    spectral class letter, subclass number, and luminosity class numeral.

    Args:
        refdb (pandas.DataFrame): PSF reference dataframe

    Returns:
        pandas.DataFrame: updated PSF reference dataframe
    """

    simbad_sptypes = refdb.SPTYPE.values

    sp_classes, sp_subclasses, sp_lclasses = decode_simbad_sptype(simbad_sptypes)

    refdb_copy = refdb.copy()
    refdb_copy['SP_CLASS'] = sp_classes
    refdb_copy['SP_SUBCLASS'] = sp_subclasses
    refdb_copy['SP_LCLASS'] = sp_lclasses

    return refdb_copy


def spectype2specnum(spclasses,spsubclasses,splclasses=None):

    spclassnums = []
    for spclass in spclasses:
        try: spclassnums.append(sp_class_letters.index(spclass) * 10)
        except ValueError: spclassnums.append(np.nan)

    spclassnums = np.array(spclassnums).astype(float)
    spsubclasses = np.array(spsubclasses).astype(float)

    numerical_spclasses = spclassnums + spsubclasses

    return numerical_spclasses


def specnum2spectype(specnums):
    if isinstance(specnums,float):
        specnums = [specnums]

    specnums = np.array(specnums)

    spclassnums = (specnums // 10).astype(int)
    spsubclasses = specnums % 10

    spclasses = []
    for spclassnum in spclassnums:
        spclasses.append(sp_class_letters[spclassnum])

    return spclasses,spsubclasses

import pandas as pd
from scipy.interpolate import LinearNDInterpolator as Interpolator

def get_sensitivity_loss_interpolator(filt,mask,return_df=False):

    # Read in correct grid for sensitivities
    grid_path = os.path.join(sensitivitygrid_path,
                        f'sensitivityloss_mags_{filt}_{mask}.csv')
    
    if not os.path.exists(grid_path):
        warnings.warn(f'No sensitivity loss grid found for filter {filt} + mask {mask}. Skipping.')
        return None
    
    df = pd.read_csv(grid_path,
                     index_col='SCI_SPTYPE'
                    )
    
    # Convert the reference and science spectral types to numerical values
    ref_spectypes = list(df.columns)
    sci_spectypes = list(df.index)
    ref_specnums = spectype2specnum(*decode_simbad_sptype(ref_spectypes))
    sci_specnums = spectype2specnum(*decode_simbad_sptype(sci_spectypes))
    
    # df.columns = ref_specnums
    # df.index = sci_specnums
    
    # Flatten the values in the table so we can interpolate
    values = np.array(df).flatten()
    x,y = np.meshgrid(ref_specnums,sci_specnums)
    points = list(zip(x.flatten(),y.flatten()))
    interpolator = Interpolator(points=points,
                                values=values
                                )
    
    if return_df:
        return interpolator, df
    else:
        return interpolator


def get_sensitivity_loss(df,sci_spectype,filt,mask):
    
    df_temp = df.copy()
    interp = get_sensitivity_loss_interpolator(filt,mask) # Returns None if no grid file found.
 
    if not interp is None:
        ref_spectypes = df_temp['SPTYPE'].copy()
        sci_spectypes = [sci_spectype] * len(ref_spectypes)

        df_temp['SENSITIVITY_LOSS'] = interp(spectype2specnum(*decode_simbad_sptype(sci_spectypes)),
                                            spectype2specnum(*decode_simbad_sptype(ref_spectypes)))
        
        # Override places where sptypes are the same to make sure the sensitivity loss is zero
        df_temp.loc[(ref_spectypes==sci_spectypes),'SENSITIVITY_LOSS']

    else:
        df_temp['SENSITIVITY_LOSS'] = 0.

    df_temp.loc[df_temp['FILTER']!=filt,'SENSITIVITY_LOSS'] = np.nan
    df_temp.loc[df_temp['CORONMSK']!=mask,'SENSITIVITY_LOSS'] = np.nan

    
    return df_temp['SENSITIVITY_LOSS']
         

def filter_opdtable_for_daterange(start_date, end_date, opdtable):
    """Filter existing opdtable for a given time range
    This includes the last measurement in the prior time range too (if applicable), so we can compute a delta
    to the first one
    """
    # Start a little early, such that we are going to have at least 1 WFS before the start date
    pre_start_date = astropy.time.Time(start_date) - astropy.time.TimeDelta(4 * u.day)
    opdtable = stpsf.mast_wss.filter_opd_table(opdtable, start_time=pre_start_date, end_time=end_date)
    if len(opdtable) == 0:
        raise ValueError('The opdtable is empty for this date range.')

    # Trim the table to have 1 and only 1 precursor measurement -
    # we'll use this to compute the drift for the first WFS in the time period
    is_pre = [astropy.time.Time(row['date']) < start_date for row in opdtable]
    opdtable['is_pre'] = is_pre
    opdtable = opdtable[np.sum(is_pre) - 1:]

    return opdtable


def get_opdtable_for_daterange(start_date, end_date):
    """Return table of OPD measurements for date range.

    This includes the last measurement preceding this date range, too, so we
    can compute the first delta at the start of this range.
    """
    # Retrieve full OPD table, then trim to the selected time period
    opdtable0 = stpsf.mast_wss.retrieve_mast_opd_table()
    opdtable0 = stpsf.mast_wss.deduplicate_opd_table(opdtable0)

    opdtable = filter_opdtable_for_daterange(start_date, end_date, opdtable0)
    return opdtable


def get_opd_map(date_obs,time_obs,duration,verbose=False):

    time = date_obs+'T'+time_obs

    startT = astropy.time.Time(astropy.time.Time(time).mjd,format='mjd') - 1*u.day
    endT = astropy.time.Time(startT.mjd + duration/60/60/24,format='mjd') + 1*u.day
    opdtable = get_opdtable_for_daterange(startT, endT)
    index = np.argmin((opdtable['date_obs_mjd']-(astropy.time.Time(time).mjd + duration/60/60/24/2))**2)
    opd_fn = opdtable['fileName'][index]

    try:
        opd, opd_hdul = stpsf.trending._read_opd(opd_fn)
    except FileNotFoundError:
        stpsf.mast_wss.mast_retrieve_opd(opd_fn, verbose=verbose)
        opd, opd_hdul = stpsf.trending._read_opd(opd_fn)

    if opd.shape==(128,128):
        opd = opd.repeat(2,axis=1).repeat(2,axis=0)

    return opd


def compute_rms_OPD(ref_db,odir='.'):

    datesobs = np.array(ref_db['DATE-OBS'])
    times_obs = np.array(ref_db['TIME-OBS'])
    durations = np.array(ref_db['DURATION'])

    opd_maps = []
    for i in tqdm(range(len(ref_db)),leave=True):
        opd_i = get_opd_map(datesobs[i],times_obs[i],durations[i])
        opd_maps+=[opd_i]

    opd_maps = np.array(opd_maps)
    mask = opd_maps.sum(axis=0) != 0
    rms_grid = []

    for i in range(len(ref_db)):
        delta_opds = opd_maps - opd_maps[i]
        delta_rmses = [stpsf.utils.rms(d, mask=mask) * 1000 for d in delta_opds]
        rms_grid+=[delta_rmses]

    fnames = list(ref_db.FILENAME)

    rms_df = pd.DataFrame(rms_grid,columns=fnames,index=fnames)
    rms_df.to_csv(os.path.join(odir,'delta_opds.csv'))

    return rms_df


def filter_on_opds(sci_fpaths,
                   threshold=50, # nm rms wavefront error
                   inclusive=True, # Use references if they are a good match for any science target
                   odir='.'):
    """Reference the delta_opds.csv file to fetch the delta rms OPD between
    the science and potential reference files.

    Args:
        ref_db (pandas.DataFrame): reference database generated by build_refdb()
        sci_fpaths (list of str): list of science file paths to get delta OPD comparisons for
        inclusive (bool): Use references if they are a good OPD match for any science target,
            otherwise use references only if they are a good match for all science targets. 
            Defaults to True.
        odir (pathlike, optional): path to refdb and opd csvs. Defaults to '.'.
    """
    opd_fpath = os.path.join(odir,'delta_opds.csv')

    opd_df = pd.read_csv(opd_fpath,index_col='Unnamed: 0')
    
    refs = []
    for sci_fpath in sci_fpaths:
        opd_col = opd_df.loc[(opd_df[sci_fpath]<=threshold),
                             sci_fpath]
        refs.append(list(opd_col.index))

    final_refs = set(refs[0])
    if len(refs) > 1:
        for ref_list in refs[1:]:
            if inclusive:
                final_refs = final_refs.union(ref_list)
            else:
                final_refs = final_refs.intersection(ref_list)

    return list(final_refs)


def load_alignments(ref_db,odir='.'):

    ref_db_out = ref_db.copy()

    alignment_csvs = sorted(glob(os.path.join(odir,'mask_landings_f*.csv')))

    if len(alignment_csvs)==0:
        print(f'WARNING: No alignment .csvs found in directory {odir} !')
        print(f'psflib will not be able to filter based on mask-star alignments.')
        return None
    
    for i,csv_fname in enumerate(alignment_csvs):
        if i==0:
            alignment_df = pd.read_csv(csv_fname)
            alignment_df.columns=['FILENAME','MASKOFF_X','MASKOFF_Y']
            alignment_df.set_index('FILENAME',inplace=True)
        else:
            df = pd.read_csv(csv_fname)
            df.columns=['FILENAME','MASKOFF_X','MASKOFF_Y']
            df.set_index('FILENAME',inplace=True)
            alignment_df = pd.concat([alignment_df,df],axis=0)

    ref_db_out = ref_db_out.join(alignment_df)
        
    return ref_db_out


def build_refdb(idir,odir='.',suffix='calints',overwrite=False,
                query_MOCA=True,
                prefer_SIMBAD=True,
                ):
    """
    Constructs a database of target-specific reference info for each
    calints file in the input directory.

    Args:
        idir (str or list): Path to directory containing JWST images, or list of
            individual file paths, to construct the database from.
        odir (path, optional): Location to save the database. Defaults to '.'.
        suffix (str, optional): Input filename suffix, e.g. 'uncal' or 'calints'. 
            Defaults to 'calints'. Does not apply if list of fpaths is provided to idir.
        overwrite (bool, optional): If true, overwrite the existing caldb.
        prefer_SIMBAD (bool, optional): Default to choosing database info SIMBAD as opposed to MOCA
            database. Defaults to True.

    Returns:
        pandas.DataFrame: database containing target- and observation-specific 
         info for each file.
    """
    
    # TODO:
    # - describe each column & its units
    # - check for alignment csvs, print warnings for missing alignments
    # - write tests for build_refdb() 
    #       - directory vs filelist input
    #       - nonexistent input directory
    #       - nonexistent output directory
    #       - empty input directory 
    #       - no calints files in input directory
    #       - header kw missing
    #       - duplicate science target with different program names
    #       - synthetic PSFs 
    #       - slightly wrong SIMBAD names
    #       - missing alignment files
    #       - missing spectral type difference loss files 
    #        
    # - logic for if 'HAS_DISK','HAS_CANDS' have a mix of 'unknown' and bool values
    
    # Check that you won't accidentally overwrite an existing csv.
    outpath = os.path.join(odir,'ref_lib.csv')
    if os.path.exists(outpath):

        if overwrite:
            msg = f'\nThis operation will overwrite {outpath}. \nIf this is not what you want, abort now!'
            warnings.warn(msg)

        else:
            raise Exception(f'spaceklip.psflib.build_refdb() is trying to overwrite {outpath}.\nIf this is what you want, set overwrite=True.')

    # Read input files 
    log.info('Reading input files...')
    suffix = suffix.strip('_')
    if isinstance(idir,str):
        fpaths = sorted(glob(os.path.join(idir,f"*_{suffix}.fits")))
        if len(fpaths) == 0:
            raise Exception(f'No "{suffix}" files found in input directory {idir} .')
    elif isinstance(idir,list):
        fpaths = idir
        for path in fpaths:
            if not (f'{suffix}.fits' in path and os.path.exists(path)):
                fpaths.remove(path)
        if len(fpaths) == 0:
            raise Exception(f'No existing "{suffix}" files found in input file list.')
            
    # Start a dataframe with the header info we want from each file
    csv_list = []
    fits_cols = [
        'TARGPROP',
        'TARGNAME', # Save 2MASS ID also
        'FILENAME',
        'OBS_ID',
        'DATE-OBS',
        'TIME-OBS',
        'DURATION', # Total exposure time 
        'TARG_RA',
        'TARG_DEC',
        'TARGURA', # RA uncertainty
        'TARGUDEC', # Dec uncertainty
        'MU_RA', # Proper motion
        'MU_DEC',                
        'MU_EPOCH',
        'INSTRUME',
        'DETECTOR',
        'MODULE',
        'CHANNEL',
        'FILTER',
        'CORONMSK',
        'PUPIL',
    ]

    for fpath in fpaths:
        row = []
        hdr = fits.getheader(fpath)
        for col in fits_cols:
            row.append(hdr[col])
        csv_list.append(row)

    df = pd.DataFrame(csv_list,columns=fits_cols)

    # Make a df with only one entry for each unique target
    targnames = np.unique(df['TARGNAME']) 
    df_unique = pd.DataFrame(np.transpose([targnames]),columns=['TARGNAME'])
    # Get 2MASS IDs
    log.info('Collecting SIMBAD IDs...')
    i = 0
    simbad_ids = []
    for targname in targnames:
        result_table = Simbad.query_objectids(targname)
        
        if result_table is None or len(result_table)==0:
            simbad_ids.append(f'UNKNOWN STAR {i}')
            i += 1
            warnings.warn(f'No SIMBAD object found for targname {targname}, this is likely a synthetic PSF.') 
            continue  

        gaia_ids_found = []
        for name in list(result_table['ID']):
            if name.startswith('Gaia DR3'):
                gaia_ids_found.append(name) 
        if len(gaia_ids_found) == 1:
            simbad_ids.extend(gaia_ids_found)
        elif len(gaia_ids_found) < 1:
            simbad_ids.append(list(result_table['ID'])[0])
        else:
            raise Exception(f'Multiple Gaia DR3 IDs found for targname {targname}: {gaia_ids_found}')
            
    df_unique['SIMBAD_ID'] = simbad_ids
    df_unique.set_index('SIMBAD_ID',inplace=True)

    # Query SIMBAD
    log.info('Querying SIMBAD...')
    customSimbad = Simbad()
    customSimbad.add_votable_fields('sptype', 
                                    'flux(K)', 'flux_error(K)', 
                                    'plx', 'plx_error')
    simbad_list = list(df_unique.index)
    short_simbad_list = []
    for st_name in simbad_list:
        if not st_name.startswith('UNKNOWN STAR'):
            short_simbad_list.append(st_name)
    scistar_simbad_table = customSimbad.query_objects(short_simbad_list)

    # Convert to pandas df and make SIMBAD IDs the index
    short_df_simbad = scistar_simbad_table.to_pandas()
    short_df_simbad['SIMBAD_ID'] = short_simbad_list
    short_df_simbad.set_index('SIMBAD_ID',inplace=True)
    short_df_simbad['SIMBAD_ID'] = short_simbad_list # Add SIMBAD_ID as a column in addition to the index
    # Add empty rows for stars not in SIMBAD (e.g. synthetic PSFs)
    df_simbad = pd.DataFrame(index=simbad_list, columns=short_df_simbad.columns, dtype='object')
    df_simbad.loc[short_df_simbad.index] = short_df_simbad.values

    # Rename some columns
    simbad_cols = { # Full column list here: http://simbad.u-strasbg.fr/Pages/guide/sim-fscript.htx 
        'SPTYPE': 'SP_TYPE', # maybe use 'simple_spt' or 'complete_spt'?
        'KMAG': 'FLUX_K', # 'kmag'
        'KMAG_ERR': 'FLUX_ERROR_K', # 'ekmag'
        'PLX': 'PLX_VALUE', # 'plx'
        'PLX_ERR': 'PLX_ERROR', # 'eplx'
        'SIMBAD_ID': 'SIMBAD_ID'
        }
    for col,simbad_col in simbad_cols.items():
        df_simbad[col] = list(df_simbad[simbad_col])

    # Add the values we want to df_unique
    df_unique = pd.concat([df_unique,df_simbad.loc[:,simbad_cols.keys()]],axis=1)

    # Sort all the dfs by index and check that they match up
    df_unique.sort_index(inplace=True)   
    df_simbad.sort_index(inplace=True)  
    assert np.all(np.array(df_unique.index)==np.array(df_simbad.index)), "Index Error"

    df_unique['COMMENTS'] = ''
    df_unique['DB_SOURCES'] = ''

    # Query mocadb.ca for extra info
    if query_MOCA:
        log.info('Querying MOCADB (this may take a minute)...')
        names_df = pd.DataFrame(list(df_unique.index),columns=['designation'])
        moca = mocapy.MocaEngine()
        mdf = moca.query("SELECT tt.designation AS input_designation, sam.* FROM tmp_table AS tt LEFT JOIN mechanics_all_designations AS mad ON(mad.designation LIKE tt.designation) LEFT JOIN summary_all_objects AS sam ON(sam.moca_oid=mad.moca_oid)", tmp_table=names_df)
        mdf.set_index('input_designation',inplace=True)

        moca_cols = {
            'SPTYPE': 'spt', # maybe use 'simple_spt' or 'complete_spt'?
            'PLX': 'plx', # 'plx'
            'PLX_ERR': 'eplx', # 'eplx'
            'AGE': 'age', # 'age'
            'AGE_ERR': 'eage', # 'eage'
        }

        # Update the column names for consistency
        for col,moca_col in moca_cols.items():
            mdf[col] = list(mdf[moca_col])

        mdf.sort_index(inplace=True)   
        assert np.all(np.array(df_unique.index)==np.array(mdf.index)), "Index Error"

        # Fill in values missing from SIMBAD with MOCA (or vice versa if prefer_SIMBAD==False)

        # Replace values and update DB_SOURCES column
        cols_overlap = list(set(list(simbad_cols.keys())).intersection(list(moca_cols.keys())))
        for col in cols_overlap:
            if prefer_SIMBAD:
                df_unique.loc[~isnone(df_simbad[col]),'DB_SOURCES'] += f"{col} adopted from SIMBAD. "
                df_unique.loc[isnone(df_simbad[col]) & ~isnone(mdf[col]),'DB_SOURCES'] += f"{col} adopted from MOCA. "
                df_unique.loc[isnone(df_simbad[col]) & ~isnone(mdf[col]),col] = mdf
            else:
                df_unique.loc[~isnone(mdf[col]),'DB_SOURCES'] += f"{col} adopted from MOCA. "
                df_unique.loc[isnone(mdf[col]) & ~isnone(df_simbad[col]),'DB_SOURCES'] += f"{col} adopted from SIMBAD. "
                df_unique.loc[isnone(mdf[col]) & ~isnone(df_simbad[col]),col] = df_simbad

        # Stellar ages only exist in MOCA database.
        for col in ['AGE','AGE_ERR']:
            df_unique[col] = mdf[col]
            df_unique.loc[~isnone(mdf[col]),'DB_SOURCES'] += f"{col} adopted from MOCA. "

    # Calculate distances from plx in mas
    df_unique['DIST'] = 1. / (df_unique['PLX'] / 1000)
    df_unique['DIST_ERR'] = df_unique['PLX_ERR'] / 1000 / ((df_unique['PLX'] / 1000)**2)

    # Decode spectral types
    df_unique = update_db_sptypes(df_unique)

    # Add empty columns
    manual_cols = [
        'FLAGS',
        'HAS_DISK',
        'HAS_CANDS']
    
    for col in manual_cols:
        df_unique[col] = 'unknown'

    # Apply dataframe of unique targets to the original file list
    df.set_index('TARGNAME',inplace=True)
    df_unique.reset_index(inplace=True)
    df_unique.set_index('TARGNAME',inplace=True)
    df_unique = df_unique.reindex(df.index)
    df_out = pd.concat([df,df_unique],axis=1)
    
    # Compute delta OPD table
    print('Computing delta OPDs...')
    compute_rms_OPD(df_out,odir=odir)
    print('Done!')

    # Load mask offsets
    df_out.reset_index(inplace=True)
    df_out.set_index('FILENAME',inplace=True)
    df_with_alignments = load_alignments(df_out,odir='.')
    if df_with_alignments is None:
        df_with_alignments = df_out
    df_with_alignments.reset_index(inplace=True)
    df_with_alignments.set_index('TARGNAME',inplace=True)
    
    # Save dataframe
    df_with_alignments.to_csv(outpath)
    log.info(f'Database saved to {outpath}')

    return df_with_alignments


def get_sciref_files(sci_target, refdb, 
                     scifiles = None,
                     idir=None, odir='.',
                     sci_dir=None,
                     spt_tolerance=None, 
                     spt_loss_tolerance=0.5,
                     filters=None, 
                     opd_threshold=None,
                     opd_inclusive=True,
                     alignment_xthreshold=None, # pix
                     alignment_ythreshold=None, # pix
                     alignment_zthreshold=None, # pix
                     alignment_inclusive=True,
                     snr_threshold=None, # not configured
                     exclude_disks=False):
    """Construct a list of science files and reference files to input to a PSF subtraction routine.

    Args:
        sci_target (str): 
            name of the science target to be PSF subtracted. Can be the proposal target name, 
            JWST resolved target name.
        refdb (pandas.DataFrame or str): 
            pandas dataframe or filepath to csv containing the reference database generated by 
            the build_refdb() function.
        idir (str):
            path to directory of input data, to be appended to file names.
        spt_tolerance (str or int, optional): 
            None (default): use all spectral types.
            'exact' : use only refs with the exact same spectral type.
            'class' : use only references with the same spectral class letter.
            'loss' : use sensitivity loss grids with a threshold equal to the spt_loss_tolerance
            int : use only refs within +- N spectral subclasses, e.g. M3-5 for an M4 science target if spt_tolerance = 1.
        spt_loss_tolerance (float):
            Threshold for sensitivity loss when referencing the sensitivity loss grids (only used if spt_tolerance='loss', 
            defaults to 0.5)
        filters (str or list, optional): 
            None (default) : include all filters.
            'F444W' or other filter name: include only that filter.
            ['filt1','filt2']: include only filt1 and filt2
        exclude_disks (bool, optional): Exclude references that are known to have disks. Defaults to False.

    Returns:
        list: filenames of science observations.
        list: filenames of reference observations.
    """

    # TODO:
        # - filter by the sensitivity loss grid if available
        # - filter by mask_offset columns if available 
        # - fix the thing where mask_offset filters don't know about different
        #   wavelength filts
        # - skip filters when reference data is missing
        # - add warning if filenames are missing from opd or alignment csvs
        # - filter out manual flags
        # - add capability to specify particular science filenames instead of science target name
        #   - choose filters automatically, require that all files are the same target.

    if isinstance(refdb,str):
        refdb = load_refdb(refdb)

    if sci_dir is None:
        sci_dir = idir

    # Locate input target 2MASS ID 
    # (input name could be in index, TARGPROP, or SIMBAD_ID column)
    if sci_target in refdb['SIMBAD_ID'].to_list():
        targname = sci_target

    elif sci_target in refdb.index.to_list():
        targname = refdb.loc[sci_target,'SIMBAD_ID']

    elif sci_target in refdb['TARGPROP'].to_list():
        refdb_temp = refdb.reset_index()
        refdb_temp.set_index('TARGPROP',inplace=True)
        targname = refdb_temp.loc[sci_target,'SIMBAD_ID']
    
    else:
        log.error(f'Science target {sci_target} not found in reference database.')
        raise Exception(f'Science target {sci_target} not found in reference database.')
    
    if isinstance(targname, pd.Series):
        targname = targname.to_list()[0]
        refdb_temp = refdb.reset_index()
    refdb_temp.set_index('FILENAME',inplace=True)

    # Collect all the science files
    if scifiles ==None:
        sci_fnames = refdb_temp.index[refdb_temp['SIMBAD_ID'] == targname].to_list()
    else:
        sci_fnames = scifiles
    
    first_scifile = sci_fnames[0]

    ### Collect the reference files

    # Start list of reference files
    ref_fnames = refdb_temp.index[refdb_temp['SIMBAD_ID'] != targname].to_list()

    ## Sort out filters
    if filters != None:
        if isinstance('filter',str):
            filters = [filters]
        filter_fnames = []
        for filter in filters:
            filter_fnames.extend(refdb_temp.index[refdb_temp['FILTER'] == filter].to_list())
        if len(filter_fnames) == 0:
            raise Warning(f'No observations found with filters {filters}.')
        
        sci_fnames = list(set(sci_fnames).intersection(filter_fnames))
        ref_fnames = list(set(ref_fnames).intersection(filter_fnames))
    else:
        filters = set(refdb_temp.loc[sci_fnames,'FILTER'])

    ## Sort out spectral types
    if spt_tolerance != None: 

        # Calculate numerical spectral types
        refdb_temp['SP_NUM'] = spectype2specnum(refdb_temp['SP_CLASS'],refdb_temp['SP_SUBCLASS'],refdb_temp['SP_LCLASS'])
        targ_sp_num = refdb_temp.loc[first_scifile,'SP_NUM']

        if isinstance(spt_tolerance,float) or isinstance(spt_tolerance,int):
            spt_fnames = refdb_temp.index[(refdb_temp['SP_NUM'] >= (targ_sp_num - spt_tolerance)) &
                                            (refdb_temp['SP_NUM'] <= (targ_sp_num + spt_tolerance))
                                            ].to_list()
        
        elif isinstance(spt_tolerance,str):
            if spt_tolerance.lower() == 'exact':

                spt_fnames = refdb_temp.index[(refdb_temp['SP_NUM'] == targ_sp_num)
                                            ].to_list()
                
            elif spt_tolerance.lower() == 'class':
                targ_sp_class = refdb_temp.loc[first_scifile,'SP_CLASS']
                spt_fnames = refdb_temp.index[(refdb_temp['SP_CLASS'] == targ_sp_class)
                                                ].to_list()
            
            elif spt_tolerance.lower() == 'loss':

                # Need to treat each filter/mask combo separately

                # Collect (filter, mask) pairs
                filts = refdb_temp.loc[sci_fnames,'FILTER']
                masks = refdb_temp.loc[sci_fnames,'CORONMSK']
                filter_mask_pairs = set(zip(filts,masks))

                spt_fnames = []
                for filt, mask in filter_mask_pairs:
                    sensitivity_loss = get_sensitivity_loss(refdb_temp,
                                                            refdb_temp.loc[first_scifile,'SPTYPE'],
                                                            filt,mask)
                    spt_fnames.extend(refdb_temp.index[(sensitivity_loss < spt_loss_tolerance)
                                                        ].to_list())
                    # print(refdb_temp.loc[first_scifile,'SPTYPE'],mask,filt)
                    # print(refdb_temp.loc[(sensitivity_loss < spt_loss_tolerance),'SPTYPE'
                    #                                     ].to_list())
                    
            else:
                raise Exception(f'spt_tolerance {spt_tolerance} not configured.')
                    
        else:
            raise Exception(f'spt_tolerance is not string, float, or int.')
        
    
        if len(spt_fnames) == 0:
            raise Warning(f'No observations found with specified spectral type filter.')
        
        sci_fnames = list(set(sci_fnames).intersection(spt_fnames))
        ref_fnames = list(set(ref_fnames).intersection(spt_fnames))


    if opd_threshold != None:

        opd_ref_fnames = filter_on_opds(sci_fpaths=sci_fnames,
               threshold=opd_threshold,
               inclusive=opd_inclusive,
               odir=odir)
        
        ref_fnames = list(set(ref_fnames).intersection(opd_ref_fnames))
        
    if alignment_xthreshold != None:
        if not 'MASKOFF_X' in refdb_temp.columns:
            print(
                'WARNING: X_OFFSET column missing from reference database, likely because alignment csvs are missing! ' \
                'Skipping alignment restriction.'
            )
        else:
            x_refs = []
            for sci_fpath in sci_fnames:
                x_off = refdb_temp.loc[sci_fpath,'MASKOFF_X']
                x_refs.append(refdb_temp.index[(refdb_temp['MASKOFF_X'] <= x_off+alignment_xthreshold) &
                                            (refdb_temp['MASKOFF_X'] >= x_off-alignment_xthreshold)
                                        ].to_list())

            final_xrefs = set(x_refs[0])
            if len(x_refs) > 1:
                for ref_list in x_refs[1:]:
                    if alignment_inclusive:
                        final_xrefs = final_xrefs.union(ref_list)
                    else:
                        final_xrefs = final_xrefs.intersection(ref_list)

            ref_fnames = list(set(ref_fnames).intersection(final_xrefs))

    if alignment_ythreshold != None:
        if not 'MASKOFF_Y' in refdb_temp.columns:
            print(
                'WARNING: Y_OFFSET column missing from reference database, likely because alignment csvs are missing! ' \
                'Skipping alignment restriction.'
            )
        else:
            y_refs = []
            for sci_fpath in sci_fnames:
                y_off = refdb_temp.loc[sci_fpath,'MASKOFF_Y']
                y_refs.append(refdb_temp.index[(refdb_temp['MASKOFF_Y'] <= y_off+alignment_ythreshold) &
                                            (refdb_temp['MASKOFF_Y'] >= y_off-alignment_ythreshold)
                                        ].to_list())

            final_yrefs = set(y_refs[0])
            if len(y_refs) > 1:
                for ref_list in y_refs[1:]:
                    if alignment_inclusive:
                        final_yrefs = final_yrefs.union(ref_list)
                    else:
                        final_yrefs = final_yrefs.intersection(ref_list)
            
            ref_fnames = list(set(ref_fnames).intersection(final_yrefs))

    if alignment_zthreshold != None:
        if not 'MASKOFF_Y' in refdb_temp.columns:
            print(
                'WARNING: X/Y_OFFSET columns missing from reference database, likely because alignment csvs are missing! ' \
                'Skipping alignment restriction.'
            )
        else:
            refdb_temp['MASKOFF_Z'] = np.sqrt(refdb_temp['MASKOFF_X']**2 + refdb_temp['MASKOFF_Y']**2)
            z_refs = []
            for sci_fpath in sci_fnames:
                z_off = refdb_temp.loc[sci_fpath,'MASKOFF_Z']
                z_refs.append(refdb_temp.index[(refdb_temp['MASKOFF_Z'] <= z_off+alignment_zthreshold) &
                                            (refdb_temp['MASKOFF_Z'] >= z_off-alignment_zthreshold)
                                        ].to_list())

            final_zrefs = set(z_refs[0])
            if len(z_refs) > 1:
                for ref_list in z_refs[1:]:
                    if alignment_inclusive:
                        final_zrefs = final_zrefs.union(ref_list)
                    else:
                        final_zrefs = final_zrefs.intersection(ref_list)
            
            ref_fnames = list(set(ref_fnames).intersection(final_zrefs))
    
    # Remove observations with disks flagged
    if exclude_disks:
        disk_fnames = refdb_temp.index[refdb_temp['HAS_DISK'] == True].to_list()
        ref_fnames = list(set(ref_fnames) - set(disk_fnames))
    
    # Make sure no observations are in both sci_fnames and ref_fnames
    if len(set(sci_fnames).intersection(ref_fnames)) > 0:
        raise Exception("One or more filenames exists in both the science and reference file list. Something is wrong.")

    if not idir is None:
        sci_fpaths = [os.path.join(sci_dir,sci_fname) for sci_fname in sci_fnames]
        ref_fpaths = [os.path.join(idir,ref_fname) for ref_fname in ref_fnames]
    else:
        sci_fpaths = sci_fnames
        ref_fpaths = ref_fnames
        
    return [sci_fpaths, ref_fpaths]


def download_mast(ref_db,token=None,
                  overwrite=False,exists_ok=True,
                  progress=False, verbose=False,
                  suffix=None, # e.g. 'calints'
                  base_dir=os.path.join('DATA','MAST_DOWNLOAD')):
    """Given the reference database, downloads MAST data for each file to base_dir.
    If a file exists already, default is to not download.
    Set overwrite=True to overwrite existing output file.
    or set exists_ok=False to raise ValueError.

    Set progress=True to show a progress bar.

    Args:
        ref_db (pandas.DataFrame): The reference database
        token (str, optional): MAST token. Default is to reference the environment 
            variable MAST_API_TOKEN.
    """
    
    fnames = list(ref_db.FILENAME)

    # Update file suffix if provided
    if not suffix == None:
            new_suffix = suffix.strip('_')

            for ff,fname in enumerate(fnames):
                fname_split = fname.split('_')
                new_fname = '_'.join(fname_split[:-1]) + f'_{new_suffix}.fits'

                fnames[ff] = new_fname    
        
    # Download each file
    for fname in fnames:
        
        mast.get_mast_filename(fname,
                               outputdir=base_dir,
                               overwrite=overwrite, exists_ok=exists_ok,
                               progress=progress, verbose=verbose,
                               mast_api_token=token)

