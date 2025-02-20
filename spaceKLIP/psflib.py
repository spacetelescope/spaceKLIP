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
import glob
import re
import mocapy
import pandas as pd
from astropy.io import fits
from spaceKLIP import mast

from astroquery.simbad import Simbad
import numpy as np

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

sp_class_letters = ['O','B','A','F','G','K','M','T','Y']

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
            sp_subclasses.append('')
            sp_lclasses.append('')
            continue

        m = re.search(r'([OBAFGKMTY])(\d\.*\d*)[-+/]*(\d\.*\d*)*[-+/]*(I*V*I*)[-/]*(I*V*I*)',simbad_spectype)
        
        if m is None:
            sp_classes.append('')
            sp_subclasses.append('')
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


def build_refdb(idir,odir='.',suffix='calints',overwrite=False,
                prefer_SIMBAD=True):
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
    # - write tests for build_refdb() 
    #       - directory vs filelist input
    #       - nonexistent input directory
    #       - empty input directory 
    #       - no calints files in input directory
    #       - header kw missing
    #       - duplicate science target with different program names
    #       - synthetic PSFs 
    #       - slightly wrong SIMBAD names
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
        fpaths = sorted(glob.glob(os.path.join(idir,f"*_{suffix}.fits")))
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
    short_simbad_list = list(df_unique.index)
    for st_name in short_simbad_list:
        if st_name.startswith('UNKNOWN STAR'):
            short_simbad_list.remove(st_name)
    scistar_simbad_table = customSimbad.query_objects(short_simbad_list)

    # Convert to pandas df and make SIMBAD IDs the index
    short_df_simbad = scistar_simbad_table.to_pandas()
    short_df_simbad['SIMBAD_ID'] = short_simbad_list
    short_df_simbad.set_index('SIMBAD_ID',inplace=True)
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
        }
    for col,simbad_col in simbad_cols.items():
        df_simbad[col] = list(df_simbad[simbad_col])

    # Add the values we want to df_unique
    df_unique = pd.concat([df_unique,df_simbad.loc[:,simbad_cols.keys()]],axis=1)

    # Query mocadb.ca for extra info
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

    # Fill in values missing from SIMBAD with MOCA (or vice versa if prefer_SIMBAD==False)

    df_unique['COMMENTS'] = ''
    df_unique['DB_SOURCES'] = ''

    # Sort all the dfs by index and check that they match up
    df_unique.sort_index(inplace=True)   
    df_simbad.sort_index(inplace=True)   
    mdf.sort_index(inplace=True)   
    assert np.all(np.array(df_unique.index)==np.array(df_simbad.index)), "Index Error"
    assert np.all(np.array(df_unique.index)==np.array(mdf.index)), "Index Error"

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
    
    # Save dataframe
    df_out.to_csv(outpath)
    log.info(f'Database saved to {outpath}')
    
    return df_out


def get_sciref_files(sci_target, refdb, idir=None, 
                     spt_tolerance=None, 
                     filters=None, 
                     exclude_disks=False):
    """Construct a list of science files and reference files to input to a PSF subtraction routine.

    Args:
        sci_target (str): 
            name of the science target to be PSF subtracted. Can be the proposal target name, 
            JWST resolved target name, or 2MASS ID.
        refdb (pandas.DataFrame or str): 
            pandas dataframe or filepath to csv containing the reference database generated by 
            the build_refdb() function.
        idir (str):
            path to directory of input data, to be appended to file names.
        spt_tolerance (str or int, optional): 
            None (default): use all spectral types.
            'exact' : use only refs with the exact same spectral type.
            'class' : use only references with the same spectral class letter.
            'subclass' : use only references with the same spectral class letter and subclass number. (ignores luminosity class)
            int : use only refs within +- N spectral subclasses, e.g. M3-5 for an M4 science target if spt_tolerance = 1.
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
        # - filter out manual flags

    if isinstance(refdb,str):
        refdb = load_refdb(refdb)

    # Locate input target 2MASS ID 
    # (input name could be in index, TARGPROP, or SIMBAD_ID column)
    if sci_target in refdb['SIMBAD_ID'].to_list():
        targ_2mass = sci_target

    elif sci_target in refdb.index.to_list():
        targ_2mass = refdb.loc[sci_target,'SIMBAD_ID'].to_list()[0]
        
    elif sci_target in refdb['TARGPROP'].to_list():
        refdb_temp = refdb.reset_index()
        refdb_temp.set_index('TARGPROP',inplace=True)
        targ_2mass = refdb_temp.loc[sci_target,'SIMBAD_ID'].to_list()[0]
    
    else:
        log.error(f'Science target {sci_target} not found in reference database.')
        raise Exception(f'Science target {sci_target} not found in reference database.')
    
    refdb_temp = refdb.reset_index()
    refdb_temp.set_index('FILENAME',inplace=True)

    # Collect all the science files
    sci_fnames = refdb_temp.index[refdb_temp['SIMBAD_ID'] == targ_2mass].to_list()
    first_scifile = sci_fnames[0]

    # Start list of reference files
    ref_fnames = refdb_temp.index[refdb_temp['SIMBAD_ID'] != targ_2mass].to_list()

    # Collect the reference files
    if spt_tolerance != None:

        # Consider handling float subclasses (e.g. M4.5) better. Take floor for now.
        refdb_temp['SP_SUBCLASS'] = np.floor(refdb_temp['SP_SUBCLASS'])

        targ_sp_class = refdb_temp.loc[first_scifile,'SP_CLASS']
        targ_sp_subclass = refdb_temp.loc[first_scifile,'SP_SUBCLASS']
        targ_sp_lclass = refdb_temp.loc[first_scifile,'SP_LCLASS']
        
        if isinstance(spt_tolerance,str):
            if spt_tolerance.lower() == 'exact':

                spt_fnames = refdb_temp.index[(refdb_temp['SP_CLASS'] == targ_sp_class) & 
                                                (refdb_temp['SP_SUBCLASS'] == targ_sp_subclass) & 
                                                (refdb_temp['SP_LCLASS'] == targ_sp_lclass)
                                                ].to_list()
                
            elif spt_tolerance.lower() == 'class':

                spt_fnames = refdb_temp.index[(refdb_temp['SP_CLASS'] == targ_sp_class)
                                                ].to_list()

            elif spt_tolerance.lower() == 'subclass':

                spt_fnames = refdb_temp.index[(refdb_temp['SP_CLASS'] == targ_sp_class) & 
                                                (refdb_temp['SP_SUBCLASS'] == targ_sp_subclass)
                                                ].to_list()
            
            else:
                raise Exception(f'spt_tolerance {spt_tolerance} not configured.')
        
        else:

            assert isinstance(spt_tolerance,int)

            spt_fnames = []
            
            for i in range(-spt_tolerance,spt_tolerance+1):

                spt_tup = (targ_sp_class,targ_sp_subclass+i)
            
                # Carry over spectral classes and subclasses correctly
                spt_tup = adjust_spttype(spt_tup)
            
                spt_fnames.extend(refdb_temp.index[(refdb_temp['SP_CLASS'] == spt_tup[0]) & 
                                                   (refdb_temp['SP_SUBCLASS'] == spt_tup[1])
                                                  ].to_list())
            
        if len(spt_fnames) == 0:
            raise Warning(f'No observations found with specified spectral type filter.')
        
        sci_fnames = list(set(sci_fnames).intersection(spt_fnames))
        ref_fnames = list(set(ref_fnames).intersection(spt_fnames))

    # Remove observations with disks flagged
    if exclude_disks:
        disk_fnames = refdb_temp.index[refdb_temp['HAS_DISK'] == True].to_list()
        ref_fnames = list(set(ref_fnames) - set(disk_fnames))
    
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

    # Make sure no observations are in both sci_fnames and ref_fnames
    if len(set(sci_fnames).intersection(ref_fnames)) > 0:
        raise Exception("One or more filenames exists in both the science and reference file list. Something is wrong.")

    if not idir is None:
        sci_fpaths = [os.path.join(idir,sci_fname) for sci_fname in sci_fnames]
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

