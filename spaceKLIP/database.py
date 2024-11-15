from __future__ import division

import matplotlib

# =============================================================================
# IMPORTS
# =============================================================================

import os
import glob
import pdb
import sys

import astropy.io.fits as pyfits
import numpy as np

import copy
import json
import pysiaf
import webbpsf, webbpsf_ext

from astropy.table import Table
from jwst.pipeline import Detector1Pipeline, Image2Pipeline, Coron3Pipeline
from stdatamodels.jwst import datamodels

from .utils import nircam_apname, get_nrcmask_from_apname, get_filter_info, config_stpipe_log

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)



# =============================================================================
# MAIN
# =============================================================================

# Initialize SIAF instruments.
siaf_nrc = pysiaf.Siaf('NIRCam')
siaf_nis = pysiaf.Siaf('NIRISS')
siaf_mir = pysiaf.Siaf('MIRI')

from webbpsf_ext.logging_utils import setup_logging
setup_logging('WARN', verbose=False)

# Load NIRCam, NIRISS, and MIRI filters
wave_nircam, weff_nircam, do_svo = get_filter_info('NIRCAM', return_more=True)
wave_niriss, weff_niriss = get_filter_info('NIRISS', do_svo=do_svo)
wave_miri,   weff_miri   = get_filter_info('MIRI',   do_svo=False) 

class Database():
    """
    The central spaceKLIP database class.
    
    """
    
    def __init__(self,
                 output_dir):
        """
        Initialize the central spaceKLIP database class. It stores the
        observational metadata and keeps track of the reduction steps.
        
        The pre-PSF subtraction data is stored in the Database.obs dictionary
        and the post-PSF subtraction data is stored in the Database.red
        dictionary. They contain a table of metadata for each concatenation,
        which are identified automatically based on instrument, filter, pupil
        mask, and image mask. The tables can be edited by the user at any
        stage of the data reduction process and spaceKLIP will continue with
        the updated metadata (e.g., modified host star position).
        
        A non-verbose mode is available by setting Database.verbose = False.
        
        Parameters
        ----------
        output_dir : path
            Directory where the reduction products shall be saved.
        
        Returns
        -------
        None.
        
        """
        
        # Output directory for saving reduction products.
        self.output_dir = output_dir

        # Check if output directory exists
        if not os.path.isdir(self.output_dir):
            log.warning(f'Output directory does not exist. Creating {self.output_dir}.')
            os.makedirs(self.output_dir)
        
        # Initialize observations dictionary which contains the individual
        # concatenations.
        self.obs = {}
        
        # Initialize reductions dictionary which contains the individual
        # concatenations.
        self.red = {}
        
        # Initialize source dictionary which contains the individual
        # companions.
        self.src = {}
        
        # Verbose mode?
        self.verbose = True
        
        pass
    
    def read_jwst_s012_data(self,
                            datapaths,
                            psflibpaths=None,
                            bgpaths=None,
                            cr_from_siaf=False,
                            assoc_using_targname=True):
        """
        Read JWST stage 0 (\*uncal), 1 (\*rate or \*rateints), or 2 (\*cal or
        \*calints) data into the Database.obs dictionary. It contains a table of
        metadata for each concatenation, which are identified automatically
        based on instrument, filter, pupil mask, and image mask. The tables can
        be edited by the user at any stage of the data reduction process and
        spaceKLIP will continue with the updated metadata (e.g., modified host
        star position).
        
        Parameters
        ----------
        datapaths : list of paths
            List of paths of the input JWST data. SpaceKLIP will try to
            automatically identify science data, PSF references, target
            acquisition frames, and MIRI background observations. If spaceKLIP
            does not get things right, you may use the 'psflibpaths' and
            'bgpaths' keywords below.
        psflibpaths : list of paths, optional
            List of paths of the input PSF references. Make sure that they are
            NOT duplicated in the 'datapaths'. The default is None.
        bgpaths : list of paths, optional
            List of paths of the input MIRI background observations. Make sure
            that they ARE duplicated in the 'datapaths' or 'psflibpaths'. The
            default is None.
        assoc_using_targname : bool, optional
            If True associate TA and BG observations to their corresponding
            SCI and REF observations based on the target name from the APT
            file. Otherwise, only consider the instrument parameters to
            distinguish between SCI and REF TA/BG. The default is True.
        
        Returns
        -------
        None.
        
        """
        
        # Check input.
        if isinstance(datapaths, str):
            datapaths = [datapaths]
        if len(datapaths) == 0:
            raise UserWarning('Could not find any data paths')
        if isinstance(psflibpaths, str):
            psflibpaths = [psflibpaths]
        if isinstance(bgpaths, str):
            bgpaths = [bgpaths]
        if bgpaths is not None:
            for i in range(len(bgpaths)):
                if psflibpaths is not None:
                    if bgpaths[i] not in datapaths and bgpaths[i] not in psflibpaths:
                        raise UserWarning('Background path ' + bgpaths[i] + ' does not occur in data or PSF library paths')
                else:
                    if bgpaths[i] not in datapaths:
                        raise UserWarning('Background path ' + bgpaths[i] + ' does not occur in data paths')
        
        # Read FITS headers.
        DATAMODL = []
        TELESCOP = []
        TARGPROP = []
        TARG_RA = []  # deg
        TARG_DEC = []  # deg
        INSTRUME = []
        DETECTOR = []
        FILTER = []
        CWAVEL = []  # micron
        DWAVEL = []  # micron
        PUPIL = []
        CORONMSK = []
        EXP_TYPE = []
        EXPSTART = []  # MJD
        NINTS = []
        EFFINTTM = []  # s
        IS_PSF = []
        SELFREF = []
        SUBARRAY = []
        NUMDTHPT = []
        XOFFSET = []  # arcsec
        YOFFSET = []  # arcsec
        APERNAME = []
        PPS_APER = []
        PIXSCALE = []  # arcsec
        PIXAR_SR = []  # sr
        BUNIT = []
        CRPIX1 = []  # pix
        CRPIX2 = []  # pix
        VPARITY = []
        V3I_YANG = []  # deg
        RA_REF = []  # deg
        DEC_REF = []  # deg
        ROLL_REF = []  # deg
        BLURFWHM = []  # pix
        HASH = []
        if psflibpaths is not None:
            allpaths = np.array(datapaths + psflibpaths)
        else:
            allpaths = np.array(datapaths)
        Nallpaths = len(allpaths)

        for i in range(Nallpaths):
            hdul = pyfits.open(allpaths[i])
            head = hdul[0].header
            # Only read in the data if needed.
            data = hdul['SCI'].data if not head.get('NINTS') else None

            if 'uncal' in allpaths[i]:
                DATAMODL += ['STAGE0']
            elif 'rate' in allpaths[i] or 'rateints' in allpaths[i]:
                DATAMODL += ['STAGE1']
            elif 'cal' in allpaths[i] or 'calints' in allpaths[i]:
                DATAMODL += ['STAGE2']
            else:
                raise UserWarning('File name must contain one of the following: uncal, rate, rateints, cal, calints')
            TELESCOP += [head.get('TELESCOP', 'JWST')]
            TARGPROP += [head.get('TARGPROP', 'UNKNOWN')]
            TARG_RA += [head.get('TARG_RA', np.nan)]
            TARG_DEC += [head.get('TARG_DEC', np.nan)]
            INSTRUME += [head.get('INSTRUME', 'UNKNOWN')]
            DETECTOR += [head.get('DETECTOR', 'UNKNOWN')]
            FILTER += [head['FILTER']]
            PUPIL += [head.get('PUPIL', 'NONE')]
            if TELESCOP[-1] == 'JWST':
                if INSTRUME[-1] == 'NIRCAM':
                    if PUPIL[-1] in wave_nircam.keys():
                        CWAVEL += [wave_nircam[PUPIL[-1]]]
                        DWAVEL += [weff_nircam[PUPIL[-1]]]
                    else:
                        CWAVEL += [wave_nircam[FILTER[-1]]]
                        DWAVEL += [weff_nircam[FILTER[-1]]]
                elif INSTRUME[-1] == 'NIRISS':
                    CWAVEL += [wave_niriss[FILTER[-1]]]
                    DWAVEL += [weff_niriss[FILTER[-1]]]
                elif INSTRUME[-1] == 'MIRI':
                    CWAVEL += [wave_miri[FILTER[-1]]]
                    DWAVEL += [weff_miri[FILTER[-1]]]
                else:
                    raise UserWarning('Data originates from unknown JWST instrument')
            else:
                raise UserWarning('Data originates from unknown telescope')
            EXP_TYPE += [head.get('EXP_TYPE', 'UNKNOWN')]
            EXPSTART += [head.get('EXPSTART', np.nan)]
            NINTS += [head.get('NINTS', data.shape[0] if data is not None and data.ndim == 3 else 1)]
            EFFINTTM += [head.get('EFFINTTM', np.nan)]
            IS_PSF += [head.get('IS_PSF', 'NONE')]
            SELFREF += [head.get('SELFREF', 'NONE')]
            SUBARRAY += [head.get('SUBARRAY', 'UNKNOWN')]
            NUMDTHPT += [head.get('NUMDTHPT', 1)]
            XOFFSET += [head.get('XOFFSET', 0.)]
            YOFFSET += [head.get('YOFFSET', 0.)]
            apname = nircam_apname(head) if INSTRUME[-1] == 'NIRCAM' else head.get('APERNAME', 'UNKNOWN')
            APERNAME += [apname]
            PPS_APER += [head.get('PPS_APER', 'UNKNOWN')]
            coronmask = get_nrcmask_from_apname(PPS_APER[-1]) if INSTRUME[-1] == 'NIRCAM' else head.get('CORONMSK', 'NONE')
            CORONMSK += [coronmask]
            if TELESCOP[-1] == 'JWST':
                if INSTRUME[-1] == 'NIRCAM':
                    ap = siaf_nrc[apname]
                elif INSTRUME[-1] == 'NIRISS':
                    ap = siaf_nis[apname]
                elif INSTRUME[-1] == 'MIRI':
                    ap = siaf_mir[apname]
                else:
                    raise UserWarning('Data originates from unknown JWST instrument')
                # Save the average of the X and Y pixel scales.
                PIXSCALE += [(ap.XSciScale + ap.YSciScale) / 2.]
            else:
                raise UserWarning('Data originates from unknown telescope')
            BLURFWHM += [head.get('BLURFWHM', np.nan)]
            head = hdul['SCI'].header
            PIXAR_SR += [head.get('PIXAR_SR', np.nan)]
            BUNIT += [head.get('BUNIT', 'NONE')]
            if cr_from_siaf:
                CRPIX1 += [ap.XSciRef]
                CRPIX2 += [ap.YSciRef]
            else:
                CRPIX1 += [head.get('CRPIX1', np.nan)]
                CRPIX2 += [head.get('CRPIX2', np.nan)]
            VPARITY += [head.get('VPARITY', -1)]
            V3I_YANG += [head.get('V3I_YANG', 0.)]
            RA_REF += [head.get('RA_REF', np.nan)]
            DEC_REF += [head.get('DEC_REF', np.nan)]
            ROLL_REF += [head.get('ROLL_REF', 0.)]
            HASH += [TELESCOP[-1] + '_' + INSTRUME[-1] + '_' + DETECTOR[-1] + '_' + FILTER[-1] + '_' + PUPIL[-1] + '_' + CORONMSK[-1] + '_' + SUBARRAY[-1]]
            hdul.close()
        DATAMODL = np.array(DATAMODL)
        TELESCOP = np.array(TELESCOP)
        TARGPROP = np.array(TARGPROP)
        TARG_RA = np.array(TARG_RA)
        TARG_DEC = np.array(TARG_DEC)
        INSTRUME = np.array(INSTRUME)
        DETECTOR = np.array(DETECTOR)
        FILTER = np.array(FILTER)
        CWAVEL = np.array(CWAVEL)
        DWAVEL = np.array(DWAVEL)
        PUPIL = np.array(PUPIL)
        CORONMSK = np.array(CORONMSK)
        EXP_TYPE = np.array(EXP_TYPE)
        EXPSTART = np.array(EXPSTART)
        NINTS = np.array(NINTS)
        EFFINTTM = np.array(EFFINTTM)
        IS_PSF = np.array(IS_PSF, dtype='<U5')
        SELFREF = np.array(SELFREF, dtype='<U5')
        SUBARRAY = np.array(SUBARRAY)
        NUMDTHPT = np.array(NUMDTHPT)
        XOFFSET = np.array(XOFFSET)
        YOFFSET = np.array(YOFFSET)
        APERNAME = np.array(APERNAME)
        PPS_APER = np.array(PPS_APER)
        PIXSCALE = np.array(PIXSCALE)
        PIXAR_SR = np.array(PIXAR_SR)
        BUNIT = np.array(BUNIT)
        CRPIX1 = np.array(CRPIX1)
        CRPIX2 = np.array(CRPIX2)
        VPARITY = np.array(VPARITY)
        V3I_YANG = np.array(V3I_YANG)
        RA_REF = np.array(RA_REF)
        DEC_REF = np.array(DEC_REF)
        ROLL_REF = np.array(ROLL_REF)
        BLURFWHM = np.array(BLURFWHM)
        HASH = np.array(HASH)
        
        # Find unique concatenations.
        HASH_unique = np.unique(HASH)
        NHASH_unique = len(HASH_unique)
        
        # Associate TA files with science or reference files.
        ww = []
        for i in range(NHASH_unique):
            if 'MASKRND_NONE' in HASH_unique[i] or 'MASKBAR_NONE' in HASH_unique[i] or 'NONE_MASK1065' in HASH_unique[i] or 'NONE_MASK1140' in HASH_unique[i] or 'NONE_MASK1550' in HASH_unique[i] or 'NONE_MASKLYOT' in HASH_unique[i]:
                ww += [-1]
                HASH_i_split = HASH_unique[i].split('_')
                for j in range(NHASH_unique):
                    HASH_j_split = HASH_unique[j].split('_')
                    if HASH_j_split[0] == HASH_i_split[0] and HASH_j_split[1] == HASH_i_split[1] and HASH_j_split[2] == HASH_i_split[2] and HASH_j_split[4] == HASH_i_split[4] and HASH_j_split[5] != 'NONE' and HASH_j_split[6][-4:] in HASH_i_split[6]:
                        ww[-1] = i
                        break
                if ww[-1] != -1:
                    HASH[HASH == HASH_unique[i]] = HASH_unique[j]
                else:
                    raise UserWarning('Could not associate TA files with science or reference files')
        HASH_unique = np.delete(HASH_unique, ww)
        NHASH_unique = len(HASH_unique)
        
        # Get PSF mask directory.
        maskbase = os.path.split(os.path.abspath(__file__))[0]
        maskbase = os.path.join(maskbase, 'resources/transmissions/')
        
        # Loop through concatenations.
        for i in range(NHASH_unique):
            ww = HASH == HASH_unique[i]
            
            # Find science and reference files.
            if psflibpaths is not None:
                ww_sci = []
                ww_ref = []
                for j in range(len(allpaths[ww])):
                    if allpaths[ww][j] in psflibpaths:
                        ww_ref.append(j)
                    else:
                        ww_sci.append(j)
                ww_sci = np.array(ww_sci, dtype='int')
                ww_ref = np.array(ww_ref, dtype='int')
            else:
                is_psf = IS_PSF[ww]
                exp_type = EXP_TYPE[ww]
                for j in range(len(exp_type)):
                    if 'TA' in exp_type[j]:
                        is_psf[j] = 'False'
                if 'NONE' not in is_psf:
                    ww_sci = np.where(is_psf == 'False')[0]
                    ww_ref = np.where(is_psf == 'True')[0]
                else:
                    log.warning('  --> Could not find IS_PSF header keyword')
                    numdthpt = NUMDTHPT[ww]
                    numdthpt_unique = np.unique(numdthpt)
                    if len(numdthpt_unique) == 2 and numdthpt_unique[0] == 1:
                        ww_sci = np.where(numdthpt == numdthpt_unique[0])[0]
                        ww_ref = np.where(numdthpt == numdthpt_unique[1])[0]
                    else:
                        log.warning('  --> Could not identify science and reference files based on dither pattern')
                        raise UserWarning('Please use psflibpaths to specify reference files')
            
            # Make Astropy tables for concatenations.
            tab = Table(names=('TYPE',
                               'EXP_TYPE',
                               'DATAMODL',
                               'TELESCOP',
                               'TARGPROP',
                               'TARG_RA',
                               'TARG_DEC',
                               'INSTRUME',
                               'DETECTOR',
                               'FILTER',
                               'CWAVEL',
                               'DWAVEL',
                               'PUPIL',
                               'CORONMSK',
                               'EXPSTART',
                               'NINTS',
                               'EFFINTTM',
                               'SUBARRAY',
                               'NUMDTHPT',
                               'XOFFSET',
                               'YOFFSET',
                               'APERNAME',
                               'PPS_APER',
                               'PIXSCALE',
                               'PIXAR_SR',
                               'BUNIT',
                               'CRPIX1',
                               'CRPIX2',
                               'RA_REF',
                               'DEC_REF',
                               'ROLL_REF',
                               'BLURFWHM',
                               'FITSFILE',
                               'MASKFILE'),
                        dtype=('object',
                               'object',
                               'object',
                               'object',
                               'object',
                               'float',
                               'float',
                               'object',
                               'object',
                               'object',
                               'float',
                               'float',
                               'object',
                               'object',
                               'float',
                               'int',
                               'float',
                               'object',
                               'int',
                               'float',
                               'float',
                               'object',
                               'object', 
                               'float',
                               'float',
                               'object',
                               'float',
                               'float',
                               'float',
                               'float',
                               'float',
                               'float',
                               'object',
                               'object'))
            for j in np.append(ww_sci, ww_ref):
                if j in ww_sci:
                    sci = True
                else:
                    sci = False
                if 'TA' in EXP_TYPE[ww][j]:
                    if sci:
                        tt = 'SCI_TA'
                    else:
                        tt = 'REF_TA'
                elif 'BG' in TARGPROP[ww][j].upper() or 'BACK' in TARGPROP[ww][j].upper() or 'BACKGROUND' in TARGPROP[ww][j].upper():
                    if sci:
                        tt = 'SCI_BG'
                    else:
                        tt = 'REF_BG'
                else:
                    if sci:
                        tt = 'SCI'
                    else:
                        tt = 'REF'
                if bgpaths is not None:
                    if allpaths[ww][j] in bgpaths:
                        if sci:
                            tt = 'SCI_BG'
                        else:
                            tt = 'REF_BG'
                maskfile = allpaths[ww][j].replace('.fits', '_psfmask.fits')
                if not os.path.exists(maskfile):    
                    if EXP_TYPE[ww][j] == 'NRC_CORON':
                    
                        config_stpipe_log(suppress=True)  # Suppress logging.

                        pipeline = Detector1Pipeline()
                        input = datamodels.open(allpaths[ww][j])
                        maskfile = pipeline.get_reference_file(input, 'psfmask')
                        config_stpipe_log(suppress=True)  # Revert to default logging.

                    elif EXP_TYPE[ww][j] == 'MIR_4QPM' or EXP_TYPE[ww][j] == 'MIR_LYOT':
                        if APERNAME[ww][j] == 'MIRIM_MASK1065':
                            maskpath = 'JWST_MIRI_F1065C_transmission_webbpsf-ext_v2.fits'
                        elif APERNAME[ww][j] == 'MIRIM_MASK1140':
                            maskpath = 'JWST_MIRI_F1140C_transmission_webbpsf-ext_v2.fits'
                        elif APERNAME[ww][j] == 'MIRIM_MASK1550':
                            maskpath = 'JWST_MIRI_F1550C_transmission_webbpsf-ext_v2.fits'
                        elif APERNAME[ww][j] == 'MIRIM_MASKLYOT':
                            maskpath = 'jwst_miri_psfmask_0009.fits'  # FIXME!
                        maskfile = os.path.join(maskbase, maskpath)
                    else:
                        maskfile = 'NONE'
                tab.add_row((tt,
                             EXP_TYPE[ww][j],
                             DATAMODL[ww][j],
                             TELESCOP[ww][j],
                             TARGPROP[ww][j],
                             TARG_RA[ww][j],
                             TARG_DEC[ww][j],
                             INSTRUME[ww][j],
                             DETECTOR[ww][j],
                             FILTER[ww][j],
                             CWAVEL[ww][j],
                             DWAVEL[ww][j],
                             PUPIL[ww][j],
                             CORONMSK[ww][j],
                             EXPSTART[ww][j],
                             NINTS[ww][j],
                             EFFINTTM[ww][j],
                             SUBARRAY[ww][j],
                             NUMDTHPT[ww][j],
                             XOFFSET[ww][j],
                             YOFFSET[ww][j],
                             APERNAME[ww][j],
                             PPS_APER[ww][j],
                             PIXSCALE[ww][j],
                             PIXAR_SR[ww][j],
                             BUNIT[ww][j],
                             CRPIX1[ww][j],
                             CRPIX2[ww][j],
                             RA_REF[ww][j],
                             DEC_REF[ww][j],
                             ROLL_REF[ww][j] - V3I_YANG[ww][j] * VPARITY[ww][j],
                             BLURFWHM[ww][j],
                             allpaths[ww][j],
                             maskfile))
            self.obs[HASH_unique[i]] = tab.copy()
            del tab
            
            # Associate background files with science or reference files.
            for j in range(len(self.obs[HASH_unique[i]])):
                if self.obs[HASH_unique[i]]['TYPE'][j] == 'SCI_BG':
                    if (self.obs[HASH_unique[i]]['EFFINTTM'][j] not in self.obs[HASH_unique[i]]['EFFINTTM'][self.obs[HASH_unique[i]]['TYPE'] == 'SCI']) or (self.obs[HASH_unique[i]]['NINTS'][j] not in self.obs[HASH_unique[i]]['NINTS'][self.obs[HASH_unique[i]]['TYPE'] == 'SCI']):
                        if (self.obs[HASH_unique[i]]['EFFINTTM'][j] in self.obs[HASH_unique[i]]['EFFINTTM'][self.obs[HASH_unique[i]]['TYPE'] == 'REF']) and (self.obs[HASH_unique[i]]['NINTS'][j] in self.obs[HASH_unique[i]]['NINTS'][self.obs[HASH_unique[i]]['TYPE'] == 'REF']):
                            self.obs[HASH_unique[i]]['TYPE'][j] = 'REF_BG'
                        else:
                            raise UserWarning('Background exposure ' + self.obs[HASH_unique[i]]['FITSFILE'][j] + ' could not be matched with PSF')
                elif self.obs[HASH_unique[i]]['TYPE'][j] == 'REF_BG':
                    if (self.obs[HASH_unique[i]]['EFFINTTM'][j] not in self.obs[HASH_unique[i]]['EFFINTTM'][self.obs[HASH_unique[i]]['TYPE'] == 'REF']) or (self.obs[HASH_unique[i]]['NINTS'][j] not in self.obs[HASH_unique[i]]['NINTS'][self.obs[HASH_unique[i]]['TYPE'] == 'REF']):
                        if (self.obs[HASH_unique[i]]['EFFINTTM'][j] in self.obs[HASH_unique[i]]['EFFINTTM'][self.obs[HASH_unique[i]]['TYPE'] == 'SCI']) and (self.obs[HASH_unique[i]]['NINTS'][j] in self.obs[HASH_unique[i]]['NINTS'][self.obs[HASH_unique[i]]['TYPE'] == 'SCI']):
                            self.obs[HASH_unique[i]]['TYPE'][j] = 'SCI_BG'
                        else:
                            raise UserWarning('Background exposure ' + self.obs[HASH_unique[i]]['FITSFILE'][j] + ' could not be matched with PSF')
            
            # Reassociate TA and background files with science or reference
            # files based on target name.
            if assoc_using_targname:
                for j in range(len(self.obs[HASH_unique[i]])):
                    if self.obs[HASH_unique[i]]['TYPE'][j] in ['SCI_TA', 'SCI_BG']:
                        targprop = self.obs[HASH_unique[i]]['TARGPROP'][self.obs[HASH_unique[i]]['TYPE'] == 'SCI']
                        ww = np.array([s in self.obs[HASH_unique[i]]['TARGPROP'][j] for s in targprop])
                        if np.sum(ww) == 0:
                            targprop = self.obs[HASH_unique[i]]['TARGPROP'][self.obs[HASH_unique[i]]['TYPE'] == 'REF']
                            ww = np.array([s in self.obs[HASH_unique[i]]['TARGPROP'][j] for s in targprop])
                            if np.sum(ww) != 0:
                                self.obs[HASH_unique[i]]['TYPE'][j] = self.obs[HASH_unique[i]]['TYPE'][j].replace('SCI', 'REF')
                    if self.obs[HASH_unique[i]]['TYPE'][j] in ['REF_TA', 'REF_BG']:
                        targprop = self.obs[HASH_unique[i]]['TARGPROP'][self.obs[HASH_unique[i]]['TYPE'] == 'REF']
                        ww = np.array([s in self.obs[HASH_unique[i]]['TARGPROP'][j] for s in targprop])
                        if np.sum(ww) == 0:
                            targprop = self.obs[HASH_unique[i]]['TARGPROP'][self.obs[HASH_unique[i]]['TYPE'] == 'SCI']
                            ww = np.array([s in self.obs[HASH_unique[i]]['TARGPROP'][j] for s in targprop])
                            if np.sum(ww) != 0:
                                self.obs[HASH_unique[i]]['TYPE'][j] = self.obs[HASH_unique[i]]['TYPE'][j].replace('REF', 'SCI')
        
        # Print Astropy tables for concatenations.
        if self.verbose:
            self.print_obs()
        
        pass
    
    def read_jwst_s3_data(self,
                          datapaths,
                          cr_from_siaf=False):
        """
        Read JWST stage 3 data (this can be \*i2d data from the official JWST
        pipeline, or data products from the pyKLIP and classical PSF
        subtraction pipelines implemented in spaceKLIP) into the Database.red
        dictionary. It contains a table of metadata for each concatenation,
        which are identified automatically based on instrument, filter, pupil
        mask, and image mask. The tables can be edited by the user at any stage
        of the data reduction process and spaceKLIP will continue with the
        updated metadata (e.g., modified host star position).
        
        Parameters
        ----------
        datapaths : list of paths
            List of paths of the input JWST data.
        
        Returns
        -------
        None.
        
        """
        
        # Check input.
        if isinstance(datapaths, str):
            datapaths = [datapaths]
        if len(datapaths) == 0:
            raise UserWarning('Could not find any data paths')
        
        # Read FITS headers.
        TYPE = []
        DATAMODL = []
        TELESCOP = []
        TARGPROP = []
        TARG_RA = []  # deg
        TARG_DEC = []  # deg
        INSTRUME = []
        DETECTOR = []
        FILTER = []
        CWAVEL = []  # micron
        DWAVEL = []  # micron
        PUPIL = []
        CORONMSK = []
        EXP_TYPE = []
        EXPSTART = []  # MJD
        NINTS = []
        EFFINTTM = []  # s
        SUBARRAY = []
        APERNAME = []
        PPS_APER = []
        PIXSCALE = []  # arcsec
        PIXAR_SR = []  # sr
        MODE = []
        ANNULI = []
        SUBSECTS = []
        KLMODES = []
        BUNIT = []
        CRPIX1 = []  # pix
        CRPIX2 = []  # pix
        BLURFWHM = []  # pix
        HASH = []
        Ndatapaths = len(datapaths)
        for i in range(Ndatapaths):
            hdul = pyfits.open(datapaths[i])
            head = hdul[0].header
            if datapaths[i].endswith('i2d.fits'):
                TYPE += ['CORON3']
            elif datapaths[i].endswith('KLmodes-all.fits'):
                TYPE += ['PYKLIP']
            else:
                raise UserWarning('File must have one of the following endings: i2d.fits, KLmodes-all.fits')
            DATAMODL += ['STAGE3']
            TELESCOP += [head.get('TELESCOP', 'JWST')]
            TARGPROP += [head.get('TARGPROP', 'UNKNOWN')]
            TARG_RA += [head.get('TARG_RA', np.nan)]
            TARG_DEC += [head.get('TARG_DEC', np.nan)]
            INSTRUME += [head.get('INSTRUME', 'UNKNOWN')]
            DETECTOR += [head.get('DETECTOR', 'UNKNOWN')]
            FILTER += [head['FILTER']]
            PUPIL += [head.get('PUPIL', 'NONE')]
            if TELESCOP[-1] == 'JWST':
                if INSTRUME[-1] == 'NIRCAM':
                    if PUPIL[-1] in wave_nircam.keys():
                        CWAVEL += [wave_nircam[PUPIL[-1]]]
                        DWAVEL += [weff_nircam[PUPIL[-1]]]
                    else:
                        CWAVEL += [wave_nircam[FILTER[-1]]]
                        DWAVEL += [weff_nircam[FILTER[-1]]]
                elif INSTRUME[-1] == 'NIRISS':
                    CWAVEL += [wave_niriss[FILTER[-1]]]
                    DWAVEL += [weff_niriss[FILTER[-1]]]
                elif INSTRUME[-1] == 'MIRI':
                    CWAVEL += [wave_miri[FILTER[-1]]]
                    DWAVEL += [weff_miri[FILTER[-1]]]
                else:
                    raise UserWarning('Data originates from unknown JWST instrument')
            else:
                raise UserWarning('Data originates from unknown telescope')
            EXP_TYPE += [head.get('EXP_TYPE', 'UNKNOWN')]
            EXPSTART += [head.get('EXPSTART', np.nan)]
            NINTS += [head.get('NINTS', 1)]
            EFFINTTM += [head.get('EFFINTTM', np.nan)]
            SUBARRAY += [head.get('SUBARRAY', 'UNKNOWN')]
            apname = nircam_apname(head) if INSTRUME[-1] == 'NIRCAM' else head.get('APERNAME', 'UNKNOWN')
            APERNAME += [apname]
            PPS_APER += [head.get('PPS_APER', 'UNKNOWN')]
            coronmask = get_nrcmask_from_apname(PPS_APER[-1]) if INSTRUME[-1] == 'NIRCAM' else head.get('CORONMSK', 'NONE')
            CORONMSK += [coronmask]
            if TELESCOP[-1] == 'JWST':
                if INSTRUME[-1] == 'NIRCAM':
                    ap = siaf_nrc[apname]
                elif INSTRUME[-1] == 'NIRISS':
                    ap = siaf_nis[apname]
                elif INSTRUME[-1] == 'MIRI':
                    ap = siaf_mir[apname]
                else:
                    raise UserWarning('Data originates from unknown JWST instrument')
                # Save the average of the X and Y pixel scales.
                PIXSCALE += [(ap.XSciScale + ap.YSciScale) / 2.]
            else:
                raise UserWarning('Data originates from unknown telescope')
            if TYPE[-1] == 'CORON3':
                MODE += ['RDI']
                ANNULI += [1]
                SUBSECTS += [1]
                try:
                    KLMODES += [str(head['KLMODE0'])]
                except KeyError:
                    log.warning('  --> Could not find KL mode in header, assuming default value of 50')
                    KLMODES += ['50']
            elif TYPE[-1] == 'PYKLIP':
                MODE += [head['MODE']]
                ANNULI += [head['ANNULI']]
                SUBSECTS += [head['SUBSECTS']]
                klmodes = str(head['KLMODE0'])
                j = 1
                while True:
                    try:
                        klmodes += ',' + str(head['KLMODE{0}'.format(j)])
                    except KeyError:
                        break
                    j += 1
                KLMODES += [klmodes]
            else:
                raise UserWarning('File must have one of the following types: CORON3, PYKLIP')
            BLURFWHM += [head.get('BLURFWHM', np.nan)]
            if TYPE[-1] == 'CORON3':
                head = hdul['SCI'].header
            PIXAR_SR += [head.get('PIXAR_SR', np.nan)]
            BUNIT += [head.get('BUNIT', 'NONE')]
            if cr_from_siaf:
                CRPIX1 += [ap.XSciRef]
                CRPIX2 += [ap.YSciRef]
            else:
                CRPIX1 += [head.get('CRPIX1', np.nan)]
                CRPIX2 += [head.get('CRPIX2', np.nan)]
            HASH += [TELESCOP[-1] + '_' + INSTRUME[-1] + '_' + DETECTOR[-1] + '_' + FILTER[-1] + '_' + PUPIL[-1] + '_' + CORONMSK[-1] + '_' + SUBARRAY[-1]]
            hdul.close()
        TYPE = np.array(TYPE)
        DATAMODL = np.array(DATAMODL)
        TELESCOP = np.array(TELESCOP)
        TARGPROP = np.array(TARGPROP)
        TARG_RA = np.array(TARG_RA)
        TARG_DEC = np.array(TARG_DEC)
        INSTRUME = np.array(INSTRUME)
        DETECTOR = np.array(DETECTOR)
        FILTER = np.array(FILTER)
        CWAVEL = np.array(CWAVEL)
        DWAVEL = np.array(DWAVEL)
        PUPIL = np.array(PUPIL)
        CORONMSK = np.array(CORONMSK)
        EXP_TYPE = np.array(EXP_TYPE)
        EXPSTART = np.array(EXPSTART)
        NINTS = np.array(NINTS)
        EFFINTTM = np.array(EFFINTTM)
        SUBARRAY = np.array(SUBARRAY)
        APERNAME = np.array(APERNAME)
        PPS_APER = np.array(PPS_APER)
        PIXSCALE = np.array(PIXSCALE)
        PIXAR_SR = np.array(PIXAR_SR)
        MODE = np.array(MODE)
        ANNULI = np.array(ANNULI)
        SUBSECTS = np.array(SUBSECTS)
        KLMODES = np.array(KLMODES)
        BUNIT = np.array(BUNIT)
        CRPIX1 = np.array(CRPIX1)
        CRPIX2 = np.array(CRPIX2)
        BLURFWHM = np.array(BLURFWHM)
        HASH = np.array(HASH)
        
        # Find unique concatenations.
        HASH_unique = np.unique(HASH)
        NHASH_unique = len(HASH_unique)
        
        # Loop through concatenations.
        for i in range(NHASH_unique):
            ww = np.where(HASH == HASH_unique[i])[0]
            
            # Make Astropy tables for concatenations.
            if HASH_unique[i] not in self.red.keys():
                tab = Table(names=('TYPE',
                                   'EXP_TYPE',
                                   'DATAMODL',
                                   'TELESCOP',
                                   'TARGPROP',
                                   'TARG_RA',
                                   'TARG_DEC',
                                   'INSTRUME',
                                   'DETECTOR',
                                   'FILTER',
                                   'CWAVEL',
                                   'DWAVEL',
                                   'PUPIL',
                                   'CORONMSK',
                                   'EXPSTART',
                                   'NINTS',
                                   'EFFINTTM',
                                   'SUBARRAY',
                                   'APERNAME',
                                   'PPS_APER',
                                   'PIXSCALE',
                                   'PIXAR_SR',
                                   'MODE',
                                   'ANNULI',
                                   'SUBSECTS',
                                   'KLMODES',
                                   'BUNIT',
                                   'BLURFWHM',
                                   'FITSFILE',
                                   'MASKFILE'),
                            dtype=('object',
                                   'object',
                                   'object',
                                   'object',
                                   'object',
                                   'float',
                                   'float',
                                   'object',
                                   'object',
                                   'object',
                                   'float',
                                   'float',
                                   'object',
                                   'object',
                                   'float',
                                   'int',
                                   'float',
                                   'object',
                                   'object',
                                   'object',
                                   'float',
                                   'float',
                                   'object',
                                   'int',
                                   'int',
                                   'object',
                                   'object',
                                   'float',
                                   'object',
                                   'object'))
            else:
                tab = self.red[HASH_unique[i]].copy()
            for j in range(len(ww)):
                maskfile = os.path.join(os.path.split(datapaths[ww[j]])[0], HASH_unique[i] + '_psfmask.fits')
                if not os.path.exists(maskfile):
                    maskfile = 'NONE'
                tab.add_row((TYPE[ww[j]],
                             EXP_TYPE[ww[j]],
                             DATAMODL[ww[j]],
                             TELESCOP[ww[j]],
                             TARGPROP[ww[j]],
                             TARG_RA[ww[j]],
                             TARG_DEC[ww[j]],
                             INSTRUME[ww[j]],
                             DETECTOR[ww[j]],
                             FILTER[ww[j]],
                             CWAVEL[ww[j]],
                             DWAVEL[ww[j]],
                             PUPIL[ww[j]],
                             CORONMSK[ww[j]],
                             EXPSTART[ww[j]],
                             NINTS[ww[j]],
                             EFFINTTM[ww[j]],
                             SUBARRAY[ww[j]],
                             APERNAME[ww[j]],
                             PPS_APER[ww[j]],
                             PIXSCALE[ww[j]],
                             PIXAR_SR[ww[j]],
                             MODE[ww[j]],
                             ANNULI[ww[j]],
                             SUBSECTS[ww[j]],
                             KLMODES[ww[j]],
                             BUNIT[ww[j]],
                             BLURFWHM[ww][j],
                             datapaths[ww[j]],
                             maskfile))
            self.red[HASH_unique[i]] = tab.copy()
            del tab
            
            # Read corresponding observations database.
            if HASH_unique[i] not in self.obs.keys():
                try:
                    file = os.path.join(os.path.split(datapaths[ww[j]])[0], HASH_unique[i] + '.dat')
                    self.obs[HASH_unique[i]] = Table.read(file, format='ascii')
                    self.obs[HASH_unique[i]]['FITSFILE'] = self.obs[HASH_unique[i]]['FITSFILE'].astype(object)
                    self.obs[HASH_unique[i]]['MASKFILE'] = self.obs[HASH_unique[i]]['MASKFILE'].astype(object)
                except FileNotFoundError:
                    raise UserWarning('Observations database for concatenation ' + HASH_unique[i] + ' not found')
        
        # Print Astropy tables for concatenations.
        if self.verbose:
            self.print_red()
        
        pass
    
    def read_jwst_s4_data(self,
                          datapaths):
        """
        Read JWST stage 4 data (spaceKLIP PSF fitting products) into the
        Database.src dictionary. It contains a list of tables of metadata for
        each concatenation, which are identified automatically based on
        instrument, filter, pupil mask, and image mask.
        
        Parameters
        ----------
        datapaths : list of paths
            List of paths of the input JWST data.
        
        Returns
        -------
        None.
        
        """
        
        # Check input.
        if isinstance(datapaths, str):
            datapaths = [datapaths]
        if len(datapaths) == 0:
            raise UserWarning('Could not find any data paths')
        
        # Find unique concatenations.
        HASH = []
        Ndatapaths = len(datapaths)
        for i in range(Ndatapaths):
            temp = os.path.split(datapaths[i])[1]
            ww = temp.find('-fitpsf_')
            HASH += [temp[:ww]]
        HASH_unique = np.unique(np.array(HASH))
        NHASH_unique = len(HASH_unique)
        
        # Loop through concatenations.
        for i in range(NHASH_unique):
            
            # Make Astropy tables for concatenations.
            if HASH_unique[i] not in self.src.keys():
                self.src[HASH_unique[i]] = []
            tab = Table(names=('ID',
                               'RA',
                               'RA_ERR',
                               'DEC',
                               'DEC_ERR',
                               'CON',
                               'CON_ERR',
                               'DELMAG',
                               'DELMAG_ERR',
                               'APPMAG',
                               'APPMAG_ERR',
                               'MSTAR',
                               'MSTAR_ERR',
                               'SNR',
                               'LN(Z/Z0)',
                               'FITSFILE'),
                        dtype=('int',
                               'float',
                               'float',
                               'float',
                               'float',
                               'float',
                               'float',
                               'float',
                               'float',
                               'float',
                               'float',
                               'float',
                               'float',
                               'float',
                               'float',
                               'object'))
            
            # Read FITS headers.
            for j in range(Ndatapaths):
                if HASH[j] == HASH_unique[i]:
                    hdul = pyfits.open(datapaths[j])
                    head = hdul[0].header
                    if head['LN(Z/Z0)'] == 'NONE':
                        evidence_ratio = np.nan
                    else:
                        evidence_ratio = head['LN(Z/Z0)']
                    tab.add_row((head['ID'],
                                 head['RA'],
                                 head['RA_ERR'],
                                 head['DEC'],
                                 head['DEC_ERR'],
                                 head['CON'],
                                 head['CON_ERR'],
                                 head['DELMAG'],
                                 head['DELMAG_ERR'],
                                 head['APPMAG'],
                                 head['APPMAG_ERR'],
                                 head['MSTAR'],
                                 head['MSTAR_ERR'],
                                 head['SNR'],
                                 evidence_ratio,
                                 head['FITSFILE']))
            self.src[HASH_unique[i]] += [tab.copy()]
            del tab
        
        # Print Astropy tables for concatenations.
        if self.verbose:
            self.print_src()
        
        pass
    
    def print_obs(self,
                  include_fitsfiles=False):
        """
        Print an abbreviated version of the observations database.
        
        Parameters
        ----------
        include_fitsfiles : bool, optional
            Include the FITS file and PSF mask paths in the output. The default
            is False.
        
        Returns
        -------
        None.
        
        """
        
        # Print Astropy tables for concatenations.
        log.info('--> Identified %.0f concatenation(s)' % len(self.obs))
        for i, key in enumerate(self.obs.keys()):
            log.info('  --> Concatenation %.0f: ' % (i + 1) + key)
            print_tab = copy.deepcopy(self.obs[key])
            if include_fitsfiles:
                print_tab.remove_columns(['TARG_RA', 'TARG_DEC', 'EXPSTART', 'APERNAME', 'PPS_APER', 
                                          'CRPIX1', 'CRPIX2', 'RA_REF', 'DEC_REF'])
            else:
                print_tab.remove_columns(['TARG_RA', 'TARG_DEC', 'EXPSTART', 'APERNAME', 'PPS_APER', 
                                          'CRPIX1', 'CRPIX2', 'RA_REF', 'DEC_REF', 'FITSFILE', 'MASKFILE'])
            print_tab['XOFFSET'] *= 1e3
            print_tab['XOFFSET'] = np.round(print_tab['XOFFSET'])
            print_tab['XOFFSET'][print_tab['XOFFSET'] == 0.] = 0.
            print_tab['YOFFSET'] *= 1e3
            print_tab['YOFFSET'] = np.round(print_tab['YOFFSET'])
            print_tab['YOFFSET'][print_tab['YOFFSET'] == 0.] = 0.
            print_tab.pprint()
        
        pass
    
    def print_red(self,
                  include_fitsfiles=False):
        """
        Print an abbreviated version of the reductions database.
        
        Parameters
        ----------
        include_fitsfiles : bool, optional
            Include the FITS file and PSF mask paths in the output. The default
            is False.
        
        Returns
        -------
        None.
        
        """
        
        # Print Astropy tables for concatenations.
        log.info('--> Identified %.0f concatenation(s)' % len(self.red))
        for i, key in enumerate(self.red.keys()):
            log.info('  --> Concatenation %.0f: ' % (i + 1) + key)
            print_tab = copy.deepcopy(self.red[key])
            if include_fitsfiles:
                print_tab.remove_columns(['TARG_RA', 'TARG_DEC', 'EXPSTART', 'APERNAME', 'PPS_APER'])
            else:
                print_tab.remove_columns(['TARG_RA', 'TARG_DEC', 'EXPSTART', 'APERNAME', 'PPS_APER', 'FITSFILE', 'MASKFILE'])
            print_tab.pprint()
        
        pass
    
    def print_src(self,
                  include_fitsfiles=False):
        """
        Print an abbreviated version of the source database.
        
        Parameters
        ----------
        include_fitsfiles : bool, optional
            Include the FITS file paths in the output. The default is False.
        
        Returns
        -------
        None.
        
        """
        
        # Print Astropy tables for concatenations.
        log.info('--> Identified %.0f concatenation(s)' % len(self.src))
        for i, key in enumerate(self.src.keys()):
            log.info('  --> Concatenation %.0f: ' % (i + 1) + key)
            log.info('  --> Identified %.0f system(s)' % len(self.src[key]))
            for j in range(len(self.src[key])):
                log.info('    --> System %.0f:' % (j + 1))
                print_tab = copy.deepcopy(self.src[key][j])
                if include_fitsfiles:
                    pass
                else:
                    print_tab.remove_columns(['FITSFILE'])
                print_tab.pprint()
        
        pass
    
    def update_obs(self,
                   key,
                   index,
                   fitsfile,
                   maskfile=None,
                   nints=None,
                   effinttm=None,
                   xoffset=None,
                   yoffset=None,
                   crpix1=None,
                   crpix2=None,
                   blurfwhm=None,
                   update_pxar=False):
        """
        Update the content of the observations database.
        
        Parameters
        ----------
        key : str
            Database key of the observation to be updated.
        index : int
            Database index of the observation to be updated.
        fitsfile : path
            New FITS file path for the observation to be updated.
        maskfile : path, optional
            New PSF mask path for the observation to be updated. The default is
            None.
        nints : int, optional
            New number of integrations for the observation to be updated. The
            default is None.
        effinttm : float, optional
            New effective integration time (s) for the observation to be
            updated. The default is None.
        xoffset : float, optional
            New PSF x-offset (mas) for the observation to be updated. The
            default is None.
        yoffset : float, optional
            New PSF y-offset (mas) for the observation to be updated. The
            default is None.
        crpix1 : float, optional
            New PSF x-position (pix, 1-indexed) for the observation to be
            updated. The default is None.
        crpix2 : float, optional
            New PSF y-position (pix, 1-indexed) for the observation to be
            updated. The default is None.
        blurfwhm : float, optional
            New FWHM for the Gaussian filter blurring (pix) for the observation
            to be updated. The default is None.
        update_pxar : bool, optional
            Update the pixel area column of the database based on the FITS file
            header information? The default is False.
        
        Returns
        -------
        None.
        
        """
        
        # Update spaceKLIP database.
        if 'uncal' in fitsfile:
            DATAMODL = 'STAGE0'
        elif 'rate' in fitsfile or 'rateints' in fitsfile:
            DATAMODL = 'STAGE1'
        elif 'cal' in fitsfile or 'calints' in fitsfile:
            DATAMODL = 'STAGE2'
        else:
            raise UserWarning('File name must contain one of the following: uncal, rate, rateints, cal, calints')
        hdul = pyfits.open(fitsfile)
        self.obs[key]['DATAMODL'][index] = DATAMODL
        if nints is not None:
            self.obs[key]['NINTS'][index] = nints
        if effinttm is not None:
            self.obs[key]['EFFINTTM'][index] = effinttm
        self.obs[key]['BUNIT'][index] = hdul['SCI'].header['BUNIT']
        if xoffset is not None:
            self.obs[key]['XOFFSET'][index] = xoffset
        if yoffset is not None:
            self.obs[key]['YOFFSET'][index] = yoffset
        if crpix1 is not None:
            self.obs[key]['CRPIX1'][index] = crpix1
        if crpix2 is not None:
            self.obs[key]['CRPIX2'][index] = crpix2
        if blurfwhm is not None:
            self.obs[key]['BLURFWHM'][index] = blurfwhm
        self.obs[key]['FITSFILE'][index] = fitsfile
        if maskfile is not None:
            self.obs[key]['MASKFILE'][index] = maskfile
        if update_pxar:
            try:
                pxar = pyfits.getheader(self.obs[key]['FITSFILE'][index], 'SCI')['PIXAR_SR']
                self.obs[key]['PIXAR_SR'][index] = pxar
            except:
                pass
        hdul.close()
        
        pass
    
    def update_src(self,
                   key,
                   index,
                   tab):
        """
        Update the content of the source database.
        
        Parameters
        ----------
        key : str
            Database key of the source to be updated.
        index : int
            Database index of the source to be updated.
        tab : astropy.table.Table
            Astropy table of the companions to be saved to the source database.
        
        Returns
        -------
        None.
        
        """
        
        # Update spaceKLIP database.
        try:
            if isinstance(self.src[key], list):
                nindex = len(self.src[key])
                if nindex > index:
                    self.src[key][index] = tab
                else:
                    self.src[key] += [None] * (index - nindex) + [tab]
        except KeyError:
            self.src[key] = [None] * index + [tab]
        
        pass
    
    def summarize(self):
        """
        Succinctly summarize the contents of the observations database, i.e.,
        how many files are present at each level of reduction, what kind (SCI,
        REF, TA), etc.
        
        Returns
        -------
        None.
        
        """
        
        def short_concat_name(concat_name):
            """
            Return a shorter, less redundant name for a coronagraphic mode,
            useful for display.
            
            Parameters
            ----------
            concat_name : str
                Concatenation name.
            
            Returns
            -------
            concat_name : str
                Less redundant concatenation name.
            
            """
            
            parts = concat_name.split('_')
            
            return "_".join([parts[1], parts[3], parts[5], ])
        
        for mode in self.obs:
            print(short_concat_name(mode))
            tab = self.obs[mode]
            for stage in [0, 1, 2]:
                stagetab = tab[tab['DATAMODL'] == f'STAGE{stage}']
                if len(stagetab):
                    nsci = np.sum(stagetab['TYPE'] == 'SCI')
                    nref = np.sum(stagetab['TYPE'] == 'REF')
                    nta = np.sum((stagetab['TYPE'] == 'SCI_TA') | (stagetab['TYPE'] == 'REF_TA'))
                    nbg = np.sum((stagetab['TYPE'] == 'SCI_BG') | (stagetab['TYPE'] == 'REF_BG'))
                    
                    summarystr = f'\tSTAGE{stage}: {len(stagetab)} files;\t{nsci} SCI, {nref} REF'
                    if nta:
                        summarystr += f', {nta} TA'
                    if nbg:
                        summarystr += f', {nbg} BG'
                    print(summarystr)
            if hasattr(self, 'red') and mode in self.red:
                tab = self.red[mode]
                stage = 3
                stagetab = tab[tab['DATAMODL'] == f'STAGE{stage}']
                if len(stagetab):
                    s3types = sorted(list(set(tab['TYPE'].value)))
                    nta = np.sum((stagetab['TYPE'] == 'SCI_TA') | (stagetab['TYPE'] == 'REF_TA'))
                    nbg = np.sum((stagetab['TYPE'] == 'SCI_BG') | (stagetab['TYPE'] == 'REF_BG'))

                    summarystr = f'\tSTAGE{stage}: {len(stagetab)} files;\t'
                    for i, typestr in enumerate(s3types):
                        ntype = np.sum(stagetab['TYPE'] == typestr)
                        summarystr += (', ' if i>0 else '') + f'{ntype} {typestr}'
                    print(summarystr)

def create_database(output_dir, 
                    pid, 
                    obsids=None,
                    input_dir=None,
                    psflibpaths=None, 
                    bgpaths=None,
                    assoc_using_targname=True,
                    verbose=True,
                    readlevel='012',
                    cr_from_siaf=False,
                    **kwargs):

    """ Create a spaceKLIP database from JWST data

    Automatically searches for uncal.fits in the input directory and creates 
    a database of the JWST data. Only works for stage0, stage1, or stage2 data
    by default; set readlevel=3 to read in stage 3 outputs.

    Parameters
    ----------
    output_dir : str
        Directory to save the database.
    pid : str
        Program ID.
    obsids : list of ints, optional
        List of observation numbers. If not set, will search for all
        observations in the input directory.
    input_dir : str
        Directory containing the JWST data. If not set, will search for
        MAST directory.
    psflibpaths : list of paths, optional
        List of paths of the input PSF references. Make sure that they are
        NOT duplicated in the 'datapaths'. The default is None.
    bgpaths : list of paths, optional
        List of paths of the input MIRI background observations. Make sure
        that they ARE duplicated in the 'datapaths' or 'psflibpaths'. The
        default is None.
    assoc_using_targname : bool, optional
        Associate observations using the TARGNAME keyword. The default is True.
    readlevel : str or int
        Level of data products to read in, either '012' (or an integer 0,1,2) to read in
        individual exposures, or '3' to read in level-3 PSF-subtracted products,
        or '4' to read in extracted PSF fitting products.
    verbose : bool, optional
        Print information to the screen. The default is True.
    
    Keyword Arguments
    -----------------
    sca : str
        Name of detector (e.g., 'along' or 'a3')
    filt : str
        Return files observed in given filter.
    file_type : str
        'uncal.fits', 'rateints.fits', 'calints.fits', etc.
    exp_type : str
        Exposure type such as NRC_TACQ, NRC_TACONFIRM
    vst_grp_act : str
        The _<gg><s><aa>_ portion of the file name.
        hdr0['VISITGRP'] + hdr0['SEQ_ID'] + hdr0['ACT_ID']
    apername : str
        Name of aperture (e.g., NRCA5_FULL)
    apername_pps : str
        Name of aperture from PPS (e.g., NRCA5_FULL)
    readlevel : str or int
        Set this to 3 invoke the code for re-reading in level 3 output
        products. By default, only levels 0,1,2 data will be read and indexed.
    """

    from webbpsf_ext.imreg_tools import get_files

    if input_dir is None:
        mast_dir = os.getenv('JWSTDOWNLOAD_OUTDIR')
        input_dir = os.path.join(mast_dir, f'{pid:05d}')

    # Check if obsids is not a list, tuple, or numpy array
    if not isinstance(obsids, (list, tuple, np.ndarray)):
        obsids = [obsids]

    # Cycle through all obsids and get the files in a single list
    fitsfiles = [get_files(input_dir, pid, obsid=oid, **kwargs) for oid in obsids]
    fitsfiles = [f for sublist in fitsfiles for f in sublist]
    datapaths = [os.path.join(input_dir, f) for f in fitsfiles]

    # Initialize the spaceKLIP database and read the input FITS files.
    db = Database(output_dir=output_dir)
    db.verbose = verbose


    if str(readlevel) in '012' or str(readlevel) == '012':
        db.read_jwst_s012_data(datapaths=datapaths,
                               psflibpaths=psflibpaths,
                               bgpaths=bgpaths,
                               cr_from_siaf=cr_from_siaf,
                               assoc_using_targname=assoc_using_targname)
    elif str(readlevel) == '3':
        # the above get_files usage won't match KLIP outputsa, so find them here
        datapaths_klip = sorted(glob.glob(os.path.join(input_dir, "*KLmodes-all.fits")))
        db.read_jwst_s3_data(datapaths=datapaths+datapaths_klip,
                             cr_from_siaf=cr_from_siaf,
                            )
    elif str(readlevel) == '4':
        db.read_jwst_s4_data(datapaths=datapaths,
                            )
    else:
        raise ValueError("Invalid/unknown value for readlevel parameter")

    return db
