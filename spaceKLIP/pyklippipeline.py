from __future__ import division

import matplotlib

# =============================================================================
# IMPORTS
# =============================================================================

import os
import pdb
import sys

from astropy.io import fits 
import numpy as np

import json
import pyklip.klip

from astropy import wcs
from jwst.pipeline import Detector1Pipeline, Image2Pipeline, Coron3Pipeline
from pyklip import parallelized, rdi
from pyklip.instruments.JWST import JWSTData
from pyklip.klip import _rotate_wcs_hdr
from spaceKLIP.psf import get_transmission
from spaceKLIP.utils import pop_pxar_kw

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# =============================================================================
# MAIN
# =============================================================================

def run_obs(database,
            restrict_to=None,
            kwargs={},
            subdir='klipsub'):
    """
    Run pyKLIP on the input observations database.
    
    Parameters
    ----------
    database : spaceKLIP.Database
        SpaceKLIP database on which pyKLIP shall be run.
    kwargs : dict, optional
        Keyword arguments for the pyklip.parallelized.klip_dataset method.
        Available keywords are:

        - mode : list of str, optional
            Subtraction modes that shall be looped over. Possible values are
            'ADI', 'RDI', and 'ADI+RDI'. The default is ['ADI+RDI'].
        - annuli : list of int, optional
            Numbers of subtraction annuli that shall be looped over. The
            default is [1].
        - annuli_spacing : str, optional
            How to distribute the annuli radially. Currently three options. 
                - 'constant' : equally spaced (Default), 
                - 'log' : logarithmical expansion with r, and 
                - 'linear' : linearly expansion with r
        - subsections : list of int, optional
            Numbers of subtraction subsections that shall be looped over. The
            default is [1].
        - numbasis : list of int, optional
            Number of KL modes that shall be looped over. The default is [1, 2,
            5, 10, 20, 50, 100].
        - movement : float, optional
            Minimum amount of movement (pix) of an astrophysical source to
            consider using that image as a reference PSF. The default is 1.
        - highpass : bool or float, optional
            If True, run a Gaussian high pass filter (default size is sigma=imgsize/10).
            Can also be a number specifying FWHM of box in pixel units. Default is False.
        - verbose : bool, optional
            Verbose mode? The default is False.
        - save_rolls : bool, optional
            Save each processed roll separately? The default is False.
        - save_full_output : bool, optional
            Save the full output cube before flattening? The default is False.

        The default is {}.
    subdir : str, optional
        Name of the directory where the data products shall be saved. The
        default is 'klipsub'.
    
    Returns
    -------
    None.

    """
    
    # Check input.
    if 'mode' not in kwargs.keys():
        kwargs['mode'] = ['ADI+RDI']
    if not isinstance(kwargs['mode'], list):
        kwargs['mode'] = [kwargs['mode']]
    if 'annuli' not in kwargs.keys():
        kwargs['annuli'] = [1]
    if not isinstance(kwargs['annuli'], list):
        kwargs['annuli'] = [kwargs['annuli']]
    if 'annuli_spacing' not in kwargs.keys():
        kwargs['annuli_spacing'] = 'constant'
    if 'subsections' not in kwargs.keys():
        kwargs['subsections'] = [1]
    if not isinstance(kwargs['subsections'], list):
        kwargs['subsections'] = [kwargs['subsections']]
    if 'numbasis' not in kwargs.keys():
        kwargs['numbasis'] = [1, 2, 5, 10, 20, 50, 100]
    if not isinstance(kwargs['numbasis'], list):
        kwargs['numbasis'] = [kwargs['numbasis']]
    if 'IWA' not in kwargs.keys():
        kwargs['IWA'] = 1.
    kwargs_temp = kwargs.copy()
    if 'movement' not in kwargs_temp.keys():
        kwargs_temp['movement'] = 1.
    kwargs_temp['calibrate_flux'] = False
    if 'verbose' not in kwargs_temp.keys():
        kwargs_temp['verbose'] = database.verbose
    if 'save_rolls' not in kwargs_temp.keys():
        kwargs_temp['save_ints'] = False
        kwargs_temp['save_rolls'] = False
    else:
        # Note pyKLIP uses save_ints as the keyword for this, but we want to use save_rolls in our pipeline for clarity,
        # so we need to set save_ints to the same value as save_rolls.
        kwargs_temp['save_ints'] = kwargs_temp['save_rolls']
    if 'save_full_output' not in kwargs_temp.keys():
        kwargs_temp['save_full_output'] = False
    if 'highpass' not in kwargs_temp.keys():
        kwargs_temp['highpass'] = False
    
    # Set output directory.
    output_dir = os.path.join(database.output_dir, subdir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    kwargs_temp['outputdir'] = output_dir
    
    # Loop through concatenations.
    datapaths = []
    for i, key in enumerate(database.obs.keys()):

        # if we limit to only processing some concatenations, check whether this concatenation matches the pattern
        if (restrict_to is not None) and (restrict_to not in key):
            continue

        log.info('--> Concatenation ' + key)
        
        filepaths, psflib_filepaths, maxnumbasis = get_pyklip_filepaths(database, key, return_maxbasis=True)
        if 'maxnumbasis' not in kwargs_temp.keys() or kwargs_temp['maxnumbasis'] is None:
            kwargs_temp['maxnumbasis'] = maxnumbasis
        
        # Run KLIP subtraction.
        for mode in kwargs['mode']:
            
            # Initialize pyKLIP dataset.
            pop_pxar_kw(np.append(filepaths, psflib_filepaths))
            dataset = JWSTData(filepaths, psflib_filepaths, 
                               highpass=kwargs_temp['highpass'],
                               center_include_offset=False,
                               center_keywords=['STARCENX','STARCENY'])
            dataset.IWA = kwargs['IWA']
            kwargs_temp['dataset'] = dataset
            kwargs_temp['aligned_center'] = dataset.psflib.aligned_center
            kwargs_temp['psf_library'] = dataset.psflib
            kwargs_temp['mode'] = mode

            # Can run pyKLIP multiple times on the same dataset with different
            # annuli and subsections.
            for annu in kwargs['annuli']:
                for subs in kwargs['subsections']:
                    log.info('  --> pyKLIP: mode = ' + mode + ', annuli = ' + str(annu) + ', subsections = ' + str(subs))
                    fileprefix = mode + '_NANNU' + str(annu) + '_NSUBS' + str(subs) + '_' + key
                    
                    # Add/update ramaining keywords.
                    kwargs_temp['fileprefix'] = fileprefix
                    kwargs_temp['annuli'] = annu
                    kwargs_temp['subsections'] = subs
                    kwargs_temp_temp = kwargs_temp.copy()

                    # Need to cleanup some kwargs that we're using but pyKLIP doesn't
                    del kwargs_temp_temp['save_full_output']
                    del kwargs_temp_temp['save_rolls']
                    del kwargs_temp_temp['IWA']
                    parallelized.klip_dataset(**kwargs_temp_temp)

                    # Get reduction path.
                    datapath = os.path.join(output_dir, fileprefix + '-KLmodes-all.fits')
                    datapaths += [datapath]
                    
                    # Update reduction header.
                    ww_sci = np.where(database.obs[key]['TYPE'] == 'SCI')[0]
                    head_primary = fits.getheader(database.obs[key]['FITSFILE'][ww_sci[0]], 0)
                    head_sci = fits.getheader(database.obs[key]['FITSFILE'][ww_sci[0]], 'SCI')
                    head_sci['NAXIS'] = 2
                    hdul = fits.open(datapath)
                    
                    # Adding comments to a handful of the FITS header keywords populated by pyKLIP.
                    hdul[0].header.comments['PSFCENTX'] = 'Star center along the X axis, 0-indexed'
                    hdul[0].header.comments['PSFCENTY'] = 'Star center along the Y axis, 0-indexed'
                    hdul[0].header.comments['STARCENX'] = 'Star center along the X axis, 1-indexed'
                    hdul[0].header.comments['STARCENY'] = 'Star center along the Y axis, 1-indexed'
                    
                    hdul[0].header['TELESCOP'] = (database.obs[key]['TELESCOP'][ww_sci[0]], head_primary.comments["TELESCOP"])
                    hdul[0].header['TARGPROP'] = (database.obs[key]['TARGPROP'][ww_sci[0]], head_primary.comments["TARGPROP"])
                    hdul[0].header['TARG_RA'] = (database.obs[key]['TARG_RA'][ww_sci[0]], head_primary.comments["TARG_RA"])
                    hdul[0].header['TARG_DEC'] = (database.obs[key]['TARG_DEC'][ww_sci[0]], head_primary.comments["TARG_DEC"])
                    hdul[0].header['INSTRUME'] = (database.obs[key]['INSTRUME'][ww_sci[0]], head_primary.comments["INSTRUME"])
                    hdul[0].header['DETECTOR'] = (database.obs[key]['DETECTOR'][ww_sci[0]], head_primary.comments["DETECTOR"])
                    hdul[0].header['FILTER'] = (database.obs[key]['FILTER'][ww_sci[0]], head_primary.comments["FILTER"])
                    hdul[0].header['CWAVEL'] = (database.obs[key]['CWAVEL'][ww_sci[0]], "[micron] Filter pivot wavelength")
                    hdul[0].header['DWAVEL'] = (database.obs[key]['DWAVEL'][ww_sci[0]], "[micron] Filter effective width")
                    hdul[0].header['PUPIL'] = (database.obs[key]['PUPIL'][ww_sci[0]], head_primary.comments["PUPIL"])
                    hdul[0].header['CORONMSK'] = (database.obs[key]['CORONMSK'][ww_sci[0]], head_primary.comments["CORONMSK"])
                    hdul[0].header['EXP_TYPE'] = (database.obs[key]['EXP_TYPE'][ww_sci[0]], head_primary.comments["EXP_TYPE"])
                    hdul[0].header['EXPSTART'] = (np.min(database.obs[key]['EXPSTART'][ww_sci]), head_primary.comments["EXPSTART"])
                    hdul[0].header['NINTS'] = (np.sum(database.obs[key]['NINTS'][ww_sci]), head_primary.comments["NINTS"])
                    hdul[0].header['EFFINTTM'] = (database.obs[key]['EFFINTTM'][ww_sci[0]], head_primary.comments["EFFINTTM"])
                    hdul[0].header['SUBARRAY'] = (database.obs[key]['SUBARRAY'][ww_sci[0]], head_primary.comments["SUBARRAY"])
                    hdul[0].header['APERNAME'] = (database.obs[key]['APERNAME'][ww_sci[0]], head_primary.comments["APERNAME"])
                    hdul[0].header['PPS_APER'] = (database.obs[key]['PPS_APER'][ww_sci[0]], head_primary.comments["PPS_APER"])
                    hdul[0].header['PIXSCALE'] = (database.obs[key]['PIXSCALE'][ww_sci[0]], "[arcsec/pixel] Pixel scale")
                    try:
                        hdul[0].header['PIXAR_SR'] = (database.obs[key]['PIXAR_SR'][ww_sci[0]], head_sci.comments["PIXAR_SR"])
                    except:
                        pass                

                    hdul[0].header['MODE'] = (mode, "PSF subtraction mode: ADI, RDI, or ADI+RDI")
                    hdul[0].header['ANNULI'] = (annu, "Number of subtraction annuli")
                    hdul[0].header['ANNSPACE'] = (kwargs['annuli_spacing'], 'Radial annulus spacing: constant, log, or linear')
                    hdul[0].header['SUBSECTS'] = (subs, "Number of subtraction subsections within each annulus")
                    hdul[0].header['HIGHPASS'] = (kwargs_temp['highpass'], 'High-pass filter setting used by pyKLIP')
                    hdul[0].header['BUNIT'] = (database.obs[key]['BUNIT'][ww_sci[0]], head_sci.comments["BUNIT"])

                    w = wcs.WCS(head_sci)
                    _rotate_wcs_hdr(w, database.obs[key]['ROLL_REF'][ww_sci[0]])
                    hdul[0].header['WCSAXES'] = (head_sci['WCSAXES'], head_sci.comments["WCSAXES"])
                    hdul[0].header['CRPIX1'] = (head_sci['STARCENX'], head_sci.comments["CRPIX1"])
                    hdul[0].header['CRPIX2'] = (head_sci['STARCENY'], head_sci.comments["CRPIX2"])
                    hdul[0].header['CRVAL1'] = (head_sci['CRVAL1'], head_sci.comments["CRVAL1"])
                    hdul[0].header['CRVAL2'] = (head_sci['CRVAL2'], head_sci.comments["CRVAL2"])
                    hdul[0].header['CTYPE1'] = (head_sci['CTYPE1'], head_sci.comments["CTYPE1"])
                    hdul[0].header['CTYPE2'] = (head_sci['CTYPE2'], head_sci.comments["CTYPE2"])
                    hdul[0].header['CUNIT1'] = (head_sci['CUNIT1'], head_sci.comments["CUNIT1"])
                    hdul[0].header['CUNIT2'] = (head_sci['CUNIT2'], head_sci.comments["CUNIT2"])
                    hdul[0].header['CD1_1'] = (w.wcs.cd[0, 0], head_sci.comments["CD1_1"])
                    hdul[0].header['CD1_2'] = (w.wcs.cd[0, 1], head_sci.comments["CD1_2"])
                    hdul[0].header['CD2_1'] = (w.wcs.cd[1, 0], head_sci.comments["CD2_1"])
                    hdul[0].header['CD2_2'] = (w.wcs.cd[1, 1], head_sci.comments["CD2_2"])
                    if not np.isnan(database.obs[key]['BLURFWHM'][ww_sci[0]]):
                        hdul[0].header['BLURFWHM'] = (database.obs[key]['BLURFWHM'][ww_sci[0]], "[pixel] FWHM of applied Gaussian blur")
                    hdul.writeto(datapath, output_verify='fix', overwrite=True)
                    hdul.close()

                    # If requested, save the full cube before flattening
                    if kwargs_temp['save_full_output']:
                        # Cube is held in dataset.output
                        intsfile = os.path.join(output_dir, fileprefix + '-KLmodes-all_fulloutput.fits')
                        hdul_full = fits.HDUList([fits.PrimaryHDU(data=dataset.output)])
                        # Set header keywords for the full cube as well, using the same header as the final output
                        # but with updated NAXIS and NINTS.
                        hdul_full[0].header = hdul[0].header.copy()
                        hdul_full[0].header['NAXIS'] = len(dataset.output.shape)
                        hdul_full[0].header['NINTS'] = dataset.output.shape[0]
                        hdul_full.writeto(intsfile, output_verify='fix', overwrite=True)

                    # If requested, save each roll separately.
                    if kwargs_temp['save_ints']:
                        n_roll = 1
                        for j in ww_sci:
                            fitsfile = os.path.split(database.obs[key]['FITSFILE'][j])[1]
                            head_primary = fits.getheader(database.obs[key]['FITSFILE'][j], 0)
                            head_sci = fits.getheader(database.obs[key]['FITSFILE'][j], 'SCI')
                            ww = [k for k in range(len(dataset._filenames)) if fitsfile in dataset._filenames[k]]
                            hdul = fits.open(datapath)
                            if dataset.allints.shape[1] == 1:
                                hdul[0].data = np.nanmedian(dataset.allints[:, :, ww, :, :], axis=(1, 2))
                            else:
                                hdul[0].data = np.nanmedian(dataset.allints[:, :, ww, :, :], axis=2)
                            hdul[0].header['NINTS'] = (database.obs[key]['NINTS'][j], head_primary.comments['NINTS'])
                            hdul[0].header['WCSAXES'] = (head_sci['WCSAXES'], head_sci.comments['WCSAXES'])
                            hdul[0].header['CRPIX1'] = (head_sci['STARCENX'], head_sci.comments['CRPIX1'])
                            hdul[0].header['CRPIX2'] = (head_sci['STARCENY'], head_sci.comments['CRPIX2'])
                            hdul[0].header['CRVAL1'] = (head_sci['CRVAL1'], head_sci.comments['CRVAL1'])
                            hdul[0].header['CRVAL2'] = (head_sci['CRVAL2'], head_sci.comments['CRVAL2'])
                            hdul[0].header['CTYPE1'] = (head_sci['CTYPE1'], head_sci.comments['CTYPE1'])
                            hdul[0].header['CTYPE2'] = (head_sci['CTYPE2'], head_sci.comments['CTYPE2'])
                            hdul[0].header['CUNIT1'] = (head_sci['CUNIT1'], head_sci.comments['CUNIT1'])
                            hdul[0].header['CUNIT2'] = (head_sci['CUNIT2'], head_sci.comments['CUNIT2'])
                            hdul[0].header['CD1_1'] = (head_sci['CD1_1'], head_sci.comments['CD1_1'])
                            hdul[0].header['CD1_2'] = (head_sci['CD1_2'], head_sci.comments['CD1_2'])
                            hdul[0].header['CD2_1'] = (head_sci['CD2_1'], head_sci.comments['CD2_1'])
                            hdul[0].header['CD2_2'] = (head_sci['CD2_2'], head_sci.comments['CD2_2'])
                            hdul.writeto(datapath.replace('-KLmodes-all.fits', '-KLmodes-all_roll%.0f.fits' % n_roll), output_verify='fix', overwrite=True)
                            hdul.close()
                            n_roll += 1

        # Save corresponding observations database.
        file = os.path.join(output_dir, key + '.dat')
        for col in ['CENTER_MASK', 'ALIGN_MASK', 'CENTER_SHIFT', 'ALIGN_SHIFT']:
            if col in database.obs[key].colnames:
                database.obs[key][col] = [
                    str([x.tolist() if hasattr(x, "tolist") else x for x in row])
                    if row is not None and hasattr(row, '__iter__')
                    else str(row)
                    for row in database.obs[key][col]
                ]

        database.obs[key].write(file, format='ascii', overwrite=True)

        # Compute and save corresponding transmission mask.
        file = os.path.join(output_dir, key + '_psfmask.fits')
        mask = get_transmission(database.obs[key])
        ww_sci = np.where(database.obs[key]['TYPE'] == 'SCI')[0]
        if mask is not None:
            hdul = fits.open(database.obs[key]['MASKFILE'][ww_sci[0]])
            hdul[0].data = None
            hdul['SCI'].data = mask
            hdul.writeto(file, output_verify='fix', overwrite=True)

    # Read reductions into database.
    database.read_jwst_s3_data(datapaths)

    pass


def get_pyklip_filepaths(database, key, return_maxbasis=False):
    '''
    Quick wrapper function to get the filepath information (in addition
    to the maxnumbasis) for pyKLIP from a spaceKLIP database. 

    Parameters
    ----------
    database : spaceKLIP.Database
        SpaceKLIP database on which pyKLIP shall be run.
    key : str
        Key for the concatenation of interest in the spaceKLIP database
    return_maxbasis : bool, optional
        Toggle for whether to additionally return the 
        maximum number of basis vectors. 

    Returns
    -------
    filepaths : 1D-array 
        List of science image file names
    psflib_filepaths : 1D-array 
        List of reference image file names
    maxnumbasis : int, optional
        The maximum number of basis vectors available. 
    '''

    filepaths = []
    psflib_filepaths = []
    first_sci = True
    nints = []
    nfitsfiles = len(database.obs[key])
    for j in range(nfitsfiles):
        if database.obs[key]['TYPE'][j] == 'SCI':
            filepaths += [database.obs[key]['FITSFILE'][j]]
            if first_sci:
                first_sci = False
            else:
                nints += [database.obs[key]['NINTS'][j]]
        elif database.obs[key]['TYPE'][j] == 'REF':
            psflib_filepaths += [database.obs[key]['FITSFILE'][j]]
            nints += [database.obs[key]['NINTS'][j]]
    filepaths = np.array(filepaths)
    psflib_filepaths = np.array(psflib_filepaths)
    nints = np.array(nints)
    maxnumbasis = np.sum(nints)

    if return_maxbasis:
        return filepaths, psflib_filepaths, maxnumbasis
    else:
        return filepaths, psflib_filepaths
