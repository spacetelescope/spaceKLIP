from __future__ import division

import matplotlib

# =============================================================================
# IMPORTS
# =============================================================================

import os
import pdb
import sys

import astropy.io.fits as pyfits
import astropy.stats
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

import json
import pysiaf
import webbpsf_ext

from copy import deepcopy
from jwst.pipeline import Detector1Pipeline, Image2Pipeline, Coron3Pipeline
import jwst.datamodels
from pyklip import parallelized
from scipy.ndimage import gaussian_filter, median_filter, fourier_shift, rotate
from scipy.ndimage import shift as spline_shift
from scipy.optimize import leastsq, minimize
from scipy.interpolate import griddata
from skimage.registration import phase_cross_correlation
from spaceKLIP import utils as ut
from spaceKLIP.psf import JWST_PSF
from spaceKLIP.xara import core
from spaceKLIP.utils import gaussian_kernel
from webbpsf_ext import robust
from webbpsf_ext.coords import dist_image
from webbpsf_ext.webbpsf_ext_core import _transmission_map
from tqdm.auto import trange
import copy
import pyklip.fakes as fakes
from spaceKLIP.psf import get_offsetpsf
from spaceKLIP.pyklippipeline import get_pyklip_filepaths
from pyklip.instruments.JWST import JWSTData
from webbpsf.constants import JWST_CIRCUMSCRIBED_DIAMETER
from astropy.io import fits
from spaceKLIP.starphot import get_stellar_magnitudes, read_spec_file
import scipy.ndimage
import lmfit
import re

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# =============================================================================
# MAIN
# =============================================================================

# Load NIRCam true mask centers and filter-dependent shifts from Jarron.
path = 'resources/crpix_jarron.json'
path = os.path.join(os.path.split(os.path.abspath(__file__))[0], path)
file = open(path, 'r')
crpix_jarron = json.load(file)
file.close()
path = 'resources/filter_shifts_jarron.json'
path = os.path.join(os.path.split(os.path.abspath(__file__))[0], path)
file = open(path, 'r')
filter_shifts_jarron = json.load(file)
file.close()

class ImageTools():
    """
    The spaceKLIP image manipulation tools class.

    """

    def __init__(self,
                 database):
        """
        Initialize the spaceKLIP image manipulation tools class.

        Parameters
        ----------
        database : spaceKLIP.Database
            SpaceKLIP database on which the image manipulation steps shall be
            run.

        Returns
        -------
        None.

        """

        # Make an internal alias of the spaceKLIP database class.
        self.database = database

        pass

    def _get_output_dir(self, subdir):
        """Utility function to get full output dir path, and create it if needed"""
        # Set output directory.
        output_dir = os.path.join(self.database.output_dir, subdir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        return output_dir

    def _iterate_function_over_files(self, types, file_transformation_function, restrict_to=None):
        """ Iterate some callable function over all files in a database.

        This is a repetitive pattern used in many of the image processing functions, so
        we abstract it here to reduce code repetition.

        The file transformation function should take one filename as an input, perform some transformation or image processing
        write out the file to some new path, and return the output filename.
        Any other arguments should be provided prior to passing in the function, for instance
        via functools.partial if necessary.

        """

        # Loop through concatenations.
        for i, key in enumerate(self.database.obs.keys()):

            # if we limit to only processing some concatenations, check whether this concatenation matches the pattern
            if (restrict_to is not None) and (restrict_to not in key):
                continue

            log.info('--> Concatenation ' + key)

            # Loop through FITS files.
            nfitsfiles = len(self.database.obs[key])
            for j in range(nfitsfiles):

                # Read FITS file.
                filename = self.database.obs[key]['FITSFILE'][j]

                # Only process files of the specified types.
                #  (skip any files with types that are not in the list of types.)
                if self.database.obs[key]['TYPE'][j] in types:

                    output_filename = file_transformation_function(filename)

                    # Update spaceKLIP database.
                    self.database.update_obs(key, j, output_filename)

    def remove_frames(self,
                      index=[0],
                      types=['SCI', 'SCI_BG', 'REF', 'REF_BG'],
                      subdir='removed'):
        """
        Remove individual frames from the data.

        Parameters
        ----------
        index : int or list of int or dict of list of list of int, optional
            Indices (0-indexed) of the frames to be removed. If int, then only
            a single frame will be removed from each observation. If list of
            int, then multiple frames can be removed from each observation. If
            dict of list of list of int, then the dictionary keys must match
            the keys of the observations database, and the number of entries in
            the lists must match the number of observations in the
            corresponding concatenation. Then, a different list of int can be
            used for each individual observation to remove different frames.
            The default is [0].
        types : list of str, optional
            List of data types from which the frames shall be removed. The
            default is ['SCI', 'SCI_BG', 'REF', 'REF_BG'].
        subdir : str, optional
            Name of the directory where the data products shall be saved. The
            default is 'removed'.

        Returns
        -------
        None.

        """

        # Check input.
        if isinstance(index, int):
            index = [index]

        # Set output directory.
        output_dir = os.path.join(self.database.output_dir, subdir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Loop through concatenations.
        for i, key in enumerate(self.database.obs.keys()):
            log.info('--> Concatenation ' + key)

            # Loop through FITS files.
            nfitsfiles = len(self.database.obs[key])
            for j in range(nfitsfiles):

                # Read FITS file and PSF mask.
                fitsfile = self.database.obs[key]['FITSFILE'][j]
                data, erro, pxdq, head_pri, head_sci, is2d, imshifts, maskoffs = ut.read_obs(fitsfile)
                maskfile = self.database.obs[key]['MASKFILE'][j]
                mask = ut.read_msk(maskfile)
                nints = self.database.obs[key]['NINTS'][j]

                # Skip file types that are not in the list of types.
                if self.database.obs[key]['TYPE'][j] in types:

                    # Remove frames.
                    head, tail = os.path.split(fitsfile)
                    log.info('  --> Frame removal: ' + tail)
                    try:
                        index_temp = index[key][j]
                    except:
                        index_temp = index.copy()
                    log.info('  --> Frame removal: removing frame(s) ' + str(index_temp))
                    data = np.delete(data, index_temp, axis=0)
                    erro = np.delete(erro, index_temp, axis=0)
                    pxdq = np.delete(pxdq, index_temp, axis=0)
                    if imshifts is not None:
                        imshifts = np.delete(imshifts, index_temp, axis=0)
                    if maskoffs is not None:
                        maskoffs = np.delete(maskoffs, index_temp, axis=0)
                    nints = data.shape[0]

                # Write FITS file and PSF mask.
                head_pri['NINTS'] = nints
                fitsfile = ut.write_obs(fitsfile, output_dir, data, erro, pxdq, head_pri, head_sci, is2d, imshifts, maskoffs)
                maskfile = ut.write_msk(maskfile, mask, fitsfile)

                # Update spaceKLIP database.
                self.database.update_obs(key, j, fitsfile, maskfile, nints=nints)

        pass

    def crop_frames(self,
                    npix=1,
                    types=['SCI', 'SCI_BG', 'REF', 'REF_BG'],
                    subdir='cropped'):
        """
        Crop all frames.

        Parameters
        ----------
        npix : int or list of four int, optional
            Number of pixels to be cropped from the frames. If int, the same
            number of pixels will be cropped on each side. If list of four int,
            a different number of pixels can be cropped from the [left, right,
            bottom, top] of the frames. The default is 1.
        types : list of str, optional
            List of data types from which the frames shall be cropped. The
            default is ['SCI', 'SCI_BG', 'REF', 'REF_BG'].
        subdir : str, optional
            Name of the directory where the data products shall be saved. The
            default is 'cropped'.

        Returns
        -------
        None.

        """

        # Check input.
        if isinstance(npix, int):
            npix = [npix, npix, npix, npix]  # left, right, bottom, top
        if len(npix) != 4:
            raise UserWarning('Parameter npix must either be an int or a list of four int (left, right, bottom, top)')

        # Set output directory.
        output_dir = os.path.join(self.database.output_dir, subdir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Loop through concatenations.
        for i, key in enumerate(self.database.obs.keys()):
            log.info('--> Concatenation ' + key)

            # Loop through FITS files.
            nfitsfiles = len(self.database.obs[key])
            for j in range(nfitsfiles):

                # Read FITS file and PSF mask.
                fitsfile = self.database.obs[key]['FITSFILE'][j]
                data, erro, pxdq, head_pri, head_sci, is2d, imshifts, maskoffs = ut.read_obs(fitsfile)
                maskfile = self.database.obs[key]['MASKFILE'][j]
                mask = ut.read_msk(maskfile)
                crpix1 = self.database.obs[key]['CRPIX1'][j]
                crpix2 = self.database.obs[key]['CRPIX2'][j]

                # Skip file types that are not in the list of types.
                if self.database.obs[key]['TYPE'][j] in types:

                    # Crop frames.
                    head, tail = os.path.split(fitsfile)
                    log.info('  --> Frame cropping: ' + tail)
                    sh = data.shape
                    data = data[:, npix[2]:-npix[3], npix[0]:-npix[1]]
                    erro = erro[:, npix[2]:-npix[3], npix[0]:-npix[1]]
                    pxdq = pxdq[:, npix[2]:-npix[3], npix[0]:-npix[1]]
                    if mask is not None:
                        mask = mask[npix[2]:-npix[3], npix[0]:-npix[1]]
                    crpix1 -= npix[0]
                    crpix2 -= npix[2]
                    log.info('  --> Frame cropping: old shape = ' + str(sh[1:]) + ', new shape = ' + str(data.shape[1:]))

                # Write FITS file and PSF mask.
                head_sci['CRPIX1'] = crpix1
                head_sci['CRPIX2'] = crpix2
                fitsfile = ut.write_obs(fitsfile, output_dir, data, erro, pxdq, head_pri, head_sci, is2d, imshifts, maskoffs)
                maskfile = ut.write_msk(maskfile, mask, fitsfile)

                # Update spaceKLIP database.
                self.database.update_obs(key, j, fitsfile, maskfile, crpix1=crpix1, crpix2=crpix2)

        pass

    def pad_frames(self,
                   npix=1,
                   cval=np.nan,
                   types=['SCI', 'SCI_BG', 'REF', 'REF_BG'],
                   subdir='padded'):
        """
        Pad all frames.

        Parameters
        ----------
        npix : int or list of four int, optional
            Number of pixels to be padded around the frames. If int, the same
            number of pixels will be padded on each side. If list of four int,
            a different number of pixels can be padded on the [left, right,
            bottom, top] of the frames. The default is 1.
        cval : float, optional
            Fill value for the padded pixels. The default is nan.
        types : list of str, optional
            List of data types from which the frames shall be padded. The
            default is ['SCI', 'SCI_BG', 'REF', 'REF_BG'].
        subdir : str, optional
            Name of the directory where the data products shall be saved. The
            default is 'padded'.

        Returns
        -------
        None.

        """

        # Check input.
        if isinstance(npix, int):
            npix = [npix, npix, npix, npix]  # left, right, bottom, top
        if len(npix) != 4:
            raise UserWarning('Parameter npix must either be an int or a list of four int (left, right, bottom, top)')

        # Set output directory.
        output_dir = os.path.join(self.database.output_dir, subdir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Loop through concatenations.
        for i, key in enumerate(self.database.obs.keys()):
            log.info('--> Concatenation ' + key)

            # Loop through FITS files.
            nfitsfiles = len(self.database.obs[key])
            for j in range(nfitsfiles):

                # Read FITS file and PSF mask.
                fitsfile = self.database.obs[key]['FITSFILE'][j]
                data, erro, pxdq, head_pri, head_sci, is2d, imshifts, maskoffs = ut.read_obs(fitsfile)
                maskfile = self.database.obs[key]['MASKFILE'][j]
                mask = ut.read_msk(maskfile)
                crpix1 = self.database.obs[key]['CRPIX1'][j]
                crpix2 = self.database.obs[key]['CRPIX2'][j]

                # Skip file types that are not in the list of types.
                if self.database.obs[key]['TYPE'][j] in types:

                    # Crop frames.
                    head, tail = os.path.split(fitsfile)
                    log.info('  --> Frame padding: ' + tail)
                    sh = data.shape
                    data = np.pad(data, ((0, 0), (npix[2], npix[3]), (npix[0], npix[1])), mode='constant', constant_values=cval)
                    erro = np.pad(erro, ((0, 0), (npix[2], npix[3]), (npix[0], npix[1])), mode='constant', constant_values=cval)
                    pxdq = np.pad(pxdq, ((0, 0), (npix[2], npix[3]), (npix[0], npix[1])), mode='constant', constant_values=0)
                    if mask is not None:
                        mask = np.pad(mask, ((npix[2], npix[3]), (npix[0], npix[1])), mode='constant', constant_values=np.nan)
                    crpix1 += npix[0]
                    crpix2 += npix[2]
                    log.info('  --> Frame padding: old shape = ' + str(sh[1:]) + ', new shape = ' + str(data.shape[1:]) + ', fill value = %.2f' % cval)

                # Write FITS file and PSF mask.
                head_sci['CRPIX1'] = crpix1
                head_sci['CRPIX2'] = crpix2
                fitsfile = ut.write_obs(fitsfile, output_dir, data, erro, pxdq, head_pri, head_sci, is2d, imshifts, maskoffs)
                maskfile = ut.write_msk(maskfile, mask, fitsfile)

                # Update spaceKLIP database.
                self.database.update_obs(key, j, fitsfile, maskfile, crpix1=crpix1, crpix2=crpix2)

        pass

    def coadd_frames(self,
                     nframes=None,
                     types=['SCI', 'SCI_BG', 'REF', 'REF_BG'],
                     subdir='coadded'):
        """
        Coadd frames.

        Parameters
        ----------
        nframes : int, optional
            Number of frames to be coadded. Modulo frames will be removed. If
            None, will coadd all frames in an observation. The default is None.
        types : list of str, optional
            List of data types from which the frames shall be coadded. The
            default is ['SCI', 'SCI_BG', 'REF', 'REF_BG'].
        subdir : str, optional
            Name of the directory where the data products shall be saved. The
            default is 'coadded'.

        Returns
        -------
        None.

        """

        # Set output directory.
        output_dir = os.path.join(self.database.output_dir, subdir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # The starting value.
        nframes0 = nframes

        # Loop through concatenations.
        for i, key in enumerate(self.database.obs.keys()):
            log.info('--> Concatenation ' + key)

            # Loop through FITS files.
            nfitsfiles = len(self.database.obs[key])
            for j in range(nfitsfiles):

                # Read FITS file and PSF mask.
                fitsfile = self.database.obs[key]['FITSFILE'][j]
                data, erro, pxdq, head_pri, head_sci, is2d, imshifts, maskoffs = ut.read_obs(fitsfile)
                maskfile = self.database.obs[key]['MASKFILE'][j]
                mask = ut.read_msk(maskfile)
                nints = self.database.obs[key]['NINTS'][j]
                effinttm = self.database.obs[key]['EFFINTTM'][j]

                # If nframes is not provided, collapse everything.
                if nframes0 is None:
                    nframes = nints

                # Skip file types that are not in the list of types.
                if self.database.obs[key]['TYPE'][j] in types:

                    # Coadd frames.
                    head, tail = os.path.split(fitsfile)
                    log.info('  --> Frame coadding: ' + tail)
                    ncoadds = data.shape[0] // nframes
                    data = np.nanmedian(data[:nframes * ncoadds].reshape((nframes, ncoadds, data.shape[-2], data.shape[-1])), axis=0)
                    erro_reshape = erro[:nframes * ncoadds].reshape((nframes, ncoadds, erro.shape[-2], erro.shape[-1]))
                    nsample = np.sum(np.logical_not(np.isnan(erro_reshape)), axis=0)
                    erro = np.true_divide(np.sqrt(np.nansum(erro_reshape**2, axis=0)), nsample)
                    pxdq_temp = pxdq[:nframes * ncoadds].reshape((nframes, ncoadds, pxdq.shape[-2], pxdq.shape[-1]))
                    pxdq = pxdq_temp[0]
                    for k in range(1, nframes):
                        pxdq = np.bitwise_or(pxdq, pxdq_temp[k])
                    if imshifts is not None:
                        imshifts = np.mean(imshifts[:nframes * ncoadds].reshape((nframes, ncoadds, imshifts.shape[-1])), axis=0)
                    if maskoffs is not None:
                        maskoffs = np.mean(maskoffs[:nframes * ncoadds].reshape((nframes, ncoadds, maskoffs.shape[-1])), axis=0)
                    nints = data.shape[0]
                    effinttm *= nframes
                    log.info('  --> Frame coadding: %.0f coadd(s) of %.0f frame(s)' % (ncoadds, nframes))

                # Write FITS file and PSF mask.
                head_pri['NINTS'] = nints
                head_pri['EFFINTTM'] = effinttm
                fitsfile = ut.write_obs(fitsfile, output_dir, data, erro, pxdq, head_pri, head_sci, is2d, imshifts, maskoffs)
                maskfile = ut.write_msk(maskfile, mask, fitsfile)

                # Update spaceKLIP database.
                self.database.update_obs(key, j, fitsfile, maskfile, nints=nints, effinttm=effinttm)

        pass

    def subtract_median(self,
                        types=['SCI', 'SCI_TA', 'SCI_BG', 'REF', 'REF_TA', 'REF_BG'],
                        method='border',
                        sigma=3.0,
                        borderwidth=32,
                        subdir='medsub'):

        """
        Subtract the median from each frame. Clip everything brighter than 5-
        sigma from the background before computing the median.

        Parameters
        ----------
        types : list of str, optional
            List of data types for which the median shall be subtracted. The
            default is ['SCI', 'SCI_TA', 'SCI_BG', 'REF', 'REF_TA', 'REF_BG'].
        subdir : str, optional
            Name of the directory where the data products shall be saved. The
            default is 'medsub'.

        method : str, optional

            - 'robust' for a robust median after masking out bright stars,
            - 'sigma_clipped' for another version of robust median using astropy
               sigma_clipped_stats on the whole image,
            - 'border' for robust median on the outer border region only, to
              ignore the bright stellar PSF in the center,
            - 'simple'  for a simple np.nanmedian.

        sigma : float, optional
            number of standard deviations to use for the clipping limit in
            sigma_clipped_stats, if the robust option is selected.
        borderwidth : int, optional
            number of pixels to use when defining the outer border region, if
            the border option is selected. Default is to use the outermost 32
            pixels around all sides of the image.
        
        Returns
        -------
        None.
        
        """

        # Set output directory.
        output_dir = os.path.join(self.database.output_dir, subdir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        log.info(f'Median subtraction using method={method}')
        # Loop through concatenations.
        for i, key in enumerate(self.database.obs.keys()):
            log.info('--> Concatenation ' + key)

            # Loop through FITS files.
            nfitsfiles = len(self.database.obs[key])
            for j in range(nfitsfiles):

                # Read FITS file and PSF mask.
                fitsfile = self.database.obs[key]['FITSFILE'][j]
                data, erro, pxdq, head_pri, head_sci, is2d, imshifts, maskoffs = ut.read_obs(fitsfile)
                maskfile = self.database.obs[key]['MASKFILE'][j]
                mask = ut.read_msk(maskfile)

                # Skip file types that are not in the list of types.
                if self.database.obs[key]['TYPE'][j] in types:

                    # Subtract median.
                    head, tail = os.path.split(fitsfile)
                    log.info('  --> Median subtraction: ' + tail)
                    data_temp = data.copy()
                    # if self.database.obs[key]['TELESCOP'][j] == 'JWST' and self.database.obs[key]['INSTRUME'][j] == 'NIRCAM':
                    # data_temp[pxdq != 0] = np.nan
                    data_temp[pxdq & 1 == 1] = np.nan
                    # else:
                    #     data_temp[pxdq & 1 == 1] = np.nan
                    if method=='robust':
                        # Robust median, using a method by Jens
                        bg_med = np.nanmedian(data_temp, axis=(1, 2), keepdims=True)
                        bg_std = robust.medabsdev(data_temp, axis=(1, 2), keepdims=True)
                        bg_ind = data_temp > (bg_med + 5. * bg_std)  # clip bright PSFs for final calculation
                        data_temp[bg_ind] = np.nan
                        bg_median = np.nanmedian(data_temp, axis=(1, 2), keepdims=True)
                    elif method == 'sigma_clipped':
                        # Robust median using astropy.stats.sigma_clipped_stats
                        if len(data.shape) == 2:
                            mean, median, stddev = astropy.stats.sigma_clipped_stats(data_temp,sigma=sigma)
                        elif len(data.shape) == 3:
                            bg_median = np.zeros([data.shape[0], 1, 1])
                            for iint in range(data.shape[0]):
                                mean_i, median_i, stddev_i = astropy.stats.sigma_clipped_stats(data[iint])
                                bg_median[iint] = median_i
                        else:
                            raise NotImplementedError("data must be 2d or 3d for this method")
                    elif method=='border':
                        # Use only the outer border region of the image, near the edges of the FOV
                        shape = data.shape
                        if len(shape) == 2:
                            # only one int
                            y, x = np.indices(shape)
                            bordermask = (x < borderwidth) | (x > shape[1] - borderwidth) | (y < borderwidth) | ( y > shape[0] - borderwidth)
                            mean, bg_median, stddev = astropy.stats.sigma_clipped_stats(data[bordermask])
                        elif len(shape) == 3:
                            # perform robust stats on border region of each int
                            y, x = np.indices(data.shape[1:])
                            bordermask = (x < borderwidth) | (x > shape[1] - borderwidth) | (y < borderwidth) | ( y > shape[0] - borderwidth)
                            bg_median = np.zeros([shape[0],1,1])
                            for iint in range(shape[0]):
                                mean_i, median_i, stddev_i = astropy.stats.sigma_clipped_stats(data[iint][bordermask])
                                bg_median[iint] = median_i
                        else:
                            raise NotImplementedError("data must be 2d or 3d for this method")
                    else:
                        # Plain vanilla median of the image
                        bg_median = np.nanmedian(data_temp, axis=(1, 2), keepdims=True)

                    data -= bg_median
                    log.info('  --> Median subtraction: mean of frame median = %.2f' % np.mean(bg_median))

                # Write FITS file and PSF mask.
                fitsfile = ut.write_obs(fitsfile, output_dir, data, erro, pxdq, head_pri, head_sci, is2d, imshifts, maskoffs)
                maskfile = ut.write_msk(maskfile, mask, fitsfile)

                # Update spaceKLIP database.
                self.database.update_obs(key, j, fitsfile, maskfile)
                
    
    def subtract_background_godoy(self,
                                  types=['SCI', 'REF'],
                                  subdir='bgsub'):

        """
        Subtract the corresponding background observations from the SCI and REF
        data in the spaceKLIP database using a method developed by Nico Godoy. 
        
        Parameters
        ----------
        types : list of str
            File types to run the subtraction over.

        subdir : str, optional
            Name of the directory where the data products shall be saved. The
            default is 'bgsub'.
        
        Returns
        -------
        None.
        
        """

        # Set output directory.
        output_dir = os.path.join(self.database.output_dir, subdir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Loop through concatenations.
        for i, key in enumerate(self.database.obs.keys()):

            # Load in bunch of stuff
            # Find science, reference, and background files.
            ww_sci = np.where(self.database.obs[key]['TYPE'] == 'SCI')[0]
            ww_ref = np.where(self.database.obs[key]['TYPE'] == 'REF')[0]
            ww_sci_bg = np.where(self.database.obs[key]['TYPE'] == 'SCI_BG')[0]
            ww_ref_bg = np.where(self.database.obs[key]['TYPE'] == 'REF_BG')[0]

            # Loop over science and reference files
            for typ in types:
                if typ == 'SCI':
                    ww, ww_bg = ww_sci, ww_sci_bg
                elif typ == 'REF':
                    ww, ww_bg = ww_ref, ww_ref_bg

                # Gather background files.
                if len(ww_bg) == 0:
                    raise UserWarning('Could not find any background files.')
                else:
                    bg_data, bg_erro, bg_pxdq  = [], [], []
                    for j in ww_bg:
                        # Read  background file.
                        fitsfile = self.database.obs[key]['FITSFILE'][j]
                        data, erro, pxdq, head_pri, head_sci, is2d, imshifts, maskoffs = ut.read_obs(fitsfile)

                        # Compute median science background.
                        bg_data += [data]
                        bg_erro += [erro]
                        bg_pxdq += [pxdq]
                    bg_data, bg_erro, bg_pxdq = np.array(bg_data), np.array(bg_erro), np.array(bg_pxdq)

                    # If multiple files, take the median. Otherwise, carry on. 
                    if bg_data.ndim == 4:
                        bg_data = np.nanmedian(bg_data, axis=0)

                # Loop over individual files
                for j in ww:
                    # Read FITS file.
                    fitsfile = self.database.obs[key]['FITSFILE'][j]
                    data, erro, pxdq, head_pri, head_sci, is2d, imshifts, maskoffs = ut.read_obs(fitsfile)

                    # Subtract the background per frame
                    head, tail = os.path.split(fitsfile)
                    log.info('  --> Background subtraction: ' + tail)

                    data -= bg_data

                    # Loop over integrations
                    data_bg_sub = np.empty_like(data)
                    for k in range(data.shape[0]):
                        # Subtract median of corresponding background frame from the frame
                        bg_submed = bg_data[k,:,:] - np.nanmedian(bg_data[k,:,:])
                        # Do the same for the data (that's already background subtracted)
                        data_submed = data[k,:,:] - np.nanmedian(data[k,:,:])

                        # Specify sections for initial guess
                        # sect1 = data_submed[108:118,12:62]/bg_submed[108:118,12:62]
                        # sect2 = data_submed[93:106,152:207]/bg_submed[93:106,152:207]
                        sect1 = data_submed[112:118,4:10]/bg_submed[112:118,4:10]
                        sect2 = data_submed[95:101,207:212]/bg_submed[95:101,207:212]

                        # Reshape into 1d arrays and concatenate
                        s1 = sect1.reshape(1,sect1.shape[0]*sect1.shape[1])
                        s2 = sect2.reshape(1,sect2.shape[0]*sect2.shape[1])
                        s12 = np.concatenate((s1[0,:],s2[0,:])) 

                        # Take median of concatenated array
                        cte = np.nanmedian(s12)

                        # Use filter to determine mask for estimating BG scaling
                        # at the moment only have it working for F1140C. 
                        filt = self.database.obs[key]['FILTER'][j]
                        if filt not in ['F1065C', 'F1140C', 'F1550C']:
                            raise NotImplementedError('Godoy subtraction is only supported for MIRI FQPMs at this time!')
                        else:
                            bgmaskbase = os.path.split(os.path.abspath(__file__))[0]
                            bgmaskfile = os.path.join(bgmaskbase, 'resources/miri_bg_masks/godoy_mask_{}.fits'.format(filt.lower()))

                        # Run minimisation function, 'res' will tell us if there is any residual 
                        # background that wasn't removed in the initial attempt. I.e. do we
                        # need to subtract a little bit more or less? 
                        res = minimize(ut.bg_minimize, 
                                       x0=cte*100,
                                       args=(data_submed, bg_submed, bgmaskfile), 
                                       method='L-BFGS-B', 
                                       tol=1e-7)

                        # Extract scale factor for the background from res
                        scale = res.x/100

                        # Scale the background, and now subtract this correction from the original
                        # background subtracted data
                        data_improved_bgsub = data_submed - bg_submed*scale

                        # Subtract median of residual frame to remove any residual median offset
                        data_bg_sub[k] = data_improved_bgsub - np.nanmedian(data_improved_bgsub)

                    # Write FITS file and PSF mask.
                    fitsfile = ut.write_obs(fitsfile, output_dir, data_bg_sub, erro, pxdq, head_pri, head_sci, is2d, imshifts, maskoffs)
                    
                    # Update spaceKLIP database.
                    self.database.update_obs(key, j, fitsfile)
        
        pass
      

    def subtract_background(self,
                            nints_per_med=None,
                            subdir='bgsub'):
        """
        Median subtract the corresponding background observations from the SCI and REF
        data in the spaceKLIP database.

        Parameters
        ----------
        nints_per_med : int
            Number of integrations per median. For example, if you have a target
            + background dataset with 20 integrations each and nints_per_med is
            set to 5, a median of every 5 background images will be subtracted from
            the corresponding 5 target images. The default is None (i.e. a median
            across all images).

        subdir : str, optional
            Name of the directory where the data products shall be saved. The
            default is 'bgsub'.

        Returns
        -------
        None.

        """

        # Set output directory.
        output_dir = os.path.join(self.database.output_dir, subdir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Store the nints_per_med parameter
        orig_nints_per_med = deepcopy(nints_per_med)

        # Loop through concatenations.
        for i, key in enumerate(self.database.obs.keys()):
            log.info('--> Concatenation ' + key)

            # Find science, reference, and background files.
            ww = np.where((self.database.obs[key]['TYPE'] == 'SCI')
                            | (self.database.obs[key]['TYPE'] == 'REF'))[0]
            ww_sci_bg = np.where(self.database.obs[key]['TYPE'] == 'SCI_BG')[0]
            ww_ref_bg = np.where(self.database.obs[key]['TYPE'] == 'REF_BG')[0]

            # Loop through science background files.
            if len(ww_sci_bg) != 0:
                sci_bg_data = []
                sci_bg_erro = []
                sci_bg_pxdq = []
                for j in ww_sci_bg:

                    # Read science background file.
                    fitsfile = self.database.obs[key]['FITSFILE'][j]
                    data, erro, pxdq, head_pri, head_sci, is2d, imshifts, maskoffs = ut.read_obs(fitsfile)

                    # Determine split indices
                    nints = data.shape[0]
                    if orig_nints_per_med == None:
                        nints_per_med = nints
                    indxs = np.arange(nints)
                    split_inds = [x+1 for x in indxs if (x+1)%nints_per_med == 0
                                              and x < (nints-nints_per_med)]

                    # Compute median science background.
                    sci_bg_data += [data]
                    sci_bg_erro += [erro]
                    sci_bg_pxdq += [pxdq]
                sci_bg_data = np.concatenate(sci_bg_data)
                sci_bg_erro = np.concatenate(sci_bg_erro)
                sci_bg_pxdq = np.concatenate(sci_bg_pxdq)

                sci_bg_data_split = np.array_split(sci_bg_data, split_inds, axis=0)
                sci_bg_erro_split = np.array_split(sci_bg_erro, split_inds, axis=0)
                sci_bg_pxdq_split = np.array_split(sci_bg_pxdq, split_inds, axis=0)
                for k in range(len(split_inds)+1):
                    sci_bg_data_split[k] = np.nanmedian(sci_bg_data_split[k], axis=0)
                    nsample = np.sum(np.logical_not(np.isnan(sci_bg_erro_split[k])), axis=0)
                    sci_bg_erro_split[k] = np.true_divide(np.sqrt(np.nansum(sci_bg_erro_split[k]**2, axis=0)), nsample)
                    sci_bg_pxdq_split[k] = np.sum(sci_bg_pxdq_split[k] & 1 == 1, axis=0) != 0
            else:
                sci_bg_data = None

            # Loop through reference background files.
            if len(ww_ref_bg) != 0:
                ref_bg_data = []
                ref_bg_erro = []
                ref_bg_pxdq = []
                for j in ww_ref_bg:

                    # Read reference background file.
                    fitsfile = self.database.obs[key]['FITSFILE'][j]
                    data, erro, pxdq, head_pri, head_sci, is2d, imshifts, maskoffs = ut.read_obs(fitsfile)

                    # Determine split indices
                    nints = data.shape[0]
                    if orig_nints_per_med == None:
                        nints_per_med = nints
                    indxs = np.arange(nints)
                    split_inds = [x+1 for x in indxs if (x+1)%nints_per_med == 0
                                                  and x < (nints-nints_per_med)]
                    # Compute median reference background.
                    ref_bg_data += [data]
                    ref_bg_erro += [erro]
                    ref_bg_pxdq += [pxdq]
                ref_bg_data = np.concatenate(ref_bg_data)
                ref_bg_erro = np.concatenate(ref_bg_erro)
                ref_bg_pxdq = np.concatenate(ref_bg_pxdq)

                ref_bg_data_split = np.array_split(ref_bg_data, split_inds, axis=0)
                ref_bg_erro_split = np.array_split(ref_bg_erro, split_inds, axis=0)
                ref_bg_pxdq_split = np.array_split(ref_bg_pxdq, split_inds, axis=0)
                for k in range(len(split_inds)+1):
                    ref_bg_data_split[k] = np.nanmedian(ref_bg_data_split[k], axis=0)
                    nsample = np.sum(np.logical_not(np.isnan(ref_bg_erro_split[k])), axis=0)
                    ref_bg_erro_split[k] = np.true_divide(np.sqrt(np.nansum(ref_bg_erro_split[k]**2, axis=0)), nsample)
                    ref_bg_pxdq_split[k] = np.sum(ref_bg_pxdq_split[k] & 1 == 1, axis=0) != 0
            else:
                ref_bg_data = None

            # Check input.
            if sci_bg_data is None and ref_bg_data is None:
                raise UserWarning('Could not find any background files')

            # Loop through science and reference files.
            for j in ww:
                # Read FITS file and PSF mask.
                fitsfile = self.database.obs[key]['FITSFILE'][j]
                data, erro, pxdq, head_pri, head_sci, is2d, imshifts, maskoffs = ut.read_obs(fitsfile)
                maskfile = self.database.obs[key]['MASKFILE'][j]
                mask = ut.read_msk(maskfile)

                wwtype = self.database.obs[key]['TYPE'][j]
                if wwtype == 'SCI':
                    sci = True
                else:
                    sci = False

                # Determine split indices
                nints = data.shape[0]
                if orig_nints_per_med == None:
                        nints_per_med = nints
                indxs = np.arange(nints)
                split_inds = [x+1 for x in indxs if (x+1)%nints_per_med == 0
                                          and x < (nints-nints_per_med)]

                # Subtract background.
                head, tail = os.path.split(fitsfile)
                log.info('  --> Background subtraction: ' + tail)

                data_split = np.array_split(data, split_inds, axis=0)
                erro_split = np.array_split(erro, split_inds, axis=0)
                pxdq_split = np.array_split(pxdq, split_inds, axis=0)
                # For each dataset, need to decide what to use as the background and subtract
                for k in range(len(split_inds)+1):
                    if (sci and sci_bg_data is not None) or (not sci and ref_bg_data is None):
                        if not sci and ref_bg_data is None:
                            log.warning('  --> Could not find reference background, attempting to use science background')
                        data_split[k] = data_split[k] - sci_bg_data_split[k]
                        erro_split[k] = np.sqrt(erro_split[k]**2 + sci_bg_erro_split[k]**2)
                        pxdq_split[k][np.logical_not(pxdq_split[k] & 1 == 1) & (sci_bg_pxdq_split[k] != 0)] += 1
                    elif (not sci and ref_bg_data is not None) or (sci and sci_bg_data is None):
                        if sci and sci_bg_data is None:
                            log.warning('  --> Could not find science background, attempting to use reference background')
                        data_split[k] = data_split[k] - ref_bg_data_split[k]
                        erro_split[k] = np.sqrt(erro_split[k]**2 + ref_bg_erro_split[k]**2)
                        pxdq_split[k][np.logical_not(pxdq_split[k] & 1 == 1) & (ref_bg_pxdq_split[k] != 0)] += 1
                data = np.concatenate(data_split, axis=0)
                erro = np.concatenate(erro_split, axis=0)
                pxdq = np.concatenate(pxdq_split, axis=0)

                # Write FITS file and PSF mask.
                fitsfile = ut.write_obs(fitsfile, output_dir, data, erro, pxdq, head_pri, head_sci, is2d, imshifts, maskoffs)
                maskfile = ut.write_msk(maskfile, mask, fitsfile)

                # Update spaceKLIP database.
                self.database.update_obs(key, j, fitsfile, maskfile)

        pass

    
    def find_bad_pixels(self,
                        method='dqarr',
                        set_dq_zero=True,
                        dqarr_kwargs={},
                        sigclip_kwargs={},
                        custom_kwargs={},
                        timeints_kwargs={},
                        gradient_kwargs={},
                        types=['SCI', 'SCI_TA', 'SCI_BG', 'REF', 'REF_TA', 'REF_BG'],
                        subdir='bpfound',
                        restrict_to=None):
        """
        Identify bad pixels for cleaning

        Parameters
        ----------
        method : str, optional
            Sequence of bad pixel cleaning methods to be run on the data.
            Different methods must be joined by a '+' sign without
            whitespace. Available methods are:

            - dqarr: uses DQ array to identify bad pixels

            - sigclip: use sigma clipping to identify additional bad pixels.

            - custom: use a custom bad pixel map

            The default is 'dqarr'.
        set_dq_zero : bool, optional
            Toggle to start a new empty DQ array, or built upon the existing array.

            The default is True
        dqarr_kwargs : dict, optional
            Keyword arguments for the 'dqarr' identification method. Available keywords are:

            The default is {}.
        sigclip_kwargs : dict, optional
            Keyword arguments for the 'sigclip' identification methods. Available keywords are:

            - sigma: float, optional
                Sigma clipping threshold. The default is 5.
            - shift_x : list of int, optional
                Pixels in x-direction to which each pixel shall be compared to.
                The default is [-1, 0, 1].
            - shift_y : list of int, optional
                Pixels in y-direction to which each pixel shall be compared to.
                The default is [-1, 0, 1].

            The default is {}.
        custom_kwargs : dict, optional
            Keyword arguments for the 'custom' method. The dictionary keys must
            match the keys of the observations database and the dictionary
            content must be binary bad pixel maps (1 = bad, 0 = good) with the
            same shape as the corresponding data. The default is {}.

            The default is {}.
        types : list of str, optional
            List of data types for which bad pixels shall be identified.
            The default is ['SCI', 'SCI_TA', 'SCI_BG', 'REF', 'REF_TA', 'REF_BG'].
        subdir : str, optional
            Name of the directory where the data products shall be saved. The
            default is 'bpfound'.

        Returns
        -------
        None

        """
        # Set output directory.
        output_dir = os.path.join(self.database.output_dir, subdir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Loop through concatenations.
        for i, key in enumerate(self.database.obs.keys()):
            # if we limit to only processing some concatenations,
            # check whether this concatenation matches the pattern
            if (restrict_to is not None) and (restrict_to not in key):
                continue

            log.info('--> Concatenation ' + key)

            # Loop through FITS files.
            nfitsfiles = len(self.database.obs[key])
            for j in range(nfitsfiles):

                # Read FITS file and PSF mask.
                fitsfile = self.database.obs[key]['FITSFILE'][j]
                data, erro, pxdq, head_pri, head_sci, is2d, imshifts, maskoffs = ut.read_obs(fitsfile)
                maskfile = self.database.obs[key]['MASKFILE'][j]
                mask = ut.read_msk(maskfile)

                if set_dq_zero: #set_dq_zero
                    # Make copy of DQ array filled with zeros, i.e. all good pixels
                    pxdq_temp = np.zeros_like(pxdq)
                else:
                    # Make copy of DQ array
                    pxdq_temp = pxdq.copy()

                # Skip file types that are not in the list of types.
                if self.database.obs[key]['TYPE'][j] in types:
                    # Call bad pixel identification routines.
                    method_split = method.split('+')
                    if method_split[0] != 'dqarr' and not set_dq_zero:
                        # If the first methond is not dqarr and you are not using a boolean mask for pxdq_temp,
                        # convert pxdq_temp to a boolean mask or some of the next steps won't work.
                        # This is just a place holder. We need to think about how this mask will look like
                        pxdq_temp = (pxdq_temp > 0)

                    for k in range(len(method_split)):
                        head, tail = os.path.split(fitsfile)
                        if method_split[k] == 'dqarr':
                            log.info('  --> Method ' + method_split[k] + ': ' + tail)
                            # Flag any pixels marked as DO_NOT_USE that aren't NONSCIENCE
                            pxdq_temp = (np.isnan(data) | (pxdq_temp & 1 == 1)) \
                                         & np.logical_not(pxdq_temp & 512 == 512)
                        elif method_split[k] == 'sigclip':
                            log.info('  --> Method ' + method_split[k] + ': ' + tail)
                            self.find_bad_pixels_sigclip(data, erro, pxdq_temp, pxdq & 512 == 512, sigclip_kwargs)
                        elif method_split[k] == 'custom':
                            log.info('  --> Method ' + method_split[k] + ': ' + tail)
                            if self.database.obs[key]['TYPE'][j] not in ['SCI_TA', 'REF_TA']:
                                self.find_bad_pixels_custom(data, erro, pxdq_temp, key, custom_kwargs)
                            else:
                                log.info('  --> Method ' + method_split[k] + ': skipped because TA file')
                        elif method_split[k] == 'timeints':
                            self.find_bad_pixels_timeints(data, erro, pxdq_temp, key, timeints_kwargs)
                        elif method_split[k] == 'gradient':
                            self.find_bad_pixels_gradient(data, erro, pxdq_temp, key, gradient_kwargs)
                        else:
                            log.info('  --> Unknown method ' + method_split[k] + ': skipped')

                if set_dq_zero:
                    # The new DQ will just be the pxdq_temp we've been modifying
                    new_dq = pxdq_temp.astype(np.uint32)
                else:
                    # The new DQ will be the original pxdq with added the flagged pixels from the pxdq_temp we've been modifying as do_not_use
                    new_dq = np.bitwise_or(pxdq.copy(), pxdq_temp).astype(np.uint32)

                # Write FITS file and PSF mask.
                fitsfile = ut.write_obs(fitsfile, output_dir, data, erro, new_dq, head_pri, head_sci, is2d, imshifts, maskoffs)
                maskfile = ut.write_msk(maskfile, mask, fitsfile)
                
                # Update spaceKLIP database.
                self.database.update_obs(key, j, fitsfile, maskfile)

        pass
      

    def fix_bad_pixels(self,
                       method='timemed+localmed+medfilt',
                       sigclip_kwargs={},
                       custom_kwargs={},
                       timemed_kwargs={},
                       localmed_kwargs={},
                       medfilt_kwargs={},
                       types=['SCI', 'SCI_TA', 'SCI_BG', 'REF', 'REF_TA', 'REF_BG'],
                       subdir='bpcleaned',
                       restrict_to=None):
        """
        TO BE DEPRECATED BY FIND_BAD_PIXELS() AND CLEAN_BAD_PIXELS()
        Identify and fix bad pixels.

        Parameters
        ----------
        method : str, optional
            Sequence of bad pixel identification and cleaning methods to be run
            on the data. Different methods must be joined by a '+' sign without
            whitespace. Available methods are:

            - sigclip: use sigma clipping to identify additional bad pixels.
            - custom: use a custom bad pixel map.
            - timemed: replace pixels which are only bad in some frames with
                       their median value from the good frames.
            - localmed: replace bad pixels with the median value of their
                     surrounding good pixels.
            - medfilt: replace bad pixels with an image plane median filter.

            The default is 'timemed+localmed+medfilt'.
        sigclip_kwargs : dict, optional
            Keyword arguments for the 'sigclip' method. Available keywords are:

            - sigclip : float, optional
                Sigma clipping threshold. The default is 5.
            - shift_x : list of int, optional
                Pixels in x-direction to which each pixel shall be compared to.
                The default is [-1, 0, 1].
            - shift_y : list of int, optional
                Pixels in y-direction to which each pixel shall be compared to.
                The default is [-1, 0, 1].

            The default is {}.
        custom_kwargs : dict, optional
            Keyword arguments for the 'custom' method. The dictionary keys must
            match the keys of the observations database and the dictionary
            content must be binary bad pixel maps (1 = bad, 0 = good) with the
            same shape as the corresponding data. The default is {}.
        timemed_kwargs : dict, optional
            Keyword arguments for the 'timemed' method. Available keywords are:

            - n/a

            The default is {}.
        localmed_kwargs : dict, optional
            Keyword arguments for the 'localmed' method. Available keywords are:

            - shift_x : list of int, optional
                Pixels in x-direction from which the median shall be computed.
                The default is [-1, 0, 1].
            - shift_y : list of int, optional
                Pixels in y-direction from which the median shall be computed.
                The default is [-1, 0, 1].

            The default is {}.
        medfilt_kwargs : dict, optional
            Keyword arguments for the 'medfilt' method. Available keywords are:

            - size : int, optional
                Kernel size of the median filter to be used. The default is 4.

            The default is {}.
        types : list of str, optional
            List of data types for which bad pixels shall be identified and
            fixed. The default is ['SCI', 'SCI_TA', 'SCI_BG', 'REF', 'REF_TA',
            'REF_BG'].
        subdir : str, optional
            Name of the directory where the data products shall be saved. The
            default is 'bpcleaned'.

        Returns
        -------
        None.

        """
        # log.info('--> WARNING! The fix_bad_pixels() routine is deprecated, the ..........')
        # log.info('--> WARNING! find_bad_pixels() and clean_bad_pixels() are preferred!!!!')
    
        # Set output directory.
        output_dir = os.path.join(self.database.output_dir, subdir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Loop through concatenations.
        for i, key in enumerate(self.database.obs.keys()):
            # if we limit to only processing some concatenations, check whether this concatenation matches the pattern
            if (restrict_to is not None) and (restrict_to not in key):
                continue

            log.info('--> Concatenation ' + key)

            # Loop through FITS files.
            nfitsfiles = len(self.database.obs[key])
            for j in range(nfitsfiles):

                # Read FITS file and PSF mask.
                fitsfile = self.database.obs[key]['FITSFILE'][j]
                data, erro, pxdq, head_pri, head_sci, is2d, imshifts, maskoffs = ut.read_obs(fitsfile)
                maskfile = self.database.obs[key]['MASKFILE'][j]
                mask = ut.read_msk(maskfile)

                # Skip file types that are not in the list of types.
                if self.database.obs[key]['TYPE'][j] in types:

                    # Call bad pixel cleaning routines.
                    pxdq_temp = pxdq.copy()
                    # if self.database.obs[key]['TELESCOP'][j] == 'JWST' and self.database.obs[key]['INSTRUME'][j] == 'NIRCAM':
                    #     pxdq_temp = (pxdq_temp != 0) & np.logical_not(pxdq_temp & 512 == 512)
                    # else:
                    pxdq_temp = (np.isnan(data) | (pxdq_temp & 1 == 1)) & np.logical_not(pxdq_temp & 512 == 512)
                    method_split = method.split('+')
                    for k in range(len(method_split)):
                        head, tail = os.path.split(fitsfile)
                        if method_split[k] == 'sigclip':
                            log.info('  --> Method ' + method_split[k] + ': ' + tail)
                            self.find_bad_pixels_sigclip(data, erro, pxdq_temp, pxdq & 512 == 512, sigclip_kwargs)
                        elif method_split[k] == 'custom':
                            log.info('  --> Method ' + method_split[k] + ': ' + tail)
                            if self.database.obs[key]['TYPE'][j] not in ['SCI_TA', 'REF_TA']:
                                self.find_bad_pixels_custom(data, erro, pxdq_temp, key, custom_kwargs)
                            else:
                                log.info('  --> Method ' + method_split[k] + ': skipped because TA file')
                        elif method_split[k] == 'timemed':
                            log.info('  --> Method ' + method_split[k] + ': ' + tail)
                            self.fix_bad_pixels_timemed(data, erro, pxdq_temp, timemed_kwargs)
                        elif method_split[k] == 'localmed':
                            log.info('  --> Method ' + method_split[k] + ': ' + tail)
                            self.fix_bad_pixels_localmed(data, erro, pxdq_temp, localmed_kwargs)
                        elif method_split[k] == 'medfilt':
                            log.info('  --> Method ' + method_split[k] + ': ' + tail)
                            self.fix_bad_pixels_medfilt(data, erro, pxdq_temp, medfilt_kwargs)
                        else:
                            log.info('  --> Unknown method ' + method_split[k] + ': skipped')
                    # if self.database.obs[key]['TELESCOP'][j] == 'JWST' and self.database.obs[key]['INSTRUME'][j] == 'NIRCAM':
                    #     pxdq[(pxdq != 0) & np.logical_not(pxdq & 512 == 512) & (pxdq_temp == 0)] = 0
                    # else:
                    # pxdq[(pxdq & 1 == 1) & np.logical_not(pxdq & 512 == 512) & (pxdq_temp == 0)] = 0

                # update the pixel DQ bit flags for the output files.
                #  The pxdq variable here is effectively just the DO_NOT_USE flag, discarding other bits.
                #  We want to make a new dq which retains the other bits as much as possible.
                #  first, retain all the other bits (bits greater than 1), then add in the new/cleaned DO_NOT_USE bit
                import jwst.datamodels
                do_not_use = jwst.datamodels.dqflags.pixel['DO_NOT_USE']
                new_dq = np.bitwise_and(pxdq.copy(), np.invert(do_not_use))  # retain all other bits except the do_not_use bit
                new_dq = np.bitwise_or(new_dq, pxdq_temp)  # add in the do_not_use bit from the cleaned version
                new_dq = new_dq.astype(np.uint32)   # ensure correct output type for saving
                                                    # (the bitwise steps otherwise return np.int64 which isn't FITS compatible)

                # Write FITS file and PSF mask.
                fitsfile = ut.write_obs(fitsfile, output_dir, data, erro, new_dq, head_pri, head_sci, is2d, imshifts, maskoffs)
                maskfile = ut.write_msk(maskfile, mask, fitsfile)

                # Update spaceKLIP database.
                self.database.update_obs(key, j, fitsfile, maskfile)

        pass


    def clean_bad_pixels(self,
                       method='timemed+localmed+medfilt',
                       timemed_kwargs={},
                       localmed_kwargs={},
                       medfilt_kwargs={},
                       interp2d_kwargs={},
                       types=['SCI', 'SCI_TA', 'SCI_BG', 'REF', 'REF_TA', 'REF_BG'],
                       subdir='bpcleaned',
                       restrict_to=None):
        """
        Clean bad pixels.

        Parameters
        ----------
        method : str, optional
            Sequence of bad pixel cleaning methods to be run on the data. 
            Different methods must be joined by a '+' sign without
            whitespace. Available methods are:

            - timemed: replace pixels which are only bad in some frames with
                       their median value from the good frames.

            - localmed: replace bad pixels with the median value of their
                        surrounding good pixels.

            - medfilt: replace bad pixels with an image plane median filter.

            - interp2d: replace bad pixels with an interpolation of neighbouring pixels.

            The default is 'timemed+localmed+medfilt'.
        timemed_kwargs : dict, optional
            Keyword arguments for the 'timemed' method. Available keywords are:

            - n/a

            The default is {}.
        localmed_kwargs: dict, optional
            Keyword arguments for the 'localmed' method. Available keywords are:

            - shift_x : list of int, optional
                Pixels in x-direction from which the median shall be computed.
                The default is [-1, 0, 1].
            - shift_y : list of int, optional
                Pixels in y-direction from which the median shall be computed.
                The default is [-1, 0, 1].

            The default is {}.
        medfilt_kwargs : dict, optional
            Keyword arguments for the 'medfilt' method. Available keywords are:

            - size : int, optional
                Kernel size of the median filter to be used. The default is 4.

            The default is {}.
        interp2d_kwargs: dict, optional
            Keyword arguments for the 'interp2d' method. Available keywords are:

            - size : int, optional
                Kernel size of the median filter to be used. The default is 4.

            The default is {}.

        types : list of str, optional
            List of data types for which bad pixels shall be identified and
            fixed. The default is ['SCI', 'SCI_TA', 'SCI_BG', 'REF', 'REF_TA',
            'REF_BG'].
        subdir : str, optional
            Name of the directory where the data products shall be saved. The
            default is 'bpcleaned'.
        
        Returns
        -------
        None.

        """
        
        # Set output directory.
        output_dir = os.path.join(self.database.output_dir, subdir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Loop through concatenations.
        for i, key in enumerate(self.database.obs.keys()):
            # if we limit to only processing some concatenations, check whether this concatenation matches the pattern
            if (restrict_to is not None) and (restrict_to not in key):
                continue

            log.info('--> Concatenation ' + key)
            
            # Loop through FITS files.
            nfitsfiles = len(self.database.obs[key])
            for j in range(nfitsfiles):
                
                # Read FITS file and PSF mask.
                fitsfile = self.database.obs[key]['FITSFILE'][j]
                data, erro, pxdq, head_pri, head_sci, is2d, imshifts, maskoffs = ut.read_obs(fitsfile)
                maskfile = self.database.obs[key]['MASKFILE'][j]
                mask = ut.read_msk(maskfile)

                fig = plt.figure()
                ax = plt.gca()
                ax.hist(data.flatten(), 
                        bins=int(np.sqrt(len(data.flatten()))),
                        histtype='step',
                        label='Pre Cleaning')

                # Make copy of DQ array
                pxdq_temp = pxdq.copy()

                # Don't want to clean anything that isn't bad or is a non-science pixel
                pxdq_temp = (np.isnan(data) | (pxdq_temp & 1 == 1)) & np.logical_not(pxdq_temp & 512 == 512)
                
                # Skip file types that are not in the list of types.
                if self.database.obs[key]['TYPE'][j] in types:
                    method_split = method.split('+')

                    spatial = ['localmed', 'medfilt', 'interp2d']
                    # If localmed and medfilt in cleaning, can't run both
                    if len(set(method_split) & set(spatial)) > 1:
                        log.info('  --> WARNING: Multiple spatial cleaning routines detected!')
                        log.info('  --> The localmed/medfilt/interp2d methods clean data in a similar manner!')
                        log.info('  --> medfilt and interp2d are redundant')
                        log.info('      --> only the first method listed will affect the data')
                        log.info('  --> localmed is partially redundant with other methods')
                        log.info('      --> if run first, large clusters of bad pixels may not be fully cleaned.')

                    # Loop over methods
                    for k in range(len(method_split)):
                        head, tail = os.path.split(fitsfile)
                        log.info('  --> Method ' + method_split[k] + ': ' + tail)
                        if method_split[k] == 'timemed':
                            self.fix_bad_pixels_timemed(data, erro, pxdq_temp, timemed_kwargs)
                        elif method_split[k] == 'localmed':
                            self.fix_bad_pixels_localmed(data, erro, pxdq_temp, localmed_kwargs)
                        elif method_split[k] == 'medfilt':
                            self.fix_bad_pixels_medfilt(data, erro, pxdq_temp, medfilt_kwargs)
                        elif method_split[k] == 'interp2d':
                            self.fix_bad_pixels_interp2d(data, erro, pxdq_temp, interp2d_kwargs)
                        else:
                            log.info('  --> Unknown method ' + method_split[k] + ': skipped')
                
                # update the pixel DQ bit flags for the output files.
                #  The pxdq variable here is effectively just the DO_NOT_USE flag, discarding other bits.
                #  We want to make a new dq which retains the other bits as much as possible.
                #  first, retain all the other bits (bits greater than 1), then add in the new/cleaned DO_NOT_USE bit

                do_not_use = jwst.datamodels.dqflags.pixel['DO_NOT_USE']
                new_dq = np.bitwise_and(pxdq.copy(), np.invert(do_not_use))  # retain all other bits except the do_not_use bit
                new_dq = np.bitwise_or(new_dq, pxdq_temp)  # add in the do_not_use bit from the cleaned version
                new_dq = new_dq.astype(np.uint32)   # ensure correct output type for saving
                                                    # (the bitwise steps otherwise return np.int64 which isn't FITS compatible)

                # Finish figure for this file
                ax.hist(data.flatten(), 
                        bins=int(np.sqrt(len(data.flatten()))),
                        histtype='step',
                        label='Post Cleaning')
                ax.legend()
                #ax.set_xscale('log')
                ax.set_yscale('log')
                ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=12)
                ax.set_xlabel("Pixel Value", fontsize=14)
                ax.set_ylabel("Frequency", fontsize=12)
                ax.set_title(f"{os.path.basename(fitsfile)} \n Original vs. Cleaned Data", fontsize=16)
                output_file = os.path.join(output_dir, tail.replace('.fits','_hist.png'))
                plt.savefig(output_file)

                # Write FITS file and PSF mask.
                fitsfile = ut.write_obs(fitsfile, output_dir, data, erro, new_dq, head_pri, head_sci, is2d, imshifts, maskoffs)
                maskfile = ut.write_msk(maskfile, mask, fitsfile)
                
                # Update spaceKLIP database.
                self.database.update_obs(key, j, fitsfile, maskfile)
        
        pass
    
    def find_bad_pixels_sigclip(self,
                                data,
                                erro,
                                pxdq,
                                NON_SCIENCE,
                                sigclip_kwargs={}):
        """
        Use an iterative sigma clipping algorithm to identify additional bad
        pixels in the data.

        Parameters
        ----------
        data : 3D-array
            Input images.
        erro : 3D-array
            Input image uncertainties.
        pxdq : 3D-array
            Input binary bad pixel maps (1 = bad, 0 = good). Will be updated by
            the routine to include the newly identified bad pixels.
        NON_SCIENCE : 3D-array
            Input binary non-science pixel maps (1 = bad, 0 = good). Will not
            be modified by the routine.
        sigclip_kwargs : dict, optional
            Keyword arguments for the 'sigclip' method. Available keywords are:

            - sigma : float, optional
                Sigma clipping threshold. The default is 5.
            - neg_sigma : float, optional
                Sigma clipping threshold for negative outliers. The default is 1. 
            - shift_x : list of int, optional
                Pixels in x-direction to which each pixel shall be compared to.
                The default is [-1, 0, 1].
            - shift_y : list of int, optional
                Pixels in y-direction to which each pixel shall be compared to.
                The default is [-1, 0, 1].

            The default is {}.

        Returns
        -------
        None.

        """

        # Check input.
        if 'sigma' not in sigclip_kwargs.keys():
            sigclip_kwargs['sigma'] = 5.
        if 'neg_sigma' not in sigclip_kwargs.keys():
            sigclip_kwargs['neg_sigma'] = 1.
        if 'shift_x' not in sigclip_kwargs.keys():
            sigclip_kwargs['shift_x'] = [-1, 0, 1]
        if 'shift_y' not in sigclip_kwargs.keys():
            sigclip_kwargs['shift_y'] = [-1, 0, 1]
        if 0 not in sigclip_kwargs['shift_x']:
            sigclip_kwargs['shift_x'] += [0]
        if 0 not in sigclip_kwargs['shift_y']:
            sigclip_kwargs['shift_y'] += [0]
        
        # Pad data.
        pad_left = np.abs(np.min(sigclip_kwargs['shift_x']))
        pad_right = np.abs(np.max(sigclip_kwargs['shift_x']))
        if pad_right == 0:
            right = None
        else:
            right = -pad_right
        pad_bottom = np.abs(np.min(sigclip_kwargs['shift_y']))
        pad_top = np.abs(np.max(sigclip_kwargs['shift_y']))
        if pad_top == 0:
            top = None
        else:
            top = -pad_top
        pad_vals = ((pad_bottom, pad_top), (pad_left, pad_right))

        # Find bad pixels using median of neighbors.
        pxdq_orig = pxdq.copy()
        ww = pxdq != 0
        data_temp = data.copy()
        data_temp[ww] = np.nan
        erro_temp = erro.copy()
        erro_temp[ww] = np.nan
        for i in range(ww.shape[0]):

            # Get median background and standard deviation.
            bg_med = np.nanmedian(data_temp[i])
            bg_std = robust.medabsdev(data_temp[i])
            bg_ind = data[i] < (bg_med + 10. * bg_std)  # clip bright PSFs for final calculation
            bg_med = np.nanmedian(data_temp[i][bg_ind])
            bg_std = robust.medabsdev(data_temp[i][bg_ind])

            # Create initial mask of large negative values.
            ww[i] = ww[i] | (data[i] < bg_med - sigclip_kwargs['neg_sigma'] * bg_std)
            ww[i][NON_SCIENCE[i]] = 0
            
            # Loop through max 10 iterations.
            for it in range(10):
                data_temp[i][ww[i]] = np.nan
                erro_temp[i][ww[i]] = np.nan
                
                # Shift data and calculate median and standard deviation of neighbours
                pad_data = np.pad(data_temp[i], pad_vals, mode='edge')
                pad_erro = np.pad(erro_temp[i], pad_vals, mode='edge')
                data_arr = []
                erro_arr = []
                for ix in sigclip_kwargs['shift_x']:
                    for iy in sigclip_kwargs['shift_y']:
                        if ix != 0 or iy != 0:
                            data_arr += [np.roll(pad_data, (iy, ix), axis=(0, 1))]
                            erro_arr += [np.roll(pad_erro, (iy, ix), axis=(0, 1))]
                data_arr = np.array(data_arr)
                data_arr_trim = data_arr[:, pad_bottom:top, pad_left:right]
                data_med = np.nanmedian(data_arr_trim, axis=0)
                diff = data[i] - data_med

                data_std = np.nanstd(data_arr_trim, axis=0)

                # # Do the same for the diff array we just made
                # pad_diff = np.pad(diff, pad_vals, mode='edge')
                # diff_arr = []
                # for ix in sigclip_kwargs['shift_x']:
                #     for iy in sigclip_kwargs['shift_y']:
                #         if ix != 0 or iy != 0:
                #             diff_arr += [np.roll(pad_diff, (iy, ix), axis=(0, 1))]
                # diff_arr = np.array(diff_arr)
                # diff_arr = diff_arr[:, pad_bottom:top, pad_left:right]
                # diff_med = np.nanmedian(diff_arr, axis=0)
                # doublediff = data[i] - data_med - diff_med
                # diff_std = np.nanstd(diff_arr, axis=0)

                # Find values N standard deviations above the mean of neighbors
                threshold = sigclip_kwargs['sigma'] * data_std
                mask_new = diff > threshold

                data_temp[i][mask_new] = np.nan

                # fig, ax = plt.subplots(1, 2)
                # ax[0].imshow(data_temp[i])
                # ax[1].imshow(data_std)
                # plt.show()

                nmask_new = np.sum(mask_new & np.logical_not(ww[i]))
                # print('Iteration %.0f: %.0f bad pixels identified, %.0f are new' % (it + 1, np.sum(mask_new), nmask_new))
                sys.stdout.write('\rFrame %.0f/%.0f, iteration %.0f' % (i + 1, ww.shape[0], it + 1))
                sys.stdout.flush()
                if it > 0 and nmask_new == 0:
                    break
                ww[i] = ww[i] | mask_new
            ww[i][NON_SCIENCE[i]] = 0
            pxdq[i][ww[i]] = 1
        print('')
        log.info('  --> Method sigclip: identified %.0f additional bad pixel(s) -- %.2f%%' % (np.sum(pxdq) - np.sum(pxdq_orig), 100. * (np.sum(pxdq) - np.sum(pxdq_orig)) / np.prod(pxdq.shape)))
        
        pass

    def find_bad_pixels_timeints(self,
                            data,
                            erro,
                            pxdq,
                            NON_SCIENCE,
                            timeints_kwargs={}):
        """
        Identify bad pixels from temporal variations across integrations.
        
        Parameters
        ----------
        data : 3D-array
            Input images.
        erro : 3D-array
            Input image uncertainties.
        pxdq : 3D-array
            Input binary bad pixel maps (1 = bad, 0 = good). Will be updated by
            the routine to include the newly identified bad pixels.
        NON_SCIENCE : 3D-array
            Input binary non-science pixel maps (1 = bad, 0 = good). Will not
            be modified by the routine.
        timeints_kwargs : dict, optional
            Keyword arguments for the 'timeints' method. Available keywords are:
            
            - sigma : float, optional
                Sigma clipping threshold. The default is 5.

            The default is {}.
        
        Returns
        -------
        None.

        """

        # Check input.
        if 'sigma' not in timeints_kwargs.keys():
            timeints_kwargs['sigma'] = 10.

        pxdq_orig = pxdq.copy()
        ww = pxdq != 0
        data_temp = data.copy()
        data_temp[ww] = np.nan
        
        # Find bad pixels across the cube

        med_ints = np.nanmedian(data_temp, axis=0)
        std_ints = np.nanstd(data_temp, axis=0)

        std2_ints = robust.medabsdev(data_temp, axis=0)

        diff = np.abs((data_temp - med_ints)) / std2_ints

        mask_new = diff > timeints_kwargs['sigma']


        # data_temp[mask_new] = 9999
        # plt.imshow(data_temp[1])
        # plt.show()
        # plt.hist(diff.flatten(), 
        #         bins=int(np.sqrt(len(diff.flatten()))),
        #         histtype='step',
        #         label='Pre Cleaning')
        # plt.yscale('log')
        # plt.show()

        
        ww = ww | mask_new
        pxdq[ww] = 1
        print('')
        log.info('  --> Method timeints: identified %.0f additional bad pixel(s) -- %.2f%%' % (np.sum(pxdq) - np.sum(pxdq_orig), 100. * (np.sum(pxdq) - np.sum(pxdq_orig)) / np.prod(pxdq.shape)))

        pass

    def find_bad_pixels_gradient(self,
                                 data,
                                 erro,
                                 pxdq,
                                 key,
                                 gradient_kwargs={}):
        print('')
        log.info('  --> Warning!: This routine has not been thoroughly tested and requires further development')
        # Check input.
        if 'sigma' not in gradient_kwargs.keys():
            gradient_kwargs['sigma'] = 0.5
        if 'threshold' not in gradient_kwargs.keys():
            gradient_kwargs['threshold'] = 0.05
        if 'negative' not in gradient_kwargs.keys():
            gradient_kwargs['negative'] = True

        sig = gradient_kwargs['sigma']
        threshold = gradient_kwargs['threshold']
        negative = gradient_kwargs['negative'] 

        pxdq_orig = pxdq.copy()
        ww = pxdq != 0
        data_temp = data.copy()
        data_temp[ww] = np.nan

        # Loop over the images
        for i in range(ww.shape[0]):
            image = data_temp[i]

            ### remove nans
            x=np.arange(0, image.shape[1])
            y=np.arange(0, image.shape[0])

            xx, yy = np.meshgrid(x, y)

            # mask nans
            image = np.ma.masked_invalid(image)

            xvalid=xx[~image.mask]
            yvalid=yy[~image.mask]

            newimage=image[~image.mask]

            image_no_nans = griddata((xvalid, yvalid), 
                                    newimage.ravel(),
                                    (xx, yy),
                                    method='linear')
            
            ### get smooth image
            smimage=gaussian_filter(image_no_nans, sigma=sig)

            ### get sharp image
            shimage=image_no_nans-smimage
            
            ### get gradients
            image_to_gradient=shimage/smimage

            gr=np.gradient((image_to_gradient))
            gr_dx=gr[1]
            gr_dy=gr[0]

            ### pad gradient adding 1 extra pixel at beginning and end
            gr_dxp=np.pad(gr_dx,(1,1))
            gr_dyp=np.pad(gr_dy,(1,1))

            ### identify bad pixels
            # positive
            bad_pixels = (gr_dxp[1:-1,2:]<-threshold) & (gr_dxp[1:-1,:-2]>threshold) & (gr_dyp[2:,1:-1]<-threshold) & (gr_dyp[:-2,1:-1]>threshold)
            # negative
            if negative:
                bad_pixels_n = (gr_dxp[1:-1,2:]>threshold) & (gr_dxp[1:-1,:-2]<-threshold) & (gr_dyp[2:,1:-1]>threshold) & (gr_dyp[:-2,1:-1]<threshold)
                bad_pixels=bad_pixels | bad_pixels_n

            image[bad_pixels] = np.nan

            # Flag DQ array
            ww[i] = ww[i] | bad_pixels
            pxdq[i][ww[i]] = 1
        print('')
        log.info('  --> Method gradient: identified %.0f additional bad pixel(s) -- %.2f%%' % (np.sum(pxdq) - np.sum(pxdq_orig), 100. * (np.sum(pxdq) - np.sum(pxdq_orig)) / np.prod(pxdq.shape)))
        pass


    def find_bad_pixels_custom(self,
                               data,
                               erro,
                               pxdq,
                               key,
                               custom_kwargs={}):
        """
        Use a custom bad pixel map to flag additional bad pixels in the data.

        Parameters
        ----------
        data : 3D-array
            Input images.
        erro : 3D-array
            Input image uncertainties.
        pxdq : 3D-array
            Input binary bad pixel maps (1 = bad, 0 = good). Will be updated by
            the routine to include the newly flagged bad pixels.
        key : str
            Database key of the observation to be updated.
        custom_kwargs : dict, optional
            Keyword arguments for the 'custom' method. The dictionary keys must
            match the keys of the observations database and the dictionary
            content must be binary bad pixel maps (1 = bad, 0 = good) with the
            same shape as the corresponding data. The default is {}.

        Returns
        -------
        None.

        """

        # Find bad pixels using median of neighbors.
        pxdq_orig = pxdq.copy()
        pxdq_custom = custom_kwargs[key] != 0
        if pxdq_custom.ndim == pxdq.ndim - 1: # Enable 3D bad pixel map to flag individual frames
            pxdq_custom = np.array([pxdq_custom] * pxdq.shape[0])
        pxdq[pxdq_custom] = 1
        log.info('  --> Method custom: flagged %.0f additional bad pixel(s) -- %.2f%%' % (np.sum(pxdq) - np.sum(pxdq_orig), 100. * (np.sum(pxdq) - np.sum(pxdq_orig)) / np.prod(pxdq.shape)))

        pass

    def fix_bad_pixels_timemed(self,
                               data,
                               erro,
                               pxdq,
                               timemed_kwargs={}):
        """
        Replace pixels which are only bad in some frames with their median
        value from the good frames.

        Parameters
        ----------
        data : 3D-array
            Input images.
        erro : 3D-array
            Input image uncertainties.
        pxdq : 3D-array
            Input binary bad pixel maps (1 = bad, 0 = good). Will be updated by
            the routine to exclude the fixed bad pixels.
        timemed_kwargs : dict, optional
            Keyword arguments for the 'timemed' method. Available keywords are:
            - n/a
            The default is {}.

        Returns
        -------
        None.

        """

        # Fix bad pixels using time median.
        ww = pxdq != 0
        ww_all_bad = np.array([np.sum(ww, axis=0) == ww.shape[0]] * ww.shape[0])
        ww_not_all_bad = ww & np.logical_not(ww_all_bad)
        log.info('  --> Method timemed: fixing %.0f bad pixel(s) -- %.2f%%' % (np.sum(ww_not_all_bad), 100. * np.sum(ww_not_all_bad) / np.prod(ww_not_all_bad.shape)))
        data[ww_not_all_bad] = np.nan
        data[ww_not_all_bad] = np.array([np.nanmedian(data, axis=0)] * data.shape[0])[ww_not_all_bad]
        erro[ww_not_all_bad] = np.nan
        erro[ww_not_all_bad] = np.array([np.nanmedian(erro, axis=0)] * erro.shape[0])[ww_not_all_bad]
        pxdq[ww_not_all_bad] = 0

        pass

    
    def fix_bad_pixels_localmed(self,
                             data,
                             erro,
                             pxdq,
                             localmed_kwargs={}):
        """
        Replace bad pixels with the median value of their surrounding good
        pixels.

        Parameters
        ----------
        data : 3D-array
            Input images.
        erro : 3D-array
            Input image uncertainties.
        pxdq : 3D-array
            Input binary bad pixel maps (1 = bad, 0 = good). Will be updated by
            the routine to exclude the fixed bad pixels.
        localmed_kwargs : dict, optional
            Keyword arguments for the 'localmed' method. Available keywords are:

            - shift_x : list of int, optional
                Pixels in x-direction from which the median shall be computed.
                The default is [-1, 0, 1].
            - shift_y : list of int, optional
                Pixels in y-direction from which the median shall be computed.
                The default is [-1, 0, 1].

            The default is {}.

        Returns
        -------
        None.

        """

        # Check input.
        if 'shift_x' not in localmed_kwargs.keys():
            localmed_kwargs['shift_x'] = [-1, 0, 1]
        if 'shift_y' not in localmed_kwargs.keys():
            localmed_kwargs['shift_y'] = [-1, 0, 1]
        if 0 not in localmed_kwargs['shift_x']:
            localmed_kwargs['shift_x'] += [0]
        if 0 not in localmed_kwargs['shift_y']:
            localmed_kwargs['shift_y'] += [0]
        
        # Pad data.
        pad_left = np.abs(np.min(localmed_kwargs['shift_x']))
        pad_right = np.abs(np.max(localmed_kwargs['shift_x']))
        if pad_right == 0:
            right = None
        else:
            right = -pad_right
        pad_bottom = np.abs(np.min(localmed_kwargs['shift_y']))
        pad_top = np.abs(np.max(localmed_kwargs['shift_y']))
        if pad_top == 0:
            top = None
        else:
            top = -pad_top
        pad_vals = ((0, 0), (pad_bottom, pad_top), (pad_left, pad_right))

        # Fix bad pixels using median of neighbors.
        ww = pxdq != 0
        data_temp = data.copy()
        data_temp[ww] = np.nan
        pad_data = np.pad(data_temp, pad_vals, mode='edge')
        erro_temp = erro.copy()
        erro_temp[ww] = np.nan
        pad_erro = np.pad(erro_temp, pad_vals, mode='edge')
        for i in range(ww.shape[0]):
            data_arr = []
            erro_arr = []
            for ix in localmed_kwargs['shift_x']:
                for iy in localmed_kwargs['shift_y']:
                    if ix != 0 or iy != 0:
                        data_arr += [np.roll(pad_data[i], (iy, ix), axis=(0, 1))]
                        erro_arr += [np.roll(pad_erro[i], (iy, ix), axis=(0, 1))]
            data_arr = np.array(data_arr)
            data_arr = data_arr[:, pad_bottom:top, pad_left:right]
            data_med = np.nanmedian(data_arr, axis=0)
            ww[i][np.isnan(data_med)] = 0
            data[i][ww[i]] = data_med[ww[i]]
            erro_arr = np.array(erro_arr)
            erro_arr = erro_arr[:, pad_bottom:top, pad_left:right]
            erro_med = np.nanmedian(erro_arr, axis=0)
            erro[i][ww[i]] = erro_med[ww[i]]
            pxdq[i][ww[i]] = 0
        log.info('  --> Method localmed: fixing %.0f bad pixel(s) -- %.2f%%' % (np.sum(ww), 100. * np.sum(ww) / np.prod(ww.shape)))
        pass

    def fix_bad_pixels_medfilt(self,
                               data,
                               erro,
                               pxdq,
                               medfilt_kwargs={}):
        """
        Replace bad pixels with an image plane median filter.

        Parameters
        ----------
        data : 3D-array
            Input images.
        erro : 3D-array
            Input image uncertainties.
        pxdq : 3D-array
            Input binary bad pixel maps (1 = bad, 0 = good). Will be updated by
            the routine to exclude the fixed bad pixels.
        medfilt_kwargs : dict, optional
            Keyword arguments for the 'medfilt' method. Available keywords are:

            - size : int, optional
                Kernel size of the median filter to be used. The default is 4.

            The default is {}.

        Returns
        -------
        None.

        """

        # Check input.
        if 'size' not in medfilt_kwargs.keys():
            medfilt_kwargs['size'] = 4

        # Fix bad pixels using median filter.
        ww = pxdq != 0
        log.info('  --> Method medfilt: fixing %.0f bad pixel(s) -- %.2f%%' % (np.sum(ww), 100. * np.sum(ww) / np.prod(ww.shape)))
        data_temp = data.copy()
        data_temp[np.isnan(data_temp)] = 0.
        erro_temp = erro.copy()
        erro_temp[np.isnan(erro_temp)] = 0.
        for i in range(ww.shape[0]):
            data[i][ww[i]] = median_filter(data_temp[i], **medfilt_kwargs)[ww[i]]
            erro[i][ww[i]] = median_filter(erro_temp[i], **medfilt_kwargs)[ww[i]]
            pxdq[i][ww[i]] = 0

        pass

    def fix_bad_pixels_interp2d(self,
                                data,
                                erro,
                                pxdq,
                                interp2d_kwargs={}):
        """
        Replace bad pixels with an interpolation of neighbouring pixels.
        
        Parameters
        ----------
        data : 3D-array
            Input images.
        erro : 3D-array
            Input image uncertainties.
        pxdq : 3D-array
            Input binary bad pixel maps (1 = bad, 0 = good). Will be updated by
            the routine to exclude the fixed bad pixels.
        interp2d_kwargs : dict, optional
            Keyword arguments for the 'interp2d' method. Available keywords are:

            - size : int, optional
                Kernel size of the median filter to be used. The default is 4.

            The default is {}.

        Returns
        -------
        None.

        """
        
        # Check input.
        if 'size' not in interp2d_kwargs.keys():
            interp2d_kwargs['size'] = 5
        
        # Fix bad pixels using interpolation of neighbors.
        ww = (pxdq != 0) & np.logical_not(pxdq & 512 == 512)
        log.info('  --> Method interp2d: fixing %.0f bad pixel(s) -- %.2f%%' % (np.sum(ww), 100. * np.sum(ww) / np.prod(ww.shape)))
        
        # NaN pixels to be replaced with interpolation
        data_temp = data.copy()
        data_temp[np.where(np.isnan(data_temp))] = 0
        data_temp[ww] = np.nan
        erro_temp = erro.copy()
        erro_temp[np.where(np.isnan(erro_temp))] = 0
        erro_temp[ww] = np.nan

        rows, cols = data_temp[0].shape
        half_box = interp2d_kwargs['size'] // 2
        for i in range(ww.shape[0]):
            for ri in range(rows):
                for ci in range(cols):
                    if np.isnan(data_temp[i][ri, ci]):
                        # Calculate the indices of the NxN box centered around the NaN pixel
                        x_min = max(0, ci - half_box)
                        x_max = min(cols, ci + half_box + 1)
                        y_min = max(0, ri - half_box)
                        y_max = min(rows, ri + half_box + 1)

                        # Extract a NxN box within the valid range
                        box = data_temp[i][y_min:y_max, x_min:x_max]
                        ebox = erro_temp[i][y_min:y_max, x_min:x_max]

                        # Extract coordinates and values from the box
                        box_coords = np.array(np.where(~np.isnan(box))).T \
                                     + np.array([[x_min, y_min]])
                        box_values = box[~np.isnan(box)]

                        ebox_coords = np.array(np.where(~np.isnan(ebox))).T \
                                     + np.array([[x_min, y_min]])
                        ebox_values = ebox[~np.isnan(ebox)]

                        # Perform interpolation if there are valid values in the box
                        if len(box_values) > interp2d_kwargs['size'] \
                           and len(ebox_values) > interp2d_kwargs['size']:
                            # Extract x and y coordinates of valid values, same coords for 
                            # data and err
                            x_coords = box_coords[:, 0]
                            y_coords = box_coords[:, 1]
                            ex_coords = ebox_coords[:, 0]
                            ey_coords = ebox_coords[:, 1]

                            # Perform interpolation of data
                            data_interp = griddata((x_coords, y_coords), 
                                                    box_values, 
                                                    (ci, ri), 
                                                    method='linear',
                                                    fill_value=np.nan)

                            # Replace data pixel with interpolated value
                            data[i][ri, ci] = data_interp

                            # Perform interpolation of error
                            err_interp = griddata((ex_coords, ey_coords), 
                                                   ebox_values, 
                                                   (ci, ri), 
                                                   method='linear',
                                                   fill_value=np.nan)

                            # Replace error pixel
                            erro[i][ri, ci] = err_interp


                            pxdq[i][ww[i]] = 0
        
        pass

    def replace_nans(self,
                     cval=0.,
                     types=['SCI', 'SCI_BG', 'REF', 'REF_BG'],
                     subdir='nanreplaced'):
        """
        Replace all nans in the data with a constant value.

        Parameters
        ----------
        cval : float, optional
            Fill value for the nan pixels. The default is 0.
        types : list of str, optional
            List of data types for which nans shall be replaced. The default is
            ['SCI', 'SCI_BG', 'REF', 'REF_BG'].
        subdir : str, optional
            Name of the directory where the data products shall be saved. The
            default is 'nanreplaced'.

        Returns
        -------
        None.

        """

        # Set output directory.
        output_dir = os.path.join(self.database.output_dir, subdir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Loop through concatenations.
        for i, key in enumerate(self.database.obs.keys()):
            log.info('--> Concatenation ' + key)

            # Loop through FITS files.
            nfitsfiles = len(self.database.obs[key])
            for j in range(nfitsfiles):

                # Read FITS file and PSF mask.
                fitsfile = self.database.obs[key]['FITSFILE'][j]
                data, erro, pxdq, head_pri, head_sci, is2d, imshifts, maskoffs = ut.read_obs(fitsfile)
                maskfile = self.database.obs[key]['MASKFILE'][j]
                mask = ut.read_msk(maskfile)

                # Skip file types that are not in the list of types.
                if self.database.obs[key]['TYPE'][j] in types:

                    # Replace nans.
                    head, tail = os.path.split(fitsfile)
                    log.info('  --> Nan replacement: ' + tail)
                    ww = np.isnan(data)
                    data[ww] = cval
                    log.info('  --> Nan replacement: replaced %.0f nan pixel(s) with value ' % (np.sum(ww)) + str(cval) + ' -- %.2f%%' % (100. * np.sum(ww)/np.prod(ww.shape)))

                # Write FITS file and PSF mask.
                fitsfile = ut.write_obs(fitsfile, output_dir, data, erro, pxdq, head_pri, head_sci, is2d, imshifts, maskoffs)
                maskfile = ut.write_msk(maskfile, mask, fitsfile)

                # Update spaceKLIP database.
                self.database.update_obs(key, j, fitsfile, maskfile)

        pass

    def blur_frames(self,
                    fact='auto',
                    types=['SCI', 'SCI_BG', 'REF', 'REF_BG'],
                    subdir='blurred'):
        """
        Blur frames with a Gaussian filter.

        Parameters
        ----------
        fact : 'auto' or 'fix23' or float or dict of list of float or None, optional
            FWHM (pix) of the Gaussian filter. If 'auto', will compute the FWHM
            automatically based on the Nyquist sampling criterion for discrete
            data, which is FWHM = lambda / 2.3D, where D = 5.2 m for NIRCam
            coronagraphy and D = 6.5 m otherwise. If 'fix23', will always blur
            the data with a Gaussian kernel of FWHM = 2.3 pix, so that even bad
            pixels cause no more Fourier ripples. If dict of list of float,
            then the dictionary keys must match the keys of the observations
            database, and the number of entries in the lists must match the
            number of observations in the corresponding concatenation. Then, a
            different FWHM can be used for each observation. If None, the
            corresponding observation will be skipped. The default is 'auto'.
        types : list of str, optional
            List of data types for which the frames shall be blurred. The
            default is ['SCI', 'SCI_BG', 'REF', 'REF_BG'].
        subdir : str, optional
            Name of the directory where the data products shall be saved. The
            default is 'blurred'.

        Returns
        -------
        None.

        """

        # Set output directory.
        output_dir = os.path.join(self.database.output_dir, subdir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Loop through concatenations.
        for i, key in enumerate(self.database.obs.keys()):
            log.info('--> Concatenation ' + key)

            # Loop through FITS files.
            Nfitsfiles = len(self.database.obs[key])
            for j in range(Nfitsfiles):

                # Read FITS file.
                fitsfile = self.database.obs[key]['FITSFILE'][j]
                data, erro, pxdq, head_pri, head_sci, is2d, imshifts, maskoffs = ut.read_obs(fitsfile)
                maskfile = self.database.obs[key]['MASKFILE'][j]
                mask = ut.read_msk(maskfile)

                # Skip file types that are not in the list of types.
                fact_temp = None
                if self.database.obs[key]['TYPE'][j] in types:

                    # Blur frames.
                    head, tail = os.path.split(fitsfile)
                    log.info('  --> Frame blurring: ' + tail)
                    try:
                        fact_temp = fact[key][j]
                    except:
                        fact_temp = fact
                    if self.database.obs[key]['TELESCOP'][j] == 'JWST':
                        if self.database.obs[key]['EXP_TYPE'][j] in ['NRC_CORON']:
                            diam = 5.2
                        else:
                            diam = JWST_CIRCUMSCRIBED_DIAMETER
                    else:
                        raise UserWarning('Data originates from unknown telescope')
                    if fact_temp is not None:
                        if str(fact_temp) == 'auto':
                            wave_min = self.database.obs[key]['CWAVEL'][j] - self.database.obs[key]['DWAVEL'][j]  # micron
                            fwhm_current = wave_min * 1e-6 / diam * 180. / np.pi * 3600. / self.database.obs[key]['PIXSCALE'][j]  # pix
                            fwhm_desired = 2.3  # pix; see, e.g., Pawley 2006
                            fwhm_desired *= 1.5  # go to 1.5 times the theoretically required bluring to safely avoid numerical ringing effects
                            fact_temp = np.sqrt(fwhm_desired**2 - fwhm_current**2)
                            fact_temp /= np.sqrt(8. * np.log(2.))  # fix from Marshall
                        if str(fact_temp) == 'fix23':
                            fwhm_current = 1.  # pix
                            fwhm_desired = 2.3  # pix; see, e.g., Pawley 2006
                            fact_temp = np.sqrt(fwhm_desired**2 - fwhm_current**2)
                            fact_temp /= np.sqrt(8. * np.log(2.))  # fix from Marshall
                        if np.isnan(fact_temp):
                            fact_temp = None
                            log.info('  --> Frame blurring: skipped')
                            continue
                        log.info('  --> Frame blurring: factor = %.3f' % fact_temp)
                        for k in range(data.shape[0]):
                            data[k] = gaussian_filter(data[k], fact_temp)
                            erro[k] = gaussian_filter(erro[k], fact_temp)
                        if mask is not None:
                            mask = gaussian_filter(mask, fact_temp)
                    else:
                        log.info('  --> Frame blurring: skipped')

                # Write FITS file.
                if fact_temp is None:
                    pass
                else:
                    head_pri['BLURFWHM'] = fact_temp * np.sqrt(8. * np.log(2.)) # Factor to convert from sigma to FWHM
                fitsfile = ut.write_obs(fitsfile, output_dir, data, erro, pxdq, head_pri, head_sci, is2d, imshifts, maskoffs)
                maskfile = ut.write_msk(maskfile, mask, fitsfile)

                # Update spaceKLIP database.
                if fact_temp is None:
                    self.database.update_obs(key, j, fitsfile, maskfile, blurfwhm=np.nan)
                else:
                    self.database.update_obs(key, j, fitsfile, maskfile, blurfwhm=fact_temp * np.sqrt(8. * np.log(2.)))

        pass

    def hpf(self,
            size='auto',
            types=['SCI', 'SCI_BG', 'REF', 'REF_BG'],
            subdir='filtered'):
        """
        Blur frames with a Gaussian filter.

        Parameters
        ----------
        size : 'auto' or float or dict of list of float or None, optional
            FWHM (pix) of the Gaussian filter. If 'auto', will compute the FWHM
            automatically based on the Nyquist sampling criterion for discrete
            data, which is FWHM = lambda / 2.3D, where D = 5.2 m for NIRCam
            coronagraphy and D = 6.5 m otherwise. If dict of list of float,
            then the dictionary keys must match the keys of the observations
            database, and the number of entries in the lists must match the
            number of observations in the corresponding concatenation. Then, a
            different FWHM can be used for each observation. If None, the
            corresponding observation will be skipped. The default is 'auto'.
        types : list of str, optional
            List of data types for which the frames shall be blurred. The
            default is ['SCI', 'SCI_BG', 'REF', 'REF_BG'].
        subdir : str, optional
            Name of the directory where the data products shall be saved. The
            default is 'blurred'.

        Returns
        -------
        None.

        """

        # Set output directory.
        output_dir = os.path.join(self.database.output_dir, subdir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Loop through concatenations.
        for i, key in enumerate(self.database.obs.keys()):
            log.info('--> Concatenation ' + key)

            # Loop through FITS files.
            Nfitsfiles = len(self.database.obs[key])
            for j in range(Nfitsfiles):

                # Read FITS file.
                fitsfile = self.database.obs[key]['FITSFILE'][j]
                data, erro, pxdq, head_pri, head_sci, is2d, imshifts, maskoffs = ut.read_obs(fitsfile)
                maskfile = self.database.obs[key]['MASKFILE'][j]
                mask = ut.read_msk(maskfile)

                # Skip file types that are not in the list of types.
                size_temp = None
                if self.database.obs[key]['TYPE'][j] in types:

                    # High-pass filter frames.
                    head, tail = os.path.split(fitsfile)
                    log.info('  --> Frame filtering: ' + tail)
                    try:
                        size_temp = size[key]
                    except:
                        size_temp = float(size)
                    if size_temp is not None:
                        log.info('  --> Frame filtering: HPF FWHM = %.2f pix' % size_temp)
                        fourier_sigma_size = (data.shape[1] / size_temp) / (2. * np.sqrt(2. * np.log(2.)))
                        data = parallelized.high_pass_filter_imgs(data, numthreads=None, filtersize=fourier_sigma_size)
                        erro = parallelized.high_pass_filter_imgs(erro, numthreads=None, filtersize=fourier_sigma_size)
                    else:
                        log.info('  --> Frame filtering: skipped')

                # Write FITS file.
                if size_temp is None:
                    pass
                else:
                    head_pri['HPFSIZE'] = size_temp
                fitsfile = ut.write_obs(fitsfile, output_dir, data, erro, pxdq, head_pri, head_sci, is2d, imshifts, maskoffs)
                maskfile = ut.write_msk(maskfile, mask, fitsfile)

                # Update spaceKLIP database.
                self.database.update_obs(key, j, fitsfile, maskfile)

        pass

    def inject_companions(self,
                          companions,
                          starfile,
                          spectral_type,
                          output_dir,
                          highpass=False,
                          subdir='test',
                          date='auto',
                          kwargs={}):
        '''
        Function to inject synthetic PSFs into a set of frames loaded from a dataset, and save the new frames with the
        injected companion.

        Parameters
        ----------
        companions : list of list of three float, optional
            List of companions to be injected.
            For each companion, there should be a three element list containing
            [RA offset (arcsec), Dec offset (arcsec), contrast].
        raw_dataset : pyKLIP dataset
            A pyKLIP dataset which companions will be injected into and KLIP
            will be performed on.
        injection_psf : 2D-array
            The PSF of the companion to be injected.
        injection_seps : 1D-array
            List of separations to inject companions at (pixels).
        injection_pas : 1D-array
            List of position angles to inject companions at (degrees).
        injection_spacing : int, None
            Spacing between companions injected in a single image. If companions
            are too close then it can pollute the recovered flux. Set to 'None'
            to inject only one companion at a time (pixels).
        injection_fluxes : 1D-array
            Same size as injection_seps, units should correspond to the image
            units. This is the *peak* flux of the injection.
        true_companions : list of list of three float, optional
            List of real companions to be masked before computing the raw contrast.
            For each companion, there should be a three element list containing
            [RA offset (pixels), Dec offset (pixels), mask radius (pixels)].
            The default is None.

        Returns
        -------
        None
        '''

        # Check input.
        if not isinstance(companions[0], list):
            if len(companions) == 3:
                companions = [companions]
        for i in range(len(companions)):
            if len(companions[i]) != 3:
                raise UserWarning('There should be three elements for each companion in the companions list')

        Ncompanions = len(companions)
        for _, key in enumerate(self.database.obs.keys()):
            ww_type=list(self.database.obs[key]['TYPE'])
            list_of_injected = []
            all_injected = False

            log.info('--> Concatenation ' + key)
            #######################################################################################################################
            filepaths, psflib_filepaths = get_pyklip_filepaths(self.database, key)
            raw_dataset = JWSTData(filepaths, psflib_filepaths)

            for ww in range(len(ww_type)):
                # Read input files and store values that we just want to save in the output_dir
                fitsfile = self.database.obs[key]['FITSFILE'][ww]
                data, erro, pxdq, head_pri, head_sci, is2d, imshifts, maskoffs = ut.read_obs(fitsfile)
                maskfile = self.database.obs[key]['MASKFILE'][ww]
                mask = ut.read_msk(maskfile)
                crpix1 = self.database.obs[key]['CRPIX1'][ww]
                crpix2 = self.database.obs[key]['CRPIX2'][ww]
                head, tail = os.path.split(fitsfile)

                # Write FITS file and PSF mask.
                head_sci['CRPIX1'] = crpix1
                head_sci['CRPIX2'] = crpix2

                # Inject only into SCI type data
                if ww_type[ww] == 'SCI':
                    # Convert the host star brightness from vegamag to MJy. Use an
                    # unocculted model PSF whose integrated flux is normalized to
                    # one in order to obtain the theoretical peak count of the
                    # star.
                    filt = self.database.obs[key]['FILTER'][ww]

                    # Get stellar magnitudes and filter zero points.
                    mstar, fzero, fzero_si = get_stellar_magnitudes(starfile, spectral_type,
                                                                    self.database.obs[key]['INSTRUME'][ww], return_si=True,
                                                                    output_dir=output_dir,
                                                                    **kwargs)  # vegamag, Jy, erg/cm^2/s/A
                    # Compute the pixel area in steradian.
                    pxsc_arcsec = self.database.obs[key]['PIXSCALE'][ww]  # arcsec
                    pxsc_rad = pxsc_arcsec / 3600. / 180. * np.pi  # rad
                    pxar = pxsc_rad ** 2  # sr

                    # Set output directory.
                    output_dir = os.path.join(self.database.output_dir, subdir)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    # Make a copy of the dataset
                    dataset = copy.deepcopy(raw_dataset)

                    apername = self.database.obs[key]['APERNAME'][ww]
                    if 'planetfile' not in kwargs.keys() or kwargs['planetfile'] is None:
                        if starfile is not None and starfile.endswith('.txt'):
                            sed = read_spec_file(starfile)
                        else:
                            sed = None
                    else:
                        sed = read_spec_file(kwargs['planetfile'])
                    ww_sci = np.where(self.database.obs[key]['TYPE'] == 'SCI')[0]
                    if date is not None:
                        if date == 'auto':
                            date = fits.getheader(self.database.obs[key]['FITSFILE'][ww_sci[0]], 0)['DATE-BEG']

                    offsetpsf_func = JWST_PSF(apername,
                                              filt,
                                              date=date,
                                              fov_pix=65,
                                              oversample=2,
                                              sp=sed,
                                              use_coeff=False)

                    # Offset PSF that is not affected by the coronagraphic
                    # mask, but only the Lyot stop.
                    psf_no_coronmsk = offsetpsf_func.psf_off
                    log.info('--> Injecting companions, writing FITS files and updating spaceKLIP database: ')

                    # Loop over companions
                    for i in trange(Ncompanions):
                        # Initial guesses for the fit parameters.
                        guess_dx = companions[i][0] / pxsc_arcsec  # pix
                        guess_dy = companions[i][1] / pxsc_arcsec  # pix
                        guess_sep = np.sqrt(guess_dx ** 2 + guess_dy ** 2)  # pix
                        guess_pa = np.rad2deg(np.arctan2(guess_dx, guess_dy))  # deg
                        guess_flux = companions[i][2]  # contrast

                        roll_ref = self.database.obs[key]['ROLL_REF'][ww]  # deg

                        # Get shift between star and coronagraphic mask
                        # position. If positive, the coronagraphic mask center
                        # is to the left/bottom of the star position.
                        _, _, _, _, _, _, _, maskoffs = ut.read_obs(self.database.obs[key]['FITSFILE'][ww])

                        # NIRCam.
                        if maskoffs is not None:
                            mask_xoff = -maskoffs[:, 0]  # pix
                            mask_yoff = -maskoffs[:, 1]  # pix

                            # Need to rotate by the roll angle (CCW) and flip
                            # the x-axis so that positive RA is to the left.
                            mask_raoff = -(mask_xoff * np.cos(np.deg2rad(roll_ref)) - mask_yoff * np.sin(
                                np.deg2rad(roll_ref)))  # pix
                            mask_deoff = mask_xoff * np.sin(np.deg2rad(roll_ref)) + mask_yoff * np.cos(
                                np.deg2rad(roll_ref))  # pix

                            # Compute the true offset between the companion and
                            # the coronagraphic mask center.
                            sim_dx = guess_dx - mask_raoff  # pix
                            sim_dy = guess_dy - mask_deoff  # pix
                            sim_sep = np.sqrt(sim_dx ** 2 + sim_dy ** 2) * pxsc_arcsec  # arcsec
                            sim_pa = np.rad2deg(np.arctan2(sim_dx, sim_dy))  # deg

                            # Take median of observation. Typically, each
                            # dither position is a separate observation.
                            sim_sep = np.median(sim_sep)
                            sim_pa = np.median(sim_pa)

                        # Otherwise.
                        else:
                            sim_sep = np.sqrt(guess_dx ** 2 + guess_dy ** 2) * pxsc_arcsec  # arcsec
                            sim_pa = np.rad2deg(np.arctan2(guess_dx, guess_dy))  # deg

                        # Generate offset PSF for this roll angle. Do not add
                        # the V3Yidl angle as it has already been added to the
                        # roll angle by spaceKLIP. This is only for estimating
                        # the coronagraphic mask throughput!
                        offsetpsf_coronmsk = offsetpsf_func.gen_psf([sim_sep, sim_pa],
                                                                    mode='rth',
                                                                    PA_V3=roll_ref,
                                                                    do_shift=False,
                                                                    quick=True,
                                                                    addV3Yidl=False)

                        # Coronagraphic mask throughput is not incorporated
                        # into the flux calibration of the JWST pipeline so
                        # that the companion flux from the detector pixels will
                        # be underestimated. Therefore, we need to scale the
                        # model offset PSF to account for the coronagraphic
                        # mask throughput (it becomes fainter). Compute scale
                        # factor by comparing a model PSF with and without
                        # coronagraphic mask.
                        scale_factor = np.sum(offsetpsf_coronmsk) / np.sum(psf_no_coronmsk)

                        # Normalize model offset PSF to a total integrated flux
                        # of 1 at infinity. Generates a new webbpsf model with
                        # PSF normalization set to 'exit_pupil'.
                        offsetpsf = offsetpsf_func.gen_psf([sim_sep, sim_pa],
                                                           mode='rth',
                                                           PA_V3=roll_ref,
                                                           do_shift=False,
                                                           quick=False,
                                                           addV3Yidl=False,
                                                           normalize='exit_pupil')

                        # Normalize model offset PSF by the flux of the star.
                        mcomp = mstar[filt] -2.5*np.log10(guess_flux)
                        offsetpsf *= fzero[filt] / 10 ** (mcomp / 2.5) / 1e6 / pxar  # MJy/sr

                        # Apply scale factor to incorporate the coronagraphic
                        # mask througput.
                        offsetpsf *= scale_factor

                        # For Test only, we apply a gaussian kernel to the psf we want to inject to test if we are able
                        # to recover it later when using Analysis.extract_companions
                        if 'sigma_xy' in kwargs.keys():
                            if 'theta_degrees' not in kwargs.keys():
                                kwargs['theta_degrees'] = 0
                            sigma_xy = kwargs['sigma_xy']
                            theta_degrees = kwargs['theta_degrees']
                            kernel = gaussian_kernel(sigma_x=sigma_xy[0], sigma_y=sigma_xy[1], theta_degrees=theta_degrees,n=6)
                            offsetpsf = scipy.ndimage.convolve(offsetpsf, kernel)

                        # Injected PSF needs to be a 3D array that matches dataset
                        inj_psf_3d = np.array([offsetpsf for k in range(dataset.input.shape[0])])

                        # Inject the PSF
                        fakes.inject_planet(frames=dataset.input, centers=dataset.centers, inputflux=inj_psf_3d,
                                            astr_hdrs=dataset.wcs, radius=guess_sep, pa=guess_pa, stampsize=65)

                    data = dataset.input


                # Write FITS file and PSF mask.
                fitsfile = ut.write_obs(fitsfile, output_dir, data, erro, pxdq, head_pri, head_sci, is2d,
                                        imshifts, maskoffs)
                maskfile = ut.write_msk(maskfile, mask, fitsfile)

                # Update spaceKLIP database.
                self.database.update_obs(key,ww, fitsfile, maskfile, crpix1=crpix1, crpix2=crpix2)

    def update_nircam_centers(self,
                              force_siaf_center=False, 
                              force_db_center=False):
        """
        Checks SIAF PRD version against FITS header PRD version and updates
        CRPIX if SIAF version is newer. Also accounts for filter-dependent distortion.
        Might not be required for simulated data.

        This step uses lookup tables of information derived from NIRCam
        commissioning activities CAR-30 and CAR-31, by J. Leisenring and J. Girard,
        and subsequent reanalyses using additional data from PSF SGD observations.

        Parameters
        ----------
        force_siaf_center : bool, optional
            Force the use of the SIAF reference pixel position irrespective of
            versions. The default is False

        force_db_center : bool, optional
            Force the use of the database reference pixel position irrespective
            of versions. The default is False

        Returns
        -------
        None.

        """

        if force_siaf_center and force_db_center:
            raise UserWarning('Both force_siaf_center and force_db_center are set to True. Only one can be True.')

        if not force_db_center:
            siaf = pysiaf.Siaf('NIRCAM')
            # Use same RegEx as pysiaf to get sorted PRDS for later comparison
            prds = [prd for i, prd in enumerate(pysiaf.prd_list) if 
                    bool(re.match(r"^[A-Z]-\d+", pysiaf.prd_list[i].split("PRDOPSSOC-")[1]))
                    is False]  # PRDs matching format: PRODOSSOC-###

        # Loop through concatenations.
        for i, key in enumerate(self.database.obs.keys()):
            log.info('--> Concatenation ' + key)

            # Loop through FITS files.
            for j in range(len(self.database.obs[key])):

                # Skip file types that are not NIRCam coronagraphy.
                if self.database.obs[key]['EXP_TYPE'][j] == 'NRC_CORON':

                    # Read FITS file and PSF mask.
                    fitsfile = self.database.obs[key]['FITSFILE'][j]
                    data, erro, pxdq, head_pri, head_sci, is2d, imshifts, maskoffs = ut.read_obs(fitsfile)
                    maskfile = self.database.obs[key]['MASKFILE'][j]
                    mask = ut.read_msk(maskfile)

                    # Update current reference pixel position.
                    head, tail = os.path.split(fitsfile)
                    log.info('  --> Update NIRCam coronagraphy centers: ' + tail)

                    # Get PRD version used for the current file.
                    file_prd_ver = head_pri['PRD_VER']

                    # use SIAF for reference pixel positions if its PRD is
                    # newer or if force_siaf_center unless force_db_center.
                    if (not force_db_center) and ((np.searchsorted(prds, file_prd_ver) < np.searchsorted(prds, pysiaf.JWST_PRD_VERSION)) 
                        or force_siaf_center):
                        log.info('  --> Update NIRCam coronagraphy centers: using CRPIX from pysiaf')
                        apsiaf = siaf[self.database.obs[key]['APERNAME'][j]]
                        crpix1 = apsiaf.XSciRef
                        crpix2 = apsiaf.YSciRef
                    else:
                        log.info('  --> Update NIRCam coronagraphy centers: using CRPIX from database')
                        crpix1 = self.database.obs[key]['CRPIX1'][j]
                        crpix2 = self.database.obs[key]['CRPIX2'][j]

                    # Get filter shift from Jarron.
                    try:
                        xshift_jarron, yshift_jarron = filter_shifts_jarron[self.database.obs[key]['FILTER'][j]]
                    except KeyError:
                        log.warning('  --> Update NIRCam coronagraphy centers: no filter shift found for ' + self.database.obs[key]['FILTER'][j])
                        xshift_jarron, yshift_jarron = 0., 0.

                    xoff, yoff = xshift_jarron, yshift_jarron
                    log.info('  --> Update NIRCam coronagraphy centers: old = (%.2f, %.2f), new = (%.2f, %.2f)' % (crpix1, crpix2, crpix1 + xoff, crpix2 + yoff))
                    crpix1 += xoff
                    crpix2 += yoff

                    # Update spaceKLIP database.
                    self.database.update_obs(key, j, fitsfile, maskfile, crpix1=crpix1, crpix2=crpix2)

        pass

    def recenter_frames(self,
                        method='fourier',
                        subpix_first_sci_only=False,
                        first_sci_only=True,
                        spectral_type='G2V',
                        shft_exp=1,
                        kwargs={},
                        highpass=False,
                        subdir='recentered'):
        """
        Recenter frames so that the host star position is data.shape // 2. For
        NIRCam coronagraphy, use a WebbPSF model to determine the star position
        behind the coronagraphic mask for the first SCI frame. Then, shift all
        other SCI and REF frames by the same amount. For MIRI coronagraphy, do
        nothing. For all other data types, simply recenter the host star PSF.

        Parameters
        ----------
        method : 'fourier' or 'spline' (not recommended), optional
            Method for shifting the frames. The default is 'fourier'.
        subpix_first_sci_only : bool, optional
            By default, all frames will be recentered to subpixel precision. If
            'subpix_first_sci_only' is True, then only the first SCI frame will
            be recentered to subpixel precision and all other SCI and REF
            frames will only be recentered to integer pixel precision by
            rolling the image. Can be helpful when working with poorly sampled
            data to avoid another interpolation step if the 'align_frames'
            routine is run subsequently. Only applicable to non-coronagraphic
            data. The default is False.
        first_sci_only : bool, optional
            Recenter all files and not just the first SCI file in each concate-
            nation. Only applicable to NIRCam coronagraphy. The default is
            True.
        spectral_type : str, optional
            Host star spectral type for the WebbPSF model used to determine the
            star position behind the coronagraphic mask. The default is 'G2V'.
        shft_exp : float, optional
            Take image to the given power before cross correlating for shifts, default is 1. For instance, 1/2 helps align nircam bar/narrow data (or other data with weird speckles)
        kwargs : dict, optional
            Keyword arguments for the scipy.ndimage.shift routine. The default
            is {}.
        subdir : str, optional
            Name of the directory where the data products shall be saved. The
            default is 'recentered'.

        Returns
        -------
        None.

        """

        # Update NIRCam coronagraphy centers, i.e., change SIAF CRPIX position
        # to true mask center determined by Jarron.
        # self.update_nircam_centers()  # shall be run purposely by the user

        # Set output directory.
        output_dir = os.path.join(self.database.output_dir, subdir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Loop through concatenations.
        for i, key in enumerate(self.database.obs.keys()):
            log.info('--> Concatenation ' + key)

            # Find science and reference files.
            ww_sci = np.where(self.database.obs[key]['TYPE'] == 'SCI')[0]
            ww_sci_ta = np.where(self.database.obs[key]['TYPE'] == 'SCI_TA')[0]
            ww_ref = np.where(self.database.obs[key]['TYPE'] == 'REF')[0]
            ww_ref_ta = np.where(self.database.obs[key]['TYPE'] == 'REF_TA')[0]

            # Loop through FITS files.
            ww_all = np.append(ww_sci, ww_ref)
            ww_all = np.append(ww_all, ww_sci_ta)
            ww_all = np.append(ww_all, ww_ref_ta)
            shifts_all = []
            for j in ww_all:

                # Read FITS file and PSF mask.
                fitsfile = self.database.obs[key]['FITSFILE'][j]
                data, erro, pxdq, head_pri, head_sci, is2d, imshifts, maskoffs = ut.read_obs(fitsfile)
                maskfile = self.database.obs[key]['MASKFILE'][j]
                mask = ut.read_msk(maskfile)

                # Recenter frames. Use different algorithms based on data type.
                head, tail = os.path.split(fitsfile)
                log.info('  --> Recenter frames: ' + tail)
                if np.sum(np.isnan(data)) != 0:
                    raise UserWarning('Please replace nan pixels before attempting to recenter frames')
                shifts = []  # shift between star position and image center (data.shape // 2)
                maskoffs_temp = []  # shift between star and coronagraphic mask position

                # SCI and REF data.
                if j in ww_sci or j in ww_ref:

                    # NIRCam coronagraphy.
                    if self.database.obs[key]['EXP_TYPE'][j] in ['NRC_CORON']:
                        for k in range(data.shape[0]):

                            # For the first SCI frame, get the star position
                            # and the shift between the star and coronagraphic
                            # mask position.

                            if (not first_sci_only or j == ww_sci[0]) and k == 0:
                                xc, yc, xshift, yshift = self.find_nircam_centers(data0=data.copy(),
                                                                                  key=key,
                                                                                  j=j,
                                                                                  shft_exp=shft_exp,
                                                                                  spectral_type=spectral_type,
                                                                                  date=head_pri['DATE-BEG'],
                                                                                  output_dir=output_dir,
                                                                                  highpass=highpass)
                            
                            # Apply the same shift to all SCI and REF frames.
                            shifts += [np.array([-(xc - (data.shape[-1] - 1.) / 2.), -(yc - (data.shape[-2] - 1.) / 2.)])]
                            maskoffs_temp += [np.array([xshift, yshift])]
                            data[k] = ut.imshift(data[k], [shifts[k][0], shifts[k][1]], method=method, kwargs=kwargs)
                            erro[k] = ut.imshift(erro[k], [shifts[k][0], shifts[k][1]], method=method, kwargs=kwargs)
                        if mask is not None:
                            # mask = ut.imshift(mask, [shifts[k][0], shifts[k][1]], method=method, kwargs=kwargs)
                            mask = spline_shift(mask, [shifts[k][1], shifts[k][0]], order=0, mode='constant', cval=np.nanmedian(mask))
                        xoffset = self.database.obs[key]['XOFFSET'][j] - self.database.obs[key]['XOFFSET'][ww_sci[0]]  # arcsec
                        yoffset = self.database.obs[key]['YOFFSET'][j] - self.database.obs[key]['YOFFSET'][ww_sci[0]]  # arcsec
                        crpix1 = (data.shape[-1] - 1.) / 2. + 1.  # 1-indexed
                        crpix2 = (data.shape[-2] - 1.) / 2. + 1.  # 1-indexed
                    
                    # MIRI coronagraphy.
                    elif self.database.obs[key]['EXP_TYPE'][j] in ['MIR_4QPM', 'MIR_LYOT']:
                        log.warning('  --> Recenter frames: not implemented for MIRI coronagraphy, skipped')
                        for k in range(data.shape[0]):

                            # Do nothing.
                            shifts += [np.array([0., 0.])]
                            maskoffs_temp += [np.array([0., 0.])]
                        xoffset = self.database.obs[key]['XOFFSET'][j]  # arcsec
                        yoffset = self.database.obs[key]['YOFFSET'][j]  # arcsec
                        crpix1 = self.database.obs[key]['CRPIX1'][j]  # 1-indexed
                        crpix2 = self.database.obs[key]['CRPIX2'][j]  # 1-indexed

                    # Other data types.
                    else:
                        for k in range(data.shape[0]):

                            # Recenter SCI and REF frames to subpixel precision
                            # using the 'BCEN' routine from XARA.
                            # https://github.com/fmartinache/xara
                            if subpix_first_sci_only == False or (j == ww_sci[0] and k == 0):
                                pp = core.determine_origin(data[k], algo='BCEN')
                                shifts += [np.array([-(pp[0] - data.shape[-1]//2), -(pp[1] - data.shape[-2]//2)])]
                                maskoffs_temp += [np.array([0., 0.])]
                                data[k] = ut.imshift(data[k], [shifts[k][0], shifts[k][1]], method=method, kwargs=kwargs)
                                erro[k] = ut.imshift(erro[k], [shifts[k][0], shifts[k][1]], method=method, kwargs=kwargs)
                            else:
                                shifts += [np.array([0., 0.])]
                                maskoffs_temp += [np.array([0., 0.])]

                            # Recenter SCI and REF frames to integer pixel
                            # precision by rolling the image.
                            ww_max = np.unravel_index(np.argmax(data[k]), data[k].shape)
                            if ww_max != (data.shape[-2]//2, data.shape[-1]//2):
                                dx, dy = data.shape[-1]//2 - ww_max[1], data.shape[-2]//2 - ww_max[0]
                                shifts[-1][0] += dx
                                shifts[-1][1] += dy
                                data[k] = np.roll(np.roll(data[k], dx, axis=1), dy, axis=0)
                                erro[k] = np.roll(np.roll(erro[k], dx, axis=1), dy, axis=0)
                        xoffset = 0.  # arcsec
                        yoffset = 0.  # arcsec
                        crpix1 = data.shape[-1]//2 + 1  # 1-indexed
                        crpix2 = data.shape[-2]//2 + 1  # 1-indexed

                # TA data.
                if j in ww_sci_ta or j in ww_ref_ta:
                    for k in range(data.shape[0]):

                        # Center TA frames on the nearest pixel center. This
                        # pixel center is not necessarily the image center,
                        # which is why a subsequent integer pixel recentering
                        # is required.
                        p0 = np.array([0., 0.])
                        pp = minimize(ut.recenterlsq,
                                      p0,
                                      args=(data[k], method, kwargs))['x']
                        shifts += [np.array([pp[0], pp[1]])]
                        maskoffs_temp += [np.array([0., 0.])]
                        data[k] = ut.imshift(data[k], [shifts[k][0], shifts[k][1]], method=method, kwargs=kwargs)
                        erro[k] = ut.imshift(erro[k], [shifts[k][0], shifts[k][1]], method=method, kwargs=kwargs)

                        # Recenter TA frames to integer pixel precision by
                        # rolling the image.
                        ww_max = np.unravel_index(np.argmax(data[k]), data[k].shape)
                        if ww_max != (data.shape[-2]//2, data.shape[-1]//2):
                            dx, dy = data.shape[-1]//2 - ww_max[1], data.shape[-2]//2 - ww_max[0]
                            shifts[-1][0] += dx
                            shifts[-1][1] += dy
                            data[k] = np.roll(np.roll(data[k], dx, axis=1), dy, axis=0)
                            erro[k] = np.roll(np.roll(erro[k], dx, axis=1), dy, axis=0)
                    xoffset = 0.  # arcsec
                    yoffset = 0.  # arcsec
                    crpix1 = data.shape[-1]//2 + 1  # 1-indexed
                    crpix2 = data.shape[-2]//2 + 1  # 1-indexed
                shifts = np.array(shifts)
                shifts_all += [shifts]
                maskoffs_temp = np.array(maskoffs_temp)
                if imshifts is not None:
                    imshifts += shifts
                else:
                    imshifts = shifts
                if maskoffs is not None:
                    maskoffs += maskoffs_temp
                else:
                    maskoffs = maskoffs_temp

                # Compute shift distances.
                dist = np.sqrt(np.sum(shifts[:, :2]**2, axis=1))  # pix
                dist *= self.database.obs[key]['PIXSCALE'][j] * 1000  # mas
                head, tail = os.path.split(self.database.obs[key]['FITSFILE'][j])
                log.info('  --> Recenter frames: ' + tail)
                log.info('  --> Recenter frames: median required shift = %.2f mas' % np.median(dist))

                # Write FITS file and PSF mask.
                head_pri['XOFFSET'] = xoffset #arcsec
                head_pri['YOFFSET'] = yoffset #arcsec
                head_sci['CRPIX1'] = crpix1
                head_sci['CRPIX2'] = crpix2
                fitsfile = ut.write_obs(fitsfile, output_dir, data, erro, pxdq, head_pri, head_sci, is2d, imshifts, maskoffs)
                maskfile = ut.write_msk(maskfile, mask, fitsfile)

                # Update spaceKLIP database.
                self.database.update_obs(key, j, fitsfile, maskfile, xoffset=xoffset, yoffset=yoffset, crpix1=crpix1, crpix2=crpix2)

        pass

    @plt.style.context('spaceKLIP.sk_style')
    def find_nircam_centers(self,
                            data0,
                            key,
                            j,
                            spectral_type='G2V',
                            shft_exp=1,
                            date=None,
                            output_dir=None,
                            fov_pix=65,
                            oversample=2,
                            use_coeff=False,
                            highpass=False,
                            save_figures=True):
        """
        Find the star position behind the coronagraphic mask using a WebbPSF
        model.

        Parameters
        ----------
        data0 : list
            List of frame for which the star position shall be determined.
        key : str
            Database key of the observation containing the first frame in data0.
        j : int
            Database index of the observation containing the first frame in data0.
        spectral_type : str, optional
            Host star spectral type for the WebbPSF model used to determine the
            star position behind the coronagraphic mask. The default is 'G2V'.
        shft_exp : float, optional
            Take image to the given power before cross correlating for shifts, default is 1.
        date : str, optional
            Observation date in the format 'YYYY-MM-DDTHH:MM:SS.MMM'. The
            default is None.
        output_dir : path, optional
            Path of the directory where the data products shall be saved. The
            default is None.
        oversample : int, optional
            Factor by which the WebbPSF model shall be oversampled. The
            default is 2.
        use_coeff : bool, optional
            Use pre-computed coefficients to generate the WebbPSF model. The
            default is False.
        save_figures : bool, optional
            Save the plots in a PDF?

        Returns
        -------
        xc : float
            Star x-position (pix, 0-indexed).
        yc : float
            Star y-position (pix, 0-indexed).
        xshift : float
            X-shift between star and coronagraphic mask position (pix).
        yshift : float
            Y-shift between star and coronagraphic mask position (pix).

        """

        # Generate host star spectrum.
        spectrum = webbpsf_ext.stellar_spectrum(spectral_type)

        # Get true mask center.
        crpix1 = self.database.obs[key]['CRPIX1'][j] - 1  # 0-indexed
        crpix2 = self.database.obs[key]['CRPIX2'][j] - 1  # 0-indexed

        # Initialize JWST_PSF object. Use odd image size so that PSF is
        # centered in pixel center.
        log.info('  --> Recenter frames: generating WebbPSF image for absolute centering (this might take a while)')
        FILTER = self.database.obs[key]['FILTER'][j]
        APERNAME = self.database.obs[key]['APERNAME'][j]
        kwargs = {
            'fov_pix': fov_pix,
            'oversample': oversample,
            'date': date,
            'use_coeff': use_coeff,
            'sp': spectrum
        }
        psf = JWST_PSF(APERNAME, FILTER, **kwargs)

        # Get SIAF reference pixel position.
        apsiaf = psf.inst_on.siaf_ap
        xsciref, ysciref = (apsiaf.XSciRef, apsiaf.YSciRef)

        # Generate model PSF. Apply offset between SIAF reference pixel
        # position and true mask center.
        xoff = (crpix1 + 1) - xsciref
        yoff = (crpix2 + 1) - ysciref
        model_psf = psf.gen_psf_idl((0, 0), coord_frame='idl', return_oversample=False, quick=True)
        if not isinstance(highpass, bool):
            highpass = float(highpass)
            fourier_sigma_size = (model_psf.shape[0] / highpass) / (2. * np.sqrt(2. * np.log(2.)))
            model_psf = parallelized.high_pass_filter_imgs(np.array([model_psf]), numthreads=None, filtersize=fourier_sigma_size)[0]
        else:
            if highpass:
                raise NotImplementedError()
        if not np.isnan(self.database.obs[key]['BLURFWHM'][j]):
            gauss_sigma = self.database.obs[key]['BLURFWHM'][j] / np.sqrt(8. * np.log(2.))
            model_psf = gaussian_filter(model_psf, gauss_sigma)

        shift_list=[]
        count=0

        for data in data0:
            # Get transmission mask.
            yi, xi = np.indices(data.shape)
            xidl, yidl = apsiaf.sci_to_idl(xi + 1 - xoff, yi + 1 - yoff)
            mask = psf.inst_on.gen_mask_transmission_map((xidl, yidl), 'idl')

            # Determine relative shift between data and model PSF. Iterate 3 times
            # to improve precision.
            xc, yc = (crpix1, crpix2)
            for i in range(3):

                # Crop data and transmission mask.
                datasub, xsub_indarr, ysub_indarr = ut.crop_image(image=data,
                                                                  xycen=(xc, yc),
                                                                  npix=fov_pix,
                                                                  return_indices=True)
                masksub = ut.crop_image(image=mask,
                                        xycen=(xc, yc),
                                        npix=fov_pix)

                if shft_exp == 1:
                    img1 = datasub* masksub
                    img2 = model_psf* masksub
                else:
                    img1 = np.power(np.abs(datasub), shft_exp)* masksub
                    img2 = np.power(np.abs(model_psf), shft_exp) * masksub

                # Determine relative shift between data and model PSF.
                shift, error, phasediff = phase_cross_correlation(img1,
                                                                  img2,
                                                                  upsample_factor=1000,
                                                                  normalization=None)
                yshift, xshift = shift

                # Update star position.
                xc = np.mean(xsub_indarr) + xshift
                yc = np.mean(ysub_indarr) + yshift
            xshift, yshift = (xc - crpix1, yc - crpix2)
            shift_list.append([xshift, yshift])
            log.info('  --> Recenter frames: star offset between frame %i and coronagraph center (dx, dy) = (%.3f, %.3f) pix' % (count,xshift, yshift))
            count+=1

        median_xshift, median_yshift = np.median(np.array(shift_list),axis=0)
        std_xshift, std_yshift = np.std(np.array(shift_list),axis=0)
        log.info( '  --> Recenter frames: median star offset from coronagraph center (dx, dy) = (%.3f, %.3f) pix' % (median_xshift, median_yshift))
        log.info( '  --> Recenter frames: std for the star offset from coronagraph center (dx, dy) = (%.3f, %.3f) pix' % (std_xshift, std_yshift))

        # Plot data, model PSF, and scene overview.
        if output_dir is not None:
            fig, ax = plt.subplots(1, 3, figsize=(3 * 6.4, 1 * 4.8))
            ax[0].imshow(datasub, origin='lower', cmap='Reds')
            ax[0].contourf(masksub, levels=[0.00, 0.25, 0.50, 0.75], cmap='Greys_r', vmin=0., vmax=2., alpha=0.5)
            ax[0].set_title('1. SCI frame & transmission mask')
            ax[1].imshow(model_psf, origin='lower', cmap='Reds')
            ax[1].contourf(masksub, levels=[0.00, 0.25, 0.50, 0.75], cmap='Greys_r', vmin=0., vmax=2., alpha=0.5)
            ax[1].set_title('Model PSF & transmission mask')
            ax[2].scatter((xsciref), (ysciref), marker='+', color='black', label='SIAF reference point')
            ax[2].scatter((crpix1 + 1), (crpix2 + 1), marker='x', color='skyblue', label='True mask center')
            ax[2].scatter((xc + 1), (yc + 1), marker='*', color='red', label='Computed star position')
            ax[2].set_aspect('equal')
            xlim = ax[2].get_xlim()
            ylim = ax[2].get_ylim()
            xrng = xlim[1]-xlim[0]
            yrng = ylim[1]-ylim[0]
            if xrng > yrng:
                ax[2].set_xlim(np.mean(xlim) - xrng, np.mean(xlim) + xrng)
                ax[2].set_ylim(np.mean(ylim) - xrng, np.mean(ylim) + xrng)
            else:
                ax[2].set_xlim(np.mean(xlim) - yrng, np.mean(xlim) + yrng)
                ax[2].set_ylim(np.mean(ylim) - yrng, np.mean(ylim) + yrng)
            ax[2].set_xlabel('x-position [pix]')
            ax[2].set_ylabel('y-position [pix]')
            ax[2].legend(loc='upper right', fontsize=12)
            ax[2].set_title('Scene overview (1-indexed)')
            plt.tight_layout()
            if save_figures:
                output_file = os.path.split(self.database.obs[key]['FITSFILE'][j])[1]
                output_file = output_file.replace('.fits', '.pdf')
                output_file = os.path.join(output_dir, output_file)
                plt.savefig(output_file)
                log.info(f" Plot saved in {output_file}")
            plt.show()
            plt.close(fig)


        # Return star position.
        return xc, yc, median_xshift, median_yshift

    @plt.style.context('spaceKLIP.sk_style')
    def align_frames(self,
                     method='fourier',
                     align_algo='leastsq',
                     mask_override=None,
                     msk_shp=8,
                     shft_exp=1,
                     align_to_file=None,
                     scale_prior=False,
                     kwargs={},
                     subdir='aligned',
                     save_figures=True):
        """
        Align all SCI and REF frames to the first SCI frame.

        Parameters
        ----------
        method : 'fourier' or 'spline' (not recommended), optional
            Method for shifting the frames. The default is 'fourier'.
        align_algo : 'leastsq' or 'header'
            Algorithm to determine the alignment offsets. Default is 'leastsq',
            'header' assumes perfect header offsets.
        mask_override : str, optional
            Mask some pixels when cross correlating for shifts
        msk_shp : int, optional
            Shape (height or radius, or [inner radius, outer radius]) for custom mask invoked by "mask_override"
        shft_exp : float, optional
            Take image to the given power before cross correlating for shifts, default is 1. For instance, 1/2 helps align nircam bar/narrow data (or other data with weird speckles)
        align_to_file : str, optional
            Path to FITS file to which all images shall be aligned. Needs to be
            a file with the same observational setup as all concatenations in
            the spaceKLIP database. Hence, this can only be applied to one
            observational setup at a time. The default is None.
        scale_prior : bool, optional
            If True, tries to find a better prior for the scale factor instead
            of simply using 1. The default is False.
        kwargs : dict, optional
            Keyword arguments for the scipy.ndimage.shift routine. The default
            is {}.
        subdir : str, optional
            Name of the directory where the data products shall be saved. The
            default is 'aligned'.
        save_figures : bool, optional
            Save the plots in a PDF?

        Returns
        -------
        None.

        """

        # Set output directory.
        output_dir = os.path.join(self.database.output_dir, subdir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # useful masks for computing shifts:
        def create_annulus_mask(h, w, center=None, radius=None):

            if center is None: # use the middle of the image
                center = (int(w/2), int(h/2))
            if radius is None: # use the smallest distance between the center and image walls
                radius = min(center[0], center[1], w-center[0], h-center[1])

            Y, X = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

            mask = (dist_from_center <= radius[0]) | (dist_from_center >= radius[1])
            return mask
        def create_circular_mask(h, w, center=None, radius=None):

            if center is None: # use the middle of the image
                center = (int(w/2), int(h/2))
            if radius is None: # use the smallest distance between the center and image walls
                radius = min(center[0], center[1], w-center[0], h-center[1])

            Y, X = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

            mask = dist_from_center <= radius
            return mask

        def create_rec_mask(h, w, center=None, z=None):
            if center is None: # use the middle of the image
                center = (int(w/2), int(h/2))
            if z is None:
                z = h//4

            mask = np.zeros((h,w), dtype=bool)
            mask[center[1]-z:center[1]+z,:] = True

            return mask

        # Loop through concatenations.
        database_temp = deepcopy(self.database.obs)
        for i, key in enumerate(self.database.obs.keys()):
            log.info('--> Concatenation ' + key)

            # Find science and reference files.
            ww_sci = np.where(self.database.obs[key]['TYPE'] == 'SCI')[0]
            if len(ww_sci) == 0:
                raise UserWarning('Could not find any science files')
            ww_ref = np.where(self.database.obs[key]['TYPE'] == 'REF')[0]
            ww_all = np.append(ww_sci, ww_ref)

            # Loop through FITS files.
            if align_to_file is not None:
                try:
                    ref_image = pyfits.getdata(align_to_file, 'SCI')
                except:
                    ref_image = pyfits.getdata(align_to_file, 0)
                if ref_image.ndim == 3:
                    ref_image = np.nanmedian(ref_image, axis=0)
            shifts_all = []
            for j in ww_all:

                # Read FITS file and PSF mask.
                fitsfile = self.database.obs[key]['FITSFILE'][j]
                data, erro, pxdq, head_pri, head_sci, is2d, imshifts, maskoffs = ut.read_obs(fitsfile)
                maskfile = self.database.obs[key]['MASKFILE'][j]
                mask = ut.read_msk(maskfile)
                if mask_override is not None:
                    if mask_override == 'ann':
                        mask_circ = create_annulus_mask(data[0].shape[0], data[0].shape[1], radius=msk_shp)
                    elif mask_override == 'circ':
                        mask_circ = create_circular_mask(data[0].shape[0],data[0].shape[1], radius=msk_shp)
                    elif mask_override == 'rec':
                        mask_circ = create_rec_mask(data[0].shape[0],data[0].shape[1], z=msk_shp)
                    else:
                        raise ValueError('There are `circ` and `rec` custom masks available')
                    mask_temp = data[0].copy()
                    mask_temp[~mask_circ] = 1
                    mask_temp[mask_circ] = 0
                elif mask is None:
                    mask_temp = np.ones_like(data[0])
                else:
                    mask_temp = mask.copy()
                
                # Align frames.
                head, tail = os.path.split(fitsfile)
                log.info('  --> Align frames: ' + tail)
                if np.sum(np.isnan(data)) != 0:
                    raise UserWarning('Please replace nan pixels before attempting to align frames')
                shifts = []
                for k in range(data.shape[0]):

                    # Take the first science frame as reference frame.
                    if j == ww_sci[0] and k == 0:
                        if align_to_file is None:
                            ref_image = data[k].copy()
                        pp = np.array([0., 0., 1.])
                        xoffset = self.database.obs[key]['XOFFSET'][j] #arcsec
                        yoffset = self.database.obs[key]['YOFFSET'][j] #arcsec
                        crpix1 = self.database.obs[key]['CRPIX1'][j] #pixels
                        crpix2 = self.database.obs[key]['CRPIX2'][j] #pixels
                        pxsc = self.database.obs[key]['PIXSCALE'][j] #arcsec

                    # Align all other SCI and REF frames to the first science
                    # frame.
                    if align_to_file is not None or j != ww_sci[0] or k != 0:
                        # Calculate shifts relative to first frame, work in pixels
                        xfirst = crpix1 + (xoffset/pxsc)
                        xoff_curr_pix = self.database.obs[key]['XOFFSET'][j]/self.database.obs[key]['PIXSCALE'][j]
                        xcurrent = self.database.obs[key]['CRPIX1'][j] + xoff_curr_pix
                        xshift = xfirst - xcurrent

                        yfirst = crpix2 + (yoffset/pxsc)
                        yoff_curr_pix = self.database.obs[key]['YOFFSET'][j]/self.database.obs[key]['PIXSCALE'][j]
                        ycurrent = self.database.obs[key]['CRPIX2'][j] + yoff_curr_pix
                        yshift = yfirst - ycurrent

                        if scale_prior:
                            ww = mask < 0.5
                            sh = mask.shape
                            bw = 100
                            ww[:bw, :] = 0.
                            ww[:, :bw] = 0.
                            ww[sh[0] - bw:, :] = 0.
                            ww[:, sh[1] - bw:] = 0.
                            # plt.imshow(ww, origin='lower')
                            # plt.show()
                            scale = np.nanmedian(np.true_divide(ref_image, data[k])[ww])
                            if shft_exp != 1:
                                scale = np.power(np.abs(scale), shft_exp)
                            p0 = np.array([xshift, yshift, scale])
                        else:
                            p0 = np.array([xshift, yshift, 1.])

                        # Fix for weird numerical behaviour if shifts are small
                        # but not exactly zero.
                        if (np.abs(xshift) < 1e-3) and (np.abs(yshift) < 1e-3):
                            p0 = np.array([0., 0., p0[-1]])
                        if align_algo == 'leastsq':
                            if shft_exp != 1:
                                args = (np.power(np.abs(data[k]), shft_exp), np.power(np.abs(ref_image), shft_exp), mask_temp, method, kwargs)
                            else:
                                args = (data[k], ref_image, mask_temp, method, kwargs)
                            # Use header values to initiate least squares fit
                            pp = leastsq(ut.alignlsq,
                                         p0,
                                         args=args)[0]
                        elif align_algo == 'header':
                            # Just assume the header values are correct
                            pp = p0

                    # Append shifts to array and apply shift to image
                    # using defined method.
                    shifts += [np.array([pp[0], pp[1], pp[2]])]
                    if align_to_file is not None or j != ww_sci[0] or k != 0:
                        data[k] = ut.imshift(data[k], [shifts[k][0], shifts[k][1]], method=method, kwargs=kwargs)
                        erro[k] = ut.imshift(erro[k], [shifts[k][0], shifts[k][1]], method=method, kwargs=kwargs)
                shifts = np.array(shifts)
                if mask is not None:
                    if align_to_file is not None or j != ww_sci[0]:
                        temp = np.median(shifts, axis=0)
                        mask = spline_shift(mask, [temp[1], temp[0]], order=0, mode='constant', cval=np.nanmedian(mask))
                shifts_all += [shifts]
                if imshifts is not None:
                    imshifts += shifts[:, :-1]
                else:
                    imshifts = shifts[:, :-1]
                if maskoffs is not None:
                    maskoffs -= shifts[:, :-1]
                else:
                    maskoffs = -shifts[:, :-1]

                # Compute shift distances.
                dist = np.sqrt(np.sum(shifts[:, :2]**2, axis=1))  # pix
                dist *= self.database.obs[key]['PIXSCALE'][j]*1000  # mas
                if j == ww_sci[0]:
                    dist = dist[1:]
                log.info('  --> Align frames: median required shift = %.2f mas' % np.median(dist))
                if self.database.obs[key]['TELESCOP'][j] == 'JWST':
                    ww = (dist < 1e-5) | (dist > 100.)
                else:
                    ww = (dist < 1e-5)
                if np.sum(ww) != 0:
                    if j == ww_sci[0]:
                        ww = np.append(np.array([False]), ww)
                    ww = np.where(ww == True)[0]
                    if align_algo != 'header':
                        log.warning('  --> The following frames might not be properly aligned: '+str(ww))

                # Write FITS file and PSF mask.
                head_pri['XOFFSET'] = xoffset #arcseconds
                head_pri['YOFFSET'] = yoffset #arcseconds
                head_sci['CRPIX1'] = crpix1
                head_sci['CRPIX2'] = crpix2
                fitsfile = ut.write_obs(fitsfile, output_dir, data, erro, pxdq, head_pri, head_sci, is2d, imshifts, maskoffs)
                maskfile = ut.write_msk(maskfile, mask, fitsfile)

                # Update spaceKLIP database.
                self.database.update_obs(key, j, fitsfile, maskfile, xoffset=xoffset, yoffset=yoffset, crpix1=crpix1, crpix2=crpix2)

            # Plot science frame alignment.
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            fig = plt.figure(figsize=(6.4, 4.8))
            ax = plt.gca()
            for index, j in enumerate(ww_sci):
                ax.scatter(shifts_all[index][:, 0] * self.database.obs[key]['PIXSCALE'][j] * 1000, 
                           shifts_all[index][:, 1] * self.database.obs[key]['PIXSCALE'][j] * 1000,
                           s=5, color=colors[index%len(colors)], marker='o',
                           label='PA = %.0f deg' % self.database.obs[key]['ROLL_REF'][j])
            ax.axhline(0., color='gray', lw=1, zorder=-1)  # set zorder to ensure lines are drawn behind all the scatter points
            ax.axvline(0., color='gray', lw=1, zorder=-1)

            ax.set_aspect('equal')
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            xrng = xlim[1]-xlim[0]
            yrng = ylim[1]-ylim[0]
            if xrng > yrng:
                ax.set_xlim(np.mean(xlim) - xrng, np.mean(xlim) + xrng)
                ax.set_ylim(np.mean(ylim) - xrng, np.mean(ylim) + xrng)
            else:
                ax.set_xlim(np.mean(xlim) - yrng, np.mean(xlim) + yrng)
                ax.set_ylim(np.mean(ylim) - yrng, np.mean(ylim) + yrng)
            ax.set_xlabel('x-shift [mas]')
            ax.set_ylabel('y-shift [mas]')
            ax.legend(loc='upper right')
            ax.set_title(f'Science frame alignment\nfor {self.database.obs[key]["TARGPROP"][ww_sci[0]]}, {self.database.obs[key]["FILTER"][ww_sci[0]]}')
            if save_figures:
                output_file = os.path.join(output_dir, key + '_align_sci.pdf')
                plt.savefig(output_file)
                log.info(f" Plot saved in {output_file}")
            plt.show()
            plt.close(fig)
            
            # Plot reference frame alignment.
            if len(ww_ref) > 0:
                colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                fig = plt.figure(figsize=(6.4, 4.8))
                ax = plt.gca()
                seen = []
                reps = []
                syms = ['o', 'v', '^', '<', '>'] * (1 + len(ww_ref) // 5)
                add = len(ww_sci)
                for index, j in enumerate(ww_ref):
                    this = '%.3f_%.3f' % (database_temp[key]['XOFFSET'][j], database_temp[key]['YOFFSET'][j])
                    if this not in seen:
                        ax.scatter(shifts_all[index + add][:, 0] * self.database.obs[key]['PIXSCALE'][j] * 1000, 
                                   shifts_all[index + add][:, 1] * self.database.obs[key]['PIXSCALE'][j] * 1000, 
                                   s=5, color=colors[len(seen)%len(colors)], marker=syms[0], 
                                   label='dither %.0f' % (len(seen) + 1))
                        ax.hlines((-database_temp[key]['YOFFSET'][j] + yoffset) * 1000, 
                                  (-database_temp[key]['XOFFSET'][j] + xoffset) * 1000 - 4., 
                                  (-database_temp[key]['XOFFSET'][j] + xoffset) * 1000 + 4.,
                                  color=colors[len(seen)%len(colors)], lw=1)
                        ax.vlines((-database_temp[key]['XOFFSET'][j] + xoffset) * 1000, 
                                  (-database_temp[key]['YOFFSET'][j] + yoffset) * 1000 - 4., 
                                  (-database_temp[key]['YOFFSET'][j] + yoffset) * 1000 + 4., 
                                  color=colors[len(seen)%len(colors)], lw=1)
                        seen += [this]
                        reps += [1]
                    else:
                        ww = np.where(np.array(seen) == this)[0][0]
                        ax.scatter(shifts_all[index + add][:, 0] * self.database.obs[key]['PIXSCALE'][j] * 1000, 
                                   shifts_all[index + add][:, 1] * self.database.obs[key]['PIXSCALE'][j] * 1000, 
                                   s=5, color=colors[ww%len(colors)], marker=syms[reps[ww]])
                        reps[ww] += 1
                ax.set_aspect('equal')
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                xrng = xlim[1]-xlim[0]
                yrng = ylim[1]-ylim[0]
                if xrng > yrng:
                    ax.set_xlim(np.mean(xlim) - xrng, np.mean(xlim) + xrng)
                    ax.set_ylim(np.mean(ylim) - xrng, np.mean(ylim) + xrng)
                else:
                    ax.set_xlim(np.mean(xlim) - yrng, np.mean(xlim) + yrng)
                    ax.set_ylim(np.mean(ylim) - yrng, np.mean(ylim) + yrng)
                ax.set_xlabel('x-shift [mas]')
                ax.set_ylabel('y-shift [mas]')
                ax.legend(loc='upper right', fontsize='small')
                ax.set_title(f'Reference frame alignment\n showing {len(ww_ref)} PSF refs for {self.database.obs[key]["FILTER"][ww_ref[0]]}')
                if save_figures:
                    output_file = os.path.join(output_dir, key + '_align_ref.pdf')
                    plt.savefig(output_file)
                    log.info(f" Plot saved in {output_file}")
                plt.show()
                plt.close(fig)
                
    @plt.style.context('spaceKLIP.sk_style')
    def subtract_nircam_coron_background(self,
                                        subdir='bgsub',
                                        mask_snr_threshold=2,
                                        r_excl_nfwhm=40,
                                        q_clip=5.,
                                        align_wrapped=True,
                                        include_global_offset=True,
                                        include_stellar_psf_component=True,
                                        generate_plot=True,
                                        save_model=False,
                                        use_jbt_background=False,
                                        bgmodel_dir=None, 
                                        background_sb={},
                                        restrict_to=None):
        """
        Fits and subtracts the astrophysical background in NIRCam coronagraphic
        data following the procedure described in Lawson et al. (2024).
        
        Note: This step should only be applied to data that has already been
        aligned. Otherwise, it will crash.
        
        For SW filters using a LW coronagraph, the field of view excludes the
        neutral density squares. In this case, the astrophysical background and
        the artificial background offset that we fit are fully degenerate in
        the regions we consider (away from the coronagraph). Since SW
        backgrounds should be low anyway, the default is to assume an
        astrophysical background of zero here. Alternatively, the JWST
        Backgrounds Tool can be used to estimate the background for affected
        data instead (if use_jbt_background=True and the JWST Backgrounds Tool
        is installed).

        Parameters
        ----------
        subdir : str, optional
            Name of the sub-directory where the data products will be saved.
            The default is 'bgsub'.
        mask_snr_threshold : float, optional
            SNR threshold for features to be masked during fitting of the
            background. SNR is estimated using the ERR FITS extension. The
            default is 2.
        r_excl_nfwhm : float, optional
            Radius (in units of the effective PSF FWHM) of the region around
            the star to exclude from the fit. The default is 50.
        q_clip : float, optional
            After computing BG model residuals, exclude q_clip% of pixels from
            both ends of the residual distribution before computing chisq. This
            is intended to avoid over-/under-estimation of the background due
            to unmasked sources or artifacts. Default is 5.
        align_wrapped : bool, optional 
            Whether input data were aligned using a Fourier shift without
            padding first (such that values wrapped at the edges). Default is
            True.
        include_global_offset : bool, optional
            Whether to fit a uniform background offset along with the
            astrophysical background model. This corrects for offsets induced
            by ramp fitting or use of the median subtraction step. Default is
            True.
        include_stellar_psf_component : bool, optional
            Whether to include a stellar PSF model component when optimizing
            the background model. Default is True.
        generate_plot : bool, optional 
            Whether to generate a plot showing the data before and after
            subtraction along with the model and masked residuals. Default is
            True.
        save_model : bool, optional
            Whether to save the optimized background model. Default is False.
        use_jbt_background : bool, optional
            Whether to use the JWST Backgrounds Tool to estimate the background
            surface brightness for data without coverage of ND squares.
            Requires that jwst_backgrounds is installed. Default is False.
        bgmodel_dir : str, optional
            Path to the directory containing the normalized background model
            component FITS files to use (or to which they should be
            downloaded). If None, uses spaceKLIP/resources/nircam_bg_models/.
            Default is None.
        background_sb : dict, optional
            A dictionary of fixed background surface brightness (SB) values (in
            the same units as the data) to adopt for any included concatenation
            keys. For each key in database.obs, if background_sb[key] is None
            or if the key is not in background_sb, the background SB will be
            fit for all observations of that concatenation if possible.
            Otherwise, background_sb[key] should be an array-like of float or
            None having the same length as database.obs[key]. If
            background_sb[key][j] is None, the jth frame's background SB will
            be fit, otherwise it will be fixed to the value
            background_sb[key][j]. Default is {}.
        restrict_to : str or None, optional
            Restrict the background subtraction to a specific key in the
            database. Default is None.
        """

        def get_jbt_background_est(t, ra, dec, wavelength):
            """
            Uses the JWST Backgrounds tool to estimate the background surface
            brightness at a given time, position, and wavelength. 
            """
            from jwst_backgrounds import jbt
            from astropy.time import Time
            tobs = Time(t, format='mjd')
            bkg = jbt.background(ra, dec, wavelength)
            calendar = bkg.bkg_data['calendar']
            tobs0 = Time(f'{tobs.datetime.year}-01-01T00:00:00')
            thisday = int(np.round((tobs.mjd-tobs0.mjd)+1))
            Fbg = bkg.bathtub['total_thiswave'][np.where(thisday == calendar)[0][0]]
            return Fbg
        
        def background_objective(p, im, bg0, psf0, optmask, q=5):
            """
            Objective function for fitting the multi-component background model
            using LMFit.
            """
            fbg, bg_offset, fpsf = [p[key] for key in p]
            res = (im - fbg*bg0 - bg_offset - fpsf*psf0)[optmask]
            low, upp = np.nanpercentile(res, [q, 100.-q])
            return np.abs(res[(res >= low) & (res <= upp)])
    
        def get_stellar_model_path(key, bgmodel_dir):
            """
            Searches for the correct stellar model component on disk and
            fetches from an online repository if needed. Returns the path to
            the FITS file.
            """
            psffile = f'{bgmodel_dir}{key}_psf0.fits'
            if not os.path.exists(psffile):
                with fits.open(f'https://github.com/kdlawson/nircam_bgsub_go4050/raw/main/nominal_bgmodels/{key}_psf0.fits') as hdul:
                    hdul.writeto(psffile)
            return psffile

        def get_background_model_path(key, bgmodel_dir):
            """
            Searches for the correct background model component on disk and
            fetches from an online repository if needed. Returns the path to
            the FITS file.
            """
            bgfile = f'{bgmodel_dir}{key}_background0.fits'
            if not os.path.exists(bgfile):
                with fits.open(f'https://github.com/kdlawson/nircam_bgsub_go4050/raw/main/nominal_bgmodels/{key}_background0.fits') as hdul:
                    hdul.writeto(bgfile)
            return bgfile

        output_dir = os.path.join(self.database.output_dir, subdir+'/')
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
            
        if bgmodel_dir is None:
            bgmodel_dir = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'resources/nircam_bg_models/')
        if not os.path.exists(bgmodel_dir):
            os.makedirs(bgmodel_dir)

        # Copy input SB dictionary so we can fill out / change values as needed.
        bg_sb_dict = background_sb.copy()

        for key in self.database.obs:
            if ((restrict_to is not None) and (restrict_to not in key)) or not np.any(np.isin(self.database.obs[key]['TYPE'], ['SCI', 'REF'])):
                continue

            log.info('--> Concatenation ' + key)

            # Fill out dictionary of SB values, replacing None with a fixed value where fitting is not possible (SW filters with LW coronagraphs).
            if (key not in bg_sb_dict) or (bg_sb_dict[key] is None):
                bg_sb_dict[key] = np.repeat(None, len(self.database.obs[key]))
            for j,entry in enumerate(self.database.obs[key]):
                if (entry['DETECTOR'] == 'NRCA2') and (entry['CORONMSK'] in ['MASK335R', 'MASK430R', 'MASKLWB']) and (entry['SUBARRAY'] != 'FULL') and (bg_sb_dict[key][j] is None): 
                    if use_jbt_background:
                        try:
                            bg_sb_dict[key][j] = get_jbt_background_est(entry['EXPSTART'], entry['TARG_RA'], entry['TARG_DEC'], entry['CWAVEL'])
                        except ModuleNotFoundError:
                            raise ModuleNotFoundError("""
                            JBT background estimation requires
                            the jwst_backgrounds package. Either
                            install jwst_backgrounds or rerun
                            with use_jbt_background=False.
                            """)
                    else:
                        bg_sb_dict[key][j] = 0.

            db_tab = self.database.obs[key]

            fwhm = db_tab['CWAVEL'][0] * 1e-6 / 5.2 * 180. / np.pi * 3600. / db_tab['PIXSCALE'][0]
            blur_fwhm = db_tab['BLURFWHM'][0]
            if np.isfinite(blur_fwhm):
                blur_sigma = blur_fwhm/np.sqrt(8.*np.log(2.))
            else:
                blur_sigma = None

            # Load the normalized stellar PSF model component
            if include_stellar_psf_component:
                psffile = get_stellar_model_path(key, bgmodel_dir=bgmodel_dir)
                with fits.open(psffile) as hdul:
                    psf0_osamp, hpsf0 = hdul['OVERSAMP'].data, hdul['OVERSAMP'].header
                osamp = hpsf0['OSAMP']
                c_psf0_osamp = np.array([hpsf0['CRPIX1'], hpsf0['CRPIX2']])-1
                if blur_sigma is not None:
                    psf0_osamp = gaussian_filter(psf0_osamp, blur_sigma*osamp)

            # Load the normalized BG model component
            if not np.all(bg_sb_dict[key] == 0):
                bgfile = get_background_model_path(key, bgmodel_dir=bgmodel_dir)
                with fits.open(bgfile) as hdul:
                    bg0_osamp, hbg0 = hdul['OVERSAMP'].data, hdul['OVERSAMP'].header
                c_coron_bg0 = np.array([hbg0['CRPIX1'], hbg0['CRPIX2']])-1
                osamp = hbg0['OSAMP']
                # Apply any blurring used for the data:
                if blur_sigma is not None:
                    bg0_osamp = gaussian_filter(bg0_osamp, blur_sigma*osamp)

            files = db_tab['FITSFILE']

            c_star = np.array([db_tab['CRPIX1'][0], db_tab['CRPIX2'][0]])-1

            h1 = fits.getheader(files[0], ext=1)
            ny, nx = h1['NAXIS2'], h1['NAXIS1']

            rmap = np.sqrt((np.arange(0, nx, dtype=np.float32)-c_star[0])**2 + (np.arange(0, ny, dtype=np.float32)-c_star[1])[:, np.newaxis]**2)
            rmap_nfwhm = rmap/fwhm # Map of each pixel's distance from the location of the star in units of the effective PSF FWHM

            # With no alignment wrapping, the stellar PSF model should be the same for all frames, so we'll just set it up once outside the loop over files.
            if not align_wrapped:
                if include_stellar_psf_component:
                    c_star_osamp = c_star*osamp + 0.5*(osamp-1)
                    psf0_osamp_crop = webbpsf_ext.image_manip.crop_image(psf0_osamp, [ny*osamp, nx*osamp], 
                                                                        xyloc=c_psf0_osamp, delx=c_star_osamp[0]-(nx*osamp-1)/2.,
                                                                        dely=c_star_osamp[1]-(ny*osamp-1)/2.)
                    psf0_crop = webbpsf_ext.image_manip.frebin(psf0_osamp_crop, scale=1/osamp, total=False)
                else:
                    psf0_crop = np.zeros((ny,nx), dtype=np.float32)
                            
            for j,f in enumerate(files):
                if db_tab[j]['TYPE'] not in ['SCI', 'REF']:
                    continue

                head, tail = os.path.split(f)
                log.info('  --> NIRCam Background Subtraction: ' + tail)

                # Assume alignment and background differences between integrations are negligible, so we can use the higher SNR coadded exposure
                with fits.open(f) as hdul:
                    ints = hdul['SCI'].data
                    errs = hdul['ERR'].data
                    h1 = hdul['SCI'].header
                    mask_offset = np.nanmedian(hdul['MASKOFFS'].data, axis=0)
                    imshift = np.nanmedian(hdul['IMSHIFTS'].data, axis=0)

                if np.ndim(ints) == 3:
                    med = np.nanmedian(ints, axis=0)
                    nsample = np.sum(np.isfinite(ints), axis=0)
                    err = np.sqrt(np.nansum(errs**2, axis=0))/nsample
                else: # In case using coadded data saved with only two dims
                    med, err = ints, errs

                c_coron = c_star - mask_offset # post-alignment mask center position

                if align_wrapped:
                    if bg_sb_dict[key][j] == 0:
                        bg0_crop = np.zeros_like(med)
                    else:
                        c_coron_osamp_preshift = (c_coron-imshift)*osamp + 0.5*(osamp-1)
                        bg0_osamp_crop = webbpsf_ext.image_manip.crop_image(bg0_osamp, [ny*osamp, nx*osamp], 
                                                                            xyloc=c_coron_bg0, delx=c_coron_osamp_preshift[0]-(nx*osamp-1)/2.,
                                                                            dely=c_coron_osamp_preshift[1]-(ny*osamp-1)/2.)
                        bg0_osamp_crop = ut.imshift(bg0_osamp_crop, imshift*osamp, pad=False)
                        bg0_crop = webbpsf_ext.image_manip.frebin(bg0_osamp_crop, scale=1/osamp, total=False) # Downsample to detector resolution

                    if include_stellar_psf_component:
                        c_star_osamp_preshift = (c_star-imshift)*osamp + 0.5*(osamp-1)
                        psf0_osamp_crop = webbpsf_ext.image_manip.crop_image(psf0_osamp, [ny*osamp, nx*osamp], 
                                                                            xyloc=c_psf0_osamp, delx=c_star_osamp_preshift[0]-(nx*osamp-1)/2.,
                                                                            dely=c_star_osamp_preshift[1]-(ny*osamp-1)/2.)
                        psf0_osamp_crop = ut.imshift(psf0_osamp_crop, imshift*osamp, pad=False)
                        psf0_crop = webbpsf_ext.image_manip.frebin(psf0_osamp_crop, scale=1/osamp, total=False)
                    else:
                        psf0_crop = np.zeros_like(med)
                else:
                    if bg_sb_dict[key][j] == 0:
                        bg0_crop = np.zeros_like(med)
                    else:
                        c_coron_osamp = c_coron*osamp + 0.5*(osamp-1)
                        bg0_osamp_crop = webbpsf_ext.image_manip.crop_image(bg0_osamp, [ny*osamp, nx*osamp], 
                                                                            xyloc=c_coron_bg0, delx=c_coron_osamp[0]-(nx*osamp-1)/2.,
                                                                            dely=c_coron_osamp[1]-(ny*osamp-1)/2.)
                        bg0_crop = webbpsf_ext.image_manip.frebin(bg0_osamp_crop, scale=1/osamp, total=False) # Downsample to detector resolution

                if bg_sb_dict[key][j] is None:
                    fbg0 = 1
                    fbg_vary = True
                else:
                    fbg0 = bg_sb_dict[key][j]
                    fbg_vary = False

                optmask = rmap_nfwhm > r_excl_nfwhm
                snr = med/err # SNR estimate using FITS ERR extension 
                med_snr = np.nanmedian(snr[optmask]) # Median SNR in the nominal background area
                low_snr = snr <= (med_snr+mask_snr_threshold) # High SNR features are those more than mask_snr_threshold sigma above the approximate BG SNR
                optmask = optmask & low_snr

                bg_offset0 = 0
                fpsf0 = 1. if not include_stellar_psf_component else np.nansum((med-np.nanmedian(med[optmask])) * psf0_crop) / np.nansum((psf0_crop ** 2))

                # Prepare the lmfit Parameters object with default values and sensible bounds
                p = lmfit.Parameters()
                p.add('fbg', value=fbg0, min=0, max=np.inf, vary=fbg_vary)
                p.add('bg_offset', value=bg_offset0, min=-np.inf, max=0, vary=include_global_offset)
                p.add('fpsf', value=fpsf0, min=0, max=fpsf0*10, vary=include_stellar_psf_component)

                # Optimize the background model
                result = lmfit.minimize(background_objective, p, args=(med, bg0_crop, psf0_crop, optmask, q_clip), method='powell')
                pfin = result.params.valuesdict()
                logstr = ', '.join([f"{key}:{value:.3f}" for key, value in pfin.items()])
                log.info('  --> NIRCam Background Subtraction: ' + logstr)

                # Compute the final background model and stellar PSF component:
                bg = pfin['fbg']*bg0_crop + pfin['bg_offset']
                psf = psf0_crop*pfin['fpsf']

                f_out = output_dir+os.path.basename(os.path.normpath(f))

                with fits.open(f) as hdul:
                    hdul[1].data -= bg # Subtract the BG model from the original file
                    hdul.writeto(f_out, overwrite=True) # Save to disk in the output directory
                    if save_model:
                        f_model = f_out.replace('.fits', '_background_model.fits')
                        hdu1 = fits.ImageHDU(bg, name='BG')
                        hdul_model = fits.HDUList([hdul[0], hdu1])
                        if include_stellar_psf_component:
                            hdul_model.append(fits.ImageHDU(psf, name='STELLAR_PSF'))

                        # Add fit params to header
                        hdul_model[0].header.update(pfin) 

                        # Add all relevant settings to the header
                        hdul_model[0].header.update(dict(include_global_offset=include_global_offset,
                                                         mask_snr_threshold=mask_snr_threshold, r_excl_nfwhm=r_excl_nfwhm, q=q_clip,
                                                         include_stellar_psf_component=include_stellar_psf_component))
                        hdul_model.writeto(f_model, overwrite=True)
                        hdul_model.close()

                if generate_plot:
                    res = med - bg
                    res_psfsub = res - psf
                    low, upp = np.nanpercentile((res_psfsub)[optmask], [q_clip, 100.-q_clip])
                    res_inliers = np.where((res_psfsub >= low) & (res_psfsub <= upp) & optmask, res_psfsub, np.nan)
                    cmap = copy.copy(mpl.colormaps.get_cmap('RdBu_r'))
                    cmap.set_bad('white')
                    clim = np.array([-1,1])*np.nanpercentile(np.abs(med), 90)
                    plot_mask = np.isfinite(med)
                    plot_ims = np.where(plot_mask, [med, bg, res_inliers, med-bg], np.nan)
                    fig,axes = plt.subplots(1,4,figsize=(15,3.5), sharex=True, sharey=True)
                    labels = ['Data', 'BG Model', 'Masked Residuals', 'Data (BG-subtracted)']
                    norm = mpl.colors.Normalize(*clim)
                    for ind,ax in enumerate(axes):
                        ax.imshow(plot_ims[ind], norm=norm, interpolation='None', origin='lower', cmap=cmap)
                        ax.set_title(labels[ind], pad=10)
                    fig.tight_layout(w_pad=1.00)
                    fig.colorbar(mpl.cm.ScalarMappable(norm, cmap), ax=axes, pad=0.015, label='[MJy / Sr]')     
                    plt.savefig(output_dir+os.path.basename(os.path.normpath(f)).replace('.fits', '_background_model.pdf'), bbox_inches='tight')
                    plt.close(fig)

                mask_in = db_tab['MASKFILE'][j]
                mask = ut.read_msk(mask_in)
                mask_out = ut.write_msk(mask_in, mask, f_out)
                self.database.update_obs(key, j, f_out, mask_out)
