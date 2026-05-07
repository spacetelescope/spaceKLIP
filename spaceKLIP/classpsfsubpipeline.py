from __future__ import division

import matplotlib

# =============================================================================
# IMPORTS
# =============================================================================

import os

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

from astropy import wcs
from pyklip.klip import _rotate_wcs_hdr
from pyklip.klip import rotate as nanrotate
from scipy.ndimage import gaussian_filter
from scipy.optimize import leastsq
from spaceKLIP import utils as ut
from spaceKLIP.psf import get_transmission

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# =============================================================================
# MAIN
# =============================================================================

def run_obs(database,
            kwargs={},
            subdir='psfsub'):
    """
    Run classical PSF subtraction on the input observations database.
    
    Parameters
    ----------
    database : spaceKLIP.Database
        SpaceKLIP database on which classical PSF subtraction shall be run.
    kwargs : dict, optional
        Keyword arguments for the classical PSF subtraction method. Available
        keywords are:

        - combine_dithers : bool, optional
            Combine all dither positions into a single reference PSF or
            subtract each dither position individually? The default is True.
        - save_rolls : bool, optional
            Save each processed roll separately? The default is False.
        - mask_bright : float, optional
            Mask all pixels brighter than this value before minimizing the
            PSF subtraction residuals. 

        The default is {}.

    subdir : str, optional
        Name of the directory where the data products shall be saved. The
        default is 'psfsub'.

    Returns
    -------
    None.

    """
    
    # Check input.
    try:
        kwargs['combine_dithers']
    except KeyError:
        kwargs['combine_dithers'] = True
    try:
        kwargs['save_rolls']
    except KeyError:
        kwargs['save_rolls'] = True
    try:
        kwargs['mask_bright']
    except KeyError:
        kwargs['mask_bright'] = None
    
    # Set output directory.
    output_dir = os.path.join(database.output_dir, subdir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Loop through concatenations.
    for i, key in enumerate(database.obs.keys()):
        log.info('--> Concatenation ' + key)
        
        # Find science and reference files.
        ww_sci = np.where(database.obs[key]['TYPE'] == 'SCI')[0]
        if len(ww_sci) == 0:
            raise UserWarning('Could not find any science files')
        ww_ref = np.where(database.obs[key]['TYPE'] == 'REF')[0]
        if len(ww_ref) == 0:
            raise UserWarning('Could not find any reference files')
        
        # Loop through reference files.
        ref_data = []
        ref_erro = []
        ref_pxdq = []
        for j in ww_ref:
            
            # Read reference file.
            fitsfile = database.obs[key]['FITSFILE'][j]
            data, erro, pxdq, head_pri, head_sci, is2d, alignshift, center_shift, align_mask, center_mask, maskoffs = ut.read_obs(fitsfile)
            
            # For now this routine does not work with nans.
            # if np.sum(np.isnan(data)) != 0:
            #     raise UserWarning('This routine does not work with nans')
            
            # Compute median reference.
            ref_data += [data]
            ref_erro += [erro]
            ref_pxdq += [pxdq]
        
        # Loop through dither positions.
        if kwargs['combine_dithers']:
            ref_data = [np.concatenate(ref_data)]
            ref_erro = [np.concatenate(ref_erro)]
            ref_pxdq = [np.concatenate(ref_pxdq)]
        for dpos in range(len(ref_data)):
            ref_data_temp = np.nanmedian(ref_data[dpos], axis=0)
            nsample = np.sum(np.logical_not(np.isnan(ref_erro[dpos])), axis=0)
            ref_erro_temp = np.true_divide(np.sqrt(np.nansum(ref_erro[dpos]**2, axis=0)), nsample)
            if database.obs[key]['TELESCOP'][ww_ref[0]] == 'JWST' and database.obs[key]['INSTRUME'][ww_ref[0]] == 'NIRCAM':
                ref_pxdq_temp = np.sum(ref_pxdq[dpos] != 0, axis=0) != 0
            else:
                ref_pxdq_temp = np.sum(ref_pxdq[dpos] & 1 == 1, axis=0) != 0
            
            # Loop through science files.
            pps = []
            sci_data = []
            sci_erro = []
            sci_pxdq = []
            sci_mask = []
            sci_effinttm = []
            for ind, j in enumerate(ww_sci):
                
                # Read science file.
                fitsfile = database.obs[key]['FITSFILE'][j]
                data, erro, pxdq, head_pri, head_sci, is2d, alignshift, center_shift, align_mask, center_mask, maskoffs = ut.read_obs(fitsfile)
                # pxdq = fits.getdata(fitsfile.replace('spaceklip_custom_flat', 'spaceklip'), 'DQ')
                maskfile = database.obs[key]['MASKFILE'][j]
                mask = ut.read_msk(maskfile)
                
                # For now this routine does not work with nans.
                # if np.sum(np.isnan(data)) != 0:
                #     raise UserWarning('This routine does not work with nans')
                
                # Compute median science.
                data = np.nanmedian(data, axis=0)
                nsample = np.sum(np.logical_not(np.isnan(erro)), axis=0)
                erro = np.true_divide(np.sqrt(np.nansum(erro**2, axis=0)), nsample)
                if database.obs[key]['TELESCOP'][j] == 'JWST' and database.obs[key]['INSTRUME'][j] == 'NIRCAM':
                    pxdq = np.sum(pxdq != 0, axis=0) != 0
                else:
                    pxdq = np.sum(pxdq & 1 == 1, axis=0) != 0
                
                # Mask data.
                if kwargs['mask_bright'] is not None:
                    temp = np.ones_like(data)
                    temp[data > kwargs['mask_bright']] = 0
                    temp = (temp > 0.5) & (pxdq < 0.5)
                    with plt.style.context('spaceKLIP.sk_style'):
                        plt.figure()
                        plt.imshow(data, origin='lower', vmin=0, vmax=50)
                        plt.imshow(temp, origin='lower', cmap='Greys_r', alpha=0.5)
                        plt.colorbar()
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, key + '_mask.pdf'))
                        # plt.show()
                        plt.close()
                else:
                    temp = None
                
                # Find best fit scaling factor.
                p0 = np.array([1.])
                pp = leastsq(ut.subtractlsq,
                             p0,
                             args=(data, ref_data_temp, temp))[0][0]
                pps += [pp]
                
                # Check best fit scaling factor.
                test = []
                # for k in np.logspace(-1, 1, 100):
                for k in np.linspace(pp - 0.5, pp + 0.5, 100):
                    temp = data - k * ref_data_temp
                    temp = temp - gaussian_filter(temp, 5)
                    test += [temp]
                test = np.array(test)
                hdu0 = pyfits.PrimaryHDU(test)
                hdul = pyfits.HDUList([hdu0])
                hdul.writeto(os.path.join(output_dir, key + '_test.fits'), output_verify='fix', overwrite=True)
                hdul.close()
                
                # Subtract reference using best fit scaling factor.
                data_temp = data - pp * ref_data_temp
                erro_temp = np.sqrt(erro**2 + (pp * ref_erro_temp)**2)
                pxdq_temp = pxdq | ref_pxdq_temp
                
                # Recenter and derotate data.
                center = [database.obs[key]['CRPIX1'][j] - 1., database.obs[key]['CRPIX2'][j] - 1.]  # pix (0-indexed)
                new_center = [data_temp.shape[1] // 2, data_temp.shape[0] // 2]  # pix (0-indexed)
                data_temp_derot = nanrotate(data_temp, database.obs[key]['ROLL_REF'][j], center=center, new_center=new_center)
                erro_temp_derot = nanrotate(erro_temp, database.obs[key]['ROLL_REF'][j], center=center, new_center=new_center)
                
                # Recenter and derotate PSF mask.
                if 'LYOT' in key:
                    center = [database.obs[key]['CRPIX1'][j] - 1., database.obs[key]['CRPIX2'][j] - 1.]  # pix (0-indexed)
                    width = 5  # pix
                    xr = np.arange(mask.shape[1])
                    yr = np.arange(mask.shape[0])
                    xx, yy = np.meshgrid(xr, yr)
                    xx = xx - (database.obs[key]['CRPIX1'][j] - 1.)
                    xx = np.abs(xx)
                    xx = xx < width
                    xx = xx.astype(float)
                    xx = gaussian_filter(xx, width)
                    xx = nanrotate(xx, -4.5, center=center)
                    xx /= np.nanmax(xx)
                    xx = 1. - xx
                    mask = xx
                center = [database.obs[key]['CRPIX1'][j] - 1., database.obs[key]['CRPIX2'][j] - 1.]  # pix (0-indexed)
                new_center = [mask.shape[1] // 2, mask.shape[0] // 2]  # pix (0-indexed)
                mask_temp = nanrotate(mask.copy(), database.obs[key]['ROLL_REF'][j], center=center, new_center=new_center)
                
                # Append data.
                sci_data += [data_temp_derot]
                sci_erro += [erro_temp_derot]
                sci_pxdq += [pxdq_temp]
                sci_mask += [mask_temp]
                sci_effinttm += [database.obs[key]['NINTS'][j] * database.obs[key]['EFFINTTM'][j]]
                
                # Write FITS file.
                if kwargs['save_rolls']:
                    hdul = pyfits.open(fitsfile)
                    hdul[0].header['NINTS'] = 1
                    hdul[0].header['EFFINTTM'] = sci_effinttm[ind]
                    hdul['SCI'].data = data_temp
                    hdul['SCI'].header['CRPIX1'] = data_temp.shape[1] // 2 + 1
                    hdul['SCI'].header['CRPIX2'] = data_temp.shape[0] // 2 + 1
                    hdul['ERR'].data = erro_temp
                    hdul['DQ'].data = sci_pxdq[ind].astype('int')
                    if kwargs['combine_dithers']:
                        hdul.writeto(os.path.join(output_dir, key + '_psfsub_roll%.0f.fits' % (ind + 1)), output_verify='fix', overwrite=True)
                    else:
                        hdul.writeto(os.path.join(output_dir, key + '_psfsub_dpos%.0f_roll%.0f.fits' % (dpos + 1, ind + 1)), output_verify='fix', overwrite=True)
                    hdul.close()
            
            # Special case with data weighted by PSF mask throughput.
            # sci_mask = np.multiply(np.array(sci_mask).T, np.array(sci_effinttm)).T
            # sci_mask = sci_mask**3
            # sci_mask_sum = np.nansum(sci_mask, axis=0)
            # sci_mask_sum[sci_mask_sum == 0.] = np.nan
            # temp = []
            # for j in range(len(sci_data)):
            #     weights = np.true_divide(sci_mask[j], sci_mask_sum)
            #     temp += [np.multiply(sci_data[j], weights)]
            # temp = np.array(temp)
            # temp = np.nansum(temp, axis=0)
            # sci_data = temp
            
            # Combine rolls.
            sci_data = np.array(sci_data)
            sci_erro = np.array(sci_erro)
            sci_pxdq = np.array(sci_pxdq)
            sci_data = np.nanmedian(sci_data, axis=0)
            nsample = np.sum(np.logical_not(np.isnan(sci_erro)), axis=0)
            sci_erro = np.true_divide(np.sqrt(np.nansum(sci_erro**2, axis=0)), nsample)
            if database.obs[key]['TELESCOP'][ww_sci[0]] == 'JWST' and database.obs[key]['INSTRUME'][ww_sci[0]] == 'NIRCAM':
                sci_pxdq = np.sum(sci_pxdq != 0, axis=0) != 0
            else:
                sci_pxdq = np.sum(sci_pxdq & 1 == 1, axis=0) != 0
            sci_effinttm = np.sum(sci_effinttm)
            
            # Write FITS file.
            hdul = pyfits.open(database.obs[key]['FITSFILE'][ww_sci[0]])
            hdul[0].header['NINTS'] = 1
            hdul[0].header['EFFINTTM'] = sci_effinttm
            hdul['SCI'].data = sci_data
            w = wcs.WCS(hdul['SCI'].header)
            _rotate_wcs_hdr(w, database.obs[key]['ROLL_REF'][ww_sci[0]])
            hdul['SCI'].header['CRPIX1'] = sci_data.shape[1] // 2 + 1
            hdul['SCI'].header['CRPIX2'] = sci_data.shape[0] // 2 + 1
            hdul['SCI'].header['CD1_1'] = w.wcs.cd[0, 0]
            hdul['SCI'].header['CD1_2'] = w.wcs.cd[0, 1]
            hdul['SCI'].header['CD2_1'] = w.wcs.cd[1, 0]
            hdul['SCI'].header['CD2_2'] = w.wcs.cd[1, 1]
            hdul['ERR'].data = sci_erro
            hdul['DQ'].data = sci_pxdq.astype('int')
            if kwargs['combine_dithers']:
                hdul.writeto(os.path.join(output_dir, key + '_psfsub.fits'), output_verify='fix', overwrite=True)
            else:
                hdul.writeto(os.path.join(output_dir, key + '_psfsub_dpos%.0f.fits' % (dpos + 1)), output_verify='fix', overwrite=True)
            hdul.close()
            log.info('--> Average best fit scaling factor (dpos%.0f) = %.2f' % (dpos + 1, np.mean(pps)))
        
        # Save corresponding observations database.
        file = os.path.join(output_dir, key + '.dat')
        database.obs[key].write(file, format='ascii', overwrite=True)
        
        # Compute and save corresponding transmission mask.
        file = os.path.join(output_dir, key + '_psfmask.fits')
        mask = get_transmission(database.obs[key])
        ww_sci = np.where(database.obs[key]['TYPE'] == 'SCI')[0]
        if mask is not None:
            hdul = pyfits.open(database.obs[key]['MASKFILE'][ww_sci[0]])
            hdul[0].data = None
            hdul['SCI'].data = mask
            hdul.writeto(file, output_verify='fix', overwrite=True)
    
    pass
