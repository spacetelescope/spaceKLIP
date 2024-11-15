from __future__ import division
import matplotlib

# =============================================================================
# IMPORTS
# =============================================================================

import os
import pdb
import sys

import astropy.io.fits as pyfits
import numpy as np

import json
import pysiaf
import webbpsf

from scipy.ndimage import shift

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

#### DEPRECATION WARNING ####
log.warning('************* DEPRECATION WARNING *************')
log.warning('****** "make_psfmasks.py" is deprecated! ******')
log.warning('***** You likely do not need to run this! *****')
log.warning('***********************************************')

# =============================================================================
# MAIN
# =============================================================================

# Set parameters.
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

# Loop through apertures and filters.
for apername in crpix_jarron.keys():
    for filt in filter_shifts_jarron.keys():
        if 'FULL_MASK210R' in apername:
            fitsfile = 'jwst_nircam_psfmask_0038.fits'
            xcen, ycen = 711.5, 1505.5
            cval = 1.
        elif 'MASK210R' in apername:
            fitsfile = 'jwst_nircam_psfmask_0049.fits'
            xcen, ycen = 320.5, 320.5
            cval = 0.99088
        elif 'FULL_MASK335R' in apername:
            fitsfile = 'jwst_nircam_psfmask_0020.fits'
            xcen, ycen = 651.5, 1661.5
            cval = 1.
        elif 'MASK335R' in apername:
            fitsfile = 'jwst_nircam_psfmask_0019.fits'
            xcen, ycen = 160.5, 160.5
            cval = 0.99088
        elif 'FULL_MASK430R' in apername:
            fitsfile = 'jwst_nircam_psfmask_0060.fits'
            xcen, ycen = 973.5, 1661.5
            cval = 1.
        elif 'MASK430R' in apername:
            fitsfile = 'jwst_nircam_psfmask_0027.fits'
            xcen, ycen = 160.5, 160.5
            cval = 0.99088

        # Get true mask center from Jarron.
        crpix1_jarron, crpix2_jarron = crpix_jarron[apername]
        print(apername, crpix1_jarron, crpix2_jarron)

        # Get filter shift from Jarron.
        xshift_jarron, yshift_jarron = filter_shifts_jarron[filt]

        # Compute required shift.
        crpix1, crpix2 = crpix1_jarron + xshift_jarron, crpix2_jarron + yshift_jarron
        xoff, yoff = crpix1 - xcen, crpix2 - ycen

        # Read PSF mask.
        maskpath = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'resources/transmissions/' + fitsfile)
        hdul = pyfits.open(maskpath)
        psfmask = hdul['SCI'].data

        # Shift the coronagraphic mask separately with a higher interpolation
        # order.
        xr = np.arange(psfmask.shape[1]) - (xcen - 1.)
        yr = np.arange(psfmask.shape[0]) - (ycen - 1.)
        xx, yy = np.meshgrid(xr, yr)
        rr = np.sqrt(xx**2 + yy**2)
        mask = psfmask.copy()
        mask[rr > 100.] = 0.99088
        mask = shift(mask, (yoff, xoff), order=3, mode='constant', cval=0.99088)

        # Shift PSF mask.
        psfmask = shift(psfmask, (yoff, xoff), order=0, mode='constant', cval=cval)

        # Insert the separately shifted coronagraphic mask into the full PSF
        # mask.
        xr = np.arange(psfmask.shape[1]) - (crpix1 - 1.)
        yr = np.arange(psfmask.shape[0]) - (crpix2 - 1.)
        xx, yy = np.meshgrid(xr, yr)
        rr = np.sqrt(xx**2 + yy**2)
        psfmask[rr < 100.] = mask[rr < 100.]

        # Write PSF mask.
        hdul['SCI'].data = psfmask
        outfile = 'resources/transmissions/' + apername + '_' + filt + '.fits'
        outfile = os.path.join(os.path.split(os.path.abspath(__file__))[0], outfile)
        print(f"Wrote to {outfile}")
        hdul.writeto(outfile, output_verify='fix', overwrite=True)

fitsfiles = ['JWST_MIRI_F1065C_transmission_webbpsf-ext_v2.fits',
             'JWST_MIRI_F1140C_transmission_webbpsf-ext_v2.fits',
             'JWST_MIRI_F1550C_transmission_webbpsf-ext_v2.fits']
for fitsfile in fitsfiles:
    thisfile = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'resources/transmissions/' + fitsfile)
    hdul = pyfits.open(thisfile)
    psfmask = hdul[0].data
    psfmask[psfmask < 0.05] = np.nan
    hdu = pyfits.ImageHDU(psfmask)
    hdu.header['EXTNAME'] = 'SCI'
    hdul.append(hdu)
    hdul.writeto(thisfile, output_verify='fix', overwrite=True)
    print(f"Wrote to resources/transmissions/{fitsfile}")


def get_bar_offset_from_siaf(siaf, filt, channel='LW'):
        """
        Get the PSF reference position with respect to the NRCA5_MASKLWB and
        the NRCA4_MASKSWB subarrays, respectively, from pySIAF.
        
        Parameters
        ----------
        siaf : pysiaf.Siaf
            NIRCam SIAF aperture.
        filt : str
            Name of the NIRCam filter.
        channel : str, optional
            Long wavelength (LW) or short wavelength (SW) channel. The default
            is 'LW'.
        
        Returns
        -------
        bar_offset : float
            Offset of the PSF reference position with respect to the
            NRCA5_MASKLWB and the NRCA4_MASKSWB subarrays, respectively, in
            arcseconds.
        
        """
        
        if channel == 'SW':
            refapername = 'NRCA4_MASKSWB'
            apername = 'NRCA4_MASKSWB_' + filt.upper()
        else:  # otherwise default to LW channel
            refapername = 'NRCA5_MASKLWB'
            apername = 'NRCA5_MASKLWB_' + filt.upper()
        offset_arcsec = np.sqrt((siaf.apertures[refapername].V2Ref - siaf.apertures[apername].V2Ref)**2 + (siaf.apertures[refapername].V3Ref - siaf.apertures[apername].V3Ref)**2)
        sign = np.sign(siaf.apertures[refapername].V2Ref - siaf.apertures[apername].V2Ref)
        
        return sign * offset_arcsec


# Get bar mask offsets.
siaf = pysiaf.Siaf('NIRCAM')
offset_swb = {filt: get_bar_offset_from_siaf(siaf, filt, channel='SW')
              for filt in ['F182M', 'F187N', 'F200W', 'F210M', 'F212N', 'narrow']}  # arcsec
offset_lwb = {filt: get_bar_offset_from_siaf(siaf, filt, channel='LW')
              for filt in ['F250M', 'F277W', 'F300M', 'F335M', 'F356W', 'F360M', 'F410M', 'F430M', 'F444W', 'F460M', 'F480M', 'narrow']}  # arcsec

# Initialize WebbPSF instruments.
nircam = webbpsf.NIRCam()

# Loop through apertures and filters.
fitsfile = 'jwst_nircam_psfmask_0036.fits'
xcen, ycen = 320.5, 320.5
for filt in offset_swb:
    
    # Get reference pixel position.
    apername = 'NRCA4_MASKSWB_' + filt.upper()
    apsiaf = siaf[apername]
    crpix1, crpix2 = (apsiaf.XSciRef, apsiaf.YSciRef)
    
    # Compute required shift.
    pxsc = nircam._pixelscale_short  # arcsec
    xoff, yoff = crpix1 - xcen - offset_swb[filt] / pxsc, crpix2 - ycen
    
    # Read PSF mask.
    thisfile = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'resources/transmissions/' + fitsfile)
    hdul = pyfits.open(thisfile)
    psfmask = hdul['SCI'].data
    
    # Shift the coronagraphic mask separately with a higher interpolation
    # order.
    xr = np.arange(psfmask.shape[1]) - (xcen - 1.)
    yr = np.arange(psfmask.shape[0]) - (ycen - 1.)
    xx, yy = np.meshgrid(xr, yr)
    psfmask[np.abs(yy) > 100.] = 1.
    psfmask = shift(psfmask, (yoff, xoff), order=3, mode='constant', cval=1.)
    
    # Write PSF mask.
    hdul['SCI'].data = psfmask
    outfile = 'resources/transmissions/' + apername + '_' + filt.upper() + '.fits'
    outfile = os.path.join(os.path.split(os.path.abspath(__file__))[0], outfile)
    hdul.writeto(outfile, output_verify='fix', overwrite=True)
    print(f"Wrote to {outfile}")

# Loop through apertures and filters.
fitsfile = 'jwst_nircam_psfmask_0003.fits'
xcen, ycen = 160.5, 160.5
for filt in offset_lwb:
    
    # Get reference pixel position.
    apername = 'NRCA5_MASKLWB_' + filt.upper()
    apsiaf = siaf[apername]
    crpix1, crpix2 = (apsiaf.XSciRef, apsiaf.YSciRef)
    
    # Compute required shift.
    pxsc = nircam._pixelscale_long  # arcsec
    xoff, yoff = crpix1 - xcen - offset_lwb[filt] / pxsc, crpix2 - ycen
    
    # Read PSF mask.
    thisfile = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'resources/transmissions/' + fitsfile)
    hdul = pyfits.open(thisfile)
    psfmask = hdul['SCI'].data
    psfmask[:, -1] = 1
    
    # Shift the coronagraphic mask separately with a higher interpolation
    # order.
    xr = np.arange(psfmask.shape[1]) - (xcen - 1.)
    yr = np.arange(psfmask.shape[0]) - (ycen - 1.)
    xx, yy = np.meshgrid(xr, yr)
    psfmask[np.abs(yy) > 50.] = 1.
    psfmask = shift(psfmask, (yoff, xoff), order=3, mode='constant', cval=1.)
    
    # Write PSF mask.
    hdul['SCI'].data = psfmask
    outfile = 'resources/transmissions/' + apername + '_' + filt.upper() + '.fits'
    outfile = os.path.join(os.path.split(os.path.abspath(__file__))[0], outfile)
    hdul.writeto(outfile, output_verify='fix', overwrite=True)
