from __future__ import division

import matplotlib

# =============================================================================
# IMPORTS
# =============================================================================

import os
import pdb
import sys
import io
import stpipe

import astropy.io.fits as pyfits
import numpy as np

import importlib
import scipy.ndimage.interpolation as sinterp

from scipy.integrate import simpson
from scipy.ndimage import fourier_shift, gaussian_filter
from scipy.ndimage import shift as spline_shift

import pysiaf
from webbpsf_ext.imreg_tools import get_coron_apname as nircam_apname
from webbpsf_ext.image_manip import expand_mask

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# =============================================================================
# MAIN
# =============================================================================

def get_nrcmask_from_apname(apname):
    """Get mask name from aperture name
    
    The aperture name is of the form:
    NRC[A/B][1-5]_[FULL]_[MASK]_[FILTER]
    where MASK is the name of the coronagraphic mask used.

    For target acquisition apertures the mask name can be
    prependend with "TA" (eg., TAMASK335R).

    Return 'NONE' if MASK not in input aperture name.

    Parameters
    ----------
    apname : str
        String aperture name as described above

    Returns
    -------
    image_mask : str
        String for image mask

    """

    if 'MASK' not in apname:
        return 'NONE'

    ap_str_arr = apname.split('_')
    for s in ap_str_arr:
        if 'MASK' in s:
            image_mask = s
            break

    # Special case for TA apertures
    if 'TA' in image_mask:
        # return 'NONE'
        # Remove TA from mask name
        image_mask = image_mask.replace('TA', '')

        # Remove FS from mask name
        if 'FS' in image_mask:
            image_mask = image_mask.replace('FS', '')

        # Remove trailing S or L from LWB and SWB TA apertures
        if ('WB' in image_mask) and (image_mask[-1]=='S' or image_mask[-1]=='L'):
            image_mask = image_mask[:-1]

    return image_mask

def read_obs(fitsfile,
             return_var=False):
    """
    Read an observation from a FITS file.
    
    Parameters
    ----------
    fitsfile : path
        Path of input FITS file.
    return_var : bool, optional
        Return VAR_POISSON and VAR_RNOISE arrays? The default is False.
    
    Returns
    -------
    data : 3D-array
        'SCI' extension data.
    erro : 3D-array
        'ERR' extension data.
    pxdq : 3D-array
        'DQ' extension data.
    head_pri : FITS header
        Primary FITS header.
    head_sci : FITS header
        'SCI' extension FITS header.
    is2d : bool
        Is the original data 2D?
    imshifts : 2D-array
        Array of shape (nints, 2) containing the total shifts applied to the
        frames. None if not available.
    maskoffs : 2D-array
        Array of shape (nints, 2) containing the offsets between the star and
        coronagraphic mask position. None if not available.
    var_poisson : 3D-array, optional
        'VAR_POISSON' extension data.
    var_rnoise : 3D-array, optional
        'VAR_RNOISE' extension data.
    
    """
    
    # Read FITS file.
    hdul = pyfits.open(fitsfile)
    data = hdul['SCI'].data
    try:
        erro = hdul['ERR'].data
    except:
        erro = np.sqrt(data)
    try:
        pxdq = hdul['DQ'].data
    except:
        pxdq = np.zeros_like(data).astype('int')
    head_pri = hdul[0].header
    head_sci = hdul['SCI'].header
    is2d = False
    if data.ndim == 2:
        data = data[np.newaxis, :]
        erro = erro[np.newaxis, :]
        pxdq = pxdq[np.newaxis, :]
        is2d = True
    if data.ndim != 3:
        raise UserWarning('Requires 2D/3D data cube')
    try:
        imshifts = hdul['IMSHIFTS'].data
    except KeyError:
        imshifts = None
    try:
        maskoffs = hdul['MASKOFFS'].data
    except KeyError:
        maskoffs = None
    if return_var:
        var_poisson = hdul['VAR_POISSON'].data
        var_rnoise = hdul['VAR_RNOISE'].data
    hdul.close()
    
    if return_var:
        return data, erro, pxdq, head_pri, head_sci, is2d, imshifts, maskoffs, var_poisson, var_rnoise
    else:
        return data, erro, pxdq, head_pri, head_sci, is2d, imshifts, maskoffs

def write_obs(fitsfile,
              output_dir,
              data,
              erro,
              pxdq,
              head_pri,
              head_sci,
              is2d,
              imshifts=None,
              maskoffs=None,
              var_poisson=None,
              var_rnoise=None):
    """
    Write an observation to a FITS file.
    
    Parameters
    ----------
    fitsfile : path
        Path of input FITS file.
    output_dir : path
        Directory where the output FITS file shall be saved.
    data : 3D-array
        'SCI' extension data.
    erro : 3D-array
        'ERR' extension data.
    pxdq : 3D-array
        'DQ' extension data.
    head_pri : FITS header
        Primary FITS header.
    head_sci : FITS header
        'SCI' extension FITS header.
    is2d : bool
        Is the original data 2D?
    imshifts : 2D-array, optional
        Array of shape (nints, 2) containing the total shifts applied to the
        frames. The default is None.
    maskoffs : 2D-array, optional
        Array of shape (nints, 2) containing the offsets between the star and
        coronagraphic mask position. The default is None.
    var_poisson : 3D-array, optional
        'VAR_POISSON' extension data. The default is None.
    var_rnoise : 3D-array, optional
        'VAR_RNOISE' extension data. The default is None.
    
    Returns
    -------
    fitsfile : path
        Path of output FITS file.
    
    """
    
    # Write FITS file.
    hdul = pyfits.open(fitsfile)
    if is2d:
        hdul['SCI'].data = data[0]
        hdul['ERR'].data = erro[0]
        hdul['DQ'].data = pxdq[0]
    else:
        hdul['SCI'].data = data
        hdul['ERR'].data = erro
        hdul['DQ'].data = pxdq
    hdul[0].header = head_pri
    hdul['SCI'].header = head_sci
    if imshifts is not None:
        try:
            hdul['IMSHIFTS'].data = imshifts
        except KeyError:
            hdu = pyfits.ImageHDU(imshifts, name='IMSHIFTS')
            hdul.append(hdu)
    if maskoffs is not None:
        try:
            hdul['MASKOFFS'].data = maskoffs
        except KeyError:
            hdu = pyfits.ImageHDU(maskoffs, name='MASKOFFS')
            hdul.append(hdu)
    if var_poisson is not None:
        hdul['VAR_POISSON'].data = var_poisson
    if var_rnoise is not None:
        hdul['VAR_RNOISE'].data = var_rnoise
    fitsfile = os.path.join(output_dir, os.path.split(fitsfile)[1])
    hdul.writeto(fitsfile, output_verify='fix', overwrite=True)
    hdul.close()
    
    return fitsfile

def read_msk(maskfile):
    """
    Read a PSF mask from a FITS file.
    
    Parameters
    ----------
    maskfile : path
        Path of input FITS file.
    
    Returns
    -------
    mask : 2D-array
        PSF mask. None if not available.
    
    """
    
    # Read FITS file.
    if maskfile != 'NONE':
        hdul = pyfits.open(maskfile)
        mask = hdul['SCI'].data
        hdul.close()
    else:
        mask = None
    
    return mask


def write_msk(maskfile,
              mask,
              fitsfile):
    """
    Write a PSF mask to a FITS file.
    
    Parameters
    ----------
    maskfile : path
        Path of input FITS file.
    mask : 2D-array
        PSF mask. None if not available.
    fitsfile : path
        Path of output FITS file (to save the PSF mask in the same directory).
    
    Returns
    -------
    maskfile : path
        Path of output FITS file.
    
    """
    
    # Write FITS file.
    if mask is not None:
        hdul = pyfits.open(maskfile)
        hdul['SCI'].data = mask
        maskfile = fitsfile.replace('.fits', '_psfmask.fits')
        hdul.writeto(maskfile, output_verify='fix', overwrite=True)
        hdul.close()
    else:
        maskfile = 'NONE'
    
    return maskfile

def read_red(fitsfile):
    """
    Read a reduction from a FITS file.
    
    Parameters
    ----------
    fitsfile : path
        Path of input FITS file.
    
    Returns
    -------
    data : 3D-array
        'SCI' extension data.
    head_pri : FITS header
        Primary FITS header.
    head_sci : FITS header
        'SCI' extension FITS header.
    is2d : bool
        Is the original data 2D?
    
    """
    
    # Read FITS file.
    hdul = pyfits.open(fitsfile)
    data = hdul[0].data
    if data is None:
        try:
            data = hdul['SCI'].data
        except:
            raise UserWarning('Could not find any data')
    head_pri = hdul[0].header
    try:
        head_sci = hdul['SCI'].header
    except:
        head_sci = None
    hdul.close()
    is2d = False
    if data.ndim == 2:
        data = data[np.newaxis, :]
        is2d = True
    if data.ndim != 3:
        raise UserWarning('Requires 2D/3D data cube')
    
    return data, head_pri, head_sci, is2d

def write_fitpsf_images(fitpsf,
                        fitsfile,
                        row):
    """
    Write a best fit FM PSF to a FITS file.
    
    Parameters
    ----------
    fitpsf : pyklip.fitpsf
        PyKLIP PSF fitting object whose best fit FM PSF shall be saved.
    fitsfile : path
        Path of output FITS file.
    row : astropy.table.Row
        Astropy table row of the companion to be saved to the FITS file.
    
    Returns
    -------
    None.
    
    """
    
    # Make best fit FM PSF.
    dx = fitpsf.fit_x.bestfit - fitpsf.data_stamp_x_center
    dy = fitpsf.fit_y.bestfit - fitpsf.data_stamp_y_center
    fm_bestfit = fitpsf.fit_flux.bestfit * sinterp.shift(fitpsf.fm_stamp, [dy, dx])
    if fitpsf.padding > 0:
        fm_bestfit = fm_bestfit[fitpsf.padding:-fitpsf.padding, fitpsf.padding:-fitpsf.padding]
    
    # Make residual image.
    residual_image = fitpsf.data_stamp - fm_bestfit
    snr = np.nanmax(fm_bestfit) / np.nanstd(residual_image)
    row['SNR'] = snr
    
    # Write FITS file.
    pri = pyfits.PrimaryHDU()
    for key in row.keys():
        if key in ['FLUX_SI', 'FLUX_SI_ERR', 'LN(Z/Z0)', 'TP_CORONMSK', 'TP_COMSUBST', 'SIGMA_X_ERROR', 'SIGMA_Y_ERROR',
                   'THETA_ERROR'] and np.isnan(row[key]):
            pri.header[key] = 'NONE'
        else:
            pri.header[key] = row[key]
    res = pyfits.ImageHDU(residual_image, name='RES')
    sci = pyfits.ImageHDU(fitpsf.data_stamp, name='SCI')
    mod = pyfits.ImageHDU(fm_bestfit, name='MOD')
    hdul = pyfits.HDUList([pri, res, sci, mod])
    hdul.writeto(fitsfile, output_verify='fix', overwrite=True)
    
    pass

def crop_image(image,
               xycen,
               npix,
               return_indices=False):
    """
    Crop an image.
    
    Parameters
    ----------
    image : 2D-array
        Input image to be cropped.
    xycen : tuple of float
        Center around which the image shall be cropped. Will be rounded.
    npix : float
        Size of the cropped image. Will be rounded.
    return_indices : bool, optional
        If True, returns the x- and y-indices of the cropped image in the
        coordinate frame of the input image. The default is False.
    
    Returns
    -------
    imsub : 2D-array
        The cropped image.
    xsub_indarr : 1D-array, optional
        The x-indices of the cropped image in the coordinate frame of the
        input image.
    ysub_indarr : 1D-array, optional
        The y-indices of the cropped image in the coordinate frame of the
        input image.
    
    """
    
    # Compute pixel coordinates.
    xc, yc = xycen
    x1 = int(xc - npix / 2. + 0.5)
    x2 = x1 + npix
    y1 = int(yc - npix / 2. + 0.5)
    y2 = y1 + npix
    
    # Crop image.
    imsub = image[y1:y2, x1:x2]
    if return_indices:
        xsub_indarr = np.arange(x1, x2).astype('int')
        ysub_indarr = np.arange(y1, y2).astype('int')
        return imsub, xsub_indarr, ysub_indarr
    else:
        return imsub

def imshift(image,
            shift,
            pad=False,
            cval=0.,
            method='fourier',
            kwargs={}):
    """
    Shift an image.
    
    Parameters
    ----------
    image : 2D-array
        Input image to be shifted.
    shift : 1D-array
        X- and y-shift to be applied.
    pad : bool, optional
        Pad the image before shifting it? Otherwise, it will wrap around
        the edges. The default is True.
    cval : float, optional
        Fill value for the padded pixels. The default is 0.
    method : 'fourier' or 'spline' (not recommended), optional
        Method for shifting the frames. The default is 'fourier'.
    kwargs : dict, optional
        Keyword arguments for the scipy.ndimage.shift routine. The default
        is {}.
    
    Returns
    -------
    imsft : 2D-array
        The shifted image.
    
    """
    
    if pad:
        
        # Pad image.
        sy, sx = image.shape
        xshift, yshift = shift
        padx = np.abs(int(xshift)) + 5
        pady = np.abs(int(yshift)) + 5
        impad = np.pad(image, ((pady, pady), (padx, padx)), mode='constant', constant_values=cval)
        
        # Shift image.
        if method == 'fourier':
            imsft = np.fft.ifftn(fourier_shift(np.fft.fftn(impad), shift[::-1])).real
        elif method == 'spline':
            imsft = spline_shift(impad, shift[::-1], **kwargs)
        else:
            raise UserWarning('Image shift method "' + method + '" is not known')
        
        # Crop image to original size.
        return imsft[pady:pady + sy, padx:padx + sx]
    else:
        if method == 'fourier':
            return np.fft.ifftn(fourier_shift(np.fft.fftn(image), shift[::-1])).real
        elif method == 'spline':
            return spline_shift(image, shift[::-1], **kwargs)
        else:
            raise UserWarning('Image shift method "' + method + '" is not known')

def alignlsq(shift,
             image,
             ref_image,
             mask=None,
             method='fourier',
             kwargs={}):
    """
    Align an image to a reference image using a Fourier shift and subtract
    method.
    
    Parameters
    ----------
    shift : 1D-array
        X- and y-shift and scaling factor to be applied.
    image : 2D-array
        Input image to be aligned to a reference image.
    ref_image : 2D-array
        Reference image.
    mask : 2D-array, optional
        Weights to be applied to the input and reference images. The
        default is None.
    method : 'fourier' or 'spline' (not recommended), optional
        Method for shifting the frames. The default is 'fourier'.
    kwargs : dict, optional
        Keyword arguments for the scipy.ndimage.shift routine. The default
        is {}.
    
    Returns
    -------
    imres : 1D-array
        Residual image collapsed into one dimension.
    
    """
    
    if mask is None:
        return (ref_image - shift[2] * imshift(image, shift[:2], method=method, kwargs=kwargs)).ravel()
    else:
        return ((ref_image - shift[2] * imshift(image, shift[:2], method=method, kwargs=kwargs)) * mask).ravel()

def recenterlsq(shift,
                image,
                method='fourier',
                kwargs={}):
    """
    Center a PSF on its nearest pixel by maximizing its peak count.
    
    Parameters
    ----------
    shift : 1D-array
        X- and y-shift to be applied.
    image : 2D-array
        Input image to be recentered.
    method : 'fourier' or 'spline' (not recommended), optional
        Method for shifting the frames. The default is 'fourier'.
    kwargs : dict, optional
        Keyword arguments for the scipy.ndimage.shift routine. The default
        is {}.
    
    Returns
    -------
    invpeak : float
        Inverse of the PSF's peak count.
    
    """
    
    return 1. / np.nanmax(imshift(image, shift, method=method, kwargs=kwargs))

def subtractlsq(shift,
                image,
                ref_image,
                mask=None):
    """
    Scale and subtract a reference from a science image.
    
    Parameters
    ----------
    shift : 1D-array
        Scaling factor between the science and the reference PSF.
    image : 2D-array
        Input image to be reference PSF-subtracted.
    ref_image : 2D-array
        Reference image.
    mask : 2D-array, optional
        Mask to be applied to the input and reference images. The default is
        None.
    
    Returns
    -------
    imres : 1D-array
        Residual image collapsed into one dimension.
    
    """
    
    res = image - shift[0] * ref_image
    res = res - gaussian_filter(res, 5)
    if mask is None:
        return res.ravel()
    else:
        return res[mask]

def _get_tp_comsubst(instrume,
                    subarray,
                    filt):
    """
    Get the COM substrate transmission averaged over the respective filter
    profile.

    *** Deprecated - use `get_tp_comsubst` instead. ***
    
    Parameters
    ----------
    instrume : 'NIRCAM', 'NIRISS', or 'MIRI'
        JWST instrument in use.
    subarray : str
        JWST subarray in use.
    filt : str
        JWST filter in use.
    
    Returns
    -------
    tp_comsubst : float
        COM substrate transmission averaged over the respective filter profile
    
    """
    
    log.warning('This function is deprecated. Use `get_tp_comsubst` instead.')

    # Default return.
    tp_comsubst = 1.
    
    # If NIRCam.
    if instrume == 'NIRCAM':
        
        # If coronagraphy subarray.
        if '210R' in subarray or '335R' in subarray or '430R' in subarray or 'SWB' in subarray or 'LWB' in subarray:
            
            # Read bandpass.
            try:
                with importlib.resources.open_text(f'spaceKLIP.resources.PCEs.{instrume}', f'{filt}.txt') as bandpass_file:
                    bandpass_data = np.genfromtxt(bandpass_file).transpose()
                    bandpass_wave = bandpass_data[0]  # micron
                    bandpass_throughput = bandpass_data[1]
            except FileNotFoundError:
                log.error('--> Filter ' + filt + ' not found for instrument ' + instrume)
            
            # Read COM substrate transmission.
            with importlib.resources.open_text(f'spaceKLIP.resources.transmissions', f'ModA_COM_Substrate_Transmission_20151028_JKrist.dat') as comsubst_file:
                comsubst_data = np.genfromtxt(comsubst_file).transpose()
                comsubst_wave = comsubst_data[0][1:]  # micron
                comsubst_throughput = comsubst_data[1][1:]
            
            # Compute COM substrate transmission averaged over the respective
            # filter profile.
            bandpass_throughput = np.interp(comsubst_wave, bandpass_wave, bandpass_throughput)
            int_tp_bandpass = simpson(bandpass_throughput, comsubst_wave)
            int_tp_bandpass_comsubst = simpson(bandpass_throughput * comsubst_throughput, comsubst_wave)
            tp_comsubst = int_tp_bandpass_comsubst / int_tp_bandpass
    
    # Return.
    return tp_comsubst

def write_starfile(starfile,  
                   new_starfile_path,
                   new_header=None):
    """
    Save stellar spectrum file to a different location, and insert
    a header to the start if needed. 
    
    Parameters
    ----------
    starfile : str
        Path to original stellar spectrum file.
    new_starfile_path : str
        Path to new stellar spectrum file.
    new_header : str
        Header to be inserted. 
    
    Returns
    -------
    None
    
    """ 
    if not os.path.exists(starfile):
        raise FileNotFoundError("The specified starfile does not exist.")
    
    with open(starfile, 'r') as orig_starfile:
        text=orig_starfile.read()
        with open(new_starfile_path, 'w') as new_starfile:
            if new_header is None:
                new_starfile.write(text)
            else:
                new_starfile.write(new_header+text)

def set_surrounded_pixels(array, user_value=np.nan):
    """
    Identifies pixels in a 2D array surrounded by NaN values 
    on all eight sides and sets them to a user-defined value.

    Parameters
    ----------
    array : numpy.ndarray
        2D array containing numeric values and NaNs.
    user_value : float or any valid value type, optional
        Value to set for pixels surrounded by NaNs on all eight sides. Defaults to NaN.

    Returns
    -------
    numpy.ndarray
        The input array with pixels surrounded by NaNs on all eight sides set to the user-defined value.
    """
    nan_mask = np.isnan(array)
    surrounded_pixels = (
        ~nan_mask[1:-1, 1:-1] &
        nan_mask[:-2, :-2] & nan_mask[:-2, 1:-1] & nan_mask[:-2, 2:] &
        nan_mask[1:-1, :-2] & nan_mask[1:-1, 2:] &
        nan_mask[2:, :-2] & nan_mask[2:, 1:-1] & nan_mask[2:, 2:]
    )
    
    array[1:-1, 1:-1][surrounded_pixels] = user_value
    return array

def get_tp_comsubst(instrume,
                    subarray,
                    filt):
    """
    Get the COM substrate transmission averaged over the respective filter
    profile.

    TODO: Spot check the COM throughput using photometric calibration data,
    assuming there are stellar offsets on and off the COM substrate.
    
    Parameters
    ----------
    instrume : 'NIRCAM', 'NIRISS', or 'MIRI'
        JWST instrument in use.
    subarray : str
        JWST subarray in use.
    filt : str
        JWST filter in use.
    
    Returns
    -------
    tp_comsubst : float
        COM substrate transmission averaged over the respective filter profile
    
    """
    
    from webbpsf_ext.bandpasses import nircam_filter, nircam_com_th

    # Default return.
    tp_comsubst = 1.
    
    # If NIRCam.
    instrume = instrume.upper()
    if instrume == 'NIRCAM':
        
        # If coronagraphy subarray.
        if '210R' in subarray or '335R' in subarray or '430R' in subarray or 'SWB' in subarray or 'LWB' in subarray:
            
            # Read bandpass.
            try:
                bp = nircam_filter(filt)
                bandpass_wave = bp.wave / 1e4  # micron
                bandpass_throughput = bp.throughput
            except FileNotFoundError:
                log.error('--> Filter ' + filt + ' not found for instrument ' + instrume)
            
            # Read COM substrate transmission interpolated at bandpass wavelengths.
            comsubst_throughput = nircam_com_th(bandpass_wave)

            # Compute weighted average of COM substrate transmission.
            tp_comsubst = np.average(comsubst_throughput, weights=bandpass_throughput)
    
    # Return.
    return tp_comsubst

def get_filter_info(instrument, timeout=1, do_svo=True, return_more=False):
    """ Load filter information from the SVO Filter Profile Service or webbpsf

    Load NIRCam, NIRISS, and MIRI filters from the SVO Filter Profile Service.
    http://svo2.cab.inta-csic.es/theory/fps/

    If timeout to server, then use local copy of filter list and load through webbpsf.

    Parameters
    ----------
    instrument : str
        Name of instrument to load filter list for. 
        Must be one of 'NIRCam', 'NIRISS', or 'MIRI'.
    timeout : float
        Timeout in seconds for connection to SVO Filter Profile Service.
    do_svo : bool
        If True, try to load filter list from SVO Filter Profile Service. 
        If False, use webbpsf without first check web server.
    return_more : bool
        If True, also return `do_svo` variable, whether SVO was used or not.
    """

    from astroquery.svo_fps import SvoFps
    import webbpsf

    iname_upper = instrument.upper()

    # Try to get filter list from SVO
    if do_svo:
        try:
            filter_list = SvoFps.get_filter_list(facility='JWST', instrument=iname_upper, timeout=timeout)
        except:
            log.warning('Using SVO Filter Profile Service timed out. Using WebbPSF instead.')
            do_svo = False

    # If unsuccessful, use webbpsf to get filter list
    if not do_svo:
        inst_func = {
            'NIRCAM': webbpsf.NIRCam,
            'NIRISS': webbpsf.NIRISS,
            'MIRI'  : webbpsf.MIRI,
        }
        inst = inst_func[iname_upper]()
        filter_list = inst.filter_list 

    wave, weff = ({}, {})
    if do_svo:
        for i in range(len(filter_list)):
            name = filter_list['filterID'][i]
            name = name[name.rfind('.') + 1:]
            wave[name] = filter_list['WavelengthMean'][i] / 1e4  # micron
            weff[name] = filter_list['WidthEff'][i] / 1e4  # micron
    else:
        for filt in filter_list:
            bp = inst._get_synphot_bandpass(filt)
            wave[filt] = bp.avgwave().to_value('micron')
            weff[filt] = bp.equivwidth().to_value('micron')

    if return_more:
        return wave, weff, do_svo
    else:
        return wave, weff

def cube_fit(tarr, data, sat_vals, sat_frac=0.95, bias=None, 
             deg=1, bpmask_arr=None, fit_zero=False, verbose=False,
             use_legendre=False, lxmap=None, return_lxmap=False,
             return_chired=False):
    """Fit unsaturated data and return coefficients"""
        
    from webbpsf_ext.maths import jl_poly_fit, jl_poly

    nz, ny, nx = data.shape
    
    # Subtract bias?
    imarr = data if bias is None else data - bias
        
    # Array of masked pixels (saturated)
    mask_good = imarr < sat_frac*sat_vals
    if bpmask_arr is not None:
        mask_good = mask_good & ~bpmask_arr
    
    # Reshape for all pixels in single dimension
    imarr = imarr.reshape([nz, -1])
    mask_good = mask_good.reshape([nz, -1])

    # Initial 
    cf = np.zeros([deg+1, nx*ny])
    if return_lxmap:
        lx_min = np.zeros([nx*ny])
        lx_max = np.zeros([nx*ny])
    if return_chired:
        chired = np.zeros([nx*ny])

    # For each 
    npix_sum = 0
    i0 = 0 if fit_zero else 1
    for i in np.arange(i0,nz)[::-1]:
        ind = (cf[1] == 0) & (mask_good[i])
        npix = np.sum(ind)
        npix_sum += npix
        
        if verbose:
            print(i+1,npix,npix_sum, 'Remaining: {}'.format(nx*ny-npix_sum))
            
        if npix>0:
            if fit_zero:
                x = np.concatenate(([0], tarr[0:i+1]))
                y = np.concatenate((np.zeros([1, np.sum(ind)]), imarr[0:i+1,ind]), axis=0)
            else:
                x, y = (tarr[0:i+1], imarr[0:i+1,ind])

            if return_lxmap:
                lx_min[ind] = np.min(x) if lxmap is None else lxmap[0]
                lx_max[ind] = np.max(x) if lxmap is None else lxmap[1]
                
            # Fit line if too few points relative to polynomial degree
            if len(x) <= deg+1:
                cf[0:2,ind] = jl_poly_fit(x,y, deg=1, use_legendre=use_legendre, lxmap=lxmap)
            else:
                cf[:,ind] = jl_poly_fit(x,y, deg=deg, use_legendre=use_legendre, lxmap=lxmap)

            # Get reduced chi-sqr metric for poorly fit data
            if return_chired:
                yfit = jl_poly(x, cf[:,ind])
                deg_chi = 1 if len(x)<=deg+1 else deg
                dof = y.shape[0] - deg_chi
                chired[ind] = chisqr_red(y, yfit=yfit, dof=dof)

    imarr = imarr.reshape([nz,ny,nx])
    mask_good = mask_good.reshape([nz,ny,nx])
    
    cf = cf.reshape([deg+1,ny,nx])
    if return_lxmap:
        lxmap_arr = np.array([lx_min, lx_max]).reshape([2,ny,nx])
        if return_chired:
            chired = chired.reshape([ny,nx])
            return cf, lxmap_arr, chired
        else:
            return cf, lxmap_arr
    else:
        if return_chired:
            chired = chired.reshape([ny,nx])
            return cf, chired
        else:
            return cf
        
def chisqr_red(yvals, yfit=None, err=None, dof=None,
               err_func=np.std):
    """ Calculate reduced chi square metric
    
    If yfit is None, then yvals assumed to be residuals.
    In this case, `err` should be specified.
    
    Parameters
    ==========
    yvals : ndarray
        Sampled values.
    yfit : ndarray
        Model fit corresponding to `yvals`.
    dof : int
        Number of degrees of freedom (nvals - nparams - 1).
    err : ndarray or float
        Uncertainties associated with `yvals`. If not specified,
        then use yvals point-to-point differences to estimate
        a single value for the uncertainty.
    err_func : func
        Error function uses to estimate `err`.
    """
    
    if (yfit is None) and (err is None):
        print("Both yfit and err cannot be set to None.")
        return
    
    diff = yvals if yfit is None else yvals - yfit
    
    sh_orig = diff.shape
    ndim = len(sh_orig)
    if ndim==1:
        if err is None:
            err = err_func(yvals[1:] - yvals[0:-1]) / np.sqrt(2)
        dev = diff / err
        chi_tot = np.sum(dev**2)
        dof = len(chi_tot) if dof is None else dof
        chi_red = chi_tot / dof
        return chi_red
    
    # Convert to 2D array
    if ndim==3:
        sh_new = [sh_orig[0], -1]
        diff = diff.reshape(sh_new)
        yvals = yvals.reshape(sh_new)
        
    # Calculate errors for each element
    if err is None:
        err_arr = np.array([yvals[i+1] - yvals[i] for i in range(sh_orig[0]-1)])
        err = err_func(err_arr, axis=0) / np.sqrt(2)
        del err_arr
    else:
        err = err.reshape(diff.shape)
    # Get reduced chi sqr for each element
    dof = sh_orig[0] if dof is None else dof
    chi_red = np.sum((diff / err)**2, axis=0) / dof
    
    if ndim==3:
        chi_red = chi_red.reshape(sh_orig[-2:])
        
    return chi_red

def cube_outlier_detection(data, sigma_cut=10, nint_min=10):
    """Get outlier pixels in a cube model (e.g., rateints or calints)
    
    Parameters
    ----------
    data : ndarray
        Data array to use for outlier detection.
        Must be a cube with shape (nint, ny, nx).

    Keyword Args
    ------------
    sigma_cut : float
        Sigma cut for outlier detection.
        Default is 5.
    nint_min : int
        Minimum number of integrations required for outlier detection.
        Default is 5.

    Returns
    -------
    Mask of bad pixels with same shape as input cube.
    """

    from webbpsf_ext import robust

    # Get bad pixels
    ndim = len(data.shape)
    if ndim < 3:
        log.warning(f'Skipping rateints outlier flagging. Only {ndim} dimensions.')
        return np.zeros_like(data, dtype=bool)
    
    nint = data.shape[0]
    if nint < nint_min:
        log.warning(f'Skipping rateints outlier flagging. Only {nint} INTS.')
        return np.zeros_like(data, dtype=bool)

    # Get outliers
    indgood = robust.mean(data, Cut=sigma_cut, axis=0, return_mask=True)
    indbad = ~indgood

    return indbad

def bg_minimize(par,X,Y,bgmaskfile):
    """Simple minimisation function for Godoy background subtraction
    
    Parameters
    ----------
    par : int
        Variable to scale background array
    X : ndarray
        Science / reference image
    Y : ndarray
        Background image
    bgmaskfile : str
        File which provides a mask to select which pixels
        to compare during minimisation

    Returns
    -------
    Sum of the squares of the residuals between images X and Y. 
    """
    mask = pyfits.getdata(bgmaskfile)
    indices = np.where(mask == 1)
    X0 = X[indices]
    Y0 = Y[indices]
    Z0 = X0 - Y0*par/100
    return np.nansum(np.sqrt(Z0**2))

def interpret_dq_value(dq_value):
    """Interpret DQ value using DQ definition

    Parameters
    ----------
    dq_value : int
        DQ value to interpret.

    Returns
    -------
    str
        Interpretation of DQ value.
    """

    from stdatamodels.jwst.datamodels.dqflags import pixel, dqflags_to_mnemonics

    if dq_value == 0:
        return {'GOOD'}
    return dqflags_to_mnemonics(dq_value, pixel)

def gaussian_kernel(sigma_x=1, sigma_y=1, theta_degrees=0, n=6):
    """
    Generates a 2D Gaussian kernel with specified standard deviations and rotation.

    Parameters:
    sigma_x (float): Standard deviation of the Gaussian in the x direction.
    sigma_y (float): Standard deviation of the Gaussian in the y direction.
    theta_degrees (float): Rotation angle of the Gaussian kernel in degrees.

    Returns:
    numpy.ndarray: The generated Gaussian kernel.
    """
    # Ensure kernel size is at least 3x3 and odd
    kernel_size_x = max(3, int(n * sigma_x + 1) | 1)  # Ensure odd size
    kernel_size_y = max(3, int(n * sigma_y + 1) | 1)  # Ensure odd size

    # Convert theta from degrees to radians
    theta = np.deg2rad(theta_degrees)

    # Create coordinate grids
    x = np.linspace(-kernel_size_x // 2, kernel_size_x // 2, kernel_size_x)
    y = np.linspace(-kernel_size_y // 2, kernel_size_y // 2, kernel_size_y)
    x, y = np.meshgrid(x, y)

    # Rotate the coordinates
    x_rot = x * np.cos(theta) + y * np.sin(theta)
    y_rot = -x * np.sin(theta) + y * np.cos(theta)

    kernel = np.exp(-(x_rot ** 2 / (2 * sigma_x ** 2) + y_rot ** 2 / (2 * sigma_y ** 2)))
    kernel /= kernel.sum()
    return kernel

def get_dqmask(dqarr, bitvalues):
    """Get DQ mask from DQ array
    
    Given some DQ array and a list of bit values, return a mask
    for the pixels that have any of the specified bit values.

    Parameters
    ----------
    dqarr : ndarray
        DQ array. Either 2D or 3D.
    bitvalues : list
        List of bit values to use for DQ mask. 
        These values must be powers of 2 (e.g., 1, 2, 4, 8, 16, ...),
        representing the specific DQ bit flags.
    """

    from astropy.nddata.bitmask import _is_bit_flag

    for v in bitvalues:
        if not _is_bit_flag(v):
            raise ValueError(
                f"Input list contains invalid (not powers of two) bit flag: {v}"
            )

    dqmask = np.zeros_like(dqarr, dtype=bool)
    for bitval in bitvalues:
        dqmask = dqmask | (dqarr & bitval)

    return dqmask

def pop_pxar_kw(filepaths):
    """
    
    Populate the PIXAR_A2 SCI header keyword which is required by pyKLIP in
    case it is not already available.

    Parameters
    ----------
    filepaths : list or array
        File paths of the FITS files whose headers shall be checked.
    """
    
    for filepath in filepaths:
        try:
            pxar = pyfits.getheader(filepath, 'SCI')['PIXAR_A2']
        except:
            hdul = pyfits.open(filepath)
            siaf_nrc = pysiaf.Siaf('NIRCam')
            siaf_nis = pysiaf.Siaf('NIRISS')
            siaf_mir = pysiaf.Siaf('MIRI')
            if hdul[0].header['INSTRUME'] == 'NIRCAM':
                ap = siaf_nrc[hdul[0].header['APERNAME']]
            elif hdul[0].header['INSTRUME'] == 'NIRISS':
                ap = siaf_nis[hdul[0].header['APERNAME']]
            elif hdul[0].header['INSTRUME'] == 'MIRI':
                ap = siaf_mir[hdul[0].header['APERNAME']]
            else:
                raise UserWarning('Data originates from unknown JWST instrument')
            pix_scale = (ap.XSciScale + ap.YSciScale) / 2.
            hdul['SCI'].header['PIXAR_A2'] = pix_scale**2
            hdul.writeto(filepath, output_verify='fix', overwrite=True)
            hdul.close()
    
    pass

def config_stpipe_log(level='WARNING', suppress=False):
    """
    Configure the logging level for the stpipe pipeline.

    Parameters
    ----------
    level : str
        The logging level as a string (e.g., 'ERROR', 'DEBUG').

    suppress : bool
        If True, suppresses the log output to ERROR level.
        If False, restores the default logging configuration.
        
    Returns
    -------
    None.
    """

    # Convert the string level to a logging level constant.
    log_level = getattr(logging, level.upper(), None)
    if log_level is None:
        raise ValueError(f"Invalid log level: {level}")

    # Prevent 'stpipe' logs from propagating to the root logger.
    # This suppresses duplicate log messages.
    log_stpipe = logging.getLogger('stpipe')
    log_stpipe.setLevel(log_level)
    log_stpipe.propagate = False

    if suppress:
        # Suppress the log output from the 'stpipe'.
        suppress_log_configuration = f"""
        [*]
        handler = append:pipeline.log
        level = {level.upper()}
        """
        stpipe.log.load_configuration(io.BytesIO(suppress_log_configuration.encode()))
    else:
        # Restore the default logging configuration.
        stpipe.log.load_configuration(stpipe.log._find_logging_config_file())
