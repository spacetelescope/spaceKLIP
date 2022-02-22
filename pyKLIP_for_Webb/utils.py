import os

import numpy as np
import astropy.io.fits as pyfits
from scipy.ndimage import rotate, shift
from scipy.interpolate import RegularGridInterpolator

import matplotlib.pyplot as plt

import webbpsf

def get_offsetpsf(offsetpsfdir,
                  obs,
                  filt,
                  mask,
                  key):
    """
    Get a derotated and integration time weighted average of an offset PSF
    from WebbPSF. Try to load it from the offsetpsfdir and generate it if
    it is not in there yet. This offset PSF is normalized to an integrated
    source of 1 and takes into account the throughput of the pupil mask.
    
    Parameters
    ----------
    filt: str
        Filter name from JWST data header.
    mask: str
        Coronagraphic mask name from JWST data header.
    key: str
        Dictionary key of the self.obs dictionary specifying the
        considered observation.
    
    Returns
    -------
    totop: array
        Stamp of the derotated and integration time weighted average of
        the offset PSF.
    """
    
    # Try to load the offset PSF from the offsetpsfdir and generate it if
    # it is not in there yet.
    if (not os.path.exists(offsetpsfdir)):
        os.makedirs(offsetpsfdir)
    if (not os.path.exists(offsetpsfdir+filt+'_'+mask+'.npy')):
        gen_offsetpsf(offsetpsfdir, filt, mask)
    offsetpsf = np.load(offsetpsfdir+filt+'_'+mask+'.npy')
    
    # Find the science target observations.
    ww_sci = np.where(obs[key]['TYP'] == 'SCI')[0]
    
    # Compute the derotated and integration time weighted average of the
    # offset PSF. Values outside of the PSF stamp are filled with zeros.
    totop = np.zeros_like(offsetpsf)
    totet = 0. # s
    for i in range(len(ww_sci)):
        inttm = obs[key]['NINTS'][ww_sci[i]]*obs[key]['EFFINTTM'][ww_sci[i]] # s
        totop += inttm*rotate(offsetpsf.copy(), -obs[key]['PA_V3'][ww_sci[i]], reshape=False, mode='constant', cval=0.)
        totet += inttm # s
    totop /= totet
    
    return totop

def gen_offsetpsf(offsetpsfdir,
                  filt,
                  mask):
    """
    Generate an offset PSF using WebbPSF. This offset PSF is normalized to
    an integrated source of 1 and takes into account the throughput of the
    pupil mask.
    
    Parameters
    ----------
    filt: str
        Filter name from JWST data header.
    mask: str
        Coronagraphic mask name from JWST data header.
    """
    
    # Get NIRCam
    nircam = webbpsf.NIRCam()

    # Apply the correct pupil mask, but no image mask (unocculted PSF).
    nircam.filter = filt
    if (mask in ['MASKA210R', 'MASKA335R', 'MASKA430R']):
        nircam.pupil_mask = 'MASKRND'
    elif (mask in ['MASKASWB']):
        nircam.pupil_mask = 'MASKSWB'
    elif (mask in ['MASKALWB']):
        nircam.pupil_mask = 'MASKLWB'
    else:
        raise UserWarning()
    nircam.image_mask = None
    
    # Compute the offset PSF using WebbPSF and save it to the offsetpsfdir.
    hdul = nircam.calc_psf(oversample=1)
    psf = hdul[0].data # PSF center is at (39, 39)

    hdul.close()
    np.save(offsetpsfdir+filt+'_'+mask+'.npy', psf)
    
    return None

def get_transmission(params, 
                     obs,
                     pxsc, # mas
                     filt,
                     mask,
                     subarr,
                     odir,
                     key):
    """
    Write coronagraphic mask transmission into self.transmission. The
    output is a 2D transmission map containing the derotated and
    integration time weighted average of the PSF masks from CRDS.
    
    TODO: assumes that (159.5, 159.5) is the center of the PSF masks from
          CRDS. This seems to be true for the round masks. For the bar
          masks, this needs to be confirmed. Then, uses the PSF position
          with respect to the NRCA4_MASKSWB and the NRCA5_MASKLWB subarray
          from pySIAF to shift the bar mask PSF masks to their new center.
    
    Parameters
    ----------
    pxsc: float
        Pixel scale of the PSF masks from CRDS.
    filt: str
        Filter name from JWST data header.
    mask: str
        Coronagraphic mask name from JWST data header.
    subarr: str
        Subarray name from JWST data header.
    odir: str
        Output directory for the plots.
    key: str
        Dictionary key of the self.obs dictionary specifying the
        considered observation.
    
    Returns
    -------
    tottp: array
        2D transmission map containing the derotated and integration time
        weighted average of the PSF masks from CRDS.
    """
    
    # Check if the fiducial point override is active.
    if ((params.fiducial_point_override == True) and (mask in ['MASKASWB', 'MASKALWB'])):
        filt_temp = 'narrow'
    else:
        filt_temp = filt
    
    # Find the science target observations.
    ww_sci = np.where(obs[key]['TYP'] == 'SCI')[0]
    
    # Open the correct PSF mask. Download it from CRDS if it is not yet
    # in the psfmaskdir.
    psfmask = params.psfmask[filt_temp+'_'+mask+'_'+subarr]
    if (not os.path.exists(params.psfmaskdir)):
        os.makedirs(params.psfmaskdir)
    if (not os.path.exists(params.psfmaskdir+psfmask+'.fits')):
        urllib.request.urlretrieve('https://jwst-crds.stsci.edu/unchecked_get/references/jwst/'+psfmask+'.fits', params.psfmaskdir+psfmask+'.fits')
    hdul = pyfits.open(params.psfmaskdir+psfmask+'.fits')
    tp = hdul['SCI'].data[1:-1, 1:-1] # crop artifact at the edge
    hdul.close()
    
    # Shift the bar mask PSF masks to their new center. Values outside of
    # the subarray are filled with zeros (i.e., no transmission).
    if (mask in ['MASKASWB']):
        tp = shift(tp, (0., -params.offset_swb[filt_temp]*1000./pxsc), mode='constant', cval=0.)
    elif (mask in ['MASKALWB']):
        tp = shift(tp, (0., -params.offset_lwb[filt_temp]*1000./pxsc), mode='constant', cval=0.)
    
    # Compute the derotated and integration time weighted average of the
    # PSF masks. Values outside of the subarray are filled with zeros
    # (i.e., no transmission). Then, create a regular grid interpolator
    # taking 2D pixel offset as input and returning the coronagraphic mask
    # transmission.
    ramp = np.arange(tp.shape[0]) # pix
    xx, yy = np.meshgrid(ramp, ramp) # pix
    xx = xx-158.5 # pix; new center because PSF mask was cropped by 2 pixel
    yy = yy-158.5 # pix; new center because PSF mask was cropped by 2 pixel
    dist = np.sqrt(xx**2+yy**2) # pix
    tottp = np.zeros_like(tp)
    totet = 0. # s
    for i in range(len(ww_sci)):
        inttm = obs[key]['NINTS'][ww_sci[i]]*obs[key]['EFFINTTM'][ww_sci[i]] # s
        tottp += inttm*rotate(tp.copy(), -obs[key]['PA_V3'][ww_sci[i]], reshape=False, mode='constant', cval=0.)
        totet += inttm # s
    tottp /= totet
    tottp[dist > params.owa] = np.nan
    params.transmission = RegularGridInterpolator((xx[0, :], yy[:, 0]), tottp)
    
    # Plot.
    plt.figure(figsize=(6.4, 4.8))
    ax = plt.gca()
    pp = ax.imshow(tottp, origin='lower', cmap='viridis', vmin=0, vmax=1)
    cc = plt.colorbar(pp, ax=ax)
    cc.set_label('Transmission', rotation=270, labelpad=20)
    ax.set_xlabel('$\Delta$RA [pix]')
    ax.set_ylabel('$\Delta$DEC [pix]')
    ax.set_title('Transmission')
    plt.tight_layout()
    plt.savefig(odir+key+'-transmission.pdf')
    plt.close()
    
    return tottp


def correct_transmission(stamp,
                         stamp_dx, # pix
                         stamp_dy,
                         params): # pix
    """
    Apply coronagraphic mask transmission. This uses 2D offset from the
    host star PSF center.
    
    Note: uses a standard cartesian coordinate system so that North is +y
          and East is -x.
    
    Note: uses the 2D transmission map stored in self.transmission. Need
          to run self.get_transmission first!
    
    Parameters
    ----------
    stamp: array
        Frame to which coronagraphic mask transmission shall be applied.
    stamp_dx: array
        Frame of the same shape as stamp containing the x-axis separation
        from the host star PSF center for each pixel.
    stamp_dy: array
        Frame of the same shape as stamp containing the y-axis separation
        from the host star PSF center for each pixel.
    
    Returns
    -------
    stamp: array
        Frame to which coronagraphic mask transmission was applied.
    """
    
    # Apply coronagraphic mask transmission.
    xy = np.vstack((stamp_dy.flatten(), stamp_dx.flatten())).T
    transmission = params.transmission(xy)
    transmission = transmission.reshape(stamp.shape)
    
    return transmission*stamp