from spaceKLIP.psf import JWST_PSF
from spaceKLIP.utils import imshift
import webbpsf_ext
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np 
from scipy import ndimage

filters = ['F1065C', 'F1140C', 'F1550C']
crpix = [[120.184,112.116],
         [119.749,112.236],
         [119.746,113.289]]
edge_widths = [7, 7, 15]
generate_psfs = False
date = '2024-05-04T17:46:11.117' # Date shouldn't be too important

def get_edges_mask(array, width=6):
    ny, nx = array.shape
    x0, y0 = nx // 2, ny // 2
    Y, X = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')

    # Thetas
    thetas = [-4.83544897, -94.83544897]

    combined_mask = np.zeros_like(array, dtype=bool)
    for theta in thetas:
        # Convert angle to radians
        theta_rad = np.deg2rad(theta)

        # Rotate coordinates so the line becomes vertical
        coord_rot = (X - x0) * np.cos(theta_rad) + (Y - y0) * np.sin(theta_rad)

        # Create mask for pixels close to the line and apply to full mask
        line_mask = np.abs(coord_rot) <= (width / 2)
        combined_mask |= line_mask

    return combined_mask

for fi, FILTER in enumerate(filters):
    if generate_psfs:
        # Initialize JWST_PSF object. Use odd image size so that PSF is
        # centered in pixel center. #106.75 105.24
        APERNAME = 'MIRIM_MASK'+FILTER[1:-1]

        fov_pix = 215
        oversample=2
        use_coeff=False
        spectrum = webbpsf_ext.stellar_spectrum('G2V')

        kwargs = {
            'fov_pix': fov_pix,
            'oversample': oversample,
            'date': date,
            'use_coeff': use_coeff,
            'sp': spectrum
        }
        psf = JWST_PSF(APERNAME, FILTER, **kwargs)

        model_psf = psf.gen_psf_idl((0, 0), coord_frame='idl', return_oversample=False, quick=True)
        fits.writeto('./model_psf_{}.fits'.format(FILTER), model_psf, overwrite=True)
    else:
        model_psf = fits.getdata('./model_psf_{}.fits'.format(FILTER))

    center_pix = crpix[fi]
    # Need to shift by assumed cropping from spaceKLIP, plus 1-indexing, minus corner vs center
    adj_crpix1 = center_pix[0]-0.5-13
    adj_crpix2 = center_pix[1]-0.5-7
    model_psf = imshift(model_psf, [adj_crpix1-107, adj_crpix2-107], crop_after_pad=True)

    # Shifting can sometimes causes NaN's, clean them up
    model_psf[np.where(np.isnan(model_psf))] = 0

    # Take the top 10% of pixels from the PSF and mask them
    model_indices = np.where(model_psf > np.percentile(model_psf, 90))
    model_psf[model_indices] = np.nan

    # Take the edges of the FQPM and mask everything not within a certain width
    fqpm_mask = get_edges_mask(model_psf, width=edge_widths[fi])
    model_psf[~fqpm_mask] = np.nan

    # Turn any non-masked values to 1
    model_psf[np.where(~np.isnan(model_psf))] = 1
    mask = model_psf

    # Save mask for background subtraction optimisation. 
    fits.writeto('./godoy_mask_{}.fits'.format(FILTER.lower()), mask, overwrite=True)