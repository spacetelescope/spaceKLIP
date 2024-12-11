import os
import functools
import urllib

import astropy
import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import webbpsf
from astropy.io import fits
from astroquery.mast import Mast, Observations

import jwst
import jwst.datamodels as datamodels
import spaceKLIP
import spaceKLIP.utils as utils
import spaceKLIP.engdb as engdb
import spaceKLIP.plotting as plotting

# Constants and lookup tables:

# TA Dither Offsets.
# Values from /PRDOPSSOC-067/TA_dithersXML/Excel/NIRCam_TA_dithers.xlsx
# Offsets in detector pixels, in the DET frame
#                           dither_num: [delta_detX, delta_detY]
_ta_dither_offsets_pix = {'NIRCAM': {0: [0,0],
                                     1: [2, 4],
                                     2: [2+3, 4-5]},
                          }

# sign convention for DET frame relative to SCI frame
_ta_dither_sign = {'NRCALONG': [1, -1],
                   'NRCBLONG': [1,1],  # TODO check this. But NRC TSO TA isn't dithered.
                    'NRCA1': [1,1],    # TODO check this. But not directly relevant since thus far there have been no dithered TAs on NRCA2
                    'NRCA2': [1,1],    # TODO check this. But not directly relevant since thus far there have been no dithered TAs on NRCA2
                    'NRCA3': [1,1],    # TODO check this. But not directly relevant since WFS TA doesn't use dithers
                    'NRCA4': [1,1],    # TODO check this. But not directly relevant since WFS TA doesn't use dithers
                    'NIS': [1,1],      # TODO check this.
                   }

def _crop_display_for_miri(ax,
                           hdul):
    """
    Crop the display region for a MIRI Target Acquisition (TA) image.

    Many MIRI TA images use full array, even though only a subarray is of interest.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib Axes object where the image is displayed.
    hdul : astropy.io.fits.HDUList
        FITS HDU list containing the TA image data and headers.

    Returns
    -------
    None.
    """
    # Note, this would be more elegant to look up from siaf,
    # but we just hard-code values here since none of this will ever change.

    apname = hdul[0].header['APERNAME']

    boxsize = 64
    if apname == 'MIRIM_TAMRS':
        # Crop to upper corner.
        ax.set_xlim(1023 - boxsize, 1023)
        ax.set_ylim(1019 - boxsize, 1019)
    elif apname == 'MIRIM_TASLITLESSPRISM' and hdul[0].header['SUBARRAY'] == 'SLITLESSPRISM':
        # Crop to upper part.
        ax.set_ylim(415 - boxsize, 415)
        ax.set_xlim(8, None)
    elif apname == 'MIRIM_SLITLESSPRISM' and hdul[0].header['SUBARRAY'] == 'SLITLESSPRISM':
        # Crop to upper part.
        ax.set_ylim(415 - boxsize - 80, 415 - 80)
        ax.set_xlim(8, None)
    elif apname == 'MIRIM_TALRS':
        ax.set_xlim(382, 382 + boxsize)
        ax.set_ylim(258, 258 + boxsize)
    elif apname == 'MIRIM_SLIT':
        ax.set_xlim(294, 294 + boxsize)
        ax.set_ylim(269, 269 + boxsize)

def set_params(parameters):
    """
    Structure parameters into MAST's expected format.

    Parameters
    ----------
    parameters : dict
        A dictionary of query parameters where keys are parameter names 
        and values are lists of possible values.

    Returns
    -------
    list
        A list of dictionaries, each containing a parameter name and its values, 
        formatted for MAST's API requirements.
    """
    return [{"paramName": key, "values": values} for key, values in parameters.items()]

def which_guider_used(visitid,
                      guidemode='FINEGUIDE'):
    """
    Query MAST for which guider was used in a given visit.

    Parameters
    ----------
    visitid : str
        Visit ID in either of the following formats:
        - 'VPPPPPOOOVVV' (e.g., V01234001001)
        - 'PPPP:O:V' (e.g., 1234:0:1)
        The function will handle both formats.
    guidemode : str
        Which kind of guide mode to check. Defaults to FINEGUIDE but
        would need to be TRACK for moving targets.

    Returns
    -------
    guider_used : str or None
        The guider used ('FGS1' or 'FGS2'), or `None` if no guider information is found.
    """

    # Check inputs.
    # Standardize the visit ID format to 'VPPPPPOOOVVV'.
    visitid = utils.get_visitid(visitid)

    # Extract program ID, observation number, and visit number from the visit ID.
    progid = visitid[1:6]  # Program ID
    obs = visitid[6:9]     # Observation number
    visit = visitid[9:12]  # Visit number

    # Set up the query keywords to filter results.
    keywords = {
        'program': [progid],               
        'observtn': [obs],                 
        'visit': [visit],                  
        'exp_type': ['FGS_' + guidemode]
    }

    # Restructure the keywords dictionary for the MAST API syntax.
    params = {
        'columns': '*',
        'filters': set_params(keywords)
    }

    # Run the web service query. This uses the specialized, lower-level webservice for the
    # guidestar queries: https://mast.stsci.edu/api/v0/_services.html#MastScienceInstrumentKeywordsGuideStar
    service = 'Mast.Jwst.Filtered.GuideStar'
    t = Mast.service_request(service, params)

    # Check the APERNAME result should be 'FGS1_FULL' or 'FGS2_FULL'.
    # Since all guiding in a visit uses the same guide star, checking the first result is sufficient.
    if len(t) > 0:
        guider_used = t['apername'][0][0:4]
    else:
        guider_used = None  # No guide star information found.

    return guider_used

@functools.lru_cache  # Cache the results so subsequent calls are faster.
def get_visit_ta_image(visitid,
                       inst='NIRCam',
                       product_type='cal',
                       localpath=None,
                       verbose=True,
                       **kwargs):

    """
    Retrieve from MAST the NIRCam/NIRSpec/NIRISS/MIRI
    target acqusition image(s) for a given visit and returns it as a HDUList variable without writing to disk.

    Behavior:
        - If only one TA file is found, that file is returned directly as an HDUList.
        - If multiple TA files are found (e.g., TACQ and TACONFIRM), a list is returned
          containing all of them.

    Parameters
    ----------
    visitid : str
        The visit ID, expected to start with "V".
    inst : str, optional
        Instrument name (e.g., 'NIRCam', 'MIRI'), default is 'NIRCam'.
    product_type : str, optional
        Type of file to retrieve ('cal', 'rate', 'uncal'), default is 'cal'.
   localpath : str, optional
        Path to a locally cached file for retrieval, default is None.
    localpath : str, optional
        Directory where files are cached locally. If the specified file is found 
        in this directory, it is loaded from disk instead of downloading from MAST. 
        Defaults to the current working directory if not provided (None).
    verbose : bool, optional
        If True, print progress and details (default: True).
    
    Returns:
    -------
    HDUList or list of HDUList
        The retrieved TA image(s) from MAST.
    """

    # Check inputs.
    # Standardize the visit ID format to 'VPPPPPOOOVVV'.
    visitid = utils.get_visitid(visitid)

    # Define query filters for TA files.
    ta_keywords = {
        'visit_id': [visitid[1:]],  # Remove the initial 'V' character from the visit ID.
        'exp_type': ['NRC_TACQ', 'MIR_TACQ', 'MIR_TACONFIRM', 
                     'NRS_WATA', 'NRS_TACONFIRM', 'NIS_TACQ']
    }

    # Restructure the keywords dictionary for the MAST API syntax.
    ta_params = {
        'columns': '*',
        'filters': set_params(ta_keywords)
    }
    
    # Query MAST based on instrument and TA keyword parameters.
    service = f"Mast.Jwst.Filtered.{inst.capitalize()}"

    if verbose:
        print(f"Querying MAST for target acquisition files for visit {visitid}")

    # Perform the service request to MAST.
    t = Mast.service_request(service, ta_params)
    nfiles = len(t)

    if verbose:
        print(f"Found {nfiles} target acquisition image(s) for this observation.")
    
    # Retrieve filenames and sort them for consistent ordering.
    filenames = t['filename']
    filenames.sort()

    # Prepare to download and/or open files.
    ta_files_found = []
    for filename in filenames:

        # Adjust filename based on the data product type requested.
        if product_type in ['rate', 'uncal']:
            filename = filename.replace('_cal.fits', f'_{product_type}.fits')

        if verbose:
            print(f"TA file: {filename}")
        
        # Define local cache path.
        local_file_cache_path = os.path.join(localpath or '.', filename)
        if os.path.exists(local_file_cache_path):
            # Open the file from the local path if it exists.
            if verbose:
                print(f"Opening file from local path: {local_file_cache_path}")
            ta_hdul = fits.open(local_file_cache_path)
        else:  # Attempt to download the file directly from MAST.
            try:
                base_url = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product/"
                mast_file_url = f"{base_url}{filename}"
                ta_hdul = fits.open(mast_file_url)
            except urllib.error.HTTPError as err:
                if err.code == 401:  # Unauthorized access error code.
                    # Use MAST API to allow retrieval of exclusive access data, if relevant.
                    mast_api_token = os.environ.get('MAST_API_TOKEN', None)
                    obs = Observations(api_token=mast_api_token)
                    uri = f"mast:JWST/product/{filename}"
                    obs.download_file(uri, local_path=local_file_cache_path, cache=False)
                    ta_hdul = fits.open(local_file_cache_path)
                else:
                    raise   # Re-raise any errors other than 401 for permissions denied.

        ta_files_found.append(ta_hdul)
    return ta_files_found[0] if nfiles == 1 else ta_files_found

def get_ta_reference_point(inst,
                           hdul,
                           ta_expnum=None):

    """
    Determine the Target Acquisition (TA) reference point.
    
    The TA reference point is the expected position of the target in the observation. 
    For most cases, this corresponds to the aperture reference location, but it may 
    be adjusted based on:
      - TA dither patterns (NIRCam and NIRISS)
      - Subarray/full array coordinate system discrepancies (MIRI).

    Parameters
    ----------
    inst : str
        The instrument name ('NIRCAM', 'NIRISS', or 'MIRI').
    hdul : astropy.io.fits.HDUList
        FITS HDU list containing observation data.
    ta_expnum : int, optional
        The Target Acquisition exposure number (1-based indexing).

    Returns
    -------
    xref, yref : float
        The reference point in the science coordinate frame (0-based indexing).
    """
    
    # Retrieve the SIAF aperture information.
    siaf = utils.get_siaf(inst)
    ap = siaf.apertures[hdul[0].header['APERNAME']]

    if inst.upper() in ['NIRCAM', 'NIRISS']:
        # Default reference point: aperture center
        # in science coordinates (adjusted for 0-based indexing).
        xref = ap.XSciRef - 1
        yref = ap.YSciRef - 1

        if ta_expnum is not None and ta_expnum > 0:
            # If there are multiple ta exposures, take into account the dither moves
            # and the sign needed to go from DET to SCI coordinate frames.
            xref -= _ta_dither_offsets_pix[inst][ta_expnum - 1][0] * _ta_dither_sign[hdul[0].header['DETECTOR']][0]
            yref -= _ta_dither_offsets_pix[inst][ta_expnum - 1][1] * _ta_dither_sign[hdul[0].header['DETECTOR']][1]

    elif inst.upper() == 'MIRI':
        try:
            if hdul[0].header['SUBARRAY'] == 'FULL' and hdul[0].header['APERNAME'] != 'MIRIM_SLIT':
                # For MIRI, sometimes we have subarray apertures for TA but the images are actually full array.
                # In which case use the detector coordinates.
                xref = ap.XDetRef - 1   # siaf uses 1-based counting
                yref = ap.YDetRef - 1   # ditto
            elif hdul[0].header['APERNAME'] == 'MIRIM_TASLITLESSPRISM' and hdul[0].header['SUBARRAY'] == 'SLITLESSPRISM':
                # This one's weird. It's pointed using TASLITLESSPRISM but read out usng SLITLESSPRISM.
                ap_slitlessprism = siaf.apertures['MIRIM_SLITLESSPRISM']
                xref, yref = ap_slitlessprism.det_to_sci(ap.XDetRef, ap.YDetRef)  # convert coords from TASLITLESSPRISM to SLITLESSPRISM
                xref, yref = xref - 1, yref - 1  # siaf uses 1-based counting
            elif hdul[0].header['APERNAME'] == 'MIRIM_SLIT':
                # This case is tricky. TACONFIRM image in slit type aperture, which doesn't have Det or Sci coords defined
                # It's made even more complex by filter-dependent MIRI offsets
                print("TODO need to get more authoritative intended target position for MIRIM TA CONFIRM")
                xref, yref = 317, 301
            elif hdul[0].header['APERNAME'].endswith('_UR') or  hdul[0].header['APERNAME'].endswith('_CUR'):
                # Coronagraphic TA, pointed using special TA subarrays but read out using the full coronagraphic subarray
                # Similar to how TASLITLESSPRISM works
                ap_subarray = siaf.apertures['MIRIM_'+hdul[0].header['SUBARRAY']]
                xref, yref = ap_subarray.det_to_sci(ap.XDetRef, ap.YDetRef)  # convert coords from TA subarray to coron subarray
                xref, yref = xref - 1, yref - 1  # siaf uses 1-based counting
            else:
                xref = ap.XSciRef - 1   # siaf uses 1-based counting
                yref = ap.YSciRef - 1   # ditto
        except TypeError: # LRS slit type doesn't have X/YSciRef
            xref = yref = 0
            print('ERROR DEBUG THIS')

    return xref, yref

def show_ta_img(visitid,
                ax=None,
                inst='NIRCam',
                return_handles=False,
                ta_expnum=None,
                mark_reference_point=True,
                mark_apername=True,
                zoom_region=False,
                **kwargs):
    """ 
    Retrieve and display a target acquisition image.
    
    Parameters
    ----------
    visitid : str
        The visit ID of the observation. The format should be either:
        - 'VPPPPPOOOVVV' (e.g., V01234001001)
        - 'PPPP:O:V' (e.g., 1234:0:1).
    ax : matplotlib.axes.Axes, optional
        The matplotlib Axes object where the TA image will be displayed. If not 
        provided, the current active Axes is used.
    inst : str
        The instrument name ('NIRCAM', 'NIRISS', or 'MIRI').
    return_handles : bool, optional
        If True, returns the data handles including the HDU list, Axes object, 
        normalization, colormap, and background level. Defaults to False.
    ta_expnum : int, optional
        The TA exposure number to display (1-based indexing). Required if multiple 
        TA exposures are available for the visit. Defaults to None.
    mark_reference_point : bool, optional
        Whether to mark the expected target position in the image. Defaults to True.
    mark_apername : bool, optional
        Whether to annotate the aperture of the guider used. Defaults to True.

    Returns
    -------
    Optional:
        If `return_handles` is True, returns a tuple (hdul, ax, norm, cmap, bglevel), where:
        - hdul : astropy.io.fits.HDUList
            The FITS HDU list containing the TA image data.
        - ax : matplotlib.axes.Axes
            The matplotlib Axes object where the image is displayed.
        - norm : matplotlib.colors.Normalize
            The normalization applied to the image.
        - cmap : matplotlib.colors.Colormap
            The colormap used for the image.
        - bglevel : float
            The background level (median) subtracted from the image.
    """

    # Check inputs.
    inst = inst.upper()
    
    # Retrieve and validate the TA image(s).
    # If there are multiple TA images, you have to pick which one you want to display.
    title_extra = ''
    hdul = get_visit_ta_image(visitid, inst=inst, **kwargs)
    if not hdul:
        raise RuntimeError("No TA images found for that visit.")
    if isinstance(hdul, list) and not isinstance(hdul, fits.HDUList):
        if ta_expnum is None:
            raise ValueError(
                f"Specify `ta_expnum=<n>` to select from {len(hdul)} TA exposures (1-based indexing)."
            )
        hdul = hdul[ta_expnum - 1]
        if inst != 'MIRI':
            title_extra = f' exp #{ta_expnum}'
        if 'CONF' in hdul[0].header['EXP_TYPE']:
            title_extra = 'CONFIRM'

    # Extract TA image and compute stats.
    ta_img = hdul['SCI'].data
    header = hdul['SCI'].header
    wcs = astropy.wcs.WCS(header)
    mask = np.isfinite(ta_img)  # Mask NaNs
    rmean, rmedian, rsig = astropy.stats.sigma_clipped_stats(ta_img[mask])
    bglevel = rmedian

    # Configure colormap and normalization.
    vmax = np.nanmax(ta_img) - bglevel
    cmap = matplotlib.cm.viridis.copy()
    cmap.set_bad('orange')
    norm = matplotlib.colors.AsinhNorm(linear_width=vmax * 0.003,
                                       vmax=vmax, vmin=-rsig)

    # Prepare annotation text.
    model = datamodels.open(hdul)
    annotation_text = (
        f"{model.meta.target.proposer_name}\n"
        f"{model.meta.instrument.filter}, {model.meta.exposure.readpatt}:"
        f"{model.meta.exposure.ngroups}:{model.meta.exposure.nints}\n"
        f"{model.meta.exposure.effective_exposure_time:.2f} s"
    )

    # Create the plot.
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
        # ax = plt.gca()
    ax.imshow(ta_img - bglevel, norm=norm, cmap=cmap, origin='lower')
    ax.set_title(
        f"{inst} TA{title_extra} on {visitid}\n"
        f"{hdul[0].header['DATE-OBS']} {hdul[0].header['TIME-OBS'][:8]}"
    )
    ax.set_ylabel("Pixels Y")
    ax.set_xlabel("Pixels X")
    plotting.annotate_secondary_axes_arcsec(ax, ta_img - bglevel, wcs)

    ax.text(0.05, 0.95, annotation_text, color='k',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(facecolor='0.75', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.3'))
    if inst == 'MIRI':
        spaceKLIP.target_aqc_tools._crop_display_for_miri(ax, hdul)

    # Mark the TA reference point in the image?
    # i.e. where was the target 'supposed to be' in any given observation.
    if mark_reference_point:
        xref, yref = get_ta_reference_point(inst, hdul, ta_expnum)
        ax.axvline(xref, color='0.75', alpha=0.7, ls='--')
        ax.axhline(yref, color='0.75', alpha=0.7, ls='--')
        ax.text(xref, yref + 2, 'SIAF', color='k',
                verticalalignment='center', horizontalalignment='right',
                bbox=dict(facecolor='0.75', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.3'))

    # Mark the aperture for the guider used?
    if mark_apername:
        ax.text(0.95, 0.95,
                hdul[0].header['APERNAME']+f"\n using {which_guider_used(visitid)}",
                color='k', transform=ax.transAxes, horizontalalignment='right', verticalalignment='top',
                bbox=dict(facecolor='0.75', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.3'))


    # Plot a zoomed in region of the image?
    if zoom_region:
        # Create the inset axis
        inset_ax = ax.inset_axes([1.3, 0.4, 0.6, 0.6], transform=ax.transAxes)
        inset_ax.imshow(np.full_like(ta_img, np.nan), norm=norm, origin='lower')
        inset_ax.set_xlim(xref-10, xref+10)
        inset_ax.set_ylim(yref-10, yref+10)
        plotting.annotate_secondary_axes_arcsec(inset_ax, ta_img - bglevel, wcs)
        inset_ax.set_ylabel("Pixels Y")
        inset_ax.set_xlabel("Pixels X")
        
        # Optionally, mark reference point in inset
        if mark_reference_point:
            inset_ax.axvline(xref, color='0.75', alpha=0.5, ls='--')
            inset_ax.axhline(yref, color='0.75', alpha=0.5, ls='--')     
            inset_ax.text(xref, yref + 2, 'SIAF', color='0.9', 
                    verticalalignment='center', horizontalalignment='right')

        
    # Return handles
    if return_handles:
        # Return both main axis and inset axis if zoom_region is specified
        if zoom_region:
            return hdul, ax, inset_ax, norm, cmap, bglevel
        else:
            return hdul, ax, norm, cmap, bglevel

def ta_analysis(data_product,
                inst='NIRCam',
                verbose=True,
                plot=False,
                return_wcs_offsets=False, 
                return_pointing_offsets=False,
                **kwargs):
    """
    Retrieve NIRCam/MIRI Target Acquisition (TA) image and evaluate performance.

    Parameters
    ----------
    data_product : str
        FITS file.
    inst : str
        The instrument name ('NIRCAM' or 'MIRI').
    verbose : bool, optional
        If True, displays detailed progress messages and logs during execution.
    plot : bool, optional
        If True, plot the TA image.
    return_wcs_offsets : bool, optional
        If True, return WCS offsets: the difference between the OSS-measured position 
        and the WCS-predicted position of the target. Incompatible with return_pointing_offsets
    return_pointing_offsets : bool, optional
        If True, return pointing offsets: the difference between the OSS-measured position 
        and the intended (reference) target position. Incompatible with return_wcs_offsets. 

    By default it downloads the file from MAST, or looks in the local directory to see if already downloaded.
    Set localpath=[some path] to search for the file in some other directory or filename.

    Returns
    -------
    targ_coords_pix_adj_sci : tuple
        The adjusted science target coordinate in pixels (in the science aperture).
    """
    
    # Check inputs.
    visitid = utils.get_visitid('V'+fits.getval(data_product, 'VISIT_ID'))  # Standardize format to 'VPPPPPOOOVVV'.
    inst = inst.upper()

    # Load SIAF data and retrieve TA images for the visit.
    siaf = utils.get_siaf(inst)
    ta_images = get_visit_ta_image(visitid, inst=inst, **kwargs)
    if not ta_images:
        raise RuntimeError(f"No TA image found for visit {visitid}")
        return None

    # Determine the number of TA images.
    n_ta_images = 1 if isinstance(ta_images, astropy.io.fits.HDUList) else len(ta_images)
    
    # Initialize plotting.
    if plot:
        nrows = 2 if inst == 'MIRI' else 3
        fig, axes = plt.subplots(figsize=(20, 20), nrows=nrows)

    # Get and plot the observed TA image.
    for i_ta_image in range(n_ta_images):
        if plot:
            hdul, ax, inset_ax, norm, cmap, bglevel = show_ta_img(visitid, ax=axes[i_ta_image],
                                                                  return_handles=True, inst=inst,
                                                                  ta_expnum=i_ta_image + 1, zoom_region=True, **kwargs)
        else:
            hdul = get_visit_ta_image(visitid, inst=inst)
            if isinstance(hdul, list) and not isinstance(hdul, fits.HDUList):
                hdul = hdul[i_ta_image]

        # Determine reference point (intended target position).
        xref, yref = get_ta_reference_point(inst, hdul, i_ta_image + 1)
        aperture_text = f'Intended target pos (from SIAF): {xref:.2f}, {yref:.2f}'    
            
        # Determine aperture.
        ta_aperture = siaf.apertures[hdul[0].header['APERNAME']]
        if inst in ['NIRCAM', 'MIRI']:
            full_ap = (siaf[ta_aperture.AperName[:5] + "_FULL"] if inst == 'NIRCAM' else siaf["MIRIM_FULL"])
            # MIRI uses a larger kernel to handle border pixels.
            interp_kernel_size = (5, 5) if inst == 'NIRCAM' else (11, 11)
            # TA computed on the *LAST* of N dithered TA exposures for NIRCAM.
            # TA computed on the *FIRST* exposure (TA), followed by a TACONFIRM for MIRI.
            show_oss_for_image = n_ta_images - 1 if inst == 'NIRCAM' else 0
    
            # Adjust aperture for MIRI slit type.
            if inst == 'MIRI' and hdul[0].header['APERNAME'] == 'MIRIM_SLIT':
                # This is a slit type aperture which doesn't have pixel transforms!
                # Therefore use the full aperture for coord transforms.
                ta_aperture = full_ap

        # Extract science data and clean the image.
        im_obs_clean = hdul['SCI'].data.copy()  # Get science data.
        im_obs_clean[hdul['DQ'].data & 1] = np.nan  # Mask DO_NOT_USE pixels.
        im_obs_clean = astropy.convolution.interpolate_replace_nans(im_obs_clean,  # Interpolate over masked pixels.
                                                                    kernel=np.ones(interp_kernel_size))
        
        ###### EXTRACT CENTROID INFORMATION ######

        try:
            ###### OSS CENTROIDS ######

            # Attempt to retrieve the on-board TA centroid measurement 
            # from the OSS logs in the engineering DB in MAST.
            osslog = engdb.get_ictm_event_log(hdul[0].header['VSTSTART'], hdul[0].header['VISITEND'])

            # Extract the OSS centroids from the log.
            try:
                oss_cen = engdb.extract_oss_TA_centroids(osslog, 'V' + hdul[0].header['VISIT_ID'])
                
                # Convert from full-frame (as used by OSS) to detector subarray coords:
                oss_cen_sci = ta_aperture.det_to_sci(*oss_cen)
            
                # For MIRI only, deal with the fact that the image data can be full array even if the aperture is subarray.
                if inst == 'MIRI':
                    subarray = hdul[0].header['SUBARRAY']
                    if subarray == 'FULL':
                        oss_cen_sci = np.asarray(full_ap.det_to_sci(*oss_cen))
                    elif subarray == 'SLITLESSPRISM':
                        # Special case, pointed using TASLITLESSPRISM but read out using SLITLESSPRISM
                        slitlessprism_ap = siaf['MIRIM_SLITLESSPRISM']
                        oss_cen_sci = np.asarray(slitlessprism_ap.det_to_sci(*oss_cen))
                    elif subarray.startswith('MASK'):
                        coron_ap = siaf['MIRIM_'+ subarray]
                        oss_cen_sci = np.asarray(coron_ap.det_to_sci(*oss_cen))
    
                # Convert from 1-based pixel indexing to 0-based.
                oss_cen_sci_pythonic = np.asarray(oss_cen_sci) - 1  # Centroid in the subarray coordinate frame.
                oss_cen_full_sci = np.asarray(full_ap.det_to_sci(*oss_cen)) - 1  # Centroid in the full-frame coordinate frame.

                # The OSS centroid is computed onboard relative to the LAST of n TA images, if there's more than 1.
                # So if we are showing the last image, then it makes sense to mark and annotate the OSS centroid location.    
                if i_ta_image == show_oss_for_image:
                    oss_centroid_text = f"OSS centroid: {oss_cen_sci_pythonic[0]:.2f}, {oss_cen_sci_pythonic[1]:.2f}"
                    
                    if plot:
                        ax.scatter(*oss_cen_sci_pythonic, color='0.5', marker='x', s=50)
                        ax.text(*oss_cen_sci_pythonic, 'OSS  ', color='0.9', 
                                              verticalalignment='center', horizontalalignment='right')
                        inset_ax.scatter(*oss_cen_sci_pythonic, color='0.5', marker='x', s=50)
                        inset_ax.text(*oss_cen_sci_pythonic, 'OSS  ', color='0.9', 
                                              verticalalignment='center', horizontalalignment='right')
                else:
                    oss_centroid_text = ""                    

            except RuntimeError:
                if verbose:
                    print("Could not parse TA coordinates from log. TA may have failed?")
                    oss_cen_sci_pythonic = (np.nan, np.nan)
                    oss_centroid_text = "No OSS centroid; TA failed"
            
            ###### WCS CENTROIDS ######
            from astropy.wcs import WCS
            from jwst.tweakreg.utils import adjust_wcs
            import jwst
            import astropy.units as u
            
            # Open the data model to access WCS and TA metadata.
            model = jwst.datamodels.open(hdul)
            wcs_ta = model.meta.wcs
    
            # Create a SkyCoord object for the target coordinates (RA, Dec) in ICRS frame.
            targ_coords = astropy.coordinates.SkyCoord(
                ra=model.meta.target.ra,
                dec=model.meta.target.dec,
                frame='icrs',
                unit=u.deg
            )
            
            # Convert celestial coordinates (RA, Dec) to pixel coordinates using WCS.
            targ_coords_pix = wcs_ta.world_to_pixel(targ_coords)  # Returns (x, y).
    
            # Format WCS information as text for output.
            wcs_text = f"Expected from WCS: {targ_coords_pix[0]:.2f}, {targ_coords_pix[1]:.2f}"
    
            # Plot WCS-derived pixel coordinates if plotting is enabled
            if plot:
                ax.scatter(targ_coords_pix[0], targ_coords_pix[1],
                                         color='magenta', marker='+', s=50)
                ax.text(targ_coords_pix[0], targ_coords_pix[1] + 2, 'WCS',
                                      color='magenta', verticalalignment='bottom',
                                      horizontalalignment='center')
                inset_ax.scatter(targ_coords_pix[0], targ_coords_pix[1],
                                         color='magenta', marker='+', s=50)
                inset_ax.text(targ_coords_pix[0], targ_coords_pix[1] + 2, 'WCS',
                                      color='magenta', verticalalignment='bottom',
                                      horizontalalignment='center')

        except ImportError:
            oss_centroid_text = ""
            wcs_text = ""

        ###### WEBBPSF CENTROIDS ######

        # Apply a mask around the border pixels to prioritize the PSF center and ignore bad/hot pixels near edges. 
        # This makes this alignment step more robust.
        nm = 6   # Pixel boarder width.
        border_mask = np.ones_like(im_obs_clean)
        border_mask[:nm] = border_mask[-nm:] = border_mask[:, :nm] = border_mask[:, -nm:] = 0

        # Special cases for different apertures.
        aperture_name = hdul[0].header['APERNAME']
        if aperture_name == 'MIRIM_TALRS':
            # Special case, it's a full frame image but enforce just centroiding in the TA region of interest.
            border_mask[:] = 0
            border_mask[260:320, 390:440] = 1  
        elif aperture_name == 'MIRIM_SLIT':
            border_mask[:] = 0
            border_mask[270:330, 290:350] = 1
        elif aperture_name.endswith('_UR'):
            # Coronagraphy upper right quadrant TA.
            imin, imax = (200, 250) if 'LYOT' in aperture_name else (150, 200)
            border_mask[:] = 0
            border_mask[imin:imax, imin:imax] = 1
        elif aperture_name.endswith('_CUR'):
            # Coronagraphy center upper right quadrant TA.
            imin, imax = (150, 225) if 'LYOT' in aperture_name else (100, 150)
            border_mask[:] = 0
            border_mask[imin:imax, imin:imax] = 1

        # Compute centroid using masked image.
        cen = webbpsf.fwcentroid.fwcentroid(im_obs_clean * border_mask)

        # Plot the centroid if plotting is enabled.
        if plot:
            ax.scatter(cen[1], cen[0], color='red', marker='+', s=50)
            ax.text(cen[1], cen[0], 'webbpsf', color='red',
                                  verticalalignment='center', horizontalalignment='left')
            inset_ax.scatter(cen[1], cen[0], color='red', marker='+', s=50)
            inset_ax.text(cen[1], cen[0], 'webbpsf', color='red',
                                  verticalalignment='center', horizontalalignment='left')
            
        if i_ta_image == show_oss_for_image:
            # Compare OSS centroid to intended position for this image.
            deltapos = (oss_cen_sci_pythonic[0] - xref, oss_cen_sci_pythonic[1] - yref)
            deltapos_type = 'OSS - Intended'
        else:
            # If more than 1 image, for the earlier images, or subsequent TACONFIRMs, show the comparison to webbpsf centroids.
            deltapos = (cen[1] - xref,  cen[0] - yref)
            deltapos_type = 'fwcentroid - Intended'

        # Construct and annotate image text.
        image_text = (f"Pixel coordinates (0-based):\n{oss_centroid_text}\n"
                      f"webbpsf measure_centroid: {cen[1]:.2f}, {cen[0]:.2f}\n"
                      f"{wcs_text}\n{aperture_text}\n"
                      f"$\\Delta$pos ({deltapos_type}): {deltapos[0]:.2f}, {deltapos[1]:.2f}")
        if plot:
            ax.text(0.95, 0.04, image_text, horizontalalignment='right', verticalalignment='bottom',
                    transform=ax.transAxes, color='k',
                    bbox=dict(facecolor='0.75', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.3'))

        if oss_cen_sci_pythonic is not None and (i_ta_image == show_oss_for_image):
        
            # Compute initial delta RA/DEC (WCS to OSS).
            # From above, we already have targ_coords as the RA and DEC of the target star at the time of the exposure.
            wcs_ta_current = wcs_ta
            ta_cen_coords_initial = wcs_ta_current.pixel_to_world(*oss_cen_sci_pythonic)
            dra_initial, ddec_initial = ta_cen_coords_initial.spherical_offsets_to(targ_coords)
            original_wcs_offset_radec = (dra_initial.to(u.arcsec), ddec_initial.to(u.arcsec))

            wcs_offset_pix = np.asarray(targ_coords_pix) - oss_cen_sci_pythonic  # Offset in pixels (x,y).

            # Total offsets.
            total_radec_offset = [ddec_initial.to(u.arcsec), dra_initial.to(u.arcsec)]
            total_px_offset_x, total_px_offset_y = wcs_offset_pix[0], wcs_offset_pix[1]

            # Obtain the initial WCS for the science observation (not TA).
            # We will perform the same RA/DEC adjustments to the TA and SCI WCS.
            wcs_sci_current = datamodels.open(data_product).meta.wcs

            # Iterative adjustment loop. 
            for i in range(20):
                ta_cen_coords = wcs_ta_current.pixel_to_world(*oss_cen_sci_pythonic)
                dra, ddec = ta_cen_coords.spherical_offsets_to(targ_coords)
                wcs_offset_radec = (dra.to(u.arcsec), ddec.to(u.arcsec))
        
                # Update total RA/Dec offsets.
                total_radec_offset[1] += dra
                total_radec_offset[0] += ddec
        
                # Apply the RA/Dec offset to adjust the WCS for TA and SCI.
                wcs_ta_current = adjust_wcs(wcs_ta_current, *wcs_offset_radec)
                wcs_sci_current = adjust_wcs(wcs_sci_current, *wcs_offset_radec)

                # Compute the residual pixel offset after adjustment.
                targ_coords_pix_adj = wcs_ta_current.world_to_pixel(targ_coords)
                residual_pix = np.asarray(targ_coords_pix_adj) - oss_cen_sci_pythonic

                # Update total pixel offsets in TA.
                total_px_offset_x += residual_pix[0]
                total_px_offset_y += residual_pix[1]

                # The new SCI target coordinate in pixels.
                targ_coords_pix_adj_sci = wcs_sci_current.world_to_pixel(targ_coords)

                # Break if residual is within tolerance.
                if np.allclose(residual_pix, [0, 0], atol=1e-3):
                    print("Converged to within tolerance.")
                    break#

            # Final adjusted WCS for TA and SCI.
            wcs_ta_adjust = wcs_ta_current
            wcs_sci_adjust = wcs_sci_current

            # Compute adjusted target coordinates
            targ_coords_adj = wcs_ta_adjust.pixel_to_world(*targ_coords_pix_adj)
            targ_coords_adj_sci = wcs_sci_adjust.pixel_to_world(*targ_coords_pix_adj_sci)

            if plot:
                ax.scatter(targ_coords_pix_adj[0], targ_coords_pix_adj[1],
                                         color='cyan', marker='+', s=50)
                ax.text(targ_coords_pix_adj[0], targ_coords_pix_adj[1] + 2, 'WCS_ADJ',
                                      color='cyan', verticalalignment='bottom',
                                      horizontalalignment='center')
                inset_ax.scatter(targ_coords_pix_adj[0], targ_coords_pix_adj[1], color='cyan', marker='+', s=50)
                inset_ax.text(targ_coords_pix_adj[0], targ_coords_pix_adj[1] + 2, 'WCS_ADJ', color='cyan',
                              verticalalignment='center', horizontalalignment='left')
                
            if verbose:
                print(f"TA Target coordinates (RA, Dec) [deg]: ({targ_coords.ra.deg:.6f}, {targ_coords.dec.deg:.6f})")
                print(f"TA Target (hms | dms): {targ_coords.to_string('hmsdms', sep=':')}")
                print(f"TA Target coordinates from WCS (x, y) [pix]: {targ_coords_pix[0]:.3f}, {targ_coords_pix[1]:.3f}")

                print(f"TA Target centroid coordinates (RA, Dec) [deg]: ({ta_cen_coords.ra.deg:.6f}, {ta_cen_coords.dec.deg:.6f})")
                print(f"OSS centroid on board (x, y):  {oss_cen}  (full det coord frame, 1-based)")
                print(f"OSS centroid converted (x, y): {oss_cen_sci_pythonic}  (sci frame in {ta_aperture.AperName}, 0-based)")
                print(f"OSS centroid converted (x, y): {oss_cen_full_sci}  (sci frame in {full_ap.AperName}, 0-based)")

                print(f"TA WCS offset (x, y) [pix] =  {total_px_offset_x:.3f}, {total_px_offset_y:.3f} (WCS - OSS)")
                print(f"TA ΔRA, ΔDEC =  {total_radec_offset[1]:.3f}, {total_radec_offset[0]:.3f} (WCS - OSS)")

                print(f"Adjusted TA Target coordinates (RA, Dec) [deg]: = {targ_coords_adj.ra.deg:.6f}, {targ_coords_adj.dec.deg:.6f}")
                print(f"TA Target coordinates from Adjusted WCS(x,y) [pix]: = {targ_coords_pix_adj[0]:.3f}, {targ_coords_pix_adj[1]:3f}")
                
            # Annotate RA/Dec offset on plot
            if plot:
                ax.text(
                    0.95, 0.80, 
                    f'WCS $\\Delta$RA, $\\Delta$Dec = {dra.to_value(u.arcsec):.3f}, {ddec.to_value(u.arcsec):.3f} arcsec',
                    horizontalalignment='right', verticalalignment='bottom',
                    transform=ax.transAxes,
                    color='cyan'
                )
    
    # Manage axes visibility for single TA image.
    if plot:
        if n_ta_images == 1:
            axes[1].set_visible(False)
            if inst != 'MIRI':
                axes[2].set_visible(False)
        for ax in axes[:n_ta_images]:
            cb = fig.colorbar(ax.images[0], ax=ax, orientation='horizontal',
                        label=hdul['SCI'].header['BUNIT'], fraction=0.05, shrink=0.9, pad=0.07)
            ticks = cb.ax.get_xticks()
            cb.ax.set_xticks([t for t in ticks if t>0.1])

        plt.tight_layout()

        outname = f'{inst.lower()}_ta_analysis_{visitid}.pdf'
        plt.savefig(outname)
        print(f" => {outname}")

    return targ_coords_pix_adj_sci

