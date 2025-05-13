# =============================================================================
# IMPORTS
# =============================================================================

import os
import jwst
import urllib
import astropy
import logging
import functools
import matplotlib
import numpy as np
import stpsf as webbpsf
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.io import fits
import jwst.datamodels as datamodels
from astroquery.mast import Mast, Observations
from jwst.tweakreg.utils import adjust_wcs
from spaceKLIP.utils import get_siaf, get_visitid
from spaceKLIP.plotting import annotate_secondary_axes_arcsec
from spaceKLIP.engdb import get_ictm_event_log, extract_oss_TA_centroids

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# =============================================================================
# MAIN
# =============================================================================

# Constants and lookup tables:

# TA Dither Offsets.
# Values from /PRDOPSSOC-067/TA_dithersXML/Excel/NIRCam_TA_dithers.xlsx
# Offsets in detector pixels, in the DET frame.
# dither_num: [delta_detX, delta_detY]
_ta_dither_offsets_pix = {'NIRCAM': {0: [0, 0],
                                     1: [2, 4],
                                     2: [2+3, 4-5]},
                          }

# Sign convention for DET frame relative to SCI frame.
_ta_dither_sign = {'NRCALONG': [1, -1],
                   # TODO check this. But NRC TSO TA isn't dithered.
                   'NRCBLONG': [1, 1],
                   # TODO check this. But not directly relevant since thus
                   # far there have been no dithered TAs on NRCA2.
                   'NRCA1': [1, 1],
                   # TODO check this. But not directly relevant since thus far
                   # there have been no dithered TAs on NRCA2.
                   'NRCA2': [1, 1],
                   # TODO check this. But not directly relevant since
                   # WFS TA doesn't use dithers.
                   'NRCA3': [1, 1],
                   # TODO check this. But not directly relevant since
                   # WFS TA doesn't use dithers.
                   'NRCA4': [1, 1],
                   }


def _crop_display_for_miri(ax,
                           hdul):
    """
    Crop the display region for a MIRI Target Acquisition (TA) image.

    Many MIRI TA images use full array, even though only a
    subarray is of interest.

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
    subarray = hdul[0].header['SUBARRAY']
    boxsize = 64

    if apname == 'MIRIM_TAMRS':
        # Crop to upper corner.
        ax.set_xlim(1023 - boxsize, 1023)
        ax.set_ylim(1019 - boxsize, 1019)
    elif apname == 'MIRIM_TASLITLESSPRISM' and subarray == 'SLITLESSPRISM':
        # Crop to upper part.
        ax.set_ylim(415 - boxsize, 415)
        ax.set_xlim(8, None)
    elif apname == 'MIRIM_SLITLESSPRISM' and subarray == 'SLITLESSPRISM':
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
        A list of dictionaries, each containing a parameter
        name and its values, formatted for MAST's API requirements.
    """
    return [{"paramName": key, "values": values} for key,
            values in parameters.items()]


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
        The guider used ('FGS1' or 'FGS2'), or `None`
        if no guider information is found.
    """

    # Check inputs.
    # Standardize the visit ID format to 'VPPPPPOOOVVV'.
    visitid = get_visitid(visitid)

    # Extract program ID, observation number,
    # and visit number from the visit ID.
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

    # Run the web service query. This uses the specialized,
    # lower-level webservice for the guidestar queries:
    # https://mast.stsci.edu/api/v0/_services.html#MastScienceInstrumentKeywordsGuideStar
    service = 'Mast.Jwst.Filtered.GuideStar'
    t = Mast.service_request(service, params)

    # Check the APERNAME result should be 'FGS1_FULL' or 'FGS2_FULL'.
    # Since all guiding in a visit uses the same guide star,
    # checking the first result is sufficient.
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
    Retrieve from MAST the NIRCam/MIRI target acqusition image(s) for a given
    visit and returns it as a HDUList variable without writing to disk.

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
    verbose : bool, optional
        If True, print progress and details (default: True).

    Returns
    -------
    HDUList or list of HDUList
        The retrieved TA image(s) from MAST.
        If only one TA file is found, that file is returned as an HDUList.
        If multiple TA files are found (e.g., TACQ and TACONFIRM), a list
        containing all HDUList is returned.
    """

    # Check inputs.
    # Standardize the visit ID format to 'VPPPPPOOOVVV'.
    visitid = get_visitid(visitid)

    # Define query filters for TA files.
    ta_keywords = {
        'visit_id': [visitid[1:]],  # Remove the initial 'V' from visit ID.
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
        log.info(f"--> Querying MAST for target acquisition files (visit {visitid})")

    # Perform the service request to MAST.
    t = Mast.service_request(service, ta_params)
    nfiles = len(t)

    if verbose:
        log.info(f"--> Found {nfiles} target acquisition image(s).")

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
            log.info(f"--> TA file: {filename}")

        # Define local cache path.
        cache_path = os.path.join(localpath or '.', filename)
        if os.path.exists(cache_path):
            # Open the file from the local path if it exists.
            if verbose:
                print(f"Opening file from local path: {cache_path}")
            ta_hdul = fits.open(cache_path)
        else:  # Attempt to download the file directly from MAST.
            try:
                base_url = ("https://mast.stsci.edu/api/v0.1/"
                            "Download/file?uri=mast:JWST/product/")
                mast_file_url = f"{base_url}{filename}"
                ta_hdul = fits.open(mast_file_url)
            except urllib.error.HTTPError as err:
                if err.code == 401:  # Unauthorized access error code.
                    # Use MAST API to retrieve exclusive access data if needed.
                    mast_api_token = os.environ.get('MAST_API_TOKEN', None)
                    obs = Observations(api_token=mast_api_token)
                    uri = f"mast:JWST/product/{filename}"
                    obs.download_file(uri, local_path=cache_path, cache=False)
                    ta_hdul = fits.open(cache_path)
                else:
                    raise  # Re-raise errors other than permission denied (401)

        ta_files_found.append(ta_hdul)
    return ta_files_found[0] if nfiles == 1 else ta_files_found


def get_ta_reference_point(inst,
                           hdul,
                           ta_expnum=None):

    """
    Determine the Target Acquisition (TA) reference point.

    The TA reference point is the expected position of the
    target in the observation. For most cases, this corresponds
    to the aperture reference location, but it may be adjusted based on:
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
    siaf = get_siaf(inst)
    ap = siaf.apertures[hdul[0].header['APERNAME']]

    if inst.upper() in ['NIRCAM', 'NIRISS']:
        # Default reference point: aperture center
        # in science coordinates (adjusted for 0-based indexing).
        xref = ap.XSciRef - 1
        yref = ap.YSciRef - 1

        if ta_expnum is not None and ta_expnum > 0:
            # If there are multiple ta exposures, take into account the
            # dither moves and the sign needed to go from
            # DET to SCI coordinate frames.
            xref -= (_ta_dither_offsets_pix[inst][ta_expnum - 1][0] *
                     _ta_dither_sign[hdul[0].header['DETECTOR']][0])
            yref -= (_ta_dither_offsets_pix[inst][ta_expnum - 1][1] *
                     _ta_dither_sign[hdul[0].header['DETECTOR']][1])

    elif inst.upper() == 'MIRI':
        try:
            if (hdul[0].header['SUBARRAY'] == 'FULL' and
                    hdul[0].header['APERNAME'] != 'MIRIM_SLIT'):
                # For MIRI, sometimes we have subarray apertures for TA
                # but the images are actually full array.
                # In which case use the detector coordinates.
                xref = ap.XDetRef - 1   # siaf uses 1-based counting
                yref = ap.YDetRef - 1   # ditto
            elif (hdul[0].header['APERNAME'] == 'MIRIM_TASLITLESSPRISM' and
                  hdul[0].header['SUBARRAY'] == 'SLITLESSPRISM'):
                # This one's weird. It's pointed using TASLITLESSPRISM
                # but read out usng SLITLESSPRISM.
                ap_slitlessprism = siaf.apertures['MIRIM_SLITLESSPRISM']
                # convert coords from TASLITLESSPRISM to SLITLESSPRISM.
                xref, yref = ap_slitlessprism.det_to_sci(ap.XDetRef,
                                                         ap.YDetRef)
                xref, yref = xref - 1, yref - 1  # siaf uses 1-based counting
            elif hdul[0].header['APERNAME'] == 'MIRIM_SLIT':
                # This case is tricky. TACONFIRM image in slit type aperture,
                # which doesn't have Det or Sci coords defined.
                # It's made even more complex by filter-dependent MIRI offsets.
                print("TODO need to get more authoritative intended target "
                      "position for MIRIM TA CONFIRM")
                xref, yref = 317, 301
            elif (hdul[0].header['APERNAME'].endswith('_UR') or
                  hdul[0].header['APERNAME'].endswith('_CUR')):
                # Coronagraphic TA, pointed using special TA subarrays
                # but read out using the full coronagraphic subarray.
                # Similar to how TASLITLESSPRISM works.
                ap_subarray = 'MIRIM_'+hdul[0].header['SUBARRAY']
                ap_subarray = siaf.apertures[ap_subarray]
                # convert coords from TA subarray to coron subarray.
                xref, yref = ap_subarray.det_to_sci(ap.XDetRef, ap.YDetRef)
                xref, yref = xref - 1, yref - 1  # siaf uses 1-based counting
            else:
                xref = ap.XSciRef - 1   # siaf uses 1-based counting
                yref = ap.YSciRef - 1   # ditto
        except TypeError:  # LRS slit type doesn't have X/YSciRef
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
                zoom=None,
                **kwargs):
    """
    Retrieve and display a target acquisition image(s).

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
        normalization, colormap, and background level. Default False.
    ta_expnum : int, optional
        The TA exposure number to display (1-based indexing).
        Required if multiple TA exposures are available. Default None.
    mark_reference_point : bool, optional
        Mark the expected target position in the image? Default True.
    mark_apername : bool, optional
        Annotate the aperture of the guider used? Default True.
    zoom : int, optional
        Zoom into the center of the image. If provided, a zoomed inset
        centered on the reference point will be displayed. The value specifies
        the width (in pixels) of the zoom region. Default is None.

    Returns
    -------
    If return_handles is True, return tuple where:
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
        - inset_ax : matplotlib.axes.Axes
            The matplotlib Axes object where the zoom image is displayed.
    """

    # Check inputs.
    inst = inst.upper()

    # Retrieve and validate the TA image(s).
    # If there are multiple TA images, pick which one to display.
    title_extra = ''
    hdul = get_visit_ta_image(visitid, inst=inst, **kwargs)
    if not hdul:
        raise RuntimeError("No TA images found for that visit.")
    if isinstance(hdul, list) and not isinstance(hdul, fits.HDUList):
        if ta_expnum is None:
            raise ValueError(
                f"Specify `ta_expnum=<n>` to select from {len(hdul)} "
                "TA exposures (1-based indexing)."
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
    ax.imshow(ta_img - bglevel, norm=norm, cmap=cmap, origin='lower')
    ax.set_title(
        f"{inst} TA{title_extra} on {visitid}\n"
        f"{hdul[0].header['DATE-OBS']} {hdul[0].header['TIME-OBS'][:8]}"
    )
    ax.set_ylabel("Pixels Y")
    ax.set_xlabel("Pixels X")
    annotate_secondary_axes_arcsec(ax, ta_img - bglevel, wcs)

    ax.text(0.05, 0.95, annotation_text, color='k',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(facecolor='0.75', alpha=0.9,
                      edgecolor='none', boxstyle='round,pad=0.3'))
    if inst == 'MIRI':
        _crop_display_for_miri(ax, hdul)

    # Mark the TA reference point in the image?
    # i.e. where was the target 'supposed to be' in any given observation.
    if mark_reference_point:
        xref, yref = get_ta_reference_point(inst, hdul, ta_expnum)
        ax.axvline(xref, color='0.75', alpha=0.7, ls='--')
        ax.axhline(yref, color='0.75', alpha=0.7, ls='--')
        ax.text(xref, yref + 2, 'SIAF', color='k',
                verticalalignment='center', horizontalalignment='right',
                bbox=dict(facecolor='0.75', alpha=0.5,
                          edgecolor='none', boxstyle='round,pad=0.3'))

    # Mark the aperture for the guider used?
    if mark_apername:
        aperture_name = hdul[0].header['APERNAME']
        ax.text(0.95, 0.95,
                aperture_name+f"\n using {which_guider_used(visitid)}",
                color='k', transform=ax.transAxes,
                horizontalalignment='right', verticalalignment='top',
                bbox=dict(facecolor='0.75', alpha=0.9,
                          edgecolor='none', boxstyle='round,pad=0.3'))

    # Plot a zoomed in region of the image?
    inset_ax = None
    if zoom is not None:
        # Create the inset axis.
        inset_ax = ax.inset_axes([1.3, 0.2, 0.8, 0.8], transform=ax.transAxes)
        inset_ax.imshow(ta_img,  # np.full_like(ta_img, np.nan),
                        norm=norm, origin='lower')
        inset_ax.set_xlim(xref-(zoom/2), xref+(zoom/2))
        inset_ax.set_ylim(yref-(zoom/2), yref+(zoom/2))
        annotate_secondary_axes_arcsec(inset_ax, ta_img-bglevel, wcs)
        inset_ax.set_ylabel("Pixels Y")
        inset_ax.set_xlabel("Pixels X")
        inset_ax.set_facecolor('black')

        # Optionally, mark reference point in inset.
        if mark_reference_point:
            inset_ax.axvline(xref, color='0.75', alpha=0.9, ls='--')
            inset_ax.axhline(yref, color='0.75', alpha=0.9, ls='--')
            inset_ax.text(xref, yref + 2, 'SIAF', color='k',
                          verticalalignment='center',
                          horizontalalignment='right',
                          bbox=dict(facecolor='0.75', alpha=0.9,
                                    edgecolor='none',
                                    boxstyle='round,pad=0.3'))

    if return_handles:
        return hdul, ax, norm, cmap, bglevel, inset_ax
    return


def ta_analysis(data_product,
                verbose=True,
                plot=False,
                output_dir='./',
                zoom=20,
                **kwargs):
    """
    Retrieve NIRCam/MIRI Target Acquisition (TA)
    image(s) and evaluate performance.

    Parameters
    ----------
    data_product : str
        JWST data product FITS file.
    verbose : bool, optional
        If True, displays detailed progress messages and logs during execution.
    plot : bool, optional
        If True, plot the TA image.
    output_dir, str, optional
        Output directory for analysis images.
    zoom : int, optional
        Zoom into the center of the image? If provided, a zoomed inset
        centered on the reference point will be displayed. The value specifies
        the width (in pixels) of the zoom region. Default is None.

    Returns
    -------
    xc, yc : float, float
        The star center in the science aperture (pixels; 0 indexed).
    """

    # Check inputs.
    # Standardize format to 'VPPPPPOOOVVV'.
    visitid = get_visitid('V'+fits.getval(data_product, 'VISIT_ID'))
    inst = fits.getval(data_product, 'INSTRUME').upper()

    # Load SIAF.
    siaf = get_siaf(inst)

    # Retrieve TA images for the visit.
    ta_imgs = get_visit_ta_image(visitid, inst=inst, **kwargs)
    if not ta_imgs:
        raise RuntimeError(f"No TA image found for visit {visitid}")
        return None
    n_ta_imgs = (1 if isinstance(ta_imgs, astropy.io.fits.HDUList) else len(ta_imgs))

    # Initialize plotting.
    if plot:
        nrows = 2 if inst == 'MIRI' else 3
        fig, axes = plt.subplots(figsize=(15, 20), nrows=nrows)

    # Get and plot the observed TA image -- plot last one.
    i_ta = n_ta_imgs - 1
    if plot:
        (hdul, ax, norm, cmap, bglevel, inset_ax) = show_ta_img(visitid, ax=axes[i_ta],
                                                                return_handles=True, inst=inst,
                                                                ta_expnum=i_ta + 1,
                                                                zoom=zoom,
                                                                verbose=False,
                                                                **kwargs)
    else:
        if isinstance(ta_imgs, list) and not isinstance(ta_imgs, fits.HDUList):
            hdul = ta_imgs[i_ta]

    # Determine reference point (intended target position).
    xref, yref = get_ta_reference_point(inst, hdul, i_ta + 1)
    aperture_text = (f"Intended target position (from SIAF): "
                     f"{xref:.2f}, {yref:.2f}")

    # Determine aperture.
    ta_aperture = siaf.apertures[hdul[0].header['APERNAME']]
    if inst in ['NIRCAM', 'MIRI']:
        full_ap = (siaf[ta_aperture.AperName[:5] + "_FULL"]
                   if inst == "NIRCAM" else siaf["MIRIM_FULL"])

        # MIRI uses a larger kernel to handle border pixels.
        interp_kernel_size = (5, 5) if inst == 'NIRCAM' else (11, 11)

        # Adjust aperture for MIRI slit type.
        if inst == 'MIRI' and hdul[0].header['APERNAME'] == 'MIRIM_SLIT':
            # This slit type aperture which doesn't have pixel transforms!
            # Therefore use the full aperture for coord transforms.
            ta_aperture = full_ap

    # Extract science data and clean the image.
    im_obs_clean = hdul['SCI'].data.copy()  # Get science data.
    im_obs_clean[hdul['DQ'].data & 1] = np.nan  # Mask DO_NOT_USE pixels.
    nan_replace = astropy.convolution.interpolate_replace_nans
    im_obs_clean = nan_replace(im_obs_clean,  # Interpolate over masked pix
                               kernel=np.ones(interp_kernel_size))
    im_obs_clean[np.isnan(im_obs_clean)] = 0  # Replace remaining NaNs.

    # ---------- EXTRACT WCS INFORMATION ----------

    # Open the data model to access WCS and TA metadata.
    model = datamodels.open(hdul)
    wcs_ta = model.meta.wcs

    # Create SkyCoord object for target coord (RA, Dec) in ICRS frame.
    targ_coords = astropy.coordinates.SkyCoord(ra=model.meta.target.ra,
                                               dec=model.meta.target.dec,
                                               frame='icrs', unit=u.deg)

    # Convert celestial coordinates (RA, Dec) to pixel coordinates using WCS.
    targ_coords_pix = wcs_ta.world_to_pixel(targ_coords)  # (x,y)

    # Format WCS information as text for output.
    wcs_text = (f"Expected from WCS: {targ_coords_pix[0]:.2f}, "
                f"{targ_coords_pix[1]:.2f}")

    # Plot WCS-derived pixel coordinates if plotting is enabled.
    if plot:
        ax.scatter(targ_coords_pix[0], targ_coords_pix[1], color='magenta', marker='+', s=50)
        ax.text(targ_coords_pix[0], targ_coords_pix[1] + 2, 'WCS', color='magenta',
                verticalalignment='bottom', horizontalalignment='center')
        inset_ax.scatter(targ_coords_pix[0], targ_coords_pix[1], color='magenta', marker='+', s=50)
        inset_ax.text(targ_coords_pix[0], targ_coords_pix[1] + 2, 'WCS', color='magenta',
                      verticalalignment='bottom', horizontalalignment='center')

    # ---------- EXTRACT OSS CENTROID INFORMATION ----------
    try:
        # Attempt to retrieve the on-board TA centroid measurement
        # from the OSS logs in the engineering DB in MAST.
        osslog = get_ictm_event_log(hdul[0].header['VSTSTART'],
                                    hdul[0].header['VISITEND'])

        # Extract the OSS centroids from the log.
        try:
            visit_id = 'V' + hdul[0].header['VISIT_ID']
            oss_cen = extract_oss_TA_centroids(osslog, visit_id)[i_ta]

            # Convert from full-frame (as used by OSS)
            # to detector subarray coords for TA:
            oss_cen_ta = ta_aperture.det_to_sci(*oss_cen)

            # For MIRI only, deal with the fact that the image data can be
            # full array even if the aperture is subarray.
            if inst == 'MIRI':
                subarray = hdul[0].header['SUBARRAY']
                if subarray == 'FULL':
                    oss_cen_ta = np.asarray(full_ap.det_to_sci(*oss_cen))
                elif subarray == 'SLITLESSPRISM':
                    # Special case, pointed using TASLITLESSPRISM
                    # but read out using SLITLESSPRISM.
                    slitlessprism_ap = siaf['MIRIM_SLITLESSPRISM']
                    oss_cen_ta = np.asarray(
                        slitlessprism_ap.det_to_sci(*oss_cen))
                elif subarray.startswith('MASK'):
                    coron_ap = siaf['MIRIM_' + subarray]
                    oss_cen_ta = np.asarray(coron_ap.det_to_sci(*oss_cen))

            # Centroid in the TA subarray coordinate frame.
            oss_cen_ta = np.asarray(oss_cen_ta) - 1  # 0 indexed

        except RuntimeError:
            if verbose:
                log.info("Could not parse TA coordinates from log. TA may have failed?")
                # Fallback tp list of NaNs.
                oss_cen_ta = (np.nan, np.nan)
                oss_centroid_text = "No OSS centroid; TA failed"

        # Compute offset of OSS centroid vs. intended position.
        if not np.isnan(oss_cen_ta[0]):
            deltapos = []
            deltapos_type = []
            deltapos.append((oss_cen_ta[0] - xref, oss_cen_ta[1] - yref))  # OSS centroid - REF
            deltapos_type.append('OSS - Intended')
            deltapos.append((oss_cen_ta[0] - targ_coords_pix[0], oss_cen_ta[1] - targ_coords_pix[1]))  # STPSF centroid - WCS
            deltapos_type.append('OSS - WCS')

        if plot:
            ax = axes[i_ta]  # Select correct subplot for TA image.
            oss_centroid_text = (f"OSS centroid: {oss_cen_ta[0]:.2f}, {oss_cen_ta[1]:.2f}" or "")
            ax.scatter(*oss_cen_ta, color='r', marker='x', s=50)
            ax.text(*oss_cen_ta, 'OSS  ', color='r', verticalalignment='center', horizontalalignment='right')
            inset_ax.scatter(*oss_cen_ta, color='r', marker='x', s=50)
            inset_ax.text(*oss_cen_ta, 'OSS  ', color='r', verticalalignment='center', horizontalalignment='right')

    except ImportError:
        oss_centroid_text = ""

    # ---------- CALCULATE WEBPSF CENTROID ----------

    # Apply a mask around the border pixels to prioritize the PSF center
    # and ignore bad/hot pixels near edges.
    # This makes this alignment step more robust.
    nm = 6   # Pixel boarder width.
    border_mask = np.ones_like(im_obs_clean)
    border_mask[:nm] = border_mask[-nm:] = 0
    border_mask[:, :nm] = border_mask[:, -nm:] = 0

    # Special cases for different apertures.
    aperture_name = hdul[0].header['APERNAME']
    if aperture_name == 'MIRIM_TALRS':
        # Special case, it's a full frame image but enforce just
        # centroiding in the TA region of interest.
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
    if np.isnan(oss_cen_ta[0]):
        # If TA failed, fallback to `fwcentroid` from webbpsf.
        oss_cen_ta = cen
        deltapos = []
        deltapos_type = []
        deltapos.append((cen[1] - xref, cen[0] - yref))  # STPSF centroid - REF
        deltapos_type.append('fwcentroid - Intended')
        deltapos.append((cen[1] - targ_coords_pix[0], cen[0] - targ_coords_pix[1]))  # STPSF centroid - WCS
        deltapos_type.append('fwcentroid - WCS')

        oss_centroid_text = (f"STPSF measured centroid: {cen[1]:.2f}, {cen[0]:.2f}" or "")

        # Plot the centroid if plotting is enabled.
        if plot:
            ax.scatter(cen[1], cen[0], color='cyan', marker='+', s=50)
            ax.text(cen[1], cen[0], 'webbpsf', color='cyan',
                    verticalalignment='center', horizontalalignment='left')
            inset_ax.scatter(cen[1], cen[0], color='cyan', marker='+', s=50)
            inset_ax.text(cen[1], cen[0], 'webbpsf', color='cyan',
                          verticalalignment='center', horizontalalignment='left')

    # ---------- CALCULATE OFFSETS ----------

    # Construct and annotate image text.
    image_text = (f"Pixel coordinates (x, y; 0-based):\n{oss_centroid_text}\n"
                  f"{wcs_text}\n{aperture_text}\n"
                  f"$\\Delta$pos ({deltapos_type[0]}): {deltapos[0][0]:.2f}, {deltapos[0][1]:.2f}\n"
                  f"$\\Delta$pos ({deltapos_type[1]}): {deltapos[1][0]:.2f}, {deltapos[1][1]:.2f}")
    if plot:
        ax.text(0.95, 0.04, image_text, horizontalalignment='right',
                verticalalignment='bottom', transform=ax.transAxes, color='k',
                bbox=dict(facecolor='0.75', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.3'))

    if oss_cen_ta is not None:

        # Compute initial delta RA/DEC (WCS to OSS).
        ta_cen_coords = wcs_ta.pixel_to_world(*oss_cen_ta)
        dra, ddec = ((targ_coords.ra - ta_cen_coords.ra),
                     (targ_coords.dec - ta_cen_coords.dec))  # WCS - OSS
        original_wcs_offset_radec = (dra.to(u.arcsec), ddec.to(u.arcsec))
        original_wcs_offset_pix = (np.asarray(targ_coords_pix) - oss_cen_ta)

        # Obtain the initial WCS for the science observation (not TA).
        # Perform the same RA/DEC adjustments to the TA and SCI WCS.
        wcs_sci = datamodels.open(data_product).meta.wcs
        sci_aperture = siaf.apertures[fits.getval(data_product, 'APERNAME', ext=0)]
        wcs_ta_adj = adjust_wcs(wcs_ta, *original_wcs_offset_radec)
        wcs_sci_adj = adjust_wcs(wcs_sci, *original_wcs_offset_radec)

        # Compute adjusted centroid and target coordinates in SCI and TA.
        ta_cen_coords_adj = wcs_ta_adj.pixel_to_world(*oss_cen_ta)
        ta_targ_coords_adj_pix = wcs_ta_adj.world_to_pixel(targ_coords)
        ta_targ_coords_adj = wcs_ta_adj.pixel_to_world(*ta_targ_coords_adj_pix)
        sci_targ_coords_adj_pix = wcs_sci_adj.world_to_pixel(targ_coords)  # 0 indexed

        # Compute any residuals after adjustment.
        residual_pix = (np.asarray(ta_targ_coords_adj_pix) - oss_cen_ta)
        dra_residual = (ta_targ_coords_adj.ra - ta_cen_coords_adj.ra)
        ddec_residual = (ta_targ_coords_adj.dec - ta_cen_coords_adj.dec)
        residual_radec = (dra_residual.to(u.arcsec), ddec_residual.to(u.arcsec))

        if plot:
            ax.scatter(ta_targ_coords_adj_pix[0], ta_targ_coords_adj_pix[1], color='k', marker='+', s=50)
            ax.text(ta_targ_coords_adj_pix[0], ta_targ_coords_adj_pix[1] + 2, 'WCS_ADJ', color='k',
                    verticalalignment='bottom', horizontalalignment='center')
            inset_ax.scatter(ta_targ_coords_adj_pix[0], ta_targ_coords_adj_pix[1], color='k', marker='+', s=50)
            inset_ax.text(ta_targ_coords_adj_pix[0], ta_targ_coords_adj_pix[1] + 2, 'WCS_ADJ', color='k',
                          verticalalignment='center', horizontalalignment='left')
            target_ax = inset_ax if zoom else ax

            # Define the annotations.
            annotations = [
                (f"WCS $\\Delta$RA, $\\Delta$Dec = "
                 f"({original_wcs_offset_radec[0]:.3f}, {original_wcs_offset_radec[1]:.3f}) arcsec", 'magenta', 0.80),
                (f"WCS_ADJ $\\Delta$RA, $\\Delta$Dec = "
                 f"({residual_radec[0]:.3f}, {residual_radec[1]:.3f}) arcsec", 'k', 0.75)]

            # Add annotations to the plot.
            for text, color, y_pos in annotations:
                target_ax.text(0.90, y_pos, text,
                               horizontalalignment='right',
                               verticalalignment='bottom',
                               transform=target_ax.transAxes,
                               fontsize=10, color=color,
                               bbox=dict(facecolor='0.75', alpha=0.9,
                                         edgecolor='none',
                                         boxstyle='round,pad=0.3'))

        if verbose:
            log.info(f'--> TA Analysis: Target coordinates (RA, Dec) [deg]: ({targ_coords.ra.deg:.6f}, {targ_coords.dec.deg:.6f})')
            log.info(f'                 Target coordinates (x, y) [pix]: ({targ_coords_pix[0]:.3f}, {targ_coords_pix[1]:.3f}) (sci frame in {ta_aperture.AperName}, 0-based)')
            cen_type = 'OSS' if 'OSS' in deltapos_type[1] else 'STPSF'
            log.info(f'                 Target coordinates from {cen_type} centroid (x, y) [pix]: ({oss_cen_ta[0]:.6f}, {oss_cen_ta[1]:.6f}) (sci frame in {ta_aperture.AperName}, 0-based)')
            log.info(f'                 Target coordinates from {cen_type} centroid (RA, Dec) [deg]: ({ta_cen_coords.ra.deg:.6f}, {ta_cen_coords.dec.deg:.6f})')
            log.info(f'                 Calculating offsets ...')
            log.info(f'                 Target coordinates WCS - {cen_type} offset (RA, Dec) [deg]: ({original_wcs_offset_radec[0]:.3f}, {original_wcs_offset_radec[1]:.3f})')
            log.info(f'                 Target coordinates WCS - {cen_type} offset (x, y) [pix]: ({original_wcs_offset_pix[0]:.3f}, {original_wcs_offset_pix[1]:.3f})')
            log.info(f'                 Adjusting TA WCS based on offset ...')
            log.info(f'                 Target coordinates from adjusted WCS (RA, Dec) [deg]: ({ta_targ_coords_adj.ra.deg:.6f}, {ta_targ_coords_adj.dec.deg:.6f})')
            log.info(f'                 Target coordinates from adjusted WCS (x, y) [pix]: ({ta_targ_coords_adj_pix[0]:.3f}, {ta_targ_coords_adj_pix[1]:3f}) (sci frame in {ta_aperture.AperName}, 0-based)')
            log.info(f'                 Target coordinates Residuals (RA, Dec) [deg]: ({residual_radec[0]:.6f}, {residual_radec[1]:.6f})')
            log.info(f'                 Target coordinates Residuals (x, y) [pix]: ({residual_pix[0]:.3f}, {residual_pix[1]:.3f})')
            log.info(f'                 Adjusting Science WCS based on same offsets ...')
            log.info(f'                 Target coordinates (x, y) [pix]: ({sci_targ_coords_adj_pix[0]:.3f}, {sci_targ_coords_adj_pix[1]:.3f}) (sci frame in {sci_aperture.AperName}, 0-based)')

    # Manage axes visibility for single TA image.
    if plot:
        axes[0].set_visible(False)

        cb = fig.colorbar(ax.images[0], ax=axes[1], orientation='horizontal',
                          label=hdul['SCI'].header['BUNIT'], fraction=0.05,
                          shrink=0.9, pad=0.07)
        ticks = cb.ax.get_xticks()
        cb.ax.set_xticks([t for t in ticks if t > 0.1])

        plt.tight_layout()

        outname = os.path.join(output_dir,
                               f'{inst.lower()}_ta_analysis_{os.path.basename(data_product[:-13])}.pdf')
        plt.savefig(outname)
        log.info(f'                 TA analysis saved: {outname}')

    if oss_cen_ta is not None:
        xc, yc = sci_targ_coords_adj_pix[0], sci_targ_coords_adj_pix[1]
        return xc, yc
