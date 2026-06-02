import logging
import sep
from pathlib import Path
from typing import Any, Literal, cast
from astroquery.gaia import Gaia
import matplotlib.pylab as plt
from astropy.visualization import simple_norm
from photutils.psf import FittableImageModel
from astropy.modeling import fitting
import spaceKLIP.utils as ut
import numpy as np
from scipy import ndimage
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.wcs import WCS
from scipy.ndimage import label
from scipy.signal import fftconvolve
from photutils.detection import DAOStarFinder

# Set up log.
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def fetch_gaia_for_image_fov(
    image: np.ndarray,
    header,
    gaia_table: str = "gaiadr3.gaia_source",
    row_limit: int = -1,
    verbose: bool = False,
) -> Table:
    """Estimate image FOV from WCS and query Gaia over that footprint.

    Parameters
    ----------
    image : 2D-array
        Image data used only for its shape.
    header : astropy.io.fits.Header
        FITS header containing the celestial WCS for the image.
    gaia_table : str, optional
        Gaia TAP table to query. Defaults to Gaia DR3 source table.
    row_limit : int, optional
        Max number of returned rows. Use ``-1`` for no row limit.
    verbose : bool, optional
        Passed to ``Gaia.launch_job_async``.

    Returns
    -------
    astropy.table.Table
        New Astropy table with selected Gaia columns plus WCS-derived ``x`` and ``y``.

    """
    data = np.asarray(image)
    if data.ndim == 3:
        data = data[0, :, :]
    if data.ndim == 2:
        pass
    else:
        raise ValueError(f"image must be a 3D or 2D, got shape {data.shape}")

    cel_wcs = WCS(header, naxis=2).celestial
    ny, nx = data.shape
    x_center = (nx) / 2.0
    y_center = (ny) / 2.0

    center_ra_deg, center_dec_deg = cel_wcs.all_pix2world(x_center, y_center, 0)
    pix_scales = np.sqrt(header['PIXAR_A2'])
    fov_x_deg = float(nx * pix_scales)/3600
    fov_y_deg = float(ny * pix_scales)/3600
    radius_deg = 0.5 * float(np.hypot(fov_x_deg, fov_y_deg))

    query = (
        "SELECT source_id, ra, dec, parallax, parallax_error, phot_g_mean_mag FROM "
        f"{gaia_table} "
        "WHERE 1=CONTAINS(" 
        "POINT('ICRS', ra, dec), "
        f"CIRCLE('ICRS', {center_ra_deg:.12f}, {center_dec_deg:.12f}, {radius_deg:.12f})"
        ")"
    )

    old_table = Gaia.MAIN_GAIA_TABLE
    old_limit = Gaia.ROW_LIMIT
    try:
        Gaia.MAIN_GAIA_TABLE = gaia_table
        Gaia.ROW_LIMIT = int(row_limit)
        job = Gaia.launch_job_async(query=query, dump_to_file=False, verbose=verbose)
        raw_result = job.get_results()
    finally:
        Gaia.MAIN_GAIA_TABLE = old_table
        Gaia.ROW_LIMIT = old_limit

    result = raw_result.copy()

    # Add detector pixel coordinates from catalog sky coordinates.
    ra_col = "ra" if "ra" in result.colnames else ("RA" if "RA" in result.colnames else None)
    dec_col = "dec" if "dec" in result.colnames else ("DEC" if "DEC" in result.colnames else None)
    if ra_col is None or dec_col is None:
        log.warning("Gaia result does not include ra/dec columns; returning sky-only table.")
        return result

    ra_arr = np.asarray(np.ma.filled(np.ma.asarray(result[ra_col]), np.nan), dtype=float)
    dec_arr = np.asarray(np.ma.filled(np.ma.asarray(result[dec_col]), np.nan), dtype=float)
    x, y = cel_wcs.all_world2pix(ra_arr, dec_arr, 0)

    if "x" in result.colnames:
        result["x"] = np.asarray(x, dtype=float)
    else:
        result["x"] = np.asarray(x, dtype=float)
    if "y" in result.colnames:
        result["y"] = np.asarray(y, dtype=float)
    else:
        result["y"] = np.asarray(y, dtype=float)

    return result

def mask_core(data,radius_core,showplots=False,cmap='Greys_r'):
    # Mask the PSF to exclude the core
    # Define a circular mask for the saturated core in the PSF data
    y_grid, x_grid = np.indices(data.shape)
    data_core_mask = (x_grid - data.shape[1]//2)**2 + (y_grid - data.shape[0]//2)**2 < radius_core**2
    # masked_data = np.ma.masked_array(data, mask=data_core_mask)
    masked_data = data.copy()
    masked_data[data_core_mask] = 0
    if showplots:
        # Display the generated PSF
        norm = simple_norm(masked_data, 'log')
        plt.imshow(masked_data, origin='lower', cmap=cmap,norm=norm)
        plt.colorbar()
        plt.title('Generated PSF for F444W (Saturated Core Excluded)')
        plt.show()
    return masked_data

def estimate_bkg_and_rms(data2d, edge_width=5):
    """Estimate background median and RMS from cutout border pixels.

    Parameters
    ----------
    data2d : 2D-array
        Image cutout.
    edge_width : int, optional
        Width (pixels) of the border used for the estimate.

    Returns
    -------
    bkg : float
        Background median.
    rms : float
        Robust RMS estimate.

    """
    data2d = np.asarray(data2d, dtype=float)
    ny, nx = data2d.shape
    ew = int(max(1, min(edge_width, ny // 2, nx // 2)))
    edge = np.zeros_like(data2d, dtype=bool)
    edge[:ew, :] = True
    edge[-ew:, :] = True
    edge[:, :ew] = True
    edge[:, -ew:] = True
    vals = data2d[edge]
    vals = vals[np.isfinite(vals)]
    if vals.size < 10:
        vals = data2d[np.isfinite(data2d)]
    if vals.size == 0:
        return 0.0, 1.0

    mean, med, std = sigma_clipped_stats(vals, sigma=3.0, maxiters=5)
    return float(med), float(std if std > 0 else 1.0)

def downsample_psf_to_detector(psf, oversampling):
    """Downsample an oversampled PSF to detector sampling by summing blocks.

    Parameters
    ----------
    psf : 2D-array
        Oversampled PSF.
    oversampling : int
        Oversampling factor (PSF pixels per detector pixel).

    Returns
    -------
    2D-array
        PSF on detector sampling.

    """
    if oversampling == 1:
        return np.asarray(psf, dtype=float)

    psf = np.asarray(psf, dtype=float)
    ny_os, nx_os = psf.shape
    if (ny_os % oversampling) != 0 or (nx_os % oversampling) != 0:
        raise ValueError(
            f"PSF shape {psf.shape} not divisible by oversampling={oversampling}."
        )
    ny = ny_os // oversampling
    nx = nx_os // oversampling
    # Sum (not mean) preserves total flux normalization.
    return psf.reshape(ny, oversampling, nx, oversampling).sum(axis=(1, 3))

def fit_psf(
    masked_psf_data,
    data,
    oversampling=1,
    radius_core=0,
    fit_radius=None,
    search_radius=None,
    bkg_subtract=True,
    edge_bkg_width=8,
    two_pass=True,
    showplots=False,
    cmap='Greys_r',
    stretch='linear'
):
    """Fit a (possibly oversampled) PSF model to an image cutout.

    Parameters
    ----------
    masked_psf_data : 2D-array
        PSF model image (may be oversampled; see ``oversampling``).
    data : 2D-array
        Image cutout to fit.
    mask_size : int, optional
        Reserved/legacy argument (kept for API compatibility).
    oversampling : int, optional
        Oversampling factor of the PSF model relative to the data.
    radius_core : float, optional
        Radius (in *data* pixels) of the saturated/NaN core to exclude from the
        fit.
    fit_radius : float, optional
        Radius (in data pixels) defining the fitting region. If None, fits the
        full cutout.
    search_radius : float, optional
        Search radius (pixels) for the matched-filter initialization.
    bkg_subtract : bool, optional
        If True, subtract a robust background estimate.
    edge_bkg_width : int, optional
        Border width (pixels) for background/RMS estimation.
    two_pass : bool, optional
        If True and ``fit_radius`` is set, do a broad pass followed by a tighter
        pass.
    showplots : bool, optional
        If True, show a diagnostic plot.

    Returns
    -------
    fitted_x_pos, fitted_y_pos, fitted_flux : float
        Best-fit PSF center and flux in cutout coordinates.

    Notes
    -----
    Photutils/Astropy convention: ``x`` is the *column* coordinate and ``y`` is
    the *row* coordinate.

    """

    data_cutout = np.asarray(data, dtype=float)

    # Robust background subtraction is critical at low S/N.
    bkg, rms = estimate_bkg_and_rms(data_cutout, edge_width=edge_bkg_width)
    if bkg_subtract:
        data_fit = data_cutout - bkg
    else:
        data_fit = data_cutout

    # Use the PSF as the model (with the core optionally masked).
    # IMPORTANT: if the PSF is oversampled w.r.t. the data, tell photutils.
    psf_model = FittableImageModel(masked_psf_data, oversampling=oversampling)

    # Use the LevMarLSQFitter to fit the PSF to the data.
    fitter = fitting.LevMarLSQFitter()

    ny, nx = data_fit.shape
    yy, xx = np.mgrid[0:ny, 0:nx]

    # For saturated stars we want to keep the masked core fixed on the NaN core.
    core_mask_x = (nx - 1) / 2
    core_mask_y = (ny - 1) / 2
    if radius_core and radius_core > 0 and np.any(~np.isfinite(data_cutout)):
        _, core_mask_x, core_mask_y = estimate_nan_core(data_cutout, margin=0)

    def _make_weights(center_x, center_y, _fit_radius):
        w = np.zeros_like(data_fit, dtype=float)
        w[finite] = 1.0 / (rms**2 + 1e-30)

        if _fit_radius is not None:
            rr2 = (xx - float(center_x)) ** 2 + (yy - float(center_y)) ** 2
            w[rr2 > float(_fit_radius) ** 2] = 0.0

        if radius_core and radius_core > 0:
            rr2 = (xx - float(core_mask_x)) ** 2 + (yy - float(core_mask_y)) ** 2
            w[rr2 < float(radius_core) ** 2] = 0.0
        return w

    # Reasonable initial guesses matter a lot for position fitting.
    x_center = (nx - 1) / 2
    y_center = (ny - 1) / 2

    finite = np.isfinite(data_fit)
    if np.any(finite):
        peak_snr = float(np.nanmax(data_fit[finite]) / (rms + 1e-12))
    else:
        peak_snr = 0.0

    psf_det = downsample_psf_to_detector(masked_psf_data, oversampling)
    psf_det = np.asarray(psf_det, dtype=float)
    if np.all(psf_det == 0) or not np.isfinite(psf_det).any():
        raise ValueError("PSF is all zeros or non-finite")
    # Normalize for correlation stability.
    psf_det = psf_det / (np.nansum(psf_det) + 1e-30)
    # Cross-correlation peak gives a good starting point for faint sources.
    corr = fftconvolve(
        np.nan_to_num(data_fit, nan=0.0),
        psf_det[::-1, ::-1],
        mode="same",
    )

    # Restrict peak search to an area where we expect the source to be.
    # This greatly reduces catastrophic failures at very low S/N.
    sr = search_radius
    if sr is None:
        sr = fit_radius
    if sr is not None:
        rr2 = (xx - x_center) ** 2 + (yy - y_center) ** 2
        corr = corr.copy()
        corr[rr2 > float(sr) ** 2] = -np.inf

    iy, ix = np.unravel_index(np.nanargmax(corr), corr.shape)

    # Subpixel peak estimate via quadratic interpolation in x and y.
    def _quad_peak(v_minus, v0, v_plus):
        denom = (v_minus - 2.0 * v0 + v_plus)
        if denom == 0:
            return 0.0
        return 0.5 * (v_minus - v_plus) / denom

    dx = 0.0
    dy = 0.0
    if 1 <= ix < (nx - 1):
        dx = _quad_peak(corr[iy, ix - 1], corr[iy, ix], corr[iy, ix + 1])
        dx = float(np.clip(dx, -1.0, 1.0))
    if 1 <= iy < (ny - 1):
        dy = _quad_peak(corr[iy - 1, ix], corr[iy, ix], corr[iy + 1, ix])
        dy = float(np.clip(dy, -1.0, 1.0))

    x0_init, y0_init = float(ix) + dx, float(iy) + dy

    psf_model.x_0.value = x0_init
    psf_model.y_0.value = y0_init

    # Flux guess: keep it positive; use peak*SOMETHING as crude initial scale.
    if np.any(finite):
        psf_model.flux.value = max(float(np.nanmax(data_fit[finite])), 0.0)
    else:
        psf_model.flux.value = 0.0

    # Parameter bounds: helps stability.
    # For saturated stars with masked cores, the position can become weakly constrained;
    # restrict it to remain near the initial guess.
    if radius_core and radius_core > 0:
        delta = float(max(3, int(radius_core)))
        psf_model.x_0.bounds = (max(0.0, x0_init - delta), min(float(nx - 1), x0_init + delta))
        psf_model.y_0.bounds = (max(0.0, y0_init - delta), min(float(ny - 1), y0_init + delta))
    else:
        psf_model.x_0.bounds = (0.0, float(nx - 1))
        psf_model.y_0.bounds = (0.0, float(ny - 1))
    psf_model.flux.bounds = (0.0, np.inf)

    # Perform the fit.
    # For low S/N, restricting too aggressively to a small radius around a potentially-wrong
    # initial guess can lock the optimizer onto the wrong solution. In that case we do a
    # broader first pass, then a tighter second pass.
    if fit_radius is not None and two_pass:
        first_pass_radius = float(fit_radius)
        if peak_snr < 10:
            first_pass_radius = float(fit_radius) * 2.0

        weights1 = _make_weights(psf_model.x_0.value, psf_model.y_0.value, first_pass_radius)
        fit1 = fitter(psf_model, xx, yy, data_fit, weights=weights1, filter_non_finite=True)

        # Recenter for second pass.
        psf_model.x_0.value = fit1.x_0.value
        psf_model.y_0.value = fit1.y_0.value
        psf_model.flux.value = max(float(fit1.flux.value), 0.0)

        weights2 = _make_weights(psf_model.x_0.value, psf_model.y_0.value, float(fit_radius))
        fit_result = fitter(psf_model, xx, yy, data_fit, weights=weights2, filter_non_finite=True)
    else:
        weights = _make_weights(psf_model.x_0.value, psf_model.y_0.value, fit_radius)
        fit_result = fitter(psf_model, xx, yy, data_fit, weights=weights, filter_non_finite=True)

    # Step 8: Output the fitted flux and position
    fitted_flux = fit_result.flux.value
    fitted_x_pos = fit_result.x_0.value
    fitted_y_pos = fit_result.y_0.value

    if showplots:
        log.info(f"Fitted Flux: {fitted_flux}")
        log.info(f"Fitted X Position: {fitted_x_pos}")
        log.info(f"Fitted Y Position: {fitted_y_pos}")
        log.info(f"Estimated background (median): {bkg}")
        log.info(f"Estimated RMS: {rms}")

        norm = simple_norm(data_fit, stretch)
        # Plot in the same convention used elsewhere in this script.
        plt.imshow(data_fit, origin='lower', cmap=cmap, norm=norm)
        plt.plot(fitted_x_pos, fitted_y_pos, 'ob')
        plt.colorbar()
        plt.title('Data to fit with fitted center')
        plt.show()

    return fitted_x_pos,fitted_y_pos,fitted_flux

def estimate_nan_core(data,
                      center=None,
                      margin=1,
                      nanmask=None
) -> tuple[int, float, float]:
    """Estimate centroid and radius of a connected non-finite (NaN/Inf) core.

    Parameters
    ----------
    data : 2D-array
        Image cutout.
    center : tuple of float, optional
        Starting point ``(x, y)`` for the flood-fill. If None, uses the image
        center.
    margin : int, optional
        Extra pixels added to the returned radius.
    nanmask: list, None, optional
        nanmask is a boolean array of the same shape as data, where True values indicate pixels to be treated as NaN
        in the analysis.

    Returns
    -------
    radius : int
        Radius in pixels of the connected non-finite region.
    x_center, y_center : float
        Region centroid in cutout coordinates.

    """
    nandata = np.asarray(data.copy())
    if nanmask is not None:
        nandata[nanmask.astype(bool)] = np.nan

    ny, nx = nandata.shape
    if center is None:
        cx, cy = (nx - 1) / 2, (ny - 1) / 2
    else:
        cx, cy = center

    bad = ~np.isfinite(nandata)

    sx = int(np.clip(round(cx), 0, nx - 1))
    sy = int(np.clip(round(cy), 0, ny - 1))

    # If the exact center is finite, look for a bad pixel close to the center.
    if not bad[sy, sx]:
        found = False
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                y = sy + dy
                x = sx + dx
                if 0 <= y < ny and 0 <= x < nx and bad[y, x]:
                    sy, sx = y, x
                    found = True
                    break
            if found:
                break
        if not found:
            return 0, float(cx), float(cy)

    # Flood-fill the connected bad region (4-connected).
    region = np.zeros_like(bad, dtype=bool)
    stack = [(sy, sx)]
    region[sy, sx] = True
    while stack:
        y, x = stack.pop()
        for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= yy < ny and 0 <= xx < nx and bad[yy, xx] and not region[yy, xx]:
                region[yy, xx] = True
                stack.append((yy, xx))

    if not np.any(region):
        return 0, float(cx), float(cy)

    yy, xx = np.indices(nandata.shape)
    x_cent = float(np.mean(xx[region]))
    y_cent = float(np.mean(yy[region]))
    rr = np.sqrt((xx - x_cent) ** 2 + (yy - y_cent) ** 2)
    radius = int(np.ceil(np.nanmax(rr[region])) + int(margin))
    return radius, x_cent, y_cent

def stars_extractor(data,
                    coords,
                    fow = 101,
                    pad_amount=0,
                    shifts = None,
                    method='fourier',
                    showplots=False,
                    cmap='Greys_r',
                    stretch='linear',
                    kwargs={}
):
    if shifts is None:
        #Just extract the tile at coordinates without shifts
        tile = data[int(round(coords[1]))-fow//2:int(round(coords[1]))+fow//2+1, int(round(coords[0]))-fow//2:int(round(coords[0]))+fow//2+1]
    else:
        #Create a bigger tile to shift, so we don't have to shift the entire image to minimize weird artifacts
        preshifttile = data[int(round(coords[1]))-(fow//2+pad_amount):int(round(coords[1]))+(fow//2+pad_amount+1),
                                   int(round(coords[0]))-(fow//2+pad_amount):int(round(coords[0]))+(fow//2+pad_amount+1)]
        shifteddata = ut.imshift(preshifttile, [shifts[0], shifts[1]], pad_amount=0, method=method, kwargs=kwargs)
        #Crop the shifted tile to the desired dimension
        tile = shifteddata[int(round(shifteddata.shape[1]//2))-fow//2:int(round(shifteddata.shape[1]//2))+fow//2+1, int(round(shifteddata.shape[0]//2))-fow//2:int(round(shifteddata.shape[0]//2))+fow//2+1]
    if showplots:
        norm = simple_norm(tile, stretch)
        plt.imshow(tile, origin='lower', norm=norm,cmap=cmap)
        plt.colorbar()
        plt.title(f'Extracted Star on integer coordinates')
        plt.show()

    return tile

def sextractor_flag_short(flag: int) -> str:
    """Return a short description for a SExtractor/SEP FLAGS bitmask.

    Parameters
    ----------
    flag : int
        SExtractor-style FLAGS value (bitmask).

    Returns
    -------
    str
        One-word summary (e.g. ``'ok'``, ``'saturated'``, ``'badpix'``).

    Notes
    -----
    The input is a bitmask; if multiple bits are set, this routine returns the
    highest-priority label.

    """
    f = int(flag)
    if f == 0:
        return "ok"

    # Priority: saturation and bad pixels often break photometry the most.
    if f & 4:
        return "saturated"
    if f & 16:
        return "badpix"
    if f & 1:
        return "edge"
    if f & 2:
        return "blended"
    if f & 8:
        return "neighbor"
    return "flagged"


def aperture_flag_short(ap_flag: int) -> str:
    """Return a short description for SEP aperture-photometry flags.

    Parameters
    ----------
    ap_flag : int
        Bitmask flag returned by ``sep.sum_circle`` (or similar).

    Returns
    -------
    str
        One-word summary (e.g. ``'ok'``, ``'truncated'``, ``'maskedpixels'``).

    Notes
    -----
    This routine decodes a bitmask and returns the highest-priority label.

    """
    f = int(ap_flag)
    if f == 0:
        return "ok"
    if f & getattr(sep, "APER_ALLMASKED", 64):
        return "allmasked"
    if f & getattr(sep, "APER_TRUNC", 16):
        return "truncated"
    if f & getattr(sep, "APER_NONPOSITIVE", 128):
        return "nonpositive"
    if f & getattr(sep, "APER_HASMASKED", 32):
        return "maskedpixels"
    return "flagged"


def _as_fixed_str_array(values: list[str], *, width: int = 64) -> np.ndarray:
    """Return a fixed-width unicode array for safe insertion into Astropy tables.

    Parameters
    ----------
    values : list of str
        Input strings.
    width : int, optional
        Fixed string width (characters).

    Returns
    -------
    numpy.ndarray
        Unicode array with dtype ``U<width>``.

    """
    if width <= 0:
        width = 1
    return np.asarray(values, dtype=f"U{int(width)}")

def select_table(
    objects_tbl: Table,
    separation_pix = None,
    center: Literal["centroid", "peak"] = "peak",
    peak_col: str = "peak",
    ap_snr: float | None= None,
    maxrat: float | None = None,
    flag_sel: list[int] | None = None,
    ap_flag_sel: list[int] | None = None,
    window_shape: Literal["circle", "square"] = "circle",
) -> Table:
    """Select detections using filters + non-maximum suppression.

    Parameters
    ----------
    objects_tbl : astropy.table.Table
        SEP detections table.
    separation_pix : float, optional
        If provided, keep only the brightest detection within this radius.
    center : {'centroid', 'peak'}, optional
        Coordinates used for the separation check.
    peak_col : str, optional
        Column used to rank detections (default: ``'peak'``).
    ap_snr : float, optional
        If provided, apply an SNR cut (uses ``'peak_snr'`` if ``peak_col='peak'``
        else uses ``'ap_snr'``).
    maxrat : float, optional
        If provided, remove elongated detections using the ``'ellipt'`` column.
    flag_sel : list of int, optional
        Keep only rows whose SEP detection flag (``'flag'``) is exactly in this list.
    ap_flag_sel : list of int, optional
        Keep only rows whose aperture flag (``'ap_flag'``) is exactly in this list.
    window_shape : {'circle', 'square'}, optional
        Neighborhood shape for the separation check.

    Returns
    -------
    astropy.table.Table
        Filtered table with a ``'ds9_id'`` column added.

    Notes
    -----
    SEP coordinates are 0-indexed numpy pixel coordinates.

    """
    # Work on a copy: this function is a selector and should not mutate inputs.
    objects_tbl = objects_tbl.copy()

    # Remove objects that are too elongated to be astrophysical.
    if maxrat is not None:
        if "ellipt" not in objects_tbl.colnames:
            raise ValueError("objects_tbl is missing required column 'ellipt' for maxrat filtering")
        objects_tbl = objects_tbl[np.asarray(objects_tbl["ellipt"], dtype=float) <= float(maxrat)].copy()

    # Remove objects with SNR lower than ap_snr.
    if ap_snr is not None:
        # If the user is ranking by peak, apply the SNR cut to the peak too.
        snr_col = "peak_snr" if peak_col == "peak" else "ap_snr"
        if snr_col not in objects_tbl.colnames:
            raise ValueError(f"objects_tbl is missing required column '{snr_col}' for snr filtering")
        objects_tbl = objects_tbl[np.asarray(objects_tbl[snr_col], dtype=float) >= float(ap_snr)].copy()

    # Filter by SEP detection flags (SExtractor-style bitmask stored as an int).
    # NOTE: As requested, this is an *exact match* on the integer value.
    if flag_sel is not None:
        if "flag" not in objects_tbl.colnames:
            raise ValueError("flag_sel was provided but objects_tbl has no 'flag' column")
        good = np.isin(np.asarray(objects_tbl["flag"], dtype=int), np.asarray(flag_sel, dtype=int))
        objects_tbl = objects_tbl[good].copy()

    # Filter by SEP aperture-photometry flags (returned by sep.sum_circle).
    # NOTE: As requested, this is an *exact match* on the integer value.
    if ap_flag_sel is not None:
        if "ap_flag" not in objects_tbl.colnames:
            raise ValueError("ap_flag_sel was provided but objects_tbl has no 'ap_flag' column")
        good = np.isin(np.asarray(objects_tbl["ap_flag"], dtype=int), np.asarray(ap_flag_sel, dtype=int))
        objects_tbl = objects_tbl[good].copy()

    if separation_pix is None or separation_pix <= 0:
        objects_tbl["ds9_id"] = np.arange(len(objects_tbl), dtype=int)
        return objects_tbl.copy()

    # After filtering, we may end up with an empty table; short-circuit.
    if len(objects_tbl) == 0:
        objects_tbl["ds9_id"] = np.arange(0, dtype=int)
        return objects_tbl.copy()

    if center not in ("centroid", "peak"):
        raise ValueError("center must be 'centroid' or 'peak'")

    if peak_col not in objects_tbl.colnames:
        raise ValueError(
            f"peak_col='{peak_col}' not in table columns. Available: {', '.join(objects_tbl.colnames)}"
        )

    if center == "peak":
        req = {"xpeak", "ypeak"}
        xcol, ycol = "xpeak", "ypeak"
    else:
        req = {"x", "y"}
        xcol, ycol = "x", "y"

    missing = req.difference(objects_tbl.colnames)
    if missing:
        raise ValueError(
            f"objects_tbl missing required columns for center='{center}': {', '.join(sorted(missing))}"
        )

    if separation_pix is not None:
        # Coordinates in SEP are 0-indexed; the relative distances are the same in DS9.
        x = np.asarray(objects_tbl[xcol], dtype=float)
        y = np.asarray(objects_tbl[ycol], dtype=float)
        coords = np.column_stack([x, y])

        peaks = np.asarray(objects_tbl[peak_col], dtype=float)
        # Highest peak first; stable tie-breaker by index (lower index first).
        order = np.lexsort((np.arange(len(peaks)), -peaks))

        if window_shape not in ("circle", "square"):
            raise ValueError("window_shape must be 'circle' or 'square'")

        sepv = float(separation_pix)
        # A square of side (2*separation_pix + 1) has half-width separation_pix + 0.5.
        sep_eff = (sepv + 0.5) if window_shape == "square" else sepv

        suppressed = np.zeros(len(objects_tbl), dtype=bool)
        keep: list[int] = []
        # Prefer scipy KDTree if available (fast for large catalogs), otherwise fallback.
        try:
            from scipy.spatial import cKDTree  # type: ignore

            tree = cKDTree(coords)
            for idx in order:
                if suppressed[idx]:
                    continue
                keep.append(int(idx))
                if window_shape == "square":
                    # Chebyshev (L-infinity) neighborhood => axis-aligned square.
                    neighbors = cast(
                        list[int],
                        tree.query_ball_point(coords[idx], r=float(sep_eff), p=np.inf),
                    )
                else:
                    neighbors = cast(list[int], tree.query_ball_point(coords[idx], r=float(sep_eff)))
                for j in neighbors:
                    suppressed[int(j)] = True
                suppressed[idx] = False
        except Exception:
            for idx in order:
                if suppressed[idx]:
                    continue
                keep.append(int(idx))
                dx = coords[:, 0] - coords[idx, 0]
                dy = coords[:, 1] - coords[idx, 1]
                if window_shape == "square":
                    mask = (np.abs(dx) <= sep_eff) & (np.abs(dy) <= sep_eff)
                else:
                    mask = (dx * dx + dy * dy) <= (sepv * sepv)
                suppressed[mask] = True
                suppressed[idx] = False

        # Return in original order for easier cross-referencing.
        keep_sorted = np.sort(np.asarray(keep, dtype=int))
        sel_objects_tbl=objects_tbl[keep_sorted].copy()
        sel_objects_tbl["ds9_id"] = np.arange(len(sel_objects_tbl), dtype=int)
        return sel_objects_tbl
    else:
        objects_tbl["ds9_id"] = np.arange(len(objects_tbl), dtype=int)
        return objects_tbl.copy()



def write_ds9_regions_from_sep_objects(
    objects_tbl: Table,
    output_path: str | Path,
    *,
    shape: Literal["ellipse", "circle", "square"] = "circle",
    center: Literal["centroid", "peak"] = "centroid",
    color: str = "red",
    scale: float = 3.0,
    circle_radius: Literal["geom", "mean", "max"] | float | None = "mean",
    circle_radius_col: str | None = None,
    round_ratio: tuple[float, float] = (0.7, 1.3),
    only_round: bool = True,
) -> Path:
    """Write a DS9 region file (image/pixel coordinates) from SEP detections.

    Parameters
    ----------
    objects_tbl : astropy.table.Table
        SEP detections table (requires at least x/y/a/b/theta; and xpeak/ypeak if
        ``center='peak'`` is used).
    output_path : str or pathlib.Path
        Output ``.reg`` path.
    shape : {'ellipse', 'circle', 'square'}, optional
        Region primitive.
    center : {'centroid', 'peak'}, optional
        Centering convention for regions.
    color : str, optional
        DS9 region color.
    scale : float, optional
        Scale factor applied to SEP a/b when deriving region sizes.
    circle_radius : {'geom', 'mean', 'max'} or float or None, optional
        Circle radius rule for ``shape='circle'``/``'square'``. If a float, use
        a fixed radius in pixels. If None, read a per-row radius from
        ``circle_radius_col``.
    circle_radius_col : str, optional
        Column name used when ``circle_radius is None``.
    round_ratio : tuple of float, optional
        Allowed ``a/b`` ratio range when ``only_round`` is True.
    only_round : bool, optional
        If True, keep only approximately round detections.

    Returns
    -------
    pathlib.Path
        Output region-file path.

    Notes
    -----
    DS9 image coordinates are 1-indexed; SEP x/y are 0-indexed. This routine
    applies the +1 conversion automatically.

    """
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# Region file format: DS9 version 4.1",
        f"global color={color} dashlist=8 3 width=1 font='helvetica 10 normal'",
        "image",
    ]

    req_cols = {"x", "y", "a", "b", "theta"}
    missing = req_cols.difference(objects_tbl.colnames)
    if missing:
        raise ValueError(
            "objects_tbl is missing required SEP columns: "
            f"{', '.join(sorted(missing))}. Available columns: {', '.join(objects_tbl.colnames)}"
        )

    if shape not in ("ellipse", "circle", "square"):
        raise ValueError("shape must be 'ellipse', 'circle', or 'square'")
    if center not in ("centroid", "peak"):
        raise ValueError("center must be 'centroid' or 'peak'")
    if circle_radius is None:
        if shape not in ("circle", "square"):
            raise ValueError("circle_radius=None is only supported for shape='circle' or shape='square'")
        if not circle_radius_col:
            raise ValueError("circle_radius=None requires circle_radius_col to be provided")
        if circle_radius_col not in objects_tbl.colnames:
            raise ValueError(
                f"circle_radius_col='{circle_radius_col}' not found in table columns. "
                f"Available columns: {', '.join(objects_tbl.colnames)}"
            )
    elif isinstance(circle_radius, (int, float)):
        if not np.isfinite(float(circle_radius)) or float(circle_radius) <= 0:
            raise ValueError("numeric circle_radius must be a finite positive number (pixels)")
    else:
        if circle_radius not in ("geom", "mean", "max"):
            raise ValueError("circle_radius must be one of: 'geom', 'mean', 'max', None, or a positive number")

    if center == "peak":
        peak_missing = {"xpeak", "ypeak"}.difference(objects_tbl.colnames)
        if peak_missing:
            raise ValueError(
                "center='peak' requested but objects_tbl is missing: "
                f"{', '.join(sorted(peak_missing))}. Available columns: {', '.join(objects_tbl.colnames)}"
            )

    rmin, rmax = round_ratio
    for i in range(len(objects_tbl)):
        n=objects_tbl['ds9_id'][i]
        a = float(objects_tbl["a"][i])
        b = float(objects_tbl["b"][i])
        if only_round:
            if b == 0:
                continue
            ratio = a / b
            if not (rmin <= ratio <= rmax):
                continue

        # DS9 is 1-indexed for image pixels.
        if center == "centroid":
            x = float(objects_tbl["x"][i]) + 1.0
            y = float(objects_tbl["y"][i]) + 1.0
        else:
            x = float(objects_tbl["xpeak"][i]) + 1.0
            y = float(objects_tbl["ypeak"][i]) + 1.0
        theta_deg = float(objects_tbl["theta"][i]) * 180.0 / np.pi
        if shape == "ellipse":
            r1 = scale * a
            r2 = scale * b
            lines.append(f"ellipse({x:.3f},{y:.3f},{r1:.3f},{r2:.3f},{theta_deg:.3f}) # text={{{n}}}")
        else:
            # Convert SEP's a/b to a single radius-like quantity `r`.
            # For circles: r is the radius.
            # For squares: side length will be (2*r + 1).
            if circle_radius is None:
                r = float(objects_tbl[circle_radius_col][i])
                if not np.isfinite(r) or r <= 0:
                    continue
            elif isinstance(circle_radius, (int, float)):
                r = float(circle_radius)
            elif circle_radius == "geom":
                r = scale * float(np.sqrt(a * b))
            elif circle_radius == "mean":
                r = scale * (a + b) / 2.0
            else:  # "max"
                r = scale * max(a, b)

            if shape == "circle":
                lines.append(f"circle({x:.3f},{y:.3f},{r:.3f}) # text={{{n}}}")
            else:
                # For squares, enforce an odd *integer* side length in pixels.
                # (This ensures the source is exactly centered on a pixel.)
                r_int = int(np.round(float(r)))
                if r_int < 0:
                    continue
                side = float(2 * r_int + 1)
                # DS9: box(x, y, width, height, angle)
                lines.append(f"box({x:.3f},{y:.3f},{side:.3f},{side:.3f},0) # text={{{n}}}")

    out.write_text("\n".join(lines) + "\n", encoding="ascii")
    return out

def SEP_source_extraction(
    header,
    data_sub,
    err,
    mask,
    thresh_sigma,
    region_center,
    aperture_radius_pix: float = 3.0,
    minarea: int = 5,
    *,
    err_mode: str = "global",
):
    """Run SEP source extraction + aperture photometry + WCS coordinate conversion.

    Parameters
    ----------
    header : astropy.io.fits.Header
        Header providing a 2D celestial WCS.
    data_sub : 2D-array
        Background-subtracted image.
    err : float or 2D-array
        Error estimate from ``load_data``.
    mask : 2D-array (bool)
        Bad-pixel mask.
    thresh_sigma : float
        Detection threshold in units of ``err``.
    region_center : {'centroid', 'peak'}
        Which coordinates are converted to RA/Dec.
    aperture_radius_pix : float, optional
        Aperture radius (pixels) for ``sep.sum_circle``.
    minarea : int, optional
        Minimum number of connected pixels above threshold.
    err_mode : {'jwst_err', 'bkg_rms', 'global', 'sqrt'}, optional
        Error model used inside SEP.

    Returns
    -------
    astropy.table.Table
        SEP detections including aperture photometry, SNR columns, and RA/DEC.

    """
    # When using a low threshold on large images (especially with a spatially
    # varying `err` map), SEP can exceed its default internal pixel buffer.
    # Bump the limit to (at least) the image size.
    sep.set_extract_pixstack(max(300000, int(data_sub.size)))

    # Choose the uncertainty model used by SEP.
    # - jwst_err: use ERR extension (per-pixel)
    # - bkg_rms: use SEP background RMS map
    # - global: use SEP background global RMS (scalar)
    # - sqrt: Poisson-like noise sqrt(max(data_sub,0)) with a noise floor
    if err_mode == "sqrt":
        # Background-subtracted images can be <=0; avoid NaNs and avoid 0 errors
        # (which would yield infinite SNR).
        err_sep = np.sqrt(np.clip(np.asarray(data_sub, dtype=float), 0.0, None))
        if np.isscalar(err):
            # Handles python floats as well as numpy scalar types.
            floor = float(np.asarray(err, dtype=float))
        else:
            arr = np.asarray(err, dtype=float)
            good = np.isfinite(arr) & (arr > 0)
            floor = float(np.nanmedian(arr[good])) if np.any(good) else np.nan
        if not np.isfinite(floor) or floor <= 0:
            floor = 1.0
        err_sep = np.where((err_sep > 0) & np.isfinite(err_sep), err_sep, floor)
    else:
        err_sep = err

    objects = sep.extract(
        data_sub,
        thresh_sigma,
        err=err_sep,
        mask=mask,
        minarea=int(minarea),
    )
    flux, fluxerr, flag = sep.sum_circle(
        data_sub,
        objects['x'],
        objects['y'],
        float(aperture_radius_pix),
        err=err_sep,
        mask=mask,
    )
    objects_tbl = Table(objects, copy=True)

    # Keep the aperture-sum flag separate to avoid confusion with the
    # SExtractor/SEP detection `flag` column.
    objects_tbl["ap_flag"] = flag

    # Human-readable SEP aperture-flag summary.
    # Prefer an explicit 'invalid' label when SEP returns NaNs.
    ap_flux = np.asarray(flux, dtype=float)
    ap_err = np.asarray(fluxerr, dtype=float)
    ap_flag_arr = np.asarray(flag, dtype=int)
    ap_desc: list[str] = []
    for f, s, se in zip(ap_flag_arr, ap_flux, ap_err):
        if not (np.isfinite(s) and np.isfinite(se)):
            ap_desc.append("invalid")
        else:
            ap_desc.append(aperture_flag_short(int(f)))
    objects_tbl["ap_flag_desc"] = _as_fixed_str_array(ap_desc, width=64)

    # Human-readable SExtractor-style flag summary.
    if "flag" in objects_tbl.colnames:
        objects_tbl["flag_desc"] = _as_fixed_str_array(
            [sextractor_flag_short(int(f)) for f in objects_tbl["flag"]],
            width=64,
        )

    objects_tbl['apflux'] = flux
    objects_tbl['apflux_err'] = fluxerr
    objects_tbl["ap_snr"] = objects_tbl["apflux"] / objects_tbl["apflux_err"]

    # Peak SNR: peak / (error at the peak pixel).
    # (Uses xpeak/ypeak; for scalar err this is just peak/err.)
    if "peak" in objects_tbl.colnames and "xpeak" in objects_tbl.colnames and "ypeak" in objects_tbl.colnames:
        peak_val = np.asarray(objects_tbl["peak"], dtype=float)
        if np.isscalar(err_sep):
            peak_err = np.full(len(objects_tbl), float(np.asarray(err_sep, dtype=float)))
        else:
            err_img = np.asarray(err_sep, dtype=float)
            h, w = err_img.shape
            xi = np.clip(np.rint(np.asarray(objects_tbl["xpeak"], dtype=float)).astype(int), 0, w - 1)
            yi = np.clip(np.rint(np.asarray(objects_tbl["ypeak"], dtype=float)).astype(int), 0, h - 1)
            peak_err = err_img[yi, xi]
        objects_tbl["peak_err"] = peak_err
        denom = np.asarray(objects_tbl["peak_err"], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            objects_tbl["peak_snr"] = np.where(np.isfinite(denom) & (denom > 0), peak_val / denom, np.nan)
    objects_tbl["ellipt"] = 1-objects_tbl['b'] / objects_tbl['a']
    # First, do a simple peak threshold.

    # Compute sky coordinates from the WCS.
    # SEP x/y and xpeak/ypeak are 0-indexed pixel coordinates, so use origin=0.
    wcs = WCS(header, naxis=2)
    if region_center == 'peak':
        ra, dec = wcs.all_pix2world(objects_tbl['xpeak'], objects_tbl['ypeak'], 0)
    else:
        ra, dec = wcs.all_pix2world(objects_tbl['x'], objects_tbl['y'], 0)

    objects_tbl['RA'] = ra
    objects_tbl['DEC'] = dec

    return objects_tbl

def as_list(x: Any) -> list[Any]:
    """Return ``x`` as a python list.

    Parameters
    ----------
    x : object
        Scalar or list-like.

    Returns
    -------
    list
        ``x`` converted to a list (scalars become a single-element list).

    """
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    return [x]

def broadcast(value: Any, n: int, newshape=True) -> list[Any]:
    """Broadcast a scalar (or selection list) to per-run values.

    Parameters
    ----------
    value : object
        Scalar or list-like input.
    n : int
        Number of runs.
    newshape : bool, optional
        Backwards-compatibility toggle. If False, list-like values are deep-copied
        to length ``n``. If True, list-like values are returned unchanged.

    Returns
    -------
    list
        Per-run values.

    """
    import copy

    if n < 0:
        raise ValueError(f"n must be >= 0 (got {n})")

    if value is None:
        return [None] * n

    if isinstance(value, (list, tuple, np.ndarray)):
        if not newshape:
            base = list(value)
            return [copy.deepcopy(base) for _ in range(n)]
        else:
            return value

    return [value] * n

def DAO_source_extraction(
    data,
    err,
    psf,
    oversampling=1,
    fit_radius=20.0,
    aperture_radius=5.0,
    dao_thresh_sigma=4.0,
    dao_fwhm=2.5,
    group_radius=15.0,
    catalog=None,
    nan_lim_percent=0.51,
):
    """Run DAOStarFinder source extraction with PSF-fitting refinement.

    This routine detects point sources with ``DAOStarFinder``, groups nearby
    detections that belong to the same physical star (including multiple
    wing-detections around bright or saturated sources), selects one
    representative per group, and refines each representative position and flux
    with ``fit_psf`` on a local image cutout.

    For each group the representative is chosen as follows:

    * If a DAO group contains one or more catalog-seeded candidates, keep all
      catalog members from that group and discard all DAO members.
    * Otherwise, if any candidate has NaN pixels nearby (saturated core), the
      NaN-core centroid is used and its radius estimated with
      ``estimate_nan_core``.
    * Otherwise the source is re-centered on the strongest group-wide
      PSF-correlation peak, with the highest ``DAOStarFinder`` peak used as a
      fallback.

    Parameters
    ----------
    data : 2D-array
        Background-subtracted science image.
    err : float or 2D-array
        Error model used for aperture-flux uncertainty estimation.
    psf : 2D-array
        PSF model image passed directly to ``fit_psf``.
    oversampling : int, optional
        Oversampling factor of ``psf`` relative to detector pixels.
    fit_radius : float, optional
        Radius (pixels) for local PSF-fitting cutouts.
    aperture_radius : float, optional
        Aperture radius (pixels) for compatibility flux columns.
    dao_thresh_sigma : float, optional
        ``DAOStarFinder`` detection threshold in units of the image RMS.
    dao_fwhm : float, optional
        PSF FWHM (pixels) passed to ``DAOStarFinder``.
    group_radius : float, optional
        Grouping radius (pixels). All ``DAOStarFinder`` detections within this
        distance of each other are treated as belonging to the same star, and
        only one representative is kept.  The same radius is also used as the
        minimum allowed separation between any two sources in the final
        catalog.  Should be set to roughly 1–2 times the PSF wing extent; a
        value of ~15 pixels works well for JWST NIRCam wide-field data.
    catalog : astropy.table.Table, str, or None, optional
        External source catalog used to override DAO detections when overlapping.
        Accepts an ``astropy.table.Table`` with ``x`` and ``y`` pixel-coordinate
        columns, or a path to a CSV file with the same columns. For any group
        that contains both DAO and catalog candidates, all catalog candidates
        in that group are kept and all DAO candidates are discarded. Catalog-
        only groups are ignored.
    nan_lim_percent : float, optional
        When checking for NaN pixels near a candidate (to identify sources to close to the edge, or outside),
        the candidate is excluded if more than this fraction of the total pixels in the tile are NaN.

    Returns
    -------
    astropy.table.Table
        Catalog with fitted detector coordinates, aperture quantities,
        detection metadata (``det_method``, ``sat_radius``, ``psf_flux``).

    Notes
    -----
    Candidates whose PSF fit fails are assigned NaN coordinates and are
    removed from the returned table before output.
    """

    def _dao(data, fwhm=2.5, sigma_thresh=4.0):
        """Recover additional faint point sources with ``DAOStarFinder``.

        Parameters
        ----------
        data : 2D-array
            Science image.
        fwhm : float, optional
            FWHM (pixels) passed to ``DAOStarFinder``.
        sigma_thresh : float, optional
            Detection threshold in units of estimated image RMS.

        Returns
        -------
        list of dict
            Candidate dictionaries with ``x``, ``y``, ``peak``,
            ``sat_radius``, and ``method``.

        """
        data = np.asarray(data, dtype=float)
        finite = np.isfinite(data)
        vals = data[finite]
        if vals.size == 0:
            return []
        med = float(np.nanmedian(vals))
        rms = float(np.nanstd(vals))
        if not np.isfinite(rms) or rms <= 0:
            rms = 1.0

        dao = DAOStarFinder(fwhm=float(fwhm), threshold=float(sigma_thresh * rms))
        tbl = dao(np.nan_to_num(data - med, nan=0.0))
        if tbl is None or len(tbl) == 0:
            return []
        out = []
        for row in tbl:
            out.append({
                "x": float(row["xcentroid"]),
                "y": float(row["ycentroid"]),
                "peak": float(row["peak"]),
                "sat_radius": 0.0,
                "method": "dao",
            })
        return out

    def _group_and_select(cands, data_arr, _group_radius, _fwhm):
        """Group nearby candidates and select one representative per star.

        DAOStarFinder often returns several detections for a single bright or
        saturated star — one near the core and several on the PSF wings.  This
        helper collapses such groups into a single candidate by:

        1. Building connected components: two candidates belong to the same
           group when their separation is less than ``_group_radius``.
        2. Within each group, selecting the best representative:

           * If any candidate has NaN pixels within ``_fwhm / 2`` pixels
             (saturated core), pick the one with the most NaN neighbours.
             ``estimate_nan_core`` is then called on that cutout to populate
             ``sat_radius``.
           * Otherwise re-center the group on the strongest local
             PSF-correlation peak.

        Parameters
        ----------
        cands : list of dict
            Full (ungrouped) candidate list from ``_dao``.
        data_arr : 2D-array
            Science image (used for NaN proximity checks).
        _group_radius : float
            Maximum separation (pixels) for two candidates to be in the
            same group.
        _fwhm : float
            PSF FWHM (pixels); sets the NaN-proximity search radius.

        Returns
        -------
        list of dict
            One representative candidate dict per group, with ``sat_radius``
            set if a saturated core was detected.

        """
        if len(cands) == 0:
            return []

        def _candidate_radius(c, base_radius, prov_sat_r=0.0):
            """Return an adaptive grouping radius for one candidate.

            Bright sources are given a larger grouping window using their DAO
            peak, and saturated sources get an additional boost from the
            estimated saturated-core size.  Because ``sat_radius`` in the raw
            candidate dict is always 0 at this stage (it is filled in only
            after grouping), callers should pass a provisional NaN-core size
            via ``prov_sat_r`` so the saturation branch can fire correctly.
            """
            r = float(base_radius)
            peak = float(c.get("peak", 0.0))
            if np.isfinite(peak) and peak > 0:
                r = max(r, float(base_radius) + 3.0 * np.log10(max(peak, 1.0)))
            # Use the larger of the stored value (always 0 here) and the
            # provisional estimate derived from NaN-pixel proximity.
            sat_r = max(float(c.get("sat_radius", 0.0)), float(prov_sat_r))
            if np.isfinite(sat_r) and sat_r > 0:
                r = max(r, float(base_radius) + 5 * sat_r)
            return float(np.clip(r, float(base_radius), 60.0))

        def _effective_radius(xvals, yvals, xref, yref, base_radius):
            """Scale the grouping window from the measured wing spread.

            Uses the 95th percentile radial extent of the group relative to a
            provisional core estimate, then adds a small margin and clips to a
            sane range.
            """
            d = np.hypot(np.asarray(xvals, dtype=float) - float(xref), np.asarray(yvals, dtype=float) - float(yref))
            if d.size == 0:
                return float(base_radius)
            wing_spread = float(np.percentile(d, 95))
            return float(np.clip(max(float(base_radius), wing_spread + 3.0), float(base_radius), 40.0))

        n = len(cands)
        xs = np.array([c["x"] for c in cands], dtype=float)
        ys = np.array([c["y"] for c in cands], dtype=float)
        ny_arr, nx_arr = data_arr.shape

        # Pre-compute a provisional NaN-core radius for every candidate so
        # that _candidate_radius can scale the grouping window correctly even
        # before the formal sat_radius is estimated inside _group_and_select.
        _quick_r = max(3, int(_group_radius // 3))
        _prov_sat = []
        for _c in cands:
            _cx, _cy = float(_c["x"]), float(_c["y"])
            _xlo = max(0, int(_cx) - _quick_r)
            _xhi = min(nx_arr, int(_cx) + _quick_r + 1)
            _ylo = max(0, int(_cy) - _quick_r)
            _yhi = min(ny_arr, int(_cy) + _quick_r + 1)
            _patch = data_arr[_ylo:_yhi, _xlo:_xhi]
            if np.any(~np.isfinite(_patch)):
                _sr, _, _ = estimate_nan_core(_patch, margin=1)
            else:
                _sr = 0.0
            _prov_sat.append(float(_sr))

        cand_radii = np.array(
            [_candidate_radius(c, _group_radius, ps) for c, ps in zip(cands, _prov_sat)],
            dtype=float,
        )

        # --- union-find for connected-component grouping ---
        parent = list(range(n))

        def _find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]  # path compression
                a = parent[a]
            return a

        def _union(a, b):
            parent[_find(a)] = _find(b)

        for i in range(n):
            for j in range(i + 1, n):
                pair_r = max(float(cand_radii[i]), float(cand_radii[j]))
                if (xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2 < pair_r ** 2:
                    _union(i, j)

        # Collect groups by root index.
        from collections import defaultdict
        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            groups[_find(i)].append(i)

        # --- select one representative per group ---
        search_r = float(_group_radius)
        selected = []

        for indices in groups.values():
            group_cands = [cands[i] for i in indices]
            group_radii = cand_radii[np.asarray(indices, dtype=int)]
            group_search_r = float(np.max(group_radii))

            # If a group contains both DAO and catalog members, keep all catalog
            # members and discard all DAO members. Catalog-only groups are
            # ignored (no DAO group to replace).
            catalog_member_indices = [k for k, c in enumerate(group_cands)
                                      if c.get("method") == "catalog"]
            dao_member_indices = [k for k, c in enumerate(group_cands)
                                  if c.get("method") != "catalog"]

            if catalog_member_indices:
                for k in catalog_member_indices:
                    best = dict(group_cands[k])

                    # Still estimate sat_radius from NaN core near each
                    # catalog position for robust masked-core PSF fitting.
                    cx, cy = float(best["x"]), float(best["y"])
                    half = int(max(8, _group_radius))
                    xlo = max(0, int(round(cx)) - half)
                    xhi = min(nx_arr, int(round(cx)) + half + 1)
                    ylo = max(0, int(round(cy)) - half)
                    yhi = min(ny_arr, int(round(cy)) + half + 1)
                    local = data_arr[ylo:yhi, xlo:xhi]
                    if np.sum(~np.isfinite(local)) > np.ceil(local.shape[0]*local.shape[1]*nan_lim_percent):
                        continue  # Avoid spurious large sat_radius estimates from mostly-NaN cutouts.
                    selected.append(best)
                continue

            # Count NaN pixels within search_r of each candidate.
            nan_counts = []
            for c in group_cands:
                cx, cy = float(c["x"]), float(c["y"])
                local_r = max(search_r, group_search_r)
                xlo = max(0, int(cx - local_r))
                xhi = min(nx_arr, int(cx + local_r) + 1)
                ylo = max(0, int(cy - local_r))
                yhi = min(ny_arr, int(cy + local_r) + 1)
                patch = data_arr[ylo:yhi, xlo:xhi]
                nan_counts.append(int(np.sum(~np.isfinite(patch))))

            max_nan = max(nan_counts)
            if max_nan > 0:
                # Saturated group: estimate the NaN-core centroid on a group-wide
                # cutout and use that as the representative source position.
                xg = np.array([float(c["x"]) for c in group_cands], dtype=float)
                yg = np.array([float(c["y"]) for c in group_cands], dtype=float)
                half = int(max(8, _group_radius))
                xlo = max(0, int(np.floor(np.min(xg))) - half)
                xhi = min(nx_arr, int(np.ceil(np.max(xg))) + half + 1)
                ylo = max(0, int(np.floor(np.min(yg))) - half)
                yhi = min(ny_arr, int(np.ceil(np.max(yg))) + half + 1)
                local = data_arr[ylo:yhi, xlo:xhi]
                if np.sum(~np.isfinite(local)) > np.ceil(local.shape[0] * local.shape[1] * nan_lim_percent):
                    continue  # Avoid spurious large sat_radius estimates from mostly-NaN cutouts.
                elif np.any(~np.isfinite(local)):
                    sr , xcore, ycore = estimate_nan_core(local, margin=1)
                    peaks = np.array([float(c.get("peak", 0.0)) for c in group_cands], dtype=float)
                    d2 = (xg - (xcore + xlo)) ** 2 + (yg - (ycore + ylo)) ** 2
                    # Use the nearest DAO detection only to inherit metadata,
                    # but move the representative coordinates onto the NaN core.
                    best_idx = int(np.lexsort((-peaks, d2))[0])
                else:
                    best_idx = int(np.argmax(nan_counts))
            else:
                # Unsaturated group: use a group-wide PSF matched-filter peak to
                # avoid keeping a bright wing knot as the representative.
                xg = np.array([float(c["x"]) for c in group_cands], dtype=float)
                yg = np.array([float(c["y"]) for c in group_cands], dtype=float)
                half = int(max(8, _group_radius))
                xlo = max(0, int(np.floor(np.min(xg))) - half)
                xhi = min(nx_arr, int(np.ceil(np.max(xg))) + half + 1)
                ylo = max(0, int(np.floor(np.min(yg))) - half)
                yhi = min(ny_arr, int(np.ceil(np.max(yg))) + half + 1)
                local = data_arr[ylo:yhi, xlo:xhi]

                peaks = np.array([float(c.get("peak", 0.0)) for c in group_cands], dtype=float)
                best_idx = int(np.argmax(peaks))

                try:
                    psf_det = downsample_psf_to_detector(psf, oversampling)
                    psf_det = np.asarray(psf_det, dtype=float)
                    psf_sum = np.nansum(psf_det)
                    if np.isfinite(psf_sum) and psf_sum > 0:
                        psf_det = psf_det / psf_sum
                        img = np.nan_to_num(local - np.nanmedian(local), nan=0.0)
                        corr = fftconvolve(img, psf_det[::-1, ::-1], mode="same")
                        iy, ix = np.unravel_index(np.nanargmax(corr), corr.shape)
                        d2 = (xg - (float(ix) + xlo)) ** 2 + (yg - (float(iy) + ylo)) ** 2
                        # Keep metadata from the nearest DAO detection, but move
                        # the representative coordinates onto the PSF-correlation peak.
                        best_idx = int(np.lexsort((-peaks, d2))[0])
                except Exception:
                    pass

            best = dict(group_cands[best_idx])  # shallow copy

            # Use the measured wing spread to enlarge the local window used for
            # the final centroid/core-radius refinement.
            xg = np.array([float(c["x"]) for c in group_cands], dtype=float)
            yg = np.array([float(c["y"]) for c in group_cands], dtype=float)
            refx, refy = float(best["x"]), float(best["y"])
            eff_group_radius = _effective_radius(xg, yg, refx, refy, _group_radius)

            # Populate sat_radius for saturated representatives.
            if max_nan > 0 and best["sat_radius"] <= 0:
                cx, cy = float(best["x"]), float(best["y"])
                half = int(max(8, eff_group_radius))
                xlo = max(0, int(round(cx)) - half)
                xhi = min(nx_arr, int(round(cx)) + half + 1)
                ylo = max(0, int(round(cy)) - half)
                yhi = min(ny_arr, int(round(cy)) + half + 1)
                local = data_arr[ylo:yhi, xlo:xhi]
                if np.any(~np.isfinite(local)):
                    sr, xcore, ycore = estimate_nan_core(
                        local, center=(cx - xlo, cy - ylo), margin=1
                    )
                    best["x"] = float(xcore + xlo)
                    best["y"] = float(ycore + ylo)
                    best["sat_radius"] = float(sr)
            elif max_nan == 0:
                # For unsaturated groups, place the representative on the local
                # PSF-correlation peak if it was measured above.
                half = int(max(8, eff_group_radius))
                xlo = max(0, int(np.floor(np.min(xg))) - half)
                xhi = min(nx_arr, int(np.ceil(np.max(xg))) + half + 1)
                ylo = max(0, int(np.floor(np.min(yg))) - half)
                yhi = min(ny_arr, int(np.ceil(np.max(yg))) + half + 1)
                local = data_arr[ylo:yhi, xlo:xhi]
                try:
                    psf_det = downsample_psf_to_detector(psf, oversampling)
                    psf_det = np.asarray(psf_det, dtype=float)
                    psf_sum = np.nansum(psf_det)
                    if np.isfinite(psf_sum) and psf_sum > 0:
                        psf_det = psf_det / psf_sum
                        img = np.nan_to_num(local - np.nanmedian(local), nan=0.0)
                        corr = fftconvolve(img, psf_det[::-1, ::-1], mode="same")
                        iy, ix = np.unravel_index(np.nanargmax(corr), corr.shape)
                        best["x"] = float(ix + xlo)
                        best["y"] = float(iy + ylo)
                except Exception:
                    pass

            selected.append(best)

        log.info(f"Using {np.sum([i['method']=='catalog' for i in selected])} catalog seeds + {np.sum([i['method']!='catalog' for i in selected])} DAO detections after selections.")
        return selected

    data = np.asarray(data, dtype=float)
    ny, nx = data.shape

    # ---- candidate detection via DAOStarFinder ----
    cat = _dao(
        data,
        fwhm=dao_fwhm,
        sigma_thresh=dao_thresh_sigma,
    )

    # ---- optional: seed candidates from an external catalog ----
    cat_from_catalog: list[dict] = []
    if catalog is not None:
        # Accept an astropy Table or a path to a CSV file.
        if isinstance(catalog, (str, Path)):
            from astropy.table import Table as _Table
            _ctbl = _Table.read(str(catalog))
        else:
            _ctbl = catalog
        for _row in _ctbl:
            try:
                _cx = float(_row["x"])
                _cy = float(_row["y"])
            except (KeyError, TypeError):
                log.warning("catalog row missing 'x'/'y' columns; skipping row.")
                continue
            cat_from_catalog.append({
                "x": _cx,
                "y": _cy,
                "peak": float(_row["peak"]) if "peak" in _ctbl.colnames else np.nan,
                "sat_radius": 0.0,
                "method": "catalog",
            })
        log.info(f"Starting from {len(cat_from_catalog)} catalog seeds + {len(cat)} DAO detections.")

    # Catalog candidates are prepended so they have priority inside each group.
    all_cands = cat_from_catalog + cat

    # Group detections from the same star (bright stars produce multiple wing
    # detections) and select one representative per group.  The representative
    # is the catalog seed (if provided), the saturated NaN core, or the
    # PSF-correlation peak for unsaturated sources.
    cands = _group_and_select(all_cands, data, group_radius, dao_fwhm)

    # Safety de-duplication: ensure no two sources are closer than group_radius.
    # This catches edge cases where two group representatives end up very close
    # (e.g. no DAO detections between two distinct stars).
    # cands = _dedup_candidates(cat, min_separation=group_radius)

    # ---- refinement with existing fit_psf ----
    rows = []
    for c in cands:
        method = c["method"]
        sat_r = float(c["sat_radius"])
        x0, y0 = c["x"], c["y"]

        # local cutout around candidate
        half = int(max(psf.shape[0] // 2, fit_radius, group_radius))
        xlo = max(0, int(round(x0)) - half)
        xhi = min(nx, int(round(x0)) + half + 1)
        ylo = max(0, int(round(y0)) - half)
        yhi = min(ny, int(round(y0)) + half + 1)
        cut = data[ylo:yhi, xlo:xhi]
        cut -= np.nanmedian(cut)
        # estimate radius from nan core in cutout if needed
        if sat_r <= 0 and np.any(~np.isfinite(cut)):
            sat_r_est, _, _ = estimate_nan_core(cut, center=(x0 - xlo, y0 - ylo), margin=1)
            sat_r = float(sat_r_est)

        if method != 'catalog':
            try:
                fx, fy, fflux = fit_psf(
                    masked_psf_data=psf,
                    data=cut,
                    oversampling=oversampling,
                    radius_core=sat_r,
                    fit_radius=min(fit_radius, max(cut.shape) / 2 - 1),
                    search_radius=min(fit_radius, max(cut.shape) / 2 - 1),
                    bkg_subtract=True,
                    two_pass=True,
                    showplots=False,
                )
                x_fit = float(fx + xlo)
                y_fit = float(fy + ylo)
                # flux = float(fflux)
                flag = 0
            except Exception:
                # x_fit, y_fit, flux = np.nan, np.nan, np.nan
                x_fit, y_fit = np.nan, np.nan
                flag = 1
        else:
            x_fit, y_fit = x0, y0
            flag = 0

        # simple aperture flux around fitted center (optional compatibility column)
        if np.isfinite(x_fit) and np.isfinite(y_fit):
            yy, xx = np.indices(data.shape)
            rr2 = (xx - x_fit) ** 2 + (yy - y_fit) ** 2
            ap = rr2 <= float(aperture_radius) ** 2
            good = ap & np.isfinite(data)
            apflux = float(np.nansum(data[good])) if np.any(good) else np.nan
            # crude error model from err input
            if np.isscalar(err):
                apflux_err = float(np.sqrt(np.sum(good)) * float(err))
            else:
                e = np.asarray(err, dtype=float)
                apflux_err = float(np.sqrt(np.nansum((e[good]) ** 2))) if np.any(good) else np.nan
        else:
            apflux, apflux_err = np.nan, np.nan


        rows.append((
            x_fit, y_fit,                # x, y
            x_fit, y_fit,                # xpeak, ypeak (compat)
            np.nan, np.nan, 0.0,         # a, b, theta (not from segmentation)
            flag,                        # flag
            0,                           # ap_flag
            apflux, apflux_err,
            apflux / apflux_err if np.isfinite(apflux) and np.isfinite(apflux_err) and apflux_err > 0 else np.nan,
            np.nan,                      # peak_snr (optional fill later)
            np.nan,                      # ellipt
            sat_r,
            c["method"],
            # flux,
        ))

    names = [
        "x", "y", "xpeak", "ypeak", "a", "b", "theta", "flag", "ap_flag",
        "apflux", "apflux_err", "ap_snr", "peak_snr", "ellipt",
        "sat_radius", "det_method"]#, "psf_flux"]
    tbl = Table(rows=rows, names=names)

    # Keep only rows with finite coordinates so bad fits are excluded from CSV/DS9.
    # Prefer peak coordinates (used with region_center="peak"), fallback to x/y.
    if "xpeak" in tbl.colnames and "ypeak" in tbl.colnames:
        xcol, ycol = "xpeak", "ypeak"
    else:
        xcol, ycol = "x", "y"

    xvals = np.asarray(tbl[xcol], dtype=float)
    yvals = np.asarray(tbl[ycol], dtype=float)
    good = np.isfinite(xvals) & np.isfinite(yvals)

    n_total = len(tbl)
    n_bad = int(np.sum(~good))
    if n_bad > 0:
        log.warning(f"Skipping {n_bad}/{n_total} sources with NaN/invalid coordinates.")
    tbl = tbl[good].copy()

    # If nothing valid remains, skip catalog + region creation for this file.
    if len(tbl) == 0:
        log.warning(f"No valid sources for CSV/DS9 output.")

    return tbl