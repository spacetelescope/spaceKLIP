import numpy as np
import logging
import sep
from pathlib import Path
from typing import Any, Literal, SupportsFloat, SupportsIndex, cast
from astropy.wcs import WCS
import matplotlib.pylab as plt
from astropy.table import Table
from astropy.visualization import simple_norm
from astropy.nddata import NDData
from photutils.psf import extract_stars
from photutils.psf import FittableImageModel
from astropy.stats import sigma_clipped_stats
from astropy.modeling import fitting
import spaceKLIP.utils as ut

# Set up log.
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

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

def fit_psf(
    masked_psf_data,
    data,
    oversampling=1,
    radius_core=0,
    fit_radius=None,
    initial_center="auto",
    search_radius=None,
    snr_threshold=10.0,
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
    initial_center : {'auto', 'max', 'center', 'matched_filter'} or (x, y), optional
        Initial guess for the center.
    search_radius : float, optional
        Search radius (pixels) for the matched-filter initialization.
    snr_threshold : float, optional
        If peak SNR is below this value, default initialization switches to a
        matched filter.
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
    x0_init = x_center
    y0_init = y_center

    # Allow explicitly providing the initial center as (x, y).
    if isinstance(initial_center, (tuple, list, np.ndarray)) and len(initial_center) == 2:
        x0_init = float(initial_center[0])
        y0_init = float(initial_center[1])
        mode = "given"
    else:
        mode = initial_center

    finite = np.isfinite(data_fit)
    if np.any(finite):
        peak_snr = float(np.nanmax(data_fit[finite]) / (rms + 1e-12))
    else:
        peak_snr = 0.0

    # Saturated stars (NaNs in the core): fitting only the wings is much more stable if
    # we (1) initialize with a matched-filter and (2) restrict the fitting region.
    if radius_core and radius_core > 0:
        if fit_radius is None:
            fit_radius = float((min(nx, ny) - 1) / 2)
        if search_radius is None:
            search_radius = min(20.0, float(fit_radius))

    if mode == "auto":
        # At high S/N, the brightest pixel is usually reliable.
        # At low S/N, use a matched-filter (cross-correlation) initial guess.
        if radius_core and radius_core > 0:
            # Wing-only (ring-like) correlations can be ambiguous; start at the NaN core.
            mode = "center"
        else:
            mode = "matched_filter" if peak_snr < float(snr_threshold) else "max"

    if mode == "max" and np.any(finite):
        iy, ix = np.unravel_index(np.nanargmax(data_fit), data_fit.shape)
        x0_init, y0_init = float(ix), float(iy)
    elif mode == "center":
        # For saturated stars: use NaN-core centroid if available.
        x0_init, y0_init = float(core_mask_x), float(core_mask_y)
    elif mode == "matched_filter" and np.any(finite):
        try:
            from scipy.signal import fftconvolve

            psf_det = _downsample_psf_to_detector(masked_psf_data, oversampling)
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
        except Exception:
            # Safe fallback if scipy is missing or correlation fails.
            pass

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
        plt.plot(fitted_x_pos, (ny - 1) - fitted_y_pos, 'ob')
        plt.colorbar()
        plt.title('Data to fit with fitted center')
        plt.show()

    return fitted_x_pos,fitted_y_pos,fitted_flux

def estimate_nan_core(data,
                      center=None,
                      margin=1,
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

    Returns
    -------
    radius : int
        Radius in pixels of the connected non-finite region.
    x_center, y_center : float
        Region centroid in cutout coordinates.

    """
    data = np.asarray(data)
    ny, nx = data.shape
    if center is None:
        cx, cy = (nx - 1) / 2, (ny - 1) / 2
    else:
        cx, cy = center

    bad = ~np.isfinite(data)

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

    yy, xx = np.indices(data.shape)
    x_cent = float(np.mean(xx[region]))
    y_cent = float(np.mean(yy[region]))
    rr = np.sqrt((xx - x_cent) ** 2 + (yy - y_cent) ** 2)
    radius = int(np.ceil(np.nanmax(rr[region])) + int(margin))
    return radius, x_cent, y_cent

def stars_extractor(data,
                    coords,
                    fow = 101,
                    shifts = None,
                    method='fourier',
                    shiftpad=5,
                    showplots=False,
                    cmap='Greys_r',
                    stretch='linear',
                    kwargs={}
):
    # #  Create a Table of star positions for extraction
    # star_tbl = Table([xs, ys], names=['x', 'y'])
    # # Extract stars from the masked data (cutout size is set to 25x25)
    # nddata = NDData(data)  # Input masked data for cutout extraction
    # stars = extract_stars(nddata, star_tbl, size=size)
    if shifts is None:
        tile = data[int(round(coords[1]))-fow//2:int(round(coords[1]))+fow//2+1, int(round(coords[0]))-fow//2:int(round(coords[0]))+fow//2+1]
    else:
        shifteddata = ut.imshift(data, [shifts[0], shifts[1]],
                               pad_amount=int(np.ceil(np.sum(np.abs(shifts)))), method=method, kwargs=kwargs)
        tile = shifteddata[int(round(coords[1]))-fow//2:int(round(coords[1]))+fow//2+1, int(round(coords[0]))-fow//2:int(round(coords[0]))+fow//2+1]

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

def sources_extraction(
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