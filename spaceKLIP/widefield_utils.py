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

# Set up log.
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

def stars_extractor(data,xs,ys, size=61, showplots=False):
    #  Create a Table of star positions for extraction
    star_tbl = Table([xs, ys], names=['x', 'y'])
    # Extract stars from the masked data (cutout size is set to 25x25)
    nddata = NDData(data)  # Input masked data for cutout extraction
    stars = extract_stars(nddata, star_tbl, size=size)
    if showplots:
        for el in range(len(stars)):
            norm = simple_norm(stars[el].data, 'log')
            plt.imshow(stars[el].data, origin='lower', norm=norm)
            plt.colorbar()
            plt.title(f'Extracted Star {el}')
            plt.show()

    return stars

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