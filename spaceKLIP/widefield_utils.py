import logging
from pathlib import Path
from typing import Literal
import matplotlib.pylab as plt
from astropy.visualization import simple_norm
from photutils.psf import FittableImageModel
from astropy.modeling import fitting
import spaceKLIP.utils as ut
import numpy as np
from astropy.table import Table, vstack
from astropy.wcs import WCS
from scipy.signal import fftconvolve
from photutils.detection import DAOStarFinder,StarFinder
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from scipy.ndimage import binary_dilation
from astropy import units as u
from astropy.coordinates import SkyCoord

# Set up log.
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def mask_within_radius(image, xdat, ydat, xcen, ycen, r, x=0, y=0, c=np.nan):
    distance = np.sqrt((xdat - (x + xcen)) ** 2 + (ydat - (y + ycen)) ** 2)
    image[np.where(distance <= r)] = c
    return image


def fetch_catalog_for_image_fov(path2table,
                                image: np.ndarray,
                                header,
                                use_gaia=False,
                                use_simbad=False,
                                border=3,
                                npix=0,
                            ):
                            """Estimate image FOV from WCS and query Gaia over that footprint.

                            Parameters
                            ----------
                            path2table : str, optional
                                Path to save the table query result CSV file.
                            image : 2D-array
                                Image data used only for its shape.
                            header : astropy.io.fits.Header
                                FITS header containing the celestial WCS for the image.
                            use_gaia : bool, optional
                               Enable Gaia query. Default is False.
                            use_simbad : bool, optional
                               Enable Simbad query. Default is False.
                            row_limit : int, optional
                                Max number of returned rows. Use ``-1`` for no row limit.
                            verbose : bool, optional
                                Passed to ``Gaia.launch_job_async``.
                            border: int, optional
                                exclude border of x pixel from image to confirm coordinates are within the fov
                            npix : int or list of four int, optional
                                Number of pixels used to pad around the frames. If int, the same
                                number of pixels will be padded on each side. If list of four int,
                                a different number of pixels can be padded on the [left, right,
                                bottom, top] of the frames. The default is 1.Need to evaluate the true border of the real data

                            Returns
                            -------
                            astropy.table.Table
                                New Astropy table with selected columns plus WCS-derived ``x`` and ``y``.

                            """

                            def query_gaia(path2table,
                                            center_ra_deg,
                                            center_dec_deg,
                                            radius_deg,
                                            gaia_table: str = "gaiadr3.gaia_source",
                                           ):
                                        """
                                        Helper to fetch Gaia DR3 source data.

                                        Parameters
                                        ----------
                                        path2table : str
                                            Path to save the table query result CSV file.
                                        center_ra_deg : float
                                            Right ascension of the center of the search region in degrees.
                                        center_dec_deg : float
                                            Declination of the center of the search region in degrees.
                                        radius_deg : float
                                            Radius of the search region in degrees.
                                        gaia_table : str, optional
                                            Gaia TAP table to query. Defaults to Gaia DR3 source table.

                                        Returns
                                        -------

                                        """
                                        from astroquery.gaia import Gaia

                                        query = (
                                            "SELECT source_id, ra, dec, parallax, parallax_error, phot_g_mean_mag FROM "
                                            f"{gaia_table} "
                                            "WHERE 1=CONTAINS(" 
                                            "POINT('ICRS', ra, dec), "
                                            f"CIRCLE('ICRS', {center_ra_deg:.12f}, {center_dec_deg:.12f}, {radius_deg:.12f})"
                                            ")"
                                        )

                                        Gaia.MAIN_GAIA_TABLE = gaia_table
                                        Gaia.ROW_LIMIT = int(-1)
                                        Gaia.launch_job_async(query=query, dump_to_file=True, verbose=False, output_format='csv',output_file=path2table)


                            def query_simbad(path2table,
                                            center_ra_deg,
                                            center_dec_deg,
                                            radius_deg,
                                            ):
                                            """
                                            Helper to fetch Simbad  source data.

                                            Parameters
                                            ----------
                                            path2table : str
                                                Path to save the table query result CSV file.
                                            center_ra_deg : float
                                                Right ascension of the center of the search region in degrees.
                                            center_dec_deg : float
                                                Declination of the center of the search region in degrees.
                                            radius_deg : float
                                                Radius of the search region in degrees.

                                            Returns
                                            -------

                                            """
                                            from astroquery.simbad import Simbad
                                            # Define center coordinates and radius
                                            coord = SkyCoord(ra=center_ra_deg, dec=center_dec_deg, unit=(u.deg, u.deg), frame='icrs')

                                            # 1. Reset fields to default, then add all 5 filters (case-sensitive)
                                            Simbad.reset_votable_fields()
                                            Simbad.add_votable_fields('flux(K)', 'flux(H)', 'flux(J)', 'flux(V)', 'flux(B)')

                                            # Execute the cone search
                                            simbad_table = Simbad.query_region(coord, radius=radius_deg * u.deg)
                                            simbad_table = simbad_table[simbad_table['FLUX_K']>0]

                                            # Only use this if your columns are returned as strings (hms/dms)
                                            if np.any([isinstance(simbad_table['RA'][0], str),isinstance(simbad_table['DEC'][0], str)]):
                                                coords = SkyCoord(simbad_table['RA'], simbad_table['DEC'],
                                                                  unit=(u.hourangle, u.deg))
                                                simbad_table['RA'] = coords.ra.deg
                                                simbad_table['DEC'] = coords.dec.deg

                                            simbad_table.write(path2table, format="csv", overwrite=True)



                            if isinstance(npix, int):
                                npix = [npix, npix, npix, npix]  # left, right, bottom, top
                            else:
                                npix = npix

                            data = np.asarray(image)
                            if data.ndim == 3:
                                data = data[0, :, :]
                            if data.ndim == 2:
                                pass
                            else:
                                raise ValueError(f"image must be a 3D or 2D, got shape {data.shape}")

                            cel_wcs = WCS(header, naxis=2).celestial
                            ny, nx = data.shape
                            x_center = nx // 2.0
                            y_center = ny // 2.0

                            center_ra_deg, center_dec_deg = cel_wcs.all_pix2world(x_center, y_center, 0)
                            pix_scales = np.sqrt(header['PIXAR_A2'])
                            fov_x_deg = float(nx * pix_scales) / 3600
                            fov_y_deg = float(ny * pix_scales) / 3600
                            radius_deg = 0.5 * float(np.hypot(fov_x_deg, fov_y_deg))

                            if use_gaia:
                                query_gaia(path2table, center_ra_deg, center_dec_deg, radius_deg)
                            elif use_simbad:
                                query_simbad(path2table, center_ra_deg, center_dec_deg, radius_deg)
                            else:
                                log.error("Neither Gaia nor Simbad query was requested. No catalog will be fetched.")
                                return Table()

                            # fetch tabes...
                            table = Table.read(path2table)

                            # Add detector pixel coordinates from catalog sky coordinates.
                            ra_col = "ra" if "ra" in table.colnames else ("RA" if "RA" in table.colnames else None)
                            dec_col = "dec" if "dec" in table.colnames else ("DEC" if "DEC" in table.colnames else None)
                            if ra_col is None or dec_col is None:
                                log.warning("Gaia table does not include ra/dec columns; returning sky-only table.")
                                return table

                            ra_arr = np.asarray(np.ma.filled(np.ma.asarray(table[ra_col]), np.nan), dtype=float)
                            dec_arr = np.asarray(np.ma.filled(np.ma.asarray(table[dec_col]), np.nan), dtype=float)
                            x, y = cel_wcs.all_world2pix(ra_arr, dec_arr, 0)
                            table["x"] = np.asarray(x, dtype=float)
                            table["y"] = np.asarray(y, dtype=float)
                            table['method'] = np.asarray(['catalog'] * len(table), dtype=str)

                            mask = (
                                    (table["x"] >= npix[0] + border)
                                    & (table["x"] <= nx - (npix[1] + border))
                                    & (table["y"] >= npix[2] + border)
                                    & (table["y"] <= ny - (npix[3] + border))
                            )
                            table_selected = table[mask]
                            table_selected.write(path2table, format="csv", overwrite=True)
                            return table_selected


def estimate_bkg_and_rms(data,mask,n=15):
    """Estimate background median and RMS from cutout border pixels.

    Parameters
    ----------
    data : 2D-array
        Image cutout.
    mask: 2D-array
        Boolean mask for data. True indicate bad pixels to ignore.
    n : int, optional
        Number of boxes to use to define box_size. The default is 11.

    Returns
    -------
    bkg : float
        Background median.
    rms : float
        Robust RMS estimate.

    """
    data = np.asarray(data, dtype=float)
    sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
    bkg_estimator = MedianBackground()
    bkg = Background2D(
        data,
        mask=mask,
        box_size=int(np.ceil(np.max(data.shape)/np.sqrt(n))),
        filter_size=(3, 3),
        sigma_clip=sigma_clip,
        bkg_estimator=bkg_estimator
    )
    return bkg.background, bkg.background_rms

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
    psf,
    data,
    nanmask,
    oversampling=1,
    coresat=None,
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
    psf : 2D-array
        PSF model image.
    data : 2D-array
        Image cutout to fit.
    nanmask: list, None, optional
        nanmask is a boolean array of the same shape as data, where True values indicate pixels to be treated as NaN
        in the analysis.
    mask_size : int, optional
        Reserved/legacy argument (kept for API compatibility).
    oversampling : int, optional
        Oversampling factor of the PSF model relative to the data.
    coresat : float, optional
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
    # TODO: understand why we are exceeding search radius when refitting the source, and the difference between fit_radius and search_radius
    def _make_weights(data_fit, rms, center_x, center_y, core_mask_x, core_mask_y, fit_radius, coresat):
        w = np.zeros_like(data_fit, dtype=float)
        w[finite] = 1.0 / (np.nanmax(rms[finite])**2 + 1e-30)

        if fit_radius > 0 :
            rr2 = (xx - float(center_x)) ** 2 + (yy - float(center_y)) ** 2
            w[rr2 > float(fit_radius) ** 2] = 0.0

        if coresat > 0:
            rr2 = (xx - float(core_mask_x)) ** 2 + (yy - float(core_mask_y)) ** 2
            w[rr2 < float(coresat) ** 2] = 0.0
        return w

    # Robust background subtraction is critical at low S/N.
    # try:
    bkg, rms = estimate_bkg_and_rms(data)
    # except:
    #     log.warning("Robust background estimation failed; proceeding basic background estimation.")
    #     bkg = np.nanmedian(data)
    #     rms = np.nanstd(data)
    #     pass

    if bkg_subtract:
        data_fit = data - bkg
    else:
        data_fit = data

    # Use the PSF as the model (with the core optionally masked).
    # IMPORTANT: if the PSF is oversampled w.r.t. the data, tell photutils.
    psf_model = FittableImageModel(psf, oversampling=oversampling)

    # Use the LevMarLSQFitter to fit the PSF to the data.
    fitter = fitting.LevMarLSQFitter()

    ny, nx = data_fit.shape
    yy, xx = np.mgrid[0:ny, 0:nx]

    # # For saturated stars we want to keep the masked core fixed on the NaN core.
    if coresat is not None:
        core_mask_x = (nx - 1) / 2
        core_mask_y = (ny - 1) / 2
    # if coresat and coresat > 0 and np.any(~np.isfinite(data)):
    # if np.any(~np.isfinite(data)):
    else:
        coresat, core_mask_x, core_mask_y = estimate_nan_core(data, margin=0)

    # Reasonable initial guesses matter a lot for position fitting.
    x_center = (nx - 1) / 2
    y_center = (ny - 1) / 2

    finite = np.isfinite(data_fit)
    if np.any(finite):
        peak_snr = float(np.nanmax(data_fit[finite]) / (np.nanmax(rms[finite]) + 1e-12))
    else:
        peak_snr = 0.0

    # Cross-correlation peak gives a good starting point for faint sources.
    corr = fftconvolve(
        np.nan_to_num(data_fit, nan=0.0),
        psf[::-1, ::-1],
        mode="same",
    )

    # Restrict peak search to an area where we expect the source to be.
    # This greatly reduces catastrophic failures at very low S/N.
    if fit_radius is None:
        fit_radius = max(data.shape)

    if search_radius is None:
        search_radius = fit_radius

    rr2 = (xx - x_center) ** 2 + (yy - y_center) ** 2
    corr = corr.copy()
    corr[(rr2 > float(search_radius) ** 2)|(nanmask==1)] = -np.inf

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
    if coresat and coresat > 0:
        delta = float(max(3, int(coresat)))
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

        if first_pass_radius != float(fit_radius):
            weights1 = _make_weights(data_fit, rms, psf_model.x_0.value, psf_model.y_0.value, core_mask_x, core_mask_y, first_pass_radius, coresat)
            fit1 = fitter(psf_model, xx, yy, data_fit, weights=weights1, filter_non_finite=True)

            # Recenter for second pass.
            psf_model.x_0.value = fit1.x_0.value
            psf_model.y_0.value = fit1.y_0.value
            psf_model.flux.value = max(float(fit1.flux.value), 0.0)

        weights2 = _make_weights(data_fit, rms, psf_model.x_0.value, psf_model.y_0.value, core_mask_x, core_mask_y, float(fit_radius), coresat)
        fit_result = fitter(psf_model, xx, yy, data_fit, weights=weights2, filter_non_finite=True)
    else:
        weights = _make_weights(data_fit, rms, psf_model.x_0.value, psf_model.y_0.value, core_mask_x, core_mask_y, float(fit_radius), coresat)
        fit_result = fitter(psf_model, xx, yy, data_fit, weights=weights, filter_non_finite=True)

    # Step 8: Output the fitted flux and position
    fitted_flux = fit_result.flux.value
    fitted_x_pos = fit_result.x_0.value
    fitted_y_pos = fit_result.y_0.value

    if showplots and (abs(fitted_x_pos-x_center)>search_radius or abs(fitted_y_pos-y_center)>search_radius):
        norm = simple_norm(data_fit, stretch)
        # Plot in the same convention used elsewhere in this script.
        plt.imshow(data_fit, origin='lower', cmap=cmap, norm=norm)
        plt.plot(fitted_x_pos, fitted_y_pos, 'ob')
        plt.colorbar()
        plt.title('Data to fit with fitted center')
        plt.show()
        pass
    if np.any(np.isnan([fitted_x_pos,fitted_y_pos])):
        pass
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
    if len(bad)>0:
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

    else:
        return 0, float(cx), float(cy)

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
        plt.title(f'Extracted Star')
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


# def aperture_flag_short(ap_flag: int) -> str:
#     """Return a short description for SEP aperture-photometry flags.
#
#     Parameters
#     ----------
#     ap_flag : int
#         Bitmask flag returned by ``sep.sum_circle`` (or similar).
#
#     Returns
#     -------
#     str
#         One-word summary (e.g. ``'ok'``, ``'truncated'``, ``'maskedpixels'``).
#
#     Notes
#     -----
#     This routine decodes a bitmask and returns the highest-priority label.
#
#     """
#     f = int(ap_flag)
#     if f == 0:
#         return "ok"
#     if f & getattr(sep, "APER_ALLMASKED", 64):
#         return "allmasked"
#     if f & getattr(sep, "APER_TRUNC", 16):
#         return "truncated"
#     if f & getattr(sep, "APER_NONPOSITIVE", 128):
#         return "nonpositive"
#     if f & getattr(sep, "APER_HASMASKED", 32):
#         return "maskedpixels"
#     return "flagged"


# def _as_fixed_str_array(values: list[str], *, width: int = 64) -> np.ndarray:
#     """Return a fixed-width unicode array for safe insertion into Astropy tables.
#
#     Parameters
#     ----------
#     values : list of str
#         Input strings.
#     width : int, optional
#         Fixed string width (characters).
#
#     Returns
#     -------
#     numpy.ndarray
#         Unicode array with dtype ``U<width>``.
#
#     """
#     if width <= 0:
#         width = 1
#     return np.asarray(values, dtype=f"U{int(width)}")

# def select_table(
#     objects_tbl: Table,
#     separation_pix = None,
#     center: Literal["centroid", "peak"] = "peak",
#     peak_col: str = "peak",
#     ap_snr: float | None= None,
#     maxrat: float | None = None,
#     flag_sel: list[int] | None = None,
#     ap_flag_sel: list[int] | None = None,
#     window_shape: Literal["circle", "square"] = "circle",
# ) -> Table:
#     """Select detections using filters + non-maximum suppression.
#
#     Parameters
#     ----------
#     objects_tbl : astropy.table.Table
#         SEP detections table.
#     separation_pix : float, optional
#         If provided, keep only the brightest detection within this radius.
#     center : {'centroid', 'peak'}, optional
#         Coordinates used for the separation check.
#     peak_col : str, optional
#         Column used to rank detections (default: ``'peak'``).
#     ap_snr : float, optional
#         If provided, apply an SNR cut (uses ``'peak_snr'`` if ``peak_col='peak'``
#         else uses ``'ap_snr'``).
#     maxrat : float, optional
#         If provided, remove elongated detections using the ``'ellipt'`` column.
#     flag_sel : list of int, optional
#         Keep only rows whose SEP detection flag (``'flag'``) is exactly in this list.
#     ap_flag_sel : list of int, optional
#         Keep only rows whose aperture flag (``'ap_flag'``) is exactly in this list.
#     window_shape : {'circle', 'square'}, optional
#         Neighborhood shape for the separation check.
#
#     Returns
#     -------
#     astropy.table.Table
#         Filtered table with a ``'id'`` column added.
#
#     Notes
#     -----
#     SEP coordinates are 0-indexed numpy pixel coordinates.
#
#     """
#     # Work on a copy: this function is a selector and should not mutate inputs.
#     objects_tbl = objects_tbl.copy()
#
#     # Remove objects that are too elongated to be astrophysical.
#     if maxrat is not None:
#         if "ellipt" not in objects_tbl.colnames:
#             raise ValueError("objects_tbl is missing required column 'ellipt' for maxrat filtering")
#         objects_tbl = objects_tbl[np.asarray(objects_tbl["ellipt"], dtype=float) <= float(maxrat)].copy()
#
#     # Remove objects with SNR lower than ap_snr.
#     if ap_snr is not None:
#         # If the user is ranking by peak, apply the SNR cut to the peak too.
#         snr_col = "peak_snr" if peak_col == "peak" else "ap_snr"
#         if snr_col not in objects_tbl.colnames:
#             raise ValueError(f"objects_tbl is missing required column '{snr_col}' for snr filtering")
#         objects_tbl = objects_tbl[np.asarray(objects_tbl[snr_col], dtype=float) >= float(ap_snr)].copy()
#
#     # Filter by SEP detection flags (SExtractor-style bitmask stored as an int).
#     # NOTE: As requested, this is an *exact match* on the integer value.
#     if flag_sel is not None:
#         if "flag" not in objects_tbl.colnames:
#             raise ValueError("flag_sel was provided but objects_tbl has no 'flag' column")
#         good = np.isin(np.asarray(objects_tbl["flag"], dtype=int), np.asarray(flag_sel, dtype=int))
#         objects_tbl = objects_tbl[good].copy()
#
#     # Filter by SEP aperture-photometry flags (returned by sep.sum_circle).
#     # NOTE: As requested, this is an *exact match* on the integer value.
#     if ap_flag_sel is not None:
#         if "ap_flag" not in objects_tbl.colnames:
#             raise ValueError("ap_flag_sel was provided but objects_tbl has no 'ap_flag' column")
#         good = np.isin(np.asarray(objects_tbl["ap_flag"], dtype=int), np.asarray(ap_flag_sel, dtype=int))
#         objects_tbl = objects_tbl[good].copy()
#
#     if separation_pix is None or separation_pix <= 0:
#         objects_tbl["id"] = np.arange(len(objects_tbl), dtype=int)
#         return objects_tbl.copy()
#
#     # After filtering, we may end up with an empty table; short-circuit.
#     if len(objects_tbl) == 0:
#         objects_tbl["id"] = np.arange(0, dtype=int)
#         return objects_tbl.copy()
#
#     if center not in ("centroid", "peak"):
#         raise ValueError("center must be 'centroid' or 'peak'")
#
#     if peak_col not in objects_tbl.colnames:
#         raise ValueError(
#             f"peak_col='{peak_col}' not in table columns. Available: {', '.join(objects_tbl.colnames)}"
#         )
#
#     if center == "peak":
#         req = {"xpeak", "ypeak"}
#         xcol, ycol = "xpeak", "ypeak"
#     else:
#         req = {"x", "y"}
#         xcol, ycol = "x", "y"
#
#     missing = req.difference(objects_tbl.colnames)
#     if missing:
#         raise ValueError(
#             f"objects_tbl missing required columns for center='{center}': {', '.join(sorted(missing))}"
#         )
#
#     if separation_pix is not None:
#         # Coordinates in SEP are 0-indexed; the relative distances are the same in DS9.
#         x = np.asarray(objects_tbl[xcol], dtype=float)
#         y = np.asarray(objects_tbl[ycol], dtype=float)
#         coords = np.column_stack([x, y])
#
#         peaks = np.asarray(objects_tbl[peak_col], dtype=float)
#         # Highest peak first; stable tie-breaker by index (lower index first).
#         order = np.lexsort((np.arange(len(peaks)), -peaks))
#
#         if window_shape not in ("circle", "square"):
#             raise ValueError("window_shape must be 'circle' or 'square'")
#
#         sepv = float(separation_pix)
#         # A square of side (2*separation_pix + 1) has half-width separation_pix + 0.5.
#         sep_eff = (sepv + 0.5) if window_shape == "square" else sepv
#
#         suppressed = np.zeros(len(objects_tbl), dtype=bool)
#         keep: list[int] = []
#         # Prefer scipy KDTree if available (fast for large catalogs), otherwise fallback.
#         try:
#             from scipy.spatial import cKDTree  # type: ignore
#
#             tree = cKDTree(coords)
#             for idx in order:
#                 if suppressed[idx]:
#                     continue
#                 keep.append(int(idx))
#                 if window_shape == "square":
#                     # Chebyshev (L-infinity) neighborhood => axis-aligned square.
#                     neighbors = cast(
#                         list[int],
#                         tree.query_ball_point(coords[idx], r=float(sep_eff), p=np.inf),
#                     )
#                 else:
#                     neighbors = cast(list[int], tree.query_ball_point(coords[idx], r=float(sep_eff)))
#                 for j in neighbors:
#                     suppressed[int(j)] = True
#                 suppressed[idx] = False
#         except Exception:
#             for idx in order:
#                 if suppressed[idx]:
#                     continue
#                 keep.append(int(idx))
#                 dx = coords[:, 0] - coords[idx, 0]
#                 dy = coords[:, 1] - coords[idx, 1]
#                 if window_shape == "square":
#                     mask = (np.abs(dx) <= sep_eff) & (np.abs(dy) <= sep_eff)
#                 else:
#                     mask = (dx * dx + dy * dy) <= (sepv * sepv)
#                 suppressed[mask] = True
#                 suppressed[idx] = False
#
#         # Return in original order for easier cross-referencing.
#         keep_sorted = np.sort(np.asarray(keep, dtype=int))
#         sel_objects_tbl=objects_tbl[keep_sorted].copy()
#         sel_objects_tbl["id"] = np.arange(len(sel_objects_tbl), dtype=int)
#         return sel_objects_tbl
#     else:
#         objects_tbl["id"] = np.arange(len(objects_tbl), dtype=int)
#         return objects_tbl.copy()



def write_ds9_regions_from_sep_objects(
    objects_tbl: Table,
    output_path: str | Path,
    shape: Literal[ "circle", "square"] = "circle",
    color: str = "red",
    circle_radius = 5,
) -> Path:
    """Write a DS9 region file (image/pixel coordinates) from SEP detections.

    Parameters
    ----------
    objects_tbl : astropy.table.Table
        SEP detections table (requires at least x/y/a/b/theta; and xpeak/ypeak if
        ``center='peak'`` is used).
    output_path : str or pathlib.Path
        Output ``.reg`` path.
    shape : {'circle', 'square'}, optional
        Region primitive.
    color : str, optional
        DS9 region color.
    circle_radius : float, optional
        Circle radius rule for ``shape='circle'``/``'square'``. If a float, use
        a fixed radius in pixels. If None, read a per-row radius from
        ``circle_radius_col``.

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

    req_cols = {"id","x", "y"}
    missing = req_cols.difference(objects_tbl.colnames)
    if missing:
        raise ValueError(
            "objects_tbl is missing required SEP columns: "
            f"{', '.join(sorted(missing))}. Available columns: {', '.join(objects_tbl.colnames)}"
        )

    if shape not in ("circle", "square"):
        raise ValueError("shape must be 'circle', or 'square'")
    if not np.isfinite(float(circle_radius)) or float(circle_radius) <= 0:
        raise ValueError("numeric circle_radius must be a finite positive number (pixels)")

    for i in range(len(objects_tbl)):
        n=objects_tbl['id'][i]

        # DS9 is 1-indexed for image pixels.
        x = float(objects_tbl["x"][i]) + 1.0
        y = float(objects_tbl["y"][i]) + 1.0

        # For circles: r is the radius.
        # For squares: side length will be (2*r + 1).
        r = float(circle_radius)

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

class DAO():
    """
    The spaceKLIP DAOStarFinder source extraction tools class for wide-field images.

    """

    def __init__(self,
                npix=0,
                oversampling=1,
                threshold=4.0,
                sharpness_range=(0.2, 1.0),
                roundness_range=(-1.0, 1.0),
                fwhm=2.5,
                group_radius=15.0,
                catalog=None,
                nan_lim_percent=0.51,
                two_pass=True,
                showplots=False,
                psf=None,
                 ):
        """
        Initialize the spaceKLIP DAOStarFinder source extraction tools class.

        Parameters
        ----------
        npix : int or list of four int, optional
            Number of pixels used to pad around the frames. If int, the same
            number of pixels will be padded on each side. If list of four int,
            a different number of pixels can be padded on the [left, right,
            bottom, top] of the frames. The default is 1.Need to evaluate the true border of the real data
        oversampling : int, optional
            Oversampling factor of ``psf`` relative to detector pixels.
        threshold : float, optional
            ``DAOStarFinder`` detection threshold in units of the image RMS.
        fwhm : float, optional
            PSF FWHM (pixels) passed to ``DAOStarFinder``.
        sharpness_range : tuple of float, optional
            Acceptable range of ``DAOStarFinder`` sharpness values.
        roundness_range : tuple of float, optional
            Acceptable range of ``DAOStarFinder`` roundness values.
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
        two_pass : bool, optional
            If True and ``fit_radius`` is set, do a broad pass followed by a tighter
            pass when refining coordinates.
        showplots : bool, optional
            If True, show a diagnostic plot when refining coordinates for problematic fits.
        psf : 2D-array
            PSF model image passed directly to ``fit_psf``.
        Returns
        -------
        None.

        """

        if isinstance(npix, int):
            self.npix = [npix, npix, npix, npix]  # left, right, bottom, top
        else:
            self.npix = npix
        if len(self.npix) != 4:
            raise UserWarning('Parameter npix must either be an int or a list of four int (left, right, bottom, top)')
        self.oversampling=oversampling
        self.threshold=threshold
        self.fwhm=fwhm
        self.sharpness_range=sharpness_range
        self.roundness_range=roundness_range
        self.group_radius=group_radius
        self.catalog=catalog
        self.nan_lim_percent=nan_lim_percent
        self.two_pass = two_pass
        self.showplots = showplots
        self.psf=psf
        pass

    def _dao(self,data,mask,mrms,border=3):
        """Recover additional faint point sources with ``DAOStarFinder``.

        Parameters
        ----------
        data : 2D-array
            Science image.
        mask : 2D-array
            Boolean mask of the science image. True indicate bad pixels to ignore.
        mrms : float
            median from RMS estimate.
        border: int, optional
            exclude border of x pixel from image to confirm coordinates are within the fov

        Returns
        -------
        list of dict
            Candidate dictionaries with ``x``, ``y``, ``peak``,
            ``coresat``, and ``method``.

        """
        data = np.asarray(data, dtype=float)
        nx , ny = data.shape
        finite = np.isfinite(data)
        vals = data[finite]
        if vals.size == 0:
            return None
        if not np.isfinite(mrms) or mrms <= 0:
            mrms = 1.0

        dao = DAOStarFinder(fwhm=float(self.fwhm), threshold=float(self.threshold * mrms),
                            sharplo=self.sharpness_range[0], sharphi=self.sharpness_range[1],
                            roundlo=self.roundness_range[0], roundhi=self.roundness_range[1])

        tbl = dao(np.nan_to_num(data, nan=0.0), mask=mask)
        if tbl is None or len(tbl) == 0:
            return None
        # out = []
        tbl['coresat'] = 0
        tbl['method'] = 'dao'
        tbl.rename_column('xcentroid', 'x')
        tbl.rename_column('ycentroid', 'y')
        tbl.rename_column('roundness1', 'roundness')

        mask = (
                (tbl["x"] >= self.npix[0] + border)
                & (tbl["x"] <= nx - (self.npix[1] + border))
                & (tbl["y"] >= self.npix[2] + border)
                & (tbl["y"] <= ny - (self.npix[3] + border))
        )
        tbl_selected = tbl[mask]['x','y','peak','coresat','method','sharpness','roundness']

        return tbl_selected

    def _starfinder(self, data, mask, mrms, border=3):
        """Recover additional faint point sources using a custom PSF with ``StarFinder``
        and manually calculate sharpness to mimic DAOStarFinder.
        """
        data = np.asarray(data, dtype=float)
        nx, ny = data.shape
        finite = np.isfinite(data)
        vals = data[finite]

        if vals.size == 0:
            return None
        if not np.isfinite(mrms) or mrms <= 0:
            mrms = 1.0

        # Initialize StarFinder using supported parameters
        finder = StarFinder(threshold=float(self.threshold * mrms),
                            kernel=self.psf)

        # Run detection
        tbl = finder(np.nan_to_num(data, nan=0.0), mask=mask)

        if tbl is None or len(tbl) == 0:
            return None

        # Conform to your pipeline's existing structure
        tbl['coresat'] = 0
        tbl['method'] = 'starfinder'
        tbl.rename_column('xcentroid', 'x')
        tbl.rename_column('ycentroid', 'y')
        tbl.rename_column('max_value', 'peak')

        # --- MANUAL SHARPNESS CALCULATION ---
        sharpness_list = []
        box_radius = 2  # Extracts a 5x5 sub-grid around the centroid core

        for row in tbl:
            xi, yi = int(round(row['x'])), int(round(row['y']))

            # Safely slice around the coordinates without spilling off image boundaries
            y_slice = slice(max(0, yi - box_radius), min(ny, yi + box_radius + 1))
            x_slice = slice(max(0, xi - box_radius), min(nx, xi + box_radius + 1))
            cutout = data[y_slice, x_slice]

            if cutout.size > 0 and np.any(cutout > 0):
                # Sharpness = central peak / total core energy sum
                total_flux = np.sum(cutout)
                peak_val = data[yi, xi]

                sh_val = peak_val / total_flux if total_flux > 0 else 0.0
            else:
                sh_val = 0.0

            sharpness_list.append(sh_val)

        tbl['sharpness'] = sharpness_list
        # -------------------------------------

        # Apply your boundary AND downstream sharpness range filters safely
        mask_indices = (
                (tbl["x"] >= self.npix[0] + border)
                & (tbl["x"] <= nx - (self.npix[1] + border))
                & (tbl["y"] >= self.npix[2] + border)
                & (tbl["y"] <= ny - (self.npix[3] + border))
                & (tbl["sharpness"] >= self.sharpness_range[0])
                & (tbl["sharpness"] <= self.sharpness_range[1])
                & (tbl["roundness"] >= self.roundness_range[0])
                & (tbl["roundness"] <= self.roundness_range[1])
        )

        # Filter table and cleanly select your required columns
        tbl_selected = tbl[mask_indices]['x', 'y', 'peak', 'coresat', 'method', 'sharpness', 'roundness']

        return tbl_selected

    def _candidate_radius(self,c, base_radius, prov_sat_r=0.0, rmax=30):
        """Return an adaptive grouping radius for one candidate.

        Bright sources are given a larger grouping window using their DAO
        peak, and saturated sources get an additional boost from the
        estimated saturated-core size.  Because ``coresat`` in the raw
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
        sat_r = max(float(c.get("coresat", 0.0)), float(prov_sat_r))
        if np.isfinite(sat_r) and sat_r > 0:
            r = max(r, float(base_radius) + 5 * sat_r)
        return float(np.clip(r, float(base_radius), rmax))

    def _effective_radius(self,xvals, yvals, xref, yref, base_radius):
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

    def _group_and_select(self,candidates, data_arr, psf):
        """Group nearby candidates and select one representative per star.

        DAOStarFinder often returns several detections for a single bright or
        saturated star — one near the core and several on the PSF wings.

        Parameters
        ----------
        candidates : list of dict
            Full (ungrouped) candidate list from ``_dao``.
        data_arr : 2D-array
            Science image (used for NaN proximity checks).
        group_radius : float
            Maximum separation (pixels) for two candidates to be in the
            same group.
        npix : int or list of four int, optional
            Number of pixels used to pad around the frames. If int, the same
            number of pixels will be padded on each side. If list of four int,
            a different number of pixels can be padded on the [left, right,
            bottom, top] of the frames. The default is 1.Need to evaluate the true border of the real data

        Returns
        -------
        list of dict
            One representative candidate dict per group, with ``coresat``
            set if a saturated core was detected.

        """
        if len(candidates) == 0:
            return []

        xs = np.array([c["x"] for c in candidates], dtype=float)
        ys = np.array([c["y"] for c in candidates], dtype=float)
        ny_arr, nx_arr = data_arr.shape

        # Pre-compute a provisional NaN-core radius for every candidate so
        # that _candidate_radius can scale the grouping window correctly even
        # before the formal coresat is estimated inside _group_and_select.
        _quick_r = max(1, int(self.group_radius))
        _prov_sat = []
        _rmax = []
        _candidates = []
        for _c in candidates:
            _cx, _cy = float(_c["x"]), float(_c["y"])
            _xlo = int(_cx) - _quick_r
            _xhi = int(_cx) + _quick_r + 1
            _ylo = int(_cy) - _quick_r
            _yhi = int(_cy) + _quick_r + 1
            _patch = data_arr[_ylo:_yhi, _xlo:_xhi]
            _sr, _, _ = estimate_nan_core(_patch, margin=1)
            _c['coresat'] = _sr
            _prov_sat.append(float(_sr))
            _rmax.append(float(max(_patch.shape)))
            _candidates.append(_c)

        candidates = _candidates
        n=len(candidates)

        cand_radii = np.array(
            [self._candidate_radius(c, _quick_r, ps, np.nanmedian(_rmax)) for c, ps in zip(candidates, _prov_sat)],
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
        selected = []
        seen_catalog_ids = set()  # Prevent cross-group duplicate entries of the same star

        for indices in groups.values():
            group_cands = [candidates[i] for i in indices]

            catalog_member_indices = [k for k, c in enumerate(group_cands)
                                      if c.get("method") == "catalog"]
            not_catalog_member_indices = [k for k, c in enumerate(group_cands)
                                          if c.get("method") != "catalog"]

            # --- 1. PRIORITIZE CATALOG MEMBERS FIRST ---
            if catalog_member_indices:
                # Sort catalog indices by peak brightness (brightest first)
                catalog_member_indices.sort(key=lambda k: float(group_cands[k].get("peak", -np.inf)), reverse=True)

                catalog_winner_found = False
                for k in catalog_member_indices:
                    candidate = dict(group_cands[k])

                    # Deduplication check
                    obj_id = candidate.get("id") or candidate.get(
                        "source_id") or f"{candidate['x']:.2f}_{candidate['y']:.2f}"
                    if obj_id in seen_catalog_ids:
                        continue

                    cx, cy = float(candidate["x"]), float(candidate["y"])

                    # Keep the border check to prevent out-of-bounds errors
                    if cx < self.npix[0] or cy < self.npix[2] or cx > nx_arr - self.npix[1] or cy > ny_arr - self.npix[
                        3]:
                        continue

                    # NOTE: We can skip the cutout extraction and the ~np.isfinite(local)
                    # validation check entirely! Catalog stars get an automatic pass.

                    # Found the single brightest valid catalog star for this group!
                    selected.append(candidate)
                    seen_catalog_ids.add(obj_id)
                    catalog_winner_found = True
                    break  # Stop checking other catalog stars in this group

                if catalog_winner_found:
                    continue  # Successfully processed this group. Skip the DAO fallback completely.

                # If all catalog stars in this group failed the NaN limit/border cuts,
                # the code naturally falls through to the 'else' block below to evaluate the DAO detections instead.

            # --- 2. FALLBACK TO DAO MEMBERS (OR IF CATALOGS FAILED QUALITY CUTS) ---
            # We change this 'else:' to a flat block since catalog success triggers 'continue'
            peaks = []
            d2 = []
            valid_not_catalog_indices = []

            for k in not_catalog_member_indices:
                candidate = dict(group_cands[k])
                cx, cy = float(candidate["x"]), float(candidate["y"])
                if cx < self.npix[0] or cy < self.npix[2] or cx > nx_arr - self.npix[1] or cy > ny_arr - self.npix[3]:
                    continue
                half = int(max(15, self.group_radius))
                xlo = max(0, int(round(cx)) - half)
                xhi = min(nx_arr, int(round(cx)) + half + 1)
                ylo = max(0, int(round(cy)) - half)
                yhi = min(ny_arr, int(round(cy)) + half + 1)
                local = data_arr[ylo:yhi, xlo:xhi]
                if np.sum(~np.isfinite(local)) > np.ceil(local.shape[0] * local.shape[1] * self.nan_lim_percent):
                    continue  # Avoid spurious large coresat estimates from mostly-NaN cutouts.

                # Track indices that actually survived the initial border and NaN filters
                valid_not_catalog_indices.append(k)

                if np.any(~np.isfinite(local)):
                    sat_flag = True
                    # Saturated group: estimate the NaN-core centroid on a group-wide
                    # cutout and use that as the representative source position.
                    sr, xcore, ycore = estimate_nan_core(local, margin=1)
                    peaks.append(float(candidate.get("peak", 0.0)))  # Flat float to prevent indexing quirks later
                    d2.append((cx - (xcore + xlo)) ** 2 + (cy - (ycore + ylo)) ** 2)
                else:
                    sat_flag = False
                    # Unsaturated group: use a group-wide PSF matched-filter peak to
                    # avoid keeping a bright wing knot as the representative.
                    peaks.append(float(candidate.get("peak", 0.0)))
                    try:
                        masked_psf_data = downsample_psf_to_detector(psf, self.oversampling)
                        masked_psf_data = np.asarray(masked_psf_data, dtype=float)
                        psf_sum = np.nansum(masked_psf_data)
                        if np.isfinite(psf_sum) and psf_sum > 0:
                            masked_psf_data = masked_psf_data / psf_sum
                            img = np.nan_to_num(local - np.nanmedian(local), nan=0.0)
                            corr = fftconvolve(img, masked_psf_data[::-1, ::-1], mode="same")
                            iy, ix = np.unravel_index(np.nanargmax(corr), corr.shape)
                            d2.append((cx - (float(ix) + xlo)) ** 2 + (cy - (float(iy) + ylo)) ** 2)
                    except Exception:
                        # If cross-correlation fails, match array length by stripping this index back out
                        valid_not_catalog_indices.pop()
                        peaks.pop()
                        continue

            peaks = np.array(peaks)
            if len(peaks) == 0:
                # All candidates have been dropped.
                continue

            # --- 3. RE-ALIGNED DAO REFINEMENT PIPELINE ---
            if len(d2) > 0:
                # lexsort sorts by d2 ascending, then by peaks descending (due to minus sign)
                best_sub_idx = int(np.lexsort((d2, -peaks))[0])
            else:
                best_sub_idx = int(np.argmax(peaks))

            # Map the inner sub-index loop choice cleanly back to the true group candidate index
            best_idx = valid_not_catalog_indices[best_sub_idx]
            best = dict(group_cands[best_idx])

            # Use the measured wing spread to enlarge the local window used for
            # the final centroid/core-radius refinement.
            xg = np.array([float(c["x"]) for c in group_cands], dtype=float)
            yg = np.array([float(c["y"]) for c in group_cands], dtype=float)
            refx, refy = float(best["x"]), float(best["y"])
            eff_group_radius = self._effective_radius(xg, yg, refx, refy, self.group_radius)

            # Populate coresat for saturated representatives.
            if sat_flag:
                cx, cy = float(best["x"]), float(best["y"])
                half = int(max(15, eff_group_radius))
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
                    if best["x"] < self.npix[0] or best["y"] < self.npix[2] or best["x"] > nx_arr - self.npix[1] or \
                            best["y"] > ny_arr - self.npix[3]:
                        continue
            else:
                # For unsaturated groups, place the representative on the local
                # PSF-correlation peak if it was measured above.
                half = int(max(8, eff_group_radius))
                xlo = max(0, int(np.floor(np.min(xg))) - half)
                xhi = min(nx_arr, int(np.ceil(np.max(xg))) + half + 1)
                ylo = max(0, int(np.floor(np.min(yg))) - half)
                yhi = min(ny_arr, int(np.ceil(np.max(yg))) + half + 1)
                local = data_arr[ylo:yhi, xlo:xhi]
                try:
                    masked_psf_data = downsample_psf_to_detector(psf, self.oversampling)
                    masked_psf_data = np.asarray(masked_psf_data, dtype=float)
                    psf_sum = np.nansum(masked_psf_data)
                    if np.isfinite(psf_sum) and psf_sum > 0:
                        masked_psf_data = masked_psf_data / psf_sum
                        img = np.nan_to_num(local - np.nanmedian(local), nan=0.0)
                        corr = fftconvolve(img, masked_psf_data[::-1, ::-1], mode="same")
                        iy, ix = np.unravel_index(np.nanargmax(corr), corr.shape)
                        best["x"] = float(ix + xlo)
                        best["y"] = float(iy + ylo)
                        if best["x"] < self.npix[0] or best["y"] < self.npix[2] or best["x"] > nx_arr - self.npix[1] or \
                                best["y"] > ny_arr - self.npix[3]:
                            continue
                except Exception:
                    continue

            selected.append(best)

        log.info(f"Using {np.sum([i['method']=='catalog' for i in selected])} catalog seeds + {np.sum([i['method']!='catalog' for i in selected])} DAO detections after selections.")
        tbl = Table(rows=selected)
        tbl['id']=[i for i in range(len(tbl))]
        return tbl

    def _refine_coordinates(self,candidates, data, nanmask, psf,search_radius=None,fit_radius=71):
        '''
        Perform coordinates refinement using PSF (wings if core is saturated) fit.

        Args:
            candidates : list of dict
                Candidate table .
            data : 2D-array
                Background-subtracted science image.
            nanmask: 2D-array (bool)
                Mask of NaN pixels  used to identify candidates with saturated cores or candidates too close to the edge.
            psf : 2D-array
                PSF model image passed directly to ``fit_psf``.
            search_radius : float, optional
                Search radius (pixels) for the matched-filter initialization.
            fit_radius : float, optional
                Radius (in data pixels) defining the fitting region. If None, fits the
                full cutout.

        Returns:
            astropy.table.Table containing the refined coordinates of the candidates
        '''

        rows = []
        for c in candidates:
            id = c["id"]
            method = c["method"]
            x_fit, y_fit = c["x"], c["y"]
            nx, ny = data.shape
            # local cutout around candidate
            half = int(min(psf.shape[0] // 2, fit_radius))
            xlo = max(0, int(round(x_fit)) - half)
            xhi = min(nx, int(round(x_fit)) + half + 1)
            ylo = max(0, int(round(y_fit)) - half)
            yhi = min(ny, int(round(y_fit)) + half + 1)
            cut = data[ylo:yhi, xlo:xhi]
            nanmaskcut = nanmask[ylo:yhi, xlo:xhi]
            sat_r, _, _ = estimate_nan_core(cut, center=(x_fit - xlo, y_fit - ylo), margin=1)

            nxpsf, nypsf = psf.shape
            xlo_psf = max(0, int(round(nxpsf//2)) - half)
            xhi_psf = min(nx, int(round(nxpsf//2)) + half + 1)
            ylo_psf = max(0, int(round(nypsf//2)) - half)
            yhi_psf = min(ny, int(round(nypsf//2)) + half + 1)
            psfcut = psf[ylo_psf:yhi_psf, xlo_psf:xhi_psf]
            ydat, xdat = np.indices(psfcut.shape)
            if sat_r > 0:
                psfcut = mask_within_radius(psfcut.copy(), xdat, ydat, psfcut.shape[1]//2, psfcut.shape[0]//2, sat_r, c=np.nan)

            if search_radius is None:
                search_radius = min(25, fit_radius)
            if method != 'catalog':
                fx, fy, _ = fit_psf(
                    psf=psfcut,
                    data=cut,
                    nanmask=nanmaskcut,
                    oversampling=self.oversampling,
                    coresat=sat_r,
                    fit_radius=fit_radius,
                    search_radius=search_radius,
                    bkg_subtract=False,
                    two_pass=self.two_pass,
                    showplots=self.showplots,
                )
                x_fit = float(fx + xlo)
                y_fit = float(fy + ylo)

            if not np.isnan(x_fit) and not np.isnan(y_fit):
                rows.append((
                    id,
                    x_fit, y_fit,
                    sat_r,
                    method,
                ))

        names = [
            "id",
            "x", "y",
            "coresat", "det_method"]
        tbl = Table(rows=rows, names=names)

        # Keep only rows with finite coordinates so bad fits are excluded from CSV/DS9.
        # Prefer peak coordinates (used with region_center="peak"), fallback to x/y.
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

    def dao_source_extractor(self,
        data,
        nanmask,
        dao=True
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
            science image.
        nanmask: 2D-array (bool)
            Mask of NaN pixels  used to identify candidates with saturated cores or candidates too close to the edge.
        dao : bool, optional
            Whether to run DAOStarFinder or StarFinder. If False, run StarFinder. Default is True.

        Returns
        -------
        astropy.table.Table
            Catalog with fitted detector coordinates, aperture quantities,
            detection metadata (``det_method``, ``coresat``, ``psf_flux``).

        Notes
        -----
        Candidates whose PSF fit fails are assigned NaN coordinates and are
        removed from the returned table before output.
        """


        data = np.asarray(data, dtype=float)
        data[nanmask==1] = np.nan

        struct_element = np.ones((3, 3), dtype=bool)
        dilated_mask = binary_dilation(nanmask.astype(bool), structure=struct_element)

        bkg, rms = estimate_bkg_and_rms(data,mask=dilated_mask)
        data_subtracted = data - bkg

        #Candidate detection via DAOStarFinder, or StarFinder
        if dao:
            dao_catalog = self._dao(data_subtracted,mask=dilated_mask,mrms=np.nanmedian(rms))
        else:
            dao_catalog = self._starfinder(data,mask=dilated_mask,mrms=np.nanmedian(rms))


        # Catalog candidates are prepended so they have priority inside each group.
        if self.catalog is not None and dao_catalog is not None:
            all_candidates = vstack([self.catalog , dao_catalog])
        elif  self.catalog is None and dao_catalog is not None:
            all_candidates = dao_catalog
        elif  self.catalog is not None and dao_catalog is None:
            all_candidates = self.catalog
        else:
            raise ValueError(f"No candidate detected in DAOStarFinder and not cstalog parsed for this data.")
        # Group detections from the same star (bright stars produce multiple wing
        # detections) and select one representative per group.  The representative
        # is the catalog seed (if provided), the saturated NaN core, or the
        # PSF-correlation peak for unsaturated sources.
        selected_candidates = self._group_and_select(all_candidates, data_subtracted, self.psf)

        return selected_candidates