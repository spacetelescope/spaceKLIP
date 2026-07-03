import logging
from pathlib import Path
from typing import Literal
import matplotlib.pylab as plt
from astropy.visualization import simple_norm
from photutils.psf import FittableImageModel
from astropy.modeling import fitting
import spaceKLIP.utils as ut
from astropy.table import Table, vstack
from astropy.wcs import WCS
from scipy.signal import fftconvolve
from photutils.detection import DAOStarFinder,StarFinder
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from astropy import units as u
from astropy.coordinates import SkyCoord
import requests
import numpy as np
from skimage.measure import label, regionprops
from scipy.ndimage import binary_dilation
from photutils.aperture import CircularAperture, aperture_photometry
from scipy.spatial import KDTree
from skimage.color import label2rgb
from astropy.visualization import ZScaleInterval

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
                                use_allwise=False,
                                use_simbad=False,
                                border=3,
                                npix=0,
                                use_mocadb: bool = False,
                                fwhm: float = 2.5,
                                threshold: float = 2.0,
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
                            use_allwise : bool, optional
                                If True, queries and filters for objects with ALLWISE W2 measurements.
                                Defaults to False.
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
                            use_mocadb : bool, optional
                                If True, queries the MOCADB database for a MOC of the image footprint.

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
                                           use_allwise: bool = False,
                                           ):
                                """
                                Helper to fetch Gaia DR3 source data, with an optional ALLWISE W2 filter proxy.

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
                                use_allwise : bool, optional
                                    If True, queries and filters for objects with ALLWISE W2 measurements.
                                    Defaults to False.

                                Returns
                                -------
                                None
                                """
                                from astroquery.gaia import Gaia
                                #TODO: fix allwise search

                                # 1. Adapt query columns and table relations based on AllWISE flag
                                if use_allwise:
                                    # Requesting fields across the corrected external catalog tables
                                    select_fields = "g.source_id, g.ra, g.dec, g.parallax, g.parallax_error, g.phot_g_mean_mag, w.w2_m_mag AS flux_w2_mag"
                                    table_joins = (
                                        f"{gaia_table} AS g "
                                        f"INNER JOIN gaiadr3.allwise_best_neighbour AS x ON g.source_id = x.source_id "
                                        f"INNER JOIN gaiadr3.allwise_original_valid AS w ON x.allwise_oid = w.allwise_oid"
                                    )
                                    # Using explicit aliases avoids table ambiguities in positional processing
                                    where_clause = (
                                        f"1=CONTAINS(POINT('ICRS', g.ra, g.dec), CIRCLE('ICRS', {center_ra_deg:.12f}, {center_dec_deg:.12f}, {radius_deg:.12f})) "
                                        f"AND w.w2_m_mag IS NOT NULL"
                                    )
                                else:
                                    # Standard fast single-table fallback
                                    select_fields = "source_id, ra, dec, parallax, parallax_error, phot_g_mean_mag"
                                    table_joins = f"{gaia_table}"
                                    where_clause = f"1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {center_ra_deg:.12f}, {center_dec_deg:.12f}, {radius_deg:.12f}))"

                                # 2. Build the unified ADQL query string
                                query = f"SELECT {select_fields} FROM {table_joins} WHERE {where_clause}"

                                # 3. Configure Gaia service settings and execute the pipeline
                                Gaia.MAIN_GAIA_TABLE = gaia_table
                                Gaia.ROW_LIMIT = int(-1)
                                Gaia.launch_job_async(
                                    query=query,
                                    dump_to_file=True,
                                    verbose=False,
                                    output_format='csv',
                                    output_file=path2table
                                )

                            def query_simbad(path2table,
                                            center_ra_deg,
                                            center_dec_deg,
                                            radius_deg,
                                            use_mocadb=False
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
                                            use_mocadb : bool, optional
                                                If True, queries and filters for objects with MOCADB measurements.
                                                Defaults to False.

                                            Returns
                                            -------

                                            """
                                            from astroquery.simbad import Simbad
                                            # Define center coordinates and radius
                                            coord = SkyCoord(ra=center_ra_deg, dec=center_dec_deg, unit=(u.deg, u.deg), frame='icrs')

                                            # 1. Reset fields to default, then add your existing filters plus M, W2, and I2
                                            Simbad.ROW_LIMIT = -1
                                            Simbad.reset_votable_fields()
                                            # Simbad.add_votable_fields('flux(K)', 'flux(H)', 'flux(J)', 'flux(V)', 'flux(B)','otype')
                                            Simbad.add_votable_fields('otype')
                                            try:
                                                # Execute the cone search
                                                simbad_table = Simbad.query_region(coord, radius=radius_deg * u.deg)
                                            except requests.exceptions.ConnectionError:
                                                log.warning("SIMBAD aborted the reused socket. Resetting connection pool...")

                                                # Close the broken session cleanly
                                                Simbad._session.close()

                                                # Re-initialize a fresh, clean session object
                                                Simbad._session = requests.Session()

                                                # Retry the query on the fresh connection
                                                Simbad.ROW_LIMIT = -1
                                                Simbad.reset_votable_fields()
                                                # Simbad.add_votable_fields('flux(K)', 'flux(H)', 'flux(J)', 'flux(V)', 'flux(B)','otype')
                                                Simbad.add_votable_fields('otype')
                                                simbad_table = Simbad.query_region(coord, radius=radius_deg * u.deg)

                                            # Filter for sources that have at least one valid (positive/non-NaN) Filter
                                            # Note: SIMBAD uses NaN for missing flux values in Astroquery

                                            # has_K = ~np.isnan(simbad_table['FLUX_K'])
                                            # simbad_table = simbad_table[has_K]
                                            all_stellar_otypes = [
                                                '*', 'MS*', 'sg*', 'gs*', 's*r', 's*y', 's*b', 'PM*', 'HV*',
                                                'YSO', 'Pr*', 'TTau*', 'HerbigAeBe', 'OrionV*', 'BrownD*',
                                                'V*', 'IrV*', 'Pu*', 'bCepV*', 'cC*', 'aCeV*', 'delSctV*',
                                                'gamDorV*', 'RRlyrV*', 'PVTelV*', 'RVTauV*', 'alphaCygV*', 'LPV*',
                                                'MiraV*', 'SRV*', 'rcbV*', 'RotV*', 'alpha2CVnV*', 'SXAriV*',
                                                'BYDraV*', 'FKComV*', 'ellVar', 'EclV*', 'AlgolV*', 'betaLyrV*',
                                                'WUMaV*', 'Er*', 'Fl*', 'FUOriV*', 'Em*', 'Be*', 'Ae*', 'WR*',
                                                'Pe*', 'HB*', 'HotSubd*', 'C*', 'S*', 'ch*', 'Am*', 'Ap*', 'Ba*',
                                                '**', 'SB*', 'EB*', 'PMB*', 'VB*', 'AstromB*', 'WD*', 'N*', 'SN*',
                                                'Psr', 'XB', 'LMXB', 'HMXB', 'out', 'EmO', 'blu','UV','X','NearIR',
                                                'MidIR','FarIR','mmRad','Low-Mass*', 'YSO_Candidate', 'YSO', 'Star'
                                            ]
                                            # Keep only row entries that match the official SIMBAD star taxonomy
                                            simbad_table = simbad_table[np.isin(simbad_table['OTYPE'],all_stellar_otypes)]

                                            # Only use this if your columns are returned as strings (hms/dms)
                                            if np.any([isinstance(simbad_table['RA'][0], str),isinstance(simbad_table['DEC'][0], str)]):
                                                coords = SkyCoord(simbad_table['RA'], simbad_table['DEC'],
                                                                  unit=(u.hourangle, u.deg))
                                                simbad_table['RA'] = coords.ra.deg
                                                simbad_table['DEC'] = coords.dec.deg
                                            #TODO: MOCADB timing out. sqlalchemy.exc.OperationalError: (pymysql.err.OperationalError) (2003, "Can't connect to MySQL server on '104.248.106.21' (timed out)")
                                            if use_mocadb:
                                                simbad_table = query_mocadb(simbad_table)
                                            simbad_table.write(path2table, format="csv", overwrite=True)

                            def query_mocadb(table):
                                from mocapy import MocaEngine
                                # Create a moca engine object
                                moca = MocaEngine()

                                ### Change this for a list of all target names
                                simbadids = table['MAIN_ID'].tolist()

                                table['MSUN'] = np.full(len(table), '', dtype=object)
                                table['SPT'] = np.full(len(table), '', dtype=object)
                                table['J'] = np.full(len(table), '', dtype=object)
                                table['K'] = np.full(len(table), '', dtype=object)
                                table['E(B-V)'] = np.full(len(table), '', dtype=object)
                                table['E(B-V)_unc'] = np.full(len(table), '', dtype=object)
                                table['MEMBERSHIP'] = np.full(len(table), '', dtype=object)

                                for simbadid in simbadids:
                                    df2 = Table.from_pandas(moca.query(
                                        f"SELECT mechanics_all_designations.designation, summary_all_objects.moca_oid, cat_2mass.j_m, cat_2mass.k_m, summary_all_objects.spectral_type, summary_all_objects.spt_ref, data_extinction.e_bv, data_extinction.e_bv_unc, data_masses.mass_msun, calc_banyan_sigma.best_ya "
                                        f"FROM mechanics_all_designations "
                                        f"JOIN summary_all_objects ON mechanics_all_designations.moca_oid = summary_all_objects.moca_oid "
                                        f"JOIN cat_2mass ON mechanics_all_designations.moca_oid = cat_2mass.moca_oid "
                                        f"JOIN data_extinction ON mechanics_all_designations.moca_oid = data_extinction.moca_oid "
                                        f"JOIN data_masses ON mechanics_all_designations.moca_oid = data_masses.moca_oid "
                                        f"JOIN calc_banyan_sigma ON mechanics_all_designations.moca_oid = calc_banyan_sigma.moca_oid "
                                        f"WHERE mechanics_all_designations.designation = '{simbadid}'"
                                        f"LIMIT 20"
                                    ))

                                    if len(df2) > 0:
                                        table['MSUN'][table['MAIN_ID'] == simbadid] = df2['mass_msun'][0]
                                        table['SPT'][table['MAIN_ID'] == simbadid] = df2['spectral_type'][0]
                                        table['J'][table['MAIN_ID'] == simbadid] = df2['j_m'][0]
                                        table['K'][table['MAIN_ID'] == simbadid] = df2['k_m'][0]
                                        table['E(B-V)'][table['MAIN_ID'] == simbadid] = df2['e_bv'][0]
                                        table['E(B-V)_unc'][table['MAIN_ID'] == simbadid] = df2['e_bv_unc'][0]
                                        table['MEMBERSHIP'][table['MAIN_ID'] == simbadid] = df2['best_ya'][0]

                                return table

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

                            if use_gaia or use_allwise:
                                query_gaia(path2table, center_ra_deg, center_dec_deg, radius_deg, use_allwise=use_allwise)
                            elif use_simbad:
                                query_simbad(path2table, center_ra_deg, center_dec_deg, radius_deg, use_mocadb=use_mocadb)
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
                            table['roundness'] = 0.0
                            table['sharpness'] = 0.5

                            mask = (
                                    (table["x"] >= npix[0] + border)
                                    & (table["x"] <= nx - (npix[1] + border))
                                    & (table["y"] >= npix[2] + border)
                                    & (table["y"] <= ny - (npix[3] + border))
                            )
                            table_selected = table[mask]
                            table_selected['flux'] = 0.0
                            table_selected['coresat'] = 0.0
                            for _c in table_selected:
                                _cx, _cy = float(_c["x"]), float(_c["y"])
                                _xlo = int(_cx) - 31
                                _xhi = int(_cx) + 32
                                _ylo = int(_cy) - 31
                                _yhi = int(_cy) + 32
                                _patch = data[_ylo:_yhi, _xlo:_xhi]
                                _sr, _x, _y, _ecc, _sol = inspect_region_for_best_prop(_patch, margin=1,fwhm=fwhm,threshold=threshold)
                                _c['coresat'] = _sr
                                # Extract quick aperture photometry
                                positions = np.transpose((_x, _y))
                                apertures = CircularAperture(positions, r=15)
                                nansat_mask = np.isnan(_patch)
                                _patch[_patch < 0] = 0
                                phot_table = aperture_photometry(_patch, apertures, mask=nansat_mask, method='exact')
                                _c['flux'] = phot_table['aperture_sum'][0]

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
    fit_radius=np.inf,
    bkg_subtract=True,
    two_pass=True,
    showplots=False,
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
    fit_radius : float, optional
        Radius (in data pixels) defining the fitting region. If None, fits the
        full cutout.
    bkg_subtract : bool, optional
        If True, subtract a robust background estimate.
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

    struct_element = np.ones((3, 3), dtype=bool)
    dilated_mask = binary_dilation(nanmask.astype(bool), structure=struct_element)

    bkg, rms = estimate_bkg_and_rms(np.copy(data), mask=dilated_mask)
    if bkg_subtract:
        data_fit = np.copy(data) - bkg
    else:
        data_fit = np.copy(data)

    data_fit[nanmask == 1] = np.nan
    # Use the PSF as the model (with the core optionally masked).
    psf_model = FittableImageModel(psf, oversampling=oversampling)

    # Use the LevMarLSQFitter to fit the PSF to the data.
    fitter = fitting.LevMarLSQFitter()

    ny, nx = data_fit.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    center_y, center_x = ny // 2.0, nx // 2.0
    core_mask_x = (nx - 1) / 2
    core_mask_y = (ny - 1) / 2

    # Reasonable initial guesses matter a lot for position fitting.
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
    rr2 = (xx - center_x) ** 2 + (yy - center_y) ** 2
    corr = corr.copy()
    corr[(rr2 > float(fit_radius) ** 2)|(nanmask==1)] = -np.inf

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
            weights1 = _make_weights(data_fit, rms, psf_model.x_0.value, psf_model.y_0.value, core_mask_x, core_mask_y, first_pass_radius, 0)
            fit1 = fitter(psf_model, xx, yy, data_fit, weights=weights1, filter_non_finite=True)
            psf_model.x_0.value = fit1.x_0.value
            psf_model.y_0.value = fit1.y_0.value
            psf_model.flux.value = max(float(fit1.flux.value), 0.0)

        weights2 = _make_weights(data_fit, rms, psf_model.x_0.value, psf_model.y_0.value, core_mask_x, core_mask_y, float(fit_radius), 0)
        fit_result = fitter(psf_model, xx, yy, data_fit, weights=weights2, filter_non_finite=True)
    else:
        weights = _make_weights(data_fit, rms, psf_model.x_0.value, psf_model.y_0.value, core_mask_x, core_mask_y, float(fit_radius), 0)
        fit_result = fitter(psf_model, xx, yy, data_fit, weights=weights, filter_non_finite=True)

    fitted_flux = fit_result.flux.value
    fitted_x_pos = fit_result.x_0.value
    fitted_y_pos = fit_result.y_0.value

    if showplots:
        base_cmap = plt.get_cmap('gray')
        cmap = base_cmap.with_extremes(bad='red')
        zscale = ZScaleInterval(contrast=0.25, n_samples=600)
        vmin, vmax = zscale.get_limits(data_fit)
        plt.imshow(data_fit, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        plt.plot(fitted_x_pos, fitted_y_pos, 'xb')
        plt.colorbar()
        plt.show()
        pass

    return fitted_x_pos,fitted_y_pos,fitted_flux

def inspect_region(nandata, labeled_mask, props, best_prop=None, x_cent=None, y_cent=None, id=None):
    with plt.style.context('spaceKLIP.sk_style'):
        """Plots only the colored mask overlay with concise ID, Flux, Roundness, and Solidity metrics."""
        plt.figure(figsize=(8, 8))

        # Clean data for display (replace NaNs and negatives with median)
        display_img = nandata.copy()
        clean_bg = np.nanmedian(np.where(display_img < 0, np.nan, display_img))
        display_img[~np.isfinite(display_img) | (display_img < 0)] = clean_bg

        # Normalize image for label2rgb blending
        img_min, img_max = display_img.min(), display_img.max()
        if img_max > img_min:
            norm_img = (display_img - img_min) / (img_max - img_min)
        else:
            norm_img = np.zeros_like(display_img)

        # CHANGE: Set bg_color to a mid-gray tuple (R, G, B) so it is highly visible
        overlay = label2rgb(
            labeled_mask,
            image=norm_img,
            bg_label=0,
            bg_color=(0.95, 0.95, 0.95),
            alpha=0.3,
        )
        plt.imshow(overlay, origin="lower")
        plt.title(f"Detected Regions Overlay for id {id}")

        print(f"\n--- INSPECTING {len(props)} REGIONS FOR ID {id}---")
        for prop in props:
            # Cast tracking properties safely
            label_id = prop.label
            is_winner = best_prop and (label_id == best_prop.label)
            status = "[WINNER]" if is_winner else ""

            # Calculate metrics
            solidity = float(prop.custom_solidity)
            brightness_factor = float(prop.brightness_factor)
            area_factor = float(prop.area_factor)
            boxy_factor = float(prop.boxy_factor)
            distance_factor = float(prop.distance_factor)
            score = float(prop.score)

            # Print detailed stats to console including photometry metrics used in decision
            print(
                f"Label {label_id:2d}: "
                f"Solidity={solidity:.2f} | "
                f"Area Factor={area_factor:.2f} | "
                f"Boxy Factor={boxy_factor:.2f} | "
                f"Distance Factor={distance_factor:.2f} | "
                f"Brightness={brightness_factor:.2f} | "
                f"Score={score:.2f} | "
                f"Centroid=({prop.centroid[1]:.1f}, {prop.centroid[0]:.1f}) {status}"
            )

            # Draw bounding boxes (FIXED: Added -0.5 offset for perfect pixel boundary alignment)
            minr, minc, maxr, maxc = prop.bbox
            rect = plt.Rectangle(
                (minc - 0.5, minr - 0.5),   # Shift anchor to the true bottom-left pixel edge
                maxc - minc,
                maxr - minr,
                fill=False,
                edgecolor="red" if is_winner else "cyan",
                linewidth=2.5 if is_winner else 1.5,
            )
            plt.gca().add_patch(rect)
            plt.plot(x_cent, y_cent, "xk", ms=7)
            # Create a clean metadata label string using original layout names
            label_text = (
                f"ID:{label_id}\n"
                f"Sc:{score:.2f}"
            )

            # Position the text neatly above or to the side of the box
            plt.text(
                maxc + 1,
                minr,
                label_text,
                color="yellow",
                fontsize=9,
                weight="bold",
                bbox=dict(facecolor="black", alpha=0.6, boxstyle="round,pad=0.2"),
            )

        plt.tight_layout()
        plt.show()
        pass

def inspect_region_for_best_prop(data, center=None, margin=1, nanmask=None, fwhm=2.5, threshold=1.5, peak_fraction=0.33,
                                 debug=False, id=None,r_max=np.inf):
    """
    Inspect a region looking for different props, identify the best one and return it's properties.

    data: 2D array of image data to analyze
    center: (x, y) tuple for the center of the region to analyze; if None, the function will find the brightest pixel in the data
    margin: number of pixels to include around the detected region for analysis
    nanmask: optional 2D boolean array of the same shape as data, where True indicates pixels to ignore (e.g., NaNs or masked regions)
    debug: if True, will make a plot with additional debug information
    r_max: maximum distance from center to accept a candidate. Default is np.inf.

    Returns:
    A list of properties for the best prop in the region

    """
    nandata = np.array(data, dtype=float)
    if nanmask is not None:
        nandata[nanmask.astype(bool)] = np.nan

    ny, nx = nandata.shape
    default_cx, default_cy = center if center is not None else ((nx - 1) / 2, (ny - 1) / 2)

    # Clean background estimation (ignoring negative border pixels)
    img_background = np.nanmedian(np.where(nandata <= 0, np.nan, nandata))

    # Calculate a rough noise estimate to find bright stars
    bright_star_thresh = threshold * img_background

    # 1. Isolate and label ONLY the NaN/Infinite cores and within some r_max from the center
    ny, nx = nandata.shape
    center_y, center_x = ny//2.0, nx// 2.0
    y, x = np.ogrid[:ny, :nx]
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    nan_mask = ~np.isfinite(nandata)&(r<=r_max)
    labeled_nan_mask = label(nan_mask)
    num_nan_cores = np.max(labeled_nan_mask)

    # 2. Isolate and label ONLY the pure bright pixels (excluding the NaNs)
    bright_mask = (nandata > bright_star_thresh) & (~nan_mask)
    labeled_bright_mask = label(bright_mask)

    # 3. Shift the bright pixel IDs up so they do not conflict with NaN IDs
    if num_nan_cores > 0:
        labeled_bright_mask[labeled_bright_mask > 0] += num_nan_cores

    # 4. Merge them into a single master mask where touching regions keep distinct IDs
    labeled_mask = np.where(labeled_nan_mask > 0, labeled_nan_mask, labeled_bright_mask)

    orig_mask = labeled_mask
    labeled_mask = np.zeros_like(orig_mask)
    new_label = 1
    for prop in regionprops(orig_mask):
        if prop.area > 1:
            labeled_mask[orig_mask == prop.label] = new_label
            new_label += 1

    if not np.any(labeled_mask):
        return 0, float(default_cx), float(default_cy), 0, 1

    # Now props contains completely unmixed regions
    props = regionprops(labeled_mask)

    if not props:
        return 0, float(default_cx), float(default_cy), 0, 1

    best_prop = None
    best_score = -float('inf')
    yy, xx = np.indices(nandata.shape)

    for prop in props:
        single_cluster_mask = (labeled_mask == prop.label)

        # --- Extract sub-masks within this specific region ---
        region_nans = single_cluster_mask & (~np.isfinite(nandata))
        region_bright = single_cluster_mask & (nandata > bright_star_thresh)

        # Decide solidity and eccentricity strategy based on presence of NaNs
        if np.any(region_nans):
            # Evaluate solidity ONLY on the NaN core cluster
            core_label = label(region_nans)
            core_props = regionprops(core_label)
            # Take the largest NaN cluster inside this prop if multiple exist
            solidity_score = max([p.solidity for p in core_props]) if core_props else 0.0
            eccentricity_score = float(prop.eccentricity)  # Saturated cores use full region bounds

            dilated = binary_dilation(region_nans, iterations=5)
            perimeter_mask = dilated & (nandata > 0)
            perimeter_data = nandata[perimeter_mask]
            avg_perimeter_brightness = np.nansum(perimeter_data - img_background)
            brightness_factor = max(0.01, avg_perimeter_brightness)
            # Saturated core: use the geometric center of the mask
            x_cent = float(np.mean(xx[core_label.astype(bool)]))
            y_cent = float(np.mean(yy[core_label.astype(bool)]))
            rr = np.sqrt((xx - x_cent) ** 2 + (yy - y_cent) ** 2)
            radius = int(np.ceil(np.max(rr[core_label.astype(bool)])) + int(margin))
            fwhm_temp = max(fwhm, min(radius, 10))
            minr, minc, maxr, maxc = prop.bbox
        else:
            # --- Dynamic peak thresholding to strip unknown nebulosity ---
            region_intensities = nandata[region_bright]

            # Find peak intensity and local cloud median background
            peak_flux = np.nanmax(region_intensities[region_intensities>0]) if len(region_intensities) > 0 else bright_star_thresh
            cloud_median = np.nanmedian(region_intensities[region_intensities>0]) if len(region_intensities) > 0 else bright_star_thresh

            # Dynamically slice based on the configurable peak_fraction parameter
            dynamic_thresh = cloud_median + peak_fraction * (peak_flux - cloud_median)

            # Re-mask this region using the localized high-contrast threshold
            clean_star_mask = region_bright & (nandata >= dynamic_thresh)

            # Evaluate solidity and geometry on this dynamically isolated peak
            star_label = label(clean_star_mask)
            star_props = regionprops(star_label)

            # Target the largest distinct peak structure found inside the cloud
            best_star_subprop = max(star_props, key=lambda p: p.area) if star_props else None

            if best_star_subprop is not None:
                # FIX: Evaluate metrics safely on the reduced area to isolate from nebulosity
                solidity_score = float(best_star_subprop.solidity)
                eccentricity_score = float(best_star_subprop.eccentricity)
                # Override bounding box variables so dx and dy measure only the clean peak
                minr, minc, maxr, maxc = best_star_subprop.bbox
                # Use clean stellar peak area for scoring metrics
                perimeter_data = nandata[clean_star_mask]
            else:
                solidity_score = 0.0
                eccentricity_score = float(prop.eccentricity)
                minr, minc, maxr, maxc = prop.bbox
                perimeter_data = nandata[region_bright]

            avg_perimeter_brightness = np.nansum(perimeter_data)  # -img_background)
            brightness_factor = max(0.01, avg_perimeter_brightness)
            fwhm_temp = fwhm

        # Extract bounding box dimensions (overridden above for unsaturated stars in nebulosity)
        dx = float(maxc - minc)
        dy = float(maxr - minr)

        # Target linear dimension (diameter) based on your 1.5x FWHM radius profile
        target_dim = 3.0 * float(fwhm_temp)
        # Added a wider, gentler standard deviation slope to prevent too quick decay to 0
        # dim_sigma = target_dim * 1.2
        dim_sigma = target_dim / 1.2

        # Evaluate independent Gaussian profiles for both X and Y dimensions
        gaussian_dx = ((dx - target_dim) / dim_sigma) ** 2
        gaussian_dy = ((dy - target_dim) / dim_sigma) ** 2

        # Combine them into a joint spatial scale factor (peaks at 1.0)
        area_factor = max(1e-5, np.exp(-0.5 * (gaussian_dx + gaussian_dy)))

        # Calculate how close to a square the region is, strictly bounded between 0 and 1
        boxy_factor = min(dx, dy) / max(1e-5, max(dx, dy))

        # Define independent spatial scales for a rectangular frame
        sigma_nx = float(nx) / 5.0
        sigma_ny = float(ny) / 5.0

        # Extract centroid coordinates (prop.centroid is ordered as (y, x))
        ry, rx = prop.centroid

        # Evaluate independent 2D Gaussian decay for rectangular geometry
        dx_norm = ((rx - default_cx) / sigma_nx) ** 2
        dy_norm = ((ry - default_cy) / sigma_ny) ** 2

        # Distance factor: 1.0 at center, smoothly dropping toward 0.0 at the edges
        distance_factor = np.exp(-0.5 * (dx_norm + dy_norm))

        # Total Score now uses the dynamically calculated parameters
        total_score = solidity_score * brightness_factor * area_factor * boxy_factor * distance_factor

        prop.custom_solidity = solidity_score
        prop.custom_eccentricity = eccentricity_score
        prop.brightness_factor = brightness_factor
        prop.area_factor = area_factor
        prop.boxy_factor = boxy_factor
        prop.distance_factor = distance_factor
        prop.score = total_score

        r = np.sqrt((rx - center_x) ** 2 + (ry - center_y) ** 2)

        if total_score > best_score and r<=r_max:
            best_score = total_score
            best_prop = prop

    if best_score <= 0.0:
        return 0, float(default_cx), float(default_cy), 0, 1

    winning_region = (labeled_mask == best_prop.label)

    # Check if the winning region corresponds to a saturated NaN core
    if np.any(np.isnan(nandata[winning_region])):
        # Saturated core: use the geometric center of the mask
        x_cent = float(np.mean(xx[winning_region]))
        y_cent = float(np.mean(yy[winning_region]))
        rr = np.sqrt((xx - x_cent) ** 2 + (yy - y_cent) ** 2)
        radius = int(np.ceil(np.max(rr[winning_region & np.isnan(nandata)])) + int(margin))
    else:
        # Unsaturated star: isolate region pixels and locate the peak flux pixel
        region_data = np.where(winning_region, nandata, -np.inf)
        y_peak, x_peak = np.unravel_index(np.argmax(region_data), region_data.shape)
        x_cent = float(x_peak)
        y_cent = float(y_peak)
        radius = 0

    if debug:
        inspect_region(nandata, labeled_mask, props, best_prop, x_cent, y_cent, id)

    return radius, x_cent, y_cent, best_prop.custom_eccentricity, best_prop.custom_solidity

def stars_extractor(data,
                    coords,
                    fov = 101,
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
        tile = data[int(round(coords[1]))-fov//2:int(round(coords[1]))+fov//2+1, int(round(coords[0]))-fov//2:int(round(coords[0]))+fov//2+1]
    else:
        #Create a bigger tile to shift, so we don't have to shift the entire image to minimize weird artifacts
        preshifttile = data[int(round(coords[1]))-(fov//2+pad_amount):int(round(coords[1]))+(fov//2+pad_amount+1),
                            int(round(coords[0]))-(fov//2+pad_amount):int(round(coords[0]))+(fov//2+pad_amount+1)]
        shifteddata = ut.imshift(preshifttile, [shifts[0], shifts[1]], pad_amount=0, method=method, kwargs=kwargs)
        #Crop the shifted tile to the desired dimension
        tile = shifteddata[int(round(shifteddata.shape[1]//2))-fov//2:int(round(shifteddata.shape[1]//2))+fov//2+1, int(round(shifteddata.shape[0]//2))-fov//2:int(round(shifteddata.shape[0]//2))+fov//2+1]
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


def extract_image_centers(hdul_list):
    """Reads headers of all matching FITS files and extracts center coordinates."""
    ra_centers = []
    dec_centers = []

    if not file_paths:
        raise FileNotFoundError(f"No FITS files found in {fits_directory}")

    for hdul in hdul_list:
        # 1. Parse the World Coordinate System from the primary or image header
        # Note: spaceKLIP / JWST data might store this in extension 1 ('SCI')
        header = hdul[0].header if 'NAXIS' in hdul[0].header else hdul[1].header
        wcs = WCS(header)

        # 2. Find the pixel dimensions of the detector array
        naxis1 = header.get('NAXIS1', 0)
        naxis2 = header.get('NAXIS2', 0)

        if naxis1 == 0 or naxis2 == 0:
            continue  # Skip files without explicit spatial footprints

        # 3. Compute the exact pixel center of this frame
        center_x = naxis1 / 2.0
        center_y = naxis2 / 2.0

        # 4. Transform pixel center to world coordinate (SkyCoord)
        center_sky = wcs.pixel_to_world(center_x, center_y)

        ra_centers.append(center_sky.ra.deg)
        dec_centers.append(center_sky.dec.deg)

    # 5. Compile everything into a single vectorized SkyCoord Array
    return SkyCoord(ra=ra_centers, dec=dec_centers, unit=(u.deg, u.deg))

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
                catalog=None,
                nan_lim_percent=0.75,
                two_pass=True,
                showplots=False,
                psf=None,
                fov=31
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
        self.catalog=catalog
        self.nan_lim_percent=nan_lim_percent
        self.two_pass = two_pass
        self.showplots = showplots
        self.psf=psf
        self.fov=fov
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

        # tbl = dao(np.nan_to_num(data, nan=0.0), mask=mask)
        tbl = dao(data, mask=mask)
        if tbl is None or len(tbl) == 0:
            return None
        # out = []
        tbl['coresat'] = 0
        tbl['eccsat'] = 0.0
        tbl['solsat'] = 1.0
        tbl['method'] = 'dao'
        tbl.rename_column('xcentroid', 'x')
        tbl.rename_column('ycentroid', 'y')
        tbl.rename_column('roundness1', 'roundness')

        mask1 = (
                (tbl["x"] >= self.npix[0] + border)
                & (tbl["x"] <= nx - (self.npix[1] + border))
                & (tbl["y"] >= self.npix[2] + border)
                & (tbl["y"] <= ny - (self.npix[3] + border))
                & (tbl["roundness"] >= self.roundness_range[0])
                & (tbl["roundness"] <= self.roundness_range[1])
                & (tbl["sharpness"] >= self.sharpness_range[0])
                & (tbl["sharpness"] <= self.sharpness_range[1])
        )
        #TODO: figure out how to set up proper mask for saturated sources. For now using the same for both.
        mask2 = (
                (tbl["x"] >= self.npix[0] + border)
                & (tbl["x"] <= nx - (self.npix[1] + border))
                & (tbl["y"] >= self.npix[2] + border)
                & (tbl["y"] <= ny - (self.npix[3] + border))
                & (tbl["roundness"] >= -0.5)
                & (tbl["roundness"] <= 0.5)
                & (tbl["sharpness"] >= self.sharpness_range[0])
                & (tbl["sharpness"] <= self.sharpness_range[1])
        )
        is_coresat = tbl["coresat"] > 0
        final_mask = np.where(is_coresat, mask1, mask1)

        tbl_selected = tbl[final_mask]['x','y','peak','coresat','eccsat','solsat','method','sharpness','roundness']
        tbl_selected['flux'] = 0.0
        for _c in tbl_selected:
            _cx, _cy = float(_c["x"]), float(_c["y"])
            _xlo = int(_cx) - 31
            _xhi = int(_cx) + 32
            _ylo = int(_cy) - 31
            _yhi = int(_cy) + 32
            _patch = data[_ylo:_yhi, _xlo:_xhi]
            _sr, _x, _y, _ecc, _sol = inspect_region_for_best_prop(_patch, margin=1,fwhm=self.fwhm,threshold=self.threshold)
            _c['coresat'] = _sr
            # Extract quick aperture photometry
            positions = np.transpose((_x, _y))
            apertures = CircularAperture(positions, r=15)
            nansat_mask = np.isnan(_patch)
            _patch[_patch < 0] = 0
            phot_table = aperture_photometry(_patch, apertures, mask=nansat_mask, method='exact')
            _c['flux'] = phot_table['aperture_sum'][0]

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
        tbl['eccsat'] = 0.0
        tbl['solsat'] = 1.0
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
        tbl_selected = tbl[mask_indices]['x', 'y', 'peak', 'coresat', 'eccsat', 'solsat', 'method', 'sharpness', 'roundness']

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

    def _group_catalog(self, catalog, ideal_roundness=0.0, ideal_sharpness=0.5):
        """
        Groups catalog sources within a box_size and selects the best stellar representative.

        Parameters:
        catalog (astropy.table.Table): Must contain columns 'x', 'y', 'flux', 'roundness', 'sharpness'
        box_size (float): The maximum distance to group sources.
        """
        # 1. Calculate a custom "star score" (Higher is better)
        # Using np.log1p for flux to balance its weight against shape metrics
        roundness_penalty = np.abs(catalog['roundness'] - ideal_roundness)
        sharpness_penalty = np.abs(catalog['sharpness'] - ideal_sharpness)
        star_score = np.log1p(catalog['flux']) - roundness_penalty - sharpness_penalty

        # 2. Build KDTree for fast spatial queries
        coords = np.vstack((catalog['x'], catalog['y'])).T
        tree = KDTree(coords)

        # 3. Sort indices by star_score from best to worst
        sorted_indices = np.argsort(star_score)[::-1]

        # Track visited points using a fast boolean mask
        num_entries = len(catalog)
        visited = np.zeros(num_entries, dtype=bool)
        keep_indices = []

        # 4. Greedy elimination loop
        for idx in sorted_indices:
            if visited[idx]:
                continue

            # Keep this source as the group representative
            keep_indices.append(idx)

            # Find all neighbors within the box_size
            neighbors = tree.query_ball_point(coords[idx], r=self.fov / 2.0, p=np.inf)
            # Mark the representative and all its neighbors as visited
            visited[neighbors] = True


        # 5. Return the filtered Astropy Table (sorted by original order)
        keep_indices = sorted(keep_indices)
        log.info(f"Using {np.sum([i['method']=='catalog' for i in catalog[keep_indices]])} catalog seeds + {np.sum([i['method']!='catalog' for i in catalog[keep_indices]])} DAO detections after selections.")
        catalog=catalog[keep_indices]
        catalog['id']=[i for i in range(len(catalog))]
        return catalog

    def _clean_catalog(self,candidates, data_arr):
        """Group nearby candidates and select one representative per star.

        DAOStarFinder often returns several detections for a single bright or
        saturated star — one near the core and several on the PSF wings.

        Parameters
        ----------
        candidates : list of dict
            Full (ungrouped) candidate list from ``_dao``.
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

        data_temp =np.copy(data_arr)
        # left, right, bottom, top
        # 1. Top border: rows from 0 up to X
        data_temp[:self.npix[3], :] = -1
        # 2. Bottom border: rows from the bottom up to X
        data_temp[-self.npix[2]:, :] = -1
        # 3. Left border: columns from 0 up to X
        data_temp[:, :self.npix[0]] = -1
        # 4. Right border: columns from the right up to X
        data_temp[:, -self.npix[1]:] = -1
        # Pre-compute a provisional NaN-core radius for every candidate so
        # that _candidate_radius can scale the grouping window correctly even
        # before the formal coresat is estimated inside _clean_catalog.
        _prov_sat = []
        _rmax = []
        _keep_mask = []
        # candidates['flux'] = 0.0
        for _c in candidates:
            _cx, _cy = float(_c["x"]), float(_c["y"])
            _xlo = int(_cx) - self.fov//2
            _xhi = int(_cx) + self.fov//2+1
            _ylo = int(_cy) - self.fov//2
            _yhi = int(_cy) + self.fov//2+1
            _patch = data_temp[_ylo:_yhi, _xlo:_xhi]
            # if _c['id'] in [75,79,87]:
            #     #for debugging purposes
            #     _sr, _x, _y, _ecc, _sol = inspect_region_for_best_prop(_patch, margin=1, fwhm=self.fwhm, threshold=self.threshold,debug=True,id=_c['id'])
            # else:
            _sr, _x, _y, _ecc, _sol = inspect_region_for_best_prop(_patch, margin=1,fwhm=self.fwhm, threshold=self.threshold)
            if _sr > 0:
                if np.sum(~np.isfinite(_patch)) <= np.ceil(_patch.shape[0] * _patch.shape[1] * self.nan_lim_percent): #_ecc >=0.7 and _sol>=0.75 and
                    _c['x'] = _x+_xlo
                    _c['y'] = _y+_ylo
                    _c['eccsat'] = _ecc
                    _c['solsat'] = _sol
                else:
                    # if _c['id'] in [19,74,71,75,79,87,111]:
                    #     pass
                    _keep_mask.append(False)
                    continue
            else:
                _c['x'] = _x+_xlo
                _c['y'] = _y+_ylo
                _c['eccsat'] = _ecc
                _c['solsat'] = _sol
            _c['coresat'] = _sr
            # Extract quick aperture photometry
            positions = np.transpose((_x, _y))
            apertures = CircularAperture(positions, r=15)
            nansat_mask = np.isnan(_patch)
            _patch[_patch<0] = 0
            phot_table = aperture_photometry(_patch, apertures, mask=nansat_mask, method='exact')
            _c['flux'] = phot_table['aperture_sum'][0]
            _prov_sat.append(float(_sr))
            _rmax.append(float(max(_patch.shape)))
            _keep_mask.append(True)

        candidates = candidates[_keep_mask]
        candidates['id']=[i for i in range(len(candidates))]
        candidates = self._group_catalog(candidates)
        return candidates

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
            sat_r, _, _, _, _ = inspect_region_for_best_prop(cut, center=(x_fit - xlo, y_fit - ylo), margin=1, fwhm=self.fwhm, threshold=self.threshold)

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
                    fwhm=self.fwhm,
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
        use_dao=True,
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
          ``inspect_region_for_best_prop``.
        * Otherwise the source is re-centered on the strongest group-wide
          PSF-correlation peak, with the highest ``DAOStarFinder`` peak used as a
          fallback.

        Parameters
        ----------
        data : 2D-array
            science image.
        nanmask: 2D-array (bool)
            Mask of NaN pixels  used to identify candidates with saturated cores or candidates too close to the edge.
        use_dao : bool, optional
            Whether to run DAOStarFinder or StarFinder. If False, run StarFinder. Default is True.
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
        data_subtracted[data_subtracted<0]=0
        #Candidate detection via DAOStarFinder, or StarFinder
        if use_dao:
            if dao:
                dao_catalog = self._dao(data_subtracted,mask=dilated_mask,mrms=np.nanmedian(rms))
            else:
                dao_catalog = self._starfinder(data,mask=dilated_mask,mrms=np.nanmedian(rms))
        else:
            dao_catalog = None

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
        all_candidates['id']=[int(i) for i in range(len(all_candidates))]
        selected_candidates = self._clean_catalog(all_candidates, data_subtracted)
        # selected_candidates = all_candidates

        return selected_candidates