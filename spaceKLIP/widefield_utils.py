import logging,os, requests

import numpy as np

from spaceKLIP import utils as ut
from spaceKLIP.plotting import load_plt_style

from photutils.detection import DAOStarFinder,StarFinder
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.background import Background2D, MedianBackground

from scipy.ndimage import binary_fill_holes, distance_transform_edt, center_of_mass, binary_dilation
from scipy.optimize import minimize
from scipy.spatial import KDTree

import astropy.io.fits as pyfits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.stats import SigmaClip
from astropy.visualization import simple_norm
from astropy.table import Table, vstack
from astropy.wcs import WCS

from skimage.color import label2rgb
from skimage.measure import label, regionprops

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from typing import Literal

from pathlib import Path

# Set up log.
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

def mask_within_radius(image, xdat, ydat, xcen, ycen, r, x=0, y=0, c=np.nan, partial=False):
    """
       Mask pixels within a specified radius of a center coordinate.

       Parameters
       ----------
       image : numpy.ndarray
           The input image to be masked.
       xdat : numpy.ndarray
           X-coordinate grid for the image.
       ydat : numpy.ndarray
           Y-coordinate grid for the image.
       xcen : float
           X-coordinate of the center in the local frame.
       ycen : float
           Y-coordinate of the center in the local frame.
       r : float
           Radius of the mask.
       x : float, optional
           X-offset for the center. Default is 0.
       y : float, optional
           Y-offset for the center. Default is 0.
       c : float, optional
           The value to fill the masked area with. Default is np.nan.
       partial : bool, optional
           If True, increases the effective radius to include any pixel touched by the circle.
           Default is False.

       Returns
       -------
       numpy.ndarray
           The masked image.
    """

    distance = np.sqrt((xdat - (x + xcen)) ** 2 + (ydat - (y + ycen)) ** 2)

    if partial:
        # Expand radius by the distance from pixel center to pixel corner (sqrt(0.5))
        # This ensures any pixel touched by the circle is included
        effective_radius = r + 0.7071
    else:
        # Standard mask based strictly on pixel centers
        effective_radius = r

    image[distance <= effective_radius] = c
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
                                            # if use_mocadb:
                                            #     simbad_table = query_mocadb(simbad_table)
                                            simbad_table.write(path2table, format="csv", overwrite=True)

                            # def query_mocadb(table):
                            #     """
                            #        Query the MOCADB database for additional stellar metadata.
                            #
                            #        Parameters
                            #        ----------
                            #        table : astropy.table.Table
                            #            Input table containing source names in the 'MAIN_ID' column.
                            #
                            #        Returns
                            #        -------
                            #        astropy.table.Table
                            #            Input table updated with 'MSUN', 'SPT', 'J', 'K', 'E(B-V)', and 'MEMBERSHIP' data.
                            #    """
                            #
                            #     from mocapy import MocaEngine
                            #     # Create a moca engine object
                            #     moca = MocaEngine()
                            #
                            #     ### Change this for a list of all target names
                            #     simbadids = table['MAIN_ID'].tolist()
                            #
                            #     table['MSUN'] = np.full(len(table), '', dtype=object)
                            #     table['SPT'] = np.full(len(table), '', dtype=object)
                            #     table['J'] = np.full(len(table), '', dtype=object)
                            #     table['K'] = np.full(len(table), '', dtype=object)
                            #     table['E(B-V)'] = np.full(len(table), '', dtype=object)
                            #     table['E(B-V)_unc'] = np.full(len(table), '', dtype=object)
                            #     table['MEMBERSHIP'] = np.full(len(table), '', dtype=object)
                            #
                            #     for simbadid in simbadids:
                            #         df2 = Table.from_pandas(moca.query(
                            #             f"SELECT mechanics_all_designations.designation, summary_all_objects.moca_oid, cat_2mass.j_m, cat_2mass.k_m, summary_all_objects.spectral_type, summary_all_objects.spt_ref, data_extinction.e_bv, data_extinction.e_bv_unc, data_masses.mass_msun, calc_banyan_sigma.best_ya "
                            #             f"FROM mechanics_all_designations "
                            #             f"JOIN summary_all_objects ON mechanics_all_designations.moca_oid = summary_all_objects.moca_oid "
                            #             f"JOIN cat_2mass ON mechanics_all_designations.moca_oid = cat_2mass.moca_oid "
                            #             f"JOIN data_extinction ON mechanics_all_designations.moca_oid = data_extinction.moca_oid "
                            #             f"JOIN data_masses ON mechanics_all_designations.moca_oid = data_masses.moca_oid "
                            #             f"JOIN calc_banyan_sigma ON mechanics_all_designations.moca_oid = calc_banyan_sigma.moca_oid "
                            #             f"WHERE mechanics_all_designations.designation = '{simbadid}'"
                            #             f"LIMIT 20"
                            #         ))
                            #
                            #         if len(df2) > 0:
                            #             table['MSUN'][table['MAIN_ID'] == simbadid] = df2['mass_msun'][0]
                            #             table['SPT'][table['MAIN_ID'] == simbadid] = df2['spectral_type'][0]
                            #             table['J'][table['MAIN_ID'] == simbadid] = df2['j_m'][0]
                            #             table['K'][table['MAIN_ID'] == simbadid] = df2['k_m'][0]
                            #             table['E(B-V)'][table['MAIN_ID'] == simbadid] = df2['e_bv'][0]
                            #             table['E(B-V)_unc'][table['MAIN_ID'] == simbadid] = df2['e_bv_unc'][0]
                            #             table['MEMBERSHIP'][table['MAIN_ID'] == simbadid] = df2['best_ya'][0]
                            #
                            #     return table

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
                            x_center = (nx-1) / 2.0
                            y_center = (ny-1) / 2.0

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


def estimate_bkg_and_rms(data,mask,n=25):
    """Estimate spatial background and RMS maps using a 2D background estimator.

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
        box_size=max(int(np.floor(np.min(data.shape) / np.sqrt(n))), 5),
        filter_size=(3, 3),
        sigma_clip=sigma_clip,
        bkg_estimator=bkg_estimator,
        exclude_percentile=50.0
    )
    return bkg.background, bkg.background_rms

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

        log.debug(f"\n--- INSPECTING {len(props)} REGIONS FOR ID {id}---")
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
            log.debug(
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
            brightness_factor = max(0.005, avg_perimeter_brightness)
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
            brightness_factor = max(0.005, avg_perimeter_brightness)
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

        # Extract centroid/peak coordinates (prop.centroid is ordered as (y, x))
        if np.any(region_nans):
            ry, rx = prop.centroid
        else:
            datamasked = np.copy(nandata)
            datamasked[~clean_star_mask] = 0
            ry, rx = np.unravel_index(np.argmax(datamasked), datamasked.shape)

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
                    fov=101,
                    pad_amount=0,
                    shifts=None,
                    method='fourier',
                    showplot=False,
                    kwargs={}
                    ):
    """
    Extract a sub-image (tile) centered on specific coordinates, optionally applying a sub-pixel shift.

    If no shifts are provided, it automatically calculates the sub-pixel shift required to
    bring the floating-point 'coords' to the exact center of the tile.
    If the shift is zero, it avoids interpolation to prevent artifacts.
    """

    # 1. Determine the integer pixel center for the crop
    x_f, y_f = float(coords[0]), float(coords[1])
    x_i, y_i = int(round(x_f)), int(round(y_f))

    # 2. Determine the sub-pixel shift
    if shifts is None:
        # Calculate shift required to move the star from its float position
        # to the center of the integer-pixel crop.
        # Example: star at 100.2, crop at 100. Shift needed: 100 - 100.2 = -0.2
        dx = x_i - x_f
        dy = y_i - y_f
    else:
        dx = (x_i - x_f) + shifts[0]
        dy = (y_i - y_f) + shifts[1]
        # dx, dy = shifts[0], shifts[1]

    # 3. Check if the shift is effectively zero
    is_zero_shift = (abs(dx) < 1e-6) and (abs(dy) < 1e-6)

    if is_zero_shift:
        # NO SHIFT: Perform a direct crop to avoid interpolation artifacts
        y_start, y_end = y_i - fov // 2, y_i + fov // 2 + 1
        x_start, x_end = x_i - fov // 2, x_i + fov // 2 + 1

        # Guard against edge of frame indexing
        tile = data[max(0, y_start):y_end, max(0, x_start):x_end]

        if tile.shape != (fov, fov):
            tile = np.pad(tile,
                          ((max(0, -y_start), max(0, y_end - data.shape[0])),
                           (max(0, -x_start), max(0, x_end - data.shape[1]))),
                          mode='constant', constant_values=0)
    else:
        # SHIFT REQUIRED: Pad, shift, and then crop
        # Extract a slightly larger tile to accommodate padding for shifting
        total_pad = fov // 2 + pad_amount
        y_low, y_high = y_i - total_pad, y_i + total_pad + 1
        x_low, x_high = x_i - total_pad, x_i + total_pad + 1

        preshifttile = data[max(0, y_low):y_high, max(0, x_low):x_high]

        # Apply padding if crop was near the detector edge
        preshifttile = np.pad(preshifttile,
                              ((max(0, -y_low), max(0, y_high - data.shape[0])),
                               (max(0, -x_low), max(0, x_high - data.shape[1]))), mode='constant', constant_values=0)
                              # mode='reflect')

        # Apply the sub-pixel shift using spaceKLIP utility
        # Note: ut.imshift takes [dx, dy]
        shifteddata = ut.imshift(preshifttile, [dx, dy], pad_amount=0, method=method, kwargs=kwargs, nan_reflected=False)

        # Crop the shifted tile back to the desired FOV
        # The star is now centered in shifteddata
        c_y, c_x = shifteddata.shape[0] // 2, shifteddata.shape[1] // 2
        tile = shifteddata[c_y - fov // 2: c_y + fov // 2 + 1,
        c_x - fov // 2: c_x + fov // 2 + 1]

    if showplot:
        load_plt_style(None)
        # 1. Safely extract the data min and max bounds
        tile_min = float(np.nanmin(tile))
        tile_max = float(np.nanmax(tile))

        # 2. FIX: Dynamically safeguard the normalization limits
        # If the image is completely flat or invalid, provide a safe default window
        if tile_max <= tile_min:
            vmin, vmax = -1.0, 1.0
        else:
            vmin = tile_min
            # Try your preferred 20% scaling threshold
            vmax_trial = tile_max * 0.2

            # If the scaled vmax falls below or equal to vmin (due to negative values),
            # fall back to a safe upper bound (e.g., halfway between min and max)
            if vmax_trial <= vmin:
                vmax = vmin + (tile_max - vmin) * 0.5
            else:
                vmax = vmax_trial

        # Double check to guarantee absolute safety before passing to Matplotlib
        if vmin >= vmax:
            vmax = vmin + 1.0

        fig, ax = plt.subplots(figsize=(5, 5))

        # Pass explicit, fully verified parameters to the renderer
        ax.imshow(tile, origin='lower', vmin=vmin, vmax=vmax, cmap='viridis')
        plt.plot(tile.shape[1] // 2, tile.shape[0] // 2, 'xr', label='Target Center')
        ax.legend()
        plt.show()
    return tile

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

def write_obs(fitsfile,
              output_dir,
              data,
              erro,
              pxdq,
              head_pri,
              head_sci,
              is2d,
              new_fitsfile=None):
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
    new_fitsfile : path, None
        If None, path to the new FITS file to save.
    Returns
    -------
    fitsfile : path
        Path of output FITS file.
    """

    # Write FITS file.
    hdul = pyfits.open(fitsfile)
    for ext_name in [hdu.name for hdu in hdul]:
        # Check if the name should be dropped
        if np.all([i not in ext_name for i in ['PRIMARY', 'SCI', 'ERR', 'DQ']]):
            hdul.pop(ext_name)

    if is2d:
        hdul['SCI'].data = data[0]
        hdul['ERR'].data = erro[0]
        hdul['DQ'].data = pxdq[0]
    else:
        hdul['SCI'].data = data
        hdul['ERR'].data = erro
        hdul['DQ'].data = pxdq
    if new_fitsfile is None:
        fitsfile = os.path.join(output_dir, os.path.split(fitsfile)[1])
    else:
        fitsfile = os.path.join(output_dir, os.path.split(new_fitsfile)[1])

    if isinstance(head_pri, (list,np.ndarray)):
        hdul[0].header = head_pri[0]
    else:
        hdul[0].header = head_pri
    if isinstance(head_sci, (list,np.ndarray)):
        hdul['SCI'].header = head_sci[0]
    else:
        hdul['SCI'].header = head_sci

    if isinstance(head_pri, (list,np.ndarray)):
        for n,header in enumerate(head_sci[1:]):
            new_pri_hdu = pyfits.ImageHDU(data=np.ones((1, 1), dtype=np.float32), header=header, name=f'PRIMARY_{n}')
            hdul.append(new_pri_hdu)
    if isinstance(head_sci, (list,np.ndarray)):
        for n,header in enumerate(head_sci[1:]):
            new_sci_hdu = pyfits.ImageHDU(data=np.ones((1, 1), dtype=np.float32), header=header, name=f'SCI_{n}')
            hdul.append(new_sci_hdu)

    hdul.writeto(fitsfile, output_verify='fix', overwrite=True)
    hdul.close()

    return fitsfile

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
                showplot=False,
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
        showplot : bool, optional
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
        self.showplot = showplot
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
                    showplot=self.showplot,
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

class FITPSF:
    """
    PSF-fitting helper for detecting and fitting a primary source and an optional companion.

    Implements a two-stage fitting strategy: (1) fit single-source model (optimize
    position, solve flux analytically) and (2) test for a companion by: early
    residual search → Stage A (freeze primary, optimize companion) → Stage B
    (joint relaxation). Model selection is done via BIC and configurable gates.

    Attributes
    ----------
    max_separation : float
        Maximum allowed separation (pixels) between primary and companion.
    min_separation : float
        Minimum allowed separation (pixels) between primary and companion.
    x_limits, y_limits : tuple of (float, float)
        Bounds for positional fits in x and y (in pixels, relative to stamp center).
    min_contrast, max_contrast : float
        Allowed contrast ratio bounds for a companion relative to primary.
    eps : float
        Minimizer step tolerance.
    maxiter : int
        Maximum iterations for nonlinear minimizer.
    background : float
        Scalar background level subtracted from tiles prior to fitting.
    r_sat : float
        Saturation radius (pixels). If > 0, saturated sources are handled via
        wing-matching instead of standard linear least-squares.
    debug : bool
        If True, enables verbose debug prints and diagnostic plots.
    showplot : bool
        If True, show final diagnostic plot after fit completion.

    Result Attributes
    -----------------
    peak1 : float
        Peak amplitude of the brightest (primary) fitted source.
    dx1, dy1 : float
        Position of primary source (offsets in pixels relative to the tile center).
    peak2 : float or 0.0
        Peak amplitude of the secondary/companion if detected. If no companion is
        detected, this is 0.0.
    dx2, dy2 : float or None
        Position of companion if detected. None if no companion is accepted.
    bintest : bool
        True if binary (two-source) model was preferred by selection criteria.

    """

    def __init__(self, max_separation=25, min_separation=1, r_sat=0, x_limits=(-3, 3), y_limits=(-3, 3),
                 min_contrast=0.05, max_contrast=1.0, eps=1e-3, maxiter=1000, background=0, bic_gate=10.0,debug=False, showplot=False):
                """
                Initialize FITPSF fitter with solver options and gating thresholds.

                Parameters
                ----------
                max_separation : float, optional
                    Maximum companion separation to consider (pixels). Default is 25.
                min_separation : float, optional
                    Minimum companion separation to accept (pixels). Default is 1.
                r_sat : float, optional
                    Saturation radius (pixels). If > 0, sources within this radius are
                    treated as saturated and fitted via wing-matching. Default is 0.
                x_limits : tuple of (float, float), optional
                    Allowed bounds [x_min, x_max] for x positional optimization relative
                    to stamp center. Default is (-3, 3).
                y_limits : tuple of (float, float), optional
                    Allowed bounds [y_min, y_max] for y positional optimization relative
                    to stamp center. Default is (-3, 3).
                min_contrast : float, optional
                    Minimum contrast (f2 / f1) required to accept a companion. Default is 0.05.
                max_contrast : float, optional
                    Maximum allowed contrast. Default is 1.0.
                eps : float, optional
                    Step tolerance passed to the nonlinear optimizer (L-BFGS-B).
                    Default is 1e-3.
                maxiter : int, optional
                    Maximum iterations for nonlinear minimizer. Default is 1000.
                background : float, optional
                    Constant background to subtract from tiles prior to fitting.
                    Default is 0.

                Attributes Initialized
                ----------------------
                peak1, dx1, dy1 : None
                    Primary fit results (populated after calling `fitpsf`).
                peak2, dx2, dy2 : None
                    Secondary/companion fit results or None if no companion accepted.
                bintest : bool
                    False initially; set to True if binary model is accepted.
                debug : bool
                    False initially; set to True to enable verbose diagnostics.
                showplot : bool
                    False initially; set to True to display final diagnostic plot.

                Notes
                -----
                This constructor only configures the fitter. The actual fitting is performed by
                calling `fitpsf(...)`, which will populate the result attributes on success.

                """

                self.max_separation = max_separation
                self.min_separation = min_separation
                self.x_limits = x_limits
                self.y_limits = y_limits
                self.min_contrast = min_contrast
                self.max_contrast = max_contrast
                self.eps = eps
                self.r_sat = r_sat
                self.maxiter = maxiter
                self.background = background
                self.bic_gate = bic_gate
                self.debug = debug
                self.showplot = showplot

                # Extracted parameters results containers (filled by fitpsf)
                self.peak1 = None
                self.dx1 = None
                self.dy1 = None
                self.peak2 = None
                self.dx2 = None
                self.dy2 = None
                self.bintest = False
                self.success = False


    def _solve_star_peak_linearly(self, clean_data, err_map, weights, imaging_psf, params,
                                   mode="single", weighted=True, r_sat1=0, r_sat2=0):
        """
        Analytical linear least-squares solver for peak amplitude.

        For saturated sources: fits PSF wings to data wings, then extrapolates peak.
        For non-saturated: standard linear least-squares.

        Parameters
        ----------
        clean_data : ndarray of shape (ny, nx)
            Background-subtracted data tile.
        err_map : ndarray of shape (ny, nx)
            Per-pixel uncertainties (standard deviations).
        weights : ndarray of shape (ny, nx)
            Mask/weights array (1 for valid, 0 for masked/NaN/saturated).
        imaging_psf : ndarray of shape (ny, nx)
            PSF template (normalized to peak=1).
        params : sequence of float
            [dx1, dy1] for single-source mode or
            [dx1, dy1, contrast, dx2, dy2] for binary mode.
        mode : {'single', 'binary'}, optional
            Fitting mode. Default is 'single'.
        weighted : bool, optional
            If True, weight by inverse variance (1/err_map²). If False, use
            binary weights only. Default is True.
        r_sat1 : float, optional
            Saturation radius for primary source. If > 0, use wing-matching
            instead of standard linear solve. Default is 0.
        r_sat2 : float, optional
            Saturation radius for companion (binary mode only). Default is 0.

        Returns
        -------
        f1_opt : float
            Fitted primary peak amplitude (true peak, not just scaling factor).
            Returns np.nan if fit fails or no valid data available.

        Notes
        -----
        - For saturated sources, the wing-matching strategy masks the saturated
          core and fits only the PSF wings to data wings, then extrapolates to
          the true peak.
        - The PSF is normalized to peak=1, so the returned scale factor directly
          equals the true peak amplitude.
        - Binary mode handles two sources with contrast ratio constraint.

        """
        imaging_psf_norm = imaging_psf / np.nanmax(imaging_psf)  # Normalize to peak=1

        if weighted:
            inv_sigma = weights / err_map
            inv_sigma[np.isnan(inv_sigma)] = 0
            d_flat = (clean_data * inv_sigma).flatten()
        else:
            inv_sigma = weights
            d_flat = (clean_data * weights).flatten()

        # Build PSF bases
        if mode == "single":
            dx1, dy1 = params
            psf_basis = ut.imshift(imaging_psf_norm, [dx1, dy1], method='spline',
                                   nan_reflected=False, pad_amount=0)

            # Check if saturated
            if r_sat1 > 0:
                return self._solve_peak_wings_single(clean_data, err_map, weights,
                                                     psf_basis, r_sat1, dx1, dy1, weighted)

            # Non-saturated: standard linear solve
            M = (psf_basis * inv_sigma).flatten()[:, np.newaxis]

        else:  # binary
            dx1, dy1, contrast, dx2, dy2 = params
            psf_basis1 = ut.imshift(imaging_psf_norm, [dx1, dy1], method='spline',
                                    nan_reflected=False, pad_amount=0)
            psf_basis2 = ut.imshift(imaging_psf_norm, [dx2, dy2], method='spline',
                                    nan_reflected=False, pad_amount=0)

            # Check if either source saturated
            if r_sat1 > 0 or r_sat2 > 0:
                return self._solve_peak_wings_binary(clean_data, err_map, weights,
                                                     psf_basis1, psf_basis2, contrast,
                                                     r_sat1, r_sat2, dx1, dy1, dx2, dy2, weighted)

            # Non-saturated: standard binary linear solve
            psf_basis = psf_basis1 + contrast * psf_basis2
            M = (psf_basis * inv_sigma).flatten()[:, np.newaxis]

        try:
            f1_opt, _, _, _ = np.linalg.lstsq(M, d_flat, rcond=None)
        except:
            f1_opt = np.nan

        return float(f1_opt)

    def _solve_peak_wings_single(self, clean_data, err_map, weights, psf_basis,
                                 r_sat, dx1, dy1, weighted):
        """
        Fit single saturated source by matching PSF wings to data wings.

        Strategy
        --------
        1. Mask saturated core (r < r_sat).
        2. Extract wing pixels where both PSF and data have signal.
        3. Fit PSF wings to data wings using linear least-squares.
        4. Return scaled peak (accounting for PSF normalization).

        Parameters
        ----------
        clean_data : ndarray of shape (ny, nx)
            Background-subtracted data tile.
        err_map : ndarray of shape (ny, nx)
            Per-pixel uncertainties.
        weights : ndarray of shape (ny, nx)
            Mask/weights array (1 for valid, 0 for masked).
        psf_basis : ndarray of shape (ny, nx)
            Shifted PSF template (already shifted to source position).
        r_sat : float
            Saturation radius (pixels). Pixels within this radius are masked out.
        dx1, dy1 : float
            Source center offset (pixels) relative to tile center.
        weighted : bool
            If True, apply inverse-variance weighting.

        Returns
        -------
        f1_opt : float
            Fitted peak amplitude of the primary source.
            Returns np.nan if wing mask is empty or fit fails.

        """
        ny, nx = clean_data.shape
        y_c, x_c = (ny - 1) / 2, (nx - 1) / 2
        y, x = np.ogrid[:ny, :nx]

        # Distance from source center
        # r_from_source = np.sqrt((x - x_c) ** 2 + (y - y_c) ** 2)
        r_from_source = np.sqrt((x - (x_c + dx1)) ** 2 + (y - (y_c + dy1)) ** 2)

        # Wing mask: outside saturated core, AND where PSF has signal
        wing_mask = (r_from_source >= r_sat) & (psf_basis > 1e-4)

        if not np.any(wing_mask):
            return np.nan

        # Build weighting for wings only
        if weighted:
            inv_sigma_wing = (weights.copy() / err_map)
            inv_sigma_wing[~np.isfinite(inv_sigma_wing)] = 0
        else:
            inv_sigma_wing = weights.copy()

        inv_sigma_wing[~wing_mask] = 0

        # Fit: data_wings = scale * psf_wings
        d_wing = (clean_data * inv_sigma_wing).flatten()
        M_wing = (psf_basis * inv_sigma_wing).flatten()[:, np.newaxis]

        # if np.all(M_wing == 0.0) or np.any(np.isnan(M_wing)):
        #     return np.nan
        try:
            scale, _, _, _ = np.linalg.lstsq(M_wing, d_wing, rcond=None)
        except:
            scale =  np.nan

        # scale is the fitted amplitude such that scale * PSF matches the data wings
        # Since PSF is normalized to peak=1, scale IS the true peak
        return float(scale)

    def _solve_peak_wings_binary(self, clean_data, err_map, weights,
                                 psf_basis1, psf_basis2, contrast,
                                 r_sat1, r_sat2, dx1, dy1, dx2, dy2, weighted):
        """
        Fit binary saturated sources sequentially for improved accuracy.

        Strategy
        --------
        1. Fit primary alone using wings (exclude companion saturation).
        2. Subtract primary model from data.
        3. Fit companion from residuals using wings (exclude primary saturation).
        4. Apply consistency check and return refined primary amplitude.

        This sequential approach improves accuracy for saturated binaries by
        reducing cross-talk between sources.

        Parameters
        ----------
        clean_data : ndarray of shape (ny, nx)
            Background-subtracted data tile.
        err_map : ndarray of shape (ny, nx)
            Per-pixel uncertainties.
        weights : ndarray of shape (ny, nx)
            Mask/weights array (1 for valid, 0 for masked).
        psf_basis1, psf_basis2 : ndarray of shape (ny, nx)
            Shifted PSF templates for primary and companion (already at source positions).
        contrast : float
            Contrast ratio (f2 / f1) used for companion model.
        r_sat1, r_sat2 : float
            Saturation radii for primary and companion (pixels).
        dx1, dy1, dx2, dy2 : float
            Source center offsets (pixels) relative to tile center.
        weighted : bool
            If True, apply inverse-variance weighting.

        Returns
        -------
        f1_opt : float
            Refined fitted peak amplitude of the primary source.
            Returns np.nan if primary wing mask is empty or fits fail.

        Notes
        -----
        A consistency check compares the primary amplitude fitted directly vs.
        fitted from companion residuals. If they differ by > 30%, the direct
        primary fit is preferred. Otherwise, a weighted average is returned.

        """
        ny, nx = clean_data.shape
        y_c, x_c = (ny - 1) / 2, (nx - 1) / 2
        y, x = np.ogrid[:ny, :nx]

        # Distance from each source center
        r1 = np.sqrt((x - (x_c + dx1)) ** 2 + (y - (y_c + dy1)) ** 2)
        r2 = np.sqrt((x - (x_c + dx2)) ** 2 + (y - (y_c + dy2)) ** 2)

        # Saturation masks
        sat_mask1 = (r1 < r_sat1) if r_sat1 > 0 else np.zeros((ny, nx), dtype=bool)
        sat_mask2 = (r2 < r_sat2) if r_sat2 > 0 else np.zeros((ny, nx), dtype=bool)

        # ========== STEP 1: Fit Primary ==========
        # Use wings where PSF1 has signal, outside companion saturation
        wing_mask_primary = (~sat_mask1) & (~sat_mask2) & (psf_basis1 > 1e-4)

        if not np.any(wing_mask_primary):
            return np.nan

        if weighted:
            inv_sigma_wing = weights.copy() / err_map
            inv_sigma_wing[~np.isfinite(inv_sigma_wing)] = 0
        else:
            inv_sigma_wing = weights.copy()

        inv_sigma_p = inv_sigma_wing.copy()
        inv_sigma_p[~wing_mask_primary] = 0

        d_primary = (clean_data * inv_sigma_p).flatten()
        M_primary = (psf_basis1 * inv_sigma_p).flatten()[:, np.newaxis]

        if np.all(M_primary == 0.0) or np.any(np.isnan(M_primary)):
            return np.nan

        try:
            f1_opt, _, _, _ = np.linalg.lstsq(M_primary, d_primary, rcond=None)
        except:
            f1_opt =  np.nan

        f1_opt = float(f1_opt)

        # ========== STEP 2: Subtract Primary & Fit Companion ==========
        # Create residual data: data - primary_model
        primary_model = f1_opt * psf_basis1
        residual_data = clean_data - primary_model

        # Use companion wings, outside primary saturation
        wing_mask_companion = (~sat_mask1) & (~sat_mask2) & (psf_basis2 > 1e-4)

        if not np.any(wing_mask_companion):
            # If companion wings not available, return primary-only fit
            return f1_opt

        inv_sigma_c = inv_sigma_wing.copy()
        inv_sigma_c[~wing_mask_companion] = 0

        d_companion = (residual_data * inv_sigma_c).flatten()
        # Note: we fit f2 = f1 * contrast, so we need to solve for the effective peak
        M_companion = (contrast * psf_basis2 * inv_sigma_c).flatten()[:, np.newaxis]

        if np.all(M_companion == 0.0) or np.any(np.isnan(M_companion)):
            # Companion fit failed, return primary estimate
            return f1_opt

        try:
            f1_companion_check, _, _, _ = np.linalg.lstsq(M_companion, d_companion, rcond=None)
            f1_companion_check = float(f1_companion_check)

            # Consistency check: companion fit should give similar primary amplitude
            # (within reasonable tolerance since it's fitted from residuals)
            if abs(f1_companion_check - f1_opt) / f1_opt < 0.3:  # Allow 30% difference
                # Use weighted average: primary fit more reliable
                f1_refined = 0.7 * f1_opt + 0.3 * f1_companion_check
                return f1_refined
        except:
            pass

        return f1_opt

    def _get_annulus_mask_by_comp_pos(self, tile, x_peak, y_peak, dr=5):
        """
        Construct an annulus mask centered on a companion candidate.

        The annulus radius is computed from the distance of the candidate to
        the tile center. Inner and outer radii are set to r ± dr. Pixels are
        included if their unit pixel-square intersects the annulus
        (i.e., partially-touching pixels are counted).

        Parameters
        ----------
        tile : ndarray of shape (ny, nx)
            Reference image (only its shape is used).
        x_peak : float
            Column index (x) of the companion/candidate in pixel coordinates.
        y_peak : float
            Row index (y) of the companion/candidate in pixel coordinates.
        dr : float, optional
            Half-width of the annulus around the candidate radius (pixels).
            Default is 5.

        Returns
        -------
        ann_mask : bool ndarray of shape (ny, nx)
            Boolean mask selecting pixels whose pixel-square intersects
            the annulus region.

        Notes
        -----
        The annulus is centered on the tile center and has inner radius r_in
        and outer radius r_out computed from the candidate distance. If debug
        mode is enabled, a diagnostic plot is displayed.

        """
        ny, nx = tile.shape
        x_center = (nx - 1) / 2.0
        y_center = (ny - 1) / 2.0
        # radial distance from center to candidate (float)
        r = np.sqrt((x_peak - x_center) ** 2 + (y_peak - y_center) ** 2)
        # inner and outer radii for annulus (ensure inner >= 0)
        r_in = max(0, r - dr)
        r_out = min(r + dr,min(nx//2,ny//2))

        y_idx, x_idx = np.indices(tile.shape)
        half = 0.5
        dx = np.abs(x_idx - x_center)
        dy = np.abs(y_idx - y_center)
        dx_min = np.maximum(dx - half, 0.0)
        dy_min = np.maximum(dy - half, 0.0)
        min_dist = np.sqrt(dx_min ** 2 + dy_min ** 2)
        dx_max = dx + half
        dy_max = dy + half
        max_dist = np.sqrt(dx_max ** 2 + dy_max ** 2)
        # annulus mask
        ann_mask = (min_dist <= r_out) & (max_dist >= r_in)

        if self.debug:
            ann_values = tile.copy()
            ann_values[~ann_mask] = np.nan
            load_plt_style(None)
            d_min = float(np.nanmin(ann_values))
            d_max = float(np.nanmax(ann_values))
            if d_min == d_max:
                # If the image is completely flat, expand the bounds symmetrically
                vmin, vmax = d_min - 1.0, d_max + 1.0
            else:
                vmin, vmax = d_min, d_max

            fig, ax = plt.subplots(figsize=(7, 7))
            # Pass explicit vmin and vmax parameters to safeguard color normalisation
            im = ax.imshow(ann_values, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
            ax.plot(x_peak, y_peak, '+r', markersize=15, markeredgewidth=3, label='Figure Center')
            plt.title('Annulus area for STD for companion')
            plt.tight_layout()
            plt.show()

        return ann_mask

    def _get_annulus_mask_by_radius(self, tile, x_peak, y_peak, r_in=None, r_out=None, dr=5.0):
        """
        Construct an annulus boolean mask around a given center.

        Pixels are included if their unit pixel-square intersects the annulus
        [r_in, r_out] (i.e., partially-touching pixels are counted).

        Parameters
        ----------
        tile : ndarray of shape (ny, nx)
            Tile used to determine mask shape (only its shape is used).
        x_peak : float
            Column index (x) of the annulus center in pixel coordinates.
        y_peak : float
            Row index (y) of the annulus center in pixel coordinates.
        r_in : float or None, optional
            Inner radius of annulus (pixels). If None, defaults to
            max(r_sat, min_separation). Default is None.
        r_out : float or None, optional
            Outer radius of annulus (pixels). If None, defaults to
            max_separation or r_in + dr. Default is None.
        dr : float, optional
            Half-width fallback used when r_in/r_out unspecified (pixels).
            Default is 5.0.

        Returns
        -------
        ann_mask : bool ndarray of shape (ny, nx)
            True for pixels whose pixel-square intersects the annulus region.

        Notes
        -----
        The annulus region is defined as pixels where the minimal distance
        to the annulus center is >= r_in and the maximal distance is <= r_out.

        """
        ny, nx = tile.shape
        if r_in is None:
            r_in = np.max(self.r_sat,float(getattr(self, "min_separation", 2.0)))
        if r_out is None:
            r_out = float(getattr(self, "max_separation", r_in + dr))

        y_idx, x_idx = np.indices((ny, nx))
        dx = np.abs(x_idx - float(x_peak))
        dy = np.abs(y_idx - float(y_peak))
        half = 0.5
        # minimal distance from center to any point in the pixel square
        dx_min = np.maximum(dx - half, 0.0)
        dy_min = np.maximum(dy - half, 0.0)
        min_dist = np.sqrt(dx_min ** 2 + dy_min ** 2)

        # maximal distance to the farthest corner
        dx_max = dx + half
        dy_max = dy + half
        max_dist = np.sqrt(dx_max ** 2 + dy_max ** 2)

        # include pixels whose square intersects [r_in, r_out]
        ann_mask = (max_dist <= float(r_out)) & (min_dist >= float(r_in))

        if self.debug:
            ann_values = tile.copy()
            ann_values[~ann_mask] = np.nan
            load_plt_style(None)
            d_min = float(np.nanmin(ann_values))
            d_max = float(np.nanmax(ann_values))
            if d_min == d_max:
                # If the image is completely flat, expand the bounds symmetrically
                vmin, vmax = d_min - 1.0, d_max + 1.0
            else:
                vmin, vmax = d_min, d_max

            fig, ax = plt.subplots(figsize=(7, 7))
            # Pass explicit vmin and vmax parameters to safeguard color normalisation
            im = ax.imshow(ann_values, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
            ax.plot(x_peak, y_peak, '+r', markersize=15, markeredgewidth=3)
            plt.title('Selected annulus area')
            plt.tight_layout()
            plt.show()
        return ann_mask

    def _get_std_in_annulus(self,tile,x_peak,y_peak,dr=5):
        """
        Compute robust local standard deviation using pixel values in an annulus.

        Parameters
        ----------
        tile : ndarray of shape (ny, nx)
            Input image (or residual map) from which the annulus sample is drawn.
        x_peak : float
            Column index (x) of the annulus center in pixel coordinates.
        y_peak : float
            Row index (y) of the annulus center in pixel coordinates.
        dr : float, optional
            Half-width used to build the annulus when r_in/r_out not
            explicitly provided (pixels). Default is 5.

        Returns
        -------
        ann_std : float
            Sample standard deviation of finite pixels inside the annulus
            (using ddof=1 for sample statistics).
            Returns np.nan if the annulus contains no finite pixels.

        """
        ann_mask = self._get_annulus_mask_by_comp_pos(tile, x_peak, y_peak, dr=dr)
        ann_values = tile.copy()
        ann_values[~ann_mask]=np.nan

        if ann_values.size == 0:
            ann_std = np.nan
        else:
            # population standard deviation (ddof=0)
            ann_std = np.nanstd(ann_values, ddof=1)
            # sample standard deviation (ddof=1), if you prefer:
            # ann_std_sample = np.std(ann_values, ddof=1)

        return ann_std

    def _plot_final_fit(self, tile):
        """
        Generate and display diagnostic plot of fitted positions over the tile.

        Shows the input tile with fitted primary (red X) and optional companion
        (blue X) positions overlaid.

        Parameters
        ----------
        tile : ndarray of shape (ny, nx)
            Input tile used for final diagnostic overlay.

        Notes
        -----
        Display is skipped if showplot is False. The plot uses dynamic vmin/vmax
        scaling to safely handle flat or negative-valued tiles.

        """
        load_plt_style(None)

        # Plot Diagnostics Window
        ny, nx = tile.shape
        fitted_x1_pos = (nx-1) / 2 + self.dx1
        fitted_y1_pos = (ny-1) / 2 + self.dy1

        # 1. Safely extract the data min and max bounds
        tile_min = float(np.nanmin(tile))
        tile_max = float(np.nanmax(tile))

        # 2. FIX: Dynamically safeguard the normalization limits
        # If the image is completely flat or invalid, provide a safe default window
        if tile_max <= tile_min:
            vmin, vmax = -1.0, 1.0
        else:
            vmin = tile_min
            # Try your preferred 20% scaling threshold
            vmax_trial = tile_max * 0.2

            # If the scaled vmax falls below or equal to vmin (due to negative values),
            # fall back to a safe upper bound (e.g., halfway between min and max)
            if vmax_trial <= vmin:
                vmax = vmin + (tile_max - vmin) * 0.5
            else:
                vmax = vmax_trial

        # Double check to guarantee absolute safety before passing to Matplotlib
        if vmin >= vmax:
            vmax = vmin + 1.0

        fig, ax = plt.subplots(figsize=(5, 5))

        # Pass explicit, fully verified parameters to the renderer
        ax.imshow(tile, origin='lower', vmin=vmin, vmax=vmax, cmap='viridis')

        ax.plot(fitted_x1_pos, fitted_y1_pos, 'xr', markersize=12, label='Fitted Star 1')
        if self.bintest:
            fitted_x2_pos = (nx-1) / 2 + self.dx2
            fitted_y2_pos = (ny-1) / 2 + self.dy2
            ax.plot(fitted_x2_pos, fitted_y2_pos, 'xb', markersize=12, label='Fitted Star 2')
        ax.legend()
        plt.show()

    def _fit_two_circles_to_mask(self, dilated_mask, min_separation=1.0, max_separation=25.0,
                                 min_radius=1.0, max_radius=30.0, clamp_value =1, debug=False):
        """
        Fit two circles sequentially to maximize coverage of saturated blob mask.

        Strategy
        --------
        1. Compute distance transform of the saturated blob mask.
        2. Find the largest circle (Circle 1) that fits entirely inside the mask.
        3. Find the largest second circle (Circle 2) that maximizes coverage of
           uncovered areas (allows overlap with Circle 1, but prioritizes new coverage).
        4. Check separation constraint; if violated, drop Circle 2.
        5. Ensure Circle 1 is the larger/more central one.

        Parameters
        ----------
        dilated_mask : bool ndarray of shape (ny, nx)
            Binary mask of saturated region (1 = saturated, 0 = valid).
        min_separation : float, optional
            Minimum allowed separation between circle centers (pixels).
            Default is 1.0.
        max_separation : float, optional
            Maximum allowed separation between circle centers (pixels).
            Default is 25.0.
        min_radius : float, optional
            Minimum allowed circle radius (pixels). Default is 1.0.
        max_radius : float, optional
            Maximum allowed circle radius (pixels). Default is 30.0.
        clamp_value: float, optional
            Fraction of maximum distance to boundary used to clamp Circle 1 and 2 radius.
            Default is 0.98 (i.e., 98% of max distance).
        debug : bool, optional
            If True, display diagnostic plot showing fitted circles and mask.
            Default is False.

        Returns
        -------
        dx1, dy1 : float
            Center offset of Circle 1 relative to tile center (pixels).
        r1 : float
            Radius of Circle 1 (pixels).
        dx2, dy2 : float or None
            Center offset of Circle 2 relative to tile center (pixels).
            Set to None if separation constraint violated or single circle sufficient.
        r2 : float
            Radius of Circle 2 (pixels). Undefined if dx2, dy2 are None.
        fit_error : float
            Fit error metric (currently always 0.0).

        Notes
        -----
        - Circles are constrained to stay fully within the mask.
        - Circle 1 center is chosen as the point with maximum distance to mask boundary.
        - Circle 2 center is selected to maximize coverage of mask areas not covered by Circle 1.
        - Tie-breaking for Circle 2: among pixels with equal coverage score, the one
          farthest from Circle 1 is chosen.

        """

        ny, nx = dilated_mask.shape
        x_center, y_center = (nx - 1) / 2.0, (ny - 1) / 2.0

        # Compute distance transform: each pixel's distance to mask boundary
        distance_map = distance_transform_edt(dilated_mask.astype(int))
        # ===== STEP 1: Find largest circle (Circle 1) =====
        # The center with maximum distance to boundary is the best candidate
        y_max, x_max = np.unravel_index(np.argmax(distance_map), distance_map.shape)
        x1 = float(x_max) + 0.005
        y1 = float(y_max) - 0.005
        r1_max = distance_map[y_max, x_max]

        # Clamp to valid range
        r1 = min(r1_max * clamp_value, max_radius)  # 98% to add small margin
        r1 = max(r1, min_radius)


        # ===== STEP 2: Find largest circle for Circle 2 =====
        # Strategy: For each candidate location, compute the maximum radius that:
        # 1. Stays within the border (distance_map)
        # 2. Stays within the mask
        # Then compute coverage of uncovered areas (areas not covered by Circle 1)

        # Create a mask of Circle 1 for coverage calculation
        yy, xx = np.indices(dilated_mask.shape, dtype=np.float64)
        circle1_mask = (xx - x1) ** 2 + (yy - y1) ** 2 <= r1 ** 2

        # For each pixel, compute maximum radius it can have
        # max_radius_at_point = distance_map.copy().astype(float)

        # Score each pixel as a potential Circle 2 center
        # based on coverage of uncovered areas
        coverage_score = np.zeros(dilated_mask.shape, dtype=np.float64)
        for iy in range(ny):
            for ix in range(nx):
                if not dilated_mask[iy, ix]:
                    coverage_score[iy, ix] = -np.inf
                else:
                    # Maximum radius at this point (stays within border)
                    r_max_at_point = distance_map[iy, ix]

                    if r_max_at_point < min_radius:
                        coverage_score[iy, ix] = -np.inf
                    else:
                        # Create a circle at this location with max radius
                        circle2_mask = (xx - ix) ** 2 + (yy - iy) ** 2 <= r_max_at_point ** 2

                        # Calculate uncovered area: Circle 2 but not Circle 1
                        uncovered = circle2_mask & dilated_mask & (~circle1_mask)

                        # Score: how much NEW area does this cover
                        coverage_score[iy, ix] = float(np.sum(uncovered))

        # Find all pixels with max score
        max_score = np.max(coverage_score)
        tie_pixels = np.argwhere(coverage_score == max_score)

        if len(tie_pixels) > 1:
            # Among ties, pick the one farthest from Circle 1
            distances_from_c1 = np.sqrt((tie_pixels[:, 1] - x1) ** 2 + (tie_pixels[:, 0] - y1) ** 2)
            best_idx = np.argmax(distances_from_c1)
            y_max_2, x_max_2 = tie_pixels[best_idx]
            log.debug( f"Coverage tie-breaking: {len(tie_pixels)} pixels with score={max_score:.1f}, chose ({x_max_2}, {y_max_2}) [dist from C1: {distances_from_c1[best_idx]:.2f}]")
        else:
            y_max_2, x_max_2 = tie_pixels[0]


        # # Find best location for Circle 2 (maximize uncovered coverage)
        # y_max_2, x_max_2 = np.unravel_index(np.argmax(coverage_score), coverage_score.shape)
        # Find ALL pixels with the max score (ties)
        # max_score = np.max(coverage_score)
        # tie_locations = np.argwhere(coverage_score == max_score)
        x2 = float(x_max_2) + 0.005
        y2 = float(y_max_2) - 0.005
        r2_max = distance_map[y_max_2, x_max_2]

        # Clamp to valid range
        r2 = min(r2_max * clamp_value, max_radius)
        r2 = max(r2, min_radius)

        # Check separation constraint
        sep = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        if sep < min_separation or sep > max_separation:
            dx1 = x1 - x_center
            dy1 = y1 - y_center
            log.debug(f"Circle 1: (dx1, dy1)=({dx1:.4f}, {dy1:.4f}), r1={r1:.2f}")
            log.debug(f"Circles too close/far away: sep={sep:.2f} < min_separation={min_separation} or sep={sep:.2f} > max_separation={max_separation}. Dropping circle 2")
            dx2 = None
            dy2 = None
            x2 = None
            y2 = None
        else:
            dx1 = x1 - x_center
            dy1 = y1 - y_center
            dx2 = x2 - x_center
            dy2 = y2 - y_center
            if np.sqrt(dx1 ** 2 + dy1 ** 2) > np.sqrt(dx2 ** 2 + dy2 ** 2):
                r2_temp=np.copy(r2)
                r2=r1
                r1=r2_temp
                dx1 = x2 - x_center
                dy1 = y2 - y_center
                dx2 = x1 - x_center
                dy2 = y1 - y_center

            log.debug(f"Circle 1: (dx1, dy1)=({dx1:.4f}, {dy1:.4f}), r1={r1:.2f}")
            log.debug(f"Circle 2: (dx2, dy2)=({dx2:.4f}, {dy2:.4f}), r2={r2:.2f}, sep={sep:.2f}")

        if debug:
            load_plt_style(None)
            canvas = dilated_mask.astype(float)
            d_min = float(np.min(canvas))
            d_max = float(np.max(canvas))
            if d_min == d_max:
                vmin, vmax = d_min - 1.0, d_max + 1.0
            else:
                vmin, vmax = d_min, d_max

            fig, ax = plt.subplots(figsize=(9, 9))
            im = ax.imshow(canvas, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
            ax.contour(dilated_mask, levels=[0.5], colors='white', linewidths=2,
                       linestyles='dashed', label='Blob Boundary')

            # Plot circles
            circle1 = patches.Circle((dx1+x_center, dy1+y_center), r1, fill=False, edgecolor='red',
                                     linewidth=2.5, label=f'Circle 1 (r={r1:.2f})')
            ax.add_patch(circle1)
            if x2 is not None and y2 is not None:
                circle2 = patches.Circle((dx2+x_center, dy2+y_center), r2, fill=False, edgecolor='blue',
                                         linewidth=2.5, label=f'Circle 2 (r={r2:.2f})')
                ax.add_patch(circle2)

            ax.plot(dx1+x_center, dy1+y_center, 'Xr', markersize=12, markeredgewidth=2,
                    label=f'Circle 1 Center: [{dx1+x_center:.4f}, {dy1+y_center:.4f}]')
            if x2 is not None and y2 is not None:
                ax.plot(dx2+x_center, dy2+y_center, 'X', color='orange', markersize=12, markeredgewidth=2,
                        label=f'Circle 2 Center: [{dx2+x_center:.4f}, {dy2+y_center:.4f}]')
            ax.plot(x_center, y_center, 'g+', markersize=15, markeredgewidth=2.5,
                    label='Tile Center')

            # ax.set_xlim(x_center, x_center)
            # ax.set_ylim(y_center, y_center)
            ax.set_title(f"Two-Circle Fit (Coverage-Based): sep={sep:.2f}px, r1={r1:.2f}, r2={r2:.2f}")
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2,
                      fontsize=9, framealpha=0.95)
            # plt.colorbar(im, ax=ax, label='Saturated Pixels')
            ax.set_aspect('equal')
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.18)
            plt.show()

        return dx1, dy1, r1, dx2, dy2, r2, 0.0

    def _make_educated_guesses(self, tile_with_nans, clean_data, err_map, weights, imaging_psf, nanmask):
        """
        Generate initial guesses for primary and companion positions/amplitudes.

        Strategy
        --------
        - If saturated pixels present in tile center: use two-circle fit on saturated mask.
        - Otherwise: search for brightest pixel in valid data as primary guess.
        - Search residuals for companion candidate at minimum distance.

        Parameters
        ----------
        tile_with_nans : ndarray of shape (ny, nx)
           Original tile with NaN values marking saturated/invalid pixels.
        clean_data : ndarray of shape (ny, nx)
           Background-subtracted data (NaN replaced with 0).
        err_map : ndarray of shape (ny, nx)
           Per-pixel uncertainties.
        weights : ndarray of shape (ny, nx)
           Mask/weights array (1 for valid, 0 for saturated/masked).
        imaging_psf : ndarray of shape (ny, nx)
           PSF template for peak estimation.
        nanmask : ndarray of shape (ny, nx)
           Binary mask (1 for saturated, 0 for valid).

        Returns
        -------
        p1_guess : float
           Estimated primary peak amplitude.
        dx1_guess, dy1_guess : float
           Estimated primary center offset (pixels).
        r1 : float
           Estimated primary saturation radius (pixels).
        p2_guess, c2_guess : float or None
           Estimated companion peak amplitude and contrast. None if no companion detected.
        dx2_guess, dy2_guess : float or None
           Estimated companion center offset. None if no companion detected.
        r2 : float
           Estimated companion saturation radius.
        labeled : ndarray
           Connected component labels of saturated mask (for debugging).
        mask_bool_second_bests : bool ndarray
           Mask of secondary saturated components (useful for tie-breaking).

        Notes
        -----
        - If center of tile is saturated, identifies the largest connected component.
        - Secondary components are tracked for companion initialization.

        """
        ny, nx = tile_with_nans.shape
        center_radius = 1  # pixels
        yy, xx = np.indices(nanmask.shape)
        center_mask = ((xx - (nx - 1) / 2) ** 2 + (yy - (ny - 1) / 2) ** 2) <= (center_radius + 0.5) ** 2

        if not np.any(np.isnan(tile_with_nans[center_mask])):
            num_sat_pixels=0
            mask_bool = (nanmask == 1)
            labeled = label(mask_bool, connectivity=1)
            mask_bool_second_bests = np.zeros_like(mask_bool, dtype=bool)
            second_best_labels = (labeled != 0)
            mask_bool_second_bests[second_best_labels] = True
            pass
        else:
            mask_bool = (nanmask == 1)
            labeled = label(mask_bool, connectivity=1)
            center_counts = np.bincount(labeled[center_mask].ravel(), minlength=(labeled.max() + 1))
            center_counts[0] = 0  # ignore background
            if center_counts.sum() == 0:
                # no component touches the center -> clear everything
                num_sat_pixels = 0
            else:
                best_label = int(np.argmax(center_counts))
                mask_bool = np.zeros_like(mask_bool, dtype=bool)
                mask_bool[labeled == best_label] = True
                num_sat_pixels = np.sum(mask_bool)

                # Get counts for all labels
                all_counts = np.bincount(labeled.ravel(), minlength=(labeled.max() + 1))

                # Find the second largest (excluding background 0 and the best_label)
                all_counts[0] = 0  # ignore background
                all_counts[best_label] = 0  # ignore best_label

                # Find the next largest
                # second_best_labels = int(np.argmax(all_counts))
                mask_bool_second_bests = np.zeros_like(mask_bool, dtype=bool)
                second_best_labels = (labeled != 0) & (labeled != best_label)
                mask_bool_second_bests[second_best_labels] = True

        if num_sat_pixels > 0:
            filled_mask = binary_fill_holes(mask_bool.copy())
            dx1_guess, dy1_guess, r1, dx2_guess, dy2_guess, r2, fit_error = self._fit_two_circles_to_mask(
                filled_mask.copy(),
                min_separation=self.min_separation,
                max_separation=np.inf,
                debug=self.debug
            )
            p1_guess = self._solve_star_peak_linearly(clean_data, err_map, weights, imaging_psf,
                                                     [dx1_guess, dy1_guess], mode="single", r_sat1=r1)

            log.debug(f"[Early Double check]")
            log.debug(f"  primary position guesses: ({dx1_guess:.4f}, {dy1_guess:.4f}), p1_guess: {p1_guess:.2f}")
            if dx2_guess is not None and dy2_guess is not None:
                p2_guess = self._solve_star_peak_linearly(clean_data, err_map, weights, imaging_psf,
                                                     [dx2_guess,dy2_guess], mode="single", r_sat1=r2)
                c2_guess = p2_guess/p1_guess
                log.debug(f"  companion position guesses: ({dx2_guess:.4f}, {dy2_guess:.4f}), p2_guess: {p2_guess:.2f}, c2_guess: {c2_guess:.2f}")
            else:
                p2_guess, c2_guess = None, None
                log.debug(f"  companion discarded: setting guesses to None for centroid approach")

        else:
            search_canvas1 = clean_data * weights
            y_max_p1, x_max_p1 = np.unravel_index(np.nanargmax(search_canvas1), (ny, nx))
            dx1_guess = float(x_max_p1 - (nx - 1) / 2) - 0.005
            dy1_guess = float(y_max_p1 - (ny - 1) / 2) + 0.005
            dx2_guess, dy2_guess = None, None
            p2_guess, c2_guess = None, None
            r1, r2 = 0 , 0

            p1_guess = self._solve_star_peak_linearly(clean_data, err_map, weights, imaging_psf,
                                                     [dx1_guess,dy1_guess], mode="single", r_sat1=r1)

            log.debug(f"[Early Single check]")
            log.debug(f"  primary position guesses: ({dx1_guess:.4f}, {dy1_guess:.4f}), p1_guess: {p1_guess:.2f}")
            log.debug(f"  no companion found: setting guesses to None for centroid approach")

        return p1_guess, dx1_guess, dy1_guess, r1, p2_guess, c2_guess, dx2_guess, dy2_guess, r2, labeled, mask_bool_second_bests

    def _one_source_model(self, clean_data, nanmask, err_map, weights, imaging_psf, p1_guess, dx1_guess, dy1_guess, num_data_points, r1):
        """
        Fit single-source model (optimize dx1, dy1; solve f1 analytically).

        Minimizes chi-squared for position parameters while solving peak
        amplitude linearly at each iteration.

        Parameters
        ----------
        clean_data : ndarray of shape (ny, nx)
           Background-subtracted data tile.
        nanmask : ndarray of shape (ny, nx)
           Binary mask (1 for saturated, 0 for valid).
        err_map : ndarray of shape (ny, nx)
           Per-pixel uncertainties.
        weights : ndarray of shape (ny, nx)
           Mask/weights array (1 for valid, 0 for masked).
        imaging_psf : ndarray of shape (ny, nx)
           PSF template (normalized to peak=1).
        p1_guess : float
           Initial guess for primary peak amplitude (used for gating).
        dx1_guess, dy1_guess : float
           Initial guess for primary center offset (pixels).
        num_data_points : int
           Number of valid (non-masked) pixels (used for BIC calculation).
        r1 : float
           Saturation radius of primary source (pixels).

        Returns
        -------
        p1_stage_a : float
           Fitted primary peak amplitude.
        dx1_stage_a, dy1_stage_a : float
           Fitted primary center offset (pixels).
        bic_1 : float
           Bayesian Information Criterion for single-source model.
           Set to np.inf if fit fails.

        Notes
        -----
        - Uses L-BFGS-B optimizer with bounds [x_limits, y_limits].
        - Peak amplitude is constrained to stay within 75%-135% of initial guess
         (if guess is available) to prevent runaway fits.
        - Chi-squared is computed only on non-saturated pixels.

        """
        bounds_1 = [self.x_limits, self.y_limits]
        guess_1 = [dx1_guess, dy1_guess]

        def chisq_1(params):
            p1 = self._solve_star_peak_linearly(clean_data, err_map, weights, imaging_psf,
                                               params, mode="single",
                                               r_sat1=r1)

            if not np.isfinite(p1):
                return 1e18
            if p1_guess is not None:
                if p1 > (p1_guess * 1.35) or p1 < (max(1e-6, p1_guess * 0.75)):
                    return 1e18

            mod = p1 * ut.imshift(imaging_psf / np.nanmax(imaging_psf), params,
                                  method='spline', nan_reflected=False, pad_amount=0)

            residuals = (clean_data - mod)  # / err_map
            chi_sq = np.nansum(((residuals[nanmask == 0] * weights[nanmask == 0]) ** 2))
            return chi_sq

        log.debug(f"[STAGE A - ONE SOURCE MODEL]")
        log.debug(f"  Initial guess_1: dx1={guess_1[0]:.4f}, dy1={guess_1[1]:.4f}")
        test_chi = chisq_1(guess_1)
        log.debug(f"  Chisq at initial guess: {test_chi:.4e}")

        eps_vector_1 = [1e-3, 1e-3]
        res_1 = minimize(chisq_1, guess_1, method='L-BFGS-B', bounds=bounds_1,
                         options={'eps': eps_vector_1, 'maxiter': self.maxiter, 'ftol': 1e-12})
        success = res_1.success
        if not (np.isfinite(res_1.fun) and res_1.fun < 1e16 and not success):
            bic_1 = np.inf
        else:
            bic_1 = res_1.fun + len(guess_1) * np.log(num_data_points)

        dx1_stage_a, dy1_stage_a = res_1.x
        p1_stage_a = self._solve_star_peak_linearly(clean_data, err_map, weights, imaging_psf,
                                                   res_1.x, mode="single",
                                                   r_sat1=r1)
        if not np.isfinite(p1_stage_a):
            success = False
        log.debug(f"  Final res_1.x: dx1_stage_a={dx1_stage_a:.4f}, dy1_stage_a={dy1_stage_a:.4f}. p1_stage_a: {p1_stage_a:.2f}")
        log.debug(f"  Chisq at final res_1.x: {chisq_1(res_1.x):.4e}. Success: {success}")
        return p1_stage_a, dx1_stage_a, dy1_stage_a, bic_1, success

    def _companion_centroid_search(self, clean_data, nanmask, err_map, weights, imaging_psf ,p1_stage_a, dx1_stage_a, dy1_stage_a, r1, r2, labeled, mask_bool_second_best):
        """
        Search residuals for companion candidate using centroid approach.

        Strategy
        --------
        1. Subtract fitted primary from data to get residuals.
        2. Mask annulus around primary (excluding primary and companion saturation).
        3. Find brightest residual pixel (companion peak).
        4. If nearby saturated blob detected: use blob centroid.
        5. Else: use weighted centroid in small window around peak.
        6. Gate companion by 5-sigma noise threshold (annulus std).

        Parameters
        ----------
        clean_data : ndarray of shape (ny, nx)
            Background-subtracted data tile.
        nanmask : ndarray of shape (ny, nx)
           Binary mask (1 for saturated, 0 for valid).
        err_map : ndarray of shape (ny, nx)
            Per-pixel uncertainties.
        weights : ndarray of shape (ny, nx)
            Mask/weights array (1 for valid, 0 for masked).
        imaging_psf : ndarray of shape (ny, nx)
            PSF template.
        p1_stage_a : float
            Fitted primary peak amplitude from stage A.
        dx1_stage_a, dy1_stage_a : float
            Fitted primary center offset (pixels).
        r1, r2 : float
            Saturation radii of primary and companion candidates (pixels).
        labeled : ndarray
            Connected component labels of saturated mask.
        mask_bool_second_best : bool ndarray
            Mask of secondary saturated components.

        Returns
        -------
        p2_stage_a : float or None
            Estimated companion peak amplitude. None if rejected.
        c2_stage_a : float or None
            Estimated companion contrast (f2/f1). None if rejected.
        dx2_stage_a, dy2_stage_a : float or None
            Estimated companion center offset (pixels). None if rejected.

        Notes
        -----
        - Two gating mechanisms:
          1. If nearby saturated patch detected (within 2 pixels): accept companion.
          2. Otherwise: require peak > 5-sigma annulus noise.
        - Contrast is floored at min_contrast to ensure physical plausibility.

        """
        ny, nx = clean_data.shape
        nan_data=clean_data.copy()
        nan_data[nanmask==1] = np.nan
        # Mild/No Saturation: Run standard residual subtraction search to catch distant companions
        s1_basis_final = ut.imshift(imaging_psf, [dx1_stage_a, dy1_stage_a], method='spline', nan_reflected=False,
                                    pad_amount=0)
        companion_residuals = (nan_data - (p1_stage_a * s1_basis_final)) * weights
        search_residuals = companion_residuals.copy()
        dynamic_shield = np.max([r1, float(getattr(self, "min_separation", 2.5))])
        mask = self._get_annulus_mask_by_radius(search_residuals, (nx - 1) / 2, (ny - 1) / 2,
                                                r_in=dynamic_shield,
                                                r_out=np.inf)
        search_residuals[~mask] = np.nan
        y_peak, x_peak = np.unravel_index(np.nanargmax(search_residuals), (ny, nx))

        log.debug(f"[STAGE A - COMPANION CENTROID SEARCH]")
        y_coords, x_coords = np.where(mask_bool_second_best)
        distances = np.sqrt((x_coords - x_peak) ** 2 + (y_coords - y_peak) ** 2)
        X=2
        if len(distances > 0) and np.min(distances) <= X:
            closest_idx = np.argmin(distances)
            selected_patch_label = labeled[y_coords[closest_idx], x_coords[closest_idx]]
            log.debug(f"Closest patch within {X} px is labeled {selected_patch_label}")
            mask_bool = np.zeros_like(labeled, dtype=bool)
            mask_bool[labeled == selected_patch_label] = True

            y_peak, x_peak = center_of_mass(mask_bool)
            y_coords, x_coords = np.where(mask_bool)
            distances = np.sqrt((x_coords - x_peak) ** 2 + (y_coords - y_peak) ** 2)
            r2 = np.mean(distances)
            dx2_guess = float(x_peak - ((nx - 1) / 2.0))
            dy2_guess = float(y_peak - ((ny - 1) / 2.0))
            peak_guess = self._solve_star_peak_linearly(nan_data, err_map, weights, imaging_psf,
                                               [dy2_guess, dx2_guess], mode="single",
                                               r_sat1=r2)
            threshold_val = None

        else:
            y_min, y_max = max(0, y_peak - 2), min(ny, y_peak + 3)
            x_min, x_max = max(0, x_peak - 2), min(nx, x_peak + 3)
            sub_window = np.fmax(search_residuals[y_min:y_max, x_min:x_max], 0.0)
            sub_sum = float(np.nansum(sub_window))
            dx2_guess = float(x_peak - ((nx - 1) / 2.0))
            dy2_guess = float(y_peak - ((ny - 1) / 2.0))
            peak_guess = float(search_residuals[y_peak, x_peak])
            threshold_val = 5 * self._get_std_in_annulus(search_residuals, x_peak, y_peak, dr=3)

        log.debug(f"  Initial guess_comp: dx2_guess={dx2_guess:.2f}, dy2_guess={dy2_guess:.2f}")
        log.debug(f"  Candidate selection with robust gating: x2={x_peak:.2f}, y2={y_peak:.2f}")
        log.debug(f"  Candidate initial peak: {peak_guess:.2f}. Estimated threshold (5-sigma): {threshold_val:.2f}")
        if threshold_val is None:
            dx2_stage_a = dx2_guess
            dy2_stage_a = dy2_guess
            p2_stage_a =peak_guess
            c2_stage_a = np.nanmax([p2_stage_a / p1_stage_a, self.min_contrast])
            log.debug(f"[Companion accepted]")
            log.debug(f"  Sigma threshold skipped, plausible saturated patch detected")
        elif peak_guess > threshold_val:
            y_mesh, x_mesh = np.mgrid[y_min:y_max, x_min:x_max]
            cx = float(np.nansum(x_mesh * sub_window) / sub_sum)
            cy = float(np.nansum(y_mesh * sub_window) / sub_sum)
            dx2_stage_a = float(cx - ((nx - 1) / 2.0))
            dy2_stage_a = float(cy - ((ny - 1) / 2.0))
            p2_stage_a = self._solve_star_peak_linearly(nan_data, err_map, weights, imaging_psf,
                                                       [dx2_stage_a, dy2_stage_a], mode="single", r_sat1=r2)
            c2_stage_a = np.nanmax([p2_stage_a / p1_stage_a, self.min_contrast])
            log.debug(f"[Companion accepted]")
            log.debug(f"  above 5-sigma annulus: {peak_guess:.2f}>{threshold_val:.2f}")
        else:
            log.debug(f"[Companion rejected]")
            log.debug(f"  below 5-sigma annulus: {peak_guess:.2f}<={threshold_val:.2f}")
            p2_stage_a, c2_stage_a, dx2_stage_a, dy2_stage_a = None, None, None, None

        return p2_stage_a, c2_stage_a, dx2_stage_a, dy2_stage_a

    def _companion_search_frozen_primary(self, clean_data, nanmask, err_map, weights, imaging_psf, p1_stage_a, dx1_stage_a, dy1_stage_a, p2_stage_a, c2_stage_a, dx2_stage_a, dy2_stage_a):
        """
        Optimize companion parameters with primary position frozen (Stage B).

        Minimizes chi-squared over (contrast, dx2, dy2) while keeping the
        primary position and amplitude fixed.

        Parameters
        ----------
        clean_data : ndarray of shape (ny, nx)
            Background-subtracted data tile.
        nanmask : ndarray of shape (ny, nx)
            Binary mask (1 for saturated, 0 for valid).
        err_map : ndarray of shape (ny, nx)
            Per-pixel uncertainties.
        weights : ndarray of shape (ny, nx)
            Mask/weights array (1 for valid, 0 for masked).
        imaging_psf : ndarray of shape (ny, nx)
            PSF template (normalized to peak=1).
        p1_stage_a, dx1_stage_a, dy1_stage_a : float
            Fitted primary parameters from stage A (held fixed).
        p2_stage_a, c2_stage_a, dx2_stage_a, dy2_stage_a : float
            Initial guesses for companion parameters.

        Returns
        -------
        c2_seed : float
            Optimized companion contrast.
        dx2_seed, dy2_seed : float
            Optimized companion center offset (pixels).

        Notes
        -----
        - Companion position is allowed to drift within [dx2_guess ± x_limits].
        - Uses L-BFGS-B optimizer with bounds on contrast and position.

        """
        guess_comp = [c2_stage_a, dx2_stage_a, dy2_stage_a]
        bounds_comp = [
            (self.min_contrast, self.max_contrast),
            (dx2_stage_a + self.x_limits[0], dx2_stage_a + self.x_limits[1]),  # Limit dx2 drift
            (dy2_stage_a + self.y_limits[0], dy2_stage_a + self.y_limits[1])  # Limit dy2 drift
        ]

        s1 = p1_stage_a * ut.imshift(imaging_psf / np.nanmax(imaging_psf), [dx1_stage_a, dy1_stage_a], method='spline',
                                     nan_reflected=False,
                                     pad_amount=0)

        def chisq_companion_stage(comp_params):
            contrast, dx2, dy2 = comp_params
            s2 = (p1_stage_a * contrast) * ut.imshift(imaging_psf / np.nanmax(imaging_psf), [dx2, dy2], method='spline',
                                                      nan_reflected=False, pad_amount=0)
            residuals = (clean_data - (s1 + s2))  # / err_map
            chi_sq = np.nansum(((residuals[nanmask == 0] * weights[nanmask == 0]) ** 2))
            return chi_sq

        log.debug(f"[STAGE B - FREEZE PRIMARY, LOCK COMPANION IN WELL]")
        log.debug(f"  dx1_stage_a={dx1_stage_a:.4f}, dy1_stage_a={dy1_stage_a:.4f}, p1_stage_a:{p1_stage_a:.2f}")
        log.debug(
            f"  dx2_stage_a={dx2_stage_a:.4f}, dy2_stage_a={dy2_stage_a:.4f}, p2_stage_a:{p2_stage_a:.2f}, c2_stage_a: {c2_stage_a:.2f}")
        log.debug(
            f"  Initial separation: {np.sqrt((dx1_stage_a - dx2_stage_a) ** 2 + (dy1_stage_a - dy2_stage_a) ** 2):.4f} px")
        log.debug(f"  Chisq at initial guess: {chisq_companion_stage(guess_comp):.4e}")

        eps_vector_comp = [1e-2, 1e-3, 1e-3]
        res_comp = minimize(chisq_companion_stage, guess_comp, method='L-BFGS-B', bounds=bounds_comp,
                            options={'eps': eps_vector_comp, 'maxiter': self.maxiter})
        success = res_comp.success
        c2_seed, dx2_seed, dy2_seed = res_comp.x

        log.debug(f"  Final res_comp.x: dx2_seed={dx2_seed:.2f}, dy2_seed={dy2_seed:.2f}, c2_seed={c2_seed:.2f}")
        log.debug(f"  Chisq at final res_comp.x: {chisq_companion_stage(res_comp.x):.4e}")
        return c2_seed, dx2_seed, dy2_seed, success

    def _companion_search_joint_relaxation(self, clean_data, nanmask, err_map, weights, imaging_psf, p1_stage_a, dx1_stage_a, dy1_stage_a, c2_seed, dx2_seed, dy2_seed, num_data_points, r1, r2):
        """
        Joint optimization of all parameters (5-parameter relaxation in Stage B).

        Minimizes chi-squared over (dx1, dy1, contrast, dx2, dy2) jointly,
        solving for f1 analytically at each iteration.

        Parameters
        ----------
        clean_data : ndarray of shape (ny, nx)
            Background-subtracted data tile.
        nanmask : ndarray of shape (ny, nx)
            Binary mask (1 for saturated, 0 for valid).
        err_map : ndarray of shape (ny, nx)
            Per-pixel uncertainties.
        weights : ndarray of shape (ny, nx)
            Mask/weights array (1 for valid, 0 for masked).
        imaging_psf : ndarray of shape (ny, nx)
            PSF template (normalized to peak=1).
        p1_stage_a, dx1_stage_a, dy1_stage_a : float
            Primary parameters from stage A (used as seed/constraint).
        c2_seed, dx2_seed, dy2_seed : float
            Companion seeds from frozen primary stage.
        num_data_points : int
            Number of valid pixels (for BIC calculation).
        r1, r2 : float
            Saturation radii of primary and companion (pixels).

        Returns
        -------
        p1_stage_b : float
            Fitted primary peak amplitude.
        dx1_stage_b, dy1_stage_b : float
            Fitted primary center offset (pixels).
        contrast_stage_b : float
            Fitted companion contrast.
        dx2_stage_b, dy2_stage_b : float
            Fitted companion center offset (pixels).
        bic_2 : float
            Bayesian Information Criterion for binary model.
            Set to np.inf if fit fails.

        Notes
        -----
        - Enforces separation constraint: min_separation <= sep <= max_separation.
        - Primary peak constrained to stay within 75%-135% of initial stage A estimate.
        - Uses L-BFGS-B optimizer with tight tolerances (ftol=1e-12).

        """
        guess_2 = [float(dx1_stage_a), float(dy1_stage_a), float(c2_seed), float(dx2_seed), float(dy2_seed)]
        bounds_2 = [
            (float(self.x_limits[0]), float(self.x_limits[1])),
            (float(self.y_limits[0]), float(self.y_limits[1])),
            (float(self.min_contrast), float(self.max_contrast)),
            (dx2_seed + self.x_limits[0], dx2_seed + self.x_limits[1]),  # Limit dx2 drift
            (dy2_seed + self.y_limits[0], dy2_seed + self.y_limits[1])  # Limit dy2 drift
        ]

        def chisq_2(params):
            dx1, dy1, contrast, dx2, dy2 = params
            sep = np.sqrt((dx1 - dx2) ** 2 + (dy1 - dy2) ** 2)

            if sep < self.min_separation or sep > self.max_separation:
                return 1e18

            p1 = self._solve_star_peak_linearly(clean_data, err_map, weights, imaging_psf,
                                               params, mode="binary",
                                               r_sat1=r1,
                                               r_sat2=r2)
            if not np.isfinite(p1):
                return 1e18
            if p1_stage_a is not None:
                if p1 > (p1_stage_a * 1.35) or p1 < (max(1e-6, p1_stage_a * 0.75)):
                    return 1e18

            s1 = p1 * ut.imshift(imaging_psf / np.nanmax(imaging_psf), [dx1, dy1], method='spline', nan_reflected=False,
                                 pad_amount=0)
            s2 = (p1 * contrast) * ut.imshift(imaging_psf / np.nanmax(imaging_psf), [dx2, dy2], method='spline',
                                              nan_reflected=False,
                                              pad_amount=0)
            residuals = (clean_data - (s1 + s2))  # / err_map
            chisq_2 = np.nansum(((residuals[nanmask == 0] * weights[nanmask == 0]) ** 2))
            return chisq_2

        eps_vector_2_pos = [1e-3, 1e-3, 1e-2, 1e-3, 1e-3]
        res_2 = minimize(chisq_2, guess_2, method='L-BFGS-B',
                         bounds=bounds_2,
                         options={'eps': eps_vector_2_pos, 'maxiter': self.maxiter, 'ftol': 1e-12})
        success = res_2.success
        dx1_stage_b, dy1_stage_b, contrast_stage_b, dx2_stage_b, dy2_stage_b = res_2.x
        p1_stage_b = self._solve_star_peak_linearly(clean_data, err_map, weights, imaging_psf,
                                                   res_2.x, mode="binary",
                                                   r_sat1=r1, r_sat2=r2)
        if not np.isfinite(p1_stage_b):
            success = False
        log.debug(f"[STAGE B - JOINT RELAXATION (5 PARAMETERS)]")
        log.debug(f"  dx1_stage_a={dx1_stage_a:.4f}, dy1_stage_a={dy1_stage_a:.4f}, p1_stage_a:{p1_stage_a:.2f}")
        log.debug(f"  dx2_seed={dx2_seed:.4f}, dy2_seed={dy2_seed:.4f}, c2_seed:{c2_seed:.2f}")
        sep_initial = np.sqrt((dx1_stage_a - dx2_seed) ** 2 + (dy1_stage_a - dy2_seed) ** 2)
        log.debug(f"  Initial separation: {sep_initial:.4f} px")
        log.debug(f"  Chisq at initial guess: {chisq_2(guess_2):.4e}")
        log.debug(f"  Final res_2.x: dx1_stage_b={dx1_stage_b:.4f}, dy1_stage_b={dy1_stage_b:.4f}, p1_stage_b: {p1_stage_b:.2f}")
        log.debug(f"  Final res_2.x: dx2_stage_b={dx2_stage_b:.4f}, dy2_stage_b={dy2_stage_b:.4f}, contrast_stage_b={contrast_stage_b:.2f}")
        log.debug(f"  Chisq at final res_2.x: {chisq_2(res_2.x):.4e}. Success: {success}")

        if not (np.isfinite(res_2.fun) and res_2.fun < 1e16):
            bic_2 = np.inf
        else:
            bic_2 = res_2.fun + len(guess_2) * np.log(num_data_points)
        return p1_stage_b, dx1_stage_b, dy1_stage_b, contrast_stage_b, dx2_stage_b, dy2_stage_b, bic_2, success

    def fitpsf(self, tile_with_nans, nanmask, err_map, imaging_psf):
        """
        Fit the tile for a primary source and optionally a companion.

        Both can be saturated or non-saturated. Implements a two-stage algorithm:
        (1) single-source model, (2) companion search (early residual + stages A & B),
        and (3) model selection via BIC with configurable gates.

        Parameters
        ----------
        tile_with_nans : ndarray of shape (ny, nx)
            Input tile; NaNs mark saturated or invalid pixels.
        nanmask : ndarray of shape (ny, nx) with dtype int
            Binary mask with 1 for masked/saturated/invalid pixels and
            0 for valid pixels.
        err_map : ndarray of shape (ny, nx)
            Per-pixel uncertainties (standard deviations) used for chi-square
            computations and weighting.
        imaging_psf : ndarray of shape (ny, nx)
            PSF stamp used as the fitting template. Should be normalized to
            peak=1 for consistency.

        Returns
        -------
        None
            Results are saved to instance attributes: `peak1`, `dx1`, `dy1`,
            `peak2`, `dx2`, `dy2`, and `bintest`.

        Attributes Set
        ---------------
        peak1 : float
            Peak amplitude of the primary (or only) fitted source.
        dx1, dy1 : float
            Position offset of primary source relative to tile center (pixels).
        peak2 : float
            Peak amplitude of companion (0.0 if no companion detected).
        dx2, dy2 : float or None
            Position offset of companion relative to tile center (pixels).
            Set to None if no companion detected.
        bintest : bool
            True if binary model was selected; False for single-source model.

        Algorithm Overview
        -------------------
        1. **Stage A (Single-Source)**:
           - Fit single-source model, optimizing position (dx1, dy1).
           - Solve peak amplitude (f1) analytically.

        2. **Companion Search**:
           - Subtract fitted primary model from data.
           - Search residuals for companion candidate (early search).
           - For each candidate: estimate position and peak via centroid.

        3. **Stage B (Companion Optimization)**:
           - **Frozen Primary**: Optimize companion (contrast, dx2, dy2) with
             primary held fixed.
           - **Joint Relaxation**: Optimize all 5 parameters (dx1, dy1, contrast,
             dx2, dy2) jointly while solving f1 analytically.

        4. **Model Selection**:
           - Compute BIC for single and binary models.
           - Accept binary if delta_BIC >= 10 and contrast >= min_contrast.
           - Ensure separation is within [min_separation, max_separation].

        Notes
        -----
        - Saturated sources (if r_sat > 0) are handled via wing-matching instead
          of standard linear least-squares.
        - Results attributes are always set (peak2 = 0.0, dx2/dy2 = None for
          single-source models).
        - Diagnostic plots (if showplot=True) display fitted positions overlaid
          on the input tile.
        - Debug logging provides detailed information on each stage if debug=True.

        """
        # TODO: implement a while loop to search for additional companions past the first until non is found.

        if self.debug:
            log.setLevel(logging.DEBUG)
        else:
            log.setLevel(logging.INFO)

        clean_data = np.nan_to_num(tile_with_nans, nan=0.0) - self.background
        weights = 1.0 - nanmask
        num_data_points = np.sum(weights)

        p1_guess, dx1_guess, dy1_guess, r1, p2_guess, c2_guess, dx2_guess, dy2_guess, r2, labeled, mask_bool_second_bests = self._make_educated_guesses(tile_with_nans, clean_data, err_map, weights, imaging_psf, nanmask)

        # -----------------------------------------------------------------
        # STAGE A - ONE SOURCE MODEL (Optimize only dx1, dy1)
        # -----------------------------------------------------------------
        p1_stage_a, dx1_stage_a, dy1_stage_a, bic_1, success_a = self._one_source_model(clean_data, nanmask, err_map, weights, imaging_psf, p1_guess, dx1_guess, dy1_guess, num_data_points, r1)

        # -----------------------------------------------------------------
        #  STAGE A - COMPANION CENTROID SEARCH (if guesses not provided)
        # -----------------------------------------------------------------
        if (dx2_guess is None or dy2_guess is None):
            p2_stage_a, c2_stage_a, dx2_stage_a, dy2_stage_a = self._companion_centroid_search(clean_data, nanmask, err_map, weights, imaging_psf ,p1_stage_a, dx1_stage_a, dy1_stage_a, r1, r2, labeled, mask_bool_second_bests)
        else:
            p2_stage_a, c2_stage_a, dx2_stage_a, dy2_stage_a = p2_guess, c2_guess, dx2_guess, dy2_guess

        if dx2_stage_a is not None and dy2_stage_a is not None:
            # -----------------------------------------------------------------
            # STAGE B - FREEZE PRIMARY, LOCK COMPANION IN WELL
            # -----------------------------------------------------------------
            c2_seed, dx2_seed, dy2_seed, _ = self._companion_search_frozen_primary(clean_data, nanmask, err_map, weights, imaging_psf, p1_stage_a, dx1_stage_a, dy1_stage_a, p2_stage_a, c2_stage_a, dx2_stage_a, dy2_stage_a)

            # -----------------------------------------------------------------
            # STAGE B - JOINT RELAXATION (5 PARAMETERS)
            # -----------------------------------------------------------------
            p1_stage_b, dx1_stage_b, dy1_stage_b, contrast_stage_b, dx2_stage_b, dy2_stage_b, bic_2, success_b = self._companion_search_joint_relaxation(clean_data, nanmask, err_map, weights, imaging_psf, p1_stage_a, dx1_stage_a, dy1_stage_a, c2_seed, dx2_seed, dy2_seed, num_data_points, r1, r2)
        else:
            bic_2 = np.inf

        # -----------------------------------------------------------------
        # MODEL SELECTION & INTEGRATED SELF-SORTING GATE
        # -----------------------------------------------------------------
        delta_bic = bic_1 - bic_2
        if delta_bic >= self.bic_gate and contrast_stage_b >= self.min_contrast and np.isfinite(bic_2) and success_b:
            self.bintest = True
            self.success = success_b
            p1, dx1, dy1, contrast, dx2, dy2 = p1_stage_b, dx1_stage_b, dy1_stage_b, contrast_stage_b, dx2_stage_b, dy2_stage_b
            p2 = p1 * contrast

            dist1 = np.sqrt(dx1 ** 2 + dy1 ** 2)
            dist2 = np.sqrt(dx2 ** 2 + dy2 ** 2)

            if dist1 > dist2:
                self.peak1, self.dx1, self.dy1 = p2, dx2, dy2
                self.peak2, self.dx2, self.dy2 = p1, dx1, dy1
            else:
                self.peak1, self.dx1, self.dy1 = p1, dx1, dy1
                self.peak2, self.dx2, self.dy2 = p2, dx2, dy2
            log.info(f"Binary-source model accepted: delta BIC={delta_bic:.2f}, (dx1,dy1)=({self.dx1:.4f},{self.dy1:.4f}), peak1: {self.peak1:.2f} |  (dx2,dy2)=({self.dx2:.4f},{self.dy2:.4f}), peak2: {self.peak2:.2f}")

        else:
            self.bintest = False
            self.success = success_a
            self.peak1, self.dx1, self.dy1 = p1_stage_a, dx1_stage_a, dy1_stage_a
            self.peak2, self.dx2, self.dy2 = 0.0, None, None
            log.info(f"Single-source model accepted: delta BIC={delta_bic:.2f}, (dx,dy)=({self.dx1:.4f},{self.dy1:.4f}), peak: {self.peak1:.2f}")

        if self.showplot or self.debug:
            self._plot_final_fit(tile_with_nans.copy())
        pass
