import logging,os
from pathlib import Path
from typing import Literal
import matplotlib.pylab as plt
from astropy.visualization import simple_norm
import spaceKLIP.utils as ut
from astropy.table import Table, vstack
from astropy.wcs import WCS
from photutils.detection import DAOStarFinder,StarFinder
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from astropy import units as u
from astropy.coordinates import SkyCoord
import requests
from skimage.measure import label, regionprops
from scipy.ndimage import binary_dilation
from photutils.aperture import CircularAperture, aperture_photometry
from scipy.spatial import KDTree
from skimage.color import label2rgb
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from scipy.ndimage import shift
import astropy.io.fits as pyfits

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
                    showplots=False,
                    cmap='Greys_r',
                    stretch='linear',
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
        dx, dy = shifts[0], shifts[1]

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

    if showplots:
        norm = simple_norm(tile, stretch)
        plt.imshow(tile, origin='lower', norm=norm, cmap=cmap)
        plt.plot(tile.shape[1] // 2, tile.shape[0] // 2, 'xr', label='Target Center')
        plt.colorbar()
        plt.title(f'Extracted Star (Shift: {dx:.3f}, {dy:.3f})')
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

class FITPSF():

    def __init__(self,
                 min_separation=1,
                 max_separation=5.0,
                 x_limits=(-1, 1),
                 y_limits=(-1, 1),
                 coarse_max_jitter = 1.5,
                 coarse_step = 0.5,
                 fine_step = 0.1,
                 fine_radius = 0.6,
                 clip_flux_min = 1e-6,
                 top_k = 5,
                 eps=1e-12,
                 min_companion_flux_frac=0.1,
                 min_companion_abs_frac_of_initial=1e-3,
                 min_companion_snr=4.0
                 ):
        self.dx1 = None
        self.dy1 = None
        self.dx2 = None
        self.dy2 = None
        self.flux1 = None
        self.flux2 = None
        self.bintest = False
        self.sep=None
        self.chi2_binary_sat = np.inf
        self.bic_binary_sats = np.inf
        self.chi2_binary = np.inf
        self.bic_binary = np.inf
        self.chi2_single_sat = np.inf
        self.bic_single_sat =np.inf
        self.chi2_single = np.inf
        self.bic_single =np.inf
        self.min_separation = min_separation
        self.max_separation = max_separation
        self.x_limits = x_limits
        self.y_limits = y_limits
        self.coarse_max_jitter = coarse_max_jitter
        self.coarse_step = coarse_step
        self.fine_step = fine_step
        self.fine_radius = fine_radius
        self.clip_flux_min = clip_flux_min
        self.top_k = top_k
        self.eps = eps

        # minimum *relative* companion flux fraction (primary is index 1).
        self.min_companion_flux_frac = min_companion_flux_frac
        # minimum *absolute* flux for any star relative to initial_flux (e.g. 0.001 x initial_flux)
        self.min_companion_abs_frac_of_initial = min_companion_abs_frac_of_initial
        # minimum SNR for companion acceptance
        self.min_companion_snr = min_companion_snr

        pass

    def find_multiple_saturated_cores(self,data,labeled_nan):
        center_x, center_y = float(self.centers[0]), float(self.centers[1])
        # Tunable parameters:
        dilation_iters = 4  # how far to dilate the NaN core to sample PSF wings (3-5 typical)
        # Build info tuples: (distance_to_center, prop, area, perimeter_sum)
        props_info = []
        # Precompute finite mask for background estimation
        finite_all = np.isfinite(data)
        any_finite = np.any(finite_all)
        for p in self.nan_props:
            py, px = p.centroid  # regionprops centroid is (row=y, col=x)
            dx = float(px) - center_x
            dy = float(py) - center_y
            r = float(np.hypot(dx, dy))
            mask = (labeled_nan == p.label)

            # Dilate the mask to create an annulus region that samples PSF wings
            dilated = binary_dilation(mask, iterations=int(dilation_iters))
            perimeter_mask = dilated & (~mask)

            # Extract perimeter pixel values
            perim_vals = np.asarray(data[perimeter_mask], dtype=float)
            if perim_vals.size == 0:
                perim_sum = 0.0
            else:
                # Estimate a local background from pixels outside the dilated region,
                # but inside the cutout and finite. Fallback to global finite median if needed.
                bg_mask = (~dilated) & finite_all
                if np.any(bg_mask):
                    bkg = float(np.nanmedian(data[bg_mask]))
                elif any_finite:
                    bkg = float(np.nanmedian(data[finite_all]))
                else:
                    bkg = 0.0
                # Compute positive, background-subtracted perimeter sum
                perim_vals[~np.isfinite(perim_vals)] = 0.0
                perim_sum = float(np.nansum(np.where(perim_vals > bkg, perim_vals - bkg, 0.0)))

            props_info.append((r, p, int(p.area), perim_sum))

        # Prefer regions that lie within max_separation (close to center),
        # and rank them by perimeter brightness (descending), then by distance (ascending).
        candidates = [t for t in props_info if t[0] < float(self.max_separation)]
        candidates.sort(key=lambda t: (-t[3], t[0]))  # high perim_sum first, nearer distance tiebreak

        if len(candidates) >= 2:
            self.selected = [candidates[0][1], candidates[1][1]]
            log.debug("selected two NaN cores by perimeter brightness within max_separation")
        elif len(candidates) == 1:
            # One good candidate inside radius: pick it and the next-best overall (by perim_sum)
            self.selected = [candidates[0][1]]
            others = [t for t in props_info if t[1].label != self.selected[0].label]
            if others:
                others.sort(key=lambda t: (-t[3], t[0]))
                self.selected.append(others[0][1])
                log.debug("one candidate inside max_separation; selected second-best overall by perim_sum")
            else:
                log.debug("only one NaN region present and selected by perim_sum")
        else:
            # No candidates inside max_separation: fallback to top two by perimeter brightness overall
            props_info.sort(key=lambda t: (-t[3], t[0]))
            if len(props_info) >= 2:
                self.selected = [props_info[0][1], props_info[1][1]]
                log.info(
                    "no cores within max_separation; selected top two by perimeter brightness overall")
            elif len(props_info) == 1:
                self.selected = [props_info[0][1]]
                log.info("only one NaN region found; selected the single region")
            else:
                self.selected = []
                log.info("no NaN regions available after scoring; skipping saturated-binary")

    def fit_multiple_saturated_cores(self,data,psf,nanmask):
        p1y, p1x = self.selected[0].centroid
        p2y, p2x = self.selected[1].centroid

        dx1_s = float(p1x - self.centers[0])
        dy1_s = float(p1y - self.centers[1])
        dx2_s = float(p2x - self.centers[0])
        dy2_s = float(p2y - self.centers[1])

        # Build finite-pixel vector for linear solves (exclude masked/NaN pixels)
        finite_mask = ~nanmask.astype(bool)
        mask_inds = np.nonzero(finite_mask)
        data_vec = data[finite_mask].ravel()
        n = int(max(1, data_vec.size))

        if data_vec.size == 0:
            log.warning("no finite pixels available -> skipping saturated-binary.")
            return
        else:
            psf_for_shift = psf.copy()
            sp_cache = {}

            def shifted_psf_cached(dx, dy):
                key = (float(dx), float(dy))
                if key not in sp_cache:
                    sp_cache[key] = shift(psf_for_shift, [dy, dx], order=3, mode='constant', cval=0.0)
                return sp_cache[key]

            # Build jitter arrays
            coarse_vals = np.arange(-self.coarse_max_jitter, self.coarse_max_jitter + 1e-12, self.coarse_step)

            # helper: solve for fluxes with non-negative constraint (prefer lsq_linear if available)
            from scipy.optimize import lsq_linear
            def solve_fluxes_nnls(A, b, clip_min=self.clip_flux_min):
                try:
                    res = lsq_linear(A, b, bounds=(clip_min, np.inf), lsmr_tol='auto', verbose=0)
                    if res.success:
                        return res.x, res.cost * 2.0
                    else:
                        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                        sol = np.clip(np.asarray(sol, dtype=float), clip_min, None)
                        resid = b - A.dot(sol)
                        return sol, float(np.sum(resid * resid))
                except Exception:
                    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                    sol = np.clip(np.asarray(sol, dtype=float), clip_min, None)
                    resid = b - A.dot(sol)
                    return sol, float(np.sum(resid * resid))

            # Coarse grid search: record top candidates
            coarse_candidates = []
            for dx1_j in coarse_vals:
                for dy1_j in coarse_vals:
                    dx1_cand = dx1_s + float(dx1_j)
                    dy1_cand = dy1_s + float(dy1_j)
                    p1_full = shifted_psf_cached(dx1_cand, dy1_cand)
                    p1 = p1_full[mask_inds].ravel()
                    if np.allclose(p1, 0.0):
                        continue
                    for dx2_j in coarse_vals:
                        for dy2_j in coarse_vals:
                            dx2_cand = dx2_s + float(dx2_j)
                            dy2_cand = dy2_s + float(dy2_j)
                            sep = np.hypot(dx1_cand - dx2_cand, dy1_cand - dy2_cand)
                            if sep < self.min_separation or sep > self.max_separation:
                                continue
                            p2_full = shifted_psf_cached(dx2_cand, dy2_cand)
                            p2 = p2_full[mask_inds].ravel()
                            A = np.vstack([p1, p2]).T
                            if np.linalg.matrix_rank(A) < 2:
                                continue
                            # Solve for fluxes with NNLS/bounded LS
                            # sol, chi2 = solve_fluxes_nnls(A, data_vec, clip_min=self.clip_flux_min)
                            # dynamic clip floor: ensure companion >= absolute floor based on initial_flux
                            clip_min_candidate = max(self.clip_flux_min,self.initial_flux * self.min_companion_abs_frac_of_initial)
                            sol, chi2 = solve_fluxes_nnls(A, data_vec, clip_min=clip_min_candidate)
                            # keep top K by chi2 (smallest)
                            coarse_candidates.append((chi2, dx1_cand, dy1_cand, dx2_cand, dy2_cand, sol[0], sol[1]))

            # If none found, fallback
            if len(coarse_candidates) == 0:
                log.warning("Saturated-binary grid-search found no valid candidate (coarse stage).")
                return
            else:
                # sort by chi2 and keep top_k
                coarse_candidates.sort(key=lambda x: x[0])
                coarse_candidates = coarse_candidates[:max(1, min(self.top_k, len(coarse_candidates)))]

                # Stage 2: refine each coarse candidate with a fine local grid
                best = {'chi2': np.inf, 'dx1': None, 'dy1': None, 'dx2': None, 'dy2': None, 'f1': None, 'f2': None}
                for (chi2_c, dx1_c, dy1_c, dx2_c, dy2_c, f1_c, f2_c) in coarse_candidates:
                    dx1_ref_vals = np.arange(dx1_c - self.fine_radius, dx1_c + self.fine_radius + 1e-12, self.fine_step)
                    dy1_ref_vals = np.arange(dy1_c - self.fine_radius, dy1_c + self.fine_radius + 1e-12, self.fine_step)
                    dx2_ref_vals = np.arange(dx2_c - self.fine_radius, dx2_c + self.fine_radius + 1e-12, self.fine_step)
                    dy2_ref_vals = np.arange(dy2_c - self.fine_radius, dy2_c + self.fine_radius + 1e-12, self.fine_step)

                    for dx1_f in dx1_ref_vals:
                        for dy1_f in dy1_ref_vals:
                            p1_full = shifted_psf_cached(dx1_f, dy1_f)
                            p1 = p1_full[mask_inds].ravel()
                            if np.allclose(p1, 0.0):
                                continue
                            for dx2_f in dx2_ref_vals:
                                for dy2_f in dy2_ref_vals:
                                    sep = np.hypot(dx1_f - dx2_f, dy1_f - dy2_f)
                                    if sep < self.min_separation or sep > self.max_separation:
                                        continue
                                    p2_full = shifted_psf_cached(dx2_f, dy2_f)
                                    p2 = p2_full[mask_inds].ravel()
                                    A = np.vstack([p1, p2]).T
                                    if np.linalg.matrix_rank(A) < 2:
                                        continue
                                    # sol, chi2_local = solve_fluxes_nnls(A, data_vec, clip_min=self.clip_flux_min)
                                    # dynamic clip floor: ensure companion >= absolute floor based on initial_flux
                                    clip_min_candidate = max(self.clip_flux_min,self.initial_flux * self.min_companion_abs_frac_of_initial)
                                    sol, chi2_local = solve_fluxes_nnls(A, data_vec, clip_min=clip_min_candidate)
                                    if chi2_local < best['chi2']:
                                        best.update({
                                            'chi2': float(chi2_local),
                                            'dx1': float(dx1_f), 'dy1': float(dy1_f),
                                            'dx2': float(dx2_f), 'dy2': float(dy2_f),
                                            'f1': float(sol[0]), 'f2': float(sol[1]),
                                            'sep': float(sep),
                                        })

                if best['dx1'] is None:
                    log.warning("Saturated-binary refinement found no valid candidate.")
                    return
                else:
                    self.dx1 = best['dx1']
                    self.dy1 = best['dy1']
                    self.dx2 = best['dx2']
                    self.dy2 = best['dy2']
                    self.flux1 = best['f1']
                    self.flux2 = best['f2']
                    self.sep = best['sep']
                    # self.chi2_binary = float(best['chi2'])
                    self.chi2_binary = float(best['chi2'])/(self.sigma**2)
                    self.bic_binary_sat = n * np.log(max(self.eps, self.chi2_binary / n)) + 6.0 * np.log(max(1, n))
                    log.info("Saturated-binary candidate preferred: ΔBIC=%.2f, sep=%.2f, f1=%.3f, f2=%.3f",
                             self.bic_binary_sat, self.sep, self.flux1, self.flux2)
                    self.bintest = True

    def fit_multiple_unsaturated_cores(self, data, psf, nanmask,weights):
        # since we have no saturated stars, we should fit for 2 not saturated stars or one.
        # perform the fit for 1 not saturated star. I already have it and is working, nothing to add here.
        def single_objective(params):
            dx, dy, flux = params
            model = flux * shift(psf, [dy, dx], order=3)
            residuals = (data[~nanmask.astype(bool)] - model[~nanmask.astype(bool)]) * weights[
                ~nanmask.astype(bool)]
            return np.sum(residuals ** 2)

        single_guess = [0.0, 0.0, self.initial_flux]
        single_bounds = [self.x_limits, self.y_limits, (self.initial_flux * 1e-1, self.initial_flux * 1e2)]

        res_single = minimize(single_objective, single_guess, bounds=single_bounds, method='L-BFGS-B')
        b_dx1, b_dy1, b_flux1 = res_single.x
        # self.chi2_single = res_single.fun
        self.chi2_single = res_single.fun/(self.sigma**2)
        self.bic_single = self.chi2_single + 3 * np.log(self.n_pixels)

        # perform the fit for 2 not saturated stars.
        # =========================================================================
        # MODEL 2: SIMULTANEOUS BINARY FIT (Parameterised with Contrast)
        # =========================================================================
        # --- Binary fit: polar reparameterization + L-BFGS-B (minimal change) ---
        # Params: [dx1, dy1, flux1, r, theta, contrast]
        # Companion x,y are computed as dx2 = dx1 + r*cos(theta), dy2 = dy1 + r*sin(theta)
        def binary_objective(params):
            dx1, dy1, flux1, r, theta, contrast = params
            flux2 = flux1 * contrast
            dx2 = dx1 + r * np.cos(theta)
            dy2 = dy1 + r * np.sin(theta)

            model = (flux1 * shift(psf, [dy1, dx1], order=3) +
                     flux2 * shift(psf, [dy2, dx2], order=3))
            residuals = (data[~nanmask.astype(bool)] - model[~nanmask.astype(bool)]) * weights[
                ~nanmask.astype(bool)]
            return float(np.sum(residuals ** 2))

        # bounds for polar parameters:
        # dx1,dy1 stay within x_limits/y_limits, flux1 positive; r in [min_separation, self.max_separation]; theta in [0, 2*pi]; contrast in (0.01, 1.0)
        binary_bounds = [
            self.x_limits, self.y_limits, (self.initial_flux * 1e-1, self.initial_flux * 1e2),  # dx1, dy1, flux1
            (max(self.min_separation, 0.0), self.max_separation),  # r
            (0.0, 2.0 * np.pi),  # theta
            (0.01, 1.0)  # contrast
        ]

        # sensible initial guess: place primary near center and companion at radius ~1.3*min_separation (or small default)
        prim_dx0 = np.clip(0.0, self.x_limits[0], self.x_limits[1])
        prim_dy0 = np.clip(0.0, self.y_limits[0], self.y_limits[1])
        prim_flux0 = self.initial_flux * 0.9
        r0 = np.clip(self.min_separation * 1.3 if self.min_separation > 0 else min(self.max_separation, 1.0),
                     max(self.min_separation, 0.0), self.max_separation)
        theta0 = 0.0
        initial_contrast_guess = 0.2

        binary_guess = [prim_dx0, prim_dy0, prim_flux0, r0, theta0, initial_contrast_guess]

        # Use L-BFGS-B (no nonlinear constraints needed; r bounded enforces separation)
        res_binary = minimize(
            binary_objective,
            binary_guess,
            bounds=binary_bounds,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-9}
        )

        # extract parameters
        if res_binary is None or not hasattr(res_binary, 'x'):
            # fallback: keep single solution
            self.chi2_binary = np.inf
            self.bic_binary = np.inf
            b_dx2 = b_dy2 = b_flux2 = None
        else:
            dx1_fit, dy1_fit, flux1_fit, r_fit, theta_fit, contrast_fit = res_binary.x
            dx2_fit = dx1_fit + r_fit * np.cos(theta_fit)
            dy2_fit = dy1_fit + r_fit * np.sin(theta_fit)
            flux2_fit = flux1_fit * contrast_fit

            # compute chi2 (weighted residuals) for BIC
            model = (flux1_fit * shift(psf, [dy1_fit, dx1_fit], order=3) +
                     flux2_fit * shift(psf, [dy2_fit, dx2_fit], order=3))
            residuals = (data[~nanmask.astype(bool)] - model[~nanmask.astype(bool)]) * weights[
                ~nanmask.astype(bool)]
            # self.chi2_binary = float(np.sum(residuals ** 2))
            self.chi2_binary = float(np.sum(residuals ** 2))/(self.sigma**2)
            self.bic_binary = self.chi2_binary + 6 * np.log(self.n_pixels)

            # assign outputs
            b_dx1, b_dy1, b_flux1 = dx1_fit, dy1_fit, flux1_fit
            b_dx2, b_dy2, b_flux2 = dx2_fit, dy2_fit, flux2_fit

        self.delta_bic = self.bic_single - self.bic_binary
        # Calculate final sorted physical separation
        self.sep = np.sqrt((b_dx1 - b_dx2) ** 2 + (b_dy1 - b_dy2) ** 2)

        # =========================================================================
        # 3. STATISTICAL COMPARISON WITH FLUX RATIO THRESHOLD
        # =========================================================================
        if self.delta_bic > 10.0 and b_flux2 > (0.01 * b_flux1):
            if self.sep < self.min_separation or not res_binary.success:
                if round(self.sep, 1) < self.min_separation:
                    log.error(f"Binary fit collapsed: separation {self.sep} < {self.min_separation:.2f} pix)")
                if not res_binary.success:
                    log.error(f"Single Star Detected. (Binary fit rejected: optimization failed to converge)")
                return b_dx1, b_dy1, b_flux1, None, None, None, False

            # Enforce that Star 1 is ALWAYS the brighter "Primary" star
            if b_flux2 > b_flux1:
                b_dx1, b_dx2 = b_dx2, b_dx1
                b_dy1, b_dy2 = b_dy2, b_dy1
                b_flux1, b_flux2 = b_flux2, b_flux1
            self.dx1 = b_dx1
            self.dy1 = b_dy1
            self.dx2 = b_dx2
            self.dy2 = b_dy2
            self.flux1 = b_flux1
            self.flux2 = b_flux2
            self.bintest = True
            log.info("Binary preferred : ΔBIC=%.2f, sep=%.2f, f1=%.3f, f2=%.3f",
                     self.delta_bic, self.sep, self.flux1, self.flux2)

        else:
            self.dx1 = b_dx1
            self.dy1 = b_dy1
            self.dx2 = None
            self.dy2 = None
            self.flux1 = b_flux1
            self.flux2 = None
            self.bintest = False
            log.info("Single preferred: ΔBIC=%.3f, f1=%.3f",
                     self.delta_bic, self.flux1)

    def fit_single_saturated_core(self, data, psf, nanmask):
        """
        Two-stage approach for the case of a single saturated core with extra checks to avoid
        spurious tiny-separation binaries:
          - Stage A: coarse binary NNLS grid to find candidates
          - Stage B: single-star refinement (coarse+fine)
          - Stage C: refine top coarse binary candidates; accept only if BIC improvement + additional checks
        """
        # centroid seed of saturated core
        p1y, p1x = self.selected[0].centroid
        dx1_s = float(p1x - self.centers[0])
        dy1_s = float(p1y - self.centers[1])

        finite_mask = ~nanmask.astype(bool)
        mask_inds = np.nonzero(finite_mask)
        data_vec = data[finite_mask].ravel()
        n = int(max(1, data_vec.size))

        if data_vec.size == 0:
            log.info("no finite pixels available -> skipping single saturated fit.")
            return

        # ---- helpers: shifted PSF cache and NNLS solver (same as other routines) ----
        psf_for_shift = psf.copy()
        sp_cache = {}

        def shifted_psf_cached(dx, dy):
            key = (float(dx), float(dy))
            if key not in sp_cache:
                sp_cache[key] = shift(psf_for_shift, [dy, dx], order=3, mode='constant', cval=0.0)
            return sp_cache[key]

        from scipy.optimize import lsq_linear

        def solve_fluxes_nnls(A, b, clip_min=self.clip_flux_min):
            try:
                res = lsq_linear(A, b, bounds=(clip_min, np.inf), lsmr_tol='auto', verbose=0)
                if res.success:
                    return res.x, res.cost * 2.0
                else:
                    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                    sol = np.clip(np.asarray(sol, dtype=float), clip_min, None)
                    resid = b - A.dot(sol)
                    return sol, float(np.sum(resid * resid))
            except Exception:
                sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                sol = np.clip(np.asarray(sol, dtype=float), clip_min, None)
                resid = b - A.dot(sol)
                return sol, float(np.sum(resid * resid))

        # -------------------------------------------------------------------------
        # Determine empirical core radius from NaN region (avoid companions inside core)
        # -------------------------------------------------------------------------
        try:
            core_area = float(self.selected[0].area)
            core_radius_emp = np.sqrt(core_area / np.pi)  # pixels
        except Exception:
            core_radius_emp = 0.0
        # enforce minimum sensible radius
        min_sep_enforced = max(self.min_separation, 1.0, 1.2 * core_radius_emp)

        # condition-number threshold for A matrix (avoid near-collinear PSF columns)
        cond_thresh = 1e3

        def single_objective(params):
            dx, dy, flux = params
            # Build model at this shift and flux using cached shift helper
            model_full = flux * shifted_psf_cached(dx, dy)
            # residuals evaluated only on finite/masked pixels (mask_inds)
            resid = data_vec - model_full[mask_inds].ravel()
            return float(np.sum(resid ** 2))

        # sensible initial guess: use centroid-derived seed and initial_flux
        single_guess = [dx1_s, dy1_s, max(1e-8, float(self.initial_flux))]

        # bounds: keep dx/dy in x/y limits, flux positive and within a wide sensible range
        flux_min_bound = max(self.clip_flux_min, self.initial_flux * 1e-3)
        flux_max_bound = self.initial_flux * 1e4
        single_bounds = [self.x_limits, self.y_limits, (flux_min_bound, flux_max_bound)]

        # run continuous optimizer (L-BFGS-B). Fall back gracefully if it fails.
        try:
            from scipy.optimize import minimize
            res_single = minimize(single_objective, single_guess, bounds=single_bounds, method='L-BFGS-B',
                                  options={'maxiter': 2000, 'ftol': 1e-9})
            if res_single is not None and hasattr(res_single, 'x'):
                b_dx = float(res_single.x[0])
                b_dy = float(res_single.x[1])
                b_flux = float(res_single.x[2])
                raw_chi2_single = float(res_single.fun)
            else:
                # optimizer did not return a valid result: fallback to seed
                b_dx, b_dy, b_flux = float(dx1_s), float(dy1_s), float(self.initial_flux)
                raw_chi2_single = np.inf
        except Exception as e:
            log.warning("single-star minimization raised exception: %s -- falling back to seed.", str(e))
            b_dx, b_dy, b_flux = float(dx1_s), float(dy1_s), float(self.initial_flux)
            raw_chi2_single = np.inf

        # Assign the robust single-star baseline
        self.dx1 = float(b_dx)
        self.dy1 = float(b_dy)
        self.dx2 = None
        self.dy2 = None
        self.flux1 = float(b_flux)
        self.flux2 = None
        self.bintest = False

        # Follow existing convention: chi2_single_sat is normalized by sigma**2 later used for BIC.
        # sigma variable was computed at function start (sigma = np.nanstd(data[~nanmask])).
        self.chi2_single_sat = float(raw_chi2_single) / (self.sigma ** 2) if np.isfinite(raw_chi2_single) else np.inf
        self.bic_single_sat = n * np.log(max(self.eps, self.chi2_single_sat / n)) + 3.0 * np.log(max(1, n))
        log.debug(
            "saturated-single candidate (minimized): chi2_raw=%.3e chi2_norm=%.3e, n=%d, BIC_sats=%.3f, dx=%.3f dy=%.3f f=%.3f",
            raw_chi2_single, self.chi2_single_sat, n, self.bic_single_sat, self.dx1, self.dy1, self.flux1)

        # -------------------------------------------------------------------------
        # STAGE A: COARSE BINARY GRID SEARCH (fast NNLS solves) to find promising candidates
        # -------------------------------------------------------------------------
        coarse_dxdy = max(0.5, self.coarse_step)  # coarse dx/dy grid spacing
        primary_jitter = min(self.coarse_max_jitter, 1.5)  # jitter around detected saturated centroid
        dx_primary_grid = np.arange(dx1_s - primary_jitter, dx1_s + primary_jitter + 1e-12, coarse_dxdy)
        dy_primary_grid = np.arange(dy1_s - primary_jitter, dy1_s + primary_jitter + 1e-12, coarse_dxdy)
        r_step_coarse = max(0.5, self.coarse_step)
        theta_step_coarse = np.deg2rad(20.0)
        r_vals = np.arange(max(self.min_separation, 0.0), self.max_separation + 1e-12, r_step_coarse)
        theta_vals = np.arange(0.0, 2.0 * np.pi, theta_step_coarse)

        coarse_candidates = []  # (chi2, bic, dx1,dy1,dx2,dy2,f1,f2,r,theta,sep,cond)

        center_x = float(self.centers[0])
        center_y = float(self.centers[1])
        ny, nx = nanmask.shape[0], nanmask.shape[1]

        for dx1_c in dx_primary_grid:
            for dy1_c in dy_primary_grid:
                p1_full = shifted_psf_cached(dx1_c, dy1_c)
                p1 = p1_full[mask_inds].ravel()
                if np.allclose(p1, 0.0):
                    continue
                for r_c in r_vals:
                    for th in theta_vals:
                        dx2_c = dx1_c + r_c * np.cos(th)
                        dy2_c = dy1_c + r_c * np.sin(th)
                        # enforce minimal separation (avoid the tiny separation collapse)
                        sep = np.hypot(dx1_c - dx2_c, dy1_c - dy2_c)
                        if sep < min_sep_enforced or sep < self.min_separation:
                            continue
                        # ensure companion center lies on finite data (not inside NaN core)
                        x2_pix = int(round(center_x + dx2_c))
                        y2_pix = int(round(center_y + dy2_c))
                        if x2_pix < 0 or x2_pix >= nx or y2_pix < 0 or y2_pix >= ny:
                            continue
                        if not finite_mask[y2_pix, x2_pix]:
                            # candidate falls on NaN region (or outside) -> not constrained, skip
                            continue
                        p2_full = shifted_psf_cached(dx2_c, dy2_c)
                        p2 = p2_full[mask_inds].ravel()
                        A = np.vstack([p1, p2]).T
                        # check linear independence / condition number
                        try:
                            condA = np.linalg.cond(A)
                        except Exception:
                            condA = np.inf
                        if condA > cond_thresh or np.linalg.matrix_rank(A) < 2:
                            continue
                        # sol, chi2_local = solve_fluxes_nnls(A, data_vec, clip_min=self.clip_flux_min)
                        # dynamic clip floor: ensure companion >= absolute floor based on initial_flux
                        clip_min_candidate = max(self.clip_flux_min,self.initial_flux * self.min_companion_abs_frac_of_initial)
                        sol, chi2_local = solve_fluxes_nnls(A, data_vec, clip_min=clip_min_candidate)
                        bic_local = n * np.log(max(self.eps, chi2_local / n)) + 6.0 * np.log(max(1, n))
                        coarse_candidates.append(
                            (chi2_local, bic_local, dx1_c, dy1_c, dx2_c, dy2_c, float(sol[0]), float(sol[1]), r_c, th,
                             sep, condA))

        # Keep top-K coarse candidates by chi2
        coarse_candidates.sort(key=lambda t: t[0])
        top_k = max(1, min(self.top_k, len(coarse_candidates)))
        top_candidates = coarse_candidates[:top_k] if len(coarse_candidates) > 0 else []

        if len(top_candidates) > 0:
            for idx, c in enumerate(top_candidates):
                chi2_c, bic_c, dx1_c, dy1_c, dx2_c, dy2_c, f1_c, f2_c, r_c, th, sep, condA = c
                log.debug("coarse cand %d: chi2=%.3e bic=%.3f r=%.2f f1=%.3f f2=%.3f sep=%.3f cond=%.1e",
                         idx, chi2_c, bic_c, r_c, f1_c, f2_c, sep, condA)
        else:
            log.warning("no coarse binary candidate found around saturated core.")
            return
        # -------------------------------------------------------------------------
        # STAGE C: Refine top coarse candidates (if any) with local fine search and accept only if robust
        # -------------------------------------------------------------------------
        if len(top_candidates) == 0:
            log.warning("no valid candidates found. Quitting.")
            return

        best_binary = {'chi2': np.inf, 'dx1': None, 'dy1': None, 'dx2': None, 'dy2': None, 'f1': None, 'f2': None,
                       'sep': None}
        for cand in top_candidates:
            _, bic_c, dx1_c, dy1_c, dx2_c, dy2_c, f1_c, f2_c, r_c, th_c, sep_c, cond_c = cand

            # skip coarse candidate if it fails minimal separation or cond checks (extra safety)
            if sep_c < min_sep_enforced or cond_c > cond_thresh:
                continue

            dx1_ref_vals = np.arange(dx1_c - self.fine_radius, dx1_c + self.fine_radius + 1e-12, self.fine_step)
            dy1_ref_vals = np.arange(dy1_c - self.fine_radius, dy1_c + self.fine_radius + 1e-12, self.fine_step)
            dx2_ref_vals = np.arange(dx2_c - self.fine_radius, dx2_c + self.fine_radius + 1e-12, self.fine_step)
            dy2_ref_vals = np.arange(dy2_c - self.fine_radius, dy2_c + self.fine_radius + 1e-12, self.fine_step)

            for dx1_f in dx1_ref_vals:
                for dy1_f in dy1_ref_vals:
                    p1_full = shifted_psf_cached(dx1_f, dy1_f)
                    p1 = p1_full[mask_inds].ravel()
                    if np.allclose(p1, 0.0):
                        continue
                    for dx2_f in dx2_ref_vals:
                        for dy2_f in dy2_ref_vals:
                            sep = np.hypot(dx1_f - dx2_f, dy1_f - dy2_f)
                            if sep < min_sep_enforced or sep < self.min_separation or sep > self.max_separation:
                                continue
                            # companion must be on finite pixels
                            x2_pix = int(round(center_x + dx2_f))
                            y2_pix = int(round(center_y + dy2_f))
                            if x2_pix < 0 or x2_pix >= nx or y2_pix < 0 or y2_pix >= ny:
                                continue
                            if not finite_mask[y2_pix, x2_pix]:
                                continue
                            p2_full = shifted_psf_cached(dx2_f, dy2_f)
                            p2 = p2_full[mask_inds].ravel()
                            A = np.vstack([p1, p2]).T
                            try:
                                condA = np.linalg.cond(A)
                            except Exception:
                                condA = np.inf
                            if condA > cond_thresh or np.linalg.matrix_rank(A) < 2:
                                continue
                            # sol, chi2_local = solve_fluxes_nnls(A, data_vec, clip_min=self.clip_flux_min)
                            # dynamic clip floor: ensure companion >= absolute floor based on initial_flux
                            clip_min_candidate = max(self.clip_flux_min,self.initial_flux * self.min_companion_abs_frac_of_initial)
                            sol, chi2_local = solve_fluxes_nnls(A, data_vec, clip_min=clip_min_candidate)
                            if chi2_local < best_binary['chi2']:
                                best_binary.update({
                                    'chi2': float(chi2_local),
                                    'dx1': float(dx1_f), 'dy1': float(dy1_f),
                                    'dx2': float(dx2_f), 'dy2': float(dy2_f),
                                    'f1': float(sol[0]), 'f2': float(sol[1]), 'sep': float(sep)
                                })

        if best_binary['dx1'] is None:
            log.warning("no valid refined binary candidate found; keeping single-star result.")
            return

        # compute bic and decide
        self.chi2_binary = best_binary['chi2'] / (self.sigma**2)
        self.bic_binary = n * np.log(max(self.eps, self.chi2_binary / n)) + 6.0 * np.log(max(1, n))
        self.delta_bic = self.bic_single_sat - self.bic_binary
        sep_best = best_binary['sep']
        # compute absolute and relative floor thresholds
        min_rel = self.min_companion_flux_frac
        min_abs = max(self.clip_flux_min, self.initial_flux * self.min_companion_abs_frac_of_initial)

        # acceptance: require strong bic improvement and companion not too faint and sep beyond core radius
        if self.delta_bic > 10.0 and best_binary['f2'] > max(min_abs, min_rel * best_binary['f1']) and sep_best >= min_sep_enforced:
            # enforce primary brighter
            f1_final, f2_final = best_binary['f1'], best_binary['f2']
            dx1_out, dy1_out = best_binary['dx1'], best_binary['dy1']
            dx2_out, dy2_out = best_binary['dx2'], best_binary['dy2']
            if f2_final > f1_final:
                dx1_out, dx2_out = dx2_out, dx1_out
                dy1_out, dy2_out = dy2_out, dy1_out
                f1_final, f2_final = f2_final, f1_final
            self.dx1 = float(dx1_out)
            self.dy1 = float(dy1_out)
            self.flux1 = float(f1_final)
            self.dx2 = float(dx2_out)
            self.dy2 = float(dy2_out)
            self.flux2 = float(f2_final)
            self.bintest = True
            log.info(
                "Saturated + Unsaturated Companion preferred: ΔBIC=%.2f, sep=%.2f, f1=%.3f, f2=%.3f",
                self.delta_bic, sep_best, self.flux1, self.flux2)
        else:
            # single-star solution is already assigned above
            log.info(
                "Saturated single preferred: ΔBIC=%.3f, f1=%.3f",
                self.delta_bic, self.flux1)
        pass

    def fitpsf(self,data, nanmask, psf):
            """
            Fits a single star and a simultaneous binary star model directly to the data.

            Uses native trust-region constraints to keep stars separated without breaking gradients.
            Decides between single and binary models based on the Bayesian Information Criterion (BIC).

            Parameters
            ----------
            data : numpy.ndarray
                2D image cutout containing the source(s) to be fitted.
            psf : numpy.ndarray
                2D PSF model image.


            Returns
            -------
            dx1 : float
                X-offset of the primary star.
            dy1 : float
                Y-offset of the primary star.
            flux1 : float
                Flux of the primary star.
            dx2 : float or None
                X-offset of the companion (if binary).
            dy2 : float or None
                Y-offset of the companion (if binary).
            flux2 : float or None
                Flux of the companion (if binary).
            is_binary : bool
                True if the binary model was preferred.
            """
            weights = np.ones_like(data, dtype=float)
            self.centers = [nanmask.shape[1] // 2, nanmask.shape[0] // 2]

            if np.any(nanmask):
                initial_flux = np.max(data[~nanmask]) / np.max(psf)
                self.sigma = np.nanstd(data[~nanmask])
            else:
                initial_flux = np.max(data) / np.max(psf)
                self.sigma = np.nanstd(data)

            self.initial_flux = max(1e-5, initial_flux)
            self.n_pixels = np.sum(~nanmask.astype(bool))

            labeled_nan = label(nanmask.astype(bool), connectivity=1)
            self.nan_props = [p for p in regionprops(labeled_nan) if p.area > 0]
            log.debug("labeled_nan max %d, num nan props %d", labeled_nan.max(), len(self.nan_props))

            # --- Perimeter / annulus brightness scoring for NaN-core selection ---
            # Score each NaN region by the flux in a small annulus around the NaN core
            # (this prefers true saturated cores with bright PSF wings over tiny bad pixels).
            if len(self.nan_props) >= 1:
                self.find_multiple_saturated_cores(data,labeled_nan)
                log.info(f"{len(self.nan_props)} NaN cores found; starting saturated star analysis.")
            elif len(self.nan_props) == 0:
                self.selected = []
                log.info("no NaN cores found; skipping saturated star analysis.")

            # If two discrete NaN cores exist, fit the two saturated stars
            if len(self.selected) >= 2:
                self.fit_multiple_saturated_cores(data, psf, nanmask)

            # If exactly one NaN core found: fit single saturated star (reuse same shift/solver helpers).
            elif len(self.selected) == 1:
                self.fit_single_saturated_core(data, psf, nanmask)

            # If no seeds found: skip binary and accept single (user requirement)
            elif len(self.selected) == 0:
                self.fit_multiple_unsaturated_cores(data, psf, nanmask, weights)


