# =============================================================================
# IMPORTS
# =============================================================================

import os
import pdb
import sys
import glob
from itertools import chain

import numpy as np
import math

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.patheffects as patheffects
from matplotlib.patches import Rectangle
from matplotlib import font_manager
import scipy.ndimage as ndi

import astropy
import astropy.io.fits as fits
import astropy.units as u
import astropy.visualization as v

from . import wcs_utils
import jwst.datamodels

import shutil
import tempfile
import ipywidgets as widgets
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# =============================================================================
# MAIN
# =============================================================================

def load_plt_style(style='spaceKLIP.sk_style'):
    """
    Load the matplotlib style for spaceKLIP plots.
    
    Load the style sheet in `sk_style.mplstyle`, which is a modified version of the
    style sheet from the `webbpsf_ext` package.
    """
    plt.style.use(style)

def annotate_compass(ax,
                     image,
                     wcs,
                     xf=0.9,
                     yf=0.1,
                     length_fraction=0.07,
                     color='white',
                     bbox_color='#4B0082',
                     fontsize=12):
    """
    Plot a compass annotation onto an image, indicating the directions of North and East.

    Makes use of the methods from jdaviz, but positions the compass differently:
    jdaviz defaults to putting it in the center, and filling most of the image.
    Here we want to make a small compass in the corner.
    
    Parameters
    ----------
    ax : matplotlib.Axes
        The axis on which to draw the compass annotation.
    image : ndarray
        A 2D image to be annotated (used just to get the image dimensions).
    wcs : astropy.wcs.WCS
        World Coordinate System (WCS) information.
    xf, yf : float, optional
        The horizontal (xf) and vertical (yf) positions of the compass as fractions of the image width and height, respectively.
        Values should be between 0 and 1, where 0 places the compass at the left or bottom edge,
        and 1 places it at the right or top edge. Default is (0.9, 0.1), placing the compass near the bottom-right corner.
    length_fraction : float, optional
        Length of the compass, as a fraction of the size of the entire image
    color : str, optional
        The color of the compass arrows and labels. Default is 'white'.
    bbox_color : str, optional
        The background color for the text labels ('N' and 'E'). Default is '#4B0082'.
        If set to None, no background box will be drawn.
    fontsize : int, optional
        The font size for the compass labels. Default is 12.

    Returns
    -------
    None.
    """
    
    # Use wcs_utils from jdaviz to compute arrow positions.
    x, y, xn, yn, xe, ye, degn, dege, xflip = wcs_utils.get_compass_info(wcs, image.shape, r_fac=length_fraction)

    # Calculate the offsets needed to reposition the compass to the desired location.
    # `xo` and `yo` are the offsets in the x and y directions, respectively.
    xo = image.shape[1] * xf - x
    yo = image.shape[0] * yf - y
    x, xn, xe = [coord + xo for coord in (x, xn, xe)]  # Adjust x-coordinates.
    y, yn, ye = [coord + yo for coord in (y, yn, ye)]  # Adjust y-coordinates.

    # Plot the compass base point on the axis as a small circle.
    ax.plot(x, y, marker='o', color=color, markersize=4)

    # Annotate the North ('N') and East ('E') directions on the image.
    # Use arrows to point from the base of the compass to the respective endpoints.
    for label, (x_end, y_end) in zip(['N', 'E'], [(xn, yn), (xe, ye)]):
        bbox = dict(facecolor=bbox_color, alpha=0.5, edgecolor='none') if bbox_color else None
        ax.annotate(label, xy=(x, y), xytext=(x_end, y_end),
                    arrowprops={'arrowstyle': '<-', 'color': color, 'lw': 1.5},
                    bbox=bbox,  # Semi-transparent background for text.
                    color=color, fontsize=fontsize, va='center', ha='center')
    
def annotate_scale_bar(ax,
                       image,
                       wcs,
                       length=1 * u.arcsec,
                       xf=0.1,
                       yf=0.1,
                       color='white',
                       bbox_color='#4B0082',
                       lw=3,
                       fontsize=10):
    """
    Plot a scale bar on an image.

    Parameters
    ----------
    ax : matplotlib.Axes
        The axis on which to draw the scale bar.
    image : ndarray
        The 2D image to be annotated (used just to get the image dimensions).
    wcs : astropy.wcs.WCS
        World Coordinate System (WCS) information.
    length : astropy.Quantity, optional
        Length of the scale bar, in arcsec or equivalent unit
    xf, yf : float, optional
        The horizontal (xf) and vertical (yf) positions of the compass as fractions of the image width and height, respectively.
        Values should be between 0 and 1, where 0 places the compass at the left or bottom edge,
        and 1 places it at the right or top edge. Default is (0.9, 0.1), placing the compass near the bottom-right corner.
    color : str, optional
        The color of the scale bar and the text label. Default is 'white'.
    bbox_color : str, optional
        The background color for the text label. Default is '#4B0082'.
        If set to None, no background box will be drawn.
    lw : float, optional
        The line width of the scale bar. Default is 3.
    fontsize : int, optional
        The font size of the text label displaying the scale bar length. Default is 10.

    Returns
    -------
    None.
    """

    # Calculate the pixel scale in arcseconds per pixel.
    pixelscale = astropy.wcs.utils.proj_plane_pixel_scales(wcs).mean() * u.deg
    sb_length = (length / pixelscale).decompose()
    
    # Calculate the position of the scale bar based on image dimensions.
    xo = image.shape[1] * xf
    yo = image.shape[0] * yf

    # Draw the scale bar on the image.
    ax.plot([xo, xo + sb_length], [yo, yo], color=color, lw=lw)
    bbox = dict(facecolor=bbox_color, alpha=0.3, edgecolor='none') if bbox_color else None
    ax.text(xo + sb_length / 2, yo + 0.02 * image.shape[0], length, color=color,
            bbox=bbox, horizontalalignment='center', fontsize=fontsize)
    
    
def annotate_secondary_axes_arcsec(ax,
                                   image,
                                   wcs):
    """
    Update an image display to add secondary axes labels in an arcsec.
    
    Parameters
    ----------
    ax : matplotlib.Axes
        The axis on which to add secondary axes labels.
    image : ndarray
        The 2D image to be annotated (used just to get the image dimensions).
    wcs : astropy.wcs.WCS
        World Coordinate System (WCS) information.
    
    Returns
    -------
    None.
    """
    
    # Define the forward and inverse transforms needed for secondary_axes.
    # see https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/secondary_axis.html
    
    pixelscale = astropy.wcs.utils.proj_plane_pixel_scales(wcs)
    pix2as_x = lambda x: (x - wcs.wcs.crpix[0]) * pixelscale[0] * 3600
    pix2as_y = lambda y: (y - wcs.wcs.crpix[1]) * pixelscale[1] * 3600
    as2pix_x = lambda x: x / pixelscale[0] / 3600 + wcs.wcs.crpix[0]
    as2pix_y = lambda y: y / pixelscale[1] / 3600 + wcs.wcs.crpix[1]
    
    secax = ax.secondary_xaxis('top', functions=(pix2as_x, as2pix_x))
    secax.set_xlabel('Offset [arcsec]', fontsize='small')
    secay = ax.secondary_yaxis('right', functions=(pix2as_y, as2pix_y))
    secay.set_ylabel('Offset [arcsec]', fontsize='small')
    secax.tick_params(labelsize='small', color='white', which='both')
    secay.tick_params(labelsize='small', color='white', which='both')


@plt.style.context('spaceKLIP.sk_style')
def display_coron_image(filename,
                        vmin=None,
                        vmax=None,
                        stretch=0.0001,
                        bbox_color='#4B0082',
                        dq_only=False,
                        zoom_center=3,
                        ax=None):
    """
    Display and annotate a coronagraphic image.
    
    Shows image on asinh scale along with some basic metadata, scale bar, and compass.
    Optionally, a zoomed inset around the image center can be displayed.

    This display function is designed to flexibly adapt to several different kinds of input data,
    including rate, rateints, cal, calints files. And pyKLIP's KLmode cubes.

    Parameters
    ----------
    filename : str
        The path to the file containing the image data.
    vmin, vmax : float, optional
        The minimum/maximum data value to use for scaling the image. If None, determined automatically.
    stretch : float, optional
        The stretch factor for the asinh normalization. If None, defaults to 0.0001.
    bbox_color : str, optional
        The background color for the text label. Default is '#4B0082'.
        If set to None, no background box will be drawn.
    dq_only : bool, optional
        If True, only the DO_NOT_USE DQ flags are displayed, not the image data itself.
    zoom_center : int, optional
        The zoom factor for the inset axis centered on the image's center. Set to None to disable.

    Returns
    -------
    None.
    """
        
    # Early exit for unsupported file types.
    if 'uncal' in filename:
        raise RuntimeError("Display code does not support stage 0 uncal files. Reduce the data further before trying to display it.")

    # Determine the input file type and set corresponding flags.
    is_pyklip = 'KLmodes' in filename
    cube_ints = 'rateints' in filename or 'calints' in filename

    if is_pyklip:
        # Handle pyKLIP output.
        # PyKLIP output, we have to open this differently, can't use JWST datamodels.
        image = fits.getdata(filename)
        header = fits.getheader(filename)
        center_x, center_y = header['CRPIX1'], header['CRPIX2']
        bunit = header['BUNIT']
        wcs = astropy.wcs.WCS(header)
        if image.ndim == 3:
            image = image[-1]  # Select the last KL mode.
            wcs = wcs.dropaxis(2)  # Drop the nints axis.
        annotation_text = f"pyKLIP results for {header['TARGPROP']}\n{header['FILTER']}"
        is_psf = False
    else:
        # Handle JWST pipeline outputs.
        # Load in JWST pipeline outputs using jwst.datamodel.
        modeltype = jwst.datamodels.CubeModel if cube_ints else jwst.datamodels.ImageModel
        model = modeltype(filename)
        image = np.nanmean(model.data, axis=0) if cube_ints else model.data
        dq = model.dq[0] if cube_ints else model.dq
        nints = model.meta.exposure.nints if cube_ints else None
        center_x, center_y = model.meta.wcsinfo.crpix1, model.meta.wcsinfo.crpix2
        bunit = model.meta.bunit_data
        is_psf = model.meta.exposure.psf_reference
        annotation_text = (
            f"{model.meta.target.proposer_name}\n"
            f"{model.meta.instrument.filter}, {model.meta.exposure.readpatt}:"
            f"{model.meta.exposure.ngroups}:{model.meta.exposure.nints}\n"
            f"{model.meta.exposure.effective_exposure_time:.2f} s"
        )
        try:
            # I don't know how to deal with the slightly different API of the GWCS class
            # so, this is crude, just cast it to a regular WCS and drop the high order distortion stuff
            # This suffices for our purposes in plotting compass annotations etc.
            # (There is almost certainly a better way to do this...)
            wcs = astropy.wcs.WCS(model.meta.wcs.to_fits()[0])
        except:
            wcs = model.get_fits_wcs()
            if cube_ints:
                wcs = wcs.dropaxis(2)

    # Create a bad pixel mask.
    # Does this file have DQ extension or not? PyKLIP outputs do not.
    bpmask = np.zeros_like(image) + np.nan
    bpmask[np.isnan(image) | ((dq & 1) == 1) if not is_pyklip else np.isnan(image)] = 1

    # Set up the asinh normalization for image display.
    stats = astropy.stats.sigma_clipped_stats(image)
    vmin = vmin if vmin is not None else stats[0] - stats[2]  # 1 sigma below mean.
    vmax = vmax if vmax is not None else np.nanmax(image)  # Max value in image.
    stretch = v.AsinhStretch(a=stretch)
    norm = v.ImageNormalize(image, interval=v.ManualInterval(vmin, vmax), stretch=stretch)

    # Create the figure and axis or use provided.
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 9))
    else:
        fig = ax.figure  # Get the figure from the provided axes.
    
    if not dq_only:
        im = ax.imshow(image, norm=norm)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.95)
        cb = fig.colorbar(im, ax=ax, cax=cax, label=bunit)
        cb.ax.set_yscale('asinh')
        cb.ax.tick_params(labelsize='small')


    # Display the bad pixel mask (DO_NOT_USE flags).
    ax.set_facecolor('#4B0082')
    imdq = ax.imshow(bpmask, vmin=0, vmax=1.5, cmap=matplotlib.cm.inferno)

    # Add image annotations.
    bbox=dict(facecolor=bbox_color, alpha=0.5, edgecolor='none') if bbox_color else None
    ax.text(0.01, 0.99, annotation_text, bbox=bbox,
            transform=ax.transAxes, color='white', verticalalignment='top', fontsize=10)
    ax.set_title(f"{os.path.basename(filename)}\n", fontsize=14)
    ax.set_xlabel("Pixels", fontsize='small')
    ax.set_ylabel("Pixels", fontsize='small')
    ax.tick_params(labelsize='small', color='white', which='both')

    labelstr = 'PSF Reference' if is_psf else 'Science target after pyKLIP PSF sub.' if is_pyklip else 'Science target'
    ax.text(0.5, 0.99, labelstr, bbox=bbox,
            style='italic', fontsize=10, color='white',
            horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)
    if cube_ints:
        ax.text(0.99, 0.99, f"Showing average of {nints} ints",
                style='italic', fontsize=10, color='white', bbox=bbox,
                horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

    # Annotate compass, scale bar, and secondary axes.
    annotate_compass(ax, image, wcs, yf=0.07, bbox_color=bbox_color)
    annotate_scale_bar(ax, image, wcs, yf=0.07, bbox_color=bbox_color)

    # Annotate secondary axes in arcseconds relative to coronagraph mask center.
    ax.xaxis.set_tick_params(which='both', bottom=True, top=False)
    ax.yaxis.set_tick_params(which='both', left=True, right=False)
    annotate_secondary_axes_arcsec(ax, image, wcs)

    if zoom_center:
        # Add a zoomed-in inset of the center.
        zoom_ax = zoomed_inset_axes(ax, zoom=zoom_center/2, loc=2, bbox_to_anchor=[-0.65, 0.91, 0.1, 0.1], bbox_transform=ax.transAxes)
        zoom_ax.imshow(image, norm=norm)
        zoom_ax.imshow(bpmask, vmin=0, vmax=1.5, cmap=matplotlib.cm.inferno)
        zoom_size = min(center_x, center_y) // zoom_center
        zoom_ax.set_xlim(center_x - zoom_size, center_x + zoom_size)
        zoom_ax.set_ylim(center_y - zoom_size, center_y + zoom_size)
        zoom_ax.tick_params(labelsize='small', color='white', which='both')
        zoom_ax.set_xlabel("Pixels", fontsize='small')
        zoom_ax.set_ylabel("Pixels", fontsize='small')

        # Annotate secondary axes in the zoomed image.
        zoom_ax.xaxis.set_tick_params(which='both', bottom=True, top=False)
        zoom_ax.yaxis.set_tick_params(which='both', left=True, right=False)
        annotate_secondary_axes_arcsec(zoom_ax, image, wcs)
        mark_inset(ax, zoom_ax, loc1=3, loc2=1, fc="none", ec="0.6", alpha=0.5)

        fig.tight_layout(rect=[0.3, 0, 1, 1])
    fig.show()
    return ax

@plt.style.context('spaceKLIP.sk_style')
def display_coron_dataset(database,
                          restrict_to=None,
                          save_filename=None,
                          stage3=None,
                          vmin=None,
                          vmax=None,
                          stretch=0.0001,
                          zoom_center=3,
                          dq_only=False,
                          interactive=False,
                          bbox_color='#4B0082'):
    """
    Display multiple files in a coronagraphic dataset.

    Parameters
    ----------
    database : spaceklip.Database
        Database of files to plot.
    restrict_to : str or dict, optional
        Optional query to filter and display specific data.
        - `None`: No filtering; all tables are processed.
        - `str`: Only datasets whose database concatenation (file group) name includes this string will be shown. Most simply, set this to a filter name to only plot images with that filter.
        - `dict`: Filters tables based on database column values, where keys are column names and values are filter criteria.
    stage3 : str, optional
        Specify if data is stage 3.
    save_filename : str
        If provided, the plots will be saved to a PDF file with this name.
    vmin, vmax : float, optional
        The minimum/maximum data value to use for scaling the image. If None, determined automatically.
    stretch : float, optional
        The stretch factor for the asinh normalization. If None, defaults to 0.0001.
    zoom_center : int, optional
        The zoom factor for the inset axis centered on the image's center. Set to None to disable.
    dq_only : bool, optional
        If True, only the DO_NOT_USE DQ flags are displayed, not the image data itself.
    interactive : bool, optional
        If `True`, the plots will be displayed interactively.
    bbox_color : str, optional
        The background color for the text label. Default is '#4B0082'.
        If set to None, no background box will be drawn.
    Returns
    -------
    None.
    """
    # Initialize PDF saving if a filename is provided.
    pdf = PdfPages(save_filename) if save_filename else None

    # Infer stage3 based on db contents.
    if stage3 is None:
        stage3 = hasattr(database, 'red') and len(database.red) > 0

    # Select the appropriate dataset and file types.
    dataset = database.red if stage3 else database.obs
    types = ['PYKLIP', 'STAGE3'] if stage3 else ['SCI', 'REF']

    # Filter files based on the 'restrict_to' criteria provided.
    filtered_files = []
    for key, table in dataset.items():
        if isinstance(restrict_to, dict):
            for col, val in restrict_to.items():
                if col not in table.colnames:
                    print(f"Warning: Column '{col}' not found in the observation table for key '{key}'. Skipping over this filtering criteria.")
                    continue  # Skip the current filter if the column doesn't exist.

                vals = val if isinstance(val, list) else [val]
                table = table[[str(cell) in map(str, vals) for cell in table[col]]]
        elif isinstance(restrict_to, str) and restrict_to not in key:
            continue  # Skip this key if it doesn't match the 'restrict_to' string.

        filtered_files += [row['FITSFILE'] for row in table if row['TYPE'] in types]

        # Loop through each file that matches the specified types.
        for fn in filtered_files:
            display_coron_image(fn, vmin=vmin, vmax=vmax, stretch=stretch, zoom_center=zoom_center)
            if pdf:
                pdf.savefig(plt.gcf())
            if interactive:
                plt.close()  # Close the figure to avoid displaying it.

    # Optional: interactively slide through the files in the database.
    if interactive:
        slider = widgets.IntSlider(value=0, min=0, max=len(filtered_files) - 1, step=1, description='Image  Index:')
        def update_image(index):
            #plt.clf()
            display_coron_image(filtered_files[index], vmin=vmin, vmax=vmax, stretch=stretch,
                                zoom_center=zoom_center, dq_only=dq_only, bbox_color=bbox_color)
            plt.show()
        out = widgets.interactive_output(update_image, {'index': slider})
        display(slider, out)
    
    if pdf:
        pdf.close()

        
@plt.style.context('spaceKLIP.sk_style')
def display_image_comparisons(database,
                              base_dirs,
                              restrict_to=None,
                              save_filename=None,
                              vmin=None,
                              vmax=None,
                              stretch=0.0001,
                              zoom_center=None,
                              interactive=False,
                              dq_only=False,
                              subtract_first=False):
    """
    Compare images before and after processing.
    
    Parameters
    ----------
    database : spaceklip.Database
        Database of files to plot.
    base_dirs : list of str
        List of base directory names.
    restrict_to : str or dict, optional
        Optional query to filter and display specific data.
        - `None`: No filtering; all tables are processed.
        - `str`: Only datasets whose database concatenation (file group) name includes this string will be shown.
                 Most simply, set this to a filter name to only plot images with that filter.
        - `dict`: Filters tables based on database column values, where keys are column names and values are filter criteria.
    save_filename : str
        If provided, the plots will be saved to a PDF file with this name.
    vmin, vmax : float, optional
        The minimum/maximum data value to use for scaling the image. If None, determined automatically.
    stretch : float, optional
        The stretch factor for the asinh normalization. If None, defaults to 0.0001.
    zoom_center : int, optional
        The zoom factor for the inset axis centered on the image's center. Set to None to disable.
    dq_only : bool, optional
        If True, only the DO_NOT_USE DQ flags are displayed, not the image data itself.
    subtract_first : bool
        Whether to subtract the first SCI frame from subsequent frames.
    interactive : bool, optional
        If `True`, the plots will be displayed interactively.
    
    Returns
    -------
    None.
    """
    
    # Initialize PDF saving if a filename is provided.
    pdf = PdfPages(save_filename) if save_filename else None

    # Initialize a dictionary to store image details for each base directory.
    image_files = {base_dir: {'bp_counts': [], 'first_sci_file': None} for base_dir in base_dirs}

    # Iterate over each key and corresponding table in the database.
    # Filter files based on the 'restrict_to' criteria provided.
    filtered_files = []
    for key, table in database.obs.items():
        if isinstance(restrict_to, dict):
            for col, val in restrict_to.items():
                if col not in table.colnames:
                    print(f"Warning: Column '{col}' not found in the observation table for key '{key}'. Skipping over this filtering criteria.")
                    continue  # Skip the current filter if the column doesn't exist.

                vals = val if isinstance(val, list) else [val]
                table = table[[str(cell) in map(str, vals) for cell in table[col]]]
        elif isinstance(restrict_to, str) and restrict_to not in key:
            continue  # Skip this key if it doesn't match the 'restrict_to' string.
        
        # Filter for SCI and REF types.
        filtered_table = [row for row in table if row['TYPE'] in ['SCI', 'REF']]
        filtered_files.extend(row['FITSFILE'] for row in filtered_table)

        # Check if any SCI data remains after filtering.
        if not any(row['TYPE'] == 'SCI' for row in filtered_table):
            print(f"No SCI type files found in key: {key}."
            f" Exiting. Check 'restrict_to' criteria.")
            return
       
        # Identify the first SCI frame for subtraction, store it for later use.
        first_sci_file = next((row['FITSFILE'] for row in filtered_table if row['TYPE'] == 'SCI'), None)
        root_dir = first_sci_file.split(os.sep)[0]
        for base_dir in base_dirs:
            image_files[base_dir]['first_sci_file'] = os.path.join(root_dir, base_dir, os.path.basename(first_sci_file))

    # Create figure of appropriate size.
    num_dirs = len(base_dirs)
    num_rows, num_cols = (1, num_dirs) if num_dirs <= 3 else (2, (num_dirs + 1) // 2)
    
    # Iterate over the filtered files to process and display images.
    def update_image(index):
        fn = filtered_files[index]
        root_dir = fn.split(os.sep)[0]  # Extract the root directory from the file path.
        base_fn = os.path.basename(fn)  # Extract the base filename from the file path.
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 10, num_rows * 10))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

   
        # Iterate over each base directory and its image information.
        for ax, (base_dir, image_info) in zip(axes, image_files.items()):
            fn_path = os.path.join(root_dir, base_dir, base_fn)  # Full file path.
            
            # Count the number of bad (DO_NOT_USE) pixels in the DQ data.
            dq = fits.getdata(fn_path, extname='DQ')
            num_bad_pixels = np.sum((dq & 1) == 1)
            image_info['bp_counts'].append(num_bad_pixels)

            # Handle subtraction only if enabled.
            if subtract_first:
                # Load and subtract the first SCI frame from the current frame.
                with tempfile.NamedTemporaryFile(suffix='_' + fn.split('_')[-1], delete=False) as tmp:
                    shutil.copy2(fn_path, tmp.name)
                    with fits.open(tmp.name, mode='update') as hdul:
                        first_sci_frame = fits.getdata(image_info['first_sci_file'], extname='SCI')
                        hdul['SCI'].data -= first_sci_frame.astype(np.float32)
                        
                        # Determine the center of the image.
                        ny, nx = hdul['SCI'].data.shape[-2:]  # Handle both 2D and 3D arrays.
                        ax.axhline(y=ny // 2, color='white', linestyle='--', linewidth=1)
                        ax.axvline(x=nx // 2, color='white', linestyle='--', linewidth=1)

                        hdul.flush()
                    fn_path = tmp.name
            ax = display_coron_image(fn_path, ax=ax, vmin=vmin, vmax=vmax, stretch=stretch, zoom_center=zoom_center, dq_only=dq_only)
            ax.images[0].set_cmap('RdBu_r' if subtract_first else 'viridis')
            plt.draw()

            ax.set_title(base_dir)
            ax.legend(handles=[patches.Patch(color='orange', label=f"DO_NOT_USE = {image_info['bp_counts'][0]} px")],
                      loc='lower center', bbox_to_anchor=(0.5, -0.18))
        fig.suptitle(
            f"{os.path.basename(fn)} - {os.path.basename(image_info['first_sci_file'])}" if subtract_first else os.path.basename(fn),
            fontsize=16)
        if interactive:
            plt.show()
    
    # Optional: interactively slide through the files in the database.
    if interactive:
        slider = widgets.IntSlider(value=0, min=1 if subtract_first else 0, max=len(filtered_files) - 1, step=1, description='Image Index:')
        out = widgets.interactive_output(update_image, {'index': slider})
        display(slider, out)
   
    # Static mode.
    else:
        for i in range(len(filtered_files)):
            update_image(i+1 if subtract_first else i)
            if pdf:
                pdf.savefig(plt.gcf())
            plt.show()
            plt.close()

        if pdf:
            pdf.close()
            
@plt.style.context('spaceKLIP.sk_style')
def plot_contrast_images(meta, data, data_masked, pxsc=None, savefile='./maskimage.pdf'):
    """
    Plot subtracted images to be used for contrast estimation, one with
    companions marked, one with the masking adopted.

    """

    # Set some quick information depending on whether a pixel scale was passed
    if pxsc == None:
        extent=(-0.5, data.shape[1]-0.5, data.shape[1]-0.5, -0.5)
        pxsc = 1
        xlabel, ylabel = 'Pixels', 'Pixels'
    else:
        extl = (data.shape[1]+1.)/2.*pxsc/1000. # arcsec
        extr = (data.shape[1]-1.)/2.*pxsc/1000. # arcsec
        extent = (extl, -extr, -extl, extr)
        xlabel, ylabel = '$\\Delta$RA [arcsec]', '$\\Delta$Dec [arcsec]'

    # Initialize plots
    f, ax = plt.subplots(1, 2, figsize=(2*6.4, 1*4.8))

    # Plot subtracted image, circle input companion locations
    ax[0].imshow(np.log10(np.abs(data[-1])), origin='lower', cmap='inferno', extent=extent)
    for j in range(len(meta.ra_off)):
        cc = plt.Circle((meta.ra_off[j]/1000., meta.de_off[j]/1000.), 10.*pxsc/1000., fill=False, edgecolor='green', linewidth=3)
        ax[0].add_artist(cc)
    if hasattr(meta, 'ra_off_mask') and hasattr(meta, 'de_off_mask'):
        for j in range(len(meta.ra_off_mask)):
            cc = plt.Circle((meta.ra_off_mask[j]/1000., meta.de_off_mask[j]/1000.), 10.*pxsc/1000., fill=False, edgecolor='red', linewidth=3)
            ax[0].add_artist(cc)
        
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
    ax[0].set_title('KLIP-subtracted')

    # Plot subtracted image, with adopted masking
    ax[1].imshow(np.log10(np.abs(data_masked[-1])), origin='lower', cmap='inferno', extent=extent)
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel(ylabel)
    if 'SWB' in savefile or 'LWB' in savefile:
        ax[1].set_title('Companions & bar masked')
    else:
        ax[1].set_title('Companions masked')
    plt.tight_layout()

    # Save and close plot
    plt.savefig(savefile)
    plt.close()

    return

@plt.style.context('spaceKLIP.sk_style')
def plot_contrast_raw(meta, seps, cons, labels='default', savefile='./rawcontrast.pdf'):
    """
    Plot raw contrast curves for different KL modes.
    """

    # Initialize figure
    plt.figure(figsize=(6.4, 4.8))
    ax = plt.gca()

    # Figure out if we're plotting one contrast curve, or multiple
    if len(cons) == 1:
        if labels == 'default':
            labels == 'contrast'
        ax.plot(seps, np.squeeze(cons), label=labels)
    elif len(cons) > 1:
        if labels == 'default':
            labels = ['contrast_{}'.format(i+1) for i in range(len(seps))]

        # Loop over contrast curves to plot
        for i in range(len(cons)):
            ax.plot(seps, cons[i], label=labels[i])

    # Plot settings
    ax.set_yscale('log')
    ax.grid(axis='y')
    ax.set_xlabel('Separation [arcsec]')
    ax.set_ylabel('Contrast [5$\sigma$]')
    ax.set_title('Raw contrast curve')
    ax.legend(loc='upper right', prop=dict(size=8))
    plt.tight_layout()

    # Save and close plot
    plt.savefig(savefile)
    ax.set_xlim([0., 6.]) # arcsec
    plt.savefig(savefile.replace('raw', 'raw_trim'))
    plt.close()

    return

@plt.style.context('spaceKLIP.sk_style')
def plot_injected_locs(meta, data, transmission, seps, pas, pxsc=None, savefile='./injected.pdf'):
    '''
    Plot subtracted image and 2D transmission alongside locations of injected planets. 
    '''
    #Set some quick information depending on whether a pixel scale was passed
    if pxsc == None:
        extent=(-0.5, data.shape[1]-0.5, data.shape[1]-0.5, -0.5)
        extent_tr=(-0.5, transmission.shape[1]-0.5, transmission.shape[1]-0.5, -0.5)
        pxsc = 1
        xlabel, ylabel = 'Pixels', 'Pixels'
    else:
        extl = (data.shape[1]+1.)/2.*pxsc/1000. # arcsec
        extr = (data.shape[1]-1.)/2.*pxsc/1000. # arcsec
        extent = (extl, -extr, -extl, extr)

        extl = (transmission.shape[1]/2.)*pxsc/1000 # arcsec
        extr = (transmission.shape[1]/2.)*pxsc/1000 # arcsec
        extent_tr = (extl, -extr, -extl, extr)

        xlabel, ylabel = '$\Delta$RA [arcsec]', '$\Delta$DEC [arcsec]'

    f, ax = plt.subplots(1, 2, figsize=(2*6.4, 1*4.8))
    ax[0].imshow(np.log10(np.abs(data)), origin='lower', cmap='inferno', extent=extent)
    for i in range(len(meta.ra_off)):
        cc = plt.Circle((meta.ra_off[i]/1000., meta.de_off[i]/1000.), 10.*pxsc/1000., fill=False, edgecolor='green', linewidth=3)
        ax[0].add_artist(cc)
    for i in range(len(seps)):
        ra = seps[i]*pxsc*np.sin(np.deg2rad(pas[i])) # mas
        de = seps[i]*pxsc*np.cos(np.deg2rad(pas[i])) # mas
        cc = plt.Circle((ra/1000., de/1000.), 10.*pxsc/1000., fill=False, edgecolor='red', linewidth=3)
        ax[0].add_artist(cc)
    # ax[0].set_xlim([5., -5.])
    # ax[0].set_ylim([-5., 5.])
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
    ax[0].set_title('KLIP-subtracted')

    p1 = ax[1].imshow(transmission, origin='lower', cmap='viridis', vmin=0., vmax=1., extent=extent_tr)
    ax[1].text(0.5, 0.9, 'THIS PLOT IS WRONG', fontsize=16, transform=plt.gca().transAxes, 
                c='k', fontweight='bold', ha='center', va='center')
    c1 = plt.colorbar(p1, ax=ax[1])
    c1.set_label('Transmission', rotation=270, labelpad=20)
    for i in range(len(meta.ra_off)):
        cc = plt.Circle((meta.ra_off[i]/1000., meta.de_off[i]/1000.), 10.*pxsc/1000., fill=False, edgecolor='green', linewidth=3)
        ax[1].add_artist(cc)
    for i in range(len(seps)):
        ra = seps[i]*pxsc*np.sin(np.deg2rad(pas[i])) # mas
        de = seps[i]*pxsc*np.cos(np.deg2rad(pas[i])) # mas
        cc = plt.Circle((ra/1000., de/1000.), 10.*pxsc/1000., fill=False, edgecolor='red', linewidth=3)
        ax[1].add_artist(cc)
    # ax[1].set_xlim([5., -5.])
    # ax[1].set_ylim([-5., 5.])
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel(ylabel)
    ax[1].set_title('Transmission')
    plt.tight_layout()
    plt.savefig(savefile)
    plt.close()

    return

@plt.style.context('spaceKLIP.sk_style')
def plot_contrast_calibrated(thrput, med_thrput, fit_thrput, con_seps, cons, corr_cons, savefile='./calcontrast.pdf'):
    '''
    Plot calibrated throughput alongside calibrated contrast curves. 
    '''
    f, ax = plt.subplots(1, 2, figsize=(2*6.4, 1*4.8))
    ax[0].plot(med_thrput['seps'], med_thrput['tps'], color='mediumaquamarine', label='Median throughput')
    ax[0].scatter(thrput['seps'], thrput['tps'], s=75, color='mediumaquamarine', alpha=0.5)
    ax[0].plot(fit_thrput['seps'], fit_thrput['tps'], color='teal', label='Best fit model')
    ax[0].set_xlim([fit_thrput['seps'][0], fit_thrput['seps'][-1]])
    ax[0].set_ylim([0.0, 1.2])
    ax[0].grid(axis='y')
    ax[0].set_xlabel('Separation [pix]')
    ax[0].set_ylabel('Throughput')
    ax[0].set_title('Algo & coronmsk throughput')
    ax[0].legend(loc='lower right')
    ax[1].plot(con_seps, cons, color='mediumaquamarine', label='Raw contrast')
    ax[1].plot(con_seps, corr_cons, color='teal', label='Calibrated contrast')
    ax[1].set_yscale('log')
    ax[1].set_xlim([0., np.max(con_seps)]) # arcsec
    ax[1].set_ylim(top=3e-3)
    ax[1].grid(axis='y')
    ax[1].set_xlabel('Separation [arcsec]')
    ax[1].set_ylabel('Contrast [5$\sigma$]')
    ax[1].set_title('Calibrated contrast curve')
    ax[1].legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(savefile)
    plt.close()

    return

@plt.style.context('spaceKLIP.sk_style')
def plot_fm_psf(meta, fm_frame, data_frame, guess_flux, pxsc=None, j=0, savefile='./fmpsf.pdf'):
    '''
    Plot forward model psf
    '''
    
    #Set some quick information depending on whether a pixel scale was passed
    if pxsc == None:
        extent=(-0.5, fm_frame.shape[1]-0.5, fm_frame.shape[1]-0.5, -0.5)
        pxsc = 1
        xlabel, ylabel = 'Pixels', 'Pixels'
    else:
        extl = (fm_frame.shape[1]+1.)/2.*pxsc/1000. # arcsec
        extr = (fm_frame.shape[1]-1.)/2.*pxsc/1000. # arcsec
        extent = (extl, -extr, -extl, extr)
        xlabel, ylabel = '$\Delta$RA [arcsec]', '$\Delta$DEC [arcsec]'

    f, ax = plt.subplots(1, 2, figsize=(2*6.4, 1*4.8))
    p0 = ax[0].imshow(fm_frame*guess_flux, origin='lower', cmap='inferno', extent=extent)
    c0 = plt.colorbar(p0, ax=ax[0])
    c0.set_label('DN', rotation=270, labelpad=20)
    cc = plt.Circle((meta.ra_off[j]/1000., meta.de_off[j]/1000.), 10.*pxsc/1000., fill=False, edgecolor='green', linewidth=3)
    ax[0].add_artist(cc)
    ax[0].set_xlim([5., -5.])
    ax[0].set_ylim([-5., 5.])
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
    ax[0].set_title(r'FM PSF ($\alpha$ = %.0e)' % guess_flux)
    p1 = ax[1].imshow(data_frame, origin='lower', cmap='inferno', extent=extent)
    c1 = plt.colorbar(p1, ax=ax[1])
    c1.set_label('DN', rotation=270, labelpad=20)
    cc = plt.Circle((meta.ra_off[j]/1000., meta.de_off[j]/1000.), 10.*pxsc/1000., fill=False, edgecolor='green', linewidth=3)
    ax[1].add_artist(cc)
    ax[1].set_xlim([5., -5.])
    ax[1].set_ylim([-5., 5.])
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel(ylabel)
    ax[1].set_title('KLIP-subtracted')
    plt.tight_layout()
    plt.savefig(savefile)
    plt.close()

    return

@plt.style.context('spaceKLIP.sk_style')
def plot_chains(chain, savefile):
    '''
    Plot MCMC chains from companion fitting
    '''
    f, ax = plt.subplots(4, 1, figsize=(1*6.4, 2*4.8))
    ax[0].plot(chain[:, :, 0].T, color='black', alpha=1./3.)
    ax[0].set_xlabel('Steps')
    ax[0].set_ylabel(r'$\Delta$RA [mas]')
    ax[1].plot(chain[:, :, 1].T, color='black', alpha=1./3.)
    ax[1].set_xlabel('Steps')
    ax[1].set_ylabel(r'$\Delta$Dec [mas]')
    ax[2].plot(chain[:, :, 2].T, color='black', alpha=1./3.)
    ax[2].set_xlabel('Steps')
    ax[2].set_ylabel(r'$\alpha$ [sec/pri]')
    ax[3].plot(chain[:, :, 3].T, color='black', alpha=1./3.)
    ax[3].set_xlabel('Steps')
    ax[3].set_ylabel(r'$l$ [pix]')
    plt.suptitle('MCMC chains')
    plt.tight_layout()
    plt.savefig(savefile)
    plt.close()

@plt.style.context('spaceKLIP.sk_style')
def plot_subimages(imgdirs, subdirs, filts, submodes, numKL, 
                   window_size=2.5, cmaps_list=['viridis'],
                   imgVmin=[-40], imgVmax=[40], subVmin=[-10], subVmax=[10],
                   labelpos=[0.04, 0.05], imtext_col='w', showKL=True, useticklabels=True, cbar_textoff=1,
                   hspace=0.05, wspace=0.05):
    '''
    Create a "publication ready" plot of the coronagraphic images, alongside
    the PSF subtracted images. A grid of images will be made. Rows will correspond to 
    filters, first column the unsubtracted PSF, following columns different
    submodes and numKLs

    Parameters
    ----------

    imgdirs : list of strings
        Parent directories of the unsubtracted images, filters won't be repeated 
    subdirs : list of strings
        Parent directories of the subtracted images
    filts : list of strings
        List of filter strings to include in the plot
    submodes : list of strings
        'ADI', 'RDI', 'RDI+ADI' (or 'ADI+RDI')
    numKL : list of ints
        output images are 3-D with the third axis corresponding to different KL
        modes used in subtraction. Indicate the number of KL modes you want to
        display, if more than one, display each for each submode on the same row .
    window_size : float
        the number of arcseconds on a side
    cmaps_list : list
        a list of strings naming the colormaps for each filter. If len(cmaps_list)==1
        the same cmap will be used for each filter.
    imgVmin : list
        a list of the min values for the the cmaps for each filter. if len(imgVmin)==1
        the same min value will be used for each filter.
    imgVmax : list
        a list of the max values for the the cmaps for each filter. if len(imgVmax)==1
        the same max value will be used for each filter.
    subVmin : list
        a list of the min values for the the subtracted image cmaps for each filter. 
        if len(imgVmin)==1 the same min value will be used for each filter.
    subVmax : list
        a list of the max values for the the subtracted image cmaps for each filter. 
        if len(imgVmin)==1 the same min value will be used for each filter.
    labelpos: list
        Position of the [RA, Dec] axis labels (figure coords)
    imtext_col: str
        Color of any text / arrows that go on top of the images
    '''

    from matplotlib.ticker import MultipleLocator, MaxNLocator

    # Get the files we care about
    imgfiles = sorted(list(chain.from_iterable([glob.glob(imgdir+'*.fits') for imgdir in imgdirs])))
    subfiles = list(chain.from_iterable([glob.glob(subdir+'*.fits') for subdir in subdirs]))
    filts = [filt.upper() for filt in filts]

    # Filter the imgfiles
    used_filts = []
    true_imgfiles = {}
    for imgfile in imgfiles:
        hdr = fits.getheader(imgfile)
        if hdr.get('SUBPXPTS') is not None:
            # This keyword only exists when larger than 1. If so,
            # this is a dithered reference observation, which we don't want.
            continue
        elif (hdr['EXP_TYPE'] != 'NRC_CORON') and (hdr['EXP_TYPE'] != 'MIR_4QPM'):
            # Don't want TACONFIRM or TACQ
            continue
        elif hdr['FILTER'] not in filts:
            # We don't want this filter
            continue
        elif hdr['FILTER'] in used_filts:
            # We've already got a file for this filter
            continue
        else:
            # We want this file
            print(imgfile)
            true_imgfiles[hdr['FILTER']] = imgfile
            used_filts.append(hdr['FILTER'])

    # Filter the subfiles
    # Note that we allow repeat filters for different reductions.
    true_subfiles = []
    for subfile in subfiles:
        print(subfile)
        if any(filt in subfile for filt in filts):
            true_subfiles.append(subfile)

    #read the subtracted images, store relevant info
    sub_dict = {}
    for flt in filts:
        for fn in true_subfiles:
            if flt in fn:
                print(flt)
                with fits.open(fn) as hdul:
                    imgcube = hdul[0].data
                    psfparam = hdul[0].header['PSFPARAM']
                    center_pix = (int(np.rint(hdul[0].header['PSFCENTY'])), 
                                  int(np.rint(hdul[0].header['PSFCENTX'])))
                mode = psfparam.split('mode=')[-1].split(',')[0].replace("'", '')#ADI/RDI/ADI+RDI
                if '+' in mode:
                    reverse_mode = mode.split('+')[-1] + '+' + mode.split('+')[0]
                else:
                    reverse_mode = None
                numbasis = [int(KLnum) for KLnum in 
                      psfparam.split('numbasis=')[-1].split(',')[0].split(']')[0].split('[')[-1].split()]
                if flt in sub_dict.keys():
                    sub_dict[flt][mode] = {'image':imgcube, 'numbasis':numbasis, 
                                           'filename':fn, 'center_pix':center_pix}
                    if reverse_mode is not None:
                        sub_dict[flt][reverse_mode] ={'image':imgcube, 'numbasis':numbasis, 
                                              'filename':fn, 'center_pix':center_pix}


                else:
                    sub_dict[flt] = {mode:{'image':imgcube, 'numbasis':numbasis, 
                                           'filename':fn, 'center_pix':center_pix}}
                    if reverse_mode is not None:
                        sub_dict[flt][reverse_mode] = {'image':imgcube, 'numbasis':numbasis, 
                                                       'filename':fn, 'center_pix':center_pix}

    #miri centers from pyklip v2.6 (y, x)
    miri_img_centers = {'F1065C': (int(np.rint(111.89-1)), int(np.rint(120.81-1))),
                        'F1140C': (int(np.rint(112.2-1)) , int(np.rint(119.99-1))) ,
                        'F1550C': (int(np.rint(113.33-1)), int(np.rint(119.84-1))),  }
    pltscale = {'NIRCAM': 0.063, 'MIRI': 0.11}
    plot_extent = (-1*window_size/2, window_size/2, -1*window_size/2, window_size/2)

    ydim = len(filts)
    xdim = len(submodes)*len(numKL) + 1

    wratios = [1]*xdim
    wratios.append(0.015*xdim) 

    # Start making the figure
    fig = plt.figure(figsize=[xdim*3, ydim*3])
    grid = gs.GridSpec(ydim, xdim+1, figure=fig, wspace=wspace, hspace=hspace, width_ratios=wratios, left=0.06, right=0.94, bottom=0.08, top=0.93)

    if len(cmaps_list) == 1:
        cmaps_list *= len(filts)
    if len(imgVmin) == 1:
        imgVmin *= len(filts)
    if len(imgVmax) == 1:
        imgVmax *= len(filts)
    if len(subVmin) == 1:
        subVmin *= len(filts)
    if len(subVmax) == 1:
        subVmax *= len(filts)

    for row, flt in enumerate(filts):
        prihdr = fits.getheader(true_imgfiles[flt])
        instrument = prihdr['INSTRUME']
        ax = plt.subplot(grid[row, 0])
        with fits.open(true_imgfiles[flt]) as hdul:
            img = np.nanmedian(hdul['SCI'].data, axis=0)
            hdr = hdul['SCI'].header
            roll = hdr['PA_V3'] + hdr['V3I_YANG']

        # get center information
        if instrument == 'NIRCAM':
            center_pix = (int(np.rint(hdr['CRPIX2']-1)), 
                          int(np.rint(hdr['CRPIX1']-1)))
            window_pix = int(np.rint(window_size / pltscale['NIRCAM'] / 2))
            offset = (window_pix - window_size / 0.063 / 2) *0.063
        else:
            center_pix = miri_img_centers[flt]
            window_pix = int(np.rint(window_size / pltscale['MIRI'] / 2))
            offset = (window_pix - window_size / 0.11 / 2) *0.11

        # Rotate unsubtracted image so North is up
        x = np.linspace(0, img.shape[0], img.shape[0])
        y = np.linspace(0, img.shape[1], img.shape[1])
        xmesh, ymesh = np.meshgrid(x, y)

        old_center = [img.shape[1]/2, img.shape[0]/2] #y, x
        xmesh += center_pix[1]
        ymesh += center_pix[0]
        xmesh -= old_center[1]
        ymesh -= old_center[0]

        new_data = ndi.map_coordinates(img, [ymesh, xmesh])

        rot_img = ndi.rotate(new_data, -roll)
        rotshape = rot_img.shape
        focus_slices_img = (slice(int(rotshape[0]/2) - window_pix, int(rotshape[0]/2) + window_pix),
                            slice(int(rotshape[1]/2) - window_pix, int(rotshape[1]/2) + window_pix))

        ax.imshow(rot_img[focus_slices_img], extent=plot_extent, cmap=cmaps_list[row], vmin=imgVmin[row], vmax=imgVmax[row])

        if row == 0:
            unsub = ax.text(0.95, 0.88, 'Unsub', fontsize=16, transform=plt.gca().transAxes, c='w', fontweight='bold', ha='right')

        # Plot N-E arrows
        ar_n = ax.arrow(0.9, 0.096, 0.0, 0.1, transform=plt.gca().transAxes, color=imtext_col, width=0.01, \
                head_width=0.04, head_length=0.04, path_effects=[patheffects.Stroke(linewidth=3, foreground='k'), patheffects.Normal()])
        n = ax.text(0.9, 0.29, 'N', fontsize=16, transform=plt.gca().transAxes, c=imtext_col, \
                va='center', ha='center', fontweight='bold')
        ar_e = ax.arrow(0.905, 0.1, -0.1, 0.0, transform=plt.gca().transAxes, color=imtext_col, width=0.01, \
                head_width=0.04, head_length=0.04, path_effects=[patheffects.Stroke(linewidth=3, foreground='k'), patheffects.Normal()])
        e = ax.text(0.71, 0.1, 'E', fontsize=16, transform=plt.gca().transAxes, c=imtext_col, \
                va='center', ha='center', fontweight='bold')
        # Plot 1" Line
        arc1text = ax.text(-int(window_size/2)+0.5, -int(window_size/2)+(window_size/75), '1"', fontsize=16, c=imtext_col, \
                va='bottom', ha='center', fontweight='bold')
        ax.add_patch(Rectangle((-int(window_size/2), -int(window_size/2)-0.075), 1, 0.15,
                      alpha=1, facecolor='w', edgecolor='k', linewidth=1))

        #ax.set_ylabel(prihdr['FILTER']+'\n\n', fontsize=18)#, fontproperties=compModern(14))
        #Draw lines around image text etc
        ftext = ax.text(0.05, 0.88, prihdr['FILTER'], fontsize=16, transform=plt.gca().transAxes, c='w', fontweight='bold')
        for ti, temp in enumerate([unsub, n, e, ftext, arc1text]):
            temp = temp.set_path_effects([patheffects.withStroke(linewidth=2, foreground='k')])
        # Draw another arrow on top to mask the line
        ax.arrow(0.9, 0.096, 0.0, 0.1, transform=plt.gca().transAxes, color=imtext_col, width=0.01, \
                head_width=0.04, head_length=0.04)
        #ax.xaxis.set_major_locator(MaxNLocator(nbins=7, min_n_ticks=3, prune='both'))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_major_locator(MultipleLocator(1))
        if row != (len(filts)-1):
            plt.setp(ax.get_xticklabels(), visible=False)
        if useticklabels == False:
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
        # else:
        #     ax.set_xlabel('RA Offset (")')#, fontproperties=compModern(14))
        #     tickFont(ax, 'x', fontproperties=compModern(10))
        # tickFont(ax, 'y', fontproperties=compModern(10))
        ax.tick_params(which='major', length=4, color=imtext_col)
        ax.tick_params(which='minor', length=0, color=imtext_col)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)

        for mm, mde in enumerate(submodes):
            focus_slices_sub = (slice(sub_dict[flt][mde]['center_pix'][0] - window_pix, 
                                      sub_dict[flt][mde]['center_pix'][0] + window_pix),
                                slice(sub_dict[flt][mde]['center_pix'][1] - window_pix, 
                                      sub_dict[flt][mde]['center_pix'][1] + window_pix))
            for kk, nkl in enumerate(numKL):
                if nkl == 'max':
                    nkl = sub_dict[flt][mde]['numbasis'][-1]
                column = 1 + (len(numKL)*mm + kk)
                ax = plt.subplot(grid[row, column])
                plotimg = sub_dict[flt][mde]['image'][sub_dict[flt][mde]['numbasis'].index(nkl),:,:][focus_slices_sub]
                plotimg[np.where(np.isnan(plotimg))] = 0
                subimg = ax.imshow(plotimg, 
                          extent=plot_extent, cmap=cmaps_list[row], vmin=subVmin[row], vmax=subVmax[row])
                star = ax.scatter(0.+offset, 0.+offset, marker='*', color='w', s=100)
                star.set_path_effects([patheffects.withStroke(linewidth=3, foreground='k')])
                ax.xaxis.set_major_locator(MultipleLocator(1))
                ax.yaxis.set_major_locator(MultipleLocator(1))
                plt.setp(ax.get_yticklabels(), visible=False)
                ax.tick_params(which='major', length=4, color=imtext_col)
                ax.tick_params(which='minor', length=0, color=imtext_col) 

                for axis in ['top','bottom','left','right']:
                    ax.spines[axis].set_linewidth(2)
                if row != (len(filts)-1):
                    plt.setp(ax.get_xticklabels(), visible=False)
                if useticklabels == False:
                    plt.setp(ax.get_xticklabels(), visible=False)
                    plt.setp(ax.get_yticklabels(), visible=False)
                # else:
                    #ax.xaxis.set_major_locator(MaxNLocator(nbins=7, min_n_ticks=3, prune='both'))
                    #tickFont(ax, 'x', fontproperties=compModern(10))
                    # ax.set_x label('RA Offset (")')#, fontproperties=compModern(14))
                if row == 0:
                    if showKL:
                        ax.set_title('{:s}, KL={:d}'.format(mde, nkl), fontweight='bold')#, fontproperties=compModern(16))
                    else:
                        temp = ax.text(0.95, 0.88, mde, fontsize=16, transform=plt.gca().transAxes, c='w', fontweight='bold', ha='right')
                        temp.set_path_effects([patheffects.withStroke(linewidth=2, foreground='k')])


        # Plot the scale bar
        cax = plt.subplot(grid[row,-1])
        cb = plt.colorbar(mappable=subimg, cax=cax)
        cb.outline.set_linewidth(2)
        cax.tick_params(which='both', color='k', labelsize=14, width=2, direction='out')
        for axis in ['top','bottom','left','right']:
                cax.spines[axis].set_linewidth(2)
        if flt != 'F1550C':
            cax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Add label for color bars
    fig.text(cbar_textoff, 0.5, "Counts (MJy/sr)", rotation=270, va='center', fontsize=18)

    if useticklabels != False:
        fig.text(0.5, labelpos[0], 'RA Offset (")', ha='center', fontsize=18)
        fig.text(labelpos[1], 0.5, 'Dec Offset (")', va='center', rotation='vertical', fontsize=18)
    fig.tight_layout()
    fig.tight_layout()
    fig.tight_layout()
    #plt.show()
    return fig


def tickFont(ax, xy, fontproperties):
    if xy == 'x':
        plt.setp(ax.get_xticklabels(), fontproperties=fontproperties)
    if xy == 'y':
        plt.setp(ax.get_yticklabels(), fontproperties=fontproperties)


def compModern(size=20):
    computerModern = font_manager.FontProperties(size=size,
                                            fname=os.path.join(os.path.dirname(__file__),
                                            'cmunrm.ttf'))
    return computerModern
