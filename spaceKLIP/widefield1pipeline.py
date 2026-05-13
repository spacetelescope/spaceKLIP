from __future__ import division

import matplotlib

# =============================================================================
# IMPORTS
# =============================================================================

import os

import astropy.io.fits as fits
import numpy as np

from tqdm import trange

from jwst.stpipe import Step
from jwst import datamodels
from jwst.datamodels import dqflags, RampModel
from jwst.pipeline import Detector1Pipeline
from .fnoise_clean import kTCSubtractStep, OneOverfStep
from .expjumpramp import ExperimentalJumpRampStep

from scipy.interpolate import interp1d
from webbpsf_ext.image_manip import expand_mask

import warnings
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# =============================================================================
# MAIN
# =============================================================================

def run_single_file(fitspath, output_dir, steps={}, verbose=False, **kwargs):
    """ Run the JWST stage 1 detector pipeline on a single file.

    WARNING: Will overwrite exiting files.

    This customized implementation can:

    - Do a custom saturation correction where only the bottom/top/left/right
      and not the diagonal pixels next to a saturated pixel are flagged.
    - Do a pseudo reference pixel correction. Therefore, flag the requested
      edge rows and columns as reference pixels, run the JWST stage 1 refpix
      step, and unflag the pseudo reference pixels again. Only applicable for
      subarray data.
    - Remove 1/f noise spatial striping in NIRCam data.

    Parameters
    ----------
    fitspath : str
        Path to the input FITS file (uncal.fits).
    output_dir : str
        Path to the output directory to save the resulting data products.
    steps : dict, optional
        See here for how to use the steps parameter:
        https://jwst-pipeline.readthedocs.io/en/latest/jwst/user_documentation/running_pipeline_python.html#configuring-a-pipeline-step-in-python
        Custom step parameters are:

        - saturation/grow_diagonal : bool, optional
            Flag also diagonal pixels (or only bottom/top/left/right)?
            The default is True.
        - saturation/flag_rcsat : bool, optional
            Flag RC pixels as always saturated? The default is False.
        - refpix/nlower : int, optional
            Number of rows at frame bottom that shall be used as additional
            reference pixels. The default is 4.
        - refpix/nupper : int, optional
            Number of rows at frame top that shall be used as additional
            reference pixels. The default is 4.
        - refpix/nrow_off : int, optional
            Number of rows to offset the reference pixel region from the
            bottom/top of the frame. The default is 0.
        - ramp_fit/save_calibrated_ramp : bool, optional
            Save the calibrated ramp? The default is False.

        Additional useful step parameters:

        - saturation/n_pix_grow_sat : int, optional
            Number of pixels to grow for saturation flagging. Default is 1.
        - ramp_fit/suppress_one_group : bool, optional
            If True, skips slope calc for pixels with only 1 available group.
            Default: False.
        - ramp_fit/maximum_cores : str, optional
            max number of parallel processes to create during ramp fitting.
            'none', 'quarter', 'half', or 'all'. Default: 'none'.

        The default is {}.

    Keyword Args
    ------------
    save_results : bool, optional
        Save the JWST pipeline step products? The default is True.
    save_intermediates : bool, optional
        Save intermediate steps, such as dq_init, saturation, refpix,
        jump, linearity, ramp, etc. Default is False.
    return_rateints : bool, optional
        Return the rateints model instead of rate? Default is False.

    Returns
    -------
    Pipeline output, either rate or rateint data model.

    """

    # from webbpsf_ext.analysis_tools import nrc_ref_info
    #
    # Print all info message if verbose, otherwise only errors or critical.

    from .logging_tools import all_logging_disabled
    log_level = logging.INFO if verbose else logging.ERROR

    # Create output directory if it doesn't exist.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize Coron1Pipeline.
    with all_logging_disabled(log_level):
        pipeline = Detector1Pipeline(output_dir=output_dir)

    # Options for saving results
    pipeline.save_results = kwargs.get('save_results', True)
    pipeline.save_calibrated_ramp = kwargs.get('save_calibrated_ramp', False)
    pipeline.save_intermediates = kwargs.get('save_intermediates', False)
    pipeline.return_rateints = kwargs.get('return_rateints', False)

    # Run Coron1Pipeline. Raise exception on error.
    # Ensure that pipeline is closed out.
    try:
        with all_logging_disabled(log_level):
            res = pipeline.call(fitspath,
                           save_results=True,
                           output_dir=output_dir
                           )
    except Exception as e:
        raise RuntimeError(
            'Caught exception during pipeline processing.'
            '\nException: {}'.format(e)
        )
    finally:
        try:
            pipeline.closeout()
        except AttributeError:
            # Method deprecated as of stpipe version 0.6.0
            pass

    return res


def run_obs(database,
            steps={},
            subdir='stage1',
            overwrite=True,
            quiet=False,
            verbose=False,
            **kwargs):
    """
    Run the JWST stage 1 detector pipeline on the input observations database.
    This customized implementation can:

    - Do a custom saturation correction where only the bottom/top/left/right
      and not the diagonal pixels next to a saturated pixel are flagged.
    - Do a pseudo reference pixel correction. Therefore, flag the requested
      edge rows and columns as reference pixels, run the JWST stage 1 refpix
      step, and unflag the pseudo reference pixels again. Only applicable for
      subarray data.
    - Remove 1/f noise spatial striping in NIRCam data.

    Parameters
    ----------
    database : spaceKLIP.Database
        SpaceKLIP database on which the JWST stage 1 pipeline shall be run.
    steps : dict, optional
        See here for how to use the steps parameter:
        https://jwst-pipeline.readthedocs.io/en/latest/jwst/user_documentation/running_pipeline_python.html#configuring-a-pipeline-step-in-python
        Custom step parameters are:

        - saturation/grow_diagonal : bool, optional
            Flag also diagonal pixels (or only bottom/top/left/right)?
            The default is True.
        - saturation/flag_rcsat : bool, optional
            Flag RC pixels as always saturated? The default is False.
        - refpix/nlower : int, optional
            Number of rows at frame bottom that shall be used as additional
            reference pixels. The default is 4.
        - refpix/nupper : int, optional
            Number of rows at frame top that shall be used as additional
            reference pixels. The default is 4.
        - refpix/nrow_off : int, optional
            Number of rows to offset the reference pixel region from the
            bottom/top of the frame. The default is 0.
        - ramp_fit/save_calibrated_ramp : bool, optional
            Save the calibrated ramp? The default is False.

        Additional useful step parameters:

        - saturation/n_pix_grow_sat : int, optional
            Number of pixels to grow for saturation flagging. Default is 1.
        - ramp_fit/suppress_one_group : bool, optional
            If True, skips slope calc for pixels with only 1 available group.
            Default: False.
        - ramp_fit/maximum_cores : str, optional
            max number of parallel processes to create during ramp fitting.
            'none', 'quarter', 'half', or 'all'. Default: 'none'.

        Default is {}.
        Each of these parameters can be passed directly through `kwargs`.
    subdir : str, optional
        Name of the directory where the data products shall be saved. The
        default is 'stage1'.
    overwrite : bool, optional
        Overwrite existing files? Default is True.
    quiet : bool, optional
        Use progress bar to track progress instead of messages.
        Overrides verbose and sets it to False. Default is False.
    verbose : bool, optional
        Print all info messages? Default is False.

    Keyword Args
    ------------
    save_results : bool, optional
        Save the JWST pipeline step products? The default is True.
    save_intermediates : bool, optional
        Save intermediate steps, such as dq_init, saturation, refpix,
        jump, linearity, ramp, etc. Default is False.
    return_rateints : bool, optional
        Return the rateints model instead of rate? Default is False.

    Returns
    -------
    None.

    """

    # Set output directory.
    output_dir = os.path.join(database.output_dir, subdir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get list of concatenation keys.
    keys = list(database.obs.keys())
    nkeys = len(keys)
    if quiet:
        verbose = False
        itervals = trange(nkeys, desc='Concatenations')
    else:
        itervals = range(nkeys)

    groupmaskflag = 0  # Set flag for group masking
    skip_revert = False  # Set flag for skipping a file
    # Loop through concatenations.
    for i in itervals:
        key = keys[i]
        if not quiet: log.info('--> Concatenation ' + key)

        # Loop through FITS files.
        nfitsfiles = len(database.obs[key])
        jtervals = trange(nfitsfiles, desc='FITS files', leave=False) if quiet else range(nfitsfiles)

        # Need to do some preparation steps if group masking is wanted before running pipeline
        steps = apply_masking_prechecks(steps)

        # Need to make sure that the database.obs[key] order deals with ref, ref_bg before sci, sci_bg files
        # Order is specific to group masking done on reference images (only option at this time)
        # if not steps['mask_groups']['skip']:
        #     # order database.obs[key] based on TYPE, i.e. REF, REF_BG, REF_TA, SCI, etc...
        #     # keeping the order of reference observations based on FITSFILE
        #     database.obs[key].sort(['TYPE', 'FITSFILE'])
        #     print(database.obs[key])

        for j in jtervals:

            # Skip non-stage 0 files.
            head, tail = os.path.split(database.obs[key]['FITSFILE'][j])
            fitspath = os.path.abspath(database.obs[key]['FITSFILE'][j])
            if database.obs[key]['DATAMODL'][j] != 'STAGE0':
                if not quiet: log.info('  --> Coron1Pipeline: skipping non-stage 0 file ' + tail)
                continue

            # Check if we are skipping the mask_groups, if not run routine.
            if not steps['mask_groups']['skip']:
                if ('mask_array' not in steps['mask_groups']) and (groupmaskflag == 0):
                    # set a flag that we are running the group optimisation
                    groupmaskflag = 1

                # Even if we are not skipping the routine, at the moment it only works on
                # REF/REF_BG data, and don't want to run on unspecified file types
                file_type = database.obs[key]['TYPE'][j]
                this_skip = file_type not in steps['mask_groups']['types']
                if not this_skip and file_type not in ['REF', 'REF_BG']:
                    log.info('  --> Group masking only works for reference images at this time! Skipping...')
                    this_skip = True

                # Don't run function prep function if we don't need to
                if not this_skip:
                    if steps['mask_groups']['mask_method'] == 'basic':
                        steps = prepare_group_masking_basic(steps,
                                                            database.obs[key],
                                                            quiet)
                    elif steps['mask_groups']['mask_method'] == 'advanced':
                        fitstype = database.obs[key]['TYPE'][j]
                        steps = prepare_group_masking_advanced(steps,
                                                               database.obs[key],
                                                               fitspath,
                                                               fitstype,
                                                               quiet)
                    elif steps['mask_groups']['mask_method'] == 'custom':
                        steps = prepare_group_masking_custom(steps,
                                                             database.obs[key],
                                                             quiet)
                else:
                    # Even though we are using mask_groups, this particular file will not have any groups masked
                    # Instruct to skip the step, but keep a record using skip_revert so we can undo for the next file.
                    steps['mask_groups']['skip'] = True
                    skip_revert = True

            # Get expected output file name
            outfile_name = tail.replace('uncal.fits', 'rateints.fits')
            fitsout_path = os.path.join(output_dir, outfile_name)

            # Skip if file already exists and overwrite is False.
            if os.path.isfile(fitsout_path) and not overwrite:
                if not quiet: log.info('  --> Coron1Pipeline: skipping already processed file '
                                       + tail)
            else:
                if not quiet: log.info('  --> Coron1Pipeline: processing ' + tail)
                _ = run_single_file(fitspath, output_dir, steps=steps,
                                    verbose=verbose, **kwargs)

            if skip_revert:
                # Need to make sure we don't skip later files if we just didn't want to mask_groups for this file
                steps['mask_groups']['skip'] = False
                skip_revert = False

            if (j == nfitsfiles - 1) and (groupmaskflag == 1):
                '''This is the last file for this concatenation, and the groupmaskflag has been
                set. This means we need to reset the mask_array back to original state,
                which was that it didn't exist, so that the routine is rerun. '''
                groupmaskflag = 0

                if steps['mask_groups']['mask_method'] == 'basic':
                    del steps['mask_groups']['mask_array']
                elif steps['mask_groups']['mask_method'] == 'advanced':
                    del steps['mask_groups']['maxgrps_faint']
                    del steps['mask_groups']['maxgrps_bright']

            # Update spaceKLIP database.
            database.update_obs(key, j, fitsout_path)


def apply_masking_prechecks(steps):
    # Need to do some preparation steps for group masking before running pipeline
    steps['mask_groups'] = steps.setdefault('mask_groups', {})
    if not steps['mask_groups']:
        # If mask_groups unspecified or has no parameters, skip by default
        steps['mask_groups']['skip'] = True
    else:
        # If mask_groups specified but skip isn't mentioned, set to False
        steps['mask_groups'].setdefault('skip', False)
    steps['mask_groups'].setdefault('mask_method', 'basic')
    steps['mask_groups'].setdefault('types', ['REF', 'REF_BG'])

    return steps


def prepare_group_masking_basic(steps, observations, quiet=False):
    if 'mask_array' not in steps['mask_groups']:
        '''First time in the file loop, or groups_to_mask has been preset, 
        run the optimisation and set groups to mask. '''
        if not quiet:
            log.info('  --> Coron1Pipeline: Optimizing number of groups to mask in ramp,'
                     ' this make take a few minutes.')

        if 'cropwidth' not in steps['mask_groups']:
            steps['mask_groups']['cropwidth'] = 20
        if 'edgewidth' not in steps['mask_groups']:
            steps['mask_groups']['edgewidth'] = 10

        # Get crop width, part of image we care about
        crop = steps['mask_groups']['cropwidth']
        edge = steps['mask_groups']['edgewidth']

        # Get our cropped science frames and reference cubes
        sci_frames = []
        ref_cubes = []
        nfitsfiles = len(observations)
        for j in range(nfitsfiles):
            if observations['TYPE'][j] == 'SCI':
                with fits.open(os.path.abspath(observations['FITSFILE'][j])) as hdul:
                    sci_frame = hdul['SCI'].data[:, -1, :, :].astype(float)

                    # Subtract a median so we focus on brightest pixels
                    sci_frame -= np.nanmedian(sci_frame, axis=(1, 2), keepdims=True)

                    # Crop around CRPIX
                    crpix_x, crpix_y = hdul["SCI"].header["CRPIX1"], hdul["SCI"].header["CRPIX2"]
                    xlo = int(crpix_x) - crop
                    xhi = int(crpix_x) + crop
                    ylo = int(crpix_y) - crop
                    yhi = int(crpix_y) + crop
                    sci_frame = sci_frame[:, ylo:yhi, xlo:xhi]

                    # Now going to set the core to 0, so we focus less on the highly variable
                    # PSF core
                    sci_frame[:, edge:-edge, edge:-edge] = np.nan

                    sci_frames.append(sci_frame)
            elif observations['TYPE'][j] == 'REF':
                with fits.open(os.path.abspath(observations['FITSFILE'][j])) as hdul:
                    ref_cube = hdul['SCI'].data.astype(float)
                    ref_shape = ref_cube.shape

                    # Subtract a median so we focus on brightest pixels
                    ref_cube -= np.nanmedian(ref_cube, axis=(2, 3), keepdims=True)

                    # Crop around CRPIX
                    crpix_x, crpix_y = hdul["SCI"].header["CRPIX1"], hdul["SCI"].header["CRPIX2"]
                    xlo = int(crpix_x) - crop
                    xhi = int(crpix_x) + crop
                    ylo = int(crpix_y) - crop
                    yhi = int(crpix_y) + crop
                    ref_cube = ref_cube[:, :, ylo:yhi, xlo:xhi]

                    # Now going to set the core to 0, so we focus less on the highly variable
                    # PSF core
                    ref_cube[:, :, edge:-edge, edge:-edge] = np.nan

                    ref_cubes.append(ref_cube)

        # Want to check against every integration of every science dataset to find whichever
        # matches the best, then use that for the scaling.
        max_grp_to_use = []
        for sci_i, sci_frame in enumerate(sci_frames):
            for int_i, sci_last_group in enumerate(sci_frame):
                # Compare every reference group to this integration
                best_diff = np.inf
                for ref_cube in ref_cubes:
                    this_cube_diffs = []
                    for ref_int in ref_cube:
                        this_int_diffs = []
                        for ref_group in ref_int:
                            diff = np.abs(np.nansum(ref_group) - np.nansum(sci_last_group))
                            this_int_diffs.append(diff)
                        this_cube_diffs.append(this_int_diffs)

                    # Is the median of these diffs better that other reference cubes?
                    if np.nanmin(this_cube_diffs) < best_diff:
                        # If yes, this reference cube is a better match to the science
                        best_diff = np.nanmin(this_cube_diffs)
                        best_maxgrp = np.median(np.argmin(this_cube_diffs, axis=1))
                max_grp_to_use.append(best_maxgrp)

        # Assemble array of groups to mask, starting one above the max group
        final_max_grp_to_use = int(np.nanmedian(max_grp_to_use))
        groups_to_mask = np.arange(final_max_grp_to_use + 1, ref_cubes[0].shape[1])

        # Make the mask array
        mask_array = np.zeros(ref_shape, dtype=bool)
        mask_array[:, groups_to_mask, :, :] = 1

        # Assign to steps so this stage doesn't get repeated.
        steps['mask_groups']['mask_array'] = mask_array

    return steps


def prepare_group_masking_custom(steps, observations, quiet=False):
    if 'mask_array' not in steps['mask_groups']:
        '''First time in the file loop, or groups_to_mask has been preset, 
        run the optimisation and set groups to mask. '''
        if not quiet:
            log.info('  --> Coron1Pipeline: Masking custom number of groups in ramp')

        if 'custom_group' not in steps['mask_groups']:
            raise KeyError(
                'mask_method "custom" requires you to set the "custom_group" keyword argument in the "mask_groups" step.')

        # Get science frames and reference cubes
        sci_frames = []
        ref_cubes = []
        nfitsfiles = len(observations)
        for j in range(nfitsfiles):
            if observations['TYPE'][j] == 'SCI':
                with fits.open(os.path.abspath(observations['FITSFILE'][j])) as hdul:
                    sci_frame = hdul['SCI'].data[:, -1, :, :].astype(float)
                    sci_frames.append(sci_frame)
            elif observations['TYPE'][j] == 'REF':
                with fits.open(os.path.abspath(observations['FITSFILE'][j])) as hdul:
                    ref_cube = hdul['SCI'].data.astype(float)
                    ref_shape = ref_cube.shape
                    ref_cubes.append(ref_cube)

        # Assemble array of groups to mask, starting one above the max group
        final_max_grp_to_use = int(steps['mask_groups']['custom_group'])
        groups_to_mask = np.arange(final_max_grp_to_use + 1, ref_cubes[0].shape[1])

        # Make the mask array
        mask_array = np.zeros(ref_shape, dtype=bool)
        mask_array[:, groups_to_mask, :, :] = 1

        # Assign to steps so this stage doesn't get repeated.
        steps['mask_groups']['mask_array'] = mask_array

    return steps


def prepare_group_masking_advanced(steps, observations, refpath, reftype, quiet=False):
    '''
    Advanced group masking method which computes the group mask on a pixel by pixel
    and reference cube by reference cube basis
    '''

    if 'cropwidth' not in steps['mask_groups']:
        steps['mask_groups']['cropwidth'] = 30
    if 'edgewidth' not in steps['mask_groups']:
        steps['mask_groups']['edgewidth'] = 20
    if 'threshold' not in steps['mask_groups']:
        steps['mask_groups']['threshold'] = 85

    # Get crop width, part of image we care about
    crop = steps['mask_groups']['cropwidth']
    edge = steps['mask_groups']['edgewidth']
    threshold = steps['mask_groups']['threshold']

    if ('maxgrps_faint' not in steps['mask_groups'] or
            'maxgrps_bright' not in steps['mask_groups'] or
            'cropmask' not in steps['mask_groups'] or
            'refbg_maxcounts' not in steps['mask_groups']):
        '''First time in the file loop, or groups_to_mask has been preset, 
        run the optimisation and set groups to mask. '''

        sci_frames = []
        sci_crpixs = []
        ref_cubes = []
        ref_crpixs = []

        nfitsfiles = len(observations)
        for j in range(nfitsfiles):
            if observations['TYPE'][j] == 'SCI':
                with fits.open(os.path.abspath(observations['FITSFILE'][j])) as hdul:
                    sci_frame = hdul['SCI'].data[:, -1, :, :].astype(float)
                    sci_crpix_x, sci_crpix_y = hdul["SCI"].header["CRPIX1"], hdul["SCI"].header["CRPIX2"]
                    sci_frames.append(sci_frame)
                    sci_crpixs.append([sci_crpix_x, sci_crpix_y])
            elif observations['TYPE'][j] == 'REF':
                with fits.open(os.path.abspath(observations['FITSFILE'][j])) as hdul:
                    ref_cube = hdul['SCI'].data.astype(float)
                    ref_crpix_x, ref_crpix_y = hdul["SCI"].header["CRPIX1"], hdul["SCI"].header["CRPIX2"]
                    ref_cubes.append(ref_cube)
                    ref_crpixs.append([ref_crpix_x, ref_crpix_y])

        # Crop and median science
        sci_frames_cropped = []
        for i, scif in enumerate(sci_frames):
            crpix_x, crpix_y = sci_crpixs[i]
            xlo = int(crpix_x) - crop
            xhi = int(crpix_x) + crop
            ylo = int(crpix_y) - crop
            yhi = int(crpix_y) + crop
            sci_frames_cropped.append(scif[:, ylo:yhi, xlo:xhi])

        sci_frames_modified = np.nanmedian(sci_frames_cropped, axis=1)  # Median over integrations

        # Crop and median reference
        ref_cubes_cropped = []
        for i, refc in enumerate(ref_cubes):
            crpix_x, crpix_y = ref_crpixs[i]
            xlo = int(crpix_x) - crop
            xhi = int(crpix_x) + crop
            ylo = int(crpix_y) - crop
            yhi = int(crpix_y) + crop
            ref_cubes_cropped.append(refc[:, :, ylo:yhi, xlo:xhi])

        ref_cubes_modified = np.nanmedian(ref_cubes_cropped, axis=1)  # Median over integrations

        # Median subtract the images
        sci_frames_medsub = sci_frames_modified - np.nanmedian(sci_frames_modified, axis=(1, 2), keepdims=True)
        ref_cubes_medsub = ref_cubes_modified - np.nanmedian(ref_cubes_modified, axis=(2, 3), keepdims=True)

        # Flatten array and find indices above percentile threshold value
        sci_frames_flat = np.reshape(sci_frames_medsub, (sci_frames_medsub.shape[0], -1))
        per = np.percentile(sci_frames_flat, [threshold])
        above_threshold_indices = np.where(sci_frames_medsub > per)

        # Create an empty mask array
        mask = np.zeros_like(sci_frames_medsub, dtype=bool)

        # Define function to expand indices and update mask
        def expand_and_update_mask(indices, mask, xpad=2, ypad=2):
            for z, x, y in zip(*indices):
                mask[z, max(0, x - xpad):min(mask.shape[1], x + xpad), max(0, y - ypad):min(mask.shape[2],
                                                                                            y + ypad)] = True

        # Expand indices and update mask for each 2D slice
        for z_slice in range(sci_frames_medsub.shape[0]):
            indices_slice = np.where(above_threshold_indices[0] == z_slice)
            expand_and_update_mask(
                (above_threshold_indices[0][indices_slice], above_threshold_indices[1][indices_slice],
                 above_threshold_indices[2][indices_slice]), mask)

        # Okay now make some hollowed out cropped frames, to focus on fainter, but still bright, pixels
        sci_frames_hollow = sci_frames_medsub.copy()
        sci_frames_hollow[:, edge:-edge, edge:-edge] = np.nan
        sci_frames_hollow[~mask] = np.nan

        ref_cubes_hollow = ref_cubes_medsub.copy()
        ref_cubes_hollow[:, :, edge:-edge, edge:-edge] = np.nan

        # Create a 4D mask with zeros for reference
        ref_cubes_shape = ref_cubes_hollow.shape
        mask_4d = np.zeros(ref_cubes_shape, dtype=bool)
        mask_ref = np.tile(mask[0:1], (ref_cubes_shape[0], 1, 1))

        for i in range(mask_4d.shape[1]):
            temp = mask_4d[:, i, :, :]
            temp[mask_ref] = True
            mask_4d[:, i, :, :] = temp
        ref_cubes_hollow[~mask_4d] = np.nan

        # Now run the routine to figure out which groups to mask
        best_faint_maxgrps = []
        best_bright_maxgrps = []
        ref_peak_pixels = []
        for i, scif in enumerate(sci_frames_hollow):
            for j, refc in enumerate(ref_cubes_hollow):
                # Need to save the peak pixel from each reference, as we'll use this for
                # the REF_BG frame interpolations
                if i == 0:
                    ref_peak_pixels.append(np.nanmax(ref_cubes_medsub[j][-1]))
                this_faint_diffs = []
                this_bright_diffs = []
                for refg in refc:
                    faint_diff = np.abs(np.nanmedian(refg) - np.nanmedian(scif))
                    this_faint_diffs.append(faint_diff)
                for refg in ref_cubes_medsub[j]:
                    bright_diff = np.abs(np.nanmax(refg) - np.nanmax(sci_frames_medsub[i]))
                    this_bright_diffs.append(bright_diff)

                best_faint_maxgrp = np.argmin(this_faint_diffs)
                best_faint_maxgrps.append(best_faint_maxgrp)

                best_bright_maxgrp = np.argmin(this_bright_diffs)
                best_bright_maxgrps.append(best_bright_maxgrp)

        maxgrps_faint = int(np.nanmedian(best_faint_maxgrps))
        maxgrps_bright = int(np.nanmedian(best_bright_maxgrps))

        steps['mask_groups']['maxgrps_faint'] = maxgrps_faint
        steps['mask_groups']['maxgrps_bright'] = maxgrps_bright
        steps['mask_groups']['cropmask'] = mask
        steps['mask_groups']['refbg_maxcounts'] = np.nanmedian(ref_peak_pixels)

    # Now read in the specific reference file we're looking at.
    with fits.open(refpath) as hdul:
        refshape = hdul['SCI'].data.shape
        ref_ints_slice = hdul['SCI'].data[:, -1, :, :].astype(float)
        ref_slice = np.nanmedian(ref_ints_slice, axis=0)
        ref_slice -= np.nanmedian(ref_slice)
        ref_crpix_x, ref_crpix_y = hdul["SCI"].header["CRPIX1"], hdul["SCI"].header["CRPIX2"]

    # Get the peak pixel count in the last group, only look at the PSF core
    xlo = int(ref_crpix_x) - crop
    xhi = int(ref_crpix_x) + crop
    ylo = int(ref_crpix_y) - crop
    yhi = int(ref_crpix_y) + crop
    ref_slice_cropped = ref_slice[ylo:yhi, xlo:xhi]

    if 'BG' in reftype:
        maxcounts = steps['mask_groups']['refbg_maxcounts']
    else:
        maxcounts = np.nanmax(ref_slice_cropped)

    # Get the median pixel count in our mask area from earlier
    ref_slice_hollow = ref_slice_cropped.copy()
    ref_slice_hollow[edge:-edge, edge:-edge] = np.nan
    all_mincounts = []
    for saved_mask in steps['mask_groups']['cropmask']:
        ref_slice_masked = ref_slice_hollow.copy()
        ref_slice_masked[~saved_mask] = np.nan
        all_mincounts.append(np.nanmedian(ref_slice_hollow))
    mincounts = np.nanmedian(all_mincounts)

    # Now make an interpolation connecting counts to the number of groups to be masked
    maxgrps_interp = interp1d([maxcounts, mincounts],
                              [steps['mask_groups']['maxgrps_bright'], steps['mask_groups']['maxgrps_faint']],
                              kind='linear',
                              bounds_error=False,
                              fill_value=(steps['mask_groups']['maxgrps_bright'],
                                          steps['mask_groups']['maxgrps_faint']))

    # Now use the interpolation to set the mask array, zero values will be included in the ramp fit
    mask_array = np.zeros(refshape, dtype=bool)
    for ri in range(mask_array.shape[2]):
        for ci in range(mask_array.shape[3]):
            if ref_slice[ri, ci] >= mincounts:
                # Determine number of groups
                this_grps = int(maxgrps_interp(ref_slice[ri, ci]))
                groups_to_mask = np.arange(this_grps + 1, refshape[1])
                mask_array[:, groups_to_mask, ri, ci] = 1

    # Assign the mask array to the steps dictionary
    steps['mask_groups']['mask_array'] = mask_array

    return steps


class MaskGroupsStep(Step):
    """
    Mask particular groups prior to ramp fitting
    """
    class_alias = "maskgroups"

    spec = """
        mask_sigma_med = float(default=3) #Only mask pixels Nsigma above the median
        mask_window = integer(default=2) #Also mask pixels within N pixels of a masked pixel
    """

    def process(self, input):
        """Mask particular groups prior to ramp fitting"""
        with datamodels.open(input) as input_model:
            datamodel = input_model.copy()

            # Set particular groups to DO_NOT_USE
            datamodel.groupdq[self.mask_array] = 1

        return datamodel



