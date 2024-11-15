from __future__ import division

import matplotlib

# =============================================================================
# IMPORTS
# =============================================================================

import os
import pdb
import sys

import astropy.io.fits as fits
import numpy as np

from tqdm import trange

from jwst.lib import reffile_utils
from jwst.stpipe import Step
from jwst import datamodels
from jwst.datamodels import dqflags, RampModel, SaturationModel
from jwst.pipeline import Detector1Pipeline, Image2Pipeline, Coron3Pipeline
from .fnoise_clean import kTCSubtractStep, OneOverfStep
from .expjumpramp import ExperimentalJumpRampStep
from webbpsf_ext import robust

from scipy.interpolate import interp1d
from skimage.metrics import structural_similarity

import warnings
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# =============================================================================
# MAIN
# =============================================================================

class Coron1Pipeline_spaceKLIP(Detector1Pipeline):
    """
    The spaceKLIP JWST stage 1 pipeline class.
    
    Apply all calibration steps to raw JWST ramps to produce 
    a 2-D slope product. Custom sub-class of ``Detector1Pipeline`` 
    with modifications for coronagraphic data.
    
    Included steps are: group_scale, dq_init, saturation, ipc, 
    superbias, refpix, rscd, firstframe, lastframe, linearity, 
    dark_current, reset, persistence, jump detection, ramp_fit, 
    and gain_scale. 
    """
    
    class_alias = "calwebb_coron1"

    spec = """
        save_intermediates = boolean(default=False) # Save all intermediate step results
        rate_int_outliers  = boolean(default=False) # Flag outlier pixels in rateints
        return_rateints    = boolean(default=False) # Return rateints or rate product?
        stage_1overf       = string(default='ints') # Where in process to perform 1/f noise removal; groups or ints
    """
    
    def __init__(self,
                 **kwargs):
        """
        Initialize the spaceKLIP JWST stage 1 pipeline class.
        
        Parameters
        ----------
        **kwargs : keyword arguments
            Default JWST stage 1 detector pipeline keyword arguments.
        
        Returns
        -------
        None.
        
        """

        self.step_defs['mask_groups'] = MaskGroupsStep
        self.step_defs['subtract_ktc'] = kTCSubtractStep
        self.step_defs['subtract_1overf'] = OneOverfStep
        self.step_defs['experimental_jumpramp'] = ExperimentalJumpRampStep
        
        # Initialize Detector1Pipeline class.
        super(Coron1Pipeline_spaceKLIP, self).__init__(**kwargs)
        
        # Set additional parameters in saturation step
        self.saturation.grow_diagonal = False
        self.saturation.flag_rcsat = False

        # Initialize reference pixel correction parameters
        self.refpix.nlower = 4
        self.refpix.nupper = 4
        self.refpix.nrow_off = 0
        # NOTE: nleft, right, and ncol_off don't actually do anything.
        #   For 1/f noise correction, use the `subtract_1overf` Step
        #   to model and subtract the 1/f noise. On by default.
        self.refpix.nleft = 0
        self.refpix.nright = 0
        self.refpix.ncol_off = 0

        # Ramp fit saving options
        # NOTE: `save_calibrated_ramp` is already a Detector1Pipeline property
        self.ramp_fit.save_calibrated_ramp = False
        
    
    def process(self,
                input):
        """
        Process an input JWST datamodel with the spaceKLIP JWST stage 1
        pipeline.
        
        Parameters
        ----------
        input : jwst.datamodel
            Input JWST datamodel to be processed.
        
        Returns
        -------
        output : jwst.datamodel
            Output JWST datamodel.
        
        """
        
        # Open input as ramp model.
        input = RampModel(input)
        
        # Process MIR & NIR exposures differently.
        instrument = input.meta.instrument.name
        if instrument == 'MIRI':
            input = self.run_step(self.group_scale, input)
            input = self.run_step(self.dq_init, input)
            input = self.run_step(self.saturation, input)
            #input = self.run_step(self.ipc, input) Not run for MIRI
            input = self.run_step(self.firstframe, input)
            input = self.run_step(self.lastframe, input)
            input = self.run_step(self.reset, input)
            input = self.run_step(self.linearity, input)
            input = self.run_step(self.rscd, input)
            input = self.run_step(self.dark_current, input)
            input = self.run_step(self.refpix, input)
            #input = self.run_step(self.charge_migration, input) Not run for MIRI
            input = self.run_step(self.jump, input)
            input = self.run_step(self.mask_groups, input)
        else:
            input = self.run_step(self.group_scale, input)
            input = self.run_step(self.dq_init, input)
            input = self.do_saturation(input)
            input = self.run_step(self.ipc, input)
            input = self.run_step(self.superbias, input)
            input = self.do_refpix(input)
            input = self.run_step(self.linearity, input)
            input = self.run_step(self.persistence, input)
            input = self.run_step(self.dark_current, input)
            input = self.run_step(self.charge_migration, input)
            input = self.run_step(self.jump, input)
            input = self.run_step(self.subtract_ktc, input)
            #1overf Only present in NIR data
            if 'groups' in self.stage_1overf:
                input = self.run_step(self.subtract_1overf, input)
        
        # save the corrected ramp data, if requested
        if self.ramp_fit.save_calibrated_ramp or self.save_calibrated_ramp or self.save_intermediates:
            self.save_model(input, suffix='ramp')
        
        # Run ramp fitting & gain scale correction.
        if not self.experimental_jumpramp.use:
            # Use the default ramp fitting procedure
            res = self.run_step(self.ramp_fit, input, save_results=False)
            rate, rateints = (res, None) if self.ramp_fit.skip else res
        else:
            # Use the experimental fitting procedure
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                res = self.run_step(self.experimental_jumpramp, input)
                rate, rateints = res
        
        if self.rate_int_outliers and rateints is not None:
            # Flag additional outliers by comparing rateints and refit ramp
            input = self.apply_rateint_outliers(rateints, input)
            if input is None:
                input, ints_model = rate, rateints
            else:
                res = self.run_step(self.ramp_fit, input, save_results=False)
                input, ints_model = (res, None) if self.ramp_fit.skip else res
        else:
            # input is the rate product, ints_model is the rateints product
            input, ints_model = rate, rateints

        if input is None:
            self.ramp_fit.log.info('NoneType returned from ramp fitting. Gain scale correction skipped')
        else:
            self.gain_scale.suffix = 'rate'
            input = self.run_step(self.gain_scale, input, save_results=False)

        # Perform 1/f correction on rateints if requested
        if ('ints' in self.stage_1overf) and ('MIRI' not in instrument):
            # Apply 1/f noise correction to rateints
            if ints_model is not None:
                ints_model = self.run_step(self.subtract_1overf, ints_model)
            if (input is not None) and (ints_model is not None):
                # TODO: Find better method to average rateints. Weighted mean?
                input.data = np.nanmean(ints_model.data, axis=0)

        # apply the gain scale step to the multi-integration product,
        # if it exists, and then save it
        if ints_model is not None:
            self.gain_scale.suffix = 'rateints'
            ints_model = self.run_step(self.gain_scale, ints_model, save_results=False)
            if self.save_results or self.save_intermediates:
                self.save_model(ints_model, suffix='rateints')

        # Setup output file.
        self.setup_output(input)

        if self.return_rateints:
            return ints_model
        
        return input

    def setup_output(self, input):
        """Determine output file name suffix"""
        if input is None:
            return None
        # Determine the proper file name suffix to use later
        if input.meta.cal_step.ramp_fit == 'COMPLETE':
            if self.return_rateints:
                self.suffix = 'rateints'
            else:
                self.suffix = 'rate'
        else:
            self.suffix = 'ramp'

    def run_step(self,
                 step_obj,
                 input,
                 save_results=None,
                 **kwargs):
        """
        Run a JWST pipeline step.
        
        Parameters
        ----------
        step_obj : jwst.step
            JWST pipeline step to be run.
        input : jwst.datamodel
            Input JWST datamodel to be processed.
        save_results : bool, optional
            Save the JWST pipeline step product? None will default to the JWST
            pipeline step default. The default is None.
        **kwargs : keyword arguments
            Default JWST pipeline step keyword arguments.
        
        Returns
        -------
        res : jwst.datamodel
            Output JWST datamodel.
        
        """
        
        # Determine if we're saving results for real
        if step_obj.skip:
            # Skip save if step is skipped
            really_save_results = False
        elif (save_results is not None):
            # Use keyword specifications
            really_save_results = save_results
        elif self.save_intermediates:
            # Use save_intermediates attribute
            really_save_results = True
        elif step_obj.save_results:
            # Use step attribute
            really_save_results = True
        else:
            # Saving is unspecified
            really_save_results = False
        
        # Run step. Don't save results yet.
        step_save_orig = step_obj.save_results
        step_obj.save_results = False
        res = step_obj(input)
        step_obj.save_results = step_save_orig
        
        # Check if group scale correction or gain scale correction were skipped.
        if step_obj is self.group_scale:
            if res.meta.cal_step.group_scale == 'SKIPPED':
                really_save_results = False
        elif step_obj is self.gain_scale:
            if res.meta.cal_step.gain_scale == 'SKIPPED':
                really_save_results = False
        
        # Save results.
        if really_save_results:
            step_obj.output_dir = self.output_dir
            if isinstance(res, (tuple)):
                self.save_model(res[0], suffix=step_obj.suffix+'0')
                self.save_model(res[1], suffix=step_obj.suffix+'1')
            else:
                self.save_model(res, suffix=step_obj.suffix)
        
        return res
    
    def do_saturation(self,
                      input,
                      **kwargs):
        """
        Do the default or a custom saturation correction.
        
        Parameters
        ----------
        input : jwst.datamodel
            Input JWST datamodel to be processed.
        **kwargs : keyword arguments
            Default JWST stage 1 saturation step keyword arguments.
        
        Returns
        -------
        res : jwst.datamodel
            Output JWST datamodel.
        
        """
        from .utils import expand_mask
        
        # Save original step parameter.
        npix_grow = self.saturation.n_pix_grow_sat
        
        # Flag RC pixels as saturated?
        flag_rcsat = self.saturation.flag_rcsat
        if flag_rcsat:
            mask_rc = (input.pixeldq & dqflags.pixel['RC']) > 0
            # Do a bitwise OR of RC mask with groupdq to flip saturation bits
            input.groupdq = input.groupdq | (mask_rc * dqflags.pixel['SATURATED'])

        # Run step with default settings.
        if self.saturation.grow_diagonal or npix_grow == 0:
            res = self.run_step(self.saturation, input, **kwargs)
        else: # No diagonal growth
            # Initial run with 0 pixel growth
            self.saturation.n_pix_grow_sat = 0
            res = self.run_step(self.saturation, input, **kwargs)
            # Reset to original value
            self.saturation.n_pix_grow_sat = npix_grow

            self.saturation.log.info(f'Growing saturation flags by {npix_grow} pixels. Ignoring diagonal growth.')
            # Update saturation dq flags to grow in vertical and horizontal directions
            # Get saturation mask
            mask_sat = (res.groupdq & dqflags.pixel['SATURATED']) > 0

            # Expand the mask by npix_grow pixels
            mask_sat = expand_mask(mask_sat, npix_grow, grow_diagonal=False)

            # Do a bitwise OR of new mask with groupdq to flip saturation bit
            res.groupdq = res.groupdq | (mask_sat * dqflags.pixel['SATURATED'])

            # Do the same for the zero frames
            zframes = res.zeroframe if res.meta.exposure.zero_frame else None
            if zframes is not None:
                # Saturated zero frames have already been set to 0
                mask_sat = (zframes==0) | mask_rc if flag_rcsat else (zframes==0)
                # Expand the mask by npix_grow pixels
                mask_sat = expand_mask(mask_sat, npix_grow, grow_diagonal=False)
                # Set saturated pixels to 0 in zero frames
                res.zeroframe[mask_sat] = 0

        return res
    
    def apply_rateint_outliers(self, 
                               rateints_model, 
                               ramp_model, 
                               **kwargs):
        """Get pixel outliers in rateint model and apply to ramp model DQ
        
        Parameters
        ----------
        rateints_model : `~jwst.datamodels.CubeModel`
            Rateints model to use for outlier detection
        ramp_model : `~jwst.datamodels.RampModel`
            Ramp model to update with outlier flags

        Keyword Args
        ------------
        sigma_cut : float
            Sigma cut for outlier detection.
            Default is 10.
        nint_min : int
            Minimum number of integrations required for outlier detection.
            Default is 10.
        """
        from .utils import cube_outlier_detection

        inst = rateints_model.meta.instrument.name
        data = rateints_model.data[1:] if 'MIRI' in inst else rateints_model.data

        indbad = cube_outlier_detection(data, **kwargs)

        # Reshape outlier mask to match ramp data
        nint, ng, ny, nx = ramp_model.data.shape
        bpmask = indbad.reshape([nint, 1, ny, nx])
        if ng>1:
            bpmask = np.repeat(bpmask, ng, axis=1)

        # Update DO_NOT_USE and JUMP_DET flags
        mask_dnu = (ramp_model.groupdq & dqflags.pixel['DO_NOT_USE']) > 0
        mask_jd  = (ramp_model.groupdq & dqflags.pixel['JUMP_DET']) > 0

        # Update DO_NOT_USE and JUMP_DET flags with outlier mask
        mask_dnu = mask_dnu | bpmask
        mask_jd  = mask_jd  | bpmask

        # Update ramp model groupdq
        ramp_model.groupdq = ramp_model.groupdq | (mask_dnu * dqflags.pixel['DO_NOT_USE'])
        ramp_model.groupdq = ramp_model.groupdq | (mask_jd  * dqflags.pixel['JUMP_DET'])

        return ramp_model

    def do_refpix(self,
                  input,
                  **kwargs):
        """ Do the default or a custom pseudo reference pixel correction.
        
        If full frame, perform RefPix as normal.
        If no ref rows or columns specified, then perform RefPix as normal.

        Otherwise, temporarily set reference rows and columns in the DQ flags
        and force the reference correction. Can set number of reference
        rows and column via the  `nlower`, `nupper`, `nleft`, `nright` 
        attributes.

        Parameters
        ----------
        input : jwst.datamodel
            Input JWST datamodel to be processed.
        **kwargs : keyword arguments
            Default JWST stage 1 refpix step keyword arguments.
        
        Returns
        -------
        res : jwst.datamodel
            Output JWST datamodel.
        
        """
        
        # Is this a full frame observation?
        is_full_frame = 'FULL' in input.meta.subarray.name.upper()

        # Get number of reference pixels explicitly specified
        nlower = self.refpix.nlower
        nupper = self.refpix.nupper
        # The pipeline does not currently support col refpix correction for subarrays
        # nleft  = self.refpix.nleft 
        # nright = self.refpix.nright
        nref_set = nlower + nupper #+ nleft + nright
        
        # Perform normal operations if full frame or no refpix pixels specified
        if is_full_frame or nref_set==0:
            return self.run_step(self.refpix, input, **kwargs)
        else:
            return self.do_pseudo_refpix(input, **kwargs)
    
    def do_pseudo_refpix(self,
                         input,
                         **kwargs):
        """ Do a pseudo reference pixel correction
        
        Flag the requested edge rows and columns as reference pixels, 
        run the JWST stage 1 refpix step, and then unflag those 
        "pseudo" reference pixels.
        
        Parameters
        ----------
        input : jwst.datamodel
            Input JWST datamodel to be processed.
        **kwargs : keyword arguments
            Default JWST stage 1 refpix step keyword arguments.
        
        Returns
        -------
        res : jwst.datamodel
            Output JWST datamodel.

        """
        
        # Get number of custom reference pixel rows & columns.
        nlower = self.refpix.nlower
        nupper = self.refpix.nupper
        nleft = self.refpix.nleft
        nright = self.refpix.nright
        nrow_off = self.refpix.nrow_off
        ncol_off = self.refpix.ncol_off

        # Flag custom reference pixel rows & columns.
        self.refpix.log.info(f'Flagging [{nlower}, {nupper}] references rows at [bottom, top] of array')
        self.refpix.log.info(f'Flagging [{nleft}, {nright}] references columns at [left, right] of array')

        # Update pixel DQ mask to manually set reference pixels
        log.info(f'Flagging [{nlower}, {nupper}] references rows at [bottom, top] of array')
        log.info(f'Flagging [{nleft}, {nright}] references rows at [left, right] of array')
        pixeldq_orig = input.pixeldq.copy()
        if nlower>0:
            ib1 = nrow_off
            ib2 = ib1 + nlower
            input.pixeldq[ib1:ib2,:] = input.pixeldq[ib1:ib2,:]  | dqflags.pixel['REFERENCE_PIXEL']
        if nupper>0:
            it1 = -1 * (nupper + nrow_off)
            it2 = None if nrow_off == 0 else -1 * nrow_off
            input.pixeldq[it1:it2,:] = input.pixeldq[it1:it2,:] | dqflags.pixel['REFERENCE_PIXEL']
        if nleft>0:
            il1 = ncol_off
            il2 = il1 + nleft
            input.pixeldq[:,il1:il2] = input.pixeldq[:,il1:il2] | dqflags.pixel['REFERENCE_PIXEL']
        if nright>0:
            ir1 = -1 * (nright + ncol_off)
            ir2 = None if ncol_off == 0 else -1 * ncol_off
            input.pixeldq[:,ir1:ir2] = input.pixeldq[:,ir1:ir2] | dqflags.pixel['REFERENCE_PIXEL']

        # Turn off side reference pixels?
        use_side_orig = self.refpix.use_side_ref_pixels 
        if nleft + nright == 0:
            self.refpix.use_side_ref_pixels = False
        
        # Run step with custom reference pixel rows & columns.
        res = self.run_step(self.refpix, input, **kwargs)
        
        # Apply original step parameter.
        self.refpix.use_side_ref_pixels = use_side_orig
        
        # Unflag custom reference pixel rows & columns.
        self.refpix.log.info('Removing custom reference pixel flags')
        if nlower>0:
            res.pixeldq[ib1:ib2,:] = pixeldq_orig[ib1:ib2,:]
        if nupper>0:
            res.pixeldq[it1:it2,:] = pixeldq_orig[it1:it2,:]
        if nleft>0:
            res.pixeldq[:,il1:il2] = pixeldq_orig[:,il1:il2]
        if nright>0:
            res.pixeldq[:,ir1:ir2] = pixeldq_orig[:,ir1:ir2]

        return res

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
    save_calibrate_ramp : bool
        Save intermediate step that is the calibrated ramp? 
        Default is False.
    save_intermediates : bool, optional
        Save intermediate steps, such as dq_init, saturation, refpix,
        jump, linearity, ramp, etc. Default is False.
    return_rateints : bool, optional
        Return the rateints model instead of rate? Default is False.
    rate_int_outliers : bool, optional
        Flag outlier pixels in rateints? Default is False.
        Uses the `cube_outlier_detection` function and requires
        a minimum of 5 integrations.
    flag_rcsat : bool, optional
        Flag known RC pixels as always saturated? Default is False.
    stage_1overf : str, optional
        Where in the pipeline process to perform 1/f noise removal?
        Either at the 'groups' or 'ints' level. Default is 'ints'.
    skip_ktc : bool, optional
        Remove kTC noise by fitting ramp data to get bias? 
        Useful for looking at linearized ramp data.
        Default: False.
    skip_fnoise : bool, optional
        Skip 1/f noise removal? Default: False.
    skip_fnoise_vert : bool, optional
        Skip removal of vertical striping? Default: False.
        Not applied if 1/f noise correction is skipped.
    skip_charge : bool, optional
        Skip charge migration flagging step? Default: False.
    skip_jump : bool, optional
        Skip jump detection step? Default: False.
    skip_dark : bool, optional
        Skip dark current subtraction step? Default is True for 
        subarrays and False for full frame data.
        Dark current cal files for subarrays are really low SNR.
    skip_ipc : bool, optional
        Skip IPC correction step? Default: False.
    skip_persistence : bool, optional
        Skip persistence correction step? Default: True.
        Doesn't currently do anything.

    Returns
    -------
    Pipeline output, either rate or rateint data model.

    """

    from webbpsf_ext.analysis_tools import nrc_ref_info

    # Print all info message if verbose, otherwise only errors or critical.
    from .logging_tools import all_logging_disabled
    log_level = logging.INFO if verbose else logging.ERROR

    # Create output directory if it doesn't exist.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize Coron1Pipeline.
    with all_logging_disabled(log_level):
        pipeline = Coron1Pipeline_spaceKLIP(output_dir=output_dir)

    # Options for saving results
    pipeline.save_results         = kwargs.get('save_results', True)
    pipeline.save_calibrated_ramp = kwargs.get('save_calibrated_ramp', False)
    pipeline.save_intermediates   = kwargs.get('save_intermediates', False)
    pipeline.return_rateints      = kwargs.get('return_rateints', False)

    # Level of data product to perform 1/f correction ('groups' or 'ints')
    pipeline.stage_1overf = kwargs.get('stage_1overf', 'ints')

    # Skip certain steps?
    pipeline.charge_migration.skip = kwargs.get('skip_charge', False)
    pipeline.jump.skip             = kwargs.get('skip_jump', False)
    pipeline.ipc.skip              = kwargs.get('skip_ipc', False)
    pipeline.persistence.skip      = kwargs.get('skip_persistence', True)
    pipeline.subtract_ktc.skip     = kwargs.get('skip_ktc', False)
    pipeline.subtract_1overf.skip  = kwargs.get('skip_fnoise', False)
    skip_vert = kwargs.get('skip_fnoise_vert', False)
    pipeline.subtract_1overf.vertical_corr = not skip_vert

    # Skip dark current for subarray by default, but not full frame
    skip_dark     = kwargs.get('skip_dark', None)
    if skip_dark is None:
        hdr0 = fits.getheader(fitspath, ext=0)
        is_full_frame = 'FULL' in hdr0['SUBARRAY']
        skip_dark = False if is_full_frame else True
    pipeline.dark_current.skip = skip_dark

    # Determine reference pixel correction parameters based on
    # instrument aperture name for NIRCam
    hdr0 = fits.getheader(fitspath, 0)
    if hdr0['INSTRUME'] == 'NIRCAM':
        # Array of reference pixel borders [lower, upper, left, right]
        nb, nt, nl, nr = nrc_ref_info(hdr0['APERNAME'], orientation='sci')
    else:
        nb, nt, nl, nr = (0, 0, 0, 0)
    # If everything is 0, set to default to 4 around the edges
    if nb + nt == 0:
        nb = nt = 4
    if nl + nr == 0:
        nl = nr = 4
    pipeline.refpix.nlower   = kwargs.get('nlower', nb)
    pipeline.refpix.nupper   = kwargs.get('nupper', nt)
    pipeline.refpix.nrow_off = kwargs.get('nrow_off', 0)

    # Set some Step parameters
    pipeline.jump.rejection_threshold              = kwargs.get('rejection_threshold', 4)
    pipeline.jump.three_group_rejection_threshold  = kwargs.get('three_group_rejection_threshold', 4)
    pipeline.jump.four_group_rejection_threshold   = kwargs.get('four_group_rejection_threshold', 4)
    pipeline.saturation.n_pix_grow_sat = kwargs.get('n_pix_grow_sat', 1)
    pipeline.saturation.grow_diagonal  = kwargs.get('grow_diagonal', False)
    pipeline.saturation.flag_rcsat     = kwargs.get('flag_rcsat', False)
    pipeline.rate_int_outliers         = kwargs.get('rate_int_outliers', False)

    # Skip pixels with only 1 group in ramp_fit?
    pipeline.ramp_fit.suppress_one_group = kwargs.get('suppress_one_group', False)
    # Number of processor cores to use during ramp fitting process
    # 'none', 'quarter', 'half', or 'all'
    pipeline.ramp_fit.maximum_cores      = kwargs.get('maximum_cores', 'none')

    # Set parameters from step dictionary
    for key1 in steps.keys():
        for key2 in steps[key1].keys():
            setattr(getattr(pipeline, key1), key2, steps[key1][key2])

    #Override jump & ramp if necessary
    if pipeline.experimental_jumpramp.use:
        log.info("Experimental jump/ramp fitting selected, regular jump and ramp will be skipped...")
        pipeline.jump.skip = True
        pipeline.ramp_fit.skip = True
    
    # Run Coron1Pipeline. Raise exception on error.
    # Ensure that pipeline is closed out.
    try:
        with all_logging_disabled(log_level):
            res = pipeline.run(fitspath)
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

    if isinstance(res, list):
        res = res[0]

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
    save_calibrate_ramp : bool
        Save intermediate step that is the calibrated ramp? 
        Default is False.
    save_intermediates : bool, optional
        Save intermediate steps, such as dq_init, saturation, refpix,
        jump, linearity, ramp, etc. Default is False.
    return_rateints : bool, optional
        Return the rateints model instead of rate? Default is False.
    rate_int_outliers : bool, optional
        Flag outlier pixels in rateints? Default is False.
        Uses the `cube_outlier_detection` function and requires
        a minimum of 5 integrations.
    flag_rcsat : bool, optional
        Flag known RC pixels as always saturated? Default is False.
    stage_1overf : str, optional
        Where in the pipeline process to perform 1/f noise removal?
        Either at the 'groups' or 'ints' level. Default is 'ints'.
    skip_ktc : bool, optional
        Remove kTC noise by fitting ramp data to get bias? 
        Useful for looking at linearized ramp data.
        Default: False.
    skip_fnoise : bool, optional
        Skip 1/f noise removal? Default: False.
    skip_fnoise_vert : bool, optional
        Skip removal of vertical striping? Default: False.
        Not applied if 1/f noise correction is skipped.
    skip_charge : bool, optional
        Skip charge migration flagging step? Default: False.
    skip_jump : bool, optional
        Skip jump detection step? Default: False.
    skip_dark : bool, optional
        Skip dark current subtraction step? Default is True for 
        subarrays and False for full frame data.
        Dark current cal files for subarrays are really low SNR.
    skip_ipc : bool, optional
        Skip IPC correction step? Default: False.
    skip_persistence : bool, optional
        Skip persistence correction step? Default: True.
        Doesn't currently do anything.

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

    groupmaskflag = 0 # Set flag for group masking
    skip_revert = False # Set flag for skipping a file
    # Loop through concatenations.
    for i in itervals:
        key = keys[i]
        if not quiet: log.info('--> Concatenation ' + key)

        # Loop through FITS files.
        nfitsfiles = len(database.obs[key])
        jtervals = trange(nfitsfiles, desc='FITS files', leave=False) if quiet else range(nfitsfiles)
        for j in jtervals:

            # Skip non-stage 0 files.
            head, tail = os.path.split(database.obs[key]['FITSFILE'][j])
            fitspath = os.path.abspath(database.obs[key]['FITSFILE'][j])
            if database.obs[key]['DATAMODL'][j] != 'STAGE0':
                if not quiet: log.info('  --> Coron1Pipeline: skipping non-stage 0 file ' + tail)
                continue

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

            if (j == jtervals[-1]) and (groupmaskflag == 1):
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
                    sci_frame -= np.nanmedian(sci_frame, axis=(1,2), keepdims=True)

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
                    ref_cube -= np.nanmedian(ref_cube, axis=(2,3), keepdims=True)

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
                            diff = np.abs(np.nansum(ref_group)-np.nansum(sci_last_group))
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
        groups_to_mask = np.arange(final_max_grp_to_use+1, ref_cubes[0].shape[1])

        # Make the mask array
        mask_array = np.zeros(ref_shape, dtype=bool)
        mask_array[:,groups_to_mask,:,:] = 1

        # Assign to steps so this stage doesn't get repeated. 
        steps['mask_groups']['mask_array'] = mask_array
        print(mask_array)

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

        sci_frames_modified = np.nanmedian(sci_frames_cropped, axis=1) #Median over integrations

        # Crop and median reference
        ref_cubes_cropped = []
        for i, refc in enumerate(ref_cubes):
            crpix_x, crpix_y = ref_crpixs[i]
            xlo = int(crpix_x) - crop
            xhi = int(crpix_x) + crop
            ylo = int(crpix_y) - crop
            yhi = int(crpix_y) + crop
            ref_cubes_cropped.append(refc[:, :, ylo:yhi, xlo:xhi])

        ref_cubes_modified = np.nanmedian(ref_cubes_cropped, axis=1) #Median over integrations

        # Median subtract the images
        sci_frames_medsub = sci_frames_modified - np.nanmedian(sci_frames_modified, axis=(1,2), keepdims=True)
        ref_cubes_medsub = ref_cubes_modified - np.nanmedian(ref_cubes_modified, axis=(2,3), keepdims=True)

        # Flatten array and find indices above percentile threshold value
        sci_frames_flat = np.reshape(sci_frames_medsub, (sci_frames_medsub.shape[0], -1))
        per = np.percentile(sci_frames_flat, [threshold])
        above_threshold_indices = np.where(sci_frames_medsub > per)

        # Create an empty mask array
        mask = np.zeros_like(sci_frames_medsub, dtype=bool)

        # Define function to expand indices and update mask
        def expand_and_update_mask(indices, mask, xpad=2, ypad=2):
            for z, x, y in zip(*indices):
                mask[z, max(0, x-xpad):min(mask.shape[1], x+xpad), max(0, y-ypad):min(mask.shape[2], y+ypad)] = True

        # Expand indices and update mask for each 2D slice
        for z_slice in range(sci_frames_medsub.shape[0]):
            indices_slice = np.where(above_threshold_indices[0] == z_slice)
            expand_and_update_mask((above_threshold_indices[0][indices_slice], above_threshold_indices[1][indices_slice], above_threshold_indices[2][indices_slice]), mask)

        # Okay now make some hollowed out cropped frames, to focus on fainter, but still bright, pixels
        sci_frames_hollow = sci_frames_medsub.copy()
        sci_frames_hollow[:, edge:-edge, edge:-edge] = np.nan
        sci_frames_hollow[~mask] = np.nan

        ref_cubes_hollow = ref_cubes_medsub.copy()
        ref_cubes_hollow[:, :, edge:-edge, edge:-edge] = np.nan

        # Create a 4D mask with zeros for reference
        ref_cubes_shape = ref_cubes_hollow.shape
        mask_4d = np.zeros(ref_cubes_shape, dtype=bool)
        mask_ref = np.tile(mask[0:1], (ref_cubes_shape[0],1,1))

        for i in range(mask_4d.shape[1]):
            temp = mask_4d[:,i,:,:]
            temp[mask_ref] = True
            mask_4d[:,i,:,:] = temp
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
                this_faint_diffs= []
                this_bright_diffs = []
                for refg in refc:
                    faint_diff = np.abs(np.nanmedian(refg)-np.nanmedian(scif))
                    this_faint_diffs.append(faint_diff)
                for refg in ref_cubes_medsub[j]:
                    bright_diff = np.abs(np.nanmax(refg)-np.nanmax(sci_frames_medsub[i]))
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
        ref_ints_slice = hdul['SCI'].data[:,-1,:,:].astype(float)
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
                              [steps['mask_groups']['maxgrps_bright'],steps['mask_groups']['maxgrps_faint']], 
                              kind='linear', 
                              bounds_error=False, 
                              fill_value=(steps['mask_groups']['maxgrps_bright'],steps['mask_groups']['maxgrps_faint']))

    # Now use the interpolation to set the mask array, zero values will be included in the ramp fit
    mask_array = np.zeros(refshape, dtype=bool)
    for ri in range(mask_array.shape[2]):
        for ci in range(mask_array.shape[3]):
            if ref_slice[ri, ci] >= mincounts:
                # Determine number of groups
                this_grps = int(maxgrps_interp(ref_slice[ri, ci]))
                groups_to_mask = np.arange(this_grps+1, refshape[1])
                mask_array[:,groups_to_mask,ri,ci] = 1

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



