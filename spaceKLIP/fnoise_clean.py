import os
import numpy as np

import matplotlib.pyplot as plt

import multiprocessing as mp

from jwst.stpipe import Step
from jwst import datamodels
from jwst.datamodels import dqflags

from webbpsf_ext import robust
from webbpsf_ext.image_manip import expand_mask

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class kTCSubtractStep(Step):
    """
    kTCSubtractStep: Subtract kTC noise from ramp data.
    """

    class_alias = "removektc"

    spec = """
        sat_frac = float(default=0.5) # Maximum saturation fraction for fitting
    """

    def process(self, input):
        """Subtract kTC noise from data."""
        
        with datamodels.open(input) as input_model:
            datamodel = input_model.copy()
            bias_arr, _ = fit_slopes_to_ramp_data(datamodel, sat_frac=self.sat_frac)

            # Subtract bias from each integration
            nints = datamodel.meta.exposure.nints
            for i in range(nints):
                datamodel.data[i] -= bias_arr[i]

        return datamodel


def nfclean_mulitprocess_helper(args):
    """ Helper function for multiprocessing 
    
    The `args` parameter should be a tuple consisting of:
    (im_diff, im_mask, noutputs, slowaxis, flatten_model, model_type, vertical_corr)
    """

    im_diff, im_mask, noutputs, slowaxis, flatten_model, model_type, vertical_corr = args

    # Select which clean function to use
    nf_clean = make_clean_class(im_diff, im_mask, noutputs, slowaxis, 
                                flatten_model=flatten_model)

    # Perform the fit and return model
    nf_clean.fit(model_type=model_type, vertical_corr=vertical_corr)
    model = nf_clean.model.copy()
    del nf_clean

    return model

class OneOverfStep(Step):
    """ OneOverfStep: Apply 1/f noise correction

    Works on `RampModel`, `CubeModel`, and `ImageModel` data products.

    For `RampModel`, it creates a master slope image, then subtracts
    a group-dependent flux image to produce a residual image. Then for
    each group residual image, we model the 1/f noise using a Savitzky-Golay 
    filter. 
    
    For `CubeModel`, we create an average of all slope images, then subtract
    from each slope image to create a series of residual images. We then apply
    a Savitzky-Golay filter to each residual image to model the 1/f noise.

    Full frame channels are handled by first removing with common channel
    noise (taking into account flips in the readout direction), then channel-dependent 
    1/f noise is modeled and applied to a complete model of the full frame image.

    If only a single integration is present, then a mask is applied to exclude
    pixels with large flux values. These pixels are excluded during the model
    fitting process.
    """

    class_alias = "1overf"

    spec = """
        model_type = option('median', 'mean', 'savgol', default='savgol') # Type of model to fit
        sat_frac = float(default=0.5) # Maximum saturation fraction for fitting
        combine_ints = boolean(default=True) # Combine integrations before ramp fitting
        vertical_corr = boolean(default=True) # Apply horizontal correction
        nproc = integer(default=4) # Number of processes to use
    """

    def __init__(self, *args, **kwargs):

        # Initialize Step class.
        super(OneOverfStep, self).__init__(*args, **kwargs)

        # Initialize worker arguments
        self._worker_arguments = None

    def _update_nproc(self, input):
        """Configure number of processes for multiprocessing"""

        # If we are in a multiprocess thread, then set self.nproc to 1
        # to avoid nested multiprocessing.
        # Spawned processes have a name like 'Process-1', 'Process-2', etc. or 'PoolWorker-'
        if ('Process-' in mp.current_process().name) or ('PoolWorker-' in mp.current_process().name):
            self.nproc = 1
            return

        if isinstance(input, datamodels.RampModel):
            nints, ngroups, _, _ = input.data.shape
            nframes_tot = nints * ngroups
        elif isinstance(input, datamodels.CubeModel):
            nints, _, _ = input.data.shape
            nframes_tot = nints
        elif isinstance(input, datamodels.ImageModel):
            nframes_tot = 1
        else:
            raise ValueError(f"Do not recognize {type(input)}")
        
        ncpu = os.cpu_count()

        # Set up number of processors for multiprocessing
        nproc = self.nproc
        nproc = nframes_tot if nproc > nframes_tot else nproc
        nproc = ncpu if nproc > ncpu else nproc
        self.nproc = nproc

    def process(self, input):
        """Apply 1/f noise correction to a JWST `RampModel`

        Parameters
        ==========
        input : JWST data model
            Input science data model to be corrected.
            Should be linearized.

        Returns
        =======
        output : JWST data model
            Output science data model with 1/f noise correction applied.
        """

        with datamodels.open(input) as input_model:
            self._update_nproc(input_model)

            if isinstance(input_model, datamodels.RampModel):
                oofn_model = self.proc_RampModel(input_model)
            elif isinstance(input_model, datamodels.CubeModel):
                oofn_model = self.proc_CubeModel(input)
            elif isinstance(input_model, datamodels.ImageModel):
                oofn_model = self.proc_ImageModel(input_model)
            else:
                raise ValueError(f"Do not recognize {type(input_model)}")
            
            datamodel = input_model.copy()

        self.oofn_model = oofn_model
        datamodel.data -= oofn_model

        del self._worker_arguments
        self._worker_arguments = None

        return datamodel

    def proc_RampModel(self, input):
        """Apply 1/f noise correction to a JWST `RampModel`

        Parameters
        ==========
        input : JWST data model
            Input science data model to be corrected.
            Should be a linearized `RampModel`.
        """

        # Get the input data model
        with datamodels.open(input) as input_model:

            # Create a copy of the input data model
            datamodel = input_model.copy()

            nints, ngroups, ny, nx = datamodel.data.shape
            noutputs = input_model.meta.exposure.noutputs
            slowaxis = np.abs(input_model.meta.subarray.slowaxis)

            # Fit slopes to get signal
            # Grab biases for each integration and get average slope
            bias_arr, slopes = fit_slopes_to_ramp_data(datamodel, 
                                                       sat_frac=self.sat_frac,
                                                       combine_ints=self.combine_ints)
            slope_mean = slopes if self.combine_ints else robust.mean(slopes, axis=0)

            # If only a single integration, then apply a mask to exclude pixels
            # with a large flux values. This will be used by the model fit.
            if nints == 1:
                data_diff = datamodel.data - bias_arr
            else:
                # Remove 1/f noise in the mean slope image
                good_mask_temp = robust.mean(slope_mean, return_mask=True)
                good_mask_temp = ~expand_mask(~good_mask_temp, 1, grow_diagonal=True)
                # Create a Clean class
                nf_clean = make_clean_class(slope_mean, good_mask_temp, noutputs, slowaxis, 
                                            flatten_model=True)
                nf_clean.fit(model_type='savgol', vertical_corr=False)
                slope_mean -= nf_clean.model
                del nf_clean, good_mask_temp

                # Generate a mean signal ramp to subtract from each group
                group_time = datamodel.meta.exposure.group_time
                tarr = np.arange(1, ngroups+1) * group_time
                signal_mean_ramp = slope_mean * tarr.reshape([-1,1,1])

                # Create a residual array for all ramps
                data_diff = datamodel.data.copy()
                for i in range(nints):
                    data_diff[i] -= (signal_mean_ramp + bias_arr[i])

            # If only a single INT, then perform model flattening along slow axis
            # to suppress flux from astrophysical sources that may get fit by
            # the 1/f noise model.
            flatten_model = True if nints==1 else False

            # Worker arguments
            worker_arguments = []
            for i in range(nints):
                # Cumulative sum of group DQ flags
                groupdq = datamodel.groupdq[i]
                mask_dnu = (groupdq & dqflags.pixel['DO_NOT_USE']) > 0
                bpmask_arr = np.cumsum(mask_dnu, axis=0) > 0
                # bpmask_arr = np.cumsum(datamodel.groupdq[i], axis=0) > 0
                for j in range(ngroups):
                    im_diff = data_diff[i,j]

                    # Good pixel mask
                    im_mask = create_bkg_mask(im_diff, bpmask=bpmask_arr[j])

                    input_args = (im_diff, im_mask, noutputs, slowaxis, flatten_model, 
                                  self.model_type, self.vertical_corr)
                    worker_arguments.append(input_args)

            self._worker_arguments = worker_arguments

            # Run multiprocessing
            model_arr = self._run_multiprocess()
            model_arr = model_arr.reshape(input_model.data.shape)

            del datamodel

        return model_arr

    def proc_CubeModel(self, input):
        """ Apply 1/f noise correction to a JWST `CubeModel` 

        Parameters
        ==========
        input : JWST data model
            Input science data model to be corrected.
            Should be a `CubeModel` such as rateints or calints.
        """
        # Get the input data model
        with datamodels.open(input) as input_model:

            nints, ny, nx = input_model.data.shape
            noutputs = input_model.meta.exposure.noutputs
            slowaxis = np.abs(input_model.meta.subarray.slowaxis)
            chsize = nx // noutputs

            # Create a copy of the input data model
            datamodel = input_model.copy()
            data = datamodel.data

            # If only a single integration, then apply a mask to exclude pixels
            # with a large flux values. This will be used by the model fit.
            if nints==1:
                data_diff = data
            else:
                # Remove the channel backgrounds
                for i in range(nints):
                    for ch in range(noutputs):
                        ix1 = int(ch*chsize)
                        ix2 = int(ix1 + chsize)
                        if slowaxis==2:
                            data[i,:,ix1:ix2] -= get_bkg(data[i,:,ix1:ix2])
                        else:
                            data[i,ix1:ix2,:] -= get_bkg(data[i,ix1:ix2,:])

                # Get average of data
                data_mean = np.nanmean(data, axis=0)

                # Remove 1/f noise in the mean data image
                good_mask_temp = create_bkg_mask(data_mean)
                good_mask_temp = ~expand_mask(~good_mask_temp, 1, grow_diagonal=True)
                # Create a Clean class
                nf_clean = make_clean_class(data_mean, good_mask_temp, noutputs, slowaxis, 
                                            flatten_model=True)
                nf_clean.fit(model_type='savgol', vertical_corr=False)

                # Subtract model from data
                data_mean -= nf_clean.model

                # Create a residual array for all images
                data_diff = data - data_mean
                del nf_clean, good_mask_temp

            # If only a single INT, then perform model flattening along slow axis
            flatten_model = True if nints==1 else False

            # Create worker arguments to pass to multiprocessing
            worker_arguments = []
            for i in range(nints):
                # Work on residual image
                im_diff = data_diff[i]
                dq_mask = datamodel.dq[i]
                bpmask = (dq_mask & dqflags.pixel['DO_NOT_USE']) > 0
                im_mask = create_bkg_mask(im_diff, bpmask=bpmask)

                input_args = (im_diff, im_mask, noutputs, slowaxis, flatten_model, 
                              self.model_type, self.vertical_corr)
                worker_arguments.append(input_args)
            # Save to class attribute
            self._worker_arguments = worker_arguments

            # Run multiprocessing
            model_arr = self._run_multiprocess()
            model_arr = model_arr.reshape(input_model.data.shape)

            del datamodel

        return model_arr

    def proc_ImageModel(self, input):
        """ Apply 1/f noise correction to a JWST `ImageModel` """
        # Get the input data model
        with datamodels.open(input) as input_model:

            noutputs = input_model.meta.exposure.noutputs
            slowaxis = np.abs(input_model.meta.subarray.slowaxis)

            # Only a single image, so apply a mask to exclude pixels
            # with a large flux values. This will be used by the model fit.
            good_mask = create_bkg_mask(input_model.data)
            good_mask = ~expand_mask(~good_mask, 1, grow_diagonal=True)

            # Subtract 1/f noise from image
            datamodel = input_model.copy()

            # Single image, so always perform model flattening along slow axis
            flatten_model = True

            # Worker arguments
            args = (
                datamodel.data, 
                good_mask, 
                noutputs, 
                slowaxis,
                flatten_model,
                self.model_type, 
                self.vertical_corr,
                )
            model = nfclean_mulitprocess_helper(args)

            del datamodel

        return model
    
    def _run_multiprocess(self):
        """Run multiprocessing on worker arguments"""

        if self._worker_arguments is None:
            raise RuntimeError('Worker arguments not set!')
        
        worker_arguments = self._worker_arguments
    
        nproc = self.nproc
        model_arr = []
        if nproc > 1:
            try:
                with mp.Pool(nproc) as pool:
                    proc_pool = pool.imap(nfclean_mulitprocess_helper, worker_arguments)
                    for res in proc_pool:
                        model_arr.append(res)
                    pool.close()
                    if model_arr[0] is None:
                        raise RuntimeError('Returned None values. Issue with multiprocess??')
            except Exception as e:
                log.error('Caught an exception during multiprocess.')
                log.info('Closing multiprocess pool.')
                raise e
            else:
                log.info('Closing multiprocess pool.')
        else:
            for args in worker_arguments:
                res = nfclean_mulitprocess_helper(args)
                if res is None:
                    raise RuntimeError('Returned None value!')
                model_arr.append(res)

        return np.asarray(model_arr)

    @plt.style.context('webbpsf_ext.wext_style')
    def quick_plot(self, image, mask, model, im_diff=None):

        fig, axes = plt.subplots(2,2, figsize=(8,8), sharex=True, sharey=True)
        axes = axes.flatten()

        im_temp = im_diff if im_diff is not None else image
        med = np.nanmedian(im_temp)
        sig = robust.medabsdev(im_temp)
        vmin, vmax = np.array([-1,1])*sig*3 + med
        axes[0].imshow(im_temp, vmin=vmin, vmax=vmax)
        axes[1].imshow(image - model, vmin=vmin, vmax=vmax)

        # Plot mask
        axes[2].imshow(mask)

        med = np.nanmedian(model)
        sig = robust.medabsdev(model)
        vmin, vmax = np.array([-1,1])*sig*3 + med
        axes[3].imshow(model, vmin=vmin, vmax=vmax, cmap='RdBu')

        # print(np.nanmedian(im_diff), np.nanmedian(data[i,j]), np.nanmedian(nf_clean.model))
        
        fig.tight_layout()

def make_clean_class(image, mask_good, noutputs, slowaxis, **kwargs):
    """ Create a Clean class instance
    
    Parameters
    ==========
    image : ndarray
        Input image to be cleaned.
    mask_good : ndarray
        Mask of good pixels. Pixels = True are modeled as background.
        Pixels = False are excluded from the background model.
    noutputs : int
        Number of output amplifier channels.
    slowaxis : int
        The slow scan axis. Must be 1 or 2. Default is 2.
        A setting of 1 implies output channels span the x-axis.
        A setting of 2 implies output channels span the y-axis.
    """

    if noutputs>1:
        return CleanFullFrame(image, mask_good, nout=noutputs, 
                              slowaxis=slowaxis, **kwargs)
    else:
        return CleanSubarray(image, mask_good, slowaxis=slowaxis, **kwargs)

def fit_slopes_to_ramp_data(input, sat_frac=0.5, combine_ints=False):
    """Fit slopes to each integration
    
    Uses custom `cube_fit` function to fit slopes to each integration.
    Returns array of slopes and bias values for each integration.
    Bias and slope arrays have shape (nints, ny, nx).

    Parameters
    ----------
    input : jwst.datamodel
        Input JWST datamodel housing the data to be fit.
    sat_frac : float
        Saturation threshold for fitting. Values above
        this fraction of the saturation level are ignored.
        Default is 0.5 to ensure that the fit is within 
        the linear range.
    combine_ints : bool
        Average all integrations into a single array before fitting.
        Return an (bias_arr, slope_mean). Default is False.
    """
    from jwst.datamodels import SaturationModel
    from jwst.saturation.saturation_step import SaturationStep
    from jwst.lib import reffile_utils
    from .utils import cube_fit

    # Get saturation reference file
    # Get the name of the saturation reference file
    sat_name = SaturationStep().get_reference_file(input, 'saturation')

    # Open the reference file data model
    sat_model = SaturationModel(sat_name)

    # Extract subarray from saturation reference file, if necessary
    if reffile_utils.ref_matches_sci(input, sat_model):
        sat_thresh = sat_model.data.copy()
    else:
        ref_sub_model = reffile_utils.get_subarray_model(input, sat_model)
        sat_thresh = ref_sub_model.data.copy()
        ref_sub_model.close()

    # Close the reference file
    sat_model.close()

    # Perform ramp fit to data to get bias offset
    group_time = input.meta.exposure.group_time
    ngroups = input.meta.exposure.ngroups
    nints = input.meta.exposure.nints
    tarr = np.arange(1, ngroups+1) * group_time
    data = input.data
    ny, nx = data.shape[-2:]

    bias_arr = []
    slope_arr = []
    for i in range(nints):
        # Get group-level bpmask for this integration
        groupdq = input.groupdq[i]
        # Make sure to accumulate the group-level dq mask
        mask_dnu = (groupdq & dqflags.pixel['DO_NOT_USE']) > 0
        bpmask_arr = np.cumsum(mask_dnu, axis=0) > 0
        # bpmask_arr = np.cumsum(groupdq, axis=0) > 0
        cf = cube_fit(tarr, data[i], bpmask_arr=bpmask_arr,
                      sat_vals=sat_thresh, sat_frac=sat_frac)
        bias_arr.append(cf[0])
        slope_arr.append(cf[1])
    bias_arr = np.asarray(bias_arr).reshape([nints, ny, nx])
    slope_arr = np.asarray(slope_arr).reshape([nints, ny, nx])

    # Subtract biases, average, and refit?
    if combine_ints and (nints > 1):
        data = input.data.copy()
        for i in range(nints):
            data[i] -= bias_arr[i]
            # Get group-level bpmask for this integration
            groupdq = input.groupdq[i]
            # Make sure to accumulate the group-level dq mask
            mask_dnu = (groupdq & dqflags.pixel['DO_NOT_USE']) > 0
            bpmask_arr = np.cumsum(mask_dnu, axis=0) > 0
            # bpmask_arr = np.cumsum(groupdq, axis=0) > 0
            # Exclude bad pixels
            data[i][bpmask_arr] = np.nan

        data_mean = np.nanmean(data, axis=0)
        bias_mean = np.nanmean(bias_arr, axis=0)
        bpmask_arr = np.isnan(data_mean)
        _, slope_mean = cube_fit(tarr, data_mean, bpmask_arr=bpmask_arr,
                                 sat_vals=sat_thresh-bias_mean, sat_frac=sat_frac)
        
        # bias has shape [nints, ny, nx]
        # slope has shape [ny, nx]
        return bias_arr, slope_mean
    elif combine_ints and (nints == 1):
        # bias has shape [nints, ny, nx]
        # slope has shape [ny, nx]
        return bias_arr, slope_arr[0]
    else:
        # bias and slope arrays have shape [nints, ny, nx]
        # bias values are in units of DN and slope in DN/sec
        return bias_arr, slope_arr

class CleanFullFrame:
    """ Clean 1/f noise from full frame images

    Use `CleanSubarray` instances to clean each channel separately.
    First removes common channel noise, then cleans the residuals of 
    each channel. Final full frame model is stored in `model` attribute. 
    Intended for use on Level 1 or 2 pipeline products on any type of 
    image product (group data, rateint, calint, rate, cal, etc). 
    Suggest removing average signal levels from data and running
    this on the residuals, which better reveal the 1/f noise structure.

    Inspired by NSClean by Bernie Rauscher (https://arxiv.org/abs/2306.03250),
    however instead of using FFTs and Matrix multiplication, this class uses
    Savitzky-Golay filtering to model the 1/f noise and subtract it from the
    data. This is significantly faster than the FFT approach and provides 
    similar results. 

    By default, function assumes such that the slowscan runs vertically along
    the y-axis (slowaxis=2), while the fast scan direction runs horizontally. 
    Direction (left to right or right to left) is irrelevant. If the slowscan 
    runs horizontally along x-axis in your data, then set `slowaxis=1` during
    intialization.
    """

    def __init__(self, data, mask, exclude_outliers=True, bg_sub=False,
                 flatten_model=False, slowaxis=2, nout=4, channel_averaging=False, 
                 **kwargs):

        """
        Parameters
        ==========
        data : ndarray
            Two-dimensional input data.
        mask : ndarray
            Two dimensional background pixels mask. 
            Pixels = True are modeled as background.
            Pixels = False are excluded from the background model.
        exclude_outliers : bool
            Exclude statistical outliers and their nearest neighbors
            from the background pixels mask.
        flatten_model : bool
            Subtract the smoothed version of each column in the model. 
            This will remove residual large scale structures form astrophysical sources 
            from the model. Default is True.
        bg_sub : bool
            Use photutils Background2D to remove spatially varying background.
            Default is False.
        slowaxis : int
            The slow scan axis. Must be 1 or 2. Default is 2.
            A setting of 1 implies output channels span the x-axis.
            A setting of 2 implies output channels span the y-axis.
        nout : int
            Number of output amplifier channels. Default is 4.
        channel_averaging : bool
            Average the channels together to remove common channel noise.
            Default is False.
        """

        # Iniailize and return subarray class
        if nout==1:
            raise ValueError("nout must be >1 for full frame. Otherwise use CleanSubarray.")

        # Definitions
        self.D = np.array(data, dtype=np.float32)
        self.M = np.array(mask, dtype=np.bool_)
        self.slowaxis = slowaxis
        self.nout = nout
        self.ny, self.nx = self.D.shape
        self.chsize = self.nx // self.nout

        self._flatten_model = flatten_model
        self._bg_sub = bg_sub
        self._exclude_outliers = exclude_outliers

        # Init the output classes
        self.chavg = np.zeros([self.ny, self.chsize])
        self.chavg_mask = np.zeros_like(self.chavg, dtype=np.bool_)
        chavg_class = None
        self.output_classes = {'chavg': chavg_class}
        
        # Create subarray class for each channel
        for ch in range(self.nout):
            x1 = int(ch*self.chsize)
            x2 = int(x1 + self.chsize)
            data = self.D[:,x1:x2] if self.slowaxis==2 else self.D[x1:x2,:]
            mask = self.M[:,x1:x2] if self.slowaxis==2 else self.M[x1:x2,:]
            self.output_classes[ch] = CleanSubarray(data, mask, 
                                                    exclude_outliers=exclude_outliers,
                                                    flatten_model=flatten_model,
                                                    bg_sub=bg_sub, slowaxis=slowaxis)
    
        # Average the channel data if requested.
        # If the average of the channel data exists, a model will
        # be subtracted from each channel and then a new model fit to
        # the residuals. This is useful for removing common channel noise.
        if channel_averaging:
            self.average_channels()
            # Create a subarray class for the average channel
            chavg_mask = np.zeros_like(self.chavg, dtype=np.bool_)
            for ch in range(nout):
                x1 = int(ch*self.chsize)
                x2 = int(x1 + self.chsize)
                ch_mask = self.M[:,x1:x2] if self.slowaxis==2 else self.M[x1:x2,:]
                if ch % 2 == 0:
                    axis_flip = 1 if self.slowaxis==2 else 0
                    ch_mask = np.flip(ch_mask, axis=axis_flip)
                chavg_mask = chavg_mask | ch_mask
            self.chavg_mask = chavg_mask
            chavg_class = CleanSubarray(self.chavg, self.chavg_mask, bg_sub=False,
                                        exclude_outliers=exclude_outliers,
                                        flatten_model=flatten_model, slowaxis=slowaxis)
            self.output_classes['chavg'] = chavg_class

    @property
    def flatten_model(self):
        """Use Coronagraphic ND acquisition square?"""
        return self._flatten_model
    @flatten_model.setter
    def flatten_model(self, value):
        if value not in [True, False]:
            raise ValueError("flatten_model must be True or False")
        self._flatten_model = value
        # Update values in output classes
        for ch_class in self.output_classes.values():
            if ch_class is not None:
                ch_class.flatten_model = value

    def average_channels(self):
        """ Create an average channel image representing the common 1/f noise"""
        data = self.D.copy()
        data[~self.M] = np.nan

        # Transpose data if slowaxis is horizontal
        if self.slowaxis==1:
            data = data.T

        # Flip every other channel along x-axes
        for ch in range(self.nout):
            x1 = int(ch*self.chsize)
            x2 = int(x1 + self.chsize)
            data[:,x1:x2] -= np.nanmedian(data[:,x1:x2])
            if ch % 2 == 0:
                data[:,x1:x2] = np.flip(data[:,x1:x2], axis=1)

        # Average channels
        data = data.reshape([self.ny, self.nout, self.chsize])
        self.chavg = np.nanmean(data, axis=1)

        # Transpose back if slowaxis is specified as horizontal
        if self.slowaxis==1:
            self.chavg = self.chavg.T

    def fit(self, model_type='savgol', vertical_corr=False, **kwargs):

        # Fit model to average channel
        chavg_class = self.output_classes['chavg']
        if chavg_class is None:
            chavg_model = np.zeros_like(self.output_classes[0].D)
        else:
            chavg_class.fit(model_type=model_type, vertical_corr=False, **kwargs)
            chavg_model = self.output_classes['chavg'].model

        # Get final models
        final_model = np.zeros_like(self.D)
        for ch in range(self.nout):
            ch_class = self.output_classes[ch]

            # Flip model every other channel
            if ch % 2 == 0:
                axis_flip = 1 if self.slowaxis==2 else 0
                avgmod = np.flip(chavg_model, axis=axis_flip)

            avgmod = np.flip(chavg_model, axis=1) if ch % 2 == 0 else chavg_model

            # Subtract average channel model from data
            ch_class.D -= avgmod
            ch_class.fit(model_type=model_type, vertical_corr=False, **kwargs)
            ch_class.D += avgmod

            # Add average model back to channel model
            x1 = int(ch*self.chsize)
            x2 = int(x1 + self.chsize)
            if self.slowaxis==2:
                final_model[:,x1:x2] = ch_class.model + avgmod
            else:
                final_model[x1:x2,:] = ch_class.model + avgmod

        # Run vertical correction on the full frame image
        if vertical_corr:
            # Subtract the final model from the data and transpose
            data = (self.D - final_model).T
            mask = self.M.T
            ff_class = CleanSubarray(data, mask, slowaxis=self.slowaxis, 
                                     exclude_outliers=self._exclude_outliers,
                                     flatten_model=self._flatten_model,
                                     bg_sub=self._bg_sub)
            ws = kwargs.get('winsize', 127)
            kwargs['winsize'] = ws if ws % 2 == 1 else ws - 1
            ff_class.fit(model_type=model_type, vertical_corr=False, **kwargs)
            vert_model = ff_class.model.T
            final_model += vert_model

            del ff_class

        self.model = final_model

    def clean(self, model_type='savgol', vertical_corr=False, **kwargs):
        """ Clean the data

        Overwrites data in-place with the cleaned data.
        
        Parameters
        ==========
        model_type : str
            Must be 'median', 'mean', or 'savgol'. For 'mean' case,
            it uses a robust mean that ignores outliers and NaNs.
            The 'median' case uses `np.nanmedian`. The 'savgol' case
            uses a Savitzky-Golay filter to model the 1/f noise, 
            iteratively rejecting outliers from the model fit relative
            to the median model. The default is 'savgol'.
        vertical_corr : bool
            Apply a horizontal correction to the data. This is useful
            for removing horizontal striping. The default is False.

        Keyword Args
        ============
        niter : int
            Number of iterations to use for rejecting outliers during
            the model fit. If the number of rejected pixels does not
            change between iterations, then the fit is considered
            converged and the loop is broken.
        winsize : int
            Size of the window filter. Should be an odd number.
        order : int
            Order of the polynomial used to fit the samples.
        per_line : bool
            Smooth each channel line separately with the hopes of avoiding
            edge discontinuities.
        mask : bool image or None
            An image mask of pixels to ignore. Should be same size as im_arr.
            This can be used to mask pixels that the filter should ignore, 
            such as stellar sources or pixel outliers. A value of True indicates
            that pixel should be ignored.
        mode : str
            Must be 'mirror', 'constant', 'nearest', 'wrap' or 'interp'.  This
            determines the type of extension to use for the padded signal to
            which the filter is applied.  When `mode` is 'constant', the padding
            value is given by `cval`. 
            When the 'interp' mode is selected (the default), no extension
            is used.  Instead, a degree `polyorder` polynomial is fit to the
            last `window_length` values of the edges, and this polynomial is
            used to evaluate the last `window_length // 2` output values.
        cval : float
            Value to fill past the edges of the input if `mode` is 'constant'.
            Default is 0.0.
        """ 
        # Fit the background model
        self.fit(model_type=model_type, vertical_corr=vertical_corr, **kwargs) 
        self.D -= self.model # Overwrite data with cleaned data
        return self.D


class CleanSubarray:
    """ 1/f noise modeling and subtraction for HAWAII-2RG subarrays.

    CleanSubarray is the base class for removing residual correlated
    read noise from generic JWST near-IR Subarray images.  It is
    intended for use on Level 1 or 2 pipeline products on any type of
    image product (group data, rateint, calint, rate, cal, etc). 
    Suggest removing average signal levels from data and running
    this on the residuals, which better reveal the 1/f noise structure.

    Inspired by NSClean by Bernie Rauscher (https://arxiv.org/abs/2306.03250),
    however instead of using FFTs and Matrix multiplication, this class uses
    Savitzky-Golay filtering to model the 1/f noise and subtract it from the
    data. This is significantly faster than the FFT approach and provides 
    similar results. 

    This function assumes such that the slowscan runs vertically along the
    y-axis, while the fast scan direction runs horizontally. Direction 
    (left to right or right to left) is irrelevant. If the slowscan runs
    horizontally along x-axis in your data, then make sure to transpose
    before passing to this class.
    """
            
    def __init__(self, data, mask, exclude_outliers=True, bg_sub=False,
                 flatten_model=False, slowaxis=2, **kwargs):
        """ Initialize the class

        Parameters
        ==========
        data : ndarray
            Two-dimensional input data.
        mask : ndarray
            Two dimensional background pixels mask. 
            Pixels = True are modeled as background.
            Pixels = False are excluded from the background model.
        exclude_outliers : bool
            Exclude statistical outliers and their nearest neighbors
            from the background pixels mask.
        flatten_model : bool
            Subtract the smoothed version of each column in the model. 
            This will remove residual large scale structures form astrophysical sources 
            from the model. Default is False.
        bg_sub : bool
            Use photutils Background2D to remove spatially varying background.
            Default is False.
        slowaxis : int
            The slow scan axis. Must be 1 or 2. Default is 2.
            A setting of 1 implies output channels span the x-axis.
            A setting of 2 implies output channels span the y-axis.
        """

        # Definitions
        self.D = np.array(data, dtype=np.float32)
        self.M = np.array(mask, dtype=np.bool_)
        self.slowaxis = np.abs(slowaxis)
        self.flatten_model = flatten_model
        
        # The mask potentially contains NaNs. Exclude them.
        self.M[np.isnan(self.D)] = False
        
        # The mask potentially contains statistical outliers.
        # Optionally exclude them.
        if exclude_outliers is True:
            gdpx = robust.mean(self.D, Cut=3, return_mask=True)
            bdpx = ~gdpx # Bad pixel mask
            bdpx = expand_mask(bdpx, 1, grow_diagonal=False) # Also flag 4 nearest neighbors
            gdpx = ~bdpx # Good pixel mask
            self.M = self.M & gdpx

        # Median subtract
        self.D = self.D - np.nanmedian(self.D[self.M])

        # Remove background variations?
        if bg_sub:
            self.bg_subtract()

    @property
    def nx(self):
        """ Number of columns in the image """
        return self.D.shape[1]
    @property
    def ny(self):
        """ Number of rows in the image """
        return self.D.shape[0]

    def bg_subtract(self, box_size=32, filter_size=5, sigma=3, maxiters=5):
        """Use photutils Background2D to remove spatially varying background"""
    
        from astropy.stats import SigmaClip
        from photutils.background import Background2D, MedianBackground

        sigclip_func = SigmaClip(sigma=sigma, maxiters=maxiters)
        bkg_estimator = MedianBackground()
        bp_mask = ~self.M # Bad pixel mask
        bkg = Background2D(self.D, box_size, mask=bp_mask, filter_size=filter_size, 
                           sigma_clip=sigclip_func, bkg_estimator=bkg_estimator)
        self.D -= bkg.background

    def fit(self, model_type='savgol', vertical_corr=False, **kwargs):
        """ Return the model which is just median of each row
        
        Parameters
        ==========
        model_type : str
            Must be 'median', 'mean', or 'savgol'. For 'mean' case,
            it uses a robust mean that ignores outliers and NaNs.
            The 'median' case uses `np.nanmedian`. The 'savgol' case
            uses a Savitzky-Golay filter to model the 1/f noise, 
            iteratively rejecting outliers from the model fit relative
            to the median model. The default is 'savgol'.
        vertical_corr : bool
            Apply a vertical correction to the data. This is useful
            for removing vertical striping. The default is False.

        Keyword Args
        ============
        niter : int
            Number of iterations to use for rejecting outliers during
            the model fit. If the number of rejected pixels does not
            change between iterations, then the fit is considered
            converged and the loop is broken. For 'savgol' only.
        """

        # Transpose data if slowaxis is horizontal
        if self.slowaxis == 1:
            self.D = self.D.T
            self.M = self.M.T

        # Fit the model
        if 'median' in model_type:
            self.model = self._fit_median(robust_mean=False)
        elif 'mean' in model_type:
            self.model = self._fit_median(robust_mean=True)
        elif 'savgol' in model_type:
            self.model = self._fit_savgol(**kwargs)
        else:
            raise ValueError(f"Do not recognize model_type={model_type}")
        
        # Remove vertical structure in model due to astrophysical sources
        if self.flatten_model:
            max_size = self.ny - 1 # Has already been transposed to slow axis along y-direction
            ws = np.min([max_size, 255])
            model_bg = channel_smooth_savgol(self.model.T, winsize=ws, order=3, per_line=True).T
            self.model -= model_bg

        # Apply vertical offset correction
        if vertical_corr:
            data = self.D.copy()
            model_horz = self.model.copy()

            # Rotate model subtracted data by 90 degrees
            data_model_sub = data - model_horz
            self.D = data_model_sub.T # np.rot90(data - model_horz, k=1, axes=(0,1))
            self.M = self.M.T # np.rot90(self.M, k=1, axes=(0,1))

            # Fit the model
            if 'median' in model_type:
                self.model = self._fit_median(robust_mean=False)
            elif 'mean' in model_type:
                self.model = self._fit_median(robust_mean=True)
            elif 'savgol' in model_type:
                kwargs['winsize'] = self.D.shape[-1] // 3 - 1
                self.model = self._fit_savgol(**kwargs)

            # Return data and mask to original orientation
            self.D = data
            self.M = self.M.T #np.rot90(self.M, k=-1, axes=(0,1))
            self.model = model_horz + self.model.T # np.rot90(self.model, k=-1, axes=(0,1))

            del model_horz

        # Transpose data back if slowaxis is horizontal
        if self.slowaxis == 1:
            self.D = self.D.T
            self.M = self.M.T
            self.model = self.model.T

    def _fit_median(self, robust_mean=False):
        """ Return the model which is just median of each row
        
        Option to use robust mean instead of median.
        """

        mean_func = robust.mean if robust_mean else np.nanmedian

        # Fit the model
        data = self.D.copy()
        data[~self.M] = np.nan
        return mean_func(data, axis=1).repeat(self.nx).reshape(data.shape)
        
    def _fit_savgol(self, niter=5, **kwargs):
        """ Use a Savitzky-Golay filter to smooth the masked row data

        Parameters
        ==========
        niter : int
            Number of iterations to use for rejecting outliers during
            the model fit. If the number of rejected pixels does not
            change between iterations, then the fit is considered
            converged and the loop is broken.

        Keyword Args
        ============
        winsize : int
            Size of the window filter. Should be an odd number.
        order : int
            Order of the polynomial used to fit the samples.
        per_line : bool
            Smooth each channel line separately with the hopes of avoiding
            edge discontinuities.
        mask : bool image or None
            An image mask of pixels to ignore. Should be same size as im_arr.
            This can be used to mask pixels that the filter should ignore, 
            such as stellar sources or pixel outliers. A value of True indicates
            that pixel should be ignored.
        mode : str
            Must be 'mirror', 'constant', 'nearest', 'wrap' or 'interp'.  This
            determines the type of extension to use for the padded signal to
            which the filter is applied.  When `mode` is 'constant', the padding
            value is given by `cval`. 
            When the 'interp' mode is selected (the default), no extension
            is used.  Instead, a degree `polyorder` polynomial is fit to the
            last `window_length` values of the edges, and this polynomial is
            used to evaluate the last `window_length // 2` output values.
        cval : float
            Value to fill past the edges of the input if `mode` is 'constant'.
            Default is 0.0.
        """

        bpmask = ~self.M # Bad pixel mask
        model = channel_smooth_savgol(self.D, mask=bpmask, **kwargs)

        for i in range(niter):
            # Get median model
            model_med = self._fit_median(robust_mean=False)

            # Find standard deviation of difference between model and median values
            bpmask = ~self.M 
            diff = model - model_med
            diff[bpmask] = np.nan
            sig = np.nanstd(diff, ddof=1)

            # Flag new outliers
            bp_sig = np.abs(diff) > 3*sig
            bpmask_new = bpmask | bp_sig
            if bpmask_new.sum() == bpmask.sum():
                break

            # Update mask and refit model
            self.M = ~bpmask_new
            model = channel_smooth_savgol(self.D, mask=bpmask_new, **kwargs)

        return model
        
    def clean(self, model_type='savgol', vertical_corr=False, **kwargs):
        """ Clean the data

        Overwrites data in-place with the cleaned data.
        
        Parameters
        ==========
        model_type : str
            Must be 'median', 'mean', or 'savgol'. For 'mean' case,
            it uses a robust mean that ignores outliers and NaNs.
            The 'median' case uses `np.nanmedian`. The 'savgol' case
            uses a Savitzky-Golay filter to model the 1/f noise, 
            iteratively rejecting outliers from the model fit relative
            to the median model. The default is 'savgol'.
        vertical_corr : bool
            Apply a horizontal correction to the data. This is useful
            for removing horizontal striping. The default is False.

        Keyword Args
        ============
        niter : int
            Number of iterations to use for rejecting outliers during
            the model fit. If the number of rejected pixels does not
            change between iterations, then the fit is considered
            converged and the loop is broken.
        winsize : int
            Size of the window filter. Should be an odd number.
        order : int
            Order of the polynomial used to fit the samples.
        per_line : bool
            Smooth each channel line separately with the hopes of avoiding
            edge discontinuities.
        mask : bool image or None
            An image mask of pixels to ignore. Should be same size as im_arr.
            This can be used to mask pixels that the filter should ignore, 
            such as stellar sources or pixel outliers. A value of True indicates
            that pixel should be ignored.
        mode : str
            Must be 'mirror', 'constant', 'nearest', 'wrap' or 'interp'.  This
            determines the type of extension to use for the padded signal to
            which the filter is applied.  When `mode` is 'constant', the padding
            value is given by `cval`. 
            When the 'interp' mode is selected (the default), no extension
            is used.  Instead, a degree `polyorder` polynomial is fit to the
            last `window_length` values of the edges, and this polynomial is
            used to evaluate the last `window_length // 2` output values.
        cval : float
            Value to fill past the edges of the input if `mode` is 'constant'.
            Default is 0.0.
        """ 
        # Fit the background model
        self.fit(model_type=model_type, vertical_corr=vertical_corr, **kwargs) 
        self.D -= self.model # Overwrite data with cleaned data
        return self.D

def mask_helper():
    """Helper to handle indices and logical indices of a mask

    Output: index, a function, with signature indices = index(logical_indices),
    to convert logical indices of a mask to 'equivalent' indices

    Example:
        >>> # linear interpolation of NaNs
        >>> mask = np.isnan(y)
        >>> x = mask_helper(y)
        >>> y[mask]= np.interp(x(mask), x(~mask), y[~mask])
    """
    return lambda z: np.nonzero(z)[0]

def channel_smooth_savgol(im_arr, winsize=127, order=3, per_line=False, 
    mask=None, **kwargs):
    """Channel smoothing using savgol filter

    Function for generating a map of the 1/f noise within a series of input images.

    Copied over from `pynrc` (https://github.com/JarronL/pynrc) by Jarron Leisenring.

    Parameters
    ==========
    im_arr : ndarray
        Input array of images (intended to be a cube of output channels).
        Shape should either be (ny, chsize) to smooth a single channel or
        (nchan, ny, chsize) for  multiple channels.
        Each image is operated on separately. If only two dimensions,
        then only a single input image is assumed. NaN's will be
        interpolated over.

    Keyword Args
    ============
    winsize : int
        Size of the window filter. Should be an odd number.
    order : int
        Order of the polynomial used to fit the samples.
    per_line : bool
        Smooth each channel line separately with the hopes of avoiding
        edge discontinuities.
    mask : bool image or None
        An image mask of pixels to ignore. Should be same size as im_arr.
        This can be used to mask pixels that the filter should ignore, 
        such as stellar sources or pixel outliers. A value of True indicates
        that pixel should be ignored.
    mode : str
        Must be 'mirror', 'constant', 'nearest', 'wrap' or 'interp'.  This
        determines the type of extension to use for the padded signal to
        which the filter is applied.  When `mode` is 'constant', the padding
        value is given by `cval`. 
        When the 'interp' mode is selected (the default), no extension
        is used.  Instead, a degree `polyorder` polynomial is fit to the
        last `window_length` values of the edges, and this polynomial is
        used to evaluate the last `window_length // 2` output values.
    cval : float
        Value to fill past the edges of the input if `mode` is 'constant'.
        Default is 0.0.
    """
    from scipy.signal import savgol_filter

    sh = im_arr.shape
    if len(sh)==2:
        nz = 1
        ny, nx = sh
    else:
        nz, ny, nx = sh

    # Check that winsize is odd
    winsize = winsize-1 if winsize % 2 == 0 else winsize

    # Reshape in case of nz=1
    im_arr = im_arr.reshape([nz, -1])
    if mask is not None:
        mask = mask.reshape([nz, -1])

    res_arr = []
    for i, im in enumerate(im_arr):
        # im should be a 1D array

        # Interpolate over masked data and NaN's
        nans = np.isnan(im)
        im_mask = nans if mask is None else nans | mask[i].flatten()
        if im_mask.any():
            # Create a copy so as to not change the original data
            im = np.copy(im)

            # Use a savgol filter to smooth out any outliers
            res = im.copy()
            res[~im_mask] = savgol_filter(im[~im_mask], 33, 3, mode='interp')

            # Replace masked pixels with linear interpolation
            x = mask_helper() # Returns the nonzero (True) indices of a mask
            im[im_mask]= np.interp(x(im_mask), x(~im_mask), res[~im_mask])

        if per_line:
            im = im.reshape([ny,-1])

            res = savgol_filter(im, winsize, order, axis=1, delta=1, **kwargs)
            res_arr.append(res)
        else:
            res = savgol_filter(im, winsize, order, delta=1, **kwargs)
            res_arr.append(res.reshape([ny,-1]))
    res_arr = np.asarray(res_arr)

    if len(sh)==2:
        return res_arr[0]
    else:
        return res_arr


def create_bkg_mask(data, bpmask=None, nsigma=3, niter=3):
    """Returns a mask of background pixels
    
    Selects background pixels based on a median value and a sigma threshold.
    First, select pixels that are below the median value of the data.
    Select pixels that deviate from the median by more than `nsigma`.
    """

    from astropy.convolution import convolve, Gaussian2DKernel
    from photutils.segmentation import detect_sources

    if bpmask is None:
        bpmask = np.zeros_like(data, dtype=np.bool_)
    else:
        # Ensure bpmask isn't all True
        if np.alltrue(bpmask):
            bpmask = np.zeros_like(data, dtype=np.bool_)

    # Excpliitly mask out NaNs
    bpmask[np.isnan(data)] = True

    mask_good = ~bpmask
    data_good = data[mask_good]
    median = np.nanmedian(data_good)
    # Select only data values less than median as background pixels
    bg_pixels = data_good < median

    # Calculate the difference between the median and each background pixel
    diff = data_good[bg_pixels] - median
    # Get 1 sigma of the distribution and its reflection
    one_sigma = robust.medabsdev(np.array([diff, -diff]))
    # Mask out values that deviate betwen some nsigma
    ind_bad = np.abs(data - median) > (nsigma*one_sigma)
    mask_good[ind_bad] = False
    data_bgsub = data - np.median(data[mask_good])

    # Smooth with Gaussian and search for sources
    kernel = Gaussian2DKernel(x_stddev=2, y_stddev=2, x_size=5, y_size=5)  
    im_conv = convolve(data_bgsub, kernel)
    segm_detect = detect_sources(im_conv, 2*one_sigma, mask=~mask_good, npixels=7, connectivity=4)
    if segm_detect is not None:
        segimage = segm_detect.data.astype(np.uint32)
        mask_good[segimage>0] = False

    for i in range(niter-1):
        npix_good = np.sum(mask_good)
        bp_mask = ~(mask_good.copy())
        mask_good = create_bkg_mask(data, bpmask=bp_mask, nsigma=3, niter=1)
        # Break out of loop if no new pixels were flagged
        if np.sum(mask_good) == npix_good:
            break

    # Return inverse of the mask
    return mask_good

def get_bkg(data, bpmask=None, nsigma=3, complex=False):
    """ Background subtraction of an image using `photutils`
    
    Similar to algorithm by Chris Willott at 
    https://github.com/chriswillott/jwst/blob/master/image1overf.py
    """

    from astropy.stats import SigmaClip
    from photutils.background import Background2D, MedianBackground

    bgk_mask = create_bkg_mask(data, bpmask=bpmask, nsigma=nsigma)
    bpmask = ~bgk_mask

    if not complex:
        # Take a simple median of the background pixels
        background = np.nanmedian(data[bgk_mask])
    else:
        # A more comlex background using photutils
        sigma_clip = SigmaClip(sigma=3., maxiters=5)
        bkg_estimator = MedianBackground()
        bkg = Background2D(data, 34, filter_size=5, mask=bpmask, sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
        
        background = bkg.background

    return background



