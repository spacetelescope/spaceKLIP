from __future__ import division

import logging
import os

import astropy.io.fits as pyfits
from jwst import datamodels
from jwst.associations.load_as_asn import LoadAsLevel2Asn
from jwst.outlier_detection.outlier_detection_step import OutlierDetectionStep
from jwst.pipeline import Image2Pipeline
from tqdm import trange

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# =============================================================================
# MAIN
# =============================================================================

class Coron2Pipeline_spaceKLIP(Image2Pipeline):
    """
    The spaceKLIP JWST stage 2 pipeline class.
    
    """
    
    class_alias = "calwebb_coron2"
    
    spec = """
        save_intermediates = boolean(default=False) # Save all intermediate step results
    """

    def __init__(self,
                 **kwargs):
        """
        Initialize the spaceKLIP JWST stage 2 pipeline class.
        
        Parameters
        ----------
        **kwargs : keyword arguments
            Default JWST stage 2 image pipeline keyword arguments.
        
        Returns
        -------
        None.
        
        """
        
        # Add outlier detection.
        self.step_defs['outlier_detection'] = OutlierDetectionStep
        
        # Initialize Image2Pipeline class.
        super(Coron2Pipeline_spaceKLIP, self).__init__(**kwargs)
        
        # Set additional step parameters.
        self.outlier_detection.skip = False
    
    def process(self,
                input):
        """
        Process an input JWST datamodel with the spaceKLIP JWST stage 2
        pipeline.
        
        Parameters
        ----------
        input : jwst.datamodel
            Input JWST datamodel to be processed.
        
        Returns
        -------
        all_res : list of jwst.datamodel
            List of output JWST datamodels.
        
        """
        
        # Open input as asn model.
        asn = LoadAsLevel2Asn.load(input, basename=self.output_file)
        
        # Loop through all products.
        all_res = []
        for product in asn['products']:
            if self.save_results:
                self.output_file = product['name']
            try:
                getattr(asn, 'filename')
            except AttributeError:
                asn.filename = 'singleton'
            
            # Process exposure.
            filebase = os.path.basename(asn.filename)
            res = self.process_exposure_product(product, asn['asn_pool'], filebase)
            
            # Run outlier detection.
            res = self.outlier_detection.run(res)
            
            # Save results.
            suffix = 'calints' if isinstance(res, datamodels.CubeModel) else 'cal'
            res.meta.filename = self.make_output_path(suffix=suffix)
            all_res.append(res)

            # If outlier detection was run but intermediates were not request
            # to be saved, remove the intermediate _median.fits files.
            if not self.save_intermediates and not self.outlier_detection.skip:
                file_median = res.meta.filename.replace('calints', 'median')
                if os.path.exists(file_median):
                    os.remove(file_median)
        
        # Setup output file.
        self.output_use_model = True
        self.suffix = False
        
        return all_res

def run_single_file(fitspath, output_dir, steps={}, verbose=False, **kwargs):
    """ Run the JWST stage 2 image pipeline on a single file.

    This customized implementation will also run the 'outlier_detection' step
    if not skipped.
    
    Parameters
    ----------
    database : spaceKLIP.Database
        SpaceKLIP database on which the JWST stage 2 image pipeline shall be
        run.
    steps : dict, optional
        See here for how to use the steps parameter:
        https://jwst-pipeline.readthedocs.io/en/latest/jwst/user_documentation/running_pipeline_python.html#configuring-a-pipeline-step-in-python
        Custom step parameters are:
        - n/a
        The default is {}.
    subdir : str, optional
        Name of the directory where the data products shall be saved. The
        default is 'stage2'.
    do_rates : bool, optional
        In addition to processing rateints files, also process rate files
        if they exist? The default is False.
    overwrite : bool, optional
        Overwrite existing files? Default is False.
    quiet : bool, optional
        Use progress bar to track progress instead of messages. 
        Overrides verbose and sets it to False. Default is False.
    verbose : bool, optional
        Print all info messages? Default is False.
    
    Keyword Args
    ------------
    save_results : bool, optional
        Save the JWST pipeline products? The default is True.
    skip_bg : bool, optional
        Skip the background subtraction step? The default is False.
    skip_photom : bool, optional
        Skip the photometric correction step? The default is False.
    skip_resample : bool, optional
        Skip the resampling (drizzle) step? While the default is set
        to False, this step only applies to normal imaging modes and
        skips coronagraphic observation. For coronagraphic observations,
        resampling occurs in Stage 3.
    skip_wcs : bool, optional
        Skip the WCS assignment step? The default is False.
    skip_flat : bool, optional
        Skip the flat field correction step? The default is False.
    skip_outlier : bool, optional
        Skip the outlier detection step? The default is False except
        for target acquisition subarray data, which will always be True.

    Returns
    -------
    None.
    """
    # Print all info message if verbose, otherwise only errors or critical.
    from .logging_tools import all_logging_disabled
    log_level = logging.INFO if verbose else logging.ERROR

    # Create output directory if it doesn't exist.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize Coron1Pipeline.
    with all_logging_disabled(log_level):
        pipeline = Coron2Pipeline_spaceKLIP(output_dir=output_dir)

    # Options for saving results
    pipeline.save_results         = kwargs.get('save_results', True)
    pipeline.save_intermediates   = kwargs.get('save_intermediates', False)

    # Skip certain steps?
    pipeline.bkg_subtract.skip = kwargs.get('skip_bg', True)
    pipeline.photom.skip       = kwargs.get('skip_photom', False)
    pipeline.resample.skip     = kwargs.get('skip_resample', False)
    pipeline.assign_wcs.skip   = kwargs.get('skip_wcs', False)
    pipeline.flat_field.skip   = kwargs.get('skip_flat', False)

    # Don't perform outlier step for TA data.
    hdr0 = pyfits.getheader(fitspath)
    if 'NRC_TA' in hdr0['EXP_TYPE']:
        pipeline.outlier_detection.skip = True
    else:
        pipeline.outlier_detection.skip = kwargs.get('skip_outlier', False)

    # Set step parameters.
    for key1 in steps.keys():
        for key2 in steps[key1].keys():
            setattr(getattr(pipeline, key1), key2, steps[key1][key2])
    
    # Run Coron2Pipeline. Raise exception on error.
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
            subdir='stage2',
            do_rates=False,
            overwrite=True,
            quiet=False,
            verbose=False,
            **kwargs):
    """
    Run the JWST stage 2 image pipeline on the input observations database.
    This customized implementation will also run the 'outlier_detection' step
    if not skipped.
    
    Parameters
    ----------
    database : spaceKLIP.Database
        SpaceKLIP database on which the JWST stage 2 image pipeline shall be
        run.
    steps : dict, optional
        See here for how to use the steps parameter:
        https://jwst-pipeline.readthedocs.io/en/latest/jwst/user_documentation/running_pipeline_python.html#configuring-a-pipeline-step-in-python
        Custom step parameters are:
        - n/a
        The default is {}.
    subdir : str, optional
        Name of the directory where the data products shall be saved. The
        default is 'stage2'.
    do_rates : bool, optional
        In addition to processing rateints files, also process rate files
        if they exist? The default is False.
    overwrite : bool, optional
        Overwrite existing files? Default is False.
    quiet : bool, optional
        Use progress bar to track progress instead of messages. 
        Overrides verbose and sets it to False. Default is False.
    verbose : bool, optional
        Print all info messages? Default is False.
    
    Keyword Args
    ------------
    save_results : bool, optional
        Save the JWST pipeline products? The default is True.
    skip_bg : bool, optional
        Skip the background subtraction step? The default is False.
    skip_photom : bool, optional
        Skip the photometric correction step? The default is False.
    skip_resample : bool, optional
        Skip the resampling (drizzle) step? While the default is set
        to False, this step only applies to normal imaging modes and
        skips coronagraphic observation. For coronagraphic observations,
        resampling occurs in Stage 3.
    skip_wcs : bool, optional
        Skip the WCS assignment step? The default is False.
    skip_flat : bool, optional
        Skip the flat field correction step? The default is False.
    skip_outlier : bool, optional
        Skip the outlier detection step? The default is False except
        for target acquisition subarray data, which will always be True.

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

    # Loop through concatenations.
    for i in itervals:
        key = keys[i]
        if not quiet: log.info('--> Concatenation ' + key)
        
        # Loop through FITS files.
        nfitsfiles = len(database.obs[key])
        jtervals = trange(nfitsfiles, desc='FITS files', leave=False) if quiet else range(nfitsfiles)
        for j in jtervals:
            
            # Skip non-stage 1 files.
            head, tail = os.path.split(database.obs[key]['FITSFILE'][j])
            fitspath = os.path.abspath(database.obs[key]['FITSFILE'][j])
            if database.obs[key]['DATAMODL'][j] != 'STAGE1':
                if not quiet: log.info('  --> Coron2Pipeline: skipping non-stage 1 file ' + tail)
            else:
                # Get expected output file name
                outfile_name = tail.replace('rateints.fits', 'calints.fits')
                fitsout_path = os.path.join(output_dir, outfile_name)

                # Skip if file already exists and overwrite is False.
                if os.path.isfile(fitsout_path) and not overwrite:
                    if not quiet: log.info('  --> Coron2Pipeline: skipping already processed file ' + tail)
                else:
                    if not quiet: log.info('  --> Coron2Pipeline: processing ' + tail)
                    res = run_single_file(fitspath, output_dir, steps=steps, 
                                          verbose=verbose, **kwargs)

                # Update spaceKLIP database.
                database.update_obs(key, j, fitsout_path, update_pxar=True)

            # Also process rate files?
            if do_rates:
                fitspath     = fitspath.replace('rateints', 'rate')
                fitsout_path = fitsout_path.replace('calints', 'cal')
                if os.path.isfile(fitsout_path) and not overwrite:
                    if not quiet: log.info('  --> Coron2Pipeline: skipping already processed file ' + tail)
                else:
                    if not quiet: log.info('  --> Coron2Pipeline: processing rate.fits file')
                    res = run_single_file(fitspath, output_dir, steps=steps, 
                                          verbose=verbose, **kwargs)
    
