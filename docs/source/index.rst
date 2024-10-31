.. This file should at least contain the root `toctree` directive.


Documentation for spaceKLIP
===========================

.. image:: _static/logo.png
   :align: center
   :width: 200px

SpaceKLIP is a Python-based data reduction and analysis toolkit for JWST high-contrast imaging data.
The development has mostly been focused on coronagraphic imaging with NIRCam and MIRI, but many tools are also applicable
to full pupil imaging observations.
It is built on top of `pyKLIP <https://pyklip.readthedocs.io/en/latest/>`_ and provides functionality and customizations specific to JWST data. 

.. note::
   **Citing spaceKLIP:** 
   If spaceKLIP is useful for your research, 
   please cite 
   `Kammerer et al. 2022, Proc SPIE. <https://ui.adsabs.harvard.edu/abs/2022SPIE12180E..3NK/abstract>`_ and
   `Carter et al. 2023, ApJL <https://ui.adsabs.harvard.edu/abs/2023ApJ...951L..20C/abstract>`_. 


On a high level, spaceKLIP consists of three different reduction steps:

  1. Tools to **run the official JWST data reduction pipeline**, with some customizations of parameters optimized for high contrast. The jwst pipeline transforms raw uncal data into flux-calibrated cal/calints images. It takes care of dark subtraction, flat-fielding, photometric calibration, and much more. See `the JWST pipeline docs <https://jwst-pipeline.readthedocs.io/en/latest/>`_ for more information. SpaceKLIP contains a customized implementation of the JWST pipeline that has been optimized for high-contrast imaging, though it is still possible to run the default JWST pipeline by deactivating all of the custom steps/modifications.
  2. Tools to **further process the cal/calints images and prepare them for PSF subtraction**, either with KLIP or classical PSF subtraction techniques. These tools can be found in the :ref:`imagetools` module. For instance, bad pixels need to be cleaned, all PSFs need to be aligned properly before running the PSF subtraction, and in the case of MIRI, a dedicated background subtraction has to be performed. There are also lots of other little functions that can be applied to the cal/calints images, e.g., high-pass filtering, cropping, or padding of the images.
  3. Finally, **PSF subtraction** can be performed using several algorithms, including `pyKLIP <https://pyklip.readthedocs.io/en/latest/>`_, the official JWST stage 3 pipeline (which is a custom implementation of the KLIP algorithm), or classical PSF subtraction method. After this step, high-contrast companions should be visible/detectable in the data.


SpaceKLIP also provides additional functions for post-pipeline scientific analyses of the PSF subtracted images. These functions can be found in the :ref:`analysistools` module. There are functions to:

 * retrieve the properties of detected companions 
 * to compute basic and calibrated contrast curves
 * to inject one or more companions into a given data set. 

There are several tutorial Jupyter notebooks which show how to use these
different spaceKLIP functionalities. While the companion retrieval/injection is
pretty standardized, the data reduction with the ImageTools module is highly
customizable. The user can decide in which order certain steps shall be run,
and even the JWST metadata like image centers and dither positions can be
adapted by the user if necessary.




Compatible Simulated Data: `Here <https://stsci.box.com/s/cktghuyrwrallb401rw5y5da2g5ije6t>`_

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   TLDR
   Installation-and-dependencies
   recommendations

.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   :hidden:

   tutorials
   tutorials/tutorial_NIRCam_reductions.ipynb
   tutorials/tutorial_MIRI_reductions.ipynb
   tutorials/tutorial_NIRCam_contrast_analyses.ipynb
   tutorials/tutorial_MIRI_contrast_analyses.ipynb
   tutorials/MAST query tools for coronagraphic datasets.ipynb

.. toctree::
   :maxdepth: 1
   :caption: spaceKLIP Functionality (Unfinished)
   :hidden:

   stage1
   stage2
   imagetools
   subtraction
   analysistools

.. toctree::
   :maxdepth: 1
   :caption: About
   :hidden:

   Contact
   Attribution

.. toctree::
   :maxdepth: 1
   :caption: Package content
   :hidden:

   spaceKLIP






