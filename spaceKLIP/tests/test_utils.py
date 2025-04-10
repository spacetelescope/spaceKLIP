import os, glob
import numpy as np
import astropy
import spaceKLIP

import pytest

testdatapath = os.getenv('SPACEKLIP_TEST_DATA_PATH')

def test_utils_read_obs():
    """test the read_obs function

    """
    fitsfiles = glob.glob(os.path.join(testdatapath,  '*nrca*calints.fits'))
    fitsfiles.sort()
    fn = fitsfiles[0]

    data, erro, pxdq, head_pri, head_sci, is2d, alignshift, center_shift, align_mask, center_mask, maskoffs =  spaceKLIP.utils.read_obs(fn)

    assert isinstance(data, np.ndarray), "read_obs didn't return a numpy data array"
    assert isinstance(head_pri, astropy.io.fits.Header), "read_obs didn't return a FITS header"
