import os, glob
import spaceKLIP

import pytest

test_csv_path = '/Users/sbogat/Documents/01_Research/JWST/4050_workspace/ref_lib_2024-07-10.csv'

def test_build_refdb():

    # Check nonexistent input directory

    # Check empty input directory 

    # Check no calints files in input directory

    # Check header kw missing

    # Check synthetic star name

    # Check no Gaia DR3 ID

    # Check duplicate science target with different program names

    # Check suffix kw behavior

    # Check overwrite kw behavior

    # Check preference for SIMBAD vs MOCA

    pass

def test_get_sciref_files():

    # - sci_target in index

    # - sci_target in 2MASS_ID

    # - sci_target in TARGPROP

    # - sci_target not in refdb 

    # - exceptions if 0 science fnames are returned
    
    # - warnings if 0 reference observations are returned

    pass