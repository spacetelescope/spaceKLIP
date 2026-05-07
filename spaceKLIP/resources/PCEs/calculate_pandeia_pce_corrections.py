"""
NIRCam has multiple detectors, but Pandeia will assume the mean PCE for a given filter
This is strictly not the case for actual observations and each mode uses the following:
- SW Subarray Imaging: Likely using SUB***P subarrays, which are on the B1 detector
- LW Subarray Imaging: Likely using SUB***P subarrays, which are on the B5 detector
- LW Coronagraphy: A5 detector
- SW Coronagraphy: A2 detector for circle masks, A4 detector for bar masks

Ideally, want to precompute some corrections to convert from the mean Pandeia PCE, to a detector dependent PCE
based on the observation actually being performed. spaceKLIP can then use the detector header keyword to
apply the appropriate correction to the PCE curve for a given filter and mode.

To do this, need to download the filters from JDocs - here:
https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-filters#gsc.tab=0
"""

import glob, os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Set the base directory for the NIRCam throughputs
base_dir = os.path.join(os.path.dirname(__file__), 'nircam_throughputs') # This is the JDocs download directory
mean_throughputs = os.path.join(base_dir, 'mean_throughputs')
detector_throughputs = os.path.join(base_dir, 'detector_based_throughputs')

# Load all the mean throughputs
mean_files = glob.glob(os.path.join(mean_throughputs, '*.txt'))

# Create a dictionary to save the correction factors for each filter and detector
correction_factors = {}
for mean_file in mean_files:
    # Extract the filter name from the file name
    filter_name = os.path.basename(mean_file).split('_')[0]

    # Expand the correction factory dictionary for this filter
    correction_factors[filter_name] = {}

    # Load the mean throughput data
    mean_data = np.genfromtxt(mean_file, skip_header=1).transpose()
    mean_wave = mean_data[0]
    mean_throughput = mean_data[1]

    # Save the wavelength array, this is shared across all the files for the same filter
    correction_factors[filter_name]['wavelength'] = mean_wave

    plt.plot(mean_wave, mean_throughput)

    # Find the corresponding detector throughputs for this filter using glob
    detector_files = glob.glob(detector_throughputs+f'/*{filter_name}_*.txt')
    print(detector_files)

    for detector_file in detector_files:
        # Extract the detector name from the file name
        detector_name = os.path.basename(detector_file).split('_')[0]

        # Load the detector throughput data
        detector_data = np.genfromtxt(detector_file, skip_header=1).transpose()
        detector_wave = detector_data[0]
        detector_throughput = detector_data[1]

        # Calculate the correction factor, if we multiply the mean throughput (Pandeia calculation) by this factor,
        # we get the individual detector throughput
        correction_factor = detector_throughput / mean_throughput

        # As we might have divided by zero, need to set any NaN or inf values to 1 (i.e. no correction)
        correction_factor[np.isnan(correction_factor)] = 1
        correction_factor[np.isinf(correction_factor)] = 1

        # Save the correction factor in the dictionary
        correction_factors[filter_name][detector_name] = correction_factor

# Now we have the correction factors for each filter and detector, we can save this to a pickle file for later use
with open(os.path.join(os.path.dirname(__file__), 'pce_correction_factors.pkl'), 'wb') as f:
    pickle.dump(correction_factors, f)

