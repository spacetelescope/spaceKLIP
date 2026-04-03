"""
This script is used to create the all_pces.npz file, which is used to store the information about the
photon conversion efficiencies (PCEs) for each instrument/mode/detector/filter combination.

NOTE: This script requires the pandeia package to be installed, which is not in the spaceKLIP environment. If
you would like to run this script, it is recommended to set up a separate environment with pandeia installed.

"""
import os
import json
import pickle
from pandeia.engine.io_utils import read_json
from pandeia.engine.calc_utils import build_default_calc
from pandeia.engine.instrument_factory import InstrumentFactory
import synphot as syn
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt


def save_and_plot_pce(instrument, mode, filt, detector, wavelengths, pce, all_data=None):
    """
    Function to save the PCEs to individual files, plot them, and optionally accumulate
    trimmed arrays into a shared dict for the consolidated all_pces.npz output.

        Parameters
        ----------
        instrument : str
            The name of the instrument (e.g. 'nircam', 'miri').
        mode : str
            The observing mode (e.g. 'imaging', 'coronagraphy').
        filt : str
            The filter name (e.g. 'f444w').
        detector : str
            The detector name (e.g. 'NRCA5', 'NRCB2', 'MIRIMAGE').
        wavelengths : array-like
            The wavelength grid for the PCE calculation (in microns).
        pce : array-like
            The calculated PCE values (in e-/photon) corresponding to the wavelength grid.
        all_data : dict, optional
            If provided, trimmed wavelength and PCE arrays will be stored here
            under keys following the pattern {instrument}__{mode}__{filt}__{detector}__wavelengths/pce.

    """

    mask = pce > 1e-5 * np.max(pce)

    # Save the PCEs to an individual npz file as well for quick access
    save_dir = os.path.join(os.path.dirname(__file__), instrument, mode)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, '{}_{}.npz'.format(filt.upper(), detector.upper()))
    np.savez_compressed(save_path,
                        wavelengths=wavelengths[mask].astype(np.float32),
                        pce=pce[mask].astype(np.float32))

    # Accumulate into the consolidated dict if provided
    if all_data is not None:
        key_prefix = f'{instrument}__{mode}__{filt}__{detector.upper()}'
        all_data[f'{key_prefix}__wavelengths'] = wavelengths[mask].astype(np.float32)
        all_data[f'{key_prefix}__pce'] = pce[mask].astype(np.float32)

    # Save a plot of each PCE, trimming values close to zero for better visualization
    figure_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(figure_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(wavelengths[mask], pce[mask], color='C0')
    plt.title(f'{instrument.upper()} {mode} {filt.upper()} PCE', fontsize=16)
    plt.xlabel('Wavelength (um)', fontsize=14)
    plt.ylabel('PCE (e-/photon)', fontsize=14)
    plt.xlim(wavelengths[mask][0], wavelengths[mask][-1])
    plt.ylim(0, 1.1 * np.max(pce[mask]))
    plt.grid()
    plot_save_path = os.path.join(figure_dir, '{}_{}_{}_{}_PCE.png'.format(instrument, mode, filt, detector))
    plt.savefig(plot_save_path)
    plt.close()

    return

# First, we need a list of instruments, modes, and filters to loop through.
pce_concatenations = {'nircam': ['sw_imaging', 'lw_imaging', 'coronagraphy'],
                      'miri': ['imaging', 'coronagraphy'],}

# Also define a common wavelength grid to use for all PCE calculations, which should cover the full range of all filters
wavelengths = np.arange(0.5, 30, 0.001)

# Load the correction factors for the NIRCam filters, which we will apply to the PCEs to get the individual detector PCEs
nircam_corrections_path = os.path.join(os.path.dirname(__file__), 'pce_correction_factors.pkl')
with open(nircam_corrections_path, 'rb') as f:
    nircam_corrections = pickle.load(f)

# Make a dictionary to store the calculations
pce_calculations = {}
# Make a dictionary to accumulate all data for the consolidated npz file
all_data = {}
# Loop over instruments
for instrument, modes in pce_concatenations.items():
    # Expand the save dictionary
    pce_calculations[instrument] = {}

    # Loop over modes
    for mode in modes:
        # Expand the save dictionary
        pce_calculations[instrument][mode] = {}

        # Obtain compatible filters from pandeia reference data
        config = read_json(os.path.join(os.environ['pandeia_refdata'], 'jwst', instrument, 'config.json'))
        filters = config['mode_config'][mode]['filters']

        print('Identified filters for {} {}: {}'.format(instrument, mode, filters))

        # Loop over filters
        for filt in filters:
            pce_calculations[instrument][mode][filt] = {}
            # Assemble the calculation configuration, starting with the default
            calc = build_default_calc('jwst', instrument, mode)
            calc['configuration']['instrument']['filter'] = filt

            # Need to assign the coronagraphic mask and aperture for the coronagraphic modes
            if mode == 'coronagraphy':
                if instrument == 'nircam':
                    # Mask doesn't affect wavelength dependence of filter throughput, so we can just choose any mask
                    calc['configuration']['instrument']['coron_mask'] = 'mask335r'
                    calc['configuration']['instrument']['aperture'] = 'mask335r'
                    calc['configuration']['instrument']['pupil_mask'] = 'none'
                elif instrument == 'miri':
                    # Need to strip the filter name to get the mask name, e.g. f1065c -> fqpm1065
                    numerical_part = ''.join(filter(str.isdigit, filt))
                    miri_mask = 'fqpm' + numerical_part if numerical_part != '2300' else 'lyot' + numerical_part
                    calc['configuration']['instrument']['coron_mask'] = miri_mask
                    calc['configuration']['instrument']['aperture'] = miri_mask

            # Also need an individual check for NIRCam to assign the detector
            if instrument == 'nircam':
                if 'imaging' in mode:
                    detector = 'sw' if mode == 'sw_imaging' else 'lw'
                elif mode == 'coronagraphy':
                    # Use filter to assign the detector based on compatible filters with an example SW aperture.
                    example_sw_aperture = 'maskswbsw'
                    sw_filters = config['config_constraints']['apertures'][example_sw_aperture]['filters']['default']
                    detector = 'sw' if filt in sw_filters else 'lw'
                # Set the detector
                calc['configuration']['instrument']['detector'] = detector

            # Run the calculation to get the PCE
            inst = InstrumentFactory(config=calc['configuration'])
            pce = inst.get_total_eff(wavelengths)

            # NIRCam has multiple detectors, but Pandeia will assume the mean PCE for a given filter
            # This is strictly not the case for actual observations, as only one detector can be used.
            if instrument == 'nircam':
                pce_corrections = nircam_corrections[filt.upper()]

                for detector, correction in pce_corrections.items():
                    # One item is actually the wavelength array, so we need to skip that one
                    if detector == 'wavelength':
                        continue

                    # Need to interpolate the correction to the wavelength grid of the PCE calculation
                    correction_interp = np.interp(wavelengths, pce_corrections['wavelength'], correction,
                                                  left=1, right=1)  # Extrapolate with 1 outside the range

                    # Apply the correction to the PCE
                    pce_corrected = pce * correction_interp

                    # Store in the dictionary with the appropriate keys
                    pce_calculations[instrument][mode][filt][detector] = pce_corrected

                    # Save and plot the PCE for this filter and detector
                    save_and_plot_pce(instrument, mode, filt, detector, wavelengths, pce_corrected, all_data)
            elif instrument == 'miri':
                # Detector is always the same for imaging and coronagraphy
                detector = 'MIRIMAGE'
                pce_calculations[instrument][mode][filt][detector] = pce

                # Save and plot the PCE for this filter and detector
                save_and_plot_pce(instrument, mode, filt, detector, wavelengths, pce, all_data)

# With calculations in hand, want to save specific information, such as the
# name, zero point, mean wavelength, pivot wavelength, and effective width.

# Get vega spectrum for zero point calculations
vega = syn.SourceSpectrum.from_vega()
# Loop through the calculations
for instrument, modes in pce_calculations.items():
    for mode, filters in modes.items():
        for filt, detectors in filters.items():
            for detector, pce in detectors.items():
                # Calculate the mean wavelength, pivot wavelength, and effective width for each filter
                lam = wavelengths
                T = np.array(pce)
                mean_wave = np.trapz(lam * T, lam) / np.trapz(T, lam)
                pivot_wave = np.sqrt(np.trapz(lam * T, lam) / np.trapz(T / lam, lam))
                eff_width = np.trapz(T, lam)

                # Calculate the zero point flux in Jy
                bp = syn.SpectralElement(syn.models.Empirical1D, points=lam*u.micron, lookup_table=T, keep_neg=True)
                obs = syn.Observation(vega, bp)
                zp_jy = obs.effstim(flux_unit='Jy')
                zp_flam = obs.effstim(flux_unit='flam')  # erg/s/cm^2/A
                zp_wm2um = zp_flam.value * 1e-7 * 1e4 * 1e4  # W/m^2/um

                print(f'{instrument} {mode} {filt}: mean_wave={mean_wave:.3f} um, '
                      f'pivot_wave={pivot_wave:.3f} um, eff_width={eff_width:.3f} um, zp_jy={zp_jy:.3e}')

                # Also store metadata in the consolidated all_data dict
                key_prefix = f'{instrument}__{mode}__{filt}__{detector}'
                all_data[f'{key_prefix}__WavelengthMean'] = np.float32(mean_wave * 1e4)  # angstrom
                all_data[f'{key_prefix}__WavelengthPivot'] = np.float32(pivot_wave * 1e4)  # angstrom
                all_data[f'{key_prefix}__WidthEff'] = np.float32(eff_width * 1e4)  # angstrom
                all_data[f'{key_prefix}__ZeroPointJy'] = np.float32(zp_jy.value)  # Jy
                all_data[f'{key_prefix}__ZeroPointFlam'] = np.float32(zp_flam.value)  # erg/s/cm^2/A
                all_data[f'{key_prefix}__ZeroPointWm2um'] = np.float32(zp_wm2um)  # W/m^2/um

# Save the consolidated all_pces.npz file
all_pces_path = os.path.join(os.path.dirname(__file__), 'all_pces.npz')
np.savez_compressed(all_pces_path, **all_data)
print(f'Saved consolidated PCE data to {all_pces_path}')

