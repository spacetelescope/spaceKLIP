from __future__ import division

import matplotlib

# =============================================================================
# IMPORTS
# =============================================================================

import os
import pdb
import sys

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

import importlib
import webbpsf_ext

import astropy.units as u

from astropy.table import Table
from astroquery.svo_fps import SvoFps
from synphot import Observation, SourceSpectrum, SpectralElement
from synphot.models import Empirical1D
from synphot.units import convert_flux

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

from spaceKLIP.plotting import load_plt_style
# =============================================================================
# MAIN
# =============================================================================

def read_spec_file(starfile):
    """
    Read a spectrum from a TXT file.
    
    Parameters
    ----------
    starfile : path
        Path of two column TXT file with wavelength (micron) and flux (Jy).
    
    Returns
    -------
    sed : synphot.SourceSpectrum
        Spectrum of the source.
    
    """
    
    try:
        data = np.genfromtxt(starfile).transpose()
        model_wave = data[0]
        model_flux = data[1]    
        sed = SourceSpectrum(Empirical1D, points=model_wave << u.Unit('micron'), lookup_table=model_flux << u.Unit('Jy'))
        sed.meta['name'] = starfile.split('/')[-1]
    except:
        raise ValueError('Unable to read provided starfile; ensure format is in two columns with wavelength (microns), flux (Jy)')
    
    return sed

def read_votable(starfile, spectral_type, instrume, output_dir, plot_style, **kwargs):
    """
    Read a spectrum from a VOTable file.

    Parameters
    ----------
    starfile : path
        Path of VizieR VOTable containing host star photometry.
    spectral_type : str
        Host star spectral type for the stellar model SED.
    instrume : 'NIRCAM' or 'MIRI'
        JWST instrument in use.
    output_dir : path, optional
        Path of the directory where the SED plot shall be saved. The default is None.
    plot_style : str, optional
        Matplotlib style to use for the SED plot. The default is None.

    Returns
    -------
    sed : synphot.SourceSpectrum
        Spectrum of the source.

    """

    # Initialize SED in random bandpass and at random magnitude.
    bp_k = webbpsf_ext.bp_2mass('k')
    bp_mag = 5.
    try:
        spec = webbpsf_ext.spectra.source_spectrum(name='Input Data & SED', sptype=spectral_type,
                                                   mag_val=bp_mag, bp=bp_k, votable_file=starfile,
                                                   **kwargs)
    except:
        spec = webbpsf_ext.spectra.source_spectrum(name='Input Data & SED', sptype=spectral_type,
                                                   mag_val=bp_mag, bp=bp_k, votable_input=starfile,
                                                   **kwargs)

    # Split between NIR and MIR exposures.
    if instrume == 'MIRI':
        wlim = [5., 30.]
    else:
        wlim = [1., 5.]

    # Fit SED to input photometry and plot SED.
    spec.fit_SED(x0=[1.], wlim=wlim, use_err=False, verbose=False)
    if output_dir is not None:
        load_plt_style(plot_style)
        spec.plot_SED()
        plt.savefig(os.path.join(output_dir, 'sed.pdf'))
        plt.close()

    # Convert units to photlam.
    input_flux = u.Quantity(spec.sp_model.flux, str(spec.sp_model.fluxunits).lower())
    photlam_flux = convert_flux(spec.sp_model.wave, input_flux, out_flux_unit='photlam')
    sed = SourceSpectrum(Empirical1D, points=spec.sp_model.wave << u.Unit(str(spec.sp_model.waveunits)),
                         lookup_table=photlam_flux << u.Unit('photlam'))

    return sed

def get_stellar_magnitudes(starfile,
                           spectral_type,
                           instrume,
                           detector,
                           exp_type,
                           output_dir=None,
                           plot_style=None,
                           **kwargs):
    """
    Get the source brightness and zero point fluxes in each filter of the JWST
    instrument in use, using per-detector PCE data.

    Parameters
    ----------
    starfile : path
        Path of VizieR VOTable containing host star photometry or two
        column TXT file with wavelength (micron) and flux (Jy).
    spectral_type : str, optional
        Host star spectral type for the stellar model SED. The default is
        'G2V'.
    instrume : 'NIRCAM' or 'MIRI'
        JWST instrument in use.
    detector : str
        Detector name (e.g. 'NRCALONG', 'NRCA1', 'MIRIMAGE').
    exp_type : str
        Exposure type from the FITS header (e.g. 'NRC_CORON', 'NRC_IMAGE',
        'MIR_4QPM', 'MIR_LYOT', 'MIR_IMAGE').
    output_dir : path, optional
        Path of the directory where the SED plot shall be saved. The default is
        None.
    
    Keyword Args
    ------------
    Teff : float
        Effective temperature ranging from 3500K to 30000K.
    metallicity : float
        Metallicity [Fe/H] value ranging from -2.5 to 0.5.
    log_g : float
        Surface gravity (log g) from 0 to 5.
    Av : float
        Add extinction to the stellar spectrum
    catname : str
        Catalog name, including 'bosz', 'ck04models', and 'phoenix'.
        Default is 'bosz', which comes from :func:`BOSZ_spectrum`.
    res : str
        Spectral resolution to use (200 or 2000 or 20000).
    interpolate : bool
        Interpolate spectrum using a weighted average of grid points
        surrounding the desired input parameters. Default: True
    radius : float
        Search radius in arcseconds for Vizier query.
        Default: 1 arcsec.

    Returns
    -------
    mstar : dict
        Dictionary of the source brightness (vegamag) in each filter of the
        JWST instrument in use.
    fzero : dict
        Dictionary of the zero point flux (Jy) of each filter of the JWST
        instrument in use.
    fzero_flam : dict
        Dictionary of the zero point flux (erg/s/cm^2/A) of each filter of the
        JWST instrument in use.
    fzero_wm2um : dict
        Dictionary of the zero point flux (W/m^2/um) of each filter of the
        JWST instrument in use.
    
    """

    # Handle deprecated return_si kwarg
    if 'return_si' in kwargs:
        log.warning('return_si is deprecated and ignored. get_stellar_magnitudes '
                    'now always returns (mstar, fzero, fzero_flam, fzero_wm2um).')
        kwargs.pop('return_si')

    # First step is to gather the SED of the source.
    # We can do this in two ways depending on the format of the input file:
    # VOTable.
    if starfile[-4:] == '.vot':
        sed = read_votable(starfile, spectral_type, instrume, output_dir, plot_style, **kwargs)
    # TXT file.
    else:
        sed = read_spec_file(starfile)

    # Load the consolidated PCE data and find all filters for this
    # instrument/detector/exp_type combination.
    from spaceKLIP.utils import load_pce_data, get_pce_info

    pce_data = load_pce_data()
    instrume_lower = instrume.lower()
    detector_upper = detector.upper()

    # Resolve mode to find matching filter keys
    if instrume_lower == 'nircam':
        if exp_type == 'NRC_CORON':
            mode = 'coronagraphy'
        elif detector_upper in ('NRCALONG', 'NRCBLONG'):
            mode = 'lw_imaging'
        else:
            mode = 'sw_imaging'
    elif instrume_lower == 'miri':
        if exp_type in ('MIR_4QPM', 'MIR_LYOT'):
            mode = 'coronagraphy'
        else:
            mode = 'imaging'
    else:
        raise ValueError(f'Unsupported instrument: {instrume}')

    # Map generic detector names to specific detector IDs
    if detector_upper == 'NRCALONG':
        detector_upper = 'NRCA5'
    elif detector_upper == 'NRCBLONG':
        detector_upper = 'NRCB5'

    # Find all filters available for this instrument/mode/detector
    key_suffix = '__wavelengths'
    key_prefix_match = f'{instrume_lower}__{mode}__'
    key_suffix_match = f'__{detector_upper}{key_suffix}'
    filts = []
    for key in pce_data.keys():
        if key.startswith(key_prefix_match) and key.endswith(key_suffix_match):
            # Extract filter name from key: instrument__mode__filt__detector__wavelengths
            filt_name = key[len(key_prefix_match):-len(key_suffix_match)]
            filts.append(filt_name)

    # Compute magnitude in each filter using the per-detector PCE bandpasses.
    mstar = {}  # vegamag
    fzero = {}  # Jy
    fzero_flam = {}  # erg/s/cm^2/A
    fzero_wm2um = {}  # W/m^2/um
    vegased = SourceSpectrum.from_vega()

    for filt in filts:
        try:
            pce_info = get_pce_info(instrume, filt, detector, exp_type)
        except KeyError as e:
            log.warning(f'Key error: Skipping filter {filt}: {e}')
            continue

        # Read bandpass wavelengths (micron) and throughput from PCE data.
        bandpass_wave = pce_info['wavelengths'] * 1e4  # micron -> Angstrom
        bandpass_throughput = pce_info['pce']

        # Create bandpass object.
        bandpass = SpectralElement(Empirical1D, points=bandpass_wave,
                                  lookup_table=bandpass_throughput)

        # Compute magnitude.
        obs = Observation(sed, bandpass, binset=bandpass.waveset)
        mag = obs.effstim(flux_unit='vegamag', vegaspec=vegased).value
        filt_upper = filt.upper()
        mstar[filt_upper] = mag
        fzero[filt_upper] = float(pce_info['ZeroPointJy'])
        fzero_flam[filt_upper] = float(pce_info['ZeroPointFlam'])
        fzero_wm2um[filt_upper] = float(pce_info['ZeroPointWm2um'])

    return mstar, fzero, fzero_flam, fzero_wm2um
