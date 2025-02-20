import matplotlib.pyplot as plt
import numpy as np
from spaceKLIP import utils as ut
from spaceKLIP.psf import JWST_PSF
from scipy.ndimage import median_filter
from astropy.stats import sigma_clip

import emcee
from scipy.ndimage import shift
import corner
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class MCMCTools():
    """
    The spaceKLIP image manipulation tools class.

    """

    def __init__(self,
                 data,
                 type=None,
                 kwargs={}):
        """
        Initialize the spaceKLIP image manipulation tools class.

        Parameters
        ----------


        Returns
        -------
        None.

        """

        # Make an internal alias of the spaceKLIP database class.
        # self.database = database
        self.crpix1 = (data.shape[-1] - 1.) / 2. + 1  # (data.shape[-1]) // 2. + 1.  # 1-indexed
        self.crpix2 = (data.shape[-1] - 1.) / 2. + 1  # (data.shape[-2]) // 2. + 1.  # 1-indexed
        if 'r' in kwargs['MCMC'].keys():
            self.r = kwargs['MCMC']['r']
        else:
            self.r = 0
        if 'nsteps' in kwargs['MCMC'].keys():
            self.nsteps = kwargs['MCMC']['nsteps']
        else:
            self.nsteps = 5000
        if 'nwalkers' in kwargs['MCMC'].keys():
            self.nwalkers = kwargs['MCMC']['nwalkers']
        else:
            self.nwalkers = 30
        if 'verbose' in kwargs['MCMC'].keys():
            self.verbose = kwargs['MCMC']['verbose']
        else:
            self.verbose = False
        if 'size' in kwargs['MCMC'].keys():
            self.size = kwargs['MCMC']['size'] + 1 if kwargs['MCMC']['size'] % 2 == 0 else kwargs['MCMC']['size']
        else:
            self.size = np.median(data, axis=0).shape[-1]
        if 'oversample' in kwargs['MCMC'].keys():
            self.oversample = kwargs['MCMC']['oversample']
        else:
            self.oversample = 1
        if 'x_guess' in kwargs['MCMC'].keys() and 'y_guess' in kwargs['MCMC'].keys():
            self.x_guess = kwargs['MCMC']['x_guess']
            self.y_guess = kwargs['MCMC']['y_guess']
            self.dx_guess = x_guess - (self.crpix1 - 1)
            self.dy_guess = y_guess - (self.crpix2 - 1)
        else:
            self.max_value = np.nanmax(np.median(data, axis=0))
            self.max_index = np.where(np.median(data, axis=0) == self.max_value)
            self.x_guess = self.max_index[1][0]
            self.y_guess = self.max_index[0][0]
            self. dx_guess = self.x_guess - (self.crpix1 - 1)
            self.dy_guess = self.y_guess - (self.crpix2 - 1)
        if 'flux_guess' in kwargs['MCMC'].keys():
            self.flux_guess = kwargs['MCMC']['flux_guess']
        else:
            self.flux_guess = np.nanmax(data)
        if 'contrast_guess' in kwargs['MCMC'].keys():
            self.contrast_guess = kwargs['MCMC']['contrast_guess']
        else:
            self.contrast_guess = 1e-1
        if 'sep_guess' in kwargs['MCMC'].keys():
            self.sep_guess = kwargs['MCMC']['sep_guess']
        else:
            self.sep_guess = 1
        if 'theta_guess' in kwargs['MCMC'].keys():
            self.theta_guess = kwargs['MCMC']['theta_guess']
        else:
            self.theta_guess = 0
        if 'x_limits' in kwargs['MCMC'].keys():
            self.dx_limits = [-kwargs['MCMC']['x_limits'], kwargs['MCMC']['x_limits']]
        else:
            self.dx_limits = [- 5, + 5]
        if 'y_limits' in kwargs['MCMC'].keys():
            self.dy_limits = [-kwargs['MCMC']['y_limits'], kwargs['MCMC']['y_limits']]
        else:
            self.dy_limits = [- 5, + 5]
        if 'flux_limits' in kwargs['MCMC'].keys():
            self.flux_limits = [self.flux_guess * 10 ** (-kwargs['MCMC']['flux_limits']),
                           self.flux_guess * 10 ** (kwargs['MCMC']['flux_limits'])]
        else:
            self.flux_limits = [self.flux_guess * 1e-1, self.flux_guess * 1e1]
        if 'contrast_limits' in kwargs['MCMC'].keys():
            # self.contrast_limits = [self.contrast_guess * 10 ** (-kwargs['MCMC']['contrast_limits']),
            #                    self.contrast_guess * 10 ** (kwargs['MCMC']['contrast_limits'])]# if self.contrast_guess * 10 ** (
                                   # kwargs['MCMC'].keys()) <= 1 else 1]
            self.contrast_limits = [kwargs['MCMC']['contrast_limits'][0], kwargs['MCMC']['contrast_limits'][1]]
        else:
            self.contrast_limits = [self.contrast_guess * 1e-1, self.contrast_guess * 1e1] # if self.contrast_guess * 1e1 <= 1 else 1]
        if 'sep_limits' in kwargs['MCMC'].keys():
            self.sep_limits = kwargs['MCMC']['sep_limits']
        else:
            self.sep_limits = [1, 10]
        if 'theta_limits' in kwargs['MCMC'].keys():
            self.theta_limits = kwargs['MCMC']['theta_limits']
        else:
            self.theta_limits = [0, 360]
        if 'binarity' in kwargs['MCMC'].keys():
            if type is not None:
                if type == 'SCI' and kwargs['MCMC']['binarity']:
                    self.binarity = True
                else:
                    self.binarity = False
            else:
                self.binarity = kwargs['MCMC']['binarity']
            if self.binarity:
                self.initial_guess = [0, 0, self.flux_guess, self.contrast_guess, self.sep_guess,
                                 self.theta_guess]  # x, y, flux, contrast, sep, theta
                self.limits = [self.dx_limits, self.dy_limits, self.flux_limits, self.contrast_limits, self.sep_limits,
                          self.theta_limits]  # x, y, flux, contrast, sep, theta
            else:
                self.binarity = False
                self.initial_guess = [0, 0, self.flux_guess]  # x, y, flux
                self.limits = [self.dx_limits, self.dy_limits, self.flux_limits]
        else:
            self.binarity = False
            self.initial_guess = [0, 0, self.flux_guess]  # x, y, flux
            self.limits = [self.dx_limits, self.dy_limits, self.flux_limits]  # x, y, flux
        if 'debug' in kwargs.keys():
            self.debug = kwargs['debug']
        else:
            self.debug = False
        if 'burnin' in kwargs['MCMC'].keys():
            self.burnin = kwargs['MCMC']['burnin']
        else:
            self.burnin = None
        if 'thin' in kwargs['MCMC'].keys():
            self.thin = kwargs['MCMC']['thin']
        else:
            self.thin = None
        pass

    def extract_subarray(self, data, center_x, center_y, size=3, flat_and_skip_center=True):
        """
        Extract a subarray from a 2D array based on the center coordinates and subarray size.

        Parameters:
        - data: 2D numpy array, the larger array.
        - center_x: The x-coordinate (row) of the center of the subarray.
        - center_y: The y-coordinate (column) of the center of the subarray.
        - size: The size of the subarray (e.g., 3 for a 3x3 subarray).

        Returns:
        - subarray: The extracted subarray.
        """
        half_size = int(size // 2)
        subarray = data[int(round(center_y)) - half_size:int(round(center_y)) + half_size + 1,
                   int(round(center_x)) - half_size:int(round(center_x)) + half_size + 1]

        if flat_and_skip_center:
            # Flatten the subarray and remove the central value
            subarray_flat = subarray.flatten()
            center_index = len(subarray_flat) // 2
            subarray = np.delete(subarray_flat, center_index)

        return subarray


    def plot_data_model_residual(self, data, psf_no_coronmsk=None, apername=None, filt=None, date=None,
                                 offsetpsf_func=None, vmin=None, vmax=None, vminres=None, vmaxres=None, mask=False,
                                 binarity=False, path2fitsfile=None, return_residuals=False):

        if psf_no_coronmsk is None:
            if offsetpsf_func is None:
                offsetpsf_func = JWST_PSF(apername,
                                          filt,
                                          date=date,
                                          fov_pix=data.shape[-1],
                                          oversample=2,
                                          sp=None,
                                          use_coeff=False)
            psf_no_coronmsk = offsetpsf_func.gen_psf([0, 0], return_oversample=False, quick=False)
        psf_no_coronmsk /= np.nanmax(psf_no_coronmsk)
        model = self.build_model_from_psf(self.best_fit_params, psf_no_coronmsk, binarity, shifted=False)

        if mask:
            w = np.where(np.median(data, axis=0) == np.nanmax(np.median(data, axis=0)))
            data_masked = self.extract_subarray(np.median(data, axis=0).copy(), w[1][0], w[0][0], size=30,
                                                flat_and_skip_center=False)

            ww = np.where(model == np.nanmax(model))
            model_masked = self.extract_subarray(model.copy(), ww[1][0], ww[0][0], size=30, flat_and_skip_center=False)
        else:
            data_masked = np.median(data, axis=0).copy()
            model_masked = model.copy()

        residual = data_masked - model_masked
        if vmin is None:
            vmin = np.nanmin(data_masked)
        if vmax is None:
            vmax = np.nanmax(data_masked)
        if vminres is None:
            vminres = np.nanmin(residual)
        if vmaxres is None:
            vmaxres = np.nanmax(residual)

        with plt.style.context('spaceKLIP.sk_style'):
            fig, ax = plt.subplots(1, 3, figsize=(21, 7))
            im0 = ax[0].imshow(data_masked, origin='lower', vmin=vmin, vmax=vmax)
            ax[0].set_title('Data')
            divider0 = make_axes_locatable(ax[0])
            cax0 = divider0.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im0, cax=cax0, orientation='vertical')
            im1 = ax[1].imshow(model_masked, origin='lower', vmin=vmin, vmax=vmax)
            ax[1].set_title('Model')
            divider1 = make_axes_locatable(ax[1])
            cax1 = divider1.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im1, cax=cax1, orientation='vertical')
            im2 = ax[2].imshow(residual, origin='lower', vmin=vminres, vmax=vmaxres)
            ax[2].set_title('Residual')
            divider2 = make_axes_locatable(ax[2])
            cax2 = divider2.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im2, cax=cax2, orientation='vertical')
            if path2fitsfile is not None:
                plt.savefig(path2fitsfile+'_residuals.png', bbox_inches='tight')
                plt.close()
            else:
                plt.show()
        if return_residuals:
            return data - model


    def mask_within_radius(self, image, xdat, ydat, xcen, ycen, r, x=0, y=0, c=np.nan):
        distance = np.sqrt((xdat - (x + xcen)) ** 2 + (ydat - (y + ycen)) ** 2)
        image[np.where(distance <= r)] = c
        return image


    def build_model_from_psf(self, params, psf, binarity, x_guess=0, y_guess=0, offsetpsf_func=None, shifted=True):
        '''
        Routine to inject a companion PSF around a single PSF, with an counter clockwise sangle and separation. 0 degree is up.
        :param params:
        :param psf:
        :param binarity:
        :param shifted:
        :return:
        '''

        #apply a median filter to remove eventual NaNs from the PSF
        nan_mask = np.isnan(psf)
        filtered = median_filter(np.where(nan_mask, np.nanmedian(psf), psf), size=5, mode='nearest')
        psf[nan_mask] = filtered[nan_mask]

        if binarity:
            # this step will inject a companion PSF around the primary PSF
            x, y, flux, contrast, sep, theta = params
            theta %= 360
            radian = theta / 180 * math.pi
            # dx, dy shift of the companion from the primary
            dx = -sep * math.sin(radian)
            dy = sep * math.cos(radian)
            psf1 = psf * flux

            if offsetpsf_func is None:
                psf2 = psf1 * contrast
                # Shift the PSF for the companion by the current dx and dy shift values
                shifted_psf2 = shift(psf2, shift=[dy, dx], mode='constant', cval=0.0)
            else:
                shifted_psf2 = offsetpsf_func.gen_psf([dx, dy], return_oversample=False, quick=False)
                shifted_psf2 /= np.nanmax(shifted_psf2)
                shifted_psf2 *= (flux * contrast)
            model = psf1 + shifted_psf2
        else:
            # this step will not inject a companion PSF around the primary PSF, and consider the primary alone
            x, y, flux = params
            model = psf * flux
        if shifted:
            # This step will shift the final model (single or binary PSF) by the current x_guess-x and y_guess-y shift values to center it somewhere in the tile
            shifted_model = shift(model, shift=[y_guess - y, x_guess - x], mode='constant', cval=0.0)
            return shifted_model
        else:
            return model

    def run(self, data, psf, x_guess=0, y_guess=0, r=0, nsteps=5000, ndim=3, nwalkers=32,
                    initial_guess=[0.0, 0.0, 1], limits=[[-15, 15], [-15, 15], [1, 1e4]], verbose=False, size=61,
                    binarity=False, filename='', pixelscale=1):
        '''
        Set up and run the MCMC Bayesian fit to find the shift to apply to the data to recenter the star at the center of the frame.
        Parameters:
        ----------
        data: 3D-array
            'SCI' extension data.
        psf: 2D-array
            PSF image from webbpsf.
        r: int, optional
            if greater than 0, mask the center of the data and the PSF (to use in case the central region is saturating)
            Default is 0.
        nsteps: int, optional
            nsteps for the MCMC fit. Default is 5000.
        ndim: int, optional
            ndim for the MCMC fit. Default is 3.
        nwalkers: int, optional
            nwalkers for the MCMC fit. Default is 32.
        initial_guess: list, optional
            initial_guess for [x_shift, y_shift, flux] for the MCMC fit. Default is [0.0, 0.0, 1].
        limits: list, optional
            limits for x_shift and y_shift for the MCMC fit:[[x_shift0,x_shift1], [y_shift0,y_shift1]]
            Default is [[-15,15], [-15,15]].
        verbose: boolean
            verbose option for the MCMC fit. Default is False.
        size: int, optional
            dimension of the tiles (data and PSF) created for the MCMC fit. Should be big enough to include at least part of the star in it
            Default is 61.
        binarity: boolean, optional
            fit two PSF to the data, adding the flux of the second star, angle theta, and the distance between the two PSFs to the parameters to fit.
            Default is False.

        Return:
        ----------
        best_fit_params: best fit parameters for x, y and flux
        '''
        global MCMC_mask_r
        MCMC_mask_r = r

        def log_likelihood(params, star_image, psf, centers, binarity, show_plots=False, vmin=None, vmax=None, vminres=None,
                           vmaxres=None, path2fitsfile=None):
            """Log-likelihood function for MCMC.

            Args:
                params (list): [x_shift, y_shift, flux]
                star_image (2D array): Observed image of the star.
                psf (2D array): Normalized PSF (flux sum is 1).

            Returns:
                float: Log-likelihood value.
            """
            model = self.build_model_from_psf(params, psf, binarity)

            ydat, xdat = np.indices(model.shape)

            if MCMC_mask_r > 0:
                shifted_masked_psf = self.mask_within_radius(model, xdat, ydat, centers[0], centers[1], MCMC_mask_r,
                                                             c=np.nan)
                masked_star_image = self.mask_within_radius(star_image, xdat, ydat, centers[0], centers[1], MCMC_mask_r,
                                                            c=np.nan)
            else:
                shifted_masked_psf = model
                masked_star_image = star_image

            # # Scale the PSF by the flux value

            # Compute the residual between the star and the shifted, scaled PSF
            residual = masked_star_image - shifted_masked_psf

            # Assuming Gaussian errors, the log-likelihood is proportional to the chi-squared
            log_likelihood = -0.5 * np.nansum(residual ** 2)

            if show_plots:
                with plt.style.context('spaceKLIP.sk_style'):
                    if vmin is None:
                        vmin = np.nanmin(masked_star_image)
                    if vmax is None:
                        vmax = np.nanmax(masked_star_image)
                    if vminres is None:
                        vminres = np.nanmin(residual)
                    if vmaxres is None:
                        vmaxres = np.nanmax(residual)
                    fig, ax = plt.subplots(1, 3, figsize=(21, 7))
                    im0 = ax[0].imshow(masked_star_image, origin='lower', vmin=vmin, vmax=vmax)
                    ax[0].set_title('Data')
                    divider0 = make_axes_locatable(ax[0])
                    cax0 = divider0.append_axes('right', size='5%', pad=0.05)
                    fig.colorbar(im0, cax=cax0, orientation='vertical')

                    im1 = ax[1].imshow(shifted_masked_psf, origin='lower', vmin=vmin, vmax=vmax)
                    ax[1].set_title('Model')
                    divider1 = make_axes_locatable(ax[1])
                    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
                    fig.colorbar(im1, cax=cax1, orientation='vertical')

                    im2 = ax[2].imshow(masked_star_image - shifted_masked_psf, origin='lower', vmin=vminres, vmax=vmaxres)
                    ax[2].set_title('Residual')
                    divider2 = make_axes_locatable(ax[2])
                    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
                    fig.colorbar(im2, cax=cax2, orientation='vertical')
                    # plt.tight_layout()
                    if path2fitsfile is not None:
                        plt.savefig(path2fitsfile, bbox_inches='tight')
                        plt.close()
                    else:
                        plt.show()
            return log_likelihood

        def log_prior(params, limits, binarity, flux_limit):
            """Log-prior function for MCMC.

            Args:
                params (list): [x_shift, y_shift, flux]
                limits (list): limits for [x_shift, y_shift]

            Returns:
                float: Log-prior value (log(1) for uniform priors, or -inf if out of bounds).
            """
            if binarity:
                # x_shift, y_shift, flux1, flux2, sep, theta = params
                x_shift, y_shift, flux, contrast, sep, theta = params

                # Define uniform priors
                if (limits[0][0] < x_shift < limits[0][1]) and (limits[1][0] < y_shift < limits[1][1]) and limits[2][
                    0] < flux < limits[2][1] and limits[3][0] < contrast < limits[3][1] and (
                        limits[4][0] < sep < limits[4][1]) and (limits[5][0] < theta <= limits[5][1]):
                    return 0.0
                else:
                    return -np.inf
            else:
                x_shift, y_shift, flux = params

                # Define uniform priors
                if (limits[0][0] < x_shift < limits[0][1]) and (limits[1][0] < y_shift < limits[1][1]) and limits[2][
                    0] < flux < limits[2][1]:
                    return 0.0
                else:
                    return -np.inf

        def log_posterior(params, star_image, psf, limits, centers, binarity, show_plots=False, vmin=None, vmax=None,
                          vminres=None, vmaxres=None, path2fitsfile=None):
            """Log-posterior function for MCMC.

            Args:
                params (list): [x_shift, y_shift, flux]
                star_image (2D array): Observed image of the star.
                psf (2D array): Normalized PSF (flux sum is 1).
                limits (list): limits for [x_shift, y_shift]

            Returns:
                float: Log-posterior value (log-likelihood + log-prior).
            """

            lp = log_prior(params, limits, binarity, flux_limit=math.ceil(np.nanmax(star_image)))
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_likelihood(params, star_image, psf, centers, binarity, show_plots=show_plots, vmin=vmin,
                                       vmax=vmax, vminres=vminres, vmaxres=vmaxres, path2fitsfile=path2fitsfile)

        log.info('--> Running MCMC fit')
        data_masked = self.extract_subarray(data.copy(), x_guess, y_guess, size=30,
                                            flat_and_skip_center=False)

        psf_masked = self.extract_subarray(psf.copy(), np.where(psf == np.nanmax(psf))[1][0],
                                           np.where(psf == np.nanmax(psf))[0][0], size=30,
                                           flat_and_skip_center=False)

        # centers=[(data_masked.shape[0]) // 2, (data_masked.shape[1]) // 2]
        centers = [(data_masked.shape[-1] - 1.) / 2., (data_masked.shape[-1] - 1.) / 2.]

        # psf_masked = psf.copy()
        # Initialize the MCMC sampler
        # Add a small random offset to the initial guess to initialize walkers
        pos = initial_guess + 1e-4 * np.random.randn(nwalkers, ndim)

        moves = [(emcee.moves.DEMove(), 0.7), (emcee.moves.DESnookerMove(), 0.3), ]
        # Create the MCMC sampler object
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, moves=moves,
                                        args=(data_masked, psf_masked, limits, centers, binarity))

        # Run the MCMC sampler for a number of steps

        sampler.run_mcmc(pos, nsteps, progress=verbose)

        # Extract the samples and compute the best-fit parameters
        samples = sampler.get_chain(flat=True)
        self.best_fit_params = []

        tau = sampler.get_autocorr_time(tol=0)
        if self.burnin is None:
            burnin = int(nsteps * 0.60)
        else:
            burnin = self.burnin
        if self.thin is None:
            thin = int(0.5 * np.nanmin(tau))
        else:
            thin = self.thin

        flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
        # filtered_flat_sample=sigma_clip(flat_samples.copy(), sigma=5, maxiters=5,axis=0)
        pranges=[]
        for i in range(flat_samples.shape[1]):
            pranges.append((np.nanmin(flat_samples[:, i][np.isfinite(flat_samples[:, i])]),
                            np.nanmax(flat_samples[:, i][np.isfinite(flat_samples[:, i])])))
        if binarity:
            labels = ["x", "y", "flux", "contrast", "sep", "theta"]
        else:
            labels = ["x", "y", "flux"]

        with plt.style.context('spaceKLIP.sk_style'):
            samples = sampler.get_chain()  # Shape: (n_steps, n_walkers, n_dim)
            n_walkers = samples.shape[1]
            fig, ax = plt.subplots(len(labels), 1, figsize=(20, 20), sharex=True)
            for elno in range(len(labels)):
                for i in range(n_walkers):
                    ax[elno].plot(samples[:, i, elno], alpha=0.5)
                    ax[elno].axvline(burnin, color='k', linestyle='--')
                ax[elno].set_ylabel(f"{labels[elno]}")
            ax[elno].set_xlabel("Step number")
            plt.savefig(filename + '_traces.png')

        for i in range(ndim):
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            txt = "\mathrm{{{3}}} = {0:.5f}_{{-{1:.5f}}}^{{{2:.5f}}}"
            txt = txt.format(mcmc[1], q[0], q[1], labels[i])
            self.best_fit_params.append(mcmc[1])
            if verbose: print(txt)

        with plt.style.context('spaceKLIP.sk_style'):
            fig = corner.corner(flat_samples, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True, prange=pranges,
                                title_kwargs={"fontsize": 12}, title_fmt=".5f")
            plt.savefig(filename + '_corners.png')
            plt.close()

        if self.debug:
            log_posterior(self.best_fit_params, data_masked, psf_masked, limits, centers, binarity, show_plots=True, vmin=0,
                          vmax=5000, path2fitsfile=filename + '_log_posterior_residuals.png')