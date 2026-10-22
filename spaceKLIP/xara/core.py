''' --------------------------------------------------------------------
       XARA: a package for eXtreme Angular Resolution Astronomy
    --------------------------------------------------------------------
    ---
    xara is a python module to create, and extract Fourier-phase data
    structures, using the theory described in the two following papers:
    - Martinache, 2010, ApJ, 724, 464.
    - Martinache, 2013, PASP, 125, 422.

    This file contains several tools used by the KPI and KPO classes to
    create and manipulate kernel- and eigen-phase data structures.
    -------------------------------------------------------------------- '''

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt2d as medfilt
from scipy.special import j1
from scipy.optimize import leastsq

''' ================================================================
    small tools and functions useful for the manipulation of
    Ker-phase data.
    ================================================================ '''

shift = np.fft.fftshift
fft = np.fft.fft2
ifft = np.fft.ifft2

dtor = np.pi/180.0
i2pi = 1j*2.0*np.pi


# =========================================================================
def _xyic(ys, xs, between_pix=False):
    ''' --------------------------------------------------------------
    Private utility: returns two arrays of (y, x) image coordinates

    Array values give the pixel coordinates relative to the center of
    the array, that can be offset by 0.5 pixel if between_pix is set
    to True

    Parameters:
    ----------
    - ys: (integer) vertical size of the array (in pixels)
    - xs: (integer) horizontal size of the array (in pixels)
    - between_pix: (boolean) places center of array between pixels

    Returns:
    -------
    A tuple of two arrays: (yy, xx) ... IN THAT ORDER!
    -------------------------------------------------------------- '''
    offset = 0
    if between_pix is True:
        offset = 0.5
    xx = np.outer(np.ones(ys), np.arange(xs)-xs//2+offset)
    yy = np.outer(np.arange(ys)-ys//2+offset, np.ones(xs))
    return (yy, xx)


# =========================================================================
def _dist(ys, xs, between_pix=False):
    ''' --------------------------------------------------------------
    Private utility: returns a distance 2D array

    Array values give the distance to array center, that can be offset
    by 0.5 pixel if between_pix is set to True

    Parameters:
    ----------
    - ys: (integer) vertical size of the array (in pixels)
    - xs: (integer) horizontal size of the array (in pixels)
    - between_pix: (boolean) places center of array between pixels
    -------------------------------------------------------------- '''
    yy, xx = _xyic(ys, xs, between_pix=between_pix)
    return np.hypot(yy, xx)


# =========================================================================
def polar_coord_map(ys, xs, scale=1.0, between_pix=False):
    ''' --------------------------------------------------------------
    Returns two 2D arrays of polar coordinates (distance & azimut)
    relative to the center of a grid map of size (ys x xs), that can
    be offset by 0.5 pixel if between_pix is set to True

    Parameters:
    ----------
    - ys    : (integer) vertical size of the array (in pixels)
    - xs    : (integer) horizontal size of the array (in pixels)
    - scale : (float) scaling factor for distance (default=1.0)
    - between_pix: (boolean) places center of array between pixels

    Returns:
    -------
    a tuple of two 2D array:
    - distance: in "pixels"
    - azimut: in degrees (0 -> 360)

    Remarks:
    -------
    The use of this function assumes that azimuts are measured from the
    map "north" and run counter-clockwise!
    -------------------------------------------------------------- '''
    yy, xx = _xyic(ys, xs, between_pix=between_pix)
    azim = -np.arctan2(xx, yy)/dtor % 360
    return scale*np.hypot(yy, xx), azim


# =========================================================================
def mas2rad(x):
    ''' Convenient little function to convert milliarcsec to radians '''
    return(x * 4.8481368110953599e-09)  # *= np.pi/(180*3600*1000)


# =========================================================================
def rad2mas(x):
    '''  convert radians to mas'''
    return(x / 4.8481368110953599e-09)  # /= np.pi/(180*3600*1000)


# =========================================================================
def colinearity_map(smaps, signal):
    ''' --------------------------------------------------------------
    Computes the colinearity map between precomputed signals on a grid
    of positions (smaps) and a signal.

    Parameters:
    ----------
    - smaps: a 3 or 4D array of precomputed signals
    - signal: a 1 or 2D array of astrophysical signal

    Note:
    ----
    Assumes that you know what you do and have matched the
    dimensions of *smaps* and *signal*
    -------------------------------------------------------------- '''
    if len(smaps.shape) == 4:
        tmp = np.tensordot(signal, smaps, axes=([0, 1], [0, 1]))
    else:
        tmp = np.tensordot(signal, smaps, axes=1)

    return tmp / np.tensordot(signal, signal)


# =========================================================================
def rebin(a, shape):
    sh = shape[0], a.shape[0]//shape[0], shape[1], a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)


# =========================================================================
def negentropy(a):
    ''' Evaluate the negentropy associated to an strict positive array "a".
    The result is a positive number.
    '''
    a1 = a.copy() / a.sum()
    entm = np.log10(a1) * a1
    return -1.0 * entm.sum()


# =========================================================================
def cvis_binary(u, v, wavel, p, detpa=None):
    ''' Calc. complex vis measured by an array for a binary star
    ----------------------------------------------------------------
    p: 3-component vector (+2 optional), the binary "parameters":
    - p[0] = sep (mas)
    - p[1] = PA (deg) E of N.
    - p[2] = contrast ratio (primary/secondary)

    optional:
    - p[3] = angular size of primary (mas)
    - p[4] = angular size of secondary (mas)

    - u,v: baseline coordinates (meters)
    - wavel: wavelength (meters)

    - detpa: detector position angle (degrees)
    ---------------------------------------------------------------- '''
    if detpa is None:
        th0 = 0.0
    else:
        th0 = detpa * dtor

    p = np.array(p)
    # relative locations
    th = p[1] * dtor
    ddec = mas2rad(p[0] * np.cos(th + th0))
    dra = -mas2rad(p[0] * np.sin(th + th0))

    # baselines into number of wavelength
    x = np.hypot(u, v)/wavel

    # decompose into two "luminosity"
    l2 = 1. / (p[2] + 1)
    l1 = 1 - l2

    # phase-factor
    phi = np.exp(-i2pi*(u*dra + v*ddec)/wavel)

    # optional effect of resolved individual sources
    if p.size == 5:
        th1, th2 = mas2rad(p[3]), mas2rad(p[4])
        v1 = 2*j1(np.pi*th1*x)/(np.pi*th1*x)
        v2 = 2*j1(np.pi*th2*x)/(np.pi*th2*x)
    else:
        v1 = np.ones(u.size, dtype=u.dtype)
        v2 = np.ones(u.size, dtype=u.dtype)

    cvis = l1 * v1 + l2 * v2 * phi

    return cvis


# =========================================================================
def grid_precalc_aux_cvis(u, v, wavel, mgrid, gscale):
    ''' Pre-calculates an auxilliary array necessary for the complex
    visibility modeling of a "complex" (i.e. grid-type) source
    ----------------------------------------------------------------
    - u,v:    baseline coordinates (meters)
    - wavel:  wavelength (meters)
    - mgrid:  square grid array describing the object
    - gscale: scalar defining the "mgrid" pitch size (in mas)
    ---------------------------------------------------------------- '''
    # relative locations
    sz, dz = mgrid.shape[0], mgrid.shape[0] / 2
    xx, yy = np.meshgrid(np.arange(sz)-dz, np.arange(sz)-dz)

    # flatten all 2D arrays
    dra = mas2rad(gscale * np.ravel(xx))
    ddec = mas2rad(gscale * np.ravel(yy))

    c0 = -2j * np.pi / wavel
    phi = np.exp(c0 * (np.outer(u, dra) + np.outer(v, ddec)))
    return phi


# =========================================================================
def grid_src_cvis(u, v, wavel, mgrid, gscale, phi=None):
    ''' Calc. complex vis measured by an array for a complex object
    ----------------------------------------------------------------
    - u,v:    baseline coordinates (meters)
    - wavel:  wavelength (meters)
    - mgrid:  square grid array describing the object
    - gscale: scalar defining the "mgrid" pitch size (in mas)
    - phi:    pre-computed auxilliary array to speed up calculation
    ---------------------------------------------------------------- '''

    if phi is None:
        phi = grid_precalc_aux_cvis(u, v, wavel, mgrid, gscale)

    return np.dot(phi, np.ravel(mgrid))


# =========================================================================
def phase_binary(u, v, wavel, p, deg=True):
    ''' Calculate the phases observed by an array on a binary star
    ----------------------------------------------------------------
    p: 3-component vector (+2 optional), the binary "parameters":
    - p[0] = sep (mas)
    - p[1] = PA (deg) E of N.
    - p[2] = contrast ratio (primary/secondary)

    optional:
    - p[3] = angular size of primary (mas)
    - p[4] = angular size of secondary (mas)

    - u,v: baseline coordinates (meters)
    - wavel: wavelength (meters)
    ---------------------------------------------------------------- '''
    cvis = cvis_binary(u, v, wavel, p)
    phase = np.angle(cvis, deg=deg)
    if deg:
        return np.mod(phase + 10980., 360.) - 180.0
    else:
        return phase


# =========================================================================
def vis2_binary(u, v, wavel, p):
    ''' Calc. squared vis. observed by an array on a binary star
    --------------------------------------------------------------
    p: 3-component vector (+2 optional), the binary "parameters":
      p[0] = sep (mas)
      p[1] = PA (deg) E of N.
      p[2] = contrast ratio (primary/secondary)

    optional:
      p[3] = angular size of primary (mas)
      p[4] = angular size of secondary (mas)

    u,v: baseline coordinates (meters)
    wavel: wavelength (meters)
    ---------------------------------------------------------------- '''

    cvis = cvis_binary(u, v, wavel, p, norm=True)
    return np.abs(cvis)**2


# =========================================================================
def uniform_disk(ys, xs, radius, between_pix=False):
    ''' Returns a centered 2D uniform disk array
    ----------------------------------------------
    Parameters:
    - (ys, xs)   : array size
    - radius     : radius of the disk
    - between_pix: (boolean) center between pixels
    ---------------------------------------------- '''
    mydist = _dist(ys, xs, between_pix=between_pix)
    ud = np.zeros_like(mydist)
    ud[mydist <= radius] = 1.0
    return ud


# =========================================================================
def super_gauss(ys, xs, radius, between_pix=False):
    ''' Returns a centered 2D super-Gaussian array
    ----------------------------------------------
    Parameters:
    - (xs, ys)   : array size
    - radius     : "radius" of the Super-Gaussian
    - between_pix: (boolean) center between pixels
    ---------------------------------------------- '''
    mydist = _dist(ys, xs, between_pix=between_pix)
    return np.exp(-(mydist/radius)**4)


# =========================================================================
def super_gauss0(xs, ys, x0, y0, w):
    ''' Returns an 2D super-Gaussian function
    ------------------------------------------
    Parameters:
    - (xs, ys) : array size
    - (x0, y0) : center of the Super-Gaussian
    - w        : width of the Super-Gaussian

    (Kept for reference for now)
    ------------------------------------------ '''

    x = np.outer(np.arange(xs), np.ones(ys))-x0
    y = np.outer(np.ones(xs), np.arange(ys))-y0
    dist = np.sqrt(x**2 + y**2)

    gg = np.exp(-(dist/w)**4)
    return gg


# =========================================================================
def centroid(image, threshold=0, binarize=False):
    ''' ------------------------------------------------------
    Determines the center of gravity of an array

    Parameters:
    ----------
    - image: the array
    - threshold: value above which pixels are taken into account
    - binarize: binarizes the image before centroid (boolean)

    Remarks:
    -------
    The binarize option can be useful for apertures, expected
    to be uniformly lit.
    ------------------------------------------------------ '''

    signal = np.where(image > threshold)
    sy, sx = image.shape[0], image.shape[1]

    temp = np.zeros((sy, sx))

    if binarize is True:
        temp[signal] = 1.0
    else:
        temp[signal] = image[signal]

    profx = 1.0 * temp.sum(axis=0)
    profy = 1.0 * temp.sum(axis=1)
    profx -= np.min(profx)
    profy -= np.min(profy)

    x0 = (profx*np.arange(sx)).sum() / profx.sum()
    y0 = (profy*np.arange(sy)).sum() / profy.sum()

    return (x0, y0)


# =========================================================================
def find_psf_center(img, verbose=True, nbit=10, visu=False, wmin=10.0):
    ''' Name of function self explanatory: locate the center of a PSF.

    ------------------------------------------------------------------
    Uses an iterative method with a window of shrinking size to
    minimize possible biases (non-uniform background, hot pixels, etc)

    Options:
    - nbit: number of iterations (default 10 is good for 512x512 imgs)
    - verbose: in case you are interested in the convergence
    ------------------------------------------------------------------ '''
    temp = img.copy()
    bckg = np.median(temp)    # background level
    temp -= bckg
    mfilt = medfilt(temp, 3)  # median filtered, kernel size = 3
    (sy, sx) = mfilt.shape    # size of "image"
    xc, yc = sx/2, sy/2       # first estimate for psf center

    signal = np.zeros_like(img)
    signal[mfilt > 0.1*mfilt.max()] = 1.0

    i0 = float(nbit-1) / np.log(sx/wmin)

    if visu:
        plt.figure()
        plt.ion()
        plt.show()

    for it in range(nbit):
        sz = np.round(sx/2 * np.exp(-it/i0))
        x0 = np.max([int(0.5 + xc - sz), 0])
        y0 = np.max([int(0.5 + yc - sz), 0])
        x1 = np.min([int(0.5 + xc + sz), sx])
        y1 = np.min([int(0.5 + yc + sz), sy])

        mask = np.zeros_like(img)
        mask[y0:y1, x0:x1] = 1.0

        if visu:
            plt.clf()
            plt.imshow((mfilt**0.2) * mask)
            plt.pause(0.1)

        profx = (mfilt*mask*signal).sum(axis=0)
        profy = (mfilt*mask*signal).sum(axis=1)

        xc = (profx*np.arange(sx)).sum() / profx.sum()
        yc = (profy*np.arange(sy)).sum() / profy.sum()

        if verbose:
            print("\rit #%2d center = (%.2f, %.2f)" % (it+1, xc, yc),
                  end="", flush=True)
    if verbose:
        print()

    return (xc, yc)


# =========================================================================
def find_fourier_origin(img, mykpo, m2pix, bmax=6.0):
    ''' ------------------------------------------------------------
    Finds the origin of the image that minimizes the amount of
    pointing-induced raw phase in the Fourier plane.

    Parameters:
    ----------
    - img   : the initial 2D array
    - mykpo : a KPO data structure
    - m2pix : 1 m in the pupil -> m2pix pixels in the image (float)
    - bmax  : the baseline beyond which Fourier phase is dropped (meters)

    Remarks:
    -------
    Developed and implemented at the request of J. Kammerer to
    better handle the wide and not so contrasted binary scenario.

    The *bmax* option is useful to filter out the phase information
    present at the longest baselines, which tends to be noisier.
    ------------------------------------------------------------ '''

    # Find image size
    sz = img.shape[0]
    if (sz != img.shape[1]):
        raise UserWarning('Requires square image')

    # Generate Fourier plane ramps in u and v direction
    uv = np.meshgrid((np.arange(sz//2+1)-sz//4)/float(sz),
                     (np.arange(sz)-sz//2)/float(sz))
    uv[0] = shift(uv[0])
    uv[1] = shift(uv[1])

    # Integer pixel re-centering
    img_filt = medfilt(img, 3)
    (yc_int, xc_int) = np.unravel_index(np.argmax(img_filt), img_filt.shape)
    img_cent = np.roll(np.roll(img, -yc_int, axis=0), -xc_int, axis=1)

    # Find Fourier plane coordinates which will be considered for the re-centering
    uv_dist = np.sqrt(mykpo.kpi.UVC[:, 0]**2+mykpo.kpi.UVC[:, 1]**2)
    uv_cutoff = np.where(uv_dist < float(r_cutoff))[0]

    # Find best sub-pixel shift
    img_fft = np.fft.rfft2(img_cent)
    best_xy_shift = leastsq(func=fourier_phase_resid_2d,
                            x0=np.array([0., 0.]),
                            args=(img_fft, m2pix, uv, uv_cutoff),
                            ftol=1E-1)[0]

    # Return best shift
    return [best_xy_shift[0]+xc_int, best_xy_shift[1]+yc_int]


# =========================================================================
def fourier_phase_resid_2d(xy, img_fft, mykpo, m2pix, uv, uv_cutoff):
    ''' ------------------------------------------------------------
    Cost function used by find_fourier_origin() defined above.

    Parameters:
    ----------
    - xy: tuple
    ------------------------------------------------------------ '''
    # Shift and inverse Fourier transform image
    img_shifted = img_fft*np.exp(i2pi*(xy[0]*uv[0]+xy[1]*uv[1]))
    img = np.abs(np.fft.fftshift(np.fft.irfft2(img_shifted)))

    # Extract Fourier plane phase
    cvis = mykpo.extract_cvis_from_img(img, m2pix, method='LDFT1')

    # Return Fourier plane phase
    return np.abs(np.angle(cvis[uv_cutoff]))


# =========================================================================
def determine_origin(img, mask=None, algo="BCEN", verbose=True, wmin=10.0):
    ''' ------------------------------------------------------------
    Determines the origin of the image, using among possible algorithms.

    Parameters:
    ----------
    - img: the initial 2D array
    - mask: an optional mask, same size as img (default = None)
    - algo: a string describing the algorithm (default = "BCEN")
      + "BCEN": centroid of the brightest speckle     (default)
      + "COGI": center of gravity of image
    - verbose: display some additional info (boolean, default=True)
    - wmin: size of the last centering window (in pixels)
    ------------------------------------------------------------ '''
    if algo.__class__ is not str:
        print("")
        algo = "BCEN"

    if mask is not None:
        img1 = img * mask
    else:
        img1 = img * 1.0

    if "cog" in algo.lower():
        (x0, y0) = centroid(img1, verbose)

    else:
        (x0, y0) = find_psf_center(img1, verbose, nbit=10, wmin=wmin)

    return (x0, y0)


# =========================================================================
def recenter(im0, mask=None, algo="BCEN", subpix=True, between=False,
             verbose=True):
    ''' ------------------------------------------------------------
    Re-centering algorithm of a 2D image im0 for kernel-analysis

    Parameters:
    ----------
    - im0: the initial 2D array
    - mask: an optional mask same size as im0 (default = None)
    - algo: centering algorithm (default = "BCEN")
      + "BCEN": centroid of the brightest speckle     (default)
      + "FPNM": Fourier-phase norm minimization       (Jens)
      + "COGI": center of gravity of image
    - subpix: sub-pixel recentering         (boolean, default=True)
    - between: center in between 4 pixels   (boolean, default=False)
    - verbose: display some additional info (boolean, default=True)

    Remarks:
    -------
    - The optional mask is *not applied* to the final image.
    - "between=True" effective only if "subpix=True"
    ------------------------------------------------------------ '''

    ysz, xsz = im0.shape

    (x0, y0) = determine_origin(im0, mask=mask, algo=algo, verbose=verbose)

    dy, dx = (y0-ysz/2), (x0-xsz/2)
    if between:
        dy += 0.5
        dx += 0.5

    if verbose:
        print("centroid: dx=%+5.2f, dy=%+5.2f\n" % (dx, dy),
              end="", flush=True)

    # integer pixel recentering first
    im0 = np.roll(np.roll(im0, -int(round(dx)), axis=1),
                  -int(round(dy)), axis=0)

    if verbose:
        print("recenter: dx=%+5d, dy=%+5d\n" % (-round(dx), -round(dy)),
              end="", flush=True)

    # optional FFT-based subpixel recentering step
    # requires insertion into a zero-padded square array (dim. power of two)
    if subpix:
        temp = np.max(im0.shape)  # max dimension of image

        for sz in 32 * 2**np.arange(6):
            if sz >= temp:
                break
        dz = sz/2.           # image half-size

        xx, yy = np.meshgrid(np.arange(sz)-dz, np.arange(sz)-dz)
        wx, wy = xx*np.pi/dz, yy*np.pi/dz

        dx -= np.round(dx)
        dy -= np.round(dy)

        if verbose:
            print("recenter: dx=%+5.2f, dy=%+5.2f\n" % (-dx, -dy),
                  end="", flush=True)
        # insert image in zero-padded array (dim. power of two)
        im = np.zeros((sz, sz))
        orix, oriy = (sz-xsz)//2, (sz-ysz)//2
        im[oriy:oriy+ysz, orix:orix+xsz] = im0

        slope = shift(dx * wx + dy * wy)
        offset = np.exp(1j*slope)
        dummy = np.real(shift(ifft(offset * fft(shift(im)))))
        im0 = dummy[oriy:oriy+ysz, orix:orix+xsz]
    return im0


# =========================================================================
def compute_DFTM2(coords, m2pix, isz, axis=0):
    ''' -----------------------------------------------------------------------
    Two-sided DFT matrix to be used with the "LDFT2" extraction method,
    DFT matrix computed for exact u (or v) coordinates.

    Based on a LL.dot(img).RR approach.

    parameters:
    ----------
    - coords : a 1D vector of baseline coordinates where to compute the FT
    - m2pix  : a scaling parameter, that depends on the wavelength, the plate
               scale and the image size
    - isz    : the image size
    - axis   : == 0 (default) produces a matrix that acts on image rows
               != 0 (anything) produces a matrix that acts on its columns

    -----------------------------------

    Example of use, for an image of size isz:

    >> LL = xara.core.compute_DFTM2(np.unique(kpi.uv[:,1]), m2pix, isz, 0)
    >> RR = xara.core.compute_DFTM2(np.unique(kpi.uv[:,0]), m2pix, isz, 1)

    >> FT = LL.dot(img).dot(RR)

    This last command returns the properly sampled 2D FT of the img.
    ----------------------------------------------------------------------- '''

    i2pi = 1j * 2 * np.pi

    bl_c = coords * m2pix
    w_v = np.exp(-i2pi/isz * bl_c, dtype=np.complex128)  # roots of DFT matrix
    ftm = np.zeros((w_v.size, isz), dtype=w_v.dtype)

    for i in range(isz):
        ftm[:, i] = w_v**(i - isz/2) / np.sqrt(isz)
    if axis != 0:
        return(ftm.T)
    else:
        return(ftm)


# =========================================================================
def compute_DFTM1(coords, m2pix, isz, inv=False, dprec=True):
    ''' ------------------------------------------------------------------
    Single-sided DFT matrix to be used with the "LDFT1" extraction method,
    DFT matrix computed for exact u (or v) coordinates.

    Based on a FF.dot(img) approach.

    parameters:
    ----------
    - coords : vector of baseline (u,v) coordinates where to compute the FT
    - m2pix  : a scaling parameter, that depends on the wavelength, the
               plate scale and the image size
    - isz    : the image size

    Option:
    ------
    - inv    : Boolean (default=False) : True -> computes inverse DFT matrix
    - dprec  : double precision (default=True)

    For an image of size (SZ x SZ), the computation requires what can be a
    fairly large (N_UV x SZ^2) auxilliary matrix.
    -----------------------------------

    Example of use, for an image of size isz:

    >> FF = xara.core.compute_DFTM1(np.unique(kpi.UVC), m2pix, isz)

    >> FT = FF.dot(img.flatten())

    This last command returns a 1D vector FT of the img.
    ------------------------------------------------------------------ '''

    i2pi = 1j * 2 * np.pi

    mydtype = np.complex64

    if dprec is True:
        mydtype = np.complex128

    xx, yy = np.meshgrid(np.arange(isz)-isz/2, np.arange(isz)-isz/2)
    uvc = coords * m2pix
    nuv = uvc.shape[0]

    if inv is True:
        WW = np.zeros((isz**2, nuv), dtype=mydtype)
        for i in range(nuv):
            WW[:, i] = np.exp(i2pi*(uvc[i, 0] * xx.flatten() +
                                    uvc[i, 1] * yy.flatten())/float(isz))
    else:
        WW = np.zeros((nuv, isz**2), dtype=mydtype)

        for i in range(nuv):
            WW[i] = np.exp(-i2pi*(uvc[i, 0] * xx.flatten() +
                                  uvc[i, 1] * yy.flatten())/float(isz))
    return(WW)


# =========================================================================
def uv_phase_regrid_matrix(UVD, UVS, rad):
    '''------------------------------------------------------------------
    !!EXPERIMENTAL FUNCTION!!

    Not sure how useful this will eventually turn however numerical experiments
    have led me to write this in attempting to diagnostic systematic errors so
    I include it here.

    Matrix to extract a smaller number of kernels from a dense model

    Provided with two sets of UV coordinates (one dense and one sparse), builds
    a matrix that regrids use the dense model to interpolate the sparse one.

    Parameters:
    ----------
    - UVD: dense  UV-coordinates array (np.array([nbuv, 2]) in meters
    - UVS: sparse UV-coordinates array (np.array([nbuv, 2]) in meters
    - rad: interpolation radius (in meters)

    Example:
    -------
    - UVS: sparse (36x2) UV coverage of a 9H mask (one sample per sub-aperture)
    - UVD: dense (1350x2) UV coverage of the same mask (12 samples per sub-apt)
    - rad: 0.42 (slightly larger than the hole diameter)

    Returns a 36x1350 matrix that averages the phase measured in the F plane
    with the dense model, so as to produce an estimate for the sparse one.
    ------------------------------------------------------------------ '''

    uvc1 = UVS.copy()
    uvc2 = UVD.copy()

    nuv1 = UVS.shape[0]
    nuv2 = UVD.shape[0]

    uvc2 = np.concatenate([uvc2, -uvc2], axis=0)  # to avoid edge effect

    GG = np.zeros((nuv1, nuv2))
    for ii in range(nuv1):
        uu, vv = uvc1[ii]
        aa = np.where((np.abs(uvc2[:, 1] - vv) <= rad) *
                      (np.abs(uvc2[:, 0] - uu) <= rad))

        for jj in range(len(aa[0])):
            index = aa[0][jj]
            if index >= nuv2:
                index -= nuv2
                GG[ii, index] = -1.0  # neg phase at uv opposite!
            else:
                GG[ii, index] = 1.0

    WW = np.abs(GG).sum(axis=1)
    GG = np.diag(1/WW).dot(GG)
    return GG


# =========================================================================
def create_discrete_model(apert, ppscale, step, binary=True, tmin=0.8):
    '''------------------------------------------------------------------

    Create the discrete (square grid) model of a provited aperture later
    used to build a kernel model.

    Parameters:
    ----------
    - apert:   square array describing the aperture of an instrument
    - ppscale: pupil pixel scale            (in meters)
    - step:    the discrete model grid size (in meters)
    - binary:  binary or grey model         (boolean, default=True)
    - tmin:    cut-off transmission         (0<tmin<1, default=80%)

    Remark #1: As pointed out by Alban Ceau:
    ---------

    Regarding the choice of "step", it is advisable to ensure that once
    projected in the pixel space of the "apert" array, the step does
    correspond to an *even and integer number of pixels*. This prevents
    subtle edge effects that may result in discrete models that do not
    fully reflect the symmetry properties of their original counterpart.

    Using the pupil functions from the xaosim.pupil module to generate
    the 2D apert array, make sure to use the between_pix flag to True
    so that the array is indeed strictly symmetric.

    Ex of reasonable use for the Subaru Telescope (diameter: 7.92 m)
    >> PSZ = 792 # array size chosen for a nice 1 cm / pixel scale!
    >> pup = xaosim.pupil.subaru((PSZ,PSZ), PSZ/2, True, True)
    >> pscale = 7.92 / PSZ
    >> model = xara.core.create_discrete_model(pup, pscale, 0.16, True)

    To verify that your model reflects the symmetry properties of the
    original aperture, look at the following superimposed plots:

    >> plt.plot( model[:,0],  model[:,1], 'bo')
    >> plt.plot(-model[:,0],  model[:,1], 'r.')

    and:

    >> plt.plot( model[:,0],  model[:,1], 'bo')
    >> plt.plot(-model[:,0], -model[:,1], 'g.')

    Any non perfectly overlapping point reveals a problem in the model,
    that requires some tweaking (including possibly manual editing).

    To fix such models, you can give the symetrizes_model() function
    defined below a shot.

    Remark #2:
    ---------

    The minimum transmission value "tmin" is used to decide where to
    chop parts of the aperture that don't make it inside the
    model. Depending on the case (binary or grey model), the threshold
    has to be adjusted.

    ------------------------------------------------------------------
    '''

    PSZ = apert.shape[0]
    nbs = int(PSZ / (step / ppscale))  # number of sample points across

    if not (nbs % 2):
        nbs += 1  # ensure odd number of samples (align with center!)

    # ============================
    #   pad the pupil array
    # ============================

    PW = int(step / ppscale)                # padding width
    padap = np.zeros((PSZ+2*PW, PSZ+2*PW))  # padded array
    padap[PW:PW+PSZ, PW:PW+PSZ] = apert
    DSZ = PSZ/2 + PW

    # ============================
    #  re-grid the pupil -> pmask
    # ============================

    pos = step * (np.arange(nbs) - nbs//2)
    xgrid, ygrid = np.meshgrid(pos, pos)
    pmask = np.zeros_like(xgrid)

    xpos = (xgrid / ppscale + DSZ).astype(int)
    ypos = (ygrid / ppscale + DSZ).astype(int)

    for jj in range(nbs):
        for ii in range(nbs):
            x0 = int(xpos[jj, ii])-PW//2
            y0 = int(ypos[jj, ii])-PW//2
            pmask[jj, ii] = padap[y0:y0+PW, x0:x0+PW].mean()

    # ==========================
    #  build the discrete model
    # ==========================

    xx, yy, tt = [], [], []

    if binary is True:
        for jj in range(nbs):
            for ii in range(nbs):
                if (pmask[jj, ii] > tmin):
                    pmask[jj, ii] = 1.0
                    xx.append(xgrid[jj, ii])
                    yy.append(ygrid[jj, ii])
                    tt.append(1.0)
                else:
                    pmask[jj, ii] = 0.0

    else:
        for jj in range(nbs):
            for ii in range(nbs):
                if (pmask[jj, ii] > tmin):
                    xx.append(xgrid[jj, ii])
                    yy.append(ygrid[jj, ii])
                    tt.append(pmask[jj, ii])

    xx = np.array(xx)
    yy = np.array(yy)
    tt = np.array(tt)

    model = np.array([xx, yy, tt]).T
    return model


# =========================================================================
def symetrizes_model(model, axis=0, cut=1e-1):
    ''' -------------------------------------------------------------------
    Returns a symetrized version of the original pupil model.

    This is what you use to automatically fix a model coming out of the
    create_discrete_model() function.

    Parameters:
    ----------
    - model: a 3-column (x,y,t) discrete pupil model
    - axis: the direction along which the model is to be symmetrized
      + 0 => horizontal
      + 1 => vertical
    - cut: distance criterion, should be < than the grid step

    Remark:
    ------

    When attempting to "fix" the grid model, one could add or remove
    sub-apertures. Here we only remove sub-apertures!
    ------------------------------------------------------------------- '''

    flag = []
    nap = model.shape[0]

    if axis > 1:
        axis = 0
        print("warning: axis can only be 0 or 1 for a 2D array!")
        print("assuming L/R symmetry (axis=0)")

    if axis == 0:
        ss, ns = 0, 1
    else:
        ss, ns = 1, 0
    for ii in range(nap):
        tmp1 = np.abs(model[:, ss] + model[ii, ss]) < cut
        tmp2 = np.abs(model[:, ns] - model[ii, ns]) < cut
        test = (tmp1 * tmp2).sum()
        if test < 1.0:
            print("problematic sub-aperture #%d flagged!" % (ii,))
            flag.append(ii)

    return np.delete(model, flag, axis=0)