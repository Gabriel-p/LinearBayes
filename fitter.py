
import sys
from astropy.io import ascii
import numpy as np
from scipy.optimize import differential_evolution as DE
import emcee
import matplotlib.pyplot as plt
import warnings


def main():
    """
    """

    file_name = "rgb6.poor.txt"
    xy_cols = "RA(deg)", "Dec(deg)"

    # Read data
    x, ex, y, ey = readData(file_name, xy_cols)
    # x, ex, y, ey = synthData()

    xy_data = np.array([x, y, ex, ey]).T

    # Some priors: slope, intercept, intrinsic scatter, outlier mean,
    # outlier scatter (standard deviation of outliers), outlier fraction
    priors = [0, 400, -400., 0., 0.001, 100., -10., 10., 0.001, 1000.]
    # priors = [-10., 10., -10., 10., 0.001, 100., -10., 10., 0.001, 1000.]

    # Make the fit
    samples, point_estim = fit_data(xy_data, priors)

    makePlots(x, ex, y, ey, samples, point_estim)


def readData(file_name, xy_cols):
    """
    """

    data = ascii.read(file_name)
    x, y = data[xy_cols[0]], data[xy_cols[1]]
    N = len(x)

    ex, ey = np.zeros(N), np.zeros(N)

    return x, ex, y, ey


def synthData():
    """
    """
    m, b = np.random.uniform(-10., 10., 2)
    print("True values: m={:.3f}, b={:.3f}".format(m, b))

    # Generate some synthetic data from the model.
    N = np.random.randint(10, 500)
    x = np.sort(10 * np.random.rand(N))
    ex = 0.1 + 0.5 * np.random.rand(N)

    y = m * x + b
    ey = 0.2 * (y.max() - y.min()) * np.random.rand(N)
    y += ey * np.random.randn(N)
    ey = ey * .5

    return x, ex, y, ey


def fit_data(
    data,
        priorlimits=[-10., 10., -10., 10., 0.001, 100., -10., 10., 0.001,
                     1000.], nwalkers=20, nsteps=5000, burn_frac=.25):
    """
    This code will fit a straight line with intrinsic dispersion to data with
    (optionally) covariant errors on both the independent and dependent
    variables, and takes care of outliers using a mixture model approach.

    The free parameters in the model are:

    * slope: slope of the fitted line.

    * intercept: intercept of the fitted line.

    * intrinsic scatter ('sigma_intrinsic'): Hogg, Bovy, Lang (2010):
      "intrinsic Gaussian variance V, orthogonal to the line."

    * outlier mean ('y_outlier'): mean of outliers.

    * outlier scatter ('sigma_outlier'): standard deviation of outliers.

    * outlier fraction ('outlier_fraction'): fraction of ouliers in data.
      Hogg, Bovy, Lang (2010): "the probability that a data point is bad (or,
      more properly, the amplitude of the bad-data distribution function in the
      mixture)."


    Parameters
    ----------

    data : np.ndarray
        Should have shape (N,4) (if no covariances on errors) or (N,5) (if
        covariant errors). Should be in the order (x, y, dx, dy) or
        (x, y, dx, dy, dxy).

    priorlimits : np.ndarray
        Upper and lower values for each of the model parameters (except the
        outlier fraction which has a flat prior between 0 and 1). The limits
        should be provided in the order:
        [slope, intercept, intrinsic scatter, outlier mean, outlier scatter]
        (so that the array has 10 elements).

    nwalkers : int
        The number of emcee walkers to use in the fit.

    nsteps : int
        The number of steps each walker should take in the MCMC.

    burn_frac : 0 < float < 1
        The fraction of initial emcee walkers to discard as burn-in.

    Returns
    -------

    samples : np.array
        All the chains (flattened) minus the burn-in period.

    point_estim : np.array
        (16th, 50th, 84th) percentile for each of the 6 fitted parameters in
        the order: (slope, intercept, intrinsic scatter, outlier mean,
        outlier deviation, outlier fraction).

    """
    # Unpack and check data.
    if data.shape[1] == 4:
        # No correlations on the errors
        x, y, dx, dy = data.T
        dxy = np.zeros_like(x)
    elif data.shape[1] == 5:
        # Data with dxy correlations
        x, y, dx, dy, dxy = data.T
    else:
        raise ValueError("'data' must have 4 or 5 columns, not {}. \
                Try transposing your data.".format(data.shape[1]))

    # Supress RuntimeWarning
    warnings.filterwarnings("ignore")

    # The number of dimensions is fixed.
    ndim = 6
    # Add outlier fraction prior limits, also fixed.
    priorlimits = priorlimits + [0., 1.]

    print("Running optimization...")

    # Estimate initial values using DE algorithm.
    def minfunc(model):
        return -full_posterior(model, x, y, dx, dy, dxy, priorlimits)
    bmin, bmax = priorlimits[0::2], priorlimits[1::2]
    bounds = list(zip(*[bmin, bmax]))
    pstart = DE(minfunc, bounds, maxiter=5000).x
    print("Initial guesses: "
          "({:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f})".format(*pstart))

    # Sample ball around the max posterior point.
    p0 = emcee.utils.sample_ball(
        pstart, 0.01 * np.ones_like(pstart), size=nwalkers)
    # Make sure there are no negative outlier fractions
    p0[:, -1] = np.abs(p0[:, -1])

    print("Running emcee...")
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, full_posterior, args=[x, y, dx, dy, dxy, priorlimits])
    for i, result in enumerate(sampler.sample(p0, iterations=nsteps)):
        updt(nsteps, i)

    # Remove burn-in and flatten all chains.
    nburn = int(burn_frac * nsteps)
    samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))

    # Shape: (6, 3)
    point_estim = np.percentile(samples, [16, 50, 84], axis=0).T

    return samples, point_estim


def full_posterior(params, x, y, dx, dy, dxy, priorlimits):
    """
    The log-posterior of the data given the full mixture model of the linear
    function and the outlier distribution.

    Parameters
    ----------

    params : np.ndarray or list
        [slope,intercept, intrinsic scatter, outlier mean,
            outlier standard deviation, outlier fraction]

    Returns
    -------

    float
        The posterior of the parameters given the data.
    """
    if log_priors(params, priorlimits) == -np.inf:
        return -np.inf
    else:
        return log_priors(params, priorlimits) +\
            full_log_likelihood(params, x, y, dx, dy, dxy)


def log_priors(params, priorlimits):
    """
    Prior probabilities on the parameters, given upper and lower limits on each
    parameter. Jeffreys priors are used for the intrinsic and outlier standard
    deviations, and a prior that is flat in Arctan(slope) is used for the
    slope. For everything else, priors are uniform within the given limits.

    Parameters
    ----------

    params : np.ndarray or list
        [slope, intercept, intrinsic scatter, outlier mean, outlier standard
         deviation, outlier fraction]

    Returns
    -------

    float
        The prior density of these parameters.
    """
    m, b, sigma_intrinsic, y_outlier, sigma_outlier, outlier_fraction = params
    mlo, mhi, blo, bhi, silo, sihi, yolo, yohi, solo, sohi, oflo, ofhi =\
        priorlimits

    if m < mlo or m > mhi or b < blo or b > bhi or sigma_intrinsic < silo or\
        sigma_intrinsic > sihi or sigma_outlier < solo or sigma_outlier > sohi\
        or y_outlier < yolo or y_outlier > yohi or outlier_fraction < oflo or\
            outlier_fraction > ofhi:
        return -np.inf
    else:
        return -np.log(1. + m * m) - np.log(sigma_intrinsic) -\
            np.log(sigma_outlier)


def full_log_likelihood(params, x, y, dx, dy, dxy):
    """
    The log-likelihood of the data given the full mixture model of the linear
    function and the outlier distribution.

    This is basically E1. (17) in Hogg, Bovy, Lang (2010), accounting for the
    intrinsic scatter term in 'likelihood_line()'.

    Returns
    -------

    float
        The likelihood of the data given this set of model parameters.
    """
    m, b, sigma_intrinsic, y_outlier, sigma_outlier, outlier_fraction = params

    lkl_line = likelihood_line([m, b, sigma_intrinsic], x, y, dx, dy, dxy)
    out_dist = outlier_distribution(
        [y_outlier, sigma_outlier], x, y, dx, dy, dxy)

    return np.sum(np.log(
        (1. - outlier_fraction) * lkl_line + outlier_fraction * out_dist))


def likelihood_line(params, x, y, dx, dy, dxy):
    """
    Likelihood for the linear function.

    Returns
    -------

    float
        The likelihood of the data given this set of model parameters.
    """
    m, b, sigma_intrinsic = params
    theta = np.arctan(m)

    sint, cost = np.sin(theta), np.cos(theta)

    # Perpendicular distance to the line
    delta = -sint * x + cost * y - cost * b

    # Projection of covariance matrix along line
    Sigma_dd = sint**2. * dx**2. - np.sin(2. * theta) * dxy + cost**2. * dy**2.

    lkl_line = (2. * np.pi * (Sigma_dd + sigma_intrinsic**2.))**-.5 *\
        np.exp(-delta**2. / (2. * (Sigma_dd + sigma_intrinsic**2.)))

    return lkl_line


def outlier_distribution(params, x, y, dx, dy, dxy):
    """
    The likelihood for the outlier distribution, which is modeled as a uniform
    distribution in x and a Gaussian distribution in y with some mean y0 and
    standard deviation sigma0.

    Returns
    -------

    float
        The likelihood of the data given this set of model parameters.
    """
    y_outlier, sigma_outlier = params
    sigma_total2 = sigma_outlier**2. + dy**2.

    out_dist = (2. * np.pi * sigma_total2)**-0.5 *\
        np.exp(-.5 * (y - y_outlier)**2. / sigma_total2)

    return out_dist


def updt(total, progress, extra=""):
    """
    Displays or updates a console progress bar.

    Original source: https://stackoverflow.com/a/15860757/1391441
    """
    barLength, status = 20, ""
    progress = float(progress) / float(total)
    if progress >= 1.:
        progress, status = 1, "\r\n"
    block = int(round(barLength * progress))
    text = "\r[{}] {:.0f}% {}{}".format(
        "#" * block + "-" * (barLength - block),
        round(progress * 100, 0), extra, status)
    sys.stdout.write(text)
    sys.stdout.flush()


def makePlots(x, ex, y, ey, samples, point_estim, m=None, b=None):
    """
    """
    try:
        import corner
        fig = plt.figure(figsize=(10, 10))
        corner.corner(samples)
        fig.tight_layout()
        plt.savefig("corner.png", dpi=150)
    except ModuleNotFoundError:
        print("No corner module")

    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    ax.minorticks_on()
    ax.grid(b=True, which='both', color='gray', linestyle='--', lw=.5)

    if m is not None:
        txt = r"$True:\;m={:.3f},\,b={:.3f}$".format(m, b)
    else:
        txt = ''
    plt.errorbar(x, y, xerr=ex, yerr=ey, fmt='o', label=txt)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    mlo, mmed, mhi = point_estim[0]
    # mlo, mhi = mmed - mlo, mhi - mmed
    blo, bmed, bhi = point_estim[1]
    # blo, bhi = bmed - blo, bhi - bmed
    txt = r"$Estim:\;m={:.3f}_{{{:.3f}}}^{{{:.3f}}}$".format(mmed, mlo, mhi)
    txt += r"$,\;b={:.3f}_{{{:.3f}}}^{{{:.3f}}}$".format(bmed, blo, bhi)
    print('\n' + txt)

    x0 = np.linspace(min(x), max(x), 10)
    plt.plot(x0, np.poly1d((mmed, bmed))(x0), label=txt, zorder=4)

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    plt.legend()
    fig.tight_layout()
    plt.savefig("final_fit.png", dpi=150)


if __name__ == '__main__':
    main()
