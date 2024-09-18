import numpy
import matplotlib.pyplot as plt
from scipy.stats import exponnorm
from scipy.optimize import basinhopping


def add_constant(data, prepend=True, has_constant="skip"):
    # Adapted from statsmodels add_constant
    x = numpy.asarray(data)
    ndim = x.ndim
    if ndim == 1:
        x = x[:, None]
    elif x.ndim > 2:
        raise ValueError("Only implemented for 2-dimensional arrays")

    is_nonzero_const = numpy.ptp(x, axis=0) == 0
    is_nonzero_const &= numpy.all(x != 0.0, axis=0)
    if is_nonzero_const.any():
        if has_constant == "skip":
            return x
        elif has_constant == "raise":
            if ndim == 1:
                raise ValueError("data is constant.")
            else:
                columns = np.arange(x.shape[1])
                cols = ",".join([str(c) for c in columns[is_nonzero_const]])
                raise ValueError(f"Column(s) {cols} are constant.")

    x = [numpy.ones(x.shape[0]), x]
    x = x if prepend else x[::-1]
    return numpy.column_stack(x)


def compute_emg_regression_linear_expo_mean(x, y):
    beta = [0, 0.1]
    sigma = 0.01
    expo_mean = [0.5, 0.5]
    bounds = [(-1, 1), (0, 1), (1e-5, 1), (0, 1), (0, 5)]
    attempts = 0
    while attempts <= 5:
        fit = fit_emg_arbitrary_variance_model(
            beta, sigma, expo_mean, linear_expo_mean, x, y, bounds=bounds
        )
        if fit.success:
            return fit.x, fit
        attempts += 1
    raise RuntimeError("fit not successful")


def minus_ll_emg_arbitrary_variance_model(params, x, y, f_expo_mean):
    beta_params = params[:2]
    sigma_params = params[2]
    expo_mean_params = params[3:]

    ll = 0

    expo_mean = f_expo_mean(x, expo_mean_params)
    scale = sigma_params
    loc = beta_params[0] + beta_params[1] * x
    K = expo_mean / (scale)
    ll = exponnorm.logpdf(y, K, loc=loc, scale=scale)
    return -numpy.sum(ll)


def fit_emg_arbitrary_variance_model(beta, sigma, expo_mean, f_expo_mean, x, y, bounds):
    x = numpy.asarray(x)
    y = numpy.asarray(y)
    params = [*beta, sigma, *expo_mean]
    minimizer_kwargs = dict(method="L-BFGS-B", args=(x, y, f_expo_mean), bounds=bounds)
    # return minimize(minus_ll_emg_arbitrary_variance_model, params, **minimizer_kwargs)
    return basinhopping(
        minus_ll_emg_arbitrary_variance_model,
        params,
        minimizer_kwargs=minimizer_kwargs,
        niter=10,
    )


##### conditional mean models ============
def linear_expo_mean(x, params):
    return params[0] + params[1] * x


def constant_expo_mean(x, params):
    return params[0]


#####  ============


# def gen_emg_arbitrary_expo_mean(params, expo_mean_function, N=100, seed=None):
#     beta_params = params[:2]
#     sigma_params = params[2]
#     expo_mean_params = params[3:]
#     rng = numpy.random.default_rng()
#     x = rng.random(N) * 7
#     y = numpy.empty((N,))
#     for nx, _x in enumerate(x):
#         loc = beta_params[0] + beta_params[1] * _x
#         scale = sigma_params
#         expo_mean = expo_mean_function(_x, expo_mean_params)
#         rate = 1 / expo_mean
#         K = 1 / (rate * scale)
#         y[nx] = exponnorm.rvs(K, loc=loc, scale=scale, random_state=seed + nx)
#     return x, y


def gen_emg_arbitrary_expo_mean_vec(params, expo_mean_function, N=100, seed=None):
    beta_params = params[:2]
    sigma_params = params[2]
    expo_mean_params = params[3:]
    rng = numpy.random.default_rng(seed=seed)
    x = rng.random(N) * 7
    y = numpy.empty((N,))
    loc = beta_params[0] + beta_params[1] * x
    scale = sigma_params
    expo_mean = expo_mean_function(x, expo_mean_params)
    K = expo_mean / scale
    y = exponnorm.rvs(K, loc=loc, scale=scale, random_state=seed)
    return x, y


if __name__ == "__main__":

    fig, axs = plt.subplots(1, 3)
    for n, ax in enumerate(axs):
        beta = [1, 0.11]
        sigma = 0.01
        expo_mean = [1, n]
        params = [*beta, sigma, *expo_mean]
        x, y = gen_emg_arbitrary_expo_mean_vec(
            params, linear_expo_mean, N=1000, seed=1234
        )

        X = add_constant(x)

        beta = [0, 0.1]
        sigma = 0.1
        expo_mean = [0, 1]
        bounds = [(-1, 1), (0, 1), (1e-5, 1), (0, 1), (0, 5)]
        fit = fit_emg_arbitrary_variance_model(
            beta, sigma, expo_mean, linear_expo_mean, x, y, bounds=bounds
        )
        print(fit.x)

        ax.plot(x, y, "o")
        ax.plot([1, 7], [1 + 1 * 0.1, 1 + 7 * 0.1], "r-")

        plt.ion()
        plt.show()
