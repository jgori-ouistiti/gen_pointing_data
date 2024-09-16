import scipy.stats as stats
import numpy
import polars
import matplotlib.pyplot as plt
import seaborn

import rpy2.robjects as robjects
from rpy2.robjects.vectors import FloatVector
from rpy2.rlike.container import TaggedList

from emg_arbitrary_variance import compute_emg_regression_linear_expo_mean

robjects.r["source"]("gen_t_copula.R")
gen_copula_fun = robjects.globalenv["gen_block_t"]
robjects.r["source"]("fit_tcopula.R")
fit_copula_fun = robjects.globalenv["fit_t_copula"]


def python_params_to_named_list_R(obj):
    params = obj["params"]
    params_list = TaggedList(list(params.values()), tags=list(params.keys()))
    obj_list = TaggedList([obj["distribution"], params_list], tags=list(obj.keys()))
    return obj_list


def fit_t_copula(ide, mt):
    return fit_copula_fun(FloatVector(ide), FloatVector(mt))


def correct_beta_lambda(ide, mt, beta, lambda_emg):
    lambda_corrected = (
        numpy.array([lambda_emg[0] for i in ide]),
        numpy.maximum(
            0,
            (mt - beta[0] - beta[1] * ide - lambda_emg[0]) / ide,
        ),
    )
    beta_corrected = beta
    return beta_corrected, lambda_corrected


def gen_emg(
    beta, sigma, lambda_emg, block_levels=None, ntrials=50, rng=None, seed=None
):
    ide = block_levels
    if ide is None:
        ide = numpy.linspace(0.1, 0.9, 10)
    elif isinstance(ide, int):
        ide = list(rng.random(ide) * 8)
    else:
        pass

    X = numpy.full((ntrials, len(ide)), fill_value=ide)
    loc = beta[0] + beta[1] * X
    scale = sigma
    expo_mean = lambda_emg[0] + lambda_emg[1] * X
    X = X.ravel()
    loc = loc.ravel()
    expo_mean = expo_mean.ravel()
    K = expo_mean / scale
    y = stats.exponnorm(K, loc=loc, scale=scale).rvs(random_state=seed)
    return X, y


def gen_emg_control(
    beta,
    sigma,
    lambda_emg,
    mvg_mu,
    mvg_cov,
    block_levels=None,
    ntrials=50,
    rng=None,
    seed=None,
):
    if rng is None:
        rng = numpy.random.default_rng(seed=seed)

    rng = rng.spawn(1)[0]
    if seed is None:
        seed = int(numpy.floor(rng.random(1)[0] * 999))

    ide = block_levels

    if ide is not None:
        if isinstance(ide, int):
            ide = list(rng.random(ide) * 8)
        else:
            ide = numpy.asarray(ide)

        mu_mt = mvg_mu[1] + mvg_cov[0, 1] / mvg_cov[0, 0] * (ide - mvg_mu[0])
        var = mvg_cov[1, 1] - mvg_cov[0, 1] ** 2 / mvg_cov[0, 0]
        y = stats.norm(loc=mu_mt, scale=var).rvs()
        x = numpy.asarray(ide)
    else:
        x, y = stats.multivariate_normal(mean=mvg_mu, cov=mvg_cov).rvs()
        x = [x]
    beta, lambda_emg = correct_beta_lambda(x, y, beta, lambda_emg)

    return gen_emg(beta, sigma, lambda_emg, block_levels=x, ntrials=ntrials, seed=seed)


def gen_t_copula(
    rho1,
    df,
    id_params,
    mt_params,  # for the marginals, pass things that make sense to R
    trials=15,
    block_levels=None,
    cdf_block=False,
    rng=None,
    seed=None,
):

    if rng is None:
        rng = numpy.random.default_rng(seed=seed)

    rng = rng.spawn(1)[0]
    if seed is None:
        seed = int(numpy.floor(rng.random(1)[0] * 999))

    if block_levels is None:
        block_levels = FloatVector(list(numpy.linspace(0.1, 0.9, 10)))
    elif isinstance(block_levels, int):
        block_levels = FloatVector(list(rng.random(block_levels)))
    else:
        if cdf_block:
            if id_params["distribution"] == "gamma":
                cdf = [
                    getattr(stats, id_params["distribution"]).cdf(
                        b,
                        id_params["params"]["shape"],
                        scale=(1 / id_params["params"]["rate"]),
                    )
                    for b in block_levels
                ]
            else:
                raise NotImplementedError
        else:
            cdf = block_levels
        block_levels = FloatVector(cdf)

    _array = numpy.array(
        gen_copula_fun(
            float(rho1),
            float(df),
            python_params_to_named_list_R(id_params),
            python_params_to_named_list_R(mt_params),
            FloatVector(block_levels),
            int(trials),
            int(seed),
        )
    )
    return _array[:, 0], _array[:, 1]


if __name__ == "__main__":

    # === loading some data
    df = polars.read_csv("../data/example_data.csv")

    ############### generate EMG data

    # -- infer emg model
    # off the shelf fitter should work. If not, refer to fit_emg_arbitrary_variance_model
    x, fit = compute_emg_regression_linear_expo_mean(
        numpy.asarray(df["IDe(2d)"]), numpy.asarray(df["Duration"])
    )
    beta = x[:2]
    sigma = x[2]
    lambda_emg = x[3:]
    # -- end infer model

    # without specifying ide levels
    emg_x, emg_y = gen_emg(
        beta, sigma, lambda_emg, block_levels=None, ntrials=50, rng=None, seed=None
    )

    # with ide levels specified:
    ide_levels = df["IDe(2d)"].unique()
    emg_x, emg_y = gen_emg(
        beta,
        sigma,
        lambda_emg,
        block_levels=numpy.asarray(ide_levels),
        ntrials=50,
        rng=None,
        seed=None,
    )

    ############### generate EMG data with r(mean(MT), IDe) control
    df_mean = df.group_by("IDe(2d)").mean()
    # -- infer mu, cov for mean MT, mean IDe using scipy.stats
    mu, cov = stats.multivariate_normal.fit(df_mean.select(["IDe(2d)", "Duration"]))
    block_levels = df_mean["IDe(2d)"]

    emg_control_x, emg_control_y = gen_emg_control(
        beta,
        sigma,
        lambda_emg,
        mu,
        cov,
        block_levels=block_levels,
        ntrials=50,
        rng=None,
        seed=None,
    )

    ############### generate data with t-copula

    rho1, df = fit_t_copula(
        df["IDe(2d)"], df["Duration"]
    )  # rho1 can be estimated by sin(pi/2 tau) where tau is kendall's tau

    # fit id and mt marginals
    u_loc, u_scale = stats.uniform.fit(df_mean.select("IDe(2d)"))
    K, loc, scale = stats.exponnorm.fit(df_mean.select("Duration"))

    id_params = {"distribution": "unif", "params": dict(min=u_loc, max=u_loc + u_scale)}
    mt_params = {
        "distribution": "emg",
        "params": {
            "mu": float(loc),
            "sigma": float(scale),
            "lambda": float(1 / (scale * K)),
        },
    }

    block_levels = stats.uniform(loc=(u_loc - 1e-3), scale=(u_scale + 2e-3)).cdf(
        block_levels
    )

    cop_x, cop_y = gen_t_copula(
        rho1,
        df,
        id_params,
        mt_params,  # for the marginals, pass things that make sense to R
        trials=15,
        block_levels=block_levels,
        cdf_block=False,
        rng=None,
        seed=None,
    )

    fig, axs = plt.subplots(1, 3)
    seaborn.scatterplot(x=emg_x, y=emg_y, ax=axs[0])
    seaborn.scatterplot(x=emg_control_x, y=emg_control_y, ax=axs[1])
    seaborn.scatterplot(x=cop_x, y=cop_y, ax=axs[2])
    plt.ion()
    plt.tight_layout()
    plt.show()
