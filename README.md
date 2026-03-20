Utilities for generating realistic movement time and index of difficulty joint distributions. Example usage

```python
import scipy.stats as stats
import numpy
import polars
import matplotlib.pyplot as plt
import seaborn

import gen_joint_mtide as gen


# === loading some data
df = polars.read_csv(gen.example_data)

############### generate EMG data

# -- infer emg model
# off the shelf fitter should work. If not, refer to fit_emg_arbitrary_variance_model
x, fit = gen.compute_emg_regression_linear_expo_mean(
    numpy.asarray(df["IDe(2d)"]), numpy.asarray(df["Duration"])
)
beta = x[:2]
sigma = x[2]
lambda_emg = x[3:]
# -- end infer model

# without specifying ide levels
emg_x, emg_y = gen.gen_emg(
    beta, sigma, lambda_emg, block_levels=None, ntrials=50, rng=None, seed=None
)

# with ide levels specified:
ide_levels = df["IDe(2d)"].unique()
emg_x, emg_y = gen.gen_emg(
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

emg_control_x, emg_control_y = gen.gen_emg_control(
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

rho1, df = gen.fit_t_copula(
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

cop_x, cop_y = gen.gen_t_copula(
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
```
