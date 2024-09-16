from scipy.stats import norm, multivariate_normal
from simulate import gen_t_copula, gen_emg, gen_emg_control


import polars
import numpy

import matplotlib.pyplot as plt
import seaborn

plt.style.use("fivethirtyeight")


def gen_strat_t_copula(
    t_copula_parameters,
    id_params_R,
    mt_params_R,
    blocks,
    strategies,
    participants,
    trials,
    seed=None,
):
    data = []
    rows = []
    rng = numpy.random.default_rng(seed=seed)

    for p in range(participants):
        for ns, s in enumerate(strategies):
            rng.spawn(1)[0]
            x, y = gen_t_copula(
                *t_copula_parameters,
                id_params_R[ns],
                mt_params_R,
                trials=trials,
                block_levels=blocks,
                rng=rng,
            )
            data.append((x, y))
            for _x, _y in zip(x, y):
                rows.append([p, ns, _x, _y])

    return data, rows


def gen_strat_emg(
    emg_params, block_levels, strategies, participants, ntrials, seed=None
):
    data = []
    rows = []
    rng = numpy.random.default_rng(seed=seed)

    for p in range(participants):
        for ns, s in enumerate(strategies):
            rng.spawn(1)[0]
            x, y = gen_emg(
                emg_params[:2],
                emg_params[2],
                emg_params[3:],
                block_levels=block_levels[p, ns, ...],
                ntrials=ntrials,
                rng=rng,
            )
            data.append((x, y))
            for _x, _y in zip(x, y):
                rows.append([p, ns, _x, _y])

    return data, rows


def gen_strat_emg_control(
    emg_params, mvg_mu, mvg_cov, blocks, strategies, participants, ntrials, seed=None
):
    data = []
    rows = []
    rng = numpy.random.default_rng(seed=seed)

    for p in range(participants):
        for ns, s in enumerate(strategies):
            for b in range(blocks):
                rng.spawn(1)[0]
                x, y = gen_emg_control(
                    emg_params[:2],
                    emg_params[2],
                    emg_params[3:],
                    mvg_mu[ns],
                    mvg_cov[ns],
                    ntrials=ntrials,
                    rng=rng,
                )
                data.append((x, y))
                for _x, _y in zip(x, y):
                    rows.append([p, ns, _x, _y])

    return data, rows


if __name__ == "__main__":

    seed = 999
    strategies = [-1, -0.5, 0, 0.5, 1]
    participants = 15
    ntrials = 16
    blocks = 5

    ## t-cop model 1

    mu_i = [4.72 + 2.3 * s for s in strategies]
    sigma_i = [1.06 + 0.39 * s for s in strategies]

    t_copula_parameters = [0.69, 17]

    id_params = [
        {"distribution": "norm", "params": dict(mean=mu, sd=sigma)}
        for mu, sigma in zip(mu_i, sigma_i)
    ]
    mt_params = {
        "distribution": "emg",
        "params": {"mu": 0.5293452, "sigma": 0.1890695, "lambda": 1.3338371},
    }

    data, rows = gen_strat_t_copula(
        t_copula_parameters,
        id_params,
        mt_params,
        blocks,
        strategies,
        participants,
        ntrials,
        seed=seed,
    )
    df_cop = polars.DataFrame(rows, schema=["Participant", "strategy", "IDe", "MT"])

    fig_cop, ax_cop = plt.subplots(1, 1)
    seaborn.scatterplot(df_cop, x="IDe", y="MT", hue="strategy", ax=ax_cop)

    # emg model 2
    block_levels = (
        norm(loc=mu_i, scale=sigma_i)
        .rvs(size=(participants, blocks, 5))
        .transpose(0, 2, 1)
    )

    emg_params = [0.08, 0.14, 0.18, 0.17, 0.08]

    data, rows = gen_strat_emg(
        emg_params, block_levels, strategies, participants, ntrials, seed=seed
    )

    df_emg = polars.DataFrame(rows, schema=["Participant", "strategy", "IDe", "MT"])

    fig_emg, ax_emg = plt.subplots(1, 1)
    seaborn.scatterplot(df_emg, x="IDe", y="MT", hue="strategy", ax=ax_emg)

    # emg control model 3
    mvg_mu = [numpy.array([4.72 + 2.3 * s, 1.3 + 0.64 * s]) for s in strategies]
    mvg_cov = [numpy.array([[(1.06 - 0.29) ** 2, 0], [0, 0.39 - 0.08]])] + [
        numpy.array(
            [
                [(1.06 + 0.29 * s) ** 2, 0.44 * (1.06 + 0.29 * s) * (0.39 + 0.08 * s)],
                [0.44 * (1.06 + 0.29 * s) * (0.39 + 0.08 * s), (0.39 + 0.08 * s) ** 2],
            ]
        )
        for s in strategies[1:]
    ]

    data, rows = gen_strat_emg_control(
        emg_params,
        mvg_mu,
        mvg_cov,
        blocks,
        strategies,
        participants,
        ntrials,
        seed=seed,
    )

    df_emg_control = polars.DataFrame(
        rows, schema=["Participant", "strategy", "IDe", "MT"]
    )

    fig_emg_control, ax_emg_control = plt.subplots(1, 1)
    seaborn.scatterplot(
        df_emg_control, x="IDe", y="MT", hue="strategy", ax=ax_emg_control
    )

    plt.ion()
    plt.show()
