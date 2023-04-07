import batman
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import batman as bat


def model(time, t0, per, rp, a, inc):
    params = bat.TransitParams()
    params.w = 0
    params.ecc = 0
    params.limb_dark = 'quadratic'
    params.u = [0.22, 0.32]
    params.t0 = t0
    params.per = per
    params.rp = rp
    params.a = a
    params.inc = inc

    m = batman.TransitModel(params, time)
    rel_flux = m.light_curve(params)
    return np.array(rel_flux)


def ln_like(theta, x, y, y_err):
    ln_like_val = -0.5 * np.sum(np.square((y - model(theta, x)) / y_err))
    return ln_like_val


def ln_prior(theta):
    a, b, c, d = theta
    if 30 <= a <= 33.5 and 2.5 <= b <= 3.5 and 23 <= c <= 26 and 6 <= d <= 11:
        return 0
    else:
        return -np.inf

#0.785, 0.72 = 0.065

def ln_prob(theta, x, y, y_err):
    lp = ln_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_like(theta, x, y, y_err)


def _p0(initial_val, n_dim_val, n_walkers_val):
    return [np.array(initial_val) + [1e-4, 75e-3, 1e-2, 1e-2] * np.random.randn(n_dim_val) for i in
            range(n_walkers_val)]


df = pd.read_csv("kepler_lc_group_4.csv")

time_arr = np.linspace(np.min(df['HJD']), np.max(df['HJD']), 1000)
