import batman
import pandas as pd
import numpy as np
import batman as bat
import emcee
import multiprocessing
import corner
import matplotlib.pyplot as plt


def model(theta, x):
    t0, per, rp, a, inc = theta
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

    m = batman.TransitModel(params, x)
    rel_flux = m.light_curve(params)
    return rel_flux


def ln_like(theta, x, y, y_err):
    ln_like_val = -0.5 * np.sum(np.square((y - model(theta, x)) / y_err))
    return ln_like_val


def ln_prior(theta):
    t0, per, rp, a, inc = theta
    if (0.71 + 2.453989e6 <= t0 <= 0.78 + 2.453989e6) and (2.1 <= per <= 2.8) and (0.09 <= rp <= 0.15) and (
            7.4 <= a <= 7.9) and 60 <= inc <= 90:
        return 0
    else:
        return -np.inf


# 0.785, 0.72 = 0.065
# p squared = a cubed
# rp/rs = change in flux root
def ln_prob(theta, x, y, y_err):
    lp = ln_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_like(theta, x, y, y_err)


def _p0(initial_val, n_dim_val, n_walkers_val):
    return [np.array(initial_val) + [1e-3, 1e-4, 1e-3, 1e-4, 0.1] * np.random.randn(n_dim_val) for i in
            range(n_walkers_val)]


def mcmc(p0, n_walkers, n_iter, n_dim, ln_prob, data_tup, pool):
    sampler_val = emcee.EnsembleSampler(n_walkers, n_dim, ln_prob, args=data_tup, pool=pool)
    print("Running burn-in...")
    p0, _, _ = sampler_val.run_mcmc(p0, 800, progress=True)
    sampler_val.reset()

    print("Running production...")
    posteriors_val, prob_val, state_val = sampler_val.run_mcmc(p0, n_iter, progress=True)
    print("Production Done")
    return sampler_val, posteriors_val, prob_val, state_val


def posteriors_func(samples_post, chain, x):
    models = []
    drw = np.floor(np.random.uniform(0, len(chain), size=samples_post)).astype(int)
    th_s = chain[drw]
    for i in th_s:
        mdl_post = model(i, x)
        models.append(mdl_post)
    spread = np.std(models, axis=0)
    med_mod = np.median(models, axis=0)
    return med_mod, spread


if __name__ == '__main__':
    with multiprocessing.Pool(4) as pool:
        df = pd.read_csv("kepler_lc_group_4.csv")

        time_arr = np.linspace(np.min(df['HJD'].to_numpy()), np.max(df['HJD'].to_numpy()), 433)
    
        data = (time_arr, df['Rel_Flux'], df['Flux_err'])
        plt.plot(time_arr, df['Rel_Flux'])
        plt.xlabel('HJD')
        plt.ylabel('Relative Flux')
        plt.minorticks_on()
        plt.grid(which='both')
        plt.savefig("Init")
        plt.show()


        n_walkers = 250
        n_iter = 5000

        initial = np.array([0.75 + 2.453989e6, 2.5, 0.12, 7.71, 90])
        n_dim = len(initial)

        p0 = _p0(initial, n_dim, n_walkers)

        sampler, posteriors, prob, state = mcmc(p0, n_walkers, n_iter, n_dim, ln_prob, data, pool)
        samples = sampler.flatchain
        print("Done")

        # Corner Plot
        labels = ['t0', 'per', 'rp', 'a', 'i']

        fig, ax = plt.subplots(1)
        ax.figure.set_size_inches(8.27, 11.69)
        corner.corner(samples, show_titles=True, labels=labels, plot_datapoints=True, quantiles=[0.32, 0.5, 0.68])
        plt.savefig("cornerbatman.png")
        plt.show()

        # Plotting 1-sigma posterior

        th_max = samples[np.argmax(sampler.flatlnprobability)]
        best_fit = model(th_max, time_arr)
        med_mod_val, spread_val = posteriors_func(600, samples, time_arr)

        fig, ax = plt.subplots(1)
        ax.figure.set_size_inches(8.27, 11.69)
        csfont = {'fontname': 'Times New Roman'}
        plt.plot(df['HJD'], df['Rel_Flux'], label='Change of observed relative flux with time')
        plt.plot(df['HJD'], best_fit, label='Highest Likelihood Model')
        plt.fill_between(df['HJD'], med_mod_val-spread_val, med_mod_val+spread_val, color='grey', alpha=0.6,
                         label=r'$1-\sigma$ Posterior Spread')
        plt.xlabel("HJD", **csfont)
        plt.ylabel("Relative Flux", **csfont)
        plt.title("A graph of the change of relative flux with date (HJD) for Kepler-1B", **csfont)
        plt.legend()
        plt.minorticks_on()
        plt.grid(which='both')
        plt.savefig("PosteriorPlot_full")
        plt.show()
        print('Theta:', th_max)
