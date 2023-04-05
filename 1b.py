import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import emcee
import corner
import multiprocessing


def model(theta, x):
    a, b, c, d = theta
    model_val = a + (b * np.sin(c * x) + d)
    return model_val


def ln_like(theta, x, y, y_err):
    ln_like_val = -0.5 * np.sum(np.square((y - model(theta, x)) / y_err))
    return ln_like_val


def ln_prior(theta):
    a, b, c, d = theta
    if 33.5 <= a <= 35.5 and 2.5 <= b <= 3.5 and 20 <= c <= 25 and 6 <= d <= 25:
        return 0
    else:
        return -np.inf


def ln_prob(theta, x, y, y_err):
    lp = ln_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_like(theta, x, y, y_err)


def _p0(initial_val, n_dim_val, n_walkers_val):
    return [np.array(initial_val) + [1e-4, 75e-3, 1e-3, 1e-2] * np.random.randn(n_dim_val) for i in
            range(n_walkers_val)]


def mcmc(p0, n_walkers, n_iter, n_dim, ln_prob, data_tup, pool):
    filename = "savefile_correct.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(n_walkers, n_dim)
    sampler_val = emcee.EnsembleSampler(n_walkers, n_dim, ln_prob, args=data_tup, backend=backend, pool=pool)
    print("Running burn-in...")
    p0, _, _ = sampler_val.run_mcmc(p0, 800, progress=True)
    sampler_val.reset()

    print("Running production...")
    posteriors_val, prob_val, state_val = sampler_val.run_mcmc(p0, n_iter, progress=True)
    print("Production Done")
    return sampler_val, posteriors_val, prob_val, state_val


df = pd.read_csv('mcmc_intro_group_5.csv')
data = (df['x'], df['y'], df['y_err'])

n_walkers = 300 #500
n_iter = 20000 #50000

initial = np.array([34.5, 3, 23, 8])
n_dim = len(initial)

p0 = _p0(initial, n_dim, n_walkers)

if __name__ == '__main__':
    with multiprocessing.Pool(4) as pool:
        sampler, posteriors, prob, state = mcmc(p0, n_walkers, n_iter, n_dim, ln_prob, data, pool)
        samples = sampler.flatchain

        # Corner Plot
        labels = ['a', 'b', 'c', 'd']
        # reader = emcee.backends.HDFBackend("savefile")
        tau = sampler.get_autocorr_time()
        # burnin = int(2 * np.max(tau))
        # thin = int(0.5 * np.min(tau))
        # samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
        # log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)
        # log_prior_samples = reader.get_blobs(discard=burnin, flat=True, thin=thin)
        print("Done")
        fig = corner.corner(samples, show_titles=True, labels=labels, plot_datapoints=True, quantiles=[0.25, 0.5, 0.75])
        plt.savefig("corner.png")
# Value Checking from plot

# plt.plot(df['x'], df['y'])
# plt.axhline(y=32)
# plt.show()

print("Debug")
