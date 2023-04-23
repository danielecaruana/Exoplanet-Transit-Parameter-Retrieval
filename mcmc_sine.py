import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import emcee
import corner
import multiprocessing


def model(theta, x):
    a, b, c, d = theta
    model_val = a + (b * (np.sin((c * x)+d)))
    return model_val


def ln_like(theta, x, y, y_err):
    ln_like_val = -0.5 * np.sum(np.square((y - model(theta, x)) / y_err))
    return ln_like_val


def ln_prior(theta):
    a, b, c, d = theta
    if 30 <= a <= 33.5 and 2.5 <= b <= 3.5 and 23 <= c <= 26 and 6 <= d <= 11:
        return 0
    else:
        return -np.inf


def ln_prob(theta, x, y, y_err):
    lp = ln_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_like(theta, x, y, y_err)


def _p0(initial_val, n_dim_val, n_walkers_val):
    return [np.array(initial_val) + [1e-4, 75e-3, 1e-2, 1e-2] * np.random.randn(n_dim_val) for i in
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


df = pd.read_csv('mcmc_intro_group_5.csv')
data = (df['x'], df['y'], df['y_err'])

# Value Checking from plot

plt.plot(df['x'], df['y'], color='black')
plt.axhline(y=32, color='black')
plt.minorticks_on()
plt.grid(which='both')
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("Sine")
plt.show()

n_walkers = 100
n_iter = 3000

initial = np.array([32.5, 3, 24, 8])
n_dim = len(initial)

p0 = _p0(initial, n_dim, n_walkers)

if __name__ == '__main__':
    with multiprocessing.Pool(4) as pool:
        sampler, posteriors, prob, state = mcmc(p0, n_walkers, n_iter, n_dim, ln_prob, data, pool)
        samples = sampler.flatchain

        # Corner Plot
        labels = ['a', 'b', 'c', 'd']
        print("Done")
        fig, ax = plt.subplots(1)
        ax.figure.set_size_inches(8.27, 11.69)
        fig = corner.corner(samples, show_titles=True, labels=labels, plot_datapoints=True, quantiles=[0.32, 0.5, 0.68])
        plt.savefig("corner_sine.png")


print("Debug")
