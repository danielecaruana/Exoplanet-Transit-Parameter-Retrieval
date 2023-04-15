import pandas as pd
import numpy as np
import batman as bat
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

df = pd.read_csv('kepler_90_pl.csv')
df.set_index('Planet', inplace=True)
k_90_rad = 1.25 * 695700

parameters = bat.TransitParams()
parameters.t0 = 0
parameters.w = 0
parameters.limb_dark = 'linear'
parameters.u = [0.7]
parameters.rp = 0.009
parameters.a = 12.73
parameters.inc = 89.4
parameters.ecc = 0
parameters.per = 7

t = np.linspace(-0.7, 0.7, 1000)

m = bat.TransitModel(parameters, t)
flux = m.light_curve(parameters)
plt.plot(t,flux)
plt.show()