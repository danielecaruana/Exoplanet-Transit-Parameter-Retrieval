import pandas as pd
import numpy as np
import batman as bat
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def planet_param(arr):
    planet_dic = {}
    names = arr.index
    for index, param in enumerate(arr):
        planet_dic[names[index]] = param
    return planet_dic


def transit(planet, profile, c1=0.0, c2=0.0):
    parameters = bat.TransitParams()
    parameters.t0 = 0
    parameters.w = 0
    if profile == 'u':
        parameters.limb_dark = 'uniform'
        parameters.u = []
    elif profile == 'l':
        parameters.limb_dark = 'linear'
        parameters.u = [c1]
    elif profile == 'q':
        parameters.limb_dark = 'quadratic'
        parameters.u = [c1, c2]
    parameters.rp = planet['Rp / Rs']
    parameters.a = planet['a / R']
    parameters.inc = planet['Orbital Inclination (deg)']
    parameters.ecc = planet['Eccentricity']
    parameters.per = planet['Orbital Period (yr)'] * 365
    return parameters


# Importing data

df = pd.read_csv('kepler_90_pl.csv')
df.set_index('Planet', inplace=True)
k_90_rad = 1.25 * 695700

# Modifying data and units

df['Rp / Rs'] = df['Planetary Radius (Earth Radius)'] * 6378.14 / k_90_rad
df['a / R'] = df['Semimajor Axis (AU)'] * 1.496e8 / k_90_rad

# Initializing arrays
time_arr = np.linspace(-0.7, 0.7, 1000)
planet_arr = np.array(df.index.values.tolist())
c1 = [0.1, 0.3, 0.5, 0.7, 0.8]
c2 = [0.8, 0.7, 0.5, 0.4, 0.3]

# Generating uniform models for all planets

fig, ax = plt.subplots(1)
ax.figure.set_size_inches(8.27, 11.69)
for i in planet_arr:
    m = bat.TransitModel(transit(df.loc[i], 'u'), time_arr)
    rel_flux = m.light_curve(transit(df.loc[i], 'u'))
    plt.plot(time_arr, rel_flux, label=i)

csfont = {'fontname': 'Times New Roman'}
plt.ylabel(r"Relative Flux", **csfont)
plt.xlabel(r"Time from central transit", **csfont)
plt.title(r"Light curves generated by a uniform limb darkening model for different planets ", **csfont)
plt.legend()
plt.minorticks_on()
plt.grid(which='both')
formatter = mticker.ScalarFormatter(useMathText=True)
ax.xaxis.set_major_formatter(formatter)
plt.savefig('uniform.png')
plt.show()

#  Generating a linear model

fig, ax = plt.subplots(1)
ax.figure.set_size_inches(8.27, 11.69)
for i in c1:
    m = bat.TransitModel(transit(df.loc['Kepler-90 e'], 'l', i), time_arr)
    rel_flux = m.light_curve(transit(df.loc['Kepler-90 e'], 'l', i))
    plt.plot(time_arr, rel_flux, label=r"$c_1 = {}$".format(i))

csfont = {'fontname': 'Times New Roman'}
plt.ylabel(r"Relative Flux", **csfont)
plt.xlabel(r"Time from central transit", **csfont)
plt.title(r"Light curves generated by a linear limb darkening model for Kepler-90 e ", **csfont)
plt.legend()
plt.minorticks_on()
plt.grid(which='both')
formatter = mticker.ScalarFormatter(useMathText=True)
ax.xaxis.set_major_formatter(formatter)
ax.get_yaxis().get_major_formatter().set_useOffset(False)
plt.savefig('linear.png')
plt.show()

# Generating a quadratic model

fig, ax = plt.subplots(1)
ax.figure.set_size_inches(8.27, 11.69)
for i in range(len(c1)):
    m = bat.TransitModel(transit(df.loc['Kepler-90 e'], 'q', c1[i], c2[i]), time_arr)
    rel_flux = m.light_curve(transit(df.loc['Kepler-90 e'], 'q', c1[i], c2[i]))
    plt.plot(time_arr, rel_flux, label=r"$c_1 = {}, c_2 = {}$".format(c1[i], c2[i]))

csfont = {'fontname': 'Times New Roman'}
plt.ylabel(r"Relative Flux", **csfont)
plt.xlabel(r"Time from central transit", **csfont)
plt.title(r"Light curves generated by a quadratic limb darkening model for Kepler-90 e ", **csfont)
plt.legend()
plt.minorticks_on()
plt.grid(which='both')
formatter = mticker.ScalarFormatter(useMathText=True)
ax.xaxis.set_major_formatter(formatter)
ax.get_yaxis().get_major_formatter().set_useOffset(False)
plt.savefig('quadratic.png')
plt.show()

print("Debug")

