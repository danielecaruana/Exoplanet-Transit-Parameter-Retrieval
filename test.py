import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('kepler_lc_group_4.csv')
# print((0.785 + 2.453989e6) - (0.72 + 2.453989e6))
plt.plot(df['HJD'], df['Rel_Flux'])

print((df['HJD'][50]) - (df['HJD'][0]) / 50)
plt.minorticks_on()
plt.grid(which='both')
plt.show()