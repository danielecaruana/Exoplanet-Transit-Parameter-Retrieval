import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('kepler_lc_group_4.csv')

plt.plot(df['HJD'], df['Rel_Flux'])
plt.show()