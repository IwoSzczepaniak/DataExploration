import numpy as np
import statsmodels.api as sm

file = 'xy-002.csv'

x,y = np.loadtxt(file,delimiter=',',unpack=True,skiprows=1)
X_plus_one = np.stack( (np.ones(x.size),x), axis=-1)
X_plus_one
ols = sm.OLS(y, X_plus_one)
ols_result = ols.fit()
print(ols_result.summary())