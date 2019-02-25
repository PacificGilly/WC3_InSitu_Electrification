############################################################################
# Project: Electrical Pre-Conditioning of Convective Clouds,
# Title: Solving the SLWC collection efficiency
# Author: James Gilmore,
# Email: james.gilmore@pgr.reading.ac.uk.
# Version: 0.1
# Date: 26/10/17
# Status: Alpha
############################################################################
from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import os, sys, time
from statsmodels.sandbox.regression.predstd import wls_prediction_std

sys.path.insert(0, '/home/users/th863480/PhD/Global_Functions')
import Gilly_Utilities as gu

t_begin = time.time()

Storage_Path = '/glusterfs/phd/users/th863480/WC3_InSitu_Electrification/'
Data_Path = 'Processed_Data/Development/SLWC_Sensor/'
Plots_Path = 'Plots/Development/SLWC_Sensor/'

#Import Data
Efficiency_Data = np.genfromtxt(Storage_Path + Data_Path + 'CollectionEfficiency_Dataset_Python_All.csv', delimiter=",", dtype=float, skip_header=1)
					
#Set X and Y axis. Note the indicies have changed. First column is always Y data. the remaining columns are X data
X = Efficiency_Data[:,0]
Y = Efficiency_Data[:,1:]

t1 = time.time()
results, ex, X2, Y2 = gu.Lineariser_Power_v2(Y, X.T, initial=0, maxits=1000, prec=10**-7)
t2 = time.time()
prstd, iv_l, iv_u = wls_prediction_std(results)

print("Power Coeff.", ex)
print(results.summary())
print(results.rsquared_adj)

"""Plot Linear Regression"""
t3 = time.time()
plt.clf()
fig, ax = plt.subplots(figsize=(8,6))
X3 = np.dot(X2, results.params)

#Plot original data with Regression
ax.plot(X3, Y2, 'o', label="data")
ax.plot(X3, results.fittedvalues, 'k', label="OLS", lw=0.5)

#Include errors
fill_kwargs = {'lw':0.0, 'edgecolor':None}
s = np.lexsort((iv_u, iv_l, X3))
ax.fill_between(X3[s], iv_u[s], iv_l[s], facecolor='black', alpha=0.3, interpolate=True, **fill_kwargs)

#Add axis and titles
ax.set_xlabel("Droplet Diameter (um)")
ax.set_ylabel("Collection Efficiency")
ax.set_title("Power Regression Model of SLWC collection efficiency")

#Plot extras
ax.legend(loc='upper left');
ax.annotate("Linear $R^2$ = %.4f \np-value = %.4f" % (results.rsquared, results.pvalues[1]), xy=(1, 1), xycoords='axes fraction', fontsize=10, xytext=(-10, -10), textcoords='offset points', ha='right', va='top')
ax.grid(which='major',axis='both',c='grey')

plt.savefig(Storage_Path + Plots_Path + 'SLWC_Collection_Efficiency_10ms.png', bbox_inches='tight',pad_inches=0.1, dpi=300)
t4 = time.time()

print("Params", results.params)
print("1/1 R^2", gu.R1to1(X3, Y2))
print("Regression Took %.5fs while plotting took %.5fs. Total Time Taken = %.5f" % (t2-t1, t4-t3, time.time()-t_begin))