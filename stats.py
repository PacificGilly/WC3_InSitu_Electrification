import numpy as np
import scipy as sp
import statsmodels.api as sm
import sys
import warnings
try:
	from sklearn.linear_model import TheilSenRegressor, HuberRegressor
	Huber = True
except ImportError:
	from sklearn.linear_model import TheilSenRegressor
	Huber = False

from collections import namedtuple	
	
from .manipulation import stride, broadcast, digitize, searchsorted
from .nanfunctions import antinan, antifinite
from .math import power

sys.path.insert(0, '/home/users/th863480/PhD/Global_Functions/Prerequisites/modules')
import somestats.bootstrap as boot

femto = sys.float_info.epsilon

__version__ = 1.1

HuberRegressionResult = namedtuple('HuberRegressionResult', ('slope', 
															'intercept',
															'rvalue', 
															'pvalue',
															'stderr'))

### Covariance Methods ###
#Below is a list of covariance algorithms which have been found from
#wikipedia (https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Covariance)
#and from other sources.

def naive_covariance(data1, data2):
	"""Fast but prone to catastrophic cancellaion when
	E[XY] ~ E[X]E[Y] i.e. the numbers are small and
	not negligible to the computer precision limit"""

	n = len(data1)
	sum12 = 0
	sum1 = sum(data1)
	sum2 = sum(data2)

	for i1, i2 in zip(data1, data2):
		sum12 += i1*i2

	covariance = (sum12 - sum1*sum2 / n) / n
    
	return covariance
	
def shifted_data_covariance(dataX, dataY):
	"""Fixes catastrophic cancellation probelem"""

	n = len(dataX)
	if (n < 2):
		return 0
	Kx = dataX[0]
	Ky = dataY[0]
	Ex = Ey = Exy = 0
	for iX, iY in zip(dataX, dataY):
		Ex += iX - Kx
		Ey += iY - Ky
		Exy += (iX - Kx) * (iY - Ky)
	
	return (Exy - Ex * Ey / n) / n
	
def shifted_data_covariance_numpy(dataX, dataY, elem=None, fill_edges=False):
	"""Expects numpy arrays input but will calculate the
	covariance much quicker
	
	N.B. This is using the biased estimator version where
	we normalised the variance by the number of values in
	our array (i.e. n). For an unbiased estimator change 
	the last n in the return line from n to n-1
	
	Parameters
	----------
	elem : int, optional
		elem is not None then dataX and dataY is subset in
		arrays of length elem and the covariance of each
		subset is found.
	fill_edges : bool, optional
        This module will split the data and centre on the
        parent dataset. Therefore, this option will buffer
        the covariance array with 0's to keep the dimensions
        the same
	"""
	
	n = dataX.size
	if n < 2: return np.nan
	
	if elem is None:
		
		Ex 	= sum(dataX - dataX[0])
		Ey 	= sum(dataY - dataY[0])
		Exy = sum((dataX - dataX[0])*(dataY - dataY[0]))
		
		return (Exy - Ex * Ey / n) / n
	
	else:
		
		as_strided = np.lib.stride_tricks.as_strided #Numpy array broadcasting trick
		strides = dataX.strides[0]					 #Used for memory checking of array 
		ncols = (n - elem)
		nrows = elem
		
		#Split dataX in array of elem elements and determine the mean subset offset by the first element of each subset
		#dataXX = np.array(zip(*(dataX[i:] for i in xrange(elem)))).T
		#dataXX = np.array([dataX[i:i+elem] for i in xrange(dataX.size-elem+1)]).T
		dataXX = as_strided(dataX, (ncols, nrows), (strides, strides)).copy().T
		Ex = sum(dataXX - dataX[:-elem])
		
		#Split dataY in array of elem elements and determine the mean subset offset by the first element of each subset
		#dataYY = np.array(zip(*(dataY[i:] for i in xrange(elem)))).T
		#dataYY = np.array([dataY[i:i+elem] for i in xrange(dataY.size-elem+1)]).T
		dataYY = as_strided(dataY, (ncols, nrows), (strides, strides)).copy().T
		Ey = sum(dataYY - dataY[:-elem])
		
		#Calculate the sum of means for each subset
		Exy = sum((dataXX - dataX[:-elem])*(dataYY - dataY[:-elem]))
		
		cov = (Exy - Ex * Ey / elem) / elem
		
		if fill_edges is True:
			return list_func(cov).rljust(dataX.size, 0, float)
		else:
			return cov
		   
def two_pass_covariance(data1, data2):
	"""Two pass algorithm computes the sample means
	and then the covariance"""

	n = len(data1)

	mean1 = sum(data1) / n
	mean2 = sum(data2) / n

	covariance = 0

	for i1, i2 in zip(data1, data2):
		a = i1 - mean1
		b = i2 - mean2
		covariance += a*b / n
    
	return covariance
	
def Lineariser_Power(endog, exog, weights=None, initial=1, maxits=1000, prec=10**-7):
	"""Linearises exog (x axis) using a power law setup and iterates
	the power function b. The newly manipulated data is then solved
	using an OLS where the remaining a and c parameters can be solved
	following the format:
	
						y = a*x^b + c

	NOTE 1 30/08/17: NumPy is unable to raise float by a fractional float
	(e.g. -0.455**0.9) which will return a np.nan. To overcome this we've
	had to temporaily convert the array exog to a complex function and
	then solve and combine the real and imaginary parts using the np.
	absolute function. This provides the correct solution. On the down
	side this will have ramifications for computational speeds. No 
	solution really exists for this issue. 
	
	see for info: https://stackoverflow.com/a/17747293
	
	"""
	
	iter = initial
	sw = 1
	iter_all = []
	sw_all = []
	for i in xrange(maxits):
		
		X2 = np.column_stack((np.ones(len(exog)), complex2float(np.array(exog, dtype=complex)**(iter-0.1/sw))))
		model = sm.OLS(endog, X2) if weights is None else sm.WLS(endog, X2, weights=weights) 
		res_min = model.fit().rsquared**0.5
		
		X2 = np.column_stack((np.ones(len(exog)), complex2float(np.array(exog, dtype=complex)**iter)))
		model = sm.OLS(endog, X2) if weights is None else sm.WLS(endog, X2, weights=weights) 
		res = model.fit().rsquared**0.5
		
		X2 = np.column_stack((np.ones(len(exog)), complex2float(np.array(exog, dtype=complex)**(iter+0.1/sw))))
		model = sm.OLS(endog, X2) if weights is None else sm.WLS(endog, X2, weights=weights) 
		res_plus = model.fit().rsquared**0.5
		
		if (res_min > res) &  (res > res_plus):
			iter -= 0.1/sw
			if sw <= 1: sw /= 2.
		elif (res_min < res) &  (res < res_plus):
			iter += 0.1/sw
			if sw <= 1: sw /= 2.
		elif (res_min > res) &  (res < res_plus):
			if (res_min - res) > (res_plus - res):
				iter -= 0.01/sw
			else:
				iter += 0.01/sw
			sw += 1
		elif (res_min < res) &  (res > res_plus):
			if (res - res_min) > (res - res_plus):
				iter += 0.01/sw
			else:
				iter -= 0.01/sw
			sw  += 1
		iter_all.append(iter)
		sw_all.append(sw)
		
		if np.abs((res - res_min) + (res - res_plus)) < prec: break
	if i == maxits-1: warnings.warn("Maximum iterations (%.0f) reached and precision of the R squared value was not found!" % (maxits), UserWarning, stacklevel=2)
	
	X2 = np.column_stack((complex2float(np.array(exog, dtype=complex)**iter), np.ones(len(exog))))
	model = sm.OLS(endog, X2) if weights is None else sm.WLS(endog, X2, weights=weights) 
	res = model.fit()
	
	return res, iter, X2, endog
	
def Lineariser_Power_v2(endog, exog, initial=1, maxits=1000, prec=10**-7, weights=None, std_err=False):
	"""Linearises exog (x axis) using a power law set-up and iterates
	the power function b. The newly manipulated data is then solved
	using an OLS where the remaining a and c parameters can be solved
	following the format:
	
						y = a*x^b + c

	NOTE 1 30/08/17: NumPy is unable to raise float by a fractional float
	(e.g. -0.455**0.9) which will return a np.nan. To overcome this we've
	had to temporarily convert the array exog to a complex function and
	then solve and combine the real and imaginary parts using the np.
	absolute function. This provides the correct solution. On the down
	side this will have ramifications for computational speeds. No 
	solution really exists for this issue. 
	
	see for info: https://stackoverflow.com/a/17747293
	
	Now added in error analysis for the power law parameter. This is
	achieved using bootstrapping, similar to EPCC_PC_Estimator. The one
	large downside is the time it takes to compute several 100 bootstraps
	which are typically required for an appropriate estimator or the power
	standard error. 
	
	std_err : int, optional
		the number of bootstrap iterations required for power error 
		analysis.	
	"""
	
	Power = initial
	sw = 1
	Power_All = []
	sw_all = []
	for i in xrange(maxits):
		
		X2 = np.column_stack((complex2float(np.array(exog, dtype=complex)**(Power-0.1/sw)), np.ones(len(exog))))
		#print("np.column_stack((endog, X2, weights))", np.column_stack((endog, X2, weights)).shape)
		#sys.exit()
		YX = antifinite(np.column_stack((endog, X2)).T)	if weights is None else antifinite(np.column_stack((endog, X2, weights)).T)
		model = sm.OLS(YX[:,0], YX[:,1:]) if weights is None else sm.WLS(YX[:,0], YX[:,1:-1], weights=YX[:,-1]) 
		res_min = model.fit().rsquared
		
		X2 = np.column_stack((complex2float(np.array(exog, dtype=complex)**Power), np.ones(len(exog))))
		YX = antifinite(np.column_stack((endog, X2)).T)	if weights is None else antifinite(np.column_stack((endog, X2, weights)).T)		
		model = sm.OLS(YX[:,0], YX[:,1:]) if weights is None else sm.WLS(YX[:,0], YX[:,1:-1], weights=YX[:,-1]) 
		res = model.fit().rsquared
		
		X2 = np.column_stack((complex2float(np.array(exog, dtype=complex)**(Power+0.1/sw)), np.ones(len(exog))))
		YX = antifinite(np.column_stack((endog, X2)).T)	if weights is None else antifinite(np.column_stack((endog, X2, weights)).T)		
		model = sm.OLS(YX[:,0], YX[:,1:]) if weights is None else sm.WLS(YX[:,0], YX[:,1:-1], weights=YX[:,-1])  
		res_plus = model.fit().rsquared
		
		if (res_min > res) &  (res > res_plus):
			Power -= 0.1/sw
			if sw <= 1: sw /= 2.
		elif (res_min < res) &  (res < res_plus):
			Power += 0.1/sw
			if sw <= 1: sw /= 2.
		elif (res_min > res) &  (res < res_plus):
			if (res_min - res) > (res_plus - res):
				Power -= 0.01/sw
			else:
				Power += 0.01/sw
			sw += 1
		elif (res_min < res) &  (res > res_plus):
			if (res - res_min) > (res - res_plus):
				Power += 0.01/sw
			else:
				Power -= 0.01/sw
			sw  += 1
		Power_All.append(Power)
		sw_all.append(sw)
		
		if np.abs((res - res_min) + (res - res_plus)) < prec: break
	if i == maxits-1: warnings.warn("Maximum iterations (%.0f) reached and precision of the R squared value was not found!" % (maxits), UserWarning, stacklevel=2)
	
	X2 = np.column_stack((complex2float(np.array(exog, dtype=complex)**Power), np.ones(len(exog))))
	YX = antifinite(np.column_stack((endog, X2)).T)	if weights is None else antifinite(np.column_stack((endog, X2, weights)).T)
	model = sm.OLS(YX[:,0], YX[:,1:]) if weights is None else sm.WLS(YX[:,0], YX[:,1:-1], weights=YX[:,-1]) 
	res = model.fit()
	Power_Best = Power
	
	#Perform error analysis using bootstrap
	if std_err is not False: 
		Residuals = endog - res.fittedvalues
		Results_All = np.zeros(std_err, dtype=object)
		
		for jj in xrange(std_err):
			Bootstrap = np.array([value + Residuals[np.random.randint(exog.size)] for value in endog], dtype=float)
			
			#Rerun power lineariser
			Power = initial
			sw = 1
			Power_All = []
			sw_all = []
			for i in xrange(maxits):
				
				X2 = np.column_stack((complex2float(np.array(exog, dtype=complex)**(Power-0.1/sw)), np.ones(len(exog))))
				YX = antifinite(np.column_stack((Bootstrap, X2)).T)	if weights is None else antifinite(np.column_stack((endog, X2, weights)).T)		
				model = sm.OLS(YX[:,0], YX[:,1:])if weights is None else sm.WLS(YX[:,0], YX[:,1:-1], weights=YX[:,-1]) 
				res_min = model.fit().rsquared
				
				X2 = np.column_stack((complex2float(np.array(exog, dtype=complex)**Power), np.ones(len(exog))))
				YX = antifinite(np.column_stack((Bootstrap, X2)).T)	if weights is None else antifinite(np.column_stack((endog, X2, weights)).T)		
				model = sm.OLS(YX[:,0], YX[:,1:]) if weights is None else sm.WLS(YX[:,0], YX[:,1:-1], weights=YX[:,-1]) 
				res = model.fit().rsquared
				
				X2 = np.column_stack((complex2float(np.array(exog, dtype=complex)**(Power+0.1/sw)), np.ones(len(exog))))
				YX = antifinite(np.column_stack((Bootstrap, X2)).T)	if weights is None else antifinite(np.column_stack((endog, X2, weights)).T)		
				model = sm.OLS(YX[:,0], YX[:,1:]) if weights is None else sm.WLS(YX[:,0], YX[:,1:-1], weights=YX[:,-1]) 
				res_plus = model.fit().rsquared
				
				if (res_min > res) &  (res > res_plus):
					Power -= 0.1/sw
					if sw <= 1: sw /= 2.
				elif (res_min < res) &  (res < res_plus):
					Power += 0.1/sw
					if sw <= 1: sw /= 2.
				elif (res_min > res) &  (res < res_plus):
					if (res_min - res) > (res_plus - res):
						Power -= 0.01/sw
					else:
						Power += 0.01/sw
					sw += 1
				elif (res_min < res) &  (res > res_plus):
					if (res - res_min) > (res - res_plus):
						Power += 0.01/sw
					else:
						Power -= 0.01/sw
					sw  += 1
				Power_All.append(Power)
				sw_all.append(sw)
				
				if np.abs((res - res_min) + (res - res_plus)) < prec: break
			if i == maxits-1: warnings.warn("Maximum iterations (%.0f) reached and precision of the R squared value was not found!" % (maxits), UserWarning, stacklevel=2)
			
			Results_All[jj] = Power
		
		#Determine the 95% confidence limits
		Power_Error = np.diff(np.percentile(Results_All, [5,95])).astype(float)/2
		
		return res, [Power_Best, Power_Error], YX[:,1:], YX[:,0]
		
	return res, Power_Best, YX[:,1:], YX[:,0]

def Lineariser_Power_v3(endog, exog, initial=1, maxits=1000, prec=10**-7, weights=None, std_err=False):
	"""Linearises exog (x axis) using a power law set-up and iterates
	the power function b. The newly manipulated data is then solved
	using an OLS where the remaining a and c parameters can be solved
	following the format:
	
						y = a*x^b + c

	NOTE 1 30/08/17: NumPy is unable to raise float by a fractional float
	(e.g. -0.455**0.9) which will return a np.nan. To overcome this we've
	had to temporarily convert the array exog to a complex function and
	then solve and combine the real and imaginary parts using the np.
	absolute function. This provides the correct solution. On the down
	side this will have ramifications for computational speeds. No 
	solution really exists for this issue. 
	
	see for info: https://stackoverflow.com/a/17747293
	
	Now added in error analysis for the power law parameter. This is
	achieved using bootstrapping, similar to EPCC_PC_Estimator. The one
	large downside is the time it takes to compute several 100 bootstraps
	which are typically required for an appropriate estimator or the power
	standard error. 
	
	std_err : int, optional
		the number of bootstrap iterations required for power error 
		analysis.	
	"""
	
	#Inital Conditions
	coeff_power = initial
	sw = 1
	Power_All = []
	sw_all = []
	
	#Optimise power coefficient
	for i in xrange(maxits):
		X2 = np.column_stack((power(exog, coeff_power-0.1/sw), np.ones(len(exog))))
		YX = antinan(np.column_stack((endog, X2)).T) if weights is None else antinan(np.column_stack((endog, X2, weights)).T)

		model = sm.OLS(YX[:,0], YX[:,1:]) if weights is None else sm.WLS(YX[:,0], YX[:,1:-1], weights=YX[:,-1]) 
		res_min = model.fit().rsquared
		
		X2 = np.column_stack((power(exog, coeff_power), np.ones(len(exog))))
		YX = antinan(np.column_stack((endog, X2)).T) if weights is None else antinan(np.column_stack((endog, X2, weights)).T)		
		model = sm.OLS(YX[:,0], YX[:,1:]) if weights is None else sm.WLS(YX[:,0], YX[:,1:-1], weights=YX[:,-1]) 
		res = model.fit().rsquared
		
		X2 = np.column_stack((power(exog, coeff_power+0.1/sw), np.ones(len(exog))))
		YX = antinan(np.column_stack((endog, X2)).T) if weights is None else antinan(np.column_stack((endog, X2, weights)).T)		
		model = sm.OLS(YX[:,0], YX[:,1:]) if weights is None else sm.WLS(YX[:,0], YX[:,1:-1], weights=YX[:,-1])  
		res_plus = model.fit().rsquared
		
		if (res_min > res) &  (res > res_plus):
			coeff_power -= 0.1/sw
			if sw <= 1: sw /= 2.
		elif (res_min < res) &  (res < res_plus):
			coeff_power += 0.1/sw
			if sw <= 1: sw /= 2.
		elif (res_min > res) &  (res < res_plus):
			if (res_min - res) > (res_plus - res):
				coeff_power -= 0.01/sw
			else:
				coeff_power += 0.01/sw
			sw += 1
		elif (res_min < res) &  (res > res_plus):
			if (res - res_min) > (res - res_plus):
				coeff_power += 0.01/sw
			else:
				coeff_power -= 0.01/sw
			sw  += 1
		Power_All.append(coeff_power)
		sw_all.append(sw)
		
		if np.abs((res - res_min) + (res - res_plus)) < prec: break
	if i == maxits-1: warnings.warn("Maximum iterations (%.0f) reached and precision of the R squared value was not found!" % (maxits), UserWarning, stacklevel=2)
	
	X2 = np.column_stack((power(exog, coeff_power), np.ones(len(exog))))
	YX = antinan(np.column_stack((endog, X2)).T)	if weights is None else antinan(np.column_stack((endog, X2, weights)).T)

	model = sm.OLS(YX[:,0], YX[:,1:]) if weights is None else sm.WLS(YX[:,0], YX[:,1:-1], weights=YX[:,-1]) 
	res_final = model.fit()
	Power_Best = coeff_power
	
	#Perform error analysis using bootstrap
	if std_err is not False: 
		Residuals = endog - res_final.fittedvalues
		Results_All = np.zeros(std_err, dtype=object)
		
		for jj in xrange(std_err):
			Bootstrap = np.array([value + Residuals[np.random.randint(exog.shape[0])] for value in endog], dtype=float)
			
			#Rerun power lineariser
			coeff_power = initial
			sw = 1
			Power_All = []
			sw_all = []
			for i in xrange(maxits):
				
				X2 = np.column_stack((power(exog, coeff_power-0.1/sw), np.ones(len(exog))))
				YX = antinan(np.column_stack((Bootstrap, X2)).T)	if weights is None else antinan(np.column_stack((endog, X2, weights)).T)		
				model = sm.OLS(YX[:,0], YX[:,1:])if weights is None else sm.WLS(YX[:,0], YX[:,1:-1], weights=YX[:,-1]) 
				res_min = model.fit().rsquared
				
				X2 = np.column_stack((power(exog, coeff_power), np.ones(len(exog))))
				YX = antinan(np.column_stack((Bootstrap, X2)).T)	if weights is None else antinan(np.column_stack((endog, X2, weights)).T)		
				model = sm.OLS(YX[:,0], YX[:,1:]) if weights is None else sm.WLS(YX[:,0], YX[:,1:-1], weights=YX[:,-1]) 
				res = model.fit().rsquared
				
				X2 = np.column_stack((power(exog, coeff_power+0.1/sw), np.ones(len(exog))))
				YX = antinan(np.column_stack((Bootstrap, X2)).T)	if weights is None else antinan(np.column_stack((endog, X2, weights)).T)		
				model = sm.OLS(YX[:,0], YX[:,1:]) if weights is None else sm.WLS(YX[:,0], YX[:,1:-1], weights=YX[:,-1]) 
				res_plus = model.fit().rsquared
				
				if (res_min > res) &  (res > res_plus):
					coeff_power -= 0.1/sw
					if sw <= 1: sw /= 2.
				elif (res_min < res) &  (res < res_plus):
					coeff_power += 0.1/sw
					if sw <= 1: sw /= 2.
				elif (res_min > res) &  (res < res_plus):
					if (res_min - res) > (res_plus - res):
						coeff_power -= 0.01/sw
					else:
						coeff_power += 0.01/sw
					sw += 1
				elif (res_min < res) &  (res > res_plus):
					if (res - res_min) > (res - res_plus):
						coeff_power += 0.01/sw
					else:
						coeff_power -= 0.01/sw
					sw  += 1
				Power_All.append(coeff_power)
				sw_all.append(sw)
				
				if np.abs((res - res_min) + (res - res_plus)) < prec: break
			if i == maxits-1: warnings.warn("Maximum iterations (%.0f) reached and precision of the R squared value was not found!" % (maxits), UserWarning, stacklevel=2)
			
			Results_All[jj] = coeff_power
		
		#Determine the 95% confidence limits
		Power_Error = np.diff(np.percentile(Results_All, [5,95])).astype(float)/2
		
		return res_final, [Power_Best, Power_Error], YX[:,1:], YX[:,0]
		
	return res_final, Power_Best, YX[:,1:], YX[:,0]
	
def TheilSenRegression(x,y, n_subsamples=None, random_state=42):
	"""Calculate the Theil-Sen Linear Regression Model. This regression model is highly robust against
	outliers.
	
	Parameters
	----------
	
	Returns
	-------
	line_x : ndarray
		Two values forming the (min(x), max(x))
	y_predict : ndarray
		Two values with the associated y values as predicted by the model
	coeff : float
		The coefficient of regression model
	intercept : float
		The intercept of the regression model
	r-value : float
		The r-value of the regression model
		
	Notes
	-----
	The p-value and standard error has not been implemented yet. Require bootstrapping, which is slow as
	the TheilSenRegression model is already slow to compute (~100 times slower then OLS).
	
	References
	----------
	1) Theil-Sen Estimators in a Multiple Linear Regression Model, 2009 Xin Dang, Hanxiang Peng, 
	Xueqin Wang and Heping Zhang http://home.olemiss.edu/~xdang/papers/MTSE.pdf
	
	2) http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TheilSenRegressor.html
	
	"""
	
	#Initialise the Theil-Sen Linear Regression Model
	estimator = TheilSenRegressor(n_subsamples=n_subsamples, random_state=random_state)
	
	#Remove nan's in data
	x, y = antifinite((x,y), unpack=True)
	
	#Fit the regression model to data. N.B. add extra dimension [:,None] to x to calculate intercept.
	estimator.fit(x[:,None],y)
	
	#Predict the data from the model
	line_x = np.array([np.nanmin(x), np.nanmax(x)])
	y_predict = estimator.predict(line_x[:,None])
	
	#Calculate r-value
	r_value = power(R1to1(y, estimator.predict(x[:,None])), 0.5)
	
	return line_x, y_predict, estimator.coef_, estimator.intercept_, r_value

def HuberRegression(x,y,epsilon=1.36, package='statsmodel'):
	"""Calculates the Huber linear regression model. Like the Theil-Sen model, the Huber is 
	resilient to outliers present in the data. 
	
	References
	----------
	1) Peter J. Huber, Elvezio M. Ronchetti, Robust Statistics Concomitant scale estimates, p.172
	2) http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html
	
	"""
	
	if Huber is False: raise ImportError("[Warning] HuberRegression is not available in this version of python. :(")
	
	#Remove nan's in data
	x, y = antifinite((x,y), unpack=True)
		
	if package == 'sklearn':
		
		#Initialise the Theil-Sen Linear Regression Model
		estimator = HuberRegressor(epsilon=epsilon)
		
		#Fit the regression model to data. N.B. add extra dimension [:,None] to x to calculate intercept.
		estimator.fit(x[:,None],y)
		
		#Predict the data from the model
		line_x = np.array([np.nanmin(x), np.nanmax(x)])
		y_predict = estimator.predict(line_x[:,None])
		
		#Calculate r-value
		r_value = power(R1to1(y, estimator.predict(x[:,None])), 0.5)
		
		#return line_x, y_predict, estimator.coef_, estimator.intercept_, r_value
		return HuberRegressionResult(estimator.coef_[0], estimator.intercept_, r_value, np.nan, np.nan)
	
	elif package == 'statsmodel':
		
		#Initalise Huber model
		rlm_model = sm.RLM(y, sm.add_constant(x), M=sm.robust.norms.HuberT(epsilon))
		
		#fit model
		rlm_results = rlm_model.fit()
		
		#Calculate r-value
		r_value = power(R1to1(y, rlm_results.predict(sm.add_constant(x))), 0.5)
	
		return HuberRegressionResult(rlm_results.params[1], rlm_results.params[0], r_value, rlm_results.pvalues[1], rlm_results.bse[1])
	
def reg_m(y, x):
	"""Multiple linear regression using statsmodel:
	
	y = dependent variable (1D array)
	x = independent variable(s) (2D array)
	
	https://stackoverflow.com/a/14971531"""

	ones = np.ones(len(x[0]))
	X = sm.add_constant(np.column_stack((x[0], ones)))
	for ele in x[1:]:
		X = sm.add_constant(np.column_stack((ele, X)))
	results = sm.OLS(y, X).fit()
    
	return results	

def rmse(predictions, targets):
	"""Root Mean Square Error (RMSE) in easy form.
	
	Reference
	---------
	https://stackoverflow.com/a/37861832/8765762
	"""
	
	differences = predictions - targets                       #the DIFFERENCEs.

	differences_squared = differences ** 2                    #the SQUAREs of ^

	mean_of_differences_squared = differences_squared.mean()  #the MEAN of ^

	rmse_val = np.sqrt(mean_of_differences_squared)           #ROOT of ^

	return rmse_val                                           #get the ^

def R1to1(X, Y):
	"""This function calculates the 1:1 R Squared value which can be
	defined as the strength of the dependent variable to the independent
	variable rather than the strenth of the linear regression to to the
	dependent variable
	
	Parameters
	----------
	X : ndarray
		The independent variable such as the modelled data
	Y : ndarray
		The dependent variable such as the measured data
		
	NOTE: Requires numpy arrays to conduct element-wise operations!	
	"""
	
	#Remove nan values from both model and measured datasets
	X,Y = antinan(np.array([X,Y]), unpack=True)
	
	#Calculate the 1:1 R^2 from the remaining data
	SStot = np.sum((X-np.mean(X))**2)
	SSres = np.sum((X-Y)**2)
	Rsquared = 1 - SSres/SStot
	
	return Rsquared	

def RMSE(f,x):
	"""Calculates the root mean square error of the data between x and y.
	
	Parameters
	----------
	f, x : array_like
		Input data which must have the same dimensions. The index values must also match up for 
		the RMSE to be meaningful. i.e. for all values of f there must be a corresponding x value.
		N.B. f is defined as your model variable and x as your measured variable.
	"""
	
	f = np.asarray(f)
	x = np.asarray(x)
	
	f, x = antinan((f,x), unpack=True)
	
	if f.size != x.size: raise ValueError("[gu.RMSE] Inputs f and x must have equal dimensions. Got (%s, %s)" % (f.size, x.size))
	if (f.size == 0) | (x.size == 0): ValueError("[gu.RMSE] Inputs f and x must have data inside them. Go sizes of (%s, %s) for f and x respectively." % (f.size, x.size))
	
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		
		return np.sqrt(np.sum((f - x)**2)/f.size)

def fstat(f, y, k=1, p_value=True):
	"""Calculates the f-statistic from the fitted data, f and observed data, y.
	
	For a function, y = f(x,k) with n elements and k parameters,
	
	F = (SSM/DFM)/(SSE/DFE)
	
	SSM = Sum of squares for model
	SSE = Sum of squares for error
	DFM = Degrees of freedom for model
	DFE = Degrees of freedom for error
	
	DFM = k - 1
	DFE = n - k
	
	SSM = sum((y-mean(y))**2)
	SSE = sum((y-yf)**2)
	
	yf is the fitted value for y.
	
	Reference
	---------
	http://facweb.cs.depaul.edu/sjost/csc423/documents/f-test-reg.htm
	https://stats.stackexchange.com/questions/254970/difference-between-ssm-and-ssr
	"""
	
	#Calculate sum of square residuals
	SSE	= np.sum((y-f)**2)
	SSM = np.sum((y-np.nanmean(y))**2)
	
	DFE = y.size - k
	DFM = k - 1
	
	#Calculate F-statistic
	f_ratio = (SSM/DFM)/(SSE/DFE)
	
	#Calculate p-value
	p_value = 1 - sp.stats.f.cdf(f_ratio, DFM, DFE)
	
	return f_ratio, p_value
		
def running_mean(x, N):
	
	#Force N to be integer
	N = int(N)

	cumsum = np.cumsum(np.insert(x, 0, 0)) 
	return (cumsum[N:] - cumsum[:-N]) / N 
	
def moving_average(a, n=3):
	"""Calculate the moving average of a 1D array with n being the number of terms to average over
	
	Reference
	---------
	https://stackoverflow.com/a/14314054/8765762
	"""
	
	ret = np.cumsum(a, dtype=float)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n - 1:] / n

############################################################################
"""Filters"""	
	
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
	r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
		
	Theory: http://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter_for_smoothing_and_differentiation
	Code: http://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
	Help: https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
			
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
		W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
		Cambridge University Press ISBN-13: 9780521880688
	"""

	from math import factorial
    
	try:
		window_size = np.abs(np.int(window_size))
		order = np.abs(np.int(order))
	except ValueError, msg:
		raise ValueError("window_size and order have to be of type int")
	if window_size % 2 != 1 or window_size < 1:
		raise TypeError("window_size size must be a positive odd number")
	if window_size < order + 2:
		raise TypeError("window_size is too small for the polynomials order")
    
	order_range = range(order+1)
	half_window = (window_size -1) // 2
    
	# precompute coefficients
	b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
	m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    
	# pad the signal at the extremes with
    # values taken from the signal itself
	firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
	lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
	y = np.concatenate((firstvals, y, lastvals))
    
	return np.convolve( m[::-1], y, mode='valid')

def savitzky_golay_weights(window_size=None, order=2, derivative=0):
    # The weights are in the first row
    # The weights for the 1st derivatives are in the second, etc.
    return savitzky_golay(window_size, order)[derivative]	
	
############################################################################
"""Kernal Density Functions

based on: https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
"""	

def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
	"""Kernel Density Estimation with Scipy"""
	
	#from kde import gaussian_kde
	from scipy.stats import gaussian_kde
	
    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.
	kde = gaussian_kde(x, **kwargs)
	return kde.evaluate(x_grid)
	
def kde_statsmodels_u(x, x_grid, bandwidth=0.2, **kwargs):
	"""Univariate Kernel Density Estimation with Statsmodels"""
	from statsmodels.nonparametric.kde import KDEUnivariate
	
	kde = KDEUnivariate(x)
	kde.fit(bw=bandwidth, **kwargs)
	return kde.evaluate(x_grid)
  
############################################################################
"""Descriptive Statistics"""
  
def Radar_Stats(Mag_Time, Mag_Height, Mag, Base_Time, Base, Base_Val=0):
	"""Calculates the radar stats from the input product.
	
	You need to input the following datasets:
	
	Mag_Time
	Mag_Height
	Magnitude (e.g. dBZ)
	Base

	"""
	
	if np.sum(np.isnan(Base)) != 1 or Base_Val == 0:
		cloud_R = np.where((Mag>-100)&(Mag_Time>=start)&(Mag_Time<=end)&(Mag_Height>=np.min(Base[np.where((Base_Time>=start)&(Base_Time<=end))])), Mag, np.nan); cloud_R = cloud_R[~np.isnan(cloud_R)]
		cloud_base = np.where((Base_Time>=start)&(Base_Time<=end), flatten(Base), np.nan); cloud_base = cloud_base[~np.isnan(cloud_base)]
	else:
		cloud_R = np.where((Mag>-100)&(Mag_Time>=start)&(Mag_Time<=end)&(Mag_Height>=Base_Val), Mag, np.nan); cloud_R = cloud_R[~np.isnan(cloud_R)]
		
		
	print("R Stats:", np.mean(cloud_R), np.median(cloud_R), np.std(cloud_R), np.max(cloud_R),np.min(cloud_R))
	print("dR Stats:", np.mean(cloud_dR), np.median(cloud_dR), np.std(cloud_dR), np.max(cloud_dR),np.min(cloud_dR))
	print("Base Stats:", np.mean(cloud_base), np.median(cloud_base), np.std(cloud_base), np.max(cloud_base),np.min(cloud_base))

def stats(array, extend=True, axis=None, output=False):
	"""Outputs the standard statistics from the input array
	
	Parameters
	----------
	array : ndarray
		The values you want to produce a statistic for
	extend : boolean, optional
		Specify outputting extra statistics. The extra ones are range and noise 
		(e.g. noise = median/std)
	axis : int, optional
		The axis you want to provide the stats. Default is None which averages as
		a single unit.
		
	Outputs
	-------
	stats : ndarray
		stats array containing the min, max, mean, median and std
	"""
	
	#Error Control
	if array is None: 
		if extend is not False:
			return np.array([np.nan]*14)
		else:
			return np.array([np.nan]*5)
	
	if extend is not False:
		if array.size == 0:
			return np.array([np.nan]*14)
		else:
			#return np.array([np.nanmin(array, axis=axis), np.nanmax(array, axis=axis), np.nanmean(array, axis=axis), np.nanmedian(array, axis=axis), np.nanstd(array, axis=axis), np.nansum(array, axis=axis), np.nanmax(array, axis=axis)-np.nanmin(array, axis=axis), np.nanmedian(array, axis=axis)/np.nanstd(array, axis=axis), iqr(array, 75, axis=axis), iqr(array, 90, axis=axis)], dtype=float)
			#return np.array([np.nanmin(array, axis=axis), np.nanmax(array, axis=axis), np.nanmean(array, axis=axis), np.nanmedian(array, axis=axis), np.nanstd(array, axis=axis), np.nansum(array, axis=axis), np.nanmax(array, axis=axis)-np.nanmin(array, axis=axis), np.nanmedian(array, axis=axis)/np.nanstd(array, axis=axis), np.nanpercentile(array, 75, axis=axis), np.nanpercentile(array, 90, axis=axis)], dtype=float)
			if output is False:
				return np.array([np.nanmin(array, axis=axis), 
					np.nanmax(array, axis=axis), 
					np.nanmean(array, axis=axis), 
					np.nanmedian(array, axis=axis), 
					np.nanstd(array, axis=axis), 
					np.nansum(array, axis=axis), 
					np.nanmax(array, axis=axis)-np.nanmin(array, axis=axis), 
					np.nanmedian(array, axis=axis)/np.nanstd(array, axis=axis), 
					np.nanpercentile(array, 5, axis=axis), 
					np.nanpercentile(array, 10, axis=axis), 
					np.nanpercentile(array, 25, axis=axis),
					np.nanpercentile(array, 75, axis=axis), 
					np.nanpercentile(array, 90, axis=axis), 
					np.nanpercentile(array, 95, axis=axis)], dtype=float)
			else:
				print("Descriptive Statistics")
				print("----------------------")
				print("Minimum = %s" % np.nanmin(array, axis=axis))
				print("Maximum = %s" % np.nanmax(array, axis=axis))
				print("Mean = %s" % np.nanmean(array, axis=axis))
				print("Median = %s" % np.nanmedian(array, axis=axis))
				print("Standard Deviation = %s" % np.nanstd(array, axis=axis))
				print("Total = %s" % np.nansum(array, axis=axis))
				print("Range = %s" % (np.nanmax(array, axis=axis)-np.nanmin(array, axis=axis)))
				print("Noise = %s" % (np.nanmedian(array, axis=axis)/np.nanstd(array, axis=axis)))
				print("5th Percentile = %s" % np.nanpercentile(array, 5, axis=axis))
				print("10th Percentile = %s" % np.nanpercentile(array, 10, axis=axis))
				print("25th Percentile = %s" % np.nanpercentile(array, 25, axis=axis))
				print("75th Percentile = %s" % np.nanpercentile(array, 75, axis=axis))
				print("90th Percentile = %s" % np.nanpercentile(array, 90, axis=axis))
				print("95th Percentile = %s" % np.nanpercentile(array, 95, axis=axis))
				print("----------------------")
	else:
		if array.size == 0:
			np.array([np.nan]*5)
		else:
			if output is False:
				return np.array([np.nanmin(array, axis=axis), 
					np.nanmax(array, axis=axis), 
					np.nanmean(array, axis=axis), 
					np.nanmedian(array, axis=axis), 
					np.nanstd(array, axis=axis)], dtype=float)
			else:
				print("Minimum = %s" % np.nanmin(array, axis=axis))
				print("Maximum = %s" % np.nanmax(array, axis=axis))
				print("Mean = %s" % np.nanmean(array, axis=axis))
				print("Median = %s" % np.nanmedian(array, axis=axis))
				print("Standard Deviation = %s" % np.nanstd(array, axis=axis))
				
def iqr(array, percentile=75, axis=None):
	"""Calculates the interquartile range of an array"""
			
	try:
		return np.subtract(*np.nanpercentile(array, [percentile,100-percentile], axis=axis))
	except TypeError:
		return np.nan

def lowess(x, y, f=2. / 3., iter=3):
    """lowess(x, y, f=2./3., iter=3) -> yest

    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a non-parametric regression curve to a scatter plot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatter plot. The function returns
    the estimated (smooth) values of y.

    The smoothing span is given by f. A larger value for f will result in a
    smoother curve. The number of robustifying iterations is given by iter. The
	function will run faster with a smaller number of iterations.
	"""
	
    n = len(x)
    r = int(np.ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(iter):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = sp.linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2

    return yest

def SSR(x, y):
	"""
	Calcualtes the sum of the square residuals between x and y
	"""
	
	# Ensure data is an ndarray
	x = np.asarray(x)
	y = np.asarray(y)
	
	# Calculate sum of the squre residuals
	return np.nansum((x - y)**2)
	
def log_likelihood(model, endog):
	"""
	Calcualtes the log-likelihood function from a one-dimensional set
	
	Parameters
	----------
	model : 1D array_like
		A one-dimensional array_like of the model output. Size must
		match endog
	endog : 1D array_like
		A one-dimensional array_like of the data to create the model.
		(e.g. for a model y = m*x + c, endog is the y-data)
	"""
	
	# Ensure data is an ndarray
	model = np.asarray(model)
	endog = np.asarray(endog)
	
	# Error checking
	if (model.ndim != 1) or (endog.ndim != 1):
		raise ValueError("log_likelihood requires the inputs to be 1 dimensional")
	
	if model.size != endog.size:
		raise ValueError("log_likelihood requires model and endog to have the same length")
	
	# Calcualte number of observations
	nobs = endog.size
	nobs2 = endog.size / 2.0

	# Calculate the mean and variance
	mean = np.nanmean(endog)
	variance = np.nanvar(endog)
	
	# Calculate sum of square residuals
	ssr = SSR(endog, model)
	
	# Calculate log-likelihood
	# return -nobs / 2 * (np.log(2 * np.pi) - np.log(variance)) - ssr / (2 * variance)		# Wikipedia method
	return -nobs2 * (np.log(2 * np.pi) + np.log(ssr / nobs) + 1)						# Statsmodel method
	   
def AIC(model, endog, k):
	"""
	Calculate the Akaike Information Criterion
	
	Parameters
	----------
	model : 1D array_like
		A one-dimensional array_like of the model output. Size must
		match endog
	endog : 1D array_like
		A one-dimensional array_like of the data to create the model.
		(e.g. for a model y = m*x + c, endog is the x-data)
	k : int
		The number of parameters used to define the model. This is the
		sum of the coefficients, intercept and variance. i.e. 
			
		k = c + i + v
	
	References
	----------
	https://www.researchgate.net/post/What_is_the_AIC_formula
	https://stats.stackexchange.com/questions/87345/calculating-aic-by-hand-in-r
	https://www.statsmodels.org/dev/_modules/statsmodels/regression/linear_model.html#OLS.loglike
	https://en.wikipedia.org/wiki/Akaike_information_criterion#cite_note-19	
	"""
	
	return 2 * k - 2 * log_likelihood(model, endog)

def AICc(model, endog, k):
	""" 
	Calcualtes the adjusted Akaike Information Criterion which is useful
	when the sample size is small [What defines small?]. This is used
	to avoid overfitting of the standard AIC diagnostic.
	
	Parameters
	----------
	See AIC
	"""
	
	return AIC(model, endog, k) + (2 * k**2 + 2 * k) / (np.size(endog) - k - 1)
		
def ensemble(xdata, ydata, bins, average=('mean', 'median'), method='ma', mode=False, undersample=False, slim=False, usestride=False, confidence_intervals='standard', unpack=False):
	"""Averages the ydata set by binning the xdata set into bins of equal elements.
	
	Parameters
	----------
	xdata : 1D array
		The xdata set to represent the midpoints of each bin
	ydata : 1D array
		The ydata set that is used for averaging
	bins : int or list or ndarray
		The number of bins you want to average the dataset over. Supply bin with a list
		or ndarray when you want to manual choose the bin locations.
	average : str or array_like of string, optional
		Specify if you want to average the data placed in each bin using the median or
		mean. Option to specify a 2 element array to specifiy the average type for the
		xdata and ydata;
		
						average = (xdata_type, ydata_type)
								= ('mean', 'median')
	method : str or int, optional
		Specify if you should average over unique elements only (default = 'unique') or
		whether you should roll the bins over all elements ('ma'). If an integer is 
		specified the number will be used to select the number of elements to roll over.
		Final method is 'bootstrap' which slowly increases the number of data elements
		in each bin until the entire population is in one bin. Then the reverse occurs
		when the first elements are removed one by one until only one element remains.
	mode : DEPRECIATED
	undersample : boolean, optional
		Specify if you should average the areas of data that are under sampled. Default
		is false and therefore does not average the under sample data. When set to True
		the number of elements are in the bins near the boundary are subsequently 
		reduced until only a 1 element bin exists.
	confidence_intervals : str, optional
		Specify how you want to define the errors of the averaged data. All confidence
		intervals are given as 95th percentiles. Options are,
		
		'standard' : use the standard error formula. If median averaging is used the
					 approximation is used to correct for the sample. SE = 1.253*SE
		'bootstrap' : use a bootstrapping algorithm to estimate the confidence
					  intervals. N.B. this method is very slow.
		
	Examples
	--------
	If we define an example dataset and the number of bins we want to place our data
	in:
	
	a = np.arange(100)
	bins = 10
	
	We can set the mode to either true or false depending on if we want each element
	of our dataset, a, to be place in one bin or several.
	
	If mode is False, then the bins will have the structure as follows,
		
							0-9, 10-19, 20-29 ... etc. 
							
	and therefore each element can only exist in one unique bin.
	
	If mode is True, then the bins will have the structure as follows,
	
							0-9, 1-10, 2-11, 3-12 ... etc.
							
	and therefore each element can be in multiple bins. This method is used to reduce the
	discontinuities in the data while providing a greater degree of resolution in the
	averaging.
	
	Issues
	------
	1)	The broadcasting of the xdata is done separately to the broadcasting of the ydata
		to reduce the chance for MemoryError when the size of the xdata and ydata are extremely
		large. Depending on the bin number and especially when mode is set to True, the size of  
		the broadcast array can be huge. This is the only disadvantage of performing the 
		averaging using the ensemble function. Otherwise, this is by far the superior method in 
		terms of processing time and simplicity.
	
	Solved Issues
	-------------
	2)	When using a small bin number the data will fail to represent the under sampled regions
		very well. This is shown when mode is True (i.e. we use a rolling window with a 
		predetermined number of elements in each bin) where a lower bin number has more elements
		available for averaging increasing the representation. This only provides a good 
		representation for the highly sampled areas of data. A potential solution for this issue
		is to downscale the number of elements selected near the boundary thus making it possible
		to represent these areas of data. It must be stressed in either report writing or to the
		user that these downscaled areas must be taken with caution as they have not been averaged
		in the same manner as the well sample areas of the dataset.
		
	3) Now supports correct assessment of confidence intervals when the median statistic is used.
	
	"""
	
	if mode is not False: 
		FutureWarning("[gu.ensemble] mode is now depreciated. Use method kwarg instead. For now we'll do the heavy lifting for you!")
		if mode is True:
			method = 'ma'
		else:
			method = mode
		
	#Sort data by ascending xdata
	mask = np.argsort(xdata, kind='mergesort')
	ydata = ydata[mask].astype(float)
	xdata = xdata[mask].astype(float)
	
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		
		if method !=  'bootstrap':
			if isinstance(bins, int):
			
				#Constants
				elems = int(np.floor(ydata.size/bins))

				if undersample is False:
					"""data_bin_x and data_bin_y are FLOAT ndarrays so element-wise operations
					CAN be performed succinctly (e.g. np.nanmean, np.nanmedian and np.nanstd)
					(i.e. THIS METHOD IS FAST)
					"""
					if usestride is False:
						xdata_average = average if isinstance(average, str) else average[0]
						
						#Calculate MidPoint
						data_bin_x = broadcast(xdata, elems, elems) if method == 'unique' else broadcast(xdata, elems, method)
						MidPoints = np.nanmedian(data_bin_x, axis=1) if xdata_average == 'median' else np.nanmean(data_bin_x, axis=1)
						
						#Calculate Boundaries
						Bin_Size = np.tile(data_bin_x.shape[1], data_bin_x.shape[0]) if slim is False else np.zeros(MidPoints.size)
						#Boundaries = [MidPoints[0], MidPoints[-1]]
						
						del data_bin_x
						
						ydata_average = average if isinstance(average, str) else average[1]
						
						#Calculate Average
						data_bin_y = broadcast(ydata, elems, elems) if method == 'unique' else broadcast(ydata, elems, method)
						Average = np.nanmedian(data_bin_y, axis=1) if ydata_average == 'median' else np.nanmean(data_bin_y, axis=1)
						
						#Calculate Confidence Limit
						if confidence_intervals == 'standard':
							SE = 1.96*np.nanstd(data_bin_y, axis=1)/np.sqrt(elems) if ydata_average == 'mean' else 1.96*(1.253*np.nanstd(data_bin_y, axis=1)/np.sqrt(elems))
						elif confidence_intervals == 'bootstrap':
							SE = boot.ci(data_bin_y.T, lambda x: np.mean(x, axis=0)).ptp(axis=0) if ydata_average == 'mean' else boot.ci(data_bin_y.T, lambda x: np.median(x, axis=0)).ptp(axis=0)
						else:
							raise ValueError("[gu.ensemble] confidence_intervals was incorrectly specified. We got %s. Available options are: 'standard', 'bootstrap'" % confidence_intervals)
					
					else:
						xdata_average = average if isinstance(average, str) else average[0]
						
						#Calculate MidPoint
						data_bin_x = stride(xdata, elems, elems) if method == 'unique' else stride(xdata, elems, method)
						MidPoints = np.nanmedian(data_bin_x, axis=1) if xdata_average == 'median' else np.nanmean(data_bin_x, axis=1)
						
						#Calculate Boundaries
						Bin_Size = np.tile(data_bin_x.shape[1], data_bin_x.shape[0]) if slim is False else np.zeros(MidPoints.size)
						#Boundaries = [MidPoints[0], MidPoints[-1]]
						
						del data_bin_x
						
						ydata_average = average if isinstance(average, str) else average[1]
						
						#Calculate Average
						data_bin_y = stride(ydata, elems, elems) if method == 'unique' else stride(ydata, elems, method)
						Average = np.nanmedian(data_bin_y, axis=1) if ydata_average == 'median' else np.nanmean(data_bin_y, axis=1)
						
						#Calculate Confidence Limit
						if confidence_intervals == 'standard':
							SE = 1.96*np.nanstd(data_bin_y, axis=1)/np.sqrt(elems) if ydata_average == 'mean' else 1.96*(1.253*np.nanstd(data_bin_y, axis=1)/np.sqrt(elems))
						elif confidence_intervals == 'bootstrap':
							SE = boot.ci(data_bin_y.T, lambda x: np.mean(x, axis=0)).ptp(axis=0) if ydata_average == 'mean' else boot.ci(data_bin_y.T, lambda x: np.median(x, axis=0)).ptp(axis=0)
						else:
							raise ValueError("[gu.ensemble] confidence_intervals was incorrectly specified. We got %s. Available options are: 'standard', 'bootstrap'" % confidence_intervals)
				
				elif undersample is True:
					"""data_bin_x and data_bin_y are OBJECT ndarrays so element-wise operations
					CANNOT be performed succinctly. Therefore, we have to calculate the mean, 
					median and std for each bin in tern (i.e. THIS METHOD IS SLOW).
					
					Boundaries are given for when the data becomes under-sampled and an equal 
					number of elements per bin does not hold true.
					"""
					
					if usestride is False:
						xdata_average = average if isinstance(average, str) else average[0]
						
						#Calculate MidPoint
						data_bin_x = broadcast(xdata, elems, elems, undersample=True) if method == 'unique' else broadcast(xdata, elems, method, undersample=True)
						MidPoints = np.nanmedian(data_bin_x, axis=1) if xdata_average == 'median' else np.nanmean(data_bin_x, axis=1)

						#Calculate Under-sampled Boundaries
						Bin_Size = np.array([len(antinan(data)) for data in data_bin_x], dtype=int) if slim is False else np.zeros(MidPoints.size)
						#MidPoints_Full = MidPoints[Bin_Size == np.max(Bin_Size)]
						#Boundaries = np.array([MidPoints_Full[0], MidPoints_Full[-1]], dtype=float)
						
						del data_bin_x
						
						ydata_average = average if isinstance(average, str) else average[1]
						
						#Calculate Average
						data_bin_y = broadcast(ydata, elems, elems, undersample=True) if method == 'unique' else broadcast(ydata, elems, method, undersample=True)
						Average = np.nanmedian(data_bin_y, axis=1) if ydata_average == 'median' else np.nanmean(data_bin_y, axis=1)
						
						#Calculate Confidence Limit
						if confidence_intervals == 'standard':
							SE = 1.96*np.nanstd(data_bin_y, axis=1)/np.sqrt(elems) if ydata_average == 'mean' else 1.96*(1.253*np.nanstd(data_bin_y, axis=1)/np.sqrt(elems))
						elif confidence_intervals == 'bootstrap':
							SE = boot.ci(data_bin_y.T, lambda x: np.mean(x, axis=0)).ptp(axis=0) if ydata_average == 'mean' else boot.ci(data_bin_y.T, lambda x: np.median(x, axis=0)).ptp(axis=0)
						else:
							raise ValueError("[gu.ensemble] confidence_intervals was incorrectly specified. We got %s. Available options are: 'standard', 'bootstrap'" % confidence_intervals)
					
					else:
						xdata_average = average if isinstance(average, str) else average[0]
						
						#Calculate MidPoint
						data_bin_x = stride(xdata, elems, elems, undersample=True) if method == 'unique' else stride(xdata, elems, method, undersample=True)
						MidPoints = np.nanmedian(data_bin_x, axis=1) if xdata_average == 'median' else np.nanmean(data_bin_x, axis=1)
						#print("MIDPOINTS DONE")
						#Calculate Under-sampled Boundaries
						Bin_Size = np.array([len(antinan(data)) for data in data_bin_x], dtype=int) if slim is False else np.zeros(MidPoints.size)
						#MidPoints_Full = MidPoints[Bin_Size == np.max(Bin_Size)]
						#Boundaries = np.array([MidPoints_Full[0], MidPoints_Full[-1]], dtype=float)
						
						del data_bin_x
						
						ydata_average = average if isinstance(average, str) else average[1]
						
						#Calculate Average
						data_bin_y = stride(ydata, elems, elems, undersample=True) if method == 'unique' else stride(ydata, elems, method, undersample=True)
						Average = np.nanmedian(data_bin_y, axis=1) if ydata_average == 'median' else np.nanmean(data_bin_y, axis=1)
						
						#Calculate Confidence Limit
						if confidence_intervals == 'standard':
							SE = 1.96*np.nanstd(data_bin_y, axis=1)/np.sqrt(elems) if ydata_average == 'mean' else 1.96*(1.253*np.nanstd(data_bin_y, axis=1)/np.sqrt(elems))
						elif confidence_intervals == 'bootstrap':
							SE = boot.ci(data_bin_y.T, lambda x: np.mean(x, axis=0)).ptp(axis=0) if ydata_average == 'mean' else boot.ci(data_bin_y.T, lambda x: np.median(x, axis=0)).ptp(axis=0)
						else:
							raise ValueError("[gu.ensemble] confidence_intervals was incorrectly specified. We got %s. Available options are: 'standard', 'bootstrap'" % confidence_intervals)
					
			else:
				#Prerequisites
				bins = np.array(bins)
				MidPoints = bins
				ydata_average = average if isinstance(average, str) else average[1]
				
				#Digitise data
				data_mask = digitize(xdata, bins)
				#Bin_Size = np.bincount(data_mask)
				max_size = np.max(np.bincount(data_mask))
				
				#Group data into rows and pad to make rows have equal length
				data_bin_y = np.array([np.pad(ydata[data_mask == i], (0, max_size - ydata[data_mask == i].size), 'constant', constant_values=np.nan) for i, bin in enumerate(bins)], dtype=np.float64)
				Bin_Size = np.array([np.sum(data_mask == i) for i, bin in enumerate(bins)], dtype=np.int32)
				
				#Average data
				if ydata_average == 'median':
					Average = np.nanmedian(data_bin_y, axis=1) 
				elif ydata_average == 'mean':
					Average = np.nanmean(data_bin_y, axis=1)
				elif ydata_average == 'sum':
					Average = np.nansum(data_bin_y, axis=1)
				SE = np.nanstd(data_bin_y, axis=1)/np.sqrt(Bin_Size) if ydata_average == 'mean' else boot.ci(data_bin_y.T, lambda x: np.median(x, axis=0)).ptp(axis=0)
								
				# sys.exit()
				
				# #Group data into specified bins

				# data_mask = searchsorted(bins, xdata)
				# max_size = np.max(np.bincount(data_mask))
				
				# MidPoints = bins
				
				# #print("Average", np.nanmin(ydata), np.nanmax(ydata), np.nanmean(ydata), np.nanmedian(ydata))
				
				# data_mask_full = [(bins[data_mask] == bin) for bin in bins]
				# Bin_Size = np.array([ydata[mask].size for mask in data_mask_full], dtype=float)
				
				# ydata_average = average if isinstance(average, str) else average[1]
				
				# #Calculate Average and Standard Error
				# print("max_size - ydata[mask].size", max_size - ydata[mask].size)
				# data_bin_y = np.array([np.pad(ydata[mask], (0, max_size - ydata[mask].size), 'constant', constant_values=np.nan) for mask in data_mask_full], dtype=float)
				# if ydata_average == 'median':
					# Average = np.nanmedian(data_bin_y, axis=1) 
				# elif ydata_average == 'mean':
					# Average = np.nanmean(data_bin_y, axis=1)
				# elif ydata_average == 'sum':
					# Average = np.nansum(data_bin_y, axis=1)
				# SE = np.nanstd(data_bin_y, axis=1)/np.sqrt(Bin_Size)
				
				#print("Average", np.nanmin(Average), np.nanmax(Average), np.nanmean(Average), np.nanmedian(Average))
		
		else:
			"""Groups the data using bootstrapping method"""
			
			#Constants
			elems = ydata.size/bins
			xdata_average = average if isinstance(average, str) else average[0]
			ydata_average = average if isinstance(average, str) else average[1]
			
			#Format x-data
			data_bin_x = stride(xdata, xdata.size, 1, undersample=True)[elems-1::elems]
			MidPoints = np.nanmedian(data_bin_x, axis=1) if xdata_average == 'median' else np.nanmean(data_bin_x, axis=1)

			#Format y-data
			data_bin_y = stride(ydata, ydata.size, 1, undersample=True)[elems-1::elems]
		
			Average = np.nanmedian(data_bin_y, axis=1) if ydata_average == 'median' else np.nanmean(data_bin_y, axis=1)
			SE = np.nanstd(data_bin_y, axis=1)/np.sqrt(ydata.size) if ydata_average == 'mean' else boot.ci(data_bin_y, lambda x: np.median(x, axis=1)).ptp(axis=1)
		
			#Force slim to be True
			slim = True
		
	if unpack is False:
		if slim is False:
			return np.vstack((MidPoints, Average, SE, Bin_Size, data_bin_y.T)).T
		else:
			return np.vstack((MidPoints, Average, SE))
	else:
		if slim is False:
			return MidPoints, Average, SE, Bin_Size, data_bin_y.T
		else:
			return MidPoints, Average, SE
	