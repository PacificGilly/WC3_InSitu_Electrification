__version__ = 1.1

import numpy as np
from datetime import datetime

from .manipulation import array, flatten
from .datetime64 import isnat, dt2hours, hours2dt

def antifinite(x, andxor=True, unpack=False, sortby=False):
	"""This will remove the np.nan and np.inf values by removing rows which contain them.
	This is a combination of antinan and antiinf and uses the numpy function np.isfinite.
	
	Parameters
	----------
	
	x : ndarray of floats, any (M x N) shape
		A matrix of data points that require sorting. Each row will be search of
		np.nan values and that row will be removed if any are found
	andxor : boolean, optional
		An optional argument used to specify whether to remove the whole row or
		leave it in tacked and just remove element. If set to True then row will
		be removed. If set to False then the input ndarray will be split up
		into array columns with different sizes.
	unpack : boolean, optional
		Determines whether to unpack the array into individual columns. useful if you 
		originally had N column arrays which you want to temporarily merge for nan
		removals.
	sortby : int, default is False
		Specify if you want to sort the data by a particular column. Default is false
		and does not sort data, but other options are to specify the column which you
		want to sort by
		
	Returns
	-------
	
	x_nonan : ndarry of floats or list of arrays dependant upon andxor
		The finalised ndarray with all nans removed will be returned unless
		andxor == False where a list of column arrays will be returned
	
	Examples
	--------
	
	>>> x = np.array([1,2,np.nan,4,5])
	>>> y = np.array([20,30,40,np.nan,np.nan])
	
	>>> xy_new = antinan(np.array([x,y])
	
	>>> xy_new
	array([[1., 2.],
	       [20., 30.]])
		   
	Sources
	-------
	http://stackoverflow.com/questions/27532503/remove-nan-row-from-x-array-and-also-the-corresponding-row-in-y
	"""
	
	#For 1D arrays
	
	if len(np.shape(x)) == 1:
		
		#Check array does not contain datetime function. If so convert to number temporarily
		x, check, dtype = dt2hours(x, epoch=np.datetime64("1900-01-01"), checkback=True)
		
		if sortby is not False:
			if check is False:
				return np.sort(x[np.isfinite(x)], kind='mergesort')
			else: 
				return hours2dt(np.sort(x[np.isfinite(x)], kind='mergesort'), epoch=np.datetime64("1900-01-01"), dtype=dtype)
		else:
			if check is False:
				return x[np.isfinite(x)]  
			else:
				return hours2dt(x[np.isfinite(x)], epoch=np.datetime64("1900-01-01 00:00:00"), dtype=dtype)
	
	#For ND arrays
	if np.shape(x)[0] > 0:
		
		#Convert datetime columns to hours
		array_has_datetime 	= np.zeros(np.shape(x)[0], dtype=bool)
		array_dtype			= np.zeros(np.shape(x)[0], dtype='S25')
		for i in xrange(np.shape(x)[0]): 
			if isinstance(x[i][0], np.datetime64) or isinstance(x[i][0], datetime):
				array_has_datetime[i] = True
				array_dtype[i] = x[i][0].dtype
			else:
				array_has_datetime[i] = False
				array_dtype[i] = x[i][0].dtype
		x = np.asarray(x, dtype=float)	
		x[x <= -10e+17] = np.nan
		
		if andxor == True:
			#Determine index for all nan values in each column
			x_mask = np.any(np.isfinite([x[0]]), axis=0)
			
			for i in xrange(1, x.shape[0]): x_mask &= np.any(np.isfinite([x[i]]), axis=0)
			
			#Remove all rows with a nan
			x_nonan = x.T[x_mask]
			
		else:
			#Split x into column arrays
			x_arrays 	 = zip(np.zeros([x.shape[0]]))
			x_nonan = zip(np.zeros([x.shape[0]]))
			for i in xrange(x.shape[0]): x_arrays[i] = x[i]
			
			#Remove nans from each column individually
			for i in xrange(x.shape[0]): x_nonan[i] = x_arrays[i][np.any(np.isfinite([x_arrays[i]]), axis=0)]
	else:
		#return np.split(x.T, x.shape[0])
		return x

	#Sort array by column if specified
	if sortby is not False: x_nonan = x_nonan[x_nonan[:,int(sortby)].argsort(kind='mergesort')]
	
	#Convert arrays that contained datetimes
	if not np.all(array_dtype == array_dtype[0]): x_nonan = x_nonan.astype(object)
	for i in xrange(x.shape[0]): x_nonan[:,i] = x_nonan[:,i].astype(np.float64).astype(array_dtype[i])
	
	if unpack == True:
		return flatten(np.split(x_nonan.T, x.shape[0]))
	else:
	
		return x_nonan			
			
def antinan(x, axis=0, andxor=True, unpack=False, sortby=False):
	"""This will remove the np.nan values by removing rows which contain them
	
	Parameters
	----------
	
	x : ndarray of floats, any (M x N) shape
		A matrix of data points that require sorting. Each row will be search of
		np.nan values and that row will be removed if any are found
	andxor : boolean, optional
		An optional argument used to specify whether to remove the whole row or
		leave it in tacked and just remove element. If set to True then row will
		be removed. If set to False then the input ndarray will be split up
		into array columns with different sizes.
	unpack : boolean, optional
		Determines whether to unpack the array into individual columns. useful if you 
		originally had N column arrays which you want to temporarily merge for nan
		removals.
	sortby : int, default is False
		Specify if you want to sort the data by a particular column. Default is false
		and does not sort data, but other options are to specify the column which you
		want to sort by
		
	Returns
	-------
	
	x_nonan : ndarry of floats or list of arrays dependant upon andxor
		The finalised ndarray with all nans removed will be returned unless
		andxor == False where a list of column arrays will be returned
	
	Examples
	--------
	
	>>> x = np.array([1,2,np.nan,4,5])
	>>> y = np.array([20,30,40,np.nan,np.nan])
	
	>>> xy_new = antinan(np.array([x,y])
	
	>>> xy_new
	array([[1., 2.],
	       [20., 30.]])
		   
	Sources
	-------
	http://stackoverflow.com/questions/27532503/remove-nan-row-from-x-array-and-also-the-corresponding-row-in-y
	"""
	
	#For 1D arrays
	x = np.asarray(x)
	if len(x.shape) == 1:
		if sortby is not False:
			return np.sort(x[~np.isnan(x)], kind='mergesort')
		else:
			return x[~np.isnan(x)]
	
	#For ND arrays
	if x.shape[-1] > 0:
		if andxor == True:
			#Determine index for all nan values in each column
			x_mask = ~np.any(np.isnan([x[0]]), axis=axis)
			
			for i in xrange(1, x.shape[0]): x_mask &= ~np.any(np.isnan([x[i]]), axis=axis)
			
			#Remove all rows with a nan
			x_nonan = x.T[x_mask]
			
		else:
			#Split x into column arrays
			x_arrays 	 = zip(np.zeros([x.shape[0]]))
			x_nonan = zip(np.zeros([x.shape[0]]))
			for i in xrange(x.shape[0]): x_arrays[i] = x[i]
			
			#Remove nans from each column individually
			for i in xrange(x.shape[0]): x_nonan[i] = x_arrays[i][~np.any(np.isnan([x_arrays[i]]), axis=axis)]
	else:
		#return np.split(x.T, x.shape[0])
		return x

	#Sort array by column if specified
	if sortby is not False: x_nonan = x_nonan[x_nonan[:,int(sortby)].argsort(kind='mergesort')]
	
	if unpack == True:
		return flatten(np.split(x_nonan.T, x.shape[0]), type='ndarray', dtype=x.dtype)
	else:
		return x_nonan

def antinat(x, andxor=True, unpack=False, sortby=False):
	"""This will remove the np.datetime64('nat') values by removing rows which contain them.
	
	This is the same as antinan but will remove only infs.
	
	Parameters
	----------
	
	x : ndarray of floats, any (M x N) shape
		A matrix of data points that require sorting. Each row will be search of
		np.inf values and that row will be removed if any are found
	andxor : boolean, optional
		An optional argument used to specify whether to remove the whole row or
		leave it intacked and just remove element. If set to True then row will
		be removed. If set to False then the input ndarray will be split up
		into array columns with different sizes.
	unpack : boolean, optional
		Determines whether to unpack the array into individual columns. useful if you 
		originally had N column arrays which you want to temporarily merge for inf
		removals.
	sortby : int, default is False
		Specify if you want to sort the data by a particular column. Default is false
		and does not sort data, but other options are to specify the column which you
		want to sort by
		
	Returns
	-------
	
	x_nonan : ndarry of floats or list of arrays dependant upon andxor
		The finalised ndarray with all infs removed will be returned unless
		andxor == False where a list of column arrays will be returned
			   
	Sources
	-------
	http://stackoverflow.com/questions/27532503/remove-nan-row-from-x-array-and-also-the-corresponding-row-in-y
	"""
	
	#For 1D arrays
	x = np.asarray(x)
	if len(x.shape) == 1:
		if sortby is not False:
			return np.sort(x[~isnat(x)], kind='mergesort')
		else:
			return x[~isnat(x)]
	
	#For ND arrays
	if x.shape[-1] > 0:
		if andxor == True:
			#Determine index for all nat values in each column
			x_mask = ~np.any(isnat([x[0]]), axis=0)
			
			for i in xrange(1, x.shape[0]): x_mask &= ~np.any(isnat([x[i]]), axis=0)
			
			#Remove all rows with a nat
			x_noinf = x.T[x_mask]
			
		else:
			#Split x into column arrays
			x_arrays 	 = zip(np.zeros([x.shape[0]]))
			x_noinf = zip(np.zeros([x.shape[0]]))
			for i in xrange(x.shape[0]): x_arrays[i] = x[i]
			
			#Remove nans from each column individually
			for i in xrange(x.shape[0]): x_noinf[i] = x_arrays[i][~np.any(isnat([x_arrays[i]]), axis=0)]
	else:
		#return np.split(x.T, x.shape[0])
		return x

	#Sort array by column if specified
	if sortby is not False: x_noinf = x_noinf[x_noinf[:,int(sortby)].argsort(kind='mergesort')]
	
	if unpack == True:
		return flatten(np.split(x_noinf.T, x.shape[0]))
	else:
		return x_noinf			
		
def antiinf(x, andxor=True, unpack=False, sortby=False):
	"""This will remove the np.inf values by removing rows which contain them.
	
	This is the same as antinan but will remove only infs.
	
	Parameters
	----------
	
	x : ndarray of floats, any (M x N) shape
		A matrix of data points that require sorting. Each row will be search of
		np.inf values and that row will be removed if any are found
	andxor : boolean, optional
		An optional argument used to specify whether to remove the whole row or
		leave it intacked and just remove element. If set to True then row will
		be removed. If set to False then the input ndarray will be split up
		into array columns with different sizes.
	unpack : boolean, optional
		Determines whether to unpack the array into individual columns. useful if you 
		originally had N column arrays which you want to temporarily merge for inf
		removals.
	sortby : int, default is False
		Specify if you want to sort the data by a particular column. Default is false
		and does not sort data, but other options are to specify the column which you
		want to sort by
		
	Returns
	-------
	
	x_nonan : ndarry of floats or list of arrays dependant upon andxor
		The finalised ndarray with all infs removed will be returned unless
		andxor == False where a list of column arrays will be returned
	
	Examples
	--------
	
	>>> x = np.array([1,2,np.inf,4,5])
	>>> y = np.array([20,30,40,np.inf,np.inf])
	
	>>> xy_new = antiinf(np.array([x,y])
	
	>>> xy_new
	array([[1., 2.],
	       [20., 30.]])
		   
	Sources
	-------
	http://stackoverflow.com/questions/27532503/remove-nan-row-from-x-array-and-also-the-corresponding-row-in-y
	"""
	
	#For 1D arrays
	x = np.asarray(x)
	if len(x.shape) == 1:
		if sortby is not False:
			return np.sort(x[~np.isinf(x)], kind='mergesort')
		else:
			return x[~np.isinf(x)]
	
	#For ND arrays
	if x.shape[-1] > 0:
		if andxor == True:
			#Determine index for all inf values in each column
			x_mask = ~np.any(np.isinf([x[0]]), axis=0)
			
			for i in xrange(1, x.shape[0]): x_mask &= ~np.any(np.isinf([x[i]]), axis=0)
			
			#Remove all rows with a inf
			x_noinf = x.T[x_mask]
			
		else:
			#Split x into column arrays
			x_arrays 	 = zip(np.zeros([x.shape[0]]))
			x_noinf = zip(np.zeros([x.shape[0]]))
			for i in xrange(x.shape[0]): x_arrays[i] = x[i]
			
			#Remove nans from each column individually
			for i in xrange(x.shape[0]): x_noinf[i] = x_arrays[i][~np.any(np.isinf([x_arrays[i]]), axis=0)]
	else:
		#return np.split(x.T, x.shape[0])
		return x

	#Sort array by column if specified
	if sortby is not False: x_noinf = x_noinf[x_noinf[:,int(sortby)].argsort(kind='mergesort')]
	
	if unpack == True:
		return flatten(np.split(x_noinf.T, x.shape[0]))
	else:
		return x_noinf	

def antival(x, val, andxor=True, unpack=False, sortby=False):
	"""This will remove the np.inf values by removing rows which contain them.
	
	This is the same as antinan but will remove only infs.
	
	Parameters
	----------
	
	x : ndarray of floats, any (M x N) shape
		A matrix of data points that require sorting. Each row will be search of
		np.inf values and that row will be removed if any are found
	andxor : boolean, optional
		An optional argument used to specify whether to remove the whole row or
		leave it intacked and just remove element. If set to True then row will
		be removed. If set to False then the input ndarray will be split up
		into array columns with different sizes.
	unpack : boolean, optional
		Determines whether to unpack the array into individual columns. useful if you 
		originally had N column arrays which you want to temporarily merge for inf
		removals.
	sortby : int, default is False
		Specify if you want to sort the data by a particular column. Default is false
		and does not sort data, but other options are to specify the column which you
		want to sort by
		
	Returns
	-------
	
	x_nonan : ndarry of floats or list of arrays dependant upon andxor
		The finalised ndarray with all infs removed will be returned unless
		andxor == False where a list of column arrays will be returned
	
	Examples
	--------
	
	>>> x = np.array([1,2,np.inf,4,5])
	>>> y = np.array([20,30,40,np.inf,np.inf])
	
	>>> xy_new = antiinf(np.array([x,y])
	
	>>> xy_new
	array([[1., 2.],
	       [20., 30.]])
		   
	Sources
	-------
	http://stackoverflow.com/questions/27532503/remove-nan-row-from-x-array-and-also-the-corresponding-row-in-y
	"""

	#For 1D arrays
	x = np.asarray(x)
	if len(x.shape) == 1:
		if sortby is not False:
			return np.sort(x[~(x == val)], kind='mergesort')
		else:
			return x[~(x == val)]
	
	#For ND arrays
	if x.shape[-1] > 0:
		x_noval = antinan(np.where(np.logical_not(x == val), x, np.nan))
	else:
		return x

	#Sort array by column if specified
	if sortby is not False: x_noval = x_noval[x_noval[:,int(sortby)].argsort(kind='mergesort')]
	
	if unpack == True:
		return flatten(np.split(x_noval.T, x.shape[0]))
	else:
		return x_noval	

def nan_helper(y):
	"""Helper to handle indices and logical indices of NaNs.

	Input:
		- y, 1d numpy array with possible NaNs
	Output:
		- nans, logical indices of NaNs
		- index, a function, with signature indices= index(logical_indices),
		  to convert logical indices of NaNs to 'equivalent' indices
	Example:
		>>> # linear interpolation of NaNs
		>>> nans, x= nan_helper(y)
		>>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
	
	https://stackoverflow.com/a/6520696
	
	"""

	return np.isnan(y), lambda z: z.nonzero()[0]	
	