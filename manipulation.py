import numpy as np
import sys, warnings
import time as systime

class array:
	"""Fastest method of combining data for very large dataset. Much faster than
	np.append, or x.extend/append. Boo
	
	Example
	-------
	import Gilly_Utilities as gu
	
	#Define array
	myarray = gu.array()
	
	#Add data to array
	myarray.update([1,2,3,4,5])
	
	#To access data you need to finalise it
	myarray = myarray.finalise(dtype=int)
	
	myarray
	>>> [1,2,3,4,5]
		
	Source
	------
	https://stackoverflow.com/questions/7133885/fastest-way-to-grow-a-numpy-numeric-array"""
	
	def __init__(self, data=None):
		self.data = [] if data is None else data
		self.flatten = lambda l: [item for sublist in l for item in sublist]
		
	def update(self, row):
		for r in row:
			self.data.append(r)

	def finalize(self, dtype=float, flat=True, dim='1D', method='fast', error_return=[]):
				
		if flat is True:
			if dim == '1D':
				if method is 'fast':
					#Used to flatten out a NON NxM array which often occurs when you .update different sized arrays
					#It first converts to an array (typical dtype is object) then uses hstack with the .flat command
					#to completely flatten to a 1D array and then we convert to the correct dtype (if possible).
					
					#This method is typically the fastest way of flattening all the data. There are some issues that
					#will impact the speed. Also, if no data exists when finalizing the array, an error will occur.
					try:
						return np.hstack(np.array(self.data).flat).astype(dtype)
					except IndexError:
						#If an error occurs in flattening data (e.g. self.data dimensions are 1 and cannot be combined)
						print("Finalize Error")
						method = 'ravel'
				if method is 'slow':
					#Typically used as a last ditch attempt at flattening array when no other method works
					#return np.array(self.flatten(self.data), dtype=dtype)
					return flatten(self.data, type='ndarray', dtype=dtype)
				if method is 'object':
					#Use in the special cases when using object arrays of different shapes (e.g. a.shape = (20,8), b.shape = (64,))
					return np.hstack(np.array(self.flatten(self.data)).flat).astype(dtype)
				if method is 'ravel':
					#A common method to flatten array which is very fast. Sometime faster than the hstack method. This
					#is why we use this method when an error occurs in the hstack method.
					return np.array(self.data, dtype=dtype).ravel()
				else:
					sys.exit("[Gilly_Utilities] You must select the correct array method. Choose either: ['fast', 'slow', 'ravel']")
			elif dim == '2D':
				if len(self.data) == 0: return error_return
				
				if method is 'fast':
					return np.hstack(self.data).astype(dtype)
				elif method is 'hstack':
					return np.hstack(self.data).astype(dtype)
				elif method is 'vstack':
					return np.vstack(self.data).astype(dtype)
				elif method is 'slow':
					return self.flatten(self.data)
				
		else:
			return np.array(self.data, dtype=dtype)


class np_func(list):
	def ljust(self, n, fillvalue='', dtype=float):
		"""Pads an array, list, by an integer number, n, with values
		set by the fillvalue. The output array type can also be
		set using dtype"""
		
		return np.array(self + [fillvalue] * (n - len(self)), dtype=dtype)
		
	def rjust(self, n, fillvalue='', dtype=float):
		"""Pads an array, list, by an integer number, n, with values
		set by the fillvalue. The output array type can also be
		set using dtype"""
		
		return np.array([fillvalue] * (n - len(self)) + self, dtype=dtype)
    
	def rljust(self, n, fillvalue='', dtype=float):
		"""Pads an array, list, by an integer number, n, with values
		set by the fillvalue. This will fill both left and right 
		sides of array"""
        
		ret = np.array([fillvalue] * np.int(np.ceil((n - len(self))/2)) + self + [fillvalue] * np.int(np.floor((n - len(self))/2)), dtype=dtype)
		
		if ret.size != n:
			return np.append(ret, fillvalue)
		else:
			return ret
		
class list_func(list):
	def ljust(self, n, fillvalue=''):
		"""Pads an array, list, by an integer number, n, with values
		set by the fillvalue. The output array type can also be
		set using array_type"""
		
		return self + [fillvalue] * (n - len(self))
		
	def rjust(self, n, fillvalue='', array_type=int):
		"""Pads an array, list, by an integer number, n, with values
		set by the fillvalue. The output array type can also be
		set using array_type"""
		
		return [fillvalue] * (n - len(self)) + self
    
	def rljust(self, n, fillvalue='', array_type=int):
		"""Pads an array, list, by an integer number, n, with values
		set by the fillvalue. This will fill both left and right 
		sides of array"""
        
		return [fillvalue] * np.int(np.ceil((n - len(self))/2)) + self + [fillvalue] * np.int(np.floor((n - len(self))/2))


def subset(haystack, range, invert=False, impose=None):
	"""Subsets an array, haystack, between the upper and lower values in range. There are parameters
	to invert the range selection (i.e. removes everything between range) and a parameter to use the
	information between haystack and range and place that over a new array in impose"""
	
	if invert is False:
		if impose is None:
			return haystack[(haystack >= range[0]) & (haystack < range[1])]
		else:
			if impose.size == haystack.size:
				return impose[(haystack >= range[0]) & (haystack < range[1])]
			else:
				sys.exit("[Error in subset] impose must be the same size as haystack")
				
	elif invert is True:
		if impose is None:
			return haystack[(haystack <= range[0]) | (haystack > range[1])]
		else:
			if impose.size == haystack.size:
				return impose[(haystack <= range[0]) | (haystack > range[1])]
			else:
				sys.exit("[Error in subset] impose must be the same size as haystack")
	
	else:
		sys.exit("[Error in subset] impose must be either None or an array of size haystack")	
		
def upset1d(haystack_large, haystack_small, sort=False, index=False):
	"""Increase the size of haystack_small to match the dimensions of haystack_large. This is done
	by riffling the smaller array into the larger array. Only works if both input arrays are 1
	dimensional."""
	
	haystack_large = np.array(haystack_large) if sort is False else np.sort(np.array(haystack_large), kind='mergesort')
	haystack_small = np.array(haystack_small) if sort is False else np.sort(np.array(haystack_small), kind='mergesort')
	
	
	#Determine index locations
	idx = np.searchsorted(haystack_small, haystack_large)
	
	#Correct for index if out of bounds
	idx[idx == haystack_small.size] -= 1
	
	if index is False:
		return haystack_small[idx]	
	else:
		return idx

def upset2d(haystack_large, haystack_small, needles, sort=False):
	"""Similar to upset1d but will work for 2d arrays. In this version you also need to specify needles.
	Needles is the dimensions of the large and small arrays which follows must be formatted into a loose
	2xM array where M is the number of dimensions.
	
	Example
	-------
	
	>>> haystack_large_time = np.arange(200)
	>>> haystack_large_height = np.arange(80)
	
	>>> haystack_small_time = np.arange(100)
	>>> haystack_small_height = np.arange(40)
	
	>>> needles = np.array([[haystack_large_time, haystack_large_height],[haystack_small_time, haystack_small_height]])
	
	>>> needles[0,0] == haystack_large_time
	True
	
	"""
	
	masks = [upset1d(needles[0,i], needles[1,i], sort=sort, index=True) for i in xrange(needles.shape[1])]
	
	try:
		return haystack_small[masks[1]][:,masks[0]]	
	except:
		return haystack_small[masks[0]]

def midpoint(array):
	"""Finds the mid point values of any array. The output will have a size, array.size + 1"""
	
	#Find difference between each element of array
	diff = np.diff(array)/2
	
	#Add difference between each element onto original array and taken into account boundaries
	return np.hstack((array[0]-diff[0], array[1]-diff[1], array[1:] + diff))
		
def mask(haystack, needle, invert=False, impose=False, cross_impose=False, sort=False, find_all=False, approx=False):
	"""Determines the all the needles in the haystack. Both haystack and needle
	need to be ndarrays for this too work. The advantage of this method is that 
	its FAST.
	
	Parameters
	----------
	haystack : the array you want to search
	
	needle : the search values
	
	invert : invert the select indices in haystack array
	
	impose : returned is the haystack array with found indices masked out with value of impose
	
	cross_impose : if you select all values in haystack_a but you want to use those index locations
		to populate a different array, use cross_impose instead.
		
		Example:
		
		t = np.arange(10)
		z = np.array([-2,-1,0])
		
		bob = gu.mask(t, [1,5,9], invert=True, cross_impose=z)
		bob = [0,-2,2,3,4,-1,6,7,8,0]
		
	sort : specify if you require the data to be sorted first
	
	find_all : for each needle, if there is multiple solutions then find_all == False will only
		find the first solution as the haystack array is assumed to be sorted. if find_all == True
		then all solutions are selected.
	
	approx : Search for elements in haystack that are closest to needle elements. Similar to 
		gu.near and gu.argnear
		
	Reference
	---------
	https://stackoverflow.com/a/31789252
	"""
	
	haystack = np.array(haystack) if sort is False else np.sort(np.array(haystack), kind='mergesort')
	needle = np.array([needle])

	if approx is True: return searchsorted(haystack, needle)
	
	#Determine index locations
	idx = searchsorted(haystack, needle, side='left') if find_all is False else np.in1d(haystack, needle)
	
	if invert is False:
		if find_all is False:
			mask = idx < haystack.size
			mask[mask] = haystack[idx[mask]] == needle[mask]
			idx = idx[mask]
		else:
			mask = np.ones(haystack.size, dtype=np.bool)
			mask[idx] = 0
			idx = np.arange(haystack.size)[~mask]
	else:
		mask_invert = np.ones(haystack.size, dtype=np.bool)
		mask_invert[idx] = 0
		idx = np.arange(haystack.size)[mask_invert]
	
	if impose is not False and cross_impose is False:
		haystack_copy = haystack.copy()
		haystack_copy[idx] = impose
		
		return haystack_copy
	
	if cross_impose is not False:
		if np.array(cross_impose).size != needle.size: raise ValueError("[ERROR in gu.mask] cross_impose must have equal size to needle or be a singlular item")
		
		mask_invert = np.ones(haystack.size, dtype=np.bool)
		mask_invert[idx] = 0
		idx_anti = np.arange(haystack.size)[mask_invert]

		haystack_copy = haystack.copy()
		haystack_copy[idx] = impose
		haystack_copy[idx_anti] = cross_impose
		
		return haystack_copy
	else:	
		return idx

def comask(haystack1, haystack2):
	"""similar to mask but makes sure exclusivity in both haystacks. mask might not align
	both arrays together"""
	
	idx = np.searchsorted(haystack1, haystack2)
	mask = idx < haystack1.size
	mask[mask] = haystack1[idx[mask]] == haystack2[mask]
	idx1 = idx[mask]
	
	idx = np.searchsorted(haystack2, haystack1)
	mask = idx < haystack2.size
	mask[mask] = haystack2[idx[mask]] == haystack1[mask]
	idx2 = idx[mask]
	
	return idx1, idx2
	
def overlaps(arr, invert=False):
	"""Checks for overlapping rows"""
	
	#Sort array by small difference between elements
	arr = arr[np.argsort(np.diff(arr,axis=1).ravel())]

	_todelete = []
	for i in xrange(arr.shape[0]): 
		_todelete.append(bool2int((arr[i,0] >= arr[:,0]) & (arr[i,1] <= arr[:,1]))[1:])
	
	todelete = unique(flatten(_todelete)).astype(int)
	if invert is not False: todelete = np.arange(arr.shape[0])[~np.in1d(np.arange(arr.shape[0]), todelete)]
	
	arr = np.delete(arr, todelete, axis=0)

	return arr
		
	
def flatten(array, type=None, dtype=None):
	"""Flattens the input array into 1D.
	
	type : 'list' or 'ndarray'
	dtype : when ndarray is used, specify the dtype or it will be found automatically"""
	
	if type is not None:
		if type == 'list':
			return [item for sublist in array for item in sublist]
		elif type == 'ndarray':
			if dtype is None:
				if isinstance(array, np.ndarray):
					return np.array([item for sublist in array for item in sublist], dtype=array.dtype)
				else:
					warnings.warn('\n[flatten]: A dtype must be specified if the input array is a list and you want the output array to be numpy. Numpy will estimate the data type automatically!', SyntaxWarning, stacklevel=2)
					return np.array([item for sublist in array for item in sublist])
			else:
				return np.array([item for sublist in array for item in sublist], dtype=dtype)
		else:
			UserWarning('[Error] Input array to flatten must be either a list or ndarray. Returning empty array.')
			return []
	else:
		if isinstance(array, list):
			return [item for sublist in array for item in sublist]
		elif isinstance(array, np.ndarray):
			if dtype is None:
				return np.array([item for sublist in array for item in sublist], dtype=array.dtype)
			else:
				return np.array([item for sublist in array for item in sublist], dtype=dtype)
		else:
			UserWarning('[Error] Input array to flatten must be either a list or ndarray. Returning empty array.')
			return []

def near(array,value):
	"""https://stackoverflow.com/a/2566508"""

	idx = np.abs(array-value).argmin()
	
	return array[idx]

def argnear(array, value):
	"""based on find_nearest but returns index rather than the array value"""
	
	return int(np.abs(array-value).argmin())

def argneararray(array, values):
	"""Same as argnear but can accepts both arrays and single values"""
	
	if isinstance(values, list) or isinstance(values, np.ndarray):
		return np.argmin([np.square(array-value) for value in values], axis=1)
	else:
		return argnear(array, values)
		
def round(a, clip):
	"""Will round number, a, to closes multiple of clip.
	e.g. round(5.9, 12) = 0
		 round(6.0, 12) = 12.0
		 
	Ref: https://stackoverflow.com/a/7859208/8765762
	"""
	
	return np.round(float(a) / clip) * clip	
	
def ceil(a, clip=1):
	"""Same as round but will raise the value"""

	return np.ceil(float(a) / clip) * clip	
	
def floor(a, clip=1):
	"""Same as round but will floor the value"""

	return np.floor(float(a) / clip) * clip	

def bool2int(bool, keepfalse=False, falseval=0):
	"""Converts a boolean array into the integer counterparts. So for every True value, the
	element will be returned as an array ready for indexing. 
	
	Parameters
	----------
	
	bool : ndarray, boolean
		an ndarray of true and false conditions relating to some argument."""
	
	bool = np.asarray(bool)
	
	if bool.dtype == 'bool':
		if keepfalse is False:
			return np.arange(bool.size, dtype=int)[bool]
		else:
			arr = np.arange(bool.size, dtype=int)
			try:
				arr[~bool] = falseval
			except:
				arr = arr.astype(float)
				arr[~bool] = falseval
			
			return arr
	else:
		raise ValueError("[bool2int] input array did not have a dtype bool. We got %s" % bool.dtype)

def int2bool(arr, int):
	"""Converts an integer array (typically used for indexing) to be used as a boolean array. This can
	be useful when subset an ND array.
	
	Parameters
	----------
	arr : 1D array
		a 1 dimensional array used for dimension keeping
	int : 1D array
		a 1 dimensional integer array used to select the elements of arr.
		
	N.B. Output will be a boolean array of size arr. This can be used to select the elements in arr in the
	same way as just using the int array."""
	
	arr[int] = True
	arr[~int] = False
	
	return np.invert(arr.astype(bool))
		
def strip(array, mask=""):
	"""Removes all mask values in an array.
	
	Parameters
	----------
	array : list or ndarray
		The list or ndarray that you want to remove values from
	mask : value, list or ndarray
		Input the values you want to mask e.g. mask=1, mask=[1,5]
		mask = np.array([1,5])
		
	Return
	------
	array_masked : ndarray
		An ndarray which have the masked values removed.
		
	Notes
	-----
	No matter what type of input array you use (e.g. list) the output
	array will always be a numpy ndarray. numpy arrays are easier to
	remove multiple values from arrays.
	
	Examples
	--------
	
	len(Data['FieldMill_Chilbolton_1sec_Step_File'][Data['FieldMill_Chilbolton_1sec_Step_File'] != ""])
		
	"""
	
	array = np.array(array)
	
	if isinstance(mask, np.ndarray) or isinstance(mask, list):
		ind_mask = np.array([array.T != val for val in mask]).T
		all_mask = np.array([all(tup) for tup in ind_mask], dtype=bool)
		#all_mask = [all(tup) for tup in ind_mask]
		
		try:
			return array[all_mask]
		except IndexError:
			return np.array([])
	else:
		return array[array != mask]

def contiguous(x, min=1, invalid=None, bounds=False):
	"""Locate regions of contiguous data and return an array of the same shape as input with ID.
	
	e.g. [0,0,0,0,1,1,1,1,0,0,0,0,2,2,2,2] 
	
	where 0 is not a valid data location and 1/2 are unique areas
	
	Parameters
	----------
	x : ndarray,
		A 1D numpy array of values that have already been pre-formatted. For example not valid elements
		must have a np.nan or np.inf to recognise this as an out-of-bounds areas
	min : integer, optional, default=1
		An integer that specifies the minimum length that a contiguous area can be in length
	invalid : any, optional, default=None
		A value specifying the invalid value. E.g. invalid=0 will see the value 0 as the spacing between
		any possible contiguous region. Default is None which will use np.finite to find any invalid values.
	bounds : bool, optional. default=False
		Specify if you want to return the index boundaries of each contiguous region
	"""
	
	x_mask = np.zeros(x.size, dtype=int)
	Temp_Mask = 0
	Ind_Mask = ()
	if invalid is None:
		for j in xrange(x.size):
			if np.isfinite(x[j]):
				Temp_Mask += 1
				Ind_Mask += (j,)
				x_mask[[Ind_Mask]] = Temp_Mask
			else:
				Temp_Mask = 0
				Ind_Mask = ()
	else:
		for j in xrange(x.size):
			if x[j] != invalid:
				Temp_Mask += 1
				Ind_Mask += (j,)
				x_mask[[Ind_Mask]] = Temp_Mask
			else:
				Temp_Mask = 0
				Ind_Mask = ()		
	
	#Remove contiguous lengths less than min
	x_mask[x_mask < min] = 0
			
	#Convert integers from cloud length to unique cloud number
	x_bounds = bool2int(np.diff(x_mask) != 0) + 1
	
	#Check to see if cloud exists along day boundary. If so we add in manual time boundaries
	if x_mask[0] != 0: x_bounds = np.insert(x_bounds, 0, 0)
	if x_mask[-1] != 0: x_bounds = np.append(x_bounds, x_mask.size-1); 
	x_bounds = x_bounds.reshape(int(x_bounds.size/2),2)

	#Give each cloud a unique integer for each time step 
	for cloud_id, cloud_bounds in enumerate(x_bounds,1):
		x_mask[cloud_bounds[0]:cloud_bounds[1]] = cloud_id
	
	if bounds is True:
		return x_bounds, x_mask
	elif bounds is False:
		return x_mask

def nearcontiguous(x, min=1, min_spacing=2, invalid=None, bounds=False):
	"""Find regions of data which are near contiguous. You can specify the maximum spacing between
	contiguous regions and the minimum spacing between non-contiguous regions. The maximum spacing
	between contiguous regions must be smaller than the minimum spacing between non-contiguous 
	regions.
	
	Parameters
	----------
	x : ndarray,
		A 1D numpy array of values that have already been pre-formatted. For example not valid elements
		must have a np.nan or np.inf to recognise this as an out-of-bounds areas
	min : integer, optional, default=1
		An integer that specifies the minimum length that a contiguous area can be in length
	min_spacing : integer, optional, default=2
		An integer that specifies the minimum spacing between contiguous areas
	invalid : any, optional, default=None
		A value specifying the invalid value. E.g. invalid=0 will see the value 0 as the spacing between
		any possible contiguous region. Default is None which will use np.finite to find any invalid values.
	bounds : bool, optional. default=False
		Specify if you want to return the index boundaries of each contiguous region
	
	Example
	-------
	
	>>> x = np.array([0,0,0,0,0,0,1,0,1,0,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,
				1,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,1,
				1,0,1,0,0,0,0,0,0,0])
	>>> gu.nearcontiguous(x, min=2, min_spacing=6, invalid=0)
	[[6,14],[35,43],[52,63],[84,91]]
	
	"""
	
	#Force x to be a numpy array. Create a sister variable to capture index values.
	x = np.asarray(x)
	sister = np.arange(x.size)
	
	check_contiguous = True
	while check_contiguous is True:
		#Calculate the time between tips
		time_spacing = np.diff(sister[x != invalid])
		
		#Find locations were time between tip is greater than threshold
		event_mask = time_spacing >= min_spacing
		
		#Look for contiguous areas. (i.e. if a rain event only has one tip)
		contiguous_mask = bool2int(contiguous(event_mask, min=min, invalid=invalid) != 0)[1::2]
		
		#If contiguous data found, remove from original rain data
		if contiguous_mask.size != 0:
			x[sister[x != invalid][contiguous_mask]] = 0
			check_contiguous = True
		else:
			check_contiguous = False
			event_mask = bool2int(event_mask)

	#Add value to start and end of x to capture the first and last near-contiguous events
	x_large = np.append(np.insert(x,0,np.append(1,np.zeros(min_spacing))), np.append(np.zeros(min_spacing),1))
	sister_large = np.arange(x_large.size)
	
	#Create reversed arrays to capture start and end of boundaries
	x_reverse = x_large[::-1]
	sister_reverse = sister_large[::-1]
	
	#Find the index value for the end boundary point
	endpoint = sister_large[x_large != invalid][bool2int(np.diff(sister_large[x_large != invalid]) >= min_spacing)]
	
	#Find the index value for the start boundary point
	startpoint = sister_reverse[x_reverse != invalid][bool2int(np.diff(sister_reverse[x_reverse != invalid]) <= -min_spacing)]
	
	#Join index together
	index = np.sort(np.append(startpoint, endpoint), kind='mergesort')[1:-1]
	
	#Reshape index and remove 1 from each index to compensate for the extra value added earlier
	index = index.reshape(index.size/2,2) - min_spacing - 1
	
	#Create array with same size as x with the location of the near-contiguous regions
	val = 1
	x_output = np.zeros(x.size, dtype=int)
	for region in index:
		x_output[region[0]:region[1]+1] = val
		val += 1
	
	return index, x_output
	
def indexswap(arr, ind, swaparr):
	"""Swaps index array that are linked with arr, to a different array, swaparr"""
	
	if len(ind) == 0: return np.array([])
	
	#Get values from array
	val = arr[ind]
		
	#Find closest value in swaparr
	swapind = argneararray(swaparr, val.ravel()).reshape(val.shape)
	
	return swapind
		
def searchsorted(a, v, side='left', sorter=None):
	"""Identical to the numpy version of search sorted except we take care of the
	out-of-bounds indices if a value in 'v' is outside all values in 'a'"""
	
	a = np.asarray(a)
	
	#Get indices from numpy search sorted
	ind = np.searchsorted(a, v, side=side, sorter=sorter)
	
	#Check that ind is an array. If only one element was found, np.searchsorted returns an scalar rather than an array.
	if not isinstance(ind, np.ndarray): ind = np.asarray(ind, dtype=int)
	
	#For any indices equalling the size of the input array, a, we remove 1 from the index
	if ind.size != 0: ind[ind == a.size] -= 1
	
	return ind	
	
def digitize(a, v, right=False):
	"""Identical to the numpy version of digitize except we take care of the
	out-of-bounds indices if a value in 'v' is outside all values in 'a'"""
	
	a = np.asarray(a)
	
	#Get indices from numpy search sorted
	ind = np.digitize(a, v, right=right)
	
	#Check that ind is an array. If only one element was found, np.searchsorted returns an scalar rather than an array.
	if not isinstance(ind, np.ndarray): ind = np.asarray(ind, dtype=int)
	
	#For any indices equalling the size of the input array, a, we remove 1 from the index
	if ind.size != 0: ind[ind == v.size] -= 1
	
	return ind	
		
def broadcast(array, window, step=1, undersample=False):  # Window len = L, Stride len/stepsize = S
	"""Restructures array into rows of length, L with step sizes, S.
	
	Parameters
	----------
	array : 1D array
		The array you want to restructure
	window : int
		The number of elements per row
	step : int, optional
		The number of steps you want in the column
	under-sample : boolean, optional
		Specify if you want to broadcast the array to show the under-sample bins.
	Example
	-------
	
	>>> array = np.arange(10)
	>>> gu.broadcast(array, 5, 2)
	array([[0, 1, 2, 3, 4],
       [2, 3, 4, 5, 6],
       [4, 5, 6, 7, 8]])
	
	Reference
	---------
	https://stackoverflow.com/questions/40084931/taking-subarrays-from-numpy-array-with-given-stride-stepsize/40085052#40085052
	
	Update
	------
	Reformatted output array when undersample is True. Now we maintain a perfect NxM array and pad 
	any undersampled array with np.nan. NOTE TO SELF: why did we not do this originally?
	"""
	
	if undersample is False:
		
		nrows = ((array.size-window)//step)+1
		return array[step*np.arange(nrows)[:,None] + np.arange(window)]
	
	elif undersample is True:
		#bins = np.array([array[i if i>0 else 0:i+window].tolist() for i in xrange(-window+1,array.size)])[0::step]
		bins = np.array([array[i if i>0 else 0:i+int(window)].tolist() for i in xrange(-int(window)+1,array.size)])[0::int(step)]
		
		# binslen_before = np.zeros(len(bins), dtype=int)
		# for i, data in enumerate(bins):
			# binslen_before[i] = len(data)
		
		#Pad array with np.nans for under sampled bins
		for i, data in enumerate(bins):
			bins[i] = np.pad(data, (window-len(data),0), 'constant', constant_values=np.nan)

		#flatten array to remove object identifiers
		bins = flatten(bins)
		
		#Convert to float and reshape to match desired arrangement
		bins = np.array(bins, dtype=float).reshape(int(len(bins)/window), window)
		
		# binslen_after = np.zeros(len(bins), dtype=int)
		# for i, data in enumerate(bins):
			# binslen_after[i] = len(data)
		
		# print("binslen_before", binslen_before.tolist())
		# print("binslen_after", binslen_after.tolist())
		# print("Length", len(binslen_before), binslen_after.shape)
		# sys.exit()
		return bins

def stride(array1, window, step=1, undersample=False): # Window len = L, Stride len/stepsize = S
	"""Restructures array into rows of length, L with step sizes, S. This is similar to 
	the broadcast function but numpy as_strided is much faster as it reorganise the 
	memory of the input array rather than forming a new array.
	
	DANGER
	------
	THIS FUNCTION CAN BE VOLITALE. OVERUSE OF THIS FUNCTION CAN LEAK MEMORY AND CAUSE MEMORY
	OVERFLOW ISSUES. USE AT YOUR OWN PERIL.
	
	Parameters
	----------
	array : 1D array
		The array you want to restructure
	window : int
		The number of elements per row
	step : int, optional
		The number of steps you want in the column
	under-sample : boolean, optional
		Specify if you want to broadcast the array to show the under-sample bins.
	Example
	-------
	
	>>> array = np.arange(10)
	>>> gu.broadcast(array, 5, 2)
	array([[0, 1, 2, 3, 4],
       [2, 3, 4, 5, 6],
       [4, 5, 6, 7, 8]])
	
	Reference
	---------
	https://stackoverflow.com/questions/40084931/taking-subarrays-from-numpy-array-with-given-stride-stepsize/40085052#40085052
	
	"""
	
	nrows = ((array1.size-window)//step)+1
	n = array1.strides[0]
	
	if undersample is False:
		
		return np.lib.stride_tricks.as_strided(array1, shape=(nrows,window), strides=(step*n,n))
		
	elif undersample is True:
		
		t1 = systime.time()
		
		mid = np.lib.stride_tricks.as_strided(array1, shape=(nrows,window), strides=(step*n,n))
		
		t2 = systime.time()
		
		data = array()
		for i in xrange(1,window):
			temp = mid[0].copy()
			temp[i:] = np.nan
			data.update([temp])
		
		t3 = systime.time()

		for elements in mid: data.update([elements])
		
		t4 = systime.time()
		
		for i in xrange(1,window):
			temp = mid[-1].copy()
			temp[:i] = np.nan
			data.update([temp])

		t5 = systime.time()
		
		data = data.finalize(flat=False)
		
		t6 = systime.time()
		#print("t2-t1 = %.6fs, t3-t2 = %.6fs, t4-t3 = %.6fs, t5-t4 = %.6fs, t6-t5 = %.6fs, total = %.6fs" % (t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, t6-t1))
		
		return data
	
def unique(ar, return_index=False, return_inverse=False,
			return_counts=False, axis=None):
	"""
    Find the unique elements of an array.
    Returns the sorted unique elements of an array. There are three optional
    outputs in addition to the unique elements: the indices of the input array
    that give the unique values, the indices of the unique array that
    reconstruct the input array, and the number of times each unique value
    comes up in the input array.
	
	Part of numpy version 1.14.0. The code was ripped from
	https://github.com/numpy/numpy/blob/v1.14.0/numpy/lib/arraysetops.py#L113-L248
	"""
	
	if np.asarray(ar).size == 0: return np.asarray(ar)
	
	ar = np.asanyarray(ar)
	if axis is None:
		return _unique1d(ar, return_index, return_inverse, return_counts)
	if not (-ar.ndim <= axis < ar.ndim):
		raise ValueError('Invalid axis kwarg specified for unique')

	ar = np.swapaxes(ar, axis, 0)
	orig_shape, orig_dtype = ar.shape, ar.dtype
    # Must reshape to a contiguous 2D array for this to work...
	ar = ar.reshape(orig_shape[0], -1)
	ar = np.ascontiguousarray(ar)

	if ar.dtype.char in (np.typecodes['AllInteger'] +
							np.typecodes['Datetime'] + 'S'):
        # Optimization: Creating a view of your data with a np.void data type of
        # size the number of bytes in a full row. Handles any type where items
        # have a unique binary representation, i.e. 0 is only 0, not +0 and -0.
		dtype = np.dtype((np.void, ar.dtype.itemsize * ar.shape[1]))
	else:
		dtype = [('f{i}'.format(i=i), ar.dtype) for i in range(ar.shape[1])]

	try:
		consolidated = ar.view(dtype)
	except TypeError:
        # There's no good way to do this for object arrays, etc...
		msg = 'The axis argument to unique is not supported for dtype {dt}'
		raise TypeError(msg.format(dt=ar.dtype))

	def reshape_uniq(uniq):
		uniq = uniq.view(orig_dtype)
		uniq = uniq.reshape(-1, *orig_shape[1:])
		uniq = np.swapaxes(uniq, 0, axis)
		return uniq

	output = _unique1d(consolidated, return_index,
						return_inverse, return_counts)
	if not (return_index or return_inverse or return_counts):
		return reshape_uniq(output)
	else:
		uniq = reshape_uniq(output[0])
		return (uniq,) + output[1:]

def _unique1d(ar, return_index=False, return_inverse=False,
              return_counts=False):
    """
    Find the unique elements of an array, ignoring shape.
	
	Part of numpy version 1.14.0. The code was ripped from
	https://github.com/numpy/numpy/blob/v1.14.0/numpy/lib/arraysetops.py#L250-L295
	
	This is needed for unique function to work
    """
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.intp),)
            if return_inverse:
                ret += (np.empty(0, np.intp),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret

    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret		

def unpack(*arg):
	"""Unpacks a 2D array. Input must be unpack(*arg.T) for column unpacking and
	unpack(*arg) for row unpacking."""
	
	return arg