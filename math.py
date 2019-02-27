from __future__ import absolute_import, division, print_function
import numpy as np
import itertools, warnings
import time as systime

from .manipulation import argnear

__doc__ = "Built for Python 2.7. Upgrading to Python 3.7 soon!"

def argpercentile(x, percentile):
	"""Outputs the index of the percentile value for the data.
	
	x is 1D
	percentile is between 0 and 100
	
	Reference
	---------
	https://stackoverflow.com/a/26071170"""
	
	idx = int(np.round(percentile/100 * (len(x) - 1))); 
	
	return np.argpartition(x,idx)[idx]

def cosarctan(x):
	"""Calculates the trigonometric function: Cos(Arctan(x)) by simplifying
	to (x^2+1)^-0.5. This function is meant to provide an improvement in 
	speed over long iterations"""
	
	return (x**2+1)**-0.5
	
def power(array, pow):
	"""Calculates the power of a function that can handle negative numbers
	in array. The method removes the sign of each element in array, calculates
	the power like normal and then returns the sign afterwards. This is a 
	workaround for C language issues"""

	#Convert to numpy array if not already
	array = np.asarray(array)
	
	#Get signs of all elements in the array
	sign = np.sign(array)
	
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
	
		return sign*(np.abs(array)**pow)

def interavg(arr, n, type='mean'):
	"""Sums over n elements in an array.
	
	Reference
	---------
	https://stackoverflow.com/a/15956341/8765762
	"""
	
	return getattr(np, type)(arr.reshape(-1, n), axis=1)
	
def isprime(a):
    return all(a % i for i in xrange(2, a))

def geomspace(start, stop, num=50, endpoint=True, dtype=None):
	"""Creates an array between start and stop using a log10
	spacing grid. This function is meant to copy np.geomspace
	when using numpy before 1.12"""
	
	array = np.logspace(np.log10(start) , np.log10(stop), num=num)
	
	return array.astype(dtype)
	
def get_multipler(array, m=2, mode='1D'):
	"""Determines multiplier at each step which produces a whole
	number.
	
	Parameters
	----------
	array : int
		The number you want to divide by
	m : int, optional
		Initial division number. By default this is set to 2
		as this is the lowest integer divisor
		
	Returns
	-------
	mul : numpy array
		The array sequence of multipliers at each step. This will be
		in reversed order so if we start at the lowest number (i.e.
		first number in mul) then we can multiply by the first number 
		in mul and so on
	
	"""
	
	#if number is prime then remove 1 from sequence
	if isprime(array) == True: 
		array += 1
		#print("ITS THE PRIME OF YOUR LIFE!")
	
	#if mode == '1D':
	mul = np.array([], dtype=int)
	while array > m:
		array_temp = array/m
		if array_temp == int(array_temp):
			array//=m
			mul = np.append(mul, m)
		else:
			while array_temp != int(array_temp):
				if m < 10:
					m += 1
					array_temp = array/m
				else:
					array += 1
					m = 2
					array_temp = array/m
			array//=m
			mul = np.append(mul, m)
			
	mul = np.append(mul, m)

	mul = np.array(list(reversed(mul)), dtype=int)
	
	return mul
	
def complex2float(array):
	"""Sums real and imaginary parts of an array together element-
	wise. This is used for finding the power of negative numbers
	which are calcuable but not in numpy which returns a nan.
	
	This is a workaround routine and is typically using as follows:
	
	complex2float(np.array(array, dtype=complex)**power)
	
	"""
	
	return array.real + array.imag

def disjoint_permutations_v2(subsets):		
	"""Next-gen of disjoint permutations
	
	This seems to be much quicker than the other permutation method. 
	A downside though is that the input subset needs to be an M x N 
	matrix rather than having subsubsets of different lengths.
	"""
	
	
	t0 = systime.time()
	
	#Initial conditions
	perms = []
	remi = lambda a: a != -1
	flatten = lambda l: [item for sublist in l for item in sublist]
	subsets_length = [range(len(subsets[i])) for i in xrange(len(subsets))]
	
	#Pad out subsets input to detect all combinations. Otherwise we will only detect combinations of length subset columns
	subsets_column = np.array(subsets).T
	subsets_column = np.lib.pad(subsets_column, [(len(subsets_column)-1, len(subsets_column)-1), (0,0)], 'constant', constant_values=-1)
	
	t1 = systime.time()
	#Create Iteration
	#subsets_iters = np.column_stack(([np.tile(np.repeat(subsets_length[i], len(subsets)**(len(subsets)-1-i)),  len(subsets)**i) for i in xrange(len(subsets))]))
	subsets_iters = np.column_stack(([np.tile(np.repeat(subsets_length[i], len(subsets[i])**(len(subsets)-1-i)),  len(subsets[i])**i) for i in xrange(len(subsets))]))
	t2 = systime.time()
	
	subsets_column_temp = subsets_column.copy()
	lensubsets = len(subsets_column)
	for i in subsets_iters: #iterate over lock formation (i.e. [1,1,1],[1,1,2],[1,1,3],[1,2,1] etc.)
		for j in xrange(len(subsets_iters[i])):
			subsets_column_temp[:,j] = np.roll(subsets_column[:,j], i[j], axis=0)
			for k in xrange(lensubsets):
				perms.append(tuple(subsets_column_temp[k].copy().tolist()))

	t3 = systime.time()	
		
	#Clean up: Remove -1 from all sublists
	perms = [filter(remi, x) for x in perms]
	
	t4 = systime.time()
	#Remove () tuples
	perms = [x for x in perms if x]

	t5 = systime.time()
	#Remove doubles
	perms = np.sort([x for x in set(x for x in perms)])

	print("Time: %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, Length: %.0f" % (t1-t0, t2-t1, t3-t2, t4-t3, t5-t4, systime.time()-t5, systime.time()-t0, len(perms)))
	return perms

def disjoint_permutations(subsets, andxor=False):	
	############################################################################
	"""Determine Permutations of the Distributions.
	
	Method
	------
	For a structured sequence where each subset is disjoint from one another we
	can use a cycle decomposition type approach.
	
	We determine the intersections between 2 cases of permutations:
	(1) Permutations of the subsets combined, e.g.:
			a = [1,2,3,4,5,6,7,8,9]
	(2) Permutations of the disjoint subsets, e.g.:
			b = [[1,2,3,4],[5,6,7],[7,8]]
	
	At the end we need to determine the common permutations in both datasets
	(i.e. A ^ B). This will lead to us getting all the correct combinations.
	
	This method can also be used to find the permutations with a subsets that have
	disjoint and union sets, e.g.:
			
			b = [[1,2,3,4],[5,6,7],[8,9],[8,9]]
			
	This basically overcomes the issue that we can't permutate within subsets when
	there are special cases which need to be self permutating.
	
	(see http://math.stackexchange.com/questions/1854141/permutation-of-disjoint-sets-of-a-symmetric-group
	for for information)
	
	Parameters
    ----------
    subsets : list
        A series of subsets that which have union with the set. Each subset can
		be disjoint or union with one another.
		
	andxor : boolean, optional
		Used when you want to satisfy the condition of input elements being
		completely distinct from each other (e.g. if a and b are distributions
		then,
			
								set(a)^(set(a)&set(b)),
								
		would be the elements in a that don't occur in b.
   		
		Basically this just doubles the size of the perms array
    Returns
    -------
	perms : list
		A series of permutations 
	
	"""
	############################################################################
	
	t0 = systime.time()
	
	flatten = lambda l: [item for sublist in l for item in sublist]
	subsets_flat = np.unique(flatten(subsets)).tolist()

	#Get all permutations with repeats
	res = []
	#print("np.array(subsets).shape[0]",np.array(subsets).shape[0])
	for i in xrange(np.array(subsets).shape[0]):
		for j in xrange(np.array(subsets).shape[0]):
			for k in xrange(np.array(subsets).shape[0]-1):
				res.append(itertools.product(*subsets[j:j+1+i]))
	
	#Get all permutations for all structures
	res2 = []
	for l in range(1, len(subsets_flat)+1):
		for x in itertools.combinations(subsets_flat, l):
			res2.append(x)

	#Find common elements
	if andxor == True: 
		perms = sorted(list(set(flatten(res))|set(res2)))
	else:
		perms = sorted(list(set(flatten(res))&set(res2)))
	
	print("Time: %.2f, Length: %.0f" % (systime.time()-t0, len(perms)))
	return perms
