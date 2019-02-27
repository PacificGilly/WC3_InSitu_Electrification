import numpy as np
import sys, warnings

from .datetime64 import dt2hours
from .manipulation import bool2int, argnear

__version__ = 1.0

def rain_rate(time, counts_or_tips, tip_bucket_size=None, drop_counter_size=None, drop_size=None, mask=True):
	"""This universal function can convert both tipping bucket, drop counter and
	disdrometer rain gauge data into a rain rate.
	
	Parameters
	----------
	time : list or array
		The time values in either hour fraction, datetime or np.datetime64 format.
		If any other float or integer specification of time is used (e.g. day 
		fraction or second) the outputted rain rate will need to corrected. The
		time is expected to be sorted.
	counts_or_tips : list or array
		The number of counts or tip identifier with the same length as time. For
		tips the input list/array needs to be 0 for no tip and 1 for tip measured.
	tip_bucket_size : float, optional
		The size of the tipping bucket per tip in mm. Default = 0.2 mm
	drop_counter_size : float, optional
		The size of each drop in a drop counter. Default is 0.0033 mm
	drop_size : list or array, optional
		Used to specify the drop sizes of the disdrometer. This must be a 1D array.
	
	Output
	------
	Time_Tip : numpy array
		The times values when a tip was detected.
	Time_RR : numpy array
		The median time between tips. This represents the central time when the rain
		occurred and naturally has the same number of elements as Rain_RR.
	Rain_RR : numpy array
		The rain rate measured between Time_Tip in mm/hr.
	
	References
	----------
	Disdrometer : Article
		Islam T. et al. (2012). A Joss-Waldvogel disdrometer derived rainfall 
		estimation study by collocated tipping bucket and rapid response rain gauges. 
		Atmos. Sci. Let. 13, pp 139-150
		
	"""
	
	if time is None: 
		#print(11111111111111111)
		return None, None, None
	if not isinstance(time, np.ndarray) or isinstance(time, list): 
		#print(22222222222222222)
		return None, None, None
	if len(time) < 10: 
		#print(33333333333333333)
		return None, None, None
	
	#Force time to be in hours
	time = dt2hours(time, time[0])
	counts_or_tips = np.asarray(counts_or_tips)
	
	if counts_or_tips.ndim == 1:
		if np.max(counts_or_tips) == 1 or tip_bucket_size is not None:
			"""TIPPING BUCKET"""
		
			#If the data follows a tipping bucket structure but tip_bucket_size was not specified, then warn user and set to standard 0.2 mm
			if tip_bucket_size is None: 
				warnings.warn('\n[gu.rainrate]: the input data has the same structure as a TIPPING BUCKET but tip_bucket_size was not specified. We will assume a standard bucket size of 0.2 mm!', SyntaxWarning, stacklevel=2)
				tip_bucket_size = 0.2
		
			Rain_Mask = counts_or_tips > 0
			Time_Tip = time[Rain_Mask]
			
			Rain_RR = tip_bucket_size/np.diff(Time_Tip)
			Time_RR = 0.5*np.diff(Time_Tip)+Time_Tip[:-1]
			
			return Time_Tip, Time_RR, Rain_RR
			
		if np.sum(counts_or_tips > 0) > 10:
			"""DROP COUNTER"""
			
			if drop_counter_size is None: 
				warnings.warn('\n[gu.rainrate]: the input data has the same structure as a DROP COUNTER but drop_counter_size was not specified. We will assume a standard bucket size of 0.0033 mm!', SyntaxWarning, stacklevel=2)
				drop_counter_size = 0.0033
			
			dt = np.nanmean(np.diff(time))
			
			Rain_Mask = counts_or_tips > 0
			Time_Counts = 0.5*np.diff(time[Rain_Mask])+time[Rain_Mask][:-1] 
			
			
			try:
				Time_Counts = np.append(Time_Counts, Time_Counts[0]-dt)
				Time_Counts = np.append(Time_Counts, Time_Counts[-1]+dt)
			except:
				print("ERROR")
				print("Time_Counts", len(Time_Counts))
				print("time", len(time))
				print("Rain_Mask", Rain_Mask, np.sum(Rain_Mask))
				print("counts_or_tips", counts_or_tips[counts_or_tips != 0])
				print("counts_or_tips", type(counts_or_tips), counts_or_tips.dtype)
				
			Mag_Counts = counts_or_tips[Rain_Mask]
			
			Rain_RR = (drop_counter_size*Mag_Counts)/dt
			Time_RR = time[Rain_Mask]
			
			return Time_Counts, Time_RR, Rain_RR
			
		else:
			#print(44444444444444444)
			return None, None, None
			
	else:
		if mask is True:
			if np.sum(np.sum(counts_or_tips, axis=1) > 20) > 10:
				"""DISDROMETER"""
				
				"""Calculation method based on Islam T. et al. (2012)"""
				
				#Constants
				A = 0.005 #m^2
				dt = np.round(np.nanmean(np.diff(time)*3600)) #~ 10s
				
				Rain_Mask = np.sum(counts_or_tips, axis=1) > 20
				Time_Disd = 0.5*np.diff(time[Rain_Mask])+time[Rain_Mask][:-1]; Time_Disd = np.append(Time_Disd, Time_Disd[0]-dt/3600); Time_Disd = np.append(Time_Disd, Time_Disd[-1]+dt/3600)
				Mag_Disd = counts_or_tips[Rain_Mask]
						
				#Static Variables (i.e. They DON'T change over each time step)
				v = 3.78*drop_size**0.67
				D = drop_size.copy()**3.67
				dD = 2*np.diff(drop_size); dD = np.append(dD, dD[-1])
				
				#Dynamic Variables (i.e. They DO change over each time step)
				Nm = Mag_Disd/(A*dt*v*dD)
				Rain_RR = (3.78*np.pi*np.nansum(D*Nm*dD, axis=1)/6)/(3600/dt)
				Time_RR = time[Rain_Mask]
				
				#print("Time_Disd", "Time_RR", "Rain_RR")
				#print(Time_Disd.shape, Time_RR.shape, Rain_RR.shape)
				
				return Time_Disd, Time_RR, Rain_RR
				
			else:
				#print(55555555555555555)
				return None, None, None
		else:
			"""DISDROMETER"""
				
			"""Calculation method based on Islam T. et al. (2012)"""
			
			#Constants
			A = 0.005 #m^2
			dt = np.round(np.nanmean(np.diff(time)*3600)) #~ 10s
			
			Time_Disd = 0.5*np.diff(time)+time[:-1]; Time_Disd = np.append(Time_Disd, Time_Disd[0]-dt/3600); Time_Disd = np.append(Time_Disd, Time_Disd[-1]+dt/3600)
			Mag_Disd = counts_or_tips
					
			#Static Variables (i.e. They DON'T change over each time step)
			v = 3.78*drop_size**0.67
			D = drop_size.copy()**3.67
			dD = 2*np.diff(drop_size); dD = np.append(dD, dD[-1])
			
			#Dynamic Variables (i.e. They DO change over each time step)
			Nm = Mag_Disd/(A*dt*v*dD)
			Rain_RR = (3.78*np.pi*np.nansum(D*Nm*dD, axis=1)/6)/(3600/dt)
			Time_RR = time
			
			#print("Time_Disd", "Time_RR", "Rain_RR")
			#print(Time_Disd.shape, Time_RR.shape, Rain_RR.shape)
			
			return Time_Disd, Time_RR, Rain_RR
		
def Drops2RR(drop_time, drop_count, drop_size):
	"""Converts a disdrometer drop counter to a rain rate
	
	Paper
	-----
	Islam T. et al. (2012). A Joss-Waldvogel disdrometer derived
	rainfall estimation study by collocated tipping bucket and
	rapid response rain gauges. Atmos. Sci. Let. 13, pp 139-150 
	
	"""
	
	#Intialise Equations and Constants
	Nm = lambda n, A, t, v, D: n/(A*t*v*D)
	v = lambda D: 3.78*D**0.67
	RR_Dis_Equ = lambda D, Nm, dD: 3.78*(np.pi/6)*np.sum([D[m]**3.67*Nm[m]*dD[m] for m in xrange(len(D))])
	A = 0.005 #m^2
	dt = 10 #s
	
	#Solve differential drop sizes
	dD = np.zeros(len(drop_size))
	dD[0] = 2*(drop_size[1]-drop_size[0])
	for o in xrange(1, len(drop_size)-1):
		dD[o] = (drop_size[o+1]-drop_size[o])+(drop_size[o]-drop_size[o-1])
	dD[-1] = 2*(drop_size[-1]-drop_size[-2])
	
	#Calculate RR
	RR_Dis = np.zeros(len(drop_time))
	for m in xrange(len(drop_time)):
		RR_Dis[m] = RR_Dis_Equ(drop_size, Nm(drop_count[m], A, dt, v(drop_size), dD), dD)/(3600/dt)
	
	drop_time = drop_time[RR_Dis>0]
	RR_Dis = RR_Dis[RR_Dis>0]
			
	return drop_time, RR_Dis		

def Cloud_Identifer(Time, Sg, Sd):
	"""Classifies the type of cloud that is overhead using solar radiation measurements,
	
	Parameters
	----------
	Time : 1D array of list
		The time for each measurement of Sg and Sd. The time needs to be in fractional hour
	Sg : 1D array or list
		The global solar (shortwave) irradiance
	Sd : 1D array or list
		The global diffuse (longwave) irradiance
		
	Output
	------
	Cloud_Time : array or float
		A float of the central time for each 15 minute period. If there is more data than
		15 minutes then the output is a float array.
	Cloud_Type : array or int
		An integer specifying the cloud type per 15 minutes of data. If there is more data
		than 15 minutes then the output is an integer array.
	
	"""
	
	#Ensure inputs are numpy arrays
	if isinstance(Time[0], np.datetime64):
		Date = Time[0]
		Time = dt2hours(Time, Date)
		
		#print(Date)
		#print(Time)
		#print(Time.size)
		
		Datetime_Found = 'datetime64'
	elif isinstance(Time[0], datetime):
		Date = toDateOnly([Time[-10]])[0]
		Time = np.array([toHourFraction(x) for x in Time], dtype=float)
		Datetime_Found = 'datetime'
	
	else:
		Time = np.array(Time, dtype=float)
		Datetime_Found = 'none'
	Sg = np.array(Sg, dtype=float)
	Sd = np.array(Sd, dtype=float)
	
	#Sort data
	sort_index = np.argsort(Time, kind='mergesort')
	Time = Time[sort_index]
	Sg = Sg[sort_index]
	Sd = Sd[sort_index]
	
	#Determine if Time is shorter than 15 minutes
	if Time[-1] - Time[0] < 0.25: 
		warnings.warn('[Cloud_Identifer]: The identification of cloud type requires 15 minutes or more data. Only %.2f minutes was found!' % ((Time[-1] - Time[0])*60), RuntimeWarning, stacklevel=2)
		return None, None
	
	#Determine if Time is exactly 15 minutes
	elif Time[-1] - Time[0] == 0.25:
		
		Diffuse_Fraction = Sd/Sg

		if np.mean(Diffuse_Fraction) >= 0.9: 	#Overcast
			Cloud_Type = 2
		elif np.mean(Diffuse_Fraction) <= 0.3: 	#Clear
			Cloud_Type = 1
		elif np.std(Diffuse_Fraction) <= 0.05: 	#Cumuliform
			Cloud_Type = 4
		elif np.std(Diffuse_Fraction) >= 0.1: 	#Stratiform
			Cloud_Type = 3
		else:									#Unclassified
			Cloud_Type = 5
		
		Cloud_Time = np.nanmean(Time)
	
	#Determine if Time is longer than 15 minutes
	elif Time[-1] - Time[0] > 0.25:
		
		#Calculate Cloud_Time
		Cloud_Num = int((Time[-1] - Time[0])/0.25)+1
		Cloud_Time = np.array([Time[0] + 0.125 + Time_Step for Time_Step in np.linspace(0, Cloud_Num*0.25, Cloud_Num, endpoint=False)], dtype=float)
		
		#Calculate Cloud_Type for each 15 minute subset
		Cloud_Type = np.zeros(Cloud_Num)
		for i, cloud_time in enumerate(Cloud_Time):
			Mask = ((cloud_time-0.125) <= Time) & ((cloud_time+0.125) > Time)
		
			Diffuse_Fraction = Sd[Mask]/Sg[Mask]
			
			if np.mean(Diffuse_Fraction) >= 0.9: 	#Overcast
				Cloud_Type[i] = 2
			elif np.mean(Diffuse_Fraction) <= 0.3: 	#Clear
				Cloud_Type[i] = 1
			elif np.std(Diffuse_Fraction) <= 0.05: 	#Cumuliform
				Cloud_Type[i] = 4
			elif np.std(Diffuse_Fraction) >= 0.1: 	#Stratiform
				Cloud_Type[i] = 3
			else:									#Unclassified
				Cloud_Type[i] = 5

		#If input time was in datetime, we now convert back to datetime
		if Datetime_Found == 'datetime': Cloud_Time = np.array([roundTime(Date + timedelta(hours=hour_frac), 1) for hour_frac in Cloud_Time], dtype=object)
		if Datetime_Found == 'datetime64': Cloud_Time = hours2dt(Cloud_Time, Date)
				
	return Cloud_Time, Cloud_Type
	
def degK2degC(Kelvin):
	"""Converts either an array, list or value from kelvin to Celsius"""
	
	if isinstance(Kelvin, list):
		return (np.asarray(Kelvin, dtype=np.float64)-273.15).tolist()
	elif isinstance(Kelvin, np.ndarray):
		return Kelvin-273.15
	elif isinstance(Kelvin, float):
		return Kelvin-273.15
	else:
		sys.exit("[ERROR in degK2degC] Kelvin needs to be either a list, ndarray or float")
		
def radarbounds(Radar_Time, Radar_Height, Radar_Mag, Cloud_TimeBounds, CloudNet_Base):
	"""Determines the cloud boundaries within a radar image. Requires all 3 dimensions of
	a radar image (e.g. time, height, magnitude), the index values denoting the boundaries
	of each cloud (in a 2D array) and the cloud base height array (1D) and must have the
	same dimensions as Radar_Time (i.e. Radar_Time.size == CloudNet_Base.size).
	
	Parameters
	----------
	Radar_Mag : ndarray or multiple ndarray
		Either a single ndarray of dimensions (height, time) or multiple ndarrays with the same
		dimensions. E.g. Radar_Mag = (Radar, Radar) when calling radar-bounds. Each radar group
		with a tuple must have equal height and time dimensions."""
	
	#Create height mask to remove lowest band of clouds
	Radar_Height_Mask = (Radar_Height > 0.3).ravel()
	Radar_Height_Subset = Radar_Height[Radar_Height_Mask].ravel()
	
	if isinstance(Radar_Mag, tuple):
		Radar_Mag_Main = Radar_Mag[0]
	
	Cloud_HeightBounds = zip(np.zeros(Cloud_TimeBounds.shape[0]))
	for cloud_id, cloud_bounds in enumerate(Cloud_TimeBounds):
		
		#[Step 1] Subset Radar Reflectivity
		Cloud_Mag = Radar_Mag_Main[Radar_Height_Mask,cloud_bounds[0]:cloud_bounds[1]]
		
		#[Step 2] Determine number of nan's for each height layer
		#ClearSkies = np.sum(np.isnan(np.where(Cloud_Mag > -30,Cloud_Mag,np.nan)),axis=1)
		ClearSkies = np.sum(np.isnan(Cloud_Mag),axis=1)
		
		#[Step 3] Find lowest cloud base layer
		CloudBaseMin = np.min(CloudNet_Base[cloud_bounds[0]:cloud_bounds[1]])
		CloudBaseMax = np.max(CloudNet_Base[cloud_bounds[0]:cloud_bounds[1]])
		CloudBase90	 = np.nanpercentile(CloudNet_Base[cloud_bounds[0]:cloud_bounds[1]], 10)
		
		#[Step 4] Find Lowest Cloud Top Layer
		mask = ClearSkies == np.diff(cloud_bounds)
		CloudTop = Radar_Height_Subset[mask]
		CloudTop = CloudTop[CloudTop > CloudBase90][0]

		#[Step 5] Remove all values below cloud base at each time-step
		Base = CloudNet_Base[cloud_bounds[0]:cloud_bounds[1]]
		Cloud_Time = np.arange(cloud_bounds[0], cloud_bounds[1], dtype=int)
		Base_Mask = np.searchsorted(Radar_Height.ravel(), Base)
		for lower_limit, timestamp in zip(Base_Mask, Cloud_Time):
			if isinstance(Radar_Mag, tuple):
				for radar_type in xrange(len(Radar_Mag)):
					Radar_Mag[radar_type][:lower_limit, timestamp] = np.nan
			else:
				Radar_Mag[:lower_limit, timestamp] = np.nan
			
		#[Step 6] Remove all values above cloud top at each time-step
		Top_Mask = np.tile(np.searchsorted(Radar_Height.ravel(), CloudTop), Cloud_Time.size)
		for upper_limit, timestamp in zip(Top_Mask, Cloud_Time):
			if isinstance(Radar_Mag, tuple):
				for radar_type in xrange(len(Radar_Mag)):
					Radar_Mag[radar_type][upper_limit:, timestamp] = np.nan
			else:
				Radar_Mag[upper_limit:, timestamp] = np.nan
		
		#Add cloud base and cloud top heights to array
		Cloud_HeightBounds[cloud_id] = [argnear(Radar_Height.ravel(),CloudBaseMin), 
			argnear(Radar_Height.ravel(),CloudBaseMax), 
			argnear(Radar_Height.ravel(),CloudTop)]
	
	Cloud_HeightBounds = np.array(Cloud_HeightBounds, dtype=int)
	
	#print("Cloud_HeightBounds", Cloud_HeightBounds)
	
	if isinstance(Radar_Mag, tuple):
		output = (Cloud_HeightBounds,)
		for i in xrange(len(Radar_Mag)): output += (Radar_Mag[i],)
	else:
		output =  (Cloud_HeightBounds, Radar_Mag)
	
	return output
	