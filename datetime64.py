import numpy as np
#import pandas as pd
import warnings, sys
from datetime import date, datetime, timedelta
from datetime import time as dttime

def time32(when='now', dtype='str'):
	"""Output the time in different formats
	
	Parameters
	----------
	when : str, optional
		Specify the time you want to configure. Currently only 'now' is an
		available option.
	dtype : str, optional
		Specify the data type you want the time to be given in. The current
		options are 'str', 'datetime' and 'datetime64' which corresponds to
		a string, python datetime and numpy datetime object output.
		
	Output
	------
	time : dtype
		Returns the time in various types depending on the input data type.
		
		str : Return hh:mm:ss string representation of the current time
		
		datetime : Return the python datetime object
		
		datetime64 : Return the numpy datetime object
		
	"""
	
	if when == 'now':
		t = datetime.now()
	
	if dtype == 'str':
		return t.time().strftime("%H:%M:%S")
	elif dtype == 'datetime':
		return t.time()
	elif dtype == 'datetime64':
		return np.datetime64(t).astype('datetime64[s]')
	else:
		raise warnings.warn('[gu.time]: The specified dtype must be either "str", "datetime" or "datetime64". We got %s' % (dtype), UserWarning, stacklevel=2)

def datetime32(when='now', dtype='str'):
	"""Output the date and time in different formats
	
	Parameters
	----------
	when : str, optional
		Specify the time you want to configure. Currently only 'now' is an
		available option.
	dtype : str, optional
		Specify the data type you want the time to be given in. The current
		options are 'str', 'datetime' and 'datetime64' which corresponds to
		a string, python datetime and numpy datetime object output.
		
	Output
	------
	time : dtype
		Returns the time in various types depending on the input data type.
		
		str : Return hh:mm:ss string representation of the current time
		
		datetime : Return the python datetime object
		
		datetime64 : Return the numpy datetime object
		
	"""
	
	if when == 'now':
		t = datetime.now()
	
	if dtype == 'str':
		return t.strftime("%Y/%m/%d %H:%M:%S")
	elif dtype == 'datetime':
		return t
	elif dtype == 'datetime64':
		return np.datetime64(t).astype('datetime64[s]')
	else:
		raise warnings.warn('[gu.time]: The specified dtype must be either "str", "datetime" or "datetime64". We got %s' % (dtype), UserWarning, stacklevel=2)
			
def Excel_to_Python_Date(excel_time, expand=None, format=None, strip=True):
	"""Converts Excel formatted time into a python format
	
	Parameters
    ----------
    time : numpy array
        1 dimensional time series array attached to data stream to determine the timing a step
		was detected
	expand : boolean, optional
		Outputs an expanded array of time variants over the broad datetime format. This is
		useful when requiring year day or fraction hours of the day
	format : str, optional
		if the excel string is in a different format then you can specify this here
	strip : boolean, optional
		strips the time component from datetime objects while leaving them intact
	
    Returns
    -------
    Python_Datetime : numpy array, float
        The date and time of each timestamp in datetime python format
	Python_Date : numpy array, float
        The date of each timestamp in datetime python format
	Python_Year : numpy array, int, optional
		The year at each time step
	Python_YD : numpy array, int, optional
		The day of the year at each timestep
	Python_Hour : numpy array, float, optional
		The fraction hour of each timestep assuming time has been given.
	"""

	Python_Datetime = np.zeros(len(excel_time), dtype=object)
	if format is not None: #Bespoke method
		for i in xrange(len(excel_time)):
			try:
				Python_Datetime[i] = datetime.strptime(excel_time[i], format)
			except TypeError:
				continue
	else: #Standard Method
		Python_Datetime = np.array([datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in excel_time], dtype=object)

	Python_Datetime	= Python_Datetime[Python_Datetime != 0]
	
	if expand is not None:
		#Convert Python_Datetime to decimal hour format
		Python_Date = np.zeros_like(Python_Datetime)
		Python_Year = np.zeros_like(Python_Datetime)
		Python_YD = np.zeros_like(Python_Datetime)
		Python_H = np.zeros_like(Python_Datetime)
		for i in xrange(len(Python_Datetime)):
			Python_Date[i] = Python_Datetime[i].date()
			Python_Year[i] = Python_Datetime[i].year
			Python_YD[i] 	= Python_Datetime[i].timetuple().tm_yday
			Python_H[i] = toHourFraction(Python_Datetime[i])
	
	if strip == True:
		Python_Datetime = toDateOnly(Python_Datetime)
	
	return Python_Datetime if expand is None else (Python_Datetime, Python_Date, Python_Year, Python_YD, Python_H)
		
def toHourFraction(date):
	"""Converts time into fractional hours between 0-24
	
	Parameters
    ----------
    date : numpy array
        1 dimensional time series array to convert

    Returns
    -------
    hour_frac : numpy array, float
        The hour fraction for each element of array 'date'
	"""
	
	sec = (date-datetime.combine(date.date(), dttime(0,0,0))).total_seconds()
	hour_frac = sec/3600
	return hour_frac

def toDateOnly(dates, dtype='datetime'):
	"""Converts an array of datetime objects and removes the time
	component keeping the datetime object intact. Similar to doing
	just <datetime_object>.date() but makes it comparable to other
	dates"""
	
	if dtype == 'datetime':
		if isinstance(dates, datetime): #Not an array of dates
			return datetime.combine(dates, dttime(0,0,0))
		else:
			return np.array([datetime.combine(dates[i], dttime(0,0,0)) for i in xrange(len(dates))], dtype=object)
	elif dtype == 'date':
		if isinstance(dates, datetime): #Not an array of dates
			return datetime.date()
		else:
			return np.array([DATETIME.date() for DATETIME in dates], dtype=object)
	else:
		raise ValueError("[toDateOnly] dtype input needs to be either 'datetime' or 'date'. We got %s" % dtype)
		
def toDatetime(date_val):
	"""converts datetime.date() object into a datetime() object"""
	
	if isinstance(date_val, date):
		return datetime.fromordinal(date_val.toordinal())
	elif isinstance(date_val, np.ndarray):
		if date_val.dtype == 'O':
			return np.array([datetime.fromordinal(date2change.toordinal()) for date2change in date_val], dtype=object)
		else:
			sys.exit("[Error: toDatetime] Incorrect dtype")
	else:
		sys.exit("[Error: toDatetime] Incorrect data type")
		
def toDatetime64(datetime_array):
	"""Converts datetime objects to numpy datetime64"""
	
	if isinstance(datetime_array, np.ndarray):
		if isinstance(datetime_array[0], date):
			return datetime_array.astype('datetime64[D]').astype('datetime64[us]')
		else:
			return datetime_array.astype('datetime64[us]')
	else:
		return None
	
def roundTime(dt=None, roundTo=60):
   """Round a datetime object to any time laps in seconds
   
   Parameters
   ----------
   dt : datetime object, optional
		The datetime you want to round. Default uses current time.
   roundTo : int, optional
		Closest number of seconds to round to, default 1 minute.
   
   Author and References
   ---------------------
   Thierry Husson 2012 - "Use it as you want but don't blame me."
   https://stackoverflow.com/a/10854034/8765762
   
   """
   
   if dt is None: dt = datetime.now()
   seconds = (dt.replace(tzinfo=None) - dt.min).seconds
   rounding = (seconds+roundTo/2) // roundTo * roundTo
   
   return dt + timedelta(0,rounding-seconds,-dt.microsecond)

def hours2dt(hours_since_epoch, epoch, dtype='datetime64'):
	"""Converts an array of values representing the number of hours since a common point
	known as an epoch.
	
	Parameters
	----------
	hours_since_epoch : ndarray
		An array of the hours since the epoch date
	epoch : str, datetime, datetime64
		The specific datetime you want to start from. The format for epoch needs to
		work with numpy datetime64.
	dtype : str, optional
		Specify if you want the output to be in numpy datetime or native "python"
		datetime format.
		
	Example
	-------
	>>> hours_since_epoch = np.linspace(0,24,86400)
	>>> epoch = datetime(1992,8,17)
	
	>>> hours2dt(hours_since_epoch, epoch)
	array([datetime(1992,8,17,0,0,0), datetime(1992,8,17,0,0,1) ...])
	
	Notes
	-----
	Function is fast if you specify dtype='datetime64' and leave it in numpy format.
		
	Reference
	---------
	https://codereview.stackexchange.com/a/77662/151534
	https://stackoverflow.com/a/13704307/8765762
	"""
	
	seconds = np.around(hours_since_epoch * (60*60))
	if dtype == 'datetime64':
		return np.datetime64(epoch) + seconds.astype('timedelta64[s]')
	elif dtype == 'datetime':
		return np.array((np.datetime64(epoch) + seconds.astype('timedelta64[s]')).astype(datetime), dtype=object)

def dt2hours(array_of_datetimes, epoch, checkback=False):
	"""Converts an array of datetime objects to hours since epoch. Reverse of hours2dt.
	
	epoch can be either a single datetime or an ndarray of datetime matching the dimensions
	of array_of_datetimes exactly.
	
	np.datetime64('NaT') are converted to np.nan on output.
	
	checkback parameter is useful when blindly converting an array without knowing if the 
	input array contains any datetimes. outputs True if datetimes were found, else False"""
	
	if isinstance(array_of_datetimes, np.ndarray):

		#check for float and int dtypes
		dtypes = ['float', 'float32', 'float128', 'int']
		if np.any(np.in1d(dtypes, array_of_datetimes.dtype)):
			if checkback is False:
				return array_of_datetimes  
			else:
				return array_of_datetimes, False, array_of_datetimes.dtype
		else:
			if not isinstance(epoch, np.ndarray): epoch = np.datetime64(epoch) 
			if array_of_datetimes.dtype == 'O': #Object array
				try:
					array_of_datetimes = toDatetime64(array_of_datetimes)
					
					#Find all np.nat values in datetime
					mask = isnat(array_of_datetimes)
					
					#Convert to hours
					hours = (array_of_datetimes - epoch.astype('datetime64[s]')).astype('timedelta64[s]').astype(float)/3600
					
					#Convert original np.nat values to np.nan
					hours[mask] = np.nan
				
				except ValueError: #Issue can arise if input array is object but does not contain datetimes.
					if checkback is False:
						return np.array(array_of_datetimes, dtype=type(array_of_datetimes[0]))  
					else:
						return np.array(array_of_datetimes, dtype=type(array_of_datetimes[0])), False, np.array(array_of_datetimes, dtype=type(array_of_datetimes[0])).dtype
				
				if checkback is False:
					return hours  
				else:
					return hours, True, 'datetime'
			else:
				#Find all np.nat values in datetime
				mask = isnat(array_of_datetimes)
				
				#Convert to hours
				hours = (array_of_datetimes.astype('datetime64[s]') - epoch.astype('datetime64[s]')).astype('timedelta64[s]').astype(float)/3600
				
				#Convert original np.nat values to np.nan
				hours[mask] = np.nan
				
				if checkback is False:
					return hours  
				else:
					return hours, True, 'datetime64'
	
	elif isinstance(array_of_datetimes, np.datetime64):
		
		array_of_datetimes = np.array(array_of_datetimes)
		
		#Convert to hours
		hours = (array_of_datetimes.astype('datetime64[s]') - epoch.astype('datetime64[s]')).astype('timedelta64[s]').astype(float)/3600
		
		if checkback is False:
			return hours 
		else:
			return hours, True, 'datetime64'
	
	else:
		raise ValueError("{Error} dt2hours requires an ndarray as input not a single value")

def isnat(your_datetime):
	"""Determines if a numpy datetime64 array has any "not a time" values inside
	
	Reference
	---------
	https://stackoverflow.com/a/42103441/8765762
	
	"""
	
	nat_as_integer = np.datetime64('NAT').view('i8')
	
	dtype_string = str(your_datetime.dtype)
	if 'datetime64' in dtype_string or 'timedelta64' in dtype_string:
		return your_datetime.view('i8') == nat_as_integer
	
	return np.array([], dtype=bool)  # it can't be a NaT if it's not a dateime

def datetime64range(start, end, units, interval=1):#, elements=None):
	"""Creates a numpy datetime64 array between start and end with the option to 
	specifiy the time units and interval
	
	Parameters
	----------
	start : datetime or datetime64
		The datetime you want to start from.
	end : datetime or datetime64
		same as above but for the end time
	units : str, optional
		This specifies the type of time or date units you want. 
		Go to https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.datetime.html#datetime-units
		for specifications for units.
	interval : float or int, optional
		How many intervals do you want.
		
	"""
	
	np.arange(start, end, timedelta(days=1)).astype(datetime)
	
	#Ensure datetime64 compliance
	start = np.datetime64(start)
	end = np.datetime64(end)
	
	#If user has not specified interval nor elements we select interval as 1
	#if interval is None and elements is None: interval = 1
	
	#if interval is not None:
	return np.arange(start.astype('datetime64[' + units + ']'), end.astype('datetime64[' + units + ']'), np.timedelta64(interval, units), dtype='datetime64[' + units + ']').astype(start.dtype)
	#elif elements is not None:
	#	return np.linspace(start.astype('datetime64[' + units + ']'), end.astype('datetime64[' + units + ']'), np.timedelta64(elements, units), dtype='datetime64[' + units + ']').astype(start.dtype)

def DatetimeFormat(date, format, check=False):
	"""Formats and checks the validity of a datetime
	format as specified by the user.
	
	Parameters
	----------
	date : str or list
		the datetime string you want to check for being 
		formatted in the correct format.
	format : str
		the format for the datetime to be formatted in.
		E.g. %d/%m/%Y_%H:%M:%S
		Check http://strftime.org/ if you're unsure.
	check : bool, optional
		whether to only check the format and return
		boolean indicators."""
	
	if isinstance(date, str):
		if check is True:
			try:
				datetime.strptime(date, format)
				return True
			except ValueError:	
				return False
		else:	
			try:
				return datetime.strptime(date, format)
			except ValueError:	
				return np.nan
	else:
		Python_Datetime = np.zeros(len(date), dtype=object)
		for i in xrange(len(date)):
			try:
				Python_Datetime[i] = datetime.strptime(date[i], format)
			except TypeError:
				continue
				
		return Python_Datetime	

def ensure_strftime(input_date, format, yearfirst=True):
	"""Ensures that the date input is in a string format which also matches the
	specified format"""
	
	from dateutil.parser import parse
	
	#Make sure format is in string format
	if not isinstance(format, str):
		raise ValueError("[gu.ensure_strftime]: Format dtype must be string. We got %s" % (type(format)))
	
	if isinstance(input_date, datetime) or isinstance(input_date, date):
		return input_date.strftime(format)
	elif isinstance(input_date, np.datetime64):
		return input_date.astype('datetime64[s]').astype(datetime).strftime(format)
	elif isinstance(input_date, str):
		if format.rfind("Y") + format.rfind("d") >= 0:
			if format.rfind("Y") < format.rfind("d"):
				return parse(input_date, dayfirst=False).strftime(format)
			else:
				return parse(input_date, dayfirst=True).strftime(format)
		else:
			return parse(input_date).strftime(format)
	else:
		raise ValueError("[gu.ensure_strftime]: The input date type must be either python datetime, numpy datetime64 or a date formatted string. We got %s" % (type(date)))		
		
def panda2datetime(s):
	"""
	This is an extremely fast approach to datetime parsing.
	For large data, the same dates are often repeated. Rather than
	re-parse these, we store all unique dates, parse them, and
	use a lookup to convert all dates.
	
	Ref
	---
	https://stackoverflow.com/a/29882676/8765762
	https://stackoverflow.com/a/37453925/8765762
	
	"""
	dates = {date:pd.to_datetime(date).to_pydatetime() for date in s.unique()}
	dates64 = s.map(dates)
	return dates64
	return np.array([np.datetime64(date) for date in dates64], dtype=object)
			
def BSTCor(time, date):
	"""Converts array, time from BST (UTC+1) to GMT (UTC+0) based on the
	daily light saving thresholds for each year which is found from the
	date parameter
	"""
	
	import calendar
	
	DST_1 = max(week[-1] for week in calendar.monthcalendar(date.year, 3))
	DST_2 = max(week[-1] for week in calendar.monthcalendar(date.year, 10))

	if date == datetime(date.year, 3, DST_1): #Date lands on daily light saving transition (UTC+0 --> UTC+1)
		print(1)
		time[time >= 2.0] -= 1
	elif date == datetime(date.year, 10, DST_2): #Date lands on daily light saving transition (UTC+1 --> UTC+0)
		print(2)
		time[time < 2.0] -= 1
	elif (date > datetime(date.year, 3, DST_1)) & (date < datetime(date.year, 10, DST_2)): #Date land on UTC+1
		print(3)
		time -= 1
	
	return time   