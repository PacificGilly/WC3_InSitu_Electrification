############################################################################
# Project: Electrical Pre-Conditioning of Convective Clouds,
# Title: WC3 Radiosonde Python Code Extras
# Author: James Gilmore,
# Email: james.gilmore@pgr.reading.ac.uk.
# Version: 1.0
# Date: 24/07/2018
# Status: Stable
# Change: Added GPS2UTC
############################################################################
from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from datetime import datetime, timedelta
import sys

sys.path.insert(0, '/home/users/th863480/PhD/Global_Functions')

#User Processing Modules
import Gilly_Utilities as gu

def GPS2UTC(GPS_Week, GPS_Second, Rollover=1, LeapSeconds=18):
	"""Converts GPS week and seconds to UTC. This is useful for Vaisala radiosonde 
	launches
	
	Based on research
	-----------------
	http://software.ligo.org/docs/glue/glue.gpstime-pysrc.html#UTCFromGps
	http://www.npl.co.uk/reference/faqs/when-and-what-is-the-gps-week-rollover-problem-(faq-time)
	https://www.timeanddate.com/time/leap-seconds-future.html
	
	Parameters
	----------
	GPS_Week : int or float
		The GPS week ranging between 0-1024. For current GPS launches the number
		of weeks passed 1980/01/06 is greater than 1024. After this time the GPS 
		week is reset to 0 and starts incrementing again. To fix this issue you
		must set Roll-over command. Currently it is set to 1 for any GPS dates 
		passed 1999/08/21.
	GPS_Second : float
		The number of seconds within a GPS week. Value should range between 0 and
		604800 representing the total number of seconds that occurs within one week
		assuming a standard 86400 seconds occurs in a single day.
	GPS_Rollover : int, optional
		If the GPS week surpassed 1024 it is reset and starts counting from 0 again.
		This loses the information of time. For any current launch the roll-over being
		set to 1 is fine. The next roll-over occurs on 2019/04/06.
	Leap_Seconds : int, optional
		The number of leap seconds that have occurred between 1980/01/06 and present. 
		This is a bit of an ambiguous term if you don't know when launch occurred. 
		Currently there is no real method of automatically acquiring the leap second.
		You could find an updating table with date reference and cycle the analysis
		until you find the correct number of leap seconds but that might be overkill.
		As of 2016/12/31 there have been +18 seconds added.
		
	Output
	------
	UTC_Time : datetime
		The converted GPS time in correct UTC format (adjusted for leap seconds) in
		datetime format.
	"""
	
	GPS_Epoch = datetime(1980, 1, 6, 0, 0, 0)
	UTC_Time = GPS_Epoch + timedelta(weeks=GPS_Week+(1024.0*Rollover), seconds=GPS_Second+LeapSeconds)
	
	return UTC_Time.replace(microsecond=0)

def CloudDrift(Radiosonde_Data):
	"""Estimates the amount of temperature drift in the cloud sensor from radiosonde launch No.3.
	This launch was chosen as there was a strong amount of drift in the sensor above 5km.
	
	Therefore, this function makes it possible to retest our conclusions rather than something that
	needs to be continuously executed.
	
	IR_Drift = 0.210855383762 * T 
	
	To use this information we would add the IR_Drift to IR:

	IR_Corrected = IR_Uncorrected + IR_Drift 
	
	"""
	
	#Collect relevant data
	Z = Radiosonde_Data[:,1]
	Tdry = Radiosonde_Data[:,3]
	IR = Radiosonde_Data[:,8]
	
	#Remove nans from all arrays
	Z, Tdry, IR = gu.antinan((Z,Tdry,IR), unpack=True)
	
	#Filter the data using the Savitzky-Golay filter
	IR_Filter = sp.signal.savgol_filter(IR, 1001, 1, deriv=0)
	
	#Calculate linear regression between Tdry and IR_Filter
	slope, intercept, r_value, p_value, std_err = sp.stats.linregress(Tdry[Z>5], IR_Filter[Z>5])
	
	#Print out the results
	gu.cprint("Temperature drift calculated for the Cloud sensor\n-------------------------------------------------", type='bold')
	print("IR_Drift = %.5f * T" % -slope)
		
	#Plot results
	plt.clf()
	plt.plot(IR, Z, label='Uncorrected')
	plt.plot(IR - slope*Tdry, Z, label='Corrected')

	plt.xlabel("IR Sensor (Counts)")
	plt.ylabel("Height ($km$)")
	plt.title("Difference between Uncorrected and Corrected Cloud Sensor")
	
	plt.grid(which='major',axis='both',c='grey')
	
	plt.legend(loc='upper right', prop={'size': 10}, fancybox=True, framealpha=0.5)
	
	plt.show()
	
	return
	
def Radiosonde_Launch(GPS_File, offset=0):
	"""Finds the datetime when the radiosonde was launched
	
	Parameters
	----------
	GPS_File : str
		The file containing the GPS file. Typically it is GPSDCC_RESULT*.tsv
	offset : int or float, default == 0
		The time offset in seconds that can be used to change the GPS time if the recording
		laptop was out of sync.
	"""
	
	#Import all the data
	if len(GPS_File) != 0: GPS_Data = np.genfromtxt(GPS_File[0], delimiter=None, skip_header=51, dtype=float, comments="#")
	
	#Find a contiguous region of data where dSondeX is measuring 
	GPS_Start = gu.contiguous(GPS_Data[:,4], min=500, invalid = -32768.00)
	
	#Find the first location in GPS_Start that's non zero
	GPS_Index = gu.bool2int(GPS_Start == 1)[0]

	#Convert GPS time into UTC
	if GPS_File is not None: Launch_Datetime = GPS2UTC(GPS_Data[GPS_Index,1], GPS_Data[GPS_Index,2]) + timedelta(seconds=offset)
	
	return Launch_Datetime, GPS_File