############################################################################
# Project: Electrical Pre-Conditioning of Convective Clouds,
# Title: Radiosonde Calibration Metrics
# Author: James Gilmore,
# Email: james.gilmore@pgr.reading.ac.uk.
# Version: 1.0
# Date: 05/03/2018
# Status: Stable
############################################################################
from __future__ import absolute_import, division, print_function
import numpy as np
import sys

sys.path.insert(0, '/home/users/th863480/PhD/Global_Functions')
import Gilly_Utilities as gu
	
class Radiosonde_Checks(object):
	
	def __init__(self, data, calibrate, package_no, height_range, check):
		"""Specify the required data and parameters for the radiosonde
		
		Parameters
		----------
		data : ndarray
			The data of the radiosonde
		calibrate : tuple
			The columns you want to calibrate. These should be the bespoke 
			data channels from Pandora
		package_no : int
			The package number of sensors built
		height_range : tuple
			The height range to plot between
		check : int
			The row identifier for checking and fixing the data
		
		"""
		
		#Make data universal
		self.data = np.array(data, dtype=float)
		
		#Run checks
		self.quality_control(calibrate, package_no, height_range, check)
	
	def return_data(self):
		return self.data
	
	def quality_control(self, calibrate, package_no, height_range, check):
	
		#Detect not a number values (i.e. find -32768 in data)
		self.data[self.data == -32768] = np.nan
		self.data = self.data[self.data[:,0] > 0]
		
		#Fixes the data for the first two radiosonde flights as a scaling factor of 2**4 was applied to certain channels as part of the I2C setup that we don't use.
		if all(x in [1,2] for x in [package_no]): #if package_no == 1 or 2 then process
			self.data[:,5] *= 2**4
			self.data[:,6][self.data[:,9] == check] = self.data[:,6][self.data[:,9] == check] * 2**4
			
		#Calibrate remaining data
		if calibrate is not None:
			if isinstance(calibrate, (tuple, list, set)) is True:
				#Calibrate PANDORA Data
				for col in calibrate:
					if col == 7:
						self.data[:,col][self.data[:,9] == 1111] *= (5/4096)
						self.data[:,col][self.data[:,9] == 1112] /= 100
					else:
						self.data[:,col] *= (5/4096)
			else:
				warnings.warn("[Radiosonde Superplotter] Calibrate expects either a tuple, list or set as an input. Therefore, skipping calibration!", SyntaxWarning, stacklevel=2)

				
		#Calibrate Radiosonde Height
		self.data[:,1] /= 1000
		self.data[:,3] -= 273.15
		self.data[:,14] -= 273.15
	
		#Filter data to specified height
		if height_range is not None: self.data = self.data[(self.data[:,1] >= height_range[0]) & (self.data[:,1] <= height_range[1])]
	
	def charge_calibration(self, package_no):
		"""Calibration values for the charge sensor for each package number. 
		Outputs the linear and log values
		
		Output
		------
		Linear, Log 
		
		Don't get confused by this order as the equations are usually reversed
		(i.e. processed_linear_data = a*np.log(raw_linear_data) + b, processed_log_data = a*raw_log_data + b """
		
		#Remove anomalous values. Only voltages between (-5, 5) are allowed.
		self.data[:,5][(-5 > self.data[:,5]) ^ (self.data[:,5] > 5)] = np.nan
		self.data[:,6][(-5 > self.data[:,6]) ^ (self.data[:,6] > 5)] = np.nan
		
		if package_no == 0:
			return self.data
		
		elif package_no == 1:
			self.data[:,5] = -0.001*self.data[:,5]**3 + 0.0075*self.data[:,5]**2 - 0.0256*self.data[:,5] + 0.0326; self.data[:,6] = -3.7318*self.data[:,6] + 4.6432
		
			return self.data
		elif package_no == 2:
			self.data[:,5] = -0.565*np.log(self.data[:,5]) + 0.2988; self.data[:,6] = -7.0975*self.data[:,6] + 8.8875
			
			return self.data
		elif package_no == 3:
			return self.data
		elif package_no == 4:
			return self.data
		elif package_no == 5:
			return self.data
		elif package_no == 6:
			return self.data
		elif package_no == 7:
			return self.data
		elif package_no == 8:
			return self.data
		elif package_no == 9:
			return self.data
		elif package_no == 10:
			return self.data
	
	def wire_calibration(self):
		"""Calibrates the vibrating wire based upon the research by Serke (2014) and calculates
		the mass equation,
		
							SLWC = -(2*b*f_0**2)/(e*D*w*f**3)*df/dt,
							
		where b is the weight of the steel wire per unit length, f_0 is the pre-launch wire 
		frequency, e is the collection efficiency, D is the wire diameter, w is the air velocity
		relative to the wire (i.e. the wind speed wind**2 = u**2+v**2+w**2) and f is the wire
		frequency at a given time, t.
		
		Research
		--------
		Serke D., et. al. (2014) Supercooled liquid water content profiling case studies with 
		a new vibrating wire sonde compared to a ground-based microwave radiometer. Atmos. Res.
		77-97, 149.
		
		"""
		
		#Parameters
		b = 44.366 #[g cm^-1]
		D = 0.1
		e = 0.95
		f = self.data[:,7][self.data[:,9] == 1112]; #Takes into account that the output off PLL is already in Hz (specifically in cHz, hence the divide by 100)
		t = self.data[:,0][self.data[:,9] == 1112]
		f_o = f[0]
		w = np.sqrt(self.data[:,15][self.data[:,9] == 1112]**2 + self.data[:,16][self.data[:,9] == 1112]**2) * 100 #Converts from m/s to cm/s for units to be correct in SLWC equation
		
		#Equations
		SLWC = lambda b, D, e, f_o, w, f, dfdt: -(2*b*f_o**2)/(e*D*w*f**3)*dfdt
		
		#Calculate the rate of change in frequency
		f_run = gu.running_mean(f, 11)

		dfdt = gu.list_func((np.roll(f_run, -1) - f_run)).rljust(len(f), np.nan, array_type=float)/gu.list_func((np.roll(t, -1) - t)).rljust(len(t), np.nan, array_type=float)
		dfdt[dfdt == 0] = np.nan
		
		#Calculate SLWC		
		SLWC = SLWC(b, D, e, f_o, w, f, dfdt)
		
		#Riffle dfdt and SLWC arrays into 1112 place to make plotting easier
		index_all = np.arange(len(self.data[:,0]))
		index_1112 = [index_all[self.data[:,0] == val][0] for val in t]
		antiindex_1112 = np.setdiff1d(index_all, index_1112)
		
		dfdt_full = np.zeros(self.data[:,0].size, dtype=float)
		SLWC_full = np.zeros(self.data[:,0].size, dtype=float)
		
		dfdt_full[index_1112] = dfdt
		dfdt_full[antiindex_1112] = np.nan
		
		SLWC_full[index_1112] = SLWC
		SLWC_full[antiindex_1112] = np.nan
		
		self.data = np.vstack((self.data.T, dfdt_full, SLWC_full)).T
		
		return self.data
		