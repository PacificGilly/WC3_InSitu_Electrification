############################################################################
# Project: Electrical Pre-Conditioning of Convective Clouds,
# Title: Radiosonde Calibration Metrics
# Author: James Gilmore,
# Email: james.gilmore@pgr.reading.ac.uk.
# Version: 1.2
# Date: 08/08/2018
# Status: Stable
############################################################################
from __future__ import absolute_import, division, print_function
import numpy as np
import scipy as sp
import sys

sys.path.insert(0, '/home/users/th863480/PhD/Global_Functions')
import Gilly_Utilities as gu
	
class Radiosonde_Checks(object):
	
	def __init__(self, data, calibrate, package_no, height_range, check, enforce_parity=True, linear=True, log=True):
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
		self.linear = linear
		self.log = log
		
		#Run checks
		self.quality_control(calibrate, package_no, height_range, check, enforce_parity)
	
	def _space_charge(self, Time, Height, Charge, lab_calibration=False, A_eff=0.0196, MA=11, keepdims=True):
		"""Calculates the space charge density from current measurements
		
		Parameters
		----------
		Time : ndarray
			A numpy array containing the timestamps for each charge measurements
		Height : ndarray
			A numpy array containing the height for each measurement
		Charge : ndarray
			A numpy array containing the charge measurements (in Volts NOT Counts)
		lab_calibration : bool, optional, default is False
			Specify whether the induced current is calculated from lab calibrations or from the circuit
		A_eff : float, optional, default = 0.0196
			The effective area of the electrode (units of m^2)
		MA : int, optional, default = 11
			The moving average window used to smooth out the variability in the charge
			measurements.
		keepdims : bool, optional
			Specify if you want to keep the same dimensions as the input array. If True,
			Time will be used to locate the best locations for each element after the 
			analysis. If MA == 1 then keeps-dims is redundant.
			
		"""
		
		print("Charge Stats", gu.stats(Charge))
		
		#Calculate moving average of Time
		Time_MA = gu.moving_average(Time, MA+1)
		
		if lab_calibration is False:
			Induced_Current = gu.moving_average((-(Charge - np.nanmedian(Charge))/(2.4*10**11))[:-1], 11)
		else:
			Induced_Current = gu.moving_average(Charge, 12)

		print("Induced_Current Stats", gu.stats(Induced_Current))	
			
		#Determine Ascent Rate of the Radiosonde
		Ascent_Rate = gu.running_mean(np.diff(Height*1000)/np.diff(Time), 11)
		
		#Calculate Space Charge Density
		Space_Charge = (Induced_Current/(A_eff*Ascent_Rate))/(1*10**-12)
		
		#Riffle Space_Charge into order if specified
		if keepdims is True: Space_Charge = gu.mask(Time, Time_MA, invert=True, impose=np.nan, cross_impose=Space_Charge)
		
		print("Median Linear Charge [V]", np.nanmedian(Charge))
		print("Induced Current, i [A]", np.nanmedian(Induced_Current))
		print("Ascent Rate, w [m/s]", np.nanmedian(Ascent_Rate))
		print("Space_Charge, rho [pC/m^3]", np.nanmedian(Space_Charge))

		return Space_Charge
		
	def return_data(self):
	
		#Filter data to specified height
		if self.height_range is not None: self.data = self.data[(self.data[:,1] >= self.height_range[0]) & (self.data[:,1] <= self.height_range[1])]
		
		return self.data
	
	def quality_control(self, calibrate, package_no, height_range, check, enforce_parity):
	
		self.height_range = height_range
	
		#Detect not a number values (i.e. find -32768 in data)
		self.data[self.data == -32768.0] = np.nan
		self.data = self.data[self.data[:,0] > 0]
		
		#Remove anomalous values in the PANDORA channels
		#Cloud Sensors
		self.data[:,7][self.data[:,7] < 1000] = np.nan
		self.data[:,8][self.data[:,8] < 1000] = np.nan
		
		#OMB Sensor
		self.data[:,7][(self.data[:,9] == 1112) & (self.data[:,7] > 3000)] = np.nan
		
		#Fixes the data for the first two radiosonde flights as a scaling factor of 2**4 was applied to certain channels as part of the I2C setup that we don't use.
		if all(x in [1,2] for x in [package_no]): #if package_no == 1 or 2 then process
			self.data[:,5] *= 2**4
			self.data[:,6][self.data[:,9] == check] = self.data[:,6][self.data[:,9] == check] * 2**4
			
		#Calibrate PANDORA Data
		if (calibrate == "volts") or (calibrate == "units"):
			self.calibrate_channels = [7,8]
			if self.linear is True: self.calibrate_channels.append(5)
			if self.log is True: self.calibrate_channels.append(6)
			
			for col in self.calibrate_channels:
				if col == 7:
					self.data[:,col][self.data[:,9] == 1111] *= (5/4096)
					self.data[:,col][self.data[:,9] == 1112] /= 100
				else:
					self.data[:,col] *= (5/4096)
		
		#Mask all zero values in OMB data
		if package_no >= 3:
			self.data[:,7][self.data[:,9] == 1112] = gu.mask(self.data[:,7][self.data[:,9] == 1112], 0, find_all=True, impose=np.nan)
	
		#Adjust frequency for temperature drift
		T = self.data[:,3][self.data[:,9] == 1112]	
		self.data[:,7][self.data[:,9] == 1112] = self.data[:,7][self.data[:,9] == 1112]-(4.3*self.data[:,7][self.data[:,9] == 1112])/T
		
		#Calibrate Radiosonde Height
		self.data[:,1] /= 1000
		self.data[:,3] -= 273.15
		self.data[:,14] -= 273.15
	
		#Filter data to specified height
		if (self.height_range is not None) & (calibrate == 'counts'): self.data = self.data[(self.data[:,1] >= self.height_range[0]) & (self.data[:,1] <= self.height_range[1])]
	
		#Remove any random data from parity channels (FOR NO.5 FLIGHT)
		if enforce_parity is True: self.data = self.data[(self.data[:,9] == 1111) ^ (self.data[:,9] == 1112)]
		
	def RH(self, package_no):
		"""Correct the relative humidity sensor for the ice phase.
		
		Reference
		---------
		Nicoll K.A. (2010). Appendix B4. In:Coupling Between the Global Atmospheric 
			Electric Circuit and Clouds. PhD Thesis. University of Reading.
		Buck, A. L. (1981), "New equations for computing vapor pressure and enhancement 
			factor", J. Appl. Meteorol., 20: 1527-1532"""
			
		#Define equations
		#Equations using Goff-Gratch formulation
		#es_w = lambda a0, T: a0*np.exp(a0*(T/(T+273.15)))
		#es_i = lambda e0, x: e0*10**x
		#x = lambda a, b, c, T0, T: a*(T0/T-1)+b*np.log10(T0/T)+c*(1-T/T0)
		
		#Equation using Arden Buck equations
		es_w = lambda T: 0.61121*np.exp((18.678-T/234.5)*(T/(257.14+T)))
		es_i = lambda T: 0.61115*np.exp((23.036-T/333.7)*(T/(279.82+T)))
		RH_ice = lambda RH, es_w, es_i: RH*es_w/es_i
		
		#Mask data for temperatures below 0
		Temp_Mask = self.data[:,3] < 0
		
		#Calculate saturate vapour pressure over water and ice
		es_w = es_w(self.data[Temp_Mask,3])
		es_i = es_i(self.data[Temp_Mask,3])

		#Calculate the relative humidity w.r.t. ice	
		RH_ice = np.append(self.data[~Temp_Mask,4], RH_ice(self.data[Temp_Mask,4], es_w, es_i))
	
		#Add extra column to data
		self.data = np.vstack((self.data.T, RH_ice)).T
	
	def cloud_calibration(self, package_no, check=1111):
		"""Tries to de-trend the cloud sensor data to increase the changes within the data"""
		
		#Collect datasets
		Time = self.data[:,0][self.data[:,9] == check]
		Cloud_Cyan = self.data[:,7][self.data[:,9] == check]
		Cloud_IR = self.data[:,8]
		
		#Determine trend offset for each 0.5km of data
		Trend_Boundaries_Cyan = np.searchsorted(self.data[:,1][self.data[:,9] == check], np.arange(0,15,0.5))
		Trend_Boundaries_IR = np.searchsorted(self.data[:,1], np.arange(0,15,0.5))
		
		#De-trend the cyan and IR after removing any nan's
		Cloud_Cyan_Detrend = sp.signal.detrend(gu.antinan(Cloud_Cyan), type='linear', bp=Trend_Boundaries_Cyan)
		Cloud_IR_Detrend = sp.signal.detrend(gu.antinan(Cloud_IR), type='linear', bp=Trend_Boundaries_IR)
		
		#Riffle Cloud_Cyan_Detrend/Cloud_IR_Detrend into Cloud_Cyan/Cloud_IR (i.e. place all nan's found in Cloud_Cyan/Cloud_IR and put them back in the same index position and therefore restore the original shape)
		Cloud_Cyan_Detrend_Riffle = gu.mask(Cloud_Cyan, gu.antinan(Cloud_Cyan), invert=True, impose=np.nan, cross_impose=Cloud_Cyan_Detrend, find_all=True)
		Cloud_IR_Detrend_Riffle = gu.mask(Cloud_IR, gu.antinan(Cloud_IR), invert=True, impose=np.nan, cross_impose=Cloud_IR_Detrend, find_all=True)
		
		#For cyan (which shares a channel with PLL) we now need to align Cloud_Cyan_Detrend_Riffle with self.data[:,7] and remember to join with the PLL dataset (see impose parameter)
		Col7 = gu.mask(self.data[:,7], Cloud_Cyan, invert=True, impose=self.data[:,7][self.data[:,9] == 1112], cross_impose=Cloud_Cyan_Detrend_Riffle, find_all=True)
		Col8 = gu.mask(self.data[:,8], Cloud_IR, invert=True, impose=np.nan, cross_impose=Cloud_IR_Detrend_Riffle, find_all=True)
		
		self.data[:,7] = Col7
		self.data[:,8] = Col8
		
	def charge_calibration(self, package_no, type='space_charge', lab_calibration=True, log=True, linear=True):
		"""Calibration values for the charge sensor for each package number. 
		Outputs the linear and log values
		
		Parameters
		----------
		package_no : int
			The radiosonde flight number. Used to pick which calibration to select.
		type : str, optional, default = 'space_charge'
			Choose how to calibrate the charge sensor, either 'space_charge' or 'current'. 
			
			'space_charge' 	: Calculate the space charge density
			'current'		: Calculate the induced current using lab based calibration values
		lab_calibration : bool, optional, default is True
			Specify to use the lab based calibration (True) or to use the circuit components to 
			determine the induced current. Only applicable when type == 'space_charge'
		log : bool, optional, default is True
			Specify to calibrate the log charge sensor
		linear : bool, optional, default is True
			Specify to calibrate the linear charge sensor
		
		Output
		------
		if type == 'space_charge':
			Linear, Log in pC/m^3
		if type == 'current':
			Linear, Log in A
		
		Don't get confused by this order as the equations are usually reversed
		(i.e. processed_linear_data = a*np.log(raw_linear_data) + b, processed_log_data = a*raw_log_data + b """
		
		print("self.calibrate_channels", self.calibrate_channels)
		
		#Convert Linear and Log to Volts
		if (linear is True) & (5 not in self.calibrate_channels): 
			self.data[:,5] *= (5/4096)
			self.calibrate_channels.append(5)
		if (log is True) & (6 not in self.calibrate_channels): 
			self.data[:,6] *= (5/4096)
			self.calibrate_channels.append(6)
		
		if type == 'space_charge':
			
			#Remove anomalous values. Only voltages between (-5, 5) are allowed.
			self.data[:,5][(-5 > self.data[:,5]) ^ (self.data[:,5] > 5)] = np.nan
			self.data[:,6][(-5 > self.data[:,6]) ^ (self.data[:,6] > 5)] = np.nan
			
			print("Log Stats [Volts]", gu.stats(self.data[:,6]))
			
			#Convert from counts to current
			if lab_calibration is True: self.charge_calibration(package_no=package_no, type='current', linear=linear, log=log)
			
			print("Log Stats [Units]", gu.stats(self.data[:,6]))
			
			#Calculate the space charge density
			if linear is True: self.data[:,5] = self._space_charge(self.data[:,0], self.data[:,1], self.data[:,5], lab_calibration=lab_calibration)
			if log is True: self.data[:,6] = self._space_charge(self.data[:,0], self.data[:,1], self.data[:,6], lab_calibration=lab_calibration)
					
			#Determine trend offset for each 0.5km of data
			#Trend_Boundaries = np.searchsorted(self.data[:,1], np.arange(0,15,0.5))
			
			#De-trend the Space_Charge after removing any nan's
			#Space_Charge_Detrend = sp.signal.detrend(gu.antinan(Space_Charge), type='linear', bp=Trend_Boundaries)
			
			#Riffle Space_Charge_Detrend into Space_Charge
			#Space_Charge_Detrend_Riffle = gu.mask(Space_Charge, gu.antinan(Space_Charge), invert=True, impose=np.nan, cross_impose=Space_Charge_Detrend, find_all=True)
			
			# #For cyan (which shares a channel with PLL) we now need to align Cloud_Cyan_Detrend_Riffle with self.data[:,7] and remember to join with the PLL dataset (see impose parameter)
			# if linear is True: 
				# DetrendSpaceCharge = gu.mask(self.data[:,5], charge_data, invert=True, impose=charge_data, cross_impose=Space_Charge_Detrend_Riffle, find_all=True)
			# else:
				# DetrendSpaceCharge = gu.mask(self.data[:,6], charge_data, invert=True, impose=charge_data, cross_impose=Space_Charge_Detrend_Riffle, find_all=True)
		
			#Riffle the 11-point running mean to package with self.data. Used for plotting
			# if linear is True:
				# self.data[:,5] = gu.mask(self.data[:,0], Time, invert=True, impose=np.nan, cross_impose=Space_Charge_Detrend_Riffle)
				# self.data[:,6] = np.full(self.data[:,6].size, np.nan)
			# else:
				# self.data[:,5] = np.full(self.data[:,6].size, np.nan)
				# self.data[:,6] = gu.mask(self.data[:,0], Time, invert=True, impose=np.nan, cross_impose=Space_Charge_Detrend_Riffle)
			
			return self.data
		
		elif type == 'current':
			if package_no == 0:
				return self.data
			
			elif package_no == 1:
				
				if linear is True: self.data[:,5] = (-0.001*self.data[:,5]**3 + 0.0075*self.data[:,5]**2 - 0.0256*self.data[:,5] + 0.0326)/10**12
				if log is True: self.data[:,6] = (-3.7318*self.data[:,6] + 4.6432)/10**12
			
				return self.data
			elif package_no == 2:
				
				if linear is True: self.data[:,5] = (-0.565*np.log(self.data[:,5]) + 0.2988)/10**12
				if log is True: self.data[:,6] = (-7.0975*self.data[:,6] + 8.8875)/10**12
				
				return self.data
			elif package_no == 3:
				
				if linear is True: self.data[:,5] = (-11.522*self.data[:,5]**3 + 103.63*self.data[:,5]**2 - 317.14*self.data[:,5] + 324.2)/10**12
				if log is True: self.data[:,6] = (-5876.3*self.data[:,6] + 7376.7)/10**12
				
				return self.data
			elif package_no == 4:
				
				#if linear is True: self.data[:,5] = (-2.3863*self.data[:,5]**3 + 19.588*self.data[:,5]**2 - 58.392*self.data[:,5] + 59.887)/10**12
				if linear is True: self.data[:,5] = (-11.173*self.data[:,5] + 28.992)/10**12
				if log is True: self.data[:,6] = (-3536.5*self.data[:,6] + 4409.6)/10**12
						
				return self.data
			elif package_no == 5:
				#if linear is True: self.data[:,5] = (-3.2083*self.data[:,5]**3 + 27.357*self.data[:,5]**2 - 83.373*self.data[:,5] + 86.647)/10**12
				if linear is True: self.data[:,5] = (-12.557*self.data[:,5] + 33.427)/10**12
				if log is True: self.data[:,6] = (-4754*self.data[:,6] + 5951)/10**12
			
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
		else:
			raise ValueError("Radiosonde Charge Calibration: type must be either 'space_charge' or 'current'. We got %s" % type)
			
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
		
		#Variables
		f = self.data[:,7][self.data[:,9] == 1112]																	#Wire frequency [Hz]
		t = self.data[:,0][self.data[:,9] == 1112]																	#Time [s]
		z = self.data[:,1][self.data[:,9] == 1112]																	#Height [m]
		
		#Parameters
		b = 0.044366 																								#Wire Weight per unit length [g cm^-1]
		D = 0.1																										#Wire Diameter [cm]
		e = 0.95																									#Collection Efficiency [unit-less]
		f_o = gu.antinan(f)[0]																						#Pre-launch un-iced wire frequency (~24.0Hz)
		w = np.sqrt(self.data[:,15][self.data[:,9] == 1112]**2 + self.data[:,16][self.data[:,9] == 1112]**2) * 100 	#Air Velocity relative to Wire [cm/s] for units to be correct in SLWC equation
	
		#Equations --> SLWC [g m^-3]
		SLWC = lambda b, D, e, f_o, w, f, dfdt: ((-2*b*f_o**2)/(e*D*w*f**3))*dfdt*10**6
		
		#Remove nans
		t, f, w, z = gu.antinan(np.array([t,f,w,z]), unpack=True)
				
		#Calculate the 11-point running mean
		f_run = gu.moving_average(f, 11)
		t_run = gu.moving_average(t, 11)
		z_run = gu.moving_average(z, 11)
		
		#Calculate the rate of change in frequency
		dfdt = np.diff(f_run)/np.diff(t_run)
		dfdt = gu.np_func(dfdt).rljust(t.size, np.nan, dtype=float)
		dfdt[dfdt == 0] = np.nan

		#Calculate SLWC
		SLWC = SLWC(b, D, e, f_o, w, f, dfdt)
				
		#Riffle dfdt and SLWC arrays into 1112 place to make plotting easier
		dfdt = gu.mask(self.data[:,0], t, invert=True, impose=np.nan, cross_impose=dfdt)
		SLWC = gu.mask(self.data[:,0], t, invert=True, impose=np.nan, cross_impose=SLWC)
		
		#Riffle the 11-point running mean to package with self.data. Used for plotting
		f_run = gu.mask(self.data[:,0], t_run, invert=True, impose=np.nan, cross_impose=f_run)
		z_run = gu.mask(self.data[:,0], t_run, invert=True, impose=np.nan, cross_impose=z_run)

		self.data = np.vstack((self.data.T, dfdt, np.abs(SLWC), z_run, f_run)).T
		
		return self.data

if __name__ == "__main__":
	sys.exit("[ERROR] Radiosonde_Calibration cannot be run as standalone. Used as part of Radiosonde_Analysis")