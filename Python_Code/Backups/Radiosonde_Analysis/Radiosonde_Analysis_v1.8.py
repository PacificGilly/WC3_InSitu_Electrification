############################################################################
# Project: Electrical Pre-Conditioning of Convective Clouds,
# Title: Plotting Radiosonde Data
# Author: James Gilmore,
# Email: james.gilmore@pgr.reading.ac.uk.
# Version: 1.8
# Date: 31/12/2018
# Status: Stable
# Change: Final version before major overhaul of the data structures
############################################################################
from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import os, sys, time, warnings, glob, argparse
from datetime import datetime, timedelta

sys.path.insert(0, '/home/users/th863480/PhD/Global_Functions')

#User Processing Modules
import Gilly_Utilities as gu

#Data Set-up Modules
from Data_Importer import EPCC_Importer
from Data_Quality import Radiosonde_Checks
from Data_Output import SPRadiosonde

#Import Global Variables
import PhD_Config as PhD_Global

#Import Tephigram Plotter
from Extras.Tephigram import Tephigram as SPTephigram

#Import WC3 Extras (for GPS2UTC)
from Extras.WC3_Extras import GPS2UTC, CloudDrift, Radiosonde_Launch

class Radiosonde(EPCC_Importer, Radiosonde_Checks, SPRadiosonde, SPTephigram):
	"""This class will process all the data aquired from a radiosonde output the data in various forms,
	including, height plot, tepihgram and indicies.
	
	Parameters
	----------
	EPCC_Importer : class
		Used to import other datasets other than the actual radiosonde
	Radiosonde_Checks : class
		Used to quality control the radiosonde data
	SPRadiosonde : class
		Used to plot the radiosonde data
`	SPTephigram : class
		Used to plot the tepihgram of the data
		
	"""
	
	def __init__(self, Radiosonde_File=None, Sensor_Package=None, Height_Range=None, Calibrate='counts', verbose=False):
		"""Set-up radiosonde data"""
		
		############################################################################
		"""Prerequisites"""
    
		#Time Controls
		t_begin = time.time()
		
		#Storage Locations
		self.Storage_Path    		= PhD_Global.Storage_Path_WC3
		self.Processed_Data_Path	= 'Processed_Data/Radiosonde/'
		self.Raw_Data_Path			= 'Raw_Data/'
		self.Radiosonde_Plots_Path  = 'Plots/Radiosonde/'
		self.Tephigram_Plots_Path   = 'Plots/Tephigram/'
		
		#Bound classes
		self.importer = EPCC_Importer()
		self.radiosonde_file = Radiosonde_File
		self.sensor_package = Sensor_Package
		self.height_range = Height_Range
		self.calibrate = Calibrate
		self.verbose = verbose
		
		############################################################################
		
		#Import Radiosonde Data
		self.Radiosonde_Data, self.Launch_Datetime = self._RadiosondeImporter(self.radiosonde_file, self.sensor_package)
	
		#Identify clouds within data
		self.Clouds_ID, self.LayerType = self._CloudIdentifier(self.height_range)
		
		#Calculate the space charge density using the log charge sensor
		self.Calibration_Log = self._ChargeCalibrator(self.calibrate, self.sensor_package, self.Clouds_ID, self.LayerType) if np.any(np.in1d(self.calibrate, ['volts', 'units'])) else None
		
	def _RadiosondeImporter(self, Radiosonde_File=None, Sensor_Package=None):
		"""Check and Import Data"""
		
		#Error check that either Radiosonde_File or Sensor_Package has been specified
		if Radiosonde_File is None and Sensor_Package is None: sys.exit("[Error] You must specify either the Radiosonde_File location or the Sensor_Package number")
		
		#Attempt to find the radiosonde file either directly or from glob
		self.Radiosonde_File = self.Storage_Path + self.Processed_Data_Path + Radiosonde_File if Radiosonde_File is not None else glob.glob(self.Storage_Path + self.Processed_Data_Path + 'Radiosonde_Flight_No.' + str(Sensor_Package).rjust(2,'0') + '_*/Radiosonde_Flight_PhD_James_No.' + str(Sensor_Package) + '*a.txt')
		
		#If no radiosonde file was found we end program
		if len(self.Radiosonde_File) == 0: sys.exit("[Error] Radiosonde package No.%s does not exist. Has the radiosonde been launched yet or has the data been misplaced?" % (Sensor_Package))
		
		#If the radiosonde file was found via glob we need to convert to str from list
		if isinstance(self.Radiosonde_File, list): self.Radiosonde_File = self.Radiosonde_File[0]
		
		#Once the radiosonde file is found we can attempt to find the GPS file in the raw file section
		self.GPS_File = glob.glob(self.Storage_Path + self.Raw_Data_Path + 'Radiosonde_Flight_No.' + str(Sensor_Package).rjust(2,'0') + '_*/GPSDCC_RESULT*.tsv')
		
		#Import all the data
		Radiosonde_Data = np.genfromtxt(self.Radiosonde_File, delimiter=None, skip_header=10, dtype=float, comments="#") if self.Radiosonde_File is not None else None
		GPS_Data = np.genfromtxt(self.GPS_File[0], delimiter=None, skip_header=51, dtype=float, comments="#") if len(self.GPS_File) != 0 else None
		
		GPS_Data = GPS_Data[GPS_Data[:,4] > 0]
		#GPS_Data = GPS_Data[7718:] #for No.4
		if self.GPS_File is not None: Launch_Datetime = GPS2UTC(GPS_Data[0,1], GPS_Data[0,2])
		
		return Radiosonde_Data, Launch_Datetime
	
	def _CloudIdentifier(self, Height_Range=None):
		"""This function will identify the cloud layers within a radiosonde ascent by using the cloud sensor and 
		relative humidity measurements
		
		Reference
		---------
		Zhang, J., H. Chen, Z. Li, X. Fan, L. Peng, Y. Yu, and M. Cribb (2010). Analysis of cloud layer structure 
			in Shouxian, China using RS92 radiosonde aided by 95 GHz cloud radar. J. Geophys. Res., 115, D00K30, 
			doi: 10.1029/2010JD014030.
		WMO, 2017. Clouds. In: Internal Cloud Atlas Manual on the Observation of Clouds and Other Meteors. 
			Hong Kong: WMO, Section 2.2.1.2.
		"""
		
		if self.verbose is True: gu.cprint("[INFO] You are running Radiosonde_CloudIdentifier from the STABLE release", type='bold')
		
		############################################################################
		"""Calibrate bespoke sensors"""
    
		#Calibrate Height, Temperature and Convert PANDORA channels from counts to volts if required.
		Radiosonde_Cal = Radiosonde_Checks(self.Radiosonde_Data.copy(), None, self.sensor_package, Height_Range, check=1111)
			
		#Calibrate Relative Humidity Sensor (e.g. find RH_ice)
		Radiosonde_Cal.RH()
		
		#Return Data (make local to function only. i.e. DON'T use self.Radiosonde_Data)
		Radiosonde_Data = Radiosonde_Cal.finalise()
		
		############################################################################
		"""[METHOD 1]: Relative Humidity (Zhang et al. 2010)"""
		
		#Define data into new variables
		Z = Radiosonde_Data[:,1]
		RH = Radiosonde_Data[:,-1]
		
		#Create Height-Resolving RH Thresholds (see Table 1 in Zhang et al. (2010))
		#N.B. use np.interp(val, RH_Thresholds['altitude'], RH_Thresholds['*RH']) where val is the height range you want the RH Threshold 
		RH_Thresholds = {'minRH' : [0.92, 0.90, 0.88, 0.75, 0.75],
			'maxRH' : [0.95, 0.93, 0.90, 0.80, 0.80],
			'interRH' : [0.84, 0.82, 0.78, 0.70, 0.70],
			'altitude' : [0, 2, 6, 12, 20]}
		
		#Define the cloud height levels as defined by WMO (2017). 
		Z_Levels = {'low' : [0,2], 'middle' : [2,7], 'high' : [5,13]}
		
		#Define the types of layers that can be detected.
		Cloud_Types = {0 : 'Clear Air', 1 : 'Moist (Not Cloud)', 2 : 'Cloud'}
		
		#Define the min, max and interRH for all measure altitudes
		minRH = np.interp(Z, RH_Thresholds['altitude'], RH_Thresholds['minRH'], left=np.nan, right=np.nan)*100
		maxRH = np.interp(Z, RH_Thresholds['altitude'], RH_Thresholds['maxRH'], left=np.nan, right=np.nan)*100
		interRH = np.interp(Z, RH_Thresholds['altitude'], RH_Thresholds['interRH'], left=np.nan, right=np.nan)*100
		
		#[Step 1]: The base of the lowest moist layer is determined as the level when RH exceeds the min-RH corresponding to this level
		minRH_mask = (RH > minRH)
		
		#[Step 2 and 3]: Above the base of the moist layer, contiguous levels with RH over the corresponding min-RH are treated as the same layer
		Z[~minRH_mask] = np.nan
		Clouds_ID = gu.contiguous(Z, 1)
		
		#[Step 4]: Moist layers with bases lower than 120m and thickness's less than 400m are discarded
		for Cloud in np.unique(Clouds_ID)[1:]:
			if Z[Clouds_ID == Cloud][0] < 0.12:
				if Z[Clouds_ID == Cloud][-1] - Z[Clouds_ID == Cloud][0] < 0.4:
					Clouds_ID[Clouds_ID == Cloud] = 0
		
		#[Step 5]: The moist layer is classified as a cloud layer is the maximum RH within this layer is greater than the corresponding max-RH at the base of this moist layer
		LayerType = np.zeros(Z.size, dtype=int) #0: Clear Air, 1: Moist Layer, 2: Cloud Layer
		for Cloud in np.unique(Clouds_ID)[1:]:
			if np.any(RH[Clouds_ID == Cloud] > maxRH[Clouds_ID == Cloud][0]):
				LayerType[Clouds_ID == Cloud] = 2
			else:
				LayerType[Clouds_ID == Cloud] = 1
		
		#[Step 6]: The base of the cloud layers is set to 280m AGL, and cloud layers are discarded if their tops are lower than 280m	
		for Cloud in np.unique(Clouds_ID)[1:]:
			if Z[Clouds_ID == Cloud][-1] < 0.280:
				Clouds_ID[Clouds_ID == Cloud] = 0
				LayerType[Clouds_ID == Cloud] = 0

		#[Step 7]: Two contiguous layers are considered as one-layer cloud if the distance between these two layers is less than 300m or the minimum RH within this distance is more than the maximum inter-RG value within this distance
		for Cloud_Below, Cloud_Above in zip(np.unique(Clouds_ID)[1:-1], np.unique(Clouds_ID)[2:]):
			
			#Define the index between clouds of interest
			Air_Between = np.arange(gu.bool2int(Clouds_ID == Cloud_Below)[-1], gu.bool2int(Clouds_ID == Cloud_Above)[0])
			
			if ((Z[Clouds_ID == Cloud_Above][0] - Z[Clouds_ID == Cloud_Below][-1]) < 0.3) or (np.nanmin(RH[Air_Between]) > np.nanmax(interRH[Air_Between])):
				Joined_Cloud_Mask = np.arange(gu.bool2int(Clouds_ID == Cloud_Below)[0], gu.bool2int(Clouds_ID == Cloud_Above)[-1])
				
				#Update the cloud ID array as the Cloud_Below and Cloud_Above are not distinct clouds
				Clouds_ID[Joined_Cloud_Mask] = Cloud_Below
				
				#Update the LayerType to reflect the new cloud merging
				if np.any(LayerType[Clouds_ID == Cloud_Below] == 2) or np.any(LayerType[Clouds_ID == Cloud_Above] == 2):
					LayerType[Joined_Cloud_Mask] = 2
				else:
					LayerType[Joined_Cloud_Mask] = 1
			
		#[Step 8] Clouds are discarded if their thickness's are less than 30.5m for low clouds and 61m for middle/high clouds
		for Cloud in np.unique(Clouds_ID)[1:]:
			if Z[Clouds_ID == Cloud][0] < Z_Levels['low'][1]:
				if Z[Clouds_ID == Cloud][-1] - Z[Clouds_ID == Cloud][0] < 0.0305:
					Clouds_ID[Clouds_ID == Cloud] = 0
					LayerType[Clouds_ID == Cloud] = 0

			else:
				if Z[Clouds_ID == Cloud][-1] - Z[Clouds_ID == Cloud][0] < 0.0610:
					Clouds_ID[Clouds_ID == Cloud] = 0
					LayerType[Clouds_ID == Cloud] = 0
		
		#Re-update numbering of each cloud identified
		Clouds_ID = gu.contiguous(Clouds_ID, invalid=0)
		
		#Output verbose to screen
		if self.verbose is True:
			print("Detected Clouds and Moist Layers\n--------------------------------")
			for Cloud in np.unique(Clouds_ID)[1:]:
				print("Cloud %s. Cloud Base = %.2fkm, Cloud Top = %.2fkm, Layer Type: %s" % (Cloud, Z[Clouds_ID == Cloud][0], Z[Clouds_ID == Cloud][-1], Cloud_Types[LayerType[Clouds_ID == Cloud][0]]))
			
		return Clouds_ID, LayerType
	
	def _ChargeCalibrator(self, Calibrate=None, Sensor_Package=None, Clouds_ID=None, LayerType=None):

		if self.verbose is True: gu.cprint("[INFO] You are running Radiosonde_ChargeCalibrator from the DEV release", type='bold')
		
		############################################################################
		"""Prerequisites"""
		
		#Time Controls
		t_begin = time.time()
			
		#Plotting requirements
		import matplotlib.pyplot as plt
		plt.style.use('classic') #necessary if Matplotlib version is >= 2.0.0
		
		#Calibration boundaries
		Height_Boundaries = {0 : [],
			1 : [],
			2 : [],
			3 : [],
			4 : [],
			5 : [10.5,12.0],
			6 : [],
			7 : [],
			8 : [],
			9 : [6,12.0],
			10 : [12,18.0]}
			
		############################################################################
		"""[Step 1] Calibrate bespoke sensors"""
		
		#Calibrate Height, Temperature and Convert PANDORA channels from counts to volts if required.
		Radiosonde_Cal = Radiosonde_Checks(self.Radiosonde_Data.copy(), Calibrate, Sensor_Package, Height_Boundaries[Sensor_Package], check=1111, linear=True, log=False)
			
		#Calibrate just the linear charge sensor
		Radiosonde_Cal.charge_calibration(Sensor_Package, type='current', linear=True, log=False)
		
		#Return Data
		Radiosonde_Data = Radiosonde_Cal.finalise()
			
		Linear = gu.moving_average(Radiosonde_Data[:,5], 11)
		Log = gu.moving_average(Radiosonde_Data[:,6], 11)
		
		PosMask = Linear >= 0
		NegMask = Linear < 0
		
		LinearPos = np.log10(Linear[PosMask])
		LogPos = Log[PosMask]
		
		LinearNeg = -np.log10(-Linear[NegMask])
		LogNeg = Log[NegMask]
			
		#Calculate Linear Regressions
		slope_all, intercept_all, r_value_all, p_value_all, std_err_all = sp.stats.linregress(Log, Linear)
		slope_pos, intercept_pos, r_value_pos, p_value_pos, std_err_pos = sp.stats.linregress(LogPos, LinearPos)
		try:
			slope_neg, intercept_neg, r_value_neg, p_value_neg, std_err_neg = sp.stats.linregress(LogNeg, LinearNeg)
		except:
			slope_neg, intercept_neg, r_value_neg, p_value_neg, std_err_neg = (0,0,0,0,0)
		
		if self.verbose is True: print(slope_all, intercept_all, r_value_all, p_value_all, std_err_all)
		if self.verbose is True: print(slope_pos, intercept_pos, r_value_pos, p_value_pos, std_err_pos)
		if self.verbose is True: print(slope_neg, intercept_neg, r_value_neg, p_value_neg, std_err_neg)
		
		############################################################################
		"""[Step 2] Plot the calibration values for positive and negative linear currents"""
		
		plt.clf()
		plt.close()
		
		f, ax = plt.subplots(1,3)
		ax[0].plot(Log, Linear , 'p', ms=1, marker='o', markeredgecolor='None', markerfacecolor='black', alpha=1, label="Clouds")
		ax[1].plot(LogPos, LinearPos , 'p', ms=1, marker='o', markeredgecolor='None', markerfacecolor='black', alpha=1, label="Clouds")
		ax[2].plot(LogNeg, LinearNeg , 'p', ms=1, marker='o', markeredgecolor='None', markerfacecolor='black', alpha=1, label="Clouds")
		
		ax[0].plot(Log, slope_all*Log+intercept_all, lw=0.5, c='red')
		ax[1].plot(LogPos, slope_pos*LogPos+intercept_pos, lw=0.5, c='red')
		ax[2].plot(LogNeg, slope_neg*LogNeg+intercept_neg, lw=0.5, c='red')
		
		ax[0].set_ylabel("Linear Sensor Current (A)", fontsize=8)
		ax[1].set_ylabel("Linear Sensor Current (log10(pA))", fontsize=8)
		ax[2].set_ylabel("Linear Sensor Current (-log10(-pA))", fontsize=8)

		for subplot in ax: subplot.minorticks_on()
		for subplot in ax: subplot.set_xlabel("Log Sensor Current (Counts)", fontsize=8)
		for subplot in ax: subplot.grid(which='major',axis='both',c='grey')
		for subplot in ax: subplot.tick_params(axis='both', which='major', labelsize=8)
		for subplot in ax: subplot.tick_params(axis='both', which='minor', labelsize=8)
		
		f.suptitle("Linear and Log Charge Sensors for Radiosonde Flight No.5", y=0.90)
		
		ax[0].get_xaxis().get_major_formatter().labelOnlyBase = False
		
		for subplot in ax:
			x0, x1 = subplot.get_xlim()
			y0, y1 = subplot.get_ylim()
			subplot.set_aspect(np.abs((x1-x0)/(y1-y0)))
		
		ax[0].annotate("All Data", xy=(0, 1), xycoords='axes fraction', xytext=(20, -20), textcoords='offset pixels', horizontalalignment='left', verticalalignment='top', fontsize=8)
		ax[1].annotate("Positive Linear Current", xy=(0, 1), xycoords='axes fraction', xytext=(20, -20), textcoords='offset pixels', horizontalalignment='left', verticalalignment='top', fontsize=8)
		ax[2].annotate("Negative Linear Current", xy=(0, 1), xycoords='axes fraction', xytext=(20, -20), textcoords='offset pixels', horizontalalignment='left', verticalalignment='top', fontsize=8)
		
		ax[0].annotate("$R^{2}$ = %.4f\n$Counts$ = %.0f" % (r_value_all**2, Log.size), xy=(1, 1), xycoords='axes fraction', fontsize=8, xytext=(-3, -3), textcoords='offset points', ha='right', va='top')
		ax[1].annotate("$R^{2}$ = %.4f\n$Counts$ = %.0f" % (r_value_pos**2, LogPos.size), xy=(1, 1), xycoords='axes fraction', fontsize=8, xytext=(-3, -3), textcoords='offset points', ha='right', va='top')
		ax[2].annotate("$R^{2}$ = %.4f\n$Counts$ = %.0f" % (r_value_neg**2, LogNeg.size), xy=(1, 1), xycoords='axes fraction', fontsize=8, xytext=(-3, -3), textcoords='offset points', ha='right', va='top')
					
		f.set_size_inches(11.7, 4.3)
		
		############################################################################
		"""[Step 3] Save plot to file"""
		
		#Specify the directory the plots are stored in 
		path = os.path.dirname(self.Radiosonde_File).replace(self.Storage_Path + self.Processed_Data_Path,"")

		#Find any other plots stored in this directory
		previous_plots = glob.glob(self.Storage_Path + self.Radiosonde_Plots_Path + path + "/*")
		
		#Find the biggest 'v' number in plots
		plot_version = []
		for plots in previous_plots:
			try:
				plot_version.append(int(os.path.basename(plots)[34:37]))
			except ValueError:
				plot_version.append(int(os.path.basename(plots)[34:36]))
		
		plot_version = str(np.max(plot_version)+1) if len(plot_version) != 0 else '1'
		
		#Create full directory and file name
		Save_Location = self.Storage_Path + self.Radiosonde_Plots_Path + path + '/' + path + '_v' + plot_version.rjust(2,'0') + '_ChargeCalibrator.png'

		#Ensure the directory exists on file system and save to that location
		gu.ensure_dir(os.path.dirname(Save_Location))
		plt.savefig(Save_Location, bbox_inches='tight', pad_inches=0.1, dpi=300)
		
		#Return regression of positive current, regression of negative current and the boundary for counts
		return (slope_all, intercept_all), (slope_pos, intercept_pos), (slope_neg, intercept_neg), (PosMask, NegMask)
		
	def Superplotter(self):
		"""This function will plot the data from a single radiosonde flight
		
		Parameters
		----------
		
		Clouds_ID : 
	
		LayerType :
		
		Calibration_Log : 2x2 tuple or array, optional
			Used to calculate the space charge density using the log sensor. Due to
			the temperature drift of the log sensor, but the wide range of measurements,
			the log sensor needs to be calibrate with the linear sensor first. Use the
			output from Radiosonde_ChargeCalibrator to populate this parameter.
		
		"""
		
		if self.verbose is True: gu.cprint("[INFO] You are running Superplotter from the DEV release", type='bold')
		
		############################################################################
		"""Prerequisites"""
		
		t_begin = time.time()

		############################################################################
		"""[Step 1] Calibrate bespoke sensors"""
		
		#Calibrate Height, Temperature and Convert PANDORA channels from counts to volts if required.
		if self.Calibration_Log is None:
			Radiosonde_Cal = Radiosonde_Checks(self.Radiosonde_Data.copy(), self.calibrate, self.sensor_package, self.height_range, check=1111)
		else:
			Radiosonde_Cal = Radiosonde_Checks(self.Radiosonde_Data.copy(), self.calibrate, self.sensor_package, self.height_range, check=1111, linear=True, log=False)
		
		#Calibrate OMB Sensor
		if (self.calibrate == "units") & (self.sensor_package < 8): Radiosonde_Cal.wire_calibration()
		
		#Calibrate Cloud Sensor
		if self.calibrate == "units": 
			if self.sensor_package < 8:
				Radiosonde_Cal.cloud_calibration(self.sensor_package, check=1111)
			else:
				Radiosonde_Cal.cloud_calibration(self.sensor_package, check=None)
				
		#Calibrate Charge Sensor
		if self.calibrate == "units": 
			if self.Calibration_Log is None:
				#Space charge density of linear sensor
				Radiosonde_Cal.charge_calibration(self.sensor_package, type='space_charge', linear=True)
			
			else:
				#Space charge density of log sensor
				
				#Get Log Charge Counts and the Time (used to correctly sort data after calibration
				Radiosonde_Time = Radiosonde_Cal.data[:,0].copy()
				Radiosonde_Height = Radiosonde_Cal.data[:,1].copy()
				LogCharge_Counts = Radiosonde_Cal.data[:,6].copy()
				
				if self.verbose is True: print("LogCharge_Counts", LogCharge_Counts)
				
				if self.verbose is True: print("LogCharge_Counts Stats", gu.stats(LogCharge_Counts))
				
				#Get the space charge density for the lab based calibration of log sensor
				Radiosonde_Cal.charge_calibration(self.sensor_package, type='space_charge', lab_calibration=True, linear=False, log=True)
				LogCharge_LabCal = Radiosonde_Cal.data[:,6].copy()
				
				#Calibrate just the linear charge sensor to determine boundaries in Calibration (e.g. We calibrate 
				PosMask = Radiosonde_Cal.data[:,5] > 0
				NegMask = Radiosonde_Cal.data[:,5] < 0
				
				if self.verbose is True: print("PosMask", np.sum(PosMask))
				if self.verbose is True: print("NegMask", np.sum(NegMask))	
				
				if self.verbose is True: print("Slope", self.Calibration_Log[0][0], "Intercept", self.Calibration_Log[0][1])
				if self.verbose is True: print("MAX", np.nanmax(LogCharge_Counts), self.Calibration_Log[0][0]*np.nanmax(LogCharge_Counts)+Calibration_Log[0][1])
				LogCharge_LinCal = self.Calibration_Log[0][0] * LogCharge_Counts + self.Calibration_Log[0][1]
				
				if self.verbose is True: print("LogCharge_LinCal Stats", gu.stats(LogCharge_LinCal))
				
				#Calculate Space Charge Density
				LogCharge_LinCal = Radiosonde_Cal._space_charge(Radiosonde_Time, Radiosonde_Height, LogCharge_LinCal, lab_calibration=True)
				
				#Subset LogCharge_LabCal and LogCharge_LinCal for correct height range
				LogCharge_LabCal = LogCharge_LabCal[(Radiosonde_Height >= self.height_range[0]) & (Radiosonde_Height <= self.height_range[1])]
				LogCharge_LinCal = LogCharge_LinCal[(Radiosonde_Height >= self.height_range[0]) & (Radiosonde_Height <= self.height_range[1])]
								
		#Calibrate Relative Humidity Sensor (e.g. find RH_ice)
		Radiosonde_Cal.RH()
		
		#Return Data
		Radiosonde_Data = Radiosonde_Cal.finalise()
		
		############################################################################
		"""[Step 2] Plot radiosonde data"""

		Title = 'Radiosonde Flight No.' + str(self.sensor_package) + ' (' + self.Launch_Datetime.strftime("%d/%m/%Y %H%MUTC") + ')' if self.GPS_File is not None else 'Radiosonde Flight (N/A)'
		if self.sensor_package < 8:
			Superplotter = SPRadiosonde(8, Title, self.height_range, Radiosonde_Data, calibrate=self.calibrate) if self.calibrate == "units" else SPRadiosonde(7, Title, self.height_range, Radiosonde_Data, calibrate=self.calibrate)
		else:	
			Superplotter = SPRadiosonde(7, Title, self.height_range, Radiosonde_Data, calibrate=self.calibrate)
				
		if self.Calibration_Log is None:
			Superplotter.Charge(Linear_Channel=0, Log_Channel=1)
		else:
			Superplotter.Charge(Linear_Channel=None, Log_Channel=(LogCharge_LabCal, LogCharge_LinCal), Calibration_Log=True)
		
		if self.sensor_package < 8:
			Superplotter.Cloud(Cyan_Channel=2, IR_Channel=3, Cyan_Check=1111)
			Superplotter.PLL(PLL_Channel=2, PLL_Check=1112, Point=False, Calibrate=self.calibrate) if self.sensor_package < 3 else Superplotter.PLL(PLL_Channel=2, PLL_Check=1112, Point=True, Calibrate=self.calibrate)
		else:
			Superplotter.Cloud(Cyan_Channel=2, IR_Channel=3)
			Superplotter.Turbulence(Turbulence_Channel=4)
			
		#Plot the processed PLL data
		if (self.calibrate == "units") & (self.sensor_package < 8): Superplotter.ch(14, 'SLWC $(g$ $m^{-3})$', 'Supercooled Liquid\nWater Concentration', check=1112, point=True)
		
		#Plot the cloud boundaries if specified
		if self.Clouds_ID is not None: Superplotter.Cloud_Boundaries(self.Clouds_ID, self.LayerType, CloudOnly=True)
		
		############################################################################
		"""[Step 3] Save plot and return"""
		
		#Specify the directory the plots are stored in 
		path = os.path.dirname(self.Radiosonde_File).replace(self.Storage_Path + self.Processed_Data_Path,"")
		
		#Find any other plots stored in this directory
		previous_plots = glob.glob(self.Storage_Path + self.Radiosonde_Plots_Path + path + "/*")
		
		#Find the biggest 'v' number in plots
		plot_version = []
		for plots in previous_plots:
			try:
				plot_version.append(int(os.path.basename(plots)[34:37]))
			except ValueError:
				plot_version.append(int(os.path.basename(plots)[34:36]))
		
		plot_version = str(np.max(plot_version)+1) if len(plot_version) != 0 else '1'
		
		#Create full directory and file name
		Save_Location = self.Storage_Path + self.Radiosonde_Plots_Path + path + '/' + path + '_v' + plot_version.rjust(2,'0') + '_' + str(self.height_range[0]).rjust(2,'0') + 'km_to_' + str(self.height_range[1]).rjust(2,'0') + 'km.png'
		
		#Ensure the directory exists on file system and save to that location
		gu.ensure_dir(os.path.dirname(Save_Location))
		Superplotter.savefig(Save_Location)
		
		if self.verbose is True: print("[INFO] Superplotter completed successfully (In %.2fs)" % (time.time()-t_begin))

	def RH_Comparison(self):
		"""This function will plot the relative humidity with respects to ice using various
		calculations of the saturated vapour pressure
		
		"""
		
		if self.verbose is True: gu.cprint("[INFO] You are running RH_Comparison from the DEV release", type='bold')
		
		############################################################################
		"""Prerequisites"""
		
		#Time Controls
		t_begin = time.time()
				
		############################################################################
		"""[Step 1] Calibrate bespoke sensors"""

		#Calibrate Height, Temperature and Convert PANDORA channels from counts to volts if required.
		Radiosonde_Cal = Radiosonde_Checks(self.Radiosonde_Data.copy(), None, self.sensor_package, self.height_range, check=1111)
		Radiosonde_Cal_Goff = Radiosonde_Checks(self.Radiosonde_Data.copy(), None, self.sensor_package, self.height_range, check=1111)
		Radiosonde_Cal_Buck = Radiosonde_Checks(self.Radiosonde_Data.copy(), None, self.sensor_package, self.height_range, check=1111)
		Radiosonde_Cal_Wexler = Radiosonde_Checks(self.Radiosonde_Data.copy(), None, self.sensor_package, self.height_range, check=1111)
		Radiosonde_Cal_Sonntag = Radiosonde_Checks(self.Radiosonde_Data.copy(), None, self.sensor_package, self.height_range, check=1111)
	
		#Calibrate Relative Humidity Sensor (e.g. find RH_ice)
		Radiosonde_Cal_Goff.RH(method='goff')
		Radiosonde_Cal_Buck.RH(method='arden-buck')
		Radiosonde_Cal_Wexler.RH(method='wexler')
		Radiosonde_Cal_Sonntag.RH(method='sonntag')
		
		#Return Data (make local to function only. i.e. DON'T use self.Radiosonde_Data)
		Radiosonde_Data = Radiosonde_Cal.finalise()
		Radiosonde_Data_Goff = Radiosonde_Cal_Goff.finalise()
		Radiosonde_Data_Buck = Radiosonde_Cal_Buck.finalise()
		Radiosonde_Data_Wexler = Radiosonde_Cal_Wexler.finalise()
		Radiosonde_Data_Sonntag = Radiosonde_Cal_Sonntag.finalise()
		
		############################################################################
		"""[Step 2] Plot radiosonde data"""
		
		Title = 'Radiosonde Flight No.' + str(self.sensor_package) + ' (' + self.Launch_Datetime.strftime("%d/%m/%Y %H%MUTC") + ')' if self.GPS_File is not None else 'Radiosonde Flight (N/A)'
		
		Height = Radiosonde_Data[:,1]
		Temperature = Radiosonde_Data[:,3]
		RH = Radiosonde_Data[:,4]
		
		#Plotting requirements
		plt.style.use('classic') #necessary if Matplotlib version is >= 2.0.0

		#Make sure we are creating new plot from scratch
		plt.clf()
		plt.close()
		
		#Define number of subplots sharing y axis
		f, ax1 = plt.subplots()

		ax1.minorticks_on()
		ax1.grid(which='major',axis='both',c='grey')

		#Rotate xticks in all subplots
		for tick in ax1.get_xticklabels(): tick.set_rotation(90)
		
		#Remove random junk from the plot
		f.subplots_adjust(hspace=0)
		plt.setp([a.get_yticklabels() for a in f.axes[1:]], visible=False)
		
		#Set axis parameters
		ax1.set_ylabel('Height $(km)$')
		ax1.set_ylim([np.nanmin(Radiosonde_Data[:,1]), np.nanmax(Radiosonde_Data[:,1])])
		
		#Define plot size
		f.set_size_inches(8, 8)
		
		#Plot RH
		ax1.plot(RH, Height, label='Original', lw=0.5)
		ax1.plot(Radiosonde_Data_Goff[:,-1], Radiosonde_Data_Goff[:,1], label='Goff-Gratch', lw=0.5)
		ax1.plot(Radiosonde_Data_Buck[:,-1], Radiosonde_Data_Buck[:,1], label='Arden-Buck', lw=0.5)
		ax1.plot(Radiosonde_Data_Wexler[:,-1], Radiosonde_Data_Wexler[:,1], label='Wexler', lw=0.5)
		ax1.plot(Radiosonde_Data_Sonntag[:,-1], Radiosonde_Data_Sonntag[:,1], label='Sonntag', lw=0.5)
		
		ax1.set_xlabel('RH $(\%)$')
		ax1.set_title(Title, fontsize=12)
		ax1.legend(loc='best')
		
		ax2 = ax1.twinx()

		ax2.plot(RH, Temperature, label='Original', lw=0.5, c='black')
		
		ax2.set_ylabel('Temperature ($^\circ$C)')
		
		ax2.set_ylim([np.nanmin(Temperature), np.nanmax(Temperature)])
		ax2.invert_yaxis()
		
		Freezing_Height = Height[gu.argnear(Temperature, 0)] if np.any(Temperature < 0) else -1
		print("Freezing_Height", Freezing_Height)
		ax1.axhline(y=Freezing_Height, c='black', ls='-', lw=1)
		
		############################################################################
		"""[Step 3] Save plot and return"""
		
		#Specify the directory the plots are stored in 
		path = os.path.dirname(self.Radiosonde_File).replace(self.Storage_Path + self.Processed_Data_Path,"")
		
		#Find any other plots stored in this directory
		previous_plots = glob.glob(self.Storage_Path + 'Plots/RH_Comparison/' + path + "/*")
		
		#Find the biggest 'v' number in plots
		plot_version = []
		for plots in previous_plots:
			try:
				plot_version.append(int(os.path.basename(plots)[34:37]))
			except ValueError:
				plot_version.append(int(os.path.basename(plots)[34:36]))
		
		plot_version = str(np.max(plot_version)+1) if len(plot_version) != 0 else '1'
		
		#Create full directory and file name
		Save_Location = self.Storage_Path + 'Plots/RH_Comparison/' + path + '/' + path + '_v' + plot_version.rjust(2,'0') + '_' + str(self.height_range[0]).rjust(2,'0') + 'km_to_' + str(self.height_range[1]).rjust(2,'0') + 'km.png'
		
		#Ensure the directory exists on file system and save to that location
		gu.ensure_dir(os.path.dirname(Save_Location))
		plt.savefig(Save_Location, bbox_inches='tight', pad_inches=0.1, dpi=300)
		
		############################################################################
		
		if self.verbose is True: print("[INFO] RH_Comparison completed successfully (In %.2fs)" % (time.time()-t_begin))

	def Tephigram(self, plot_tephigram=False, plot_camborne=False):
		"""The Radiosonde_Tephigram function will plot a tephigram from the dry bulb temperature,
		T_dry and the Dew point Temperature, T_dew for pressure values, P at each corresponding 
		height. 
		
		Certain tephigram outputs are available from this function including:
		1) Lower Condensation Level (LCL) in m
		2) Level of Free Convection (LFC) in m
		3) Environmental Level (EL) in m
		4) Convective Available Potential Energy (CAPE) in J/kg
		5) Convective INhibition (CIN) in J/kg

		Parameters
		----------
		
		plot_tephigram : bool, optional, default is False
			Specify True to plot a tephigram of the sounding data. Otherwise
			just calculate the sounding indices
		plot_camborne : bool, optional, default is False
			Specify True to add the sounding from Camborne at the closest time
			to the launch time. Only used if plot_tephigram is True.
		
		Outputs
		-------
		
		References
		----------
		Ambaum, M. H. P., 2010. Water in the Atmosphere. In: Thermal Physics of the Atmosphere. Oxford: Wiley & Sons, pp. 93-109
		Marlton, G. 2018. Tephigram. Original Matlab code found in Matlab_Code directory
		Hunt, K. 2018. Tephigram. Original Python code found in the same directory.
		"""
		
		if self.verbose is True: gu.cprint("[INFO] You are running Radiosonde_Tephigram from the STABLE release", type='bold')
			
		############################################################################
		"""Prerequisites"""
		
		#Time Controls
		t_begin = time.time()
		
		#Set-up data importer
		EPCC_Data = EPCC_Importer()
		

		############################################################################
		"""[Step 1] Calibrate bespoke sensors"""
    
		#Calibrate Height, Temperature and Convert PANDORA channels from counts to volts if required.
		Radiosonde_Cal = Radiosonde_Checks(self.Radiosonde_Data.copy(), calibrate=None, package_no=self.sensor_package, height_range=[0,12], check=1111)
		
		#Return Data (make local to function only. i.e. DON'T use self.Radiosonde_Data)
		Radiosonde_Data = Radiosonde_Cal.finalise()
		
		Z = Radiosonde_Data[:,1]
		Tdry = Radiosonde_Data[:,3]
		Tdew = Radiosonde_Data[:,14]
		Pres = Radiosonde_Data[:,2]
		RH = Radiosonde_Data[:,4]/100; RH -= np.max(RH) - 0.01
		Wind_Mag = (Radiosonde_Data[:,15]**2 + Radiosonde_Data[:,16]**2)**0.5
		Wind_Dir = np.arctan2(Radiosonde_Data[:,15], Radiosonde_Data[:,16]) * 180 / np.pi
			
		############################################################################
		"""[Step 2] Create Tephigram"""
		
		if plot_tephigram is True:
		
			print("[INFO] Plotting Tephigram...")
		
			#Mask nan data (ONLY FOR PLOTTING)
			Radiosonde_Data_Plotting = gu.antinan(Radiosonde_Data.T)
			
			#Unpack variables
			Z_Plot = Radiosonde_Data[:,1]
			Tdry_Plot = Radiosonde_Data[:,3]
			Tdew_Plot = Radiosonde_Data[:,14]
			Pres_Plot = Radiosonde_Data[:,2]
			
			#Subset the tephigram to specified location
			locator = gu.argneararray(Z_Plot, np.array(self.height_range)*1000)
			anchor = np.array([(Pres_Plot[locator]),(Tdry_Plot[locator])]).T
			
			Pres_Plot_Antinan, Tdry_Plot_Antinan, Tdew_Plot_Antinan = gu.antinan(np.array([Pres_Plot, Tdry_Plot, Tdew_Plot]), unpack=True)
			
			#Group the dews, temps and wind profile measurements
			dews = zip(Pres_Plot_Antinan, Tdew_Plot_Antinan)
			temps = zip(Pres_Plot_Antinan, Tdry_Plot_Antinan)
			barb_vals = zip(Pres,Wind_Dir,Pres_Plot)
					
			#Create Tephigram plot
			Tephigram = SPTephigram()
			
			if plot_camborne is True:
				
				#Determine ULS data
				ULS_File = sorted(glob.glob(PhD_Global.Raw_Data_Path + 'Met_Data/ULS/*'))
				
				ULS_Date = np.zeros(len(ULS_File), dtype=object)
				for i, file in enumerate(ULS_File):
					try:
						ULS_Date[i] = datetime.strptime(os.path.basename(file), '%Y%m%d_%H_03808_UoW_ULS.csv')
					except:
						ULS_Date[i] = datetime(1900,1,1)
				
				#Find Nearest Upper Level Sounding Flight to Radiosonde Flight
				ID = gu.argnear(ULS_Date, self.Launch_Datetime)
				
				print("[INFO] Radiosonde Launch Time:", self.Launch_Datetime, "Camborne Launch Time:", ULS_Date[ID])
				
				#Import Camborne Radiosonde Data
				press_camborne, temps_camborne, dews_camborne = EPCC_Data.ULS_Calibrate(ULS_File[ID], unpack=True, PRES=True, TEMP=True, DWPT=True)
				
				#Match Camborne pressures with Reading pressures
				mask = [gu.argnear(press_camborne, Pres_Plot[0]), gu.argnear(press_camborne, Pres_Plot[-1])]
				press_camborne = press_camborne[mask[0]:mask[1]]
				temps_camborne = temps_camborne[mask[0]:mask[1]]
				dews_camborne = dews_camborne[mask[0]:mask[1]]
					
				dews_camborne = zip(press_camborne, dews_camborne)
				temps_camborne = zip(press_camborne, temps_camborne)
				
				#Plot Reading Radiosonde
				profile_t1 = Tephigram.plot(temps, color="red", linewidth=1, label='Reading Dry Bulb Temperature', zorder=5)
				profile_d1 = Tephigram.plot(dews, color="blue", linewidth=1, label='Reading Dew Bulb Temperature', zorder=5)
				
				#Plot Camborne Radiosonde
				profile_t1 = Tephigram.plot(temps_camborne, color="red", linestyle=':', linewidth=1, label='Camborne Dry Bulb Temperature', zorder=5)
				profile_d1 = Tephigram.plot(dews_camborne, color="blue", linestyle=':', linewidth=1, label='Camborne Dew Bulb Temperature', zorder=5)
			else:	
				profile_t1 = Tephigram.plot(temps, color="red", linewidth=2, **{'label':'Dry Bulb Temperature'})
				profile_d1 = Tephigram.plot(dews, color="blue", linewidth=2, **{'label':'Dew Bulb Temperature'})
			
			#Add extra information to Tephigram plot
			#Tephigram.axes.set(title=Title, xlabel="Potential Temperature $(^\circ C)$", ylabel="Dry Bulb Temperature $(^\circ C)$")
			Title = 'Radiosonde Tephigram Flight No.' + str(self.sensor_package) + ' (' + self.Launch_Datetime.strftime("%d/%m/%Y %H%MUTC") + ')' if self.GPS_File is not None else 'Radiosonde Tephigram Flight (N/A)'
			Tephigram.axes.set(title=Title)
					
			#[OPTIONAL] Add wind profile information to Tephigram.
			#profile_t1.barbs(barb_vals)
			
			############################################################################
			"""Save plot to file"""

			#Specify the directory the plots are stored in 
			path = os.path.dirname(self.Radiosonde_File).replace(self.Storage_Path + self.Processed_Data_Path,"")
			
			#Find any other plots stored in this directory
			previous_plots = glob.glob(self.Storage_Path + self.Tephigram_Plots_Path + path + "/*")
			
			#Find the biggest 'v' number in plots
			plot_version = []
			for plots in previous_plots:
				try:
					plot_version.append(int(os.path.basename(plots)[34:37]))
				except ValueError:
					plot_version.append(int(os.path.basename(plots)[34:36]))
			
			plot_version = str(np.max(plot_version)+1) if len(plot_version) != 0 else '1'
			
			#Create full directory and file name
			Save_Location = self.Storage_Path + self.Tephigram_Plots_Path + path + '/' + path + '_v' + plot_version.rjust(2,'0') + '_' + str(self.height_range[0]).rjust(2,'0') + 'km_to_' + str(self.height_range[1]).rjust(2,'0') + 'km.png'
			
			#Ensure the directory exists on file system and save to that location
			gu.ensure_dir(os.path.dirname(Save_Location))
			
			Tephigram.savefig(Save_Location)      

		############################################################################
		"""[Step 3] Calculate Stability Indices"""
		
		print("[INFO] Calculating Stability Indices...")
		
		#Common Pressure Levels
		P_500 = gu.argnear(Pres, 500)
		P_700 = gu.argnear(Pres, 700)
		P_850 = gu.argnear(Pres, 850)
		
		#Showalter stability index
		#S = Tdry[P_500] - Tl
		
		#K-Index
		K = (Tdry[P_850] - Tdry[P_500]) + Tdew[P_850] - (Tdry[P_700] - Tdew[P_700])
		
		#Cross Totals Index
		CT = Tdew[P_850] - Tdry[P_500]
		
		#Vertical Totals Index
		VT = Tdry[P_850] - Tdry[P_500]
		
		#Total Totals Index
		TT = VT + CT
		
		#SWEAT Index
		ms2kn = 1.94384	#Conversion between m/s to knots
			
		SW_1 = 20*(TT-49)
		SW_2 = 12*Tdew[P_850]
		SW_3 = 2*Wind_Mag[P_850]*ms2kn
		SW_4 = Wind_Mag[P_500]*ms2kn
		SW_5 = 125*(np.sin(Wind_Dir[P_500]-Wind_Dir[P_850]) + 0.2)
		
		#Condition SWEAT Term 1 from several conditions
		SW_1 = 0 if SW_1 < 49 else SW_1

		#Condition SWEAT Term 5 with several conditions
		if (Wind_Dir[P_850] > 130) & (Wind_Dir[P_850] < 250):
			if (Wind_Dir[P_500] > 210) & (Wind_Dir[P_500] < 310):
				if Wind_Dir[P_500]-Wind_Dir[P_850] > 0:
					if (Wind_Mag[P_500]*ms2kn > 15) & (Wind_Mag[P_850]*ms2kn > 15):
						SW_5 = SW_5
					else:
						SW_5 = 0
				else:
					SW_5 = 0
			else:
				SW_5 = 0
		else:
			SW_5 = 0
		
		#Calulate Final Product
		SW = SW_1 + SW_2 + SW_3 + SW_4 + SW_5
		
		print("Stability Indices")
		print("-----------------")
		print("K-Index:", K)
		print("Cross Totals Index:", CT)
		print("Vettical Totals Index:", VT)
		print("Total Totals Index:", TT)
		print("SWEAT Index:", SW)
		print("\n")
		
		############################################################################
		"""[Step 4] Calculate Tephigram Indices"""
		
		#Convert Temperature back to Kelvin
		Tdry += 273.15
		Tdew += 273.15
		
		#Convert Height into metres
		Z *= 1000
		
		#Constants
		over27 = 0.286 # Value used for calculating potential temperature 2/7
		L = 2.5e6  #Latent evaporation 2.5x10^6
		epsilon = 0.622
		E = 6.014  #e in hpa 
		Rd = 287 #R constant for dry air
		
		#Equations
		es = lambda T: 6.112*np.exp((17.67*(T-273.15))/(T-29.65)) #Teten's Formula for Saturated Vapour Pressure converted for the units of T in Kelvin rather than Centigrade
		
		#Calculate Theta and Theta Dew
		theta = Tdry*(1000/Pres)**over27
		thetadew = Tdew*(1000/Pres)**over27
			
		#Find the Lifting Condensation Level (LCL)
		qs_base = 0.622*es(Tdew[0])/Pres[0]
		
		theta_base = theta[0]
		Pqs_base = 0.622*es(Tdry)/qs_base  #Calculate a pressure for constant qs
		Pqs_base = Tdry*(1000/Pqs_base)**(2/7) #Calculates pressure in term of P temp
		
		#print("Tdew[0]", Tdew[0])
		#print("Pres[0]", Pres[0])
		#print("qs_base",qs_base)
		#print("theta_base", theta_base)
		#print("Pqs_base", Pqs_base)
		
		#Find first location where Pqs_base > theta_base
		y1 = np.arange(Pqs_base.size)[Pqs_base > theta_base][0]
		#print(Pqs_base[y1])
		#print(y1)
		#print(gu.argnear(Pqs_base, theta_base))
		
		LCL = Z[y1]
			
		Tarr = np.zeros(Tdry.size)
		thetaarr = np.zeros(Tdry.size)

		T_temp = Tdry[y1]
		P_temp = 1000*(T_temp/Pqs_base[y1])**3.5
		qs0 = 0.622*es(T_temp)/P_temp
		
		thetaarr[y1] = Pqs_base[y1]
		Tarr[y1] = T_temp
		
		for i in xrange(y1+1, y1+100):
			T_temp -= 1
			P_temp = 1000*(T_temp/thetaarr[i-1])**3.5
			qs = 0.622*es(T_temp)/P_temp
			thetaarr[i] = thetaarr[i-1] - ((2.5E6/1004)*(thetaarr[i-1]/T_temp) * (qs-qs0))
			qs0 = qs
			Tarr[i] = T_temp
			
		#Now need to integrate back to 1000hpa
		T_temp = Tdry[y1]
		P_temp = 1000*(T_temp/Pqs_base[y1])**3.5
		qs0 = 0.622*es(T_temp)/P_temp
		thetaarr[y1] = Pqs_base[y1]
		Tarr[y1] = T_temp
		for i in xrange(y1-1):
			T_temp += 1
			P_temp = 1000*(T_temp/thetaarr[(y1-i)+1])**3.5
			qs = 0.622*es(T_temp)/P_temp
			thetaarr[y1-i] = thetaarr[(y1-i)+1] - ((2.5E6/1004)*(thetaarr[(y1-i)+1]/T_temp) * (qs-qs0))
			qs0 = qs
			Tarr[y1-i] = T_temp
		
		y8 = (thetaarr>253) & (thetaarr<380)
		thetaarr = thetaarr[y8]
		Tarr = Tarr[y8]
		
		#Now find environmental levels and LFC begin by converting thetaarr into P
		Pthetaeq = 1000/(thetaarr/Tarr)**3.5
		l5 = np.isnan(Pthetaeq)
		Pthetaeq[l5] = []
		
		#Now interpolate on to rs height co-ordinates	
		TEMP = sp.interpolate.interp1d(Pthetaeq,[thetaarr,Tarr], fill_value="extrapolate")(Pres)
		thetaarr = TEMP[0]
		Tarr = TEMP[1]

		del(TEMP)
			
		y5 = np.arange(Tdry.size)[Tdry < Tarr]

		if np.any(y5):
			LFC = Z[y5[0]]
			EL = Z[y5[-1]]
			
			#Finds CIN area above LCL
			y6 = np.arange(Tdry.size)[(Z < LFC) & (Z >= LCL) & (Tdry > Tarr)]
			y7 = np.arange(Tdry.size)[(Z < LCL) & (Tdry > Tarr)]
			
			Pstart = Pres[y5[-1]]
			
			#Now need to calculate y5 temperatures into virtual temperatures
			Tvdash = Tarr/(1-(E/Pres)*(1-epsilon))
			Tv = Tdry/(1-(E/Pres)*(1-epsilon))
			T_adiabat = ((theta_base/(1000/Pres)**over27))
			Tv_adiabat = T_adiabat/(1-(E/Pres)*(1-epsilon))
			
			#Now need to calculate CAPE... and CIN to use CAPE = R_d = intergral(LFC,EL)(T'_v - T_v) d ln p
			CAPE = 0
			for i in xrange(y5[-2], y5[0], -1):
				CAPE += (Rd*(Tvdash[i] - Tv[i]) * np.log(Pres[i]/Pres[i+1]));
			
			#Now we use same technique to calculate CIN
			CIN=0;
			if len(y6) != 0:
				for i in xrange(y6[-2], y6[0], -1):
					CIN += (Rd*(Tvdash[i] - Tv[i]) * np.log(Pres[i]/Pres[i+1]))
		
			#Now calculate temperature along the dry adiabat
			y7 = np.arange(Tdry.size)[(Z < LCL) & (Tv > Tv_adiabat)]
			if len(y7) != 0:
				for i in xrange(y7[-2], y7[0], -1): 
					CIN += (Rd*(Tv_adiabat[i] - Tv[i]) * np.log(Pres[i]/Pres[i+1]));   
			
		else:
			LFC = np.nan
			EL = np.nan
			CAPE = 0
			CIN = 0
		
		#Print out information
		print("Parcel Information")
		print("------------------")
		print("LCL = %.2fm" % LCL)
		print("LFC = %.2fm" % LFC)
		print("EL = %.2fm" % EL)
		print("CAPE %.2f J/kg" % CAPE)
		print("CIN %.2f J/kg" % CIN)
		print("\n")
		
		print("[INFO] Radiosonde_Tephigram has been completed successfully (In %.2fs)" % (time.time()-t_begin))
		
		return LCL, LFC, EL, CAPE, CIN		
	
def Radiosonde_Superplotter(Radiosonde_File=None, Calibrate=None, Height_Range=None, Sensor_Package=None, Clouds_ID=None, LayerType=None, Calibration_Log=None):
	"""This function will plot the data from a single radiosonde flight
    
    Parameters
    ----------
    
    Radiosonde_File : str, optional
        Location of the radiosonde file to be processed and plotted
	Calibrate : tuple, optional
		tuple of column numbers which require calibration using the
		calibration metric y = x*(5/4096)
	Height_Range : tuple, optional
		lower and upper limits of the height you want to plot
	Sensor_Package : int, optional
		specify the package number related to the data. This is then used
		to calibrate the charge sensor
	Clouds_ID : 
	
	LayerType :
	
	Calibration_Log : 2x2 tuple or array, optional
		Used to calculate the space charge density using the log sensor. Due to
		the temperature drift of the log sensor, but the wide range of measurements,
		the log sensor needs to be calibrate with the linear sensor first. Use the
		output from Radiosonde_ChargeCalibrator to populate this parameter.
	
	"""

	gu.cprint("[INFO] You are running Radiosonde_Superplotter from the DEV release", type='bold')
    
    ############################################################################
	"""Prerequisites"""
    
	#Time Controls
	t_begin = time.time()
		
	#Storage Locations
	Storage_Path    		= PhD_Global.Storage_Path_WC3
	Processed_Data_Path		= 'Processed_Data/Radiosonde/'
	Raw_Data_Path			= 'Raw_Data/'
	Plots_Path      		= 'Plots/Radiosonde/'
    
	#Plot Labels
	if Calibrate == "counts":
		Pandora_Labels = ['Current (Counts)',
			'Cloud Sensor\n(Counts)',
			'PLL (counts)']
	elif Calibrate == "volts":
		Pandora_Labels = ['Current (V)',
			'Cloud Sensor\n(V)',
			'PLL (Counts)']
	elif Calibrate == "units":
		Pandora_Labels = ['Charge Density\n$(pC$ $m^{-3})$',
			'Cloud Sensor\n(V)',
			'Vibrating\nWire$(Hz)$']
		
    ############################################################################
	"""[Step 1] Check and Import Data"""
    
	#Error check that either Radiosonde_File or Sensor_Package has been specified
	if Radiosonde_File is None and Sensor_Package is None: sys.exit("[Error] You must specify either the Radiosonde_File location or the Sensor_Package number")
	
	#Attempt to find the radiosonde file either directly or from glob
	Radiosonde_File = Storage_Path + Processed_Data_Path + Radiosonde_File if Radiosonde_File is not None else glob.glob(Storage_Path + Processed_Data_Path + 'Radiosonde_Flight_No.' + str(Sensor_Package).rjust(2,'0') + '_*/Radiosonde_Flight_PhD_James_No.' + str(Sensor_Package) + '*a.txt')
	
	#If no radiosonde file was found we end program
	if len(Radiosonde_File) == 0: sys.exit("[Error] Radiosonde package No.%s does not exist. Has the radiosonde been launched yet or has the data been misplaced?" % (Sensor_Package))
	
	#If the radiosonde file was found via glob we need to convert to str from list
	if isinstance(Radiosonde_File, list): Radiosonde_File = Radiosonde_File[0]
	
	#Once the radiosonde file is found we can attempt to find the GPS file in the raw file section
	GPS_File = glob.glob(Storage_Path + Raw_Data_Path + 'Radiosonde_Flight_No.' + str(Sensor_Package).rjust(2,'0') + '_*/GPSDCC_RESULT*.tsv')
	
	#Import all the data
	if Radiosonde_File is not None: Radiosonde_Data = np.genfromtxt(Radiosonde_File, delimiter=None, skip_header=10, dtype=float, comments="#")
	if len(GPS_File) != 0: GPS_Data = np.genfromtxt(GPS_File[0], delimiter=None, skip_header=51, dtype=float, comments="#")
	
	############################################################################
	"""[Step 2] Calibrate bespoke sensors"""
    
	#Calibrate Height, Temperature and Convert PANDORA channels from counts to volts if required.
	if Calibration_Log is None:
		Radiosonde_Cal = Radiosonde_Checks(Radiosonde_Data, Calibrate, Sensor_Package, Height_Range, check=1111)
	else:
		Radiosonde_Cal = Radiosonde_Checks(Radiosonde_Data, Calibrate, Sensor_Package, Height_Range, check=1111, linear=True, log=False)
	
	#Calibrate OMB Sensor
	if (Calibrate == "units") & (Sensor_Package < 8): Radiosonde_Cal.wire_calibration()
	
	#Calibrate Cloud Sensor
	if Calibrate == "units": 
		if Sensor_Package < 8:
			Radiosonde_Cal.cloud_calibration(Sensor_Package, check=1111)
		else:
			Radiosonde_Cal.cloud_calibration(Sensor_Package, check=None)
			
	#Calibrate Charge Sensor
	if Calibrate == "units": 
		if Calibration_Log is None:
			#Space charge density of linear sensor
			Radiosonde_Cal.charge_calibration(Sensor_Package, type='space_charge', linear=True)
		
		else:
			#Space charge density of log sensor
			
			#Get Log Charge Counts and the Time (used to correctly sort data after calibration
			Radiosonde_Time = Radiosonde_Cal.data[:,0].copy()
			Radiosonde_Height = Radiosonde_Cal.data[:,1].copy()
			LogCharge_Counts = Radiosonde_Cal.data[:,6].copy()
			
			print("LogCharge_Counts", LogCharge_Counts)
			
			print("LogCharge_Counts Stats", gu.stats(LogCharge_Counts))
			
			#Get the space charge density for the lab based calibration of log sensor
			Radiosonde_Cal.charge_calibration(Sensor_Package, type='space_charge', lab_calibration=True, linear=False, log=True)
			LogCharge_LabCal = Radiosonde_Cal.data[:,6].copy()
			
			#Calibrate just the linear charge sensor to determine boundaries in Calibration (e.g. We calibrate 
			PosMask = Radiosonde_Cal.data[:,5] > 0
			NegMask = Radiosonde_Cal.data[:,5] < 0
			
			print("PosMask", np.sum(PosMask))
			print("NegMask", np.sum(NegMask))	
			
			print("Slope", Calibration_Log[0][0], "Intercept", Calibration_Log[0][1])
			print("MAX", np.nanmax(LogCharge_Counts), Calibration_Log[0][0]*np.nanmax(LogCharge_Counts)+Calibration_Log[0][1])
			LogCharge_LinCal = Calibration_Log[0][0] * LogCharge_Counts + Calibration_Log[0][1]
			
			print("LogCharge_LinCal Stats", gu.stats(LogCharge_LinCal))
			
			#Calculate Space Charge Density
			LogCharge_LinCal = Radiosonde_Cal._space_charge(Radiosonde_Time, Radiosonde_Height, LogCharge_LinCal, lab_calibration=True)
			
			#Subset LogCharge_LabCal and LogCharge_LinCal for correct height range
			LogCharge_LabCal = LogCharge_LabCal[(Radiosonde_Height >= Height_Range[0]) & (Radiosonde_Height <= Height_Range[1])]
			LogCharge_LinCal = LogCharge_LinCal[(Radiosonde_Height >= Height_Range[0]) & (Radiosonde_Height <= Height_Range[1])]
			
	
			# print("LogCharge_LinCal", LogCharge_LinCal[:50])
			# sys.exit()
			
			# #Split Radiosonde_Time and LogCharge_Counts into positive and negative regions (found from Radiosonde_ChargeCalibrator)
			# Radiosonde_Time_Pos = Radiosonde_Time[PosMask]
			# Radiosonde_Time_Neg = Radiosonde_Time[NegMask]
			# LogCharge_Counts_Pos = LogCharge_Counts[PosMask]
			# LogCharge_Counts_Neg = LogCharge_Counts[NegMask]
			
			# #Calculate calibration from linear regression			
			# LogCharge_LinCal_Pos = LogCurrent[0][0] * LogCharge_Counts_Pos + LogCurrent[0][1]
			# LogCharge_LinCal_Neg = LogCurrent[1][0] * LogCharge_Counts_Neg + LogCurrent[1][1]

			# #Combine positive and negative calibrations back together
			# Radiosonde_Time = np.hstack((Radiosonde_Time_Pos, Radiosonde_Time_Neg))
			# LogCharge_LinCal = np.hstack((10**LogCharge_LinCal_Pos, -10**-LogCharge_LinCal_Neg))
			
			# #Sort the arrays by time
			# mask = np.argsort(Radiosonde_Time, kind='mergesort')
			# Radiosonde_Time = Radiosonde_Time[mask]
			# LogCharge_LinCal = LogCharge_LinCal[mask] #Units are Amps
				
			# print("LogCharge_LinCal_Neg", LogCharge_LinCal_Neg[:50])
			# print("LogCharge_LinCal_Neg", (-10**-LogCharge_LinCal_Neg)[:50])
			# sys.exit()
				
			# #Calculate Space Charge Density
			# LogCharge_LinCal = Radiosonde_Cal._space_charge(Radiosonde_Time, Radiosonde_Height, LogCharge_LinCal, lab_calibration=True)
			
			# #Subset LogCharge_LabCal and LogCharge_LinCal for correct height range
			# LogCharge_LabCal = LogCharge_LabCal[(Radiosonde_Height >= Height_Range[0]) & (Radiosonde_Height <= Height_Range[1])]
			# LogCharge_LinCal = LogCharge_LinCal[(Radiosonde_Height >= Height_Range[0]) & (Radiosonde_Height <= Height_Range[1])]
			
			# print("LogCharge_LinCal", LogCharge_LinCal.tolist())
			# print("LogCharge_LabCal", LogCharge_LabCal.tolist())
			# sys.exit()
			
	#Calibrate Relative Humidity Sensor (e.g. find RH_ice)
	Radiosonde_Cal.RH()
	
	#Return Data
	Radiosonde_Data = Radiosonde_Cal.return_data()

	GPS_Data = GPS_Data[GPS_Data[:,4] > 0]
	#GPS_Data = GPS_Data[7718:] #for No.4
	if GPS_File is not None: Launch_Datetime = GPS2UTC(GPS_Data[0,1], GPS_Data[0,2])

	############################################################################
	"""[Step 3] Plot radiosonde data"""

	Title = 'Radiosonde Flight No.' + str(Sensor_Package) + ' (' + Launch_Datetime.strftime("%d/%m/%Y %H%MUTC") + ')' if GPS_File is not None else 'Radiosonde Flight (N/A)'
	Superplotter = SPRadiosonde(8, Title, Height_Range, Radiosonde_Data) if Calibrate == "units" else SPRadiosonde(7, Title, Height_Range, Radiosonde_Data)
	
	#Plot the PANDORA data
	#Superplotter.Charge(Linear_Channel=None, Log_Channel=1, XLabel=Pandora_Labels[0])
	
	if Calibration_Log is None:
		Superplotter.Charge(Linear_Channel=0, Log_Channel=1, XLabel=Pandora_Labels[0])
	else:
		Superplotter.Charge(Linear_Channel=None, Log_Channel=(LogCharge_LabCal, LogCharge_LinCal), XLabel=Pandora_Labels[0], Calibration_Log=True)
	
	if Sensor_Package < 8:
		Superplotter.Cloud(Cyan_Channel=2, IR_Channel=3, XLabel=Pandora_Labels[1], Cyan_Check=1111)
		Superplotter.PLL(PLL_Channel=2, XLabel=Pandora_Labels[2], PLL_Check=1112, Point=False, Calibrate=Calibrate) if Sensor_Package < 3 else Superplotter.PLL(PLL_Channel=2, XLabel=Pandora_Labels[2], PLL_Check=1112, Point=True, Calibrate=Calibrate)
	else:
		Superplotter.Cloud(Cyan_Channel=2, IR_Channel=3, XLabel=Pandora_Labels[1])
		Superplotter.Turbulence(Turbulence_Channel=4)
		
	#Plot the processed PLL data
	#if Calibrate == "units": Superplotter.ch(13, 'dPLLdt $(Hz$ $s^{-1})$', 'dPLL/dt', check=1112, point=True)
	if (Calibrate == "units") & (Sensor_Package < 8): Superplotter.ch(14, 'SLWC $(g$ $m^{-3})$', 'Supercooled Liquid\nWater Concentration', check=1112, point=True)
	
	#Plot the cloud boundaries if specified
	if Clouds_ID is not None: Superplotter.Cloud_Boundaries(Clouds_ID, LayerType, CloudOnly=True)
	
	############################################################################
	"""[Step 4] Save plot and return"""
	
	#Specify the directory the plots are stored in 
	path = os.path.dirname(Radiosonde_File).replace(Storage_Path + Processed_Data_Path,"")
	
	#Find any other plots stored in this directory
	previous_plots = glob.glob(Storage_Path + Plots_Path + path + "/*")
	
	#Find the biggest 'v' number in plots
	plot_version = []
	for plots in previous_plots:
		try:
			plot_version.append(int(os.path.basename(plots)[34:37]))
		except ValueError:
			plot_version.append(int(os.path.basename(plots)[34:36]))
	
	plot_version = str(np.max(plot_version)+1) if len(plot_version) != 0 else '1'
	
	#Create full directory and file name
	Save_Location = Storage_Path + Plots_Path + path + '/' + path + '_v' + plot_version.rjust(2,'0') + '_' + str(Height_Range[0]).rjust(2,'0') + 'km_to_' + str(Height_Range[1]).rjust(2,'0') + 'km.png'
	
	#Ensure the directory exists on file system and save to that location
	gu.ensure_dir(os.path.dirname(Save_Location))
	Superplotter.savefig(Save_Location)
	
	print("[INFO] Radiosonde_Superplotter completed successfully (In %.2fs)" % (time.time()-t_begin))

def Radiosonde_Tephigram(Radiosonde_File=None, Height_Range=None, Sensor_Package=None, plot_tephigram=False, plot_camborne=False):
	"""The Radiosonde_Tephigram function will plot a tephigram from the dry bulb temperature,
	T_dry and the Dew point Temperature, T_dew for pressure values, P at each corresponding 
	height. 
	
	Certain tephigram outputs are available from this function including:
	1) Lower Condensation Level (LCL) in m
	2) Level of Free Convection (LFC) in m
	3) Environmental Level (EL) in m
	4) Convective Available Potential Energy (CAPE) in J/kg
	5) Convective INhibition (CIN) in J/kg

	Parameters
	----------
	
	Radiosonde_File : str, optional, default is None
        Location of the radiosonde file to be processed and plotted
	Height_Range : tuple, optional, default is None
		lower and upper limits of the height you want to plot
	Sensor_Package : int, optional, default is None
		specify the package number related to the data. This is then used
		to calibrate the charge sensor
	plot_tephigram : bool, optional, default is False
		Specify True to plot a tephigram of the sounding data. Otherwise
		just calculate the sounding indices
	plot_camborne : bool, optional, default is False
		Specify True to add the sounding from Camborne at the closest time
		to the launch time. Only used if plot_tephigram is True.
	
	Outputs
	-------
	
	References
	----------
	Ambaum, M. H. P., 2010. Water in the Atmosphere. In: Thermal Physics of the Atmosphere. Oxford: Wiley & Sons, pp. 93-109
	Marlton, G. 2018. Tephigram. Original Matlab code found in Matlab_Code directory
	Hunt, K. 2018. Tephigram. Original Python code found in the same directory.
	"""
	
	gu.cprint("[INFO] You are running Radiosonde_Tephigram from the STABLE release", type='bold')
		
	############################################################################
	"""Prerequisites"""
	
	#Time Controls
	t_begin = time.time()
	
	#Storage Locations
	Storage_Path 			= PhD_Global.Storage_Path_WC3
	Processed_Data_Path		= 'Processed_Data/Radiosonde/'
	Raw_Data_Path			= 'Raw_Data/'
	Plots_Path 				= 'Plots/Tephigram/'
	
	#Set-up data importer
	EPCC_Data = EPCC_Importer()
	
	############################################################################
	"""[Step 1] Check and Import Data"""
    
	#Error check that either Radiosonde_File or Sensor_Package has been specified
	if Radiosonde_File is None and Sensor_Package is None: sys.exit("[Error] You must specify either the Radiosonde_File location or the Sensor_Package number")
	
	#Attempt to find the radiosonde file either directly or from glob
	Radiosonde_File = Storage_Path + Processed_Data_Path + Radiosonde_File if Radiosonde_File is not None else glob.glob(Storage_Path + Processed_Data_Path + 'Radiosonde_Flight_No.' + str(Sensor_Package).rjust(2,'0') + '_*/Radiosonde_Flight_PhD_James_No.' + str(Sensor_Package) + '*a.txt')
	
	#If no radiosonde file was found we end program
	if len(Radiosonde_File) == 0: sys.exit("[Error] Radiosonde package No.%s does not exist. Has the radiosonde been launched yet or has the data been misplaced?" % (Sensor_Package))
	
	#If the radiosonde file was found via glob we need to convert to str from list
	if isinstance(Radiosonde_File, list): Radiosonde_File = Radiosonde_File[0]
	
	print("Importing Radiosonde File @ ", Radiosonde_File)
	
	#Once the radiosonde file is found we can attempt to find the GPS file in the raw file section
	GPS_File = glob.glob(Storage_Path + Raw_Data_Path + 'Radiosonde_Flight_No.' + str(Sensor_Package).rjust(2,'0') + '_*/GPSDCC_RESULT*.tsv')
	
	#Import all the data
	if Radiosonde_File is not None: Radiosonde_Data = np.genfromtxt(Radiosonde_File, delimiter=None, skip_header=9, dtype=float, comments="#")
	if len(GPS_File) != 0: GPS_Data = np.genfromtxt(GPS_File[0], delimiter=None, skip_header=51, dtype=float, comments="#")
		
	#Calibrate Height, Temperature and Convert PANDORA channels from counts to volts if required.
	Radiosonde_Cal = Radiosonde_Checks(Radiosonde_Data, calibrate=None, package_no=Sensor_Package, height_range=[0,12], check=1111)
	
	#Return Data
	Radiosonde_Data = Radiosonde_Cal.finalise()

	#Calculate launch datetime
	GPS_Data = GPS_Data[GPS_Data[:,4] > 0]
	if GPS_File is not None: Launch_Datetime = GPS2UTC(GPS_Data[0,1], GPS_Data[0,2])
	
	#Unpack variables
	Z = Radiosonde_Data[:,1]
	Tdry = Radiosonde_Data[:,3]
	Tdew = Radiosonde_Data[:,14]
	Pres = Radiosonde_Data[:,2]
	RH = Radiosonde_Data[:,4]/100; RH -= np.max(RH) - 0.01
	Wind_Mag = (Radiosonde_Data[:,15]**2 + Radiosonde_Data[:,16]**2)**0.5
	Wind_Dir = np.arctan2(Radiosonde_Data[:,15], Radiosonde_Data[:,16]) * 180 / np.pi
		
	############################################################################
	"""[Step 2] Create Tephigram"""
	
	if plot_tephigram is True:
	
		print("[INFO] Plotting Tephigram...")
	
		#Mask nan data (ONLY FOR PLOTTING)
		Radiosonde_Data_Plotting = gu.antinan(Radiosonde_Data.T)
		
		#Unpack variables
		Z_Plot = Radiosonde_Data[:,1]
		Tdry_Plot = Radiosonde_Data[:,3]
		Tdew_Plot = Radiosonde_Data[:,14]
		Pres_Plot = Radiosonde_Data[:,2]
		
		#Subset the tephigram to specified location
		locator = gu.argneararray(Z_Plot, np.array(Height_Range)*1000)
		anchor = np.array([(Pres_Plot[locator]),(Tdry_Plot[locator])]).T
		
		Pres_Plot_Antinan, Tdry_Plot_Antinan, Tdew_Plot_Antinan = gu.antinan(np.array([Pres_Plot, Tdry_Plot, Tdew_Plot]), unpack=True)
		
		#Group the dews, temps and wind profile measurements
		dews = zip(Pres_Plot_Antinan, Tdew_Plot_Antinan)
		temps = zip(Pres_Plot_Antinan, Tdry_Plot_Antinan)
		barb_vals = zip(Pres,Wind_Dir,Pres_Plot)
		        
		#Create Tephigram plot
		Tephigram = SPTephigram()
		if plot_camborne is True:
			
			#Determine ULS data
			ULS_File = sorted(glob.glob(PhD_Global.Raw_Data_Path + 'Met_Data/ULS/*'))
			
			ULS_Date = np.zeros(len(ULS_File), dtype=object)
			for i, file in enumerate(ULS_File):
				try:
					ULS_Date[i] = datetime.strptime(os.path.basename(file), '%Y%m%d_%H_UoW_ULS.csv')
				except:
					ULS_Date[i] = datetime(1900,1,1)
			
			#Find Nearest Upper Level Sounding Flight to Radiosonde Flight
			ID = gu.argnear(ULS_Date, Launch_Datetime)
			
			print("[INFO] Radiosonde Launch Time:", Launch_Datetime, "Camborne Launch Time:", ULS_Date[ID])
			
			#Import Camborne Radiosonde Data
			press_camborne, temps_camborne, dews_camborne = EPCC_Data.ULS_Calibrate(ULS_File[ID], unpack=True, PRES=True, TEMP=True, DWPT=True)
			
			#Match Camborne pressures with Reading pressures
			mask = [gu.argnear(press_camborne, Pres_Plot[0]), gu.argnear(press_camborne, Pres_Plot[-1])]
			press_camborne = press_camborne[mask[0]:mask[1]]
			temps_camborne = temps_camborne[mask[0]:mask[1]]
			dews_camborne = dews_camborne[mask[0]:mask[1]]
				
			dews_camborne = zip(press_camborne, dews_camborne)
			temps_camborne = zip(press_camborne, temps_camborne)
			
			#Plot Reading Radiosonde
			profile_t1 = Tephigram.plot(temps, color="red", linewidth=1, label='Reading Dry Bulb Temperature', zorder=5)
			profile_d1 = Tephigram.plot(dews, color="blue", linewidth=1, label='Reading Dew Bulb Temperature', zorder=5)
			
			#Plot Camborne Radiosonde
			profile_t1 = Tephigram.plot(temps_camborne, color="red", linestyle=':', linewidth=1, label='Camborne Dry Bulb Temperature', zorder=5)
			profile_d1 = Tephigram.plot(dews_camborne, color="blue", linestyle=':', linewidth=1, label='Camborne Dew Bulb Temperature', zorder=5)
		else:	
			profile_t1 = Tephigram.plot(temps,color="red",linewidth=2, **{'label':'Dry Bulb Temperature'})
			profile_d1 = Tephigram.plot(dews,color="blue",linewidth=2, **{'label':'Dew Bulb Temperature'})
		
		#Add extra information to Tephigram plot
		#Tephigram.axes.set(title=Title, xlabel="Potential Temperature $(^\circ C)$", ylabel="Dry Bulb Temperature $(^\circ C)$")
		Title = 'Radiosonde Tephigram Flight No.' + str(Sensor_Package) + ' (' + Launch_Datetime.strftime("%d/%m/%Y %H%MUTC") + ')' if GPS_File is not None else 'Radiosonde Tephigram Flight (N/A)'
		Tephigram.axes.set(title=Title)
				
		#[OPTIONAL] Add wind profile information to Tephigram.
		#profile_t1.barbs(barb_vals)
		
		############################################################################
		"""Save plot to file"""

		#Specify the directory the plots are stored in 
		path = os.path.dirname(Radiosonde_File).replace(Storage_Path + Processed_Data_Path,"")
		
		#Find any other plots stored in this directory
		previous_plots = glob.glob(Storage_Path + Plots_Path + path + "/*")
		
		#Find the biggest 'v' number in plots
		plot_version = []
		for plots in previous_plots:
			try:
				plot_version.append(int(os.path.basename(plots)[34:37]))
			except ValueError:
				plot_version.append(int(os.path.basename(plots)[34:36]))
		
		plot_version = str(np.max(plot_version)+1) if len(plot_version) != 0 else '1'
		
		#Create full directory and file name
		Save_Location = Storage_Path + Plots_Path + path + '/' + path + '_v' + plot_version.rjust(2,'0') + '_' + str(Height_Range[0]).rjust(2,'0') + 'km_to_' + str(Height_Range[1]).rjust(2,'0') + 'km.png'
		
		#Ensure the directory exists on file system and save to that location
		gu.ensure_dir(os.path.dirname(Save_Location))
		
		Tephigram.savefig(Save_Location)      

	############################################################################
	"""[Step 3] Calculate Stability Indices"""
	
	print("[INFO] Calculating Stability Indices...")
	
	#Common Pressure Levels
	P_500 = gu.argnear(Pres, 500)
	P_700 = gu.argnear(Pres, 700)
	P_850 = gu.argnear(Pres, 850)
	
	#Showalter stability index
	#S = Tdry[P_500] - Tl
	
	#K-Index
	K = (Tdry[P_850] - Tdry[P_500]) + Tdew[P_850] - (Tdry[P_700] - Tdew[P_700])
	
	#Cross Totals Index
	CT = Tdew[P_850] - Tdry[P_500]
	
	#Vertical Totals Index
	VT = Tdry[P_850] - Tdry[P_500]
	
	#Total Totals Index
	TT = VT + CT
	
	#SWEAT Index
	ms2kn = 1.94384	#Conversion between m/s to knots
		
	SW_1 = 20*(TT-49)
	SW_2 = 12*Tdew[P_850]
	SW_3 = 2*Wind_Mag[P_850]*ms2kn
	SW_4 = Wind_Mag[P_500]*ms2kn
	SW_5 = 125*(np.sin(Wind_Dir[P_500]-Wind_Dir[P_850]) + 0.2)
	
	#Condition SWEAT Term 1 from several conditions
	SW_1 = 0 if SW_1 < 49 else SW_1

	#Condition SWEAT Term 5 with several conditions
	if (Wind_Dir[P_850] > 130) & (Wind_Dir[P_850] < 250):
		if (Wind_Dir[P_500] > 210) & (Wind_Dir[P_500] < 310):
			if Wind_Dir[P_500]-Wind_Dir[P_850] > 0:
				if (Wind_Mag[P_500]*ms2kn > 15) & (Wind_Mag[P_850]*ms2kn > 15):
					SW_5 = SW_5
				else:
					SW_5 = 0
			else:
				SW_5 = 0
		else:
			SW_5 = 0
	else:
		SW_5 = 0
	
	#Calulate Final Product
	SW = SW_1 + SW_2 + SW_3 + SW_4 + SW_5
	
	print("Stability Indices")
	print("-----------------")
	print("K-Index:", K)
	print("Cross Totals Index:", CT)
	print("Vettical Totals Index:", VT)
	print("Total Totals Index:", TT)
	print("SWEAT Index:", SW)
	print("\n")
	
	############################################################################
	"""[Step 4] Calculate Tephigram Indices"""
	
	#Convert Temperature back to Kelvin
	Tdry += 273.15
	Tdew += 273.15
	
	#Convert Height into metres
	Z *= 1000
	
	#Constants
	over27 = 0.286 # Value used for calculating potential temperature 2/7
	L = 2.5e6  #Latent evaporation 2.5x10^6
	epsilon = 0.622
	E = 6.014  #e in hpa 
	Rd = 287 #R constant for dry air
	
	#Equations
	es = lambda T: 6.112*np.exp((17.67*(T-273.15))/(T-29.65)) #Teten's Formula for Saturated Vapour Pressure converted for the units of T in Kelvin rather than Centigrade
	
	#Calculate Theta and Theta Dew
	theta = Tdry*(1000/Pres)**over27
	thetadew = Tdew*(1000/Pres)**over27
		
	#Find the Lifting Condensation Level (LCL)
	qs_base = 0.622*es(Tdew[0])/Pres[0]
	
	theta_base = theta[0]
	Pqs_base = 0.622*es(Tdry)/qs_base  #Calculate a pressure for constant qs
	Pqs_base = Tdry*(1000/Pqs_base)**(2/7) #Calculates pressure in term of P temp
	
	#print("Tdew[0]", Tdew[0])
	#print("Pres[0]", Pres[0])
	#print("qs_base",qs_base)
	#print("theta_base", theta_base)
	#print("Pqs_base", Pqs_base)
	
	#Find first location where Pqs_base > theta_base
	y1 = np.arange(Pqs_base.size)[Pqs_base > theta_base][0]
	#print(Pqs_base[y1])
	#print(y1)
	#print(gu.argnear(Pqs_base, theta_base))
	
	LCL = Z[y1]
		
	Tarr = np.zeros(Tdry.size)
	thetaarr = np.zeros(Tdry.size)

	T_temp = Tdry[y1]
	P_temp = 1000*(T_temp/Pqs_base[y1])**3.5
	qs0 = 0.622*es(T_temp)/P_temp
	
	thetaarr[y1] = Pqs_base[y1]
	Tarr[y1] = T_temp
	
	for i in xrange(y1+1, y1+100):
		T_temp -= 1
		P_temp = 1000*(T_temp/thetaarr[i-1])**3.5
		qs = 0.622*es(T_temp)/P_temp
		thetaarr[i] = thetaarr[i-1] - ((2.5E6/1004)*(thetaarr[i-1]/T_temp) * (qs-qs0))
		qs0 = qs
		Tarr[i] = T_temp
		
	#Now need to integrate back to 1000hpa
	T_temp = Tdry[y1]
	P_temp = 1000*(T_temp/Pqs_base[y1])**3.5
	qs0 = 0.622*es(T_temp)/P_temp
	thetaarr[y1] = Pqs_base[y1]
	Tarr[y1] = T_temp
	for i in xrange(y1-1):
		T_temp += 1
		P_temp = 1000*(T_temp/thetaarr[(y1-i)+1])**3.5
		qs = 0.622*es(T_temp)/P_temp
		thetaarr[y1-i] = thetaarr[(y1-i)+1] - ((2.5E6/1004)*(thetaarr[(y1-i)+1]/T_temp) * (qs-qs0))
		qs0 = qs
		Tarr[y1-i] = T_temp
	
	y8 = (thetaarr>253) & (thetaarr<380)
	thetaarr = thetaarr[y8]
	Tarr = Tarr[y8]
	
	#Now find environmental levels and LFC begin by converting thetaarr into P
	Pthetaeq = 1000/(thetaarr/Tarr)**3.5
	l5 = np.isnan(Pthetaeq)
	Pthetaeq[l5] = []
	
	#Now interpolate on to rs height co-ordinates	
	TEMP = sp.interpolate.interp1d(Pthetaeq,[thetaarr,Tarr], fill_value="extrapolate")(Pres)
	thetaarr = TEMP[0]
	Tarr = TEMP[1]

	del(TEMP)
		
	y5 = np.arange(Tdry.size)[Tdry < Tarr]

	if np.any(y5):
		LFC = Z[y5[0]]
		EL = Z[y5[-1]]
		
		#Finds CIN area above LCL
		y6 = np.arange(Tdry.size)[(Z < LFC) & (Z >= LCL) & (Tdry > Tarr)]
		y7 = np.arange(Tdry.size)[(Z < LCL) & (Tdry > Tarr)]
		
		Pstart = Pres[y5[-1]]
		
		#Now need to calculate y5 temperatures into virtual temperatures
		Tvdash = Tarr/(1-(E/Pres)*(1-epsilon))
		Tv = Tdry/(1-(E/Pres)*(1-epsilon))
		T_adiabat = ((theta_base/(1000/Pres)**over27))
		Tv_adiabat = T_adiabat/(1-(E/Pres)*(1-epsilon))
		
		#Now need to calculate CAPE... and CIN to use CAPE = R_d = intergral(LFC,EL)(T'_v - T_v) d ln p
		CAPE = 0
		for i in xrange(y5[-2], y5[0], -1):
			CAPE += (Rd*(Tvdash[i] - Tv[i]) * np.log(Pres[i]/Pres[i+1]));
		
		#Now we use same technique to calculate CIN
		CIN=0;
		if len(y6) != 0:
			for i in xrange(y6[-2], y6[0], -1):
				CIN += (Rd*(Tvdash[i] - Tv[i]) * np.log(Pres[i]/Pres[i+1]))
	
		#Now calculate temperature along the dry adiabat
		y7 = np.arange(Tdry.size)[(Z < LCL) & (Tv > Tv_adiabat)]
		if len(y7) != 0:
			for i in xrange(y7[-2], y7[0], -1): 
				CIN += (Rd*(Tv_adiabat[i] - Tv[i]) * np.log(Pres[i]/Pres[i+1]));   
		
	else:
		LFC = np.nan
		EL = np.nan
		CAPE = 0
		CIN = 0
	
	#Print out information
	print("Parcel Information")
	print("------------------")
	print("LCL = %.2fm" % LCL)
	print("LFC = %.2fm" % LFC)
	print("EL = %.2fm" % EL)
	print("CAPE %.2f J/kg" % CAPE)
	print("CIN %.2f J/kg" % CIN)
	print("\n")
	
	print("[INFO] Radiosonde_Tephigram has been completed successfully (In %.2fs)" % (time.time()-t_begin))
	
	return LCL, LFC, EL, CAPE, CIN		

def Radiosonde_CloudIdentifier(Radiosonde_File=None, Height_Range=None, Sensor_Package=None):
	"""This function will identify the cloud layers within a radiosonde ascent by using the cloud sensor and 
	relative humidity measurements
	
	Reference
	---------
	Zhang, J., H. Chen, Z. Li, X. Fan, L. Peng, Y. Yu, and M. Cribb (2010). Analysis of cloud layer structure 
		in Shouxian, China using RS92 radiosonde aided by 95 GHz cloud radar. J. Geophys. Res., 115, D00K30, 
		doi: 10.1029/2010JD014030.
	WMO, 2017. Clouds. In: Internal Cloud Atlas Manual on the Observation of Clouds and Other Meteors. 
		Hong Kong: WMO, Section 2.2.1.2.
	"""
	
	gu.cprint("[INFO] You are running Radiosonde_CloudIdentifier from the STABLE release", type='bold')
		
	############################################################################
	"""Prerequisites"""
	
	#Time Controls
	t_begin = time.time()
	
	#Storage Locations
	Storage_Path 			= PhD_Global.Storage_Path_WC3
	Processed_Data_Path		= 'Processed_Data/Radiosonde/'
	Raw_Data_Path			= 'Raw_Data/'
	Plots_Path 				= 'Plots/Tephigram/'
	
	#Set-up data importer
	EPCC_Data = EPCC_Importer()
	
	############################################################################
	"""[Step 1] Check and Import Data"""
    
	#Error check that either Radiosonde_File or Sensor_Package has been specified
	if Radiosonde_File is None and Sensor_Package is None: sys.exit("[Error] You must specify either the Radiosonde_File location or the Sensor_Package number")
	
	#Attempt to find the radiosonde file either directly or from glob
	Radiosonde_File = Storage_Path + Processed_Data_Path + Radiosonde_File if Radiosonde_File is not None else glob.glob(Storage_Path + Processed_Data_Path + 'Radiosonde_Flight_No.' + str(Sensor_Package).rjust(2,'0') + '_*/Radiosonde_Flight_PhD_James_No.' + str(Sensor_Package) + '*a.txt')
	
	#If no radiosonde file was found we end program
	if len(Radiosonde_File) == 0: sys.exit("[Error] Radiosonde package No.%s does not exist. Has the radiosonde been launched yet or has the data been misplaced?" % (Sensor_Package))
	
	#If the radiosonde file was found via glob we need to convert to str from list
	if isinstance(Radiosonde_File, list): Radiosonde_File = Radiosonde_File[0]
	
	#Once the radiosonde file is found we can attempt to find the GPS file in the raw file section
	GPS_File = glob.glob(Storage_Path + Raw_Data_Path + 'Radiosonde_Flight_No.' + str(Sensor_Package).rjust(2,'0') + '_*/GPSDCC_RESULT*.tsv')
	
	#Import all the data
	if Radiosonde_File is not None: Radiosonde_Data = np.genfromtxt(Radiosonde_File, delimiter=None, skip_header=10, dtype=float, comments="#")
	if len(GPS_File) != 0: GPS_Data = np.genfromtxt(GPS_File[0], delimiter=None, skip_header=51, dtype=float, comments="#")
	
	############################################################################
	"""[Step 2] Calibrate bespoke sensors"""
    
	#Calibrate Height, Temperature and Convert PANDORA channels from counts to volts if required.
	Radiosonde_Cal = Radiosonde_Checks(Radiosonde_Data, None, Sensor_Package, Height_Range, check=1111)
		
	#Calibrate Relative Humidity Sensor (e.g. find RH_ice)
	Radiosonde_Cal.RH()
	
	#Return Data
	Radiosonde_Data = Radiosonde_Cal.return_data()
			
	GPS_Data = GPS_Data[GPS_Data[:,4] > 0]
	if GPS_File is not None: Launch_Datetime = GPS2UTC(GPS_Data[0,1], GPS_Data[0,2])
	
	############################################################################
	"""[METHOD 1]: Relative Humidity (Zhang et al. 2010)"""
	
	#Define data into new variables
	Z = Radiosonde_Data[:,1]
	RH = Radiosonde_Data[:,-1]
	
	#Create Height-Resolving RH Thresholds (see Table 1 in Zhang et al. (2010))
	#N.B. use np.interp(val, RH_Thresholds['altitude'], RH_Thresholds['*RH']) where val is the height range you want the RH Threshold 
	RH_Thresholds = {'minRH' : [0.92, 0.90, 0.88, 0.75, 0.75],
		'maxRH' : [0.95, 0.93, 0.90, 0.80, 0.80],
		'interRH' : [0.84, 0.82, 0.78, 0.70, 0.70],
		'altitude' : [0, 2, 6, 12, 20]}
	
	#Define the cloud height levels as defined by WMO (2017). 
	Z_Levels = {'low' : [0,2], 'middle' : [2,7], 'high' : [5,13]}
	
	#Define the types of layers that can be detected.
	Cloud_Types = {0 : 'Clear Air', 1 : 'Moist (Not Cloud)', 2 : 'Cloud'}
	
	#Define the min, max and interRH for all measure altitudes
	minRH = np.interp(Z, RH_Thresholds['altitude'], RH_Thresholds['minRH'], left=np.nan, right=np.nan)*100
	maxRH = np.interp(Z, RH_Thresholds['altitude'], RH_Thresholds['maxRH'], left=np.nan, right=np.nan)*100
	interRH = np.interp(Z, RH_Thresholds['altitude'], RH_Thresholds['interRH'], left=np.nan, right=np.nan)*100
	
	#[Step 1]: The base of the lowest moist layer is determined as the level when RH exceeds the min-RH corresponding to this level
	minRH_mask = (RH > minRH)
	
	#[Step 2 and 3]: Above the base of the moist layer, contiguous levels with RH over the corresponding min-RH are treated as the same layer
	Z[~minRH_mask] = np.nan
	Clouds_ID = gu.contiguous(Z, 1)
	
	#[Step 4]: Moist layers with bases lower than 120m and thickness's less than 400m are discarded
	for Cloud in np.unique(Clouds_ID)[1:]:
		if Z[Clouds_ID == Cloud][0] < 0.12:
			if Z[Clouds_ID == Cloud][-1] - Z[Clouds_ID == Cloud][0] < 0.4:
				Clouds_ID[Clouds_ID == Cloud] = 0
	
	#[Step 5]: The moist layer is classified as a cloud layer is the maximum RH within this layer is greater than the corresponding max-RH at the base of this moist layer
	LayerType = np.zeros(Z.size, dtype=int) #0: Clear Air, 1: Moist Layer, 2: Cloud Layer
	for Cloud in np.unique(Clouds_ID)[1:]:
		if np.any(RH[Clouds_ID == Cloud] > maxRH[Clouds_ID == Cloud][0]):
			LayerType[Clouds_ID == Cloud] = 2
		else:
			LayerType[Clouds_ID == Cloud] = 1
	
	#[Step 6]: The base of the cloud layers is set to 280m AGL, and cloud layers are discarded if their tops are lower than 280m	
	for Cloud in np.unique(Clouds_ID)[1:]:
		if Z[Clouds_ID == Cloud][-1] < 0.280:
			Clouds_ID[Clouds_ID == Cloud] = 0
			LayerType[Clouds_ID == Cloud] = 0

	#[Step 7]: Two contiguous layers are considered as one-layer cloud if the distance between these two layers is less than 300m or the minimum RH within this distance is more than the maximum inter-RG value within this distance
	for Cloud_Below, Cloud_Above in zip(np.unique(Clouds_ID)[1:-1], np.unique(Clouds_ID)[2:]):
		
		#Define the index between clouds of interest
		Air_Between = np.arange(gu.bool2int(Clouds_ID == Cloud_Below)[-1], gu.bool2int(Clouds_ID == Cloud_Above)[0])
		
		if ((Z[Clouds_ID == Cloud_Above][0] - Z[Clouds_ID == Cloud_Below][-1]) < 0.3) or (np.nanmin(RH[Air_Between]) > np.nanmax(interRH[Air_Between])):
			Joined_Cloud_Mask = np.arange(gu.bool2int(Clouds_ID == Cloud_Below)[0], gu.bool2int(Clouds_ID == Cloud_Above)[-1])
			
			#Update the cloud ID array as the Cloud_Below and Cloud_Above are not distinct clouds
			Clouds_ID[Joined_Cloud_Mask] = Cloud_Below
			
			#Update the LayerType to reflect the new cloud merging
			if np.any(LayerType[Clouds_ID == Cloud_Below] == 2) or np.any(LayerType[Clouds_ID == Cloud_Above] == 2):
				LayerType[Joined_Cloud_Mask] = 2
			else:
				LayerType[Joined_Cloud_Mask] = 1
		
	#[Step 8] Clouds are discarded if their thickness's are less than 30.5m for low clouds and 61m for middle/high clouds
	for Cloud in np.unique(Clouds_ID)[1:]:
		if Z[Clouds_ID == Cloud][0] < Z_Levels['low'][1]:
			if Z[Clouds_ID == Cloud][-1] - Z[Clouds_ID == Cloud][0] < 0.0305:
				Clouds_ID[Clouds_ID == Cloud] = 0
				LayerType[Clouds_ID == Cloud] = 0

		else:
			if Z[Clouds_ID == Cloud][-1] - Z[Clouds_ID == Cloud][0] < 0.0610:
				Clouds_ID[Clouds_ID == Cloud] = 0
				LayerType[Clouds_ID == Cloud] = 0
	
	#Re-update numbering of each cloud identified
	Clouds_ID = gu.contiguous(Clouds_ID, invalid=0)
	
	print("Detected Clouds and Moist Layers\n--------------------------------")
	for Cloud in np.unique(Clouds_ID)[1:]:
		print("Cloud %s. Cloud Base = %.2fkm, Cloud Top = %.2fkm, Layer Type: %s" % (Cloud, Z[Clouds_ID == Cloud][0], Z[Clouds_ID == Cloud][-1], Cloud_Types[LayerType[Clouds_ID == Cloud][0]]))
	
	return Clouds_ID, LayerType
	
	############################################################################
	"""[METHOD 2] Cloud Sensor (Own Method. Probably defunct)"""
	
	Z = Radiosonde_Data[:,1]
	Tdry = Radiosonde_Data[:,3]
	Cyan = Radiosonde_Data[:,7][Radiosonde_Data[:,9] == 1111]
	IR = Radiosonde_Data[:,8]
	
	#Correct IR sensor for temperature drift
	IR_Drift = 0.239601750967*Tdry
	IR += IR_Drift
	
	Z,Tdry,IR = gu.antinan((Z,Tdry,IR), unpack=True)
	
	IR_Mean = np.nanmean(np.sort(IR.copy(), kind='mergesort')[:IR.size//8])
	IR_Median = np.nanmedian(np.sort(IR.copy(), kind='mergesort')[:IR.size//8])
	
	IR_Mask = np.full(IR.size, 1100, dtype=np.float64)
	IR_Mask[IR < (IR_Median + 18)] = np.nan
	
	IR_Mask_Mask = np.zeros(IR_Mask.size, dtype=int)
	Temp_Mask = 0
	Ind_Mask = ()
	for j in xrange(IR_Mask.size):
		if np.isfinite(IR_Mask[j]):
			Temp_Mask += 1
			Ind_Mask += (j,)
			IR_Mask_Mask[[Ind_Mask]] = Temp_Mask
		else:
			Temp_Mask = 0
			Ind_Mask = ()
	
	#Remove cloud base lengths less than 10 (~8 minutes)
	IR_Mask_Mask[IR_Mask_Mask < 2] = 0
			
	#Convert integers from cloud length to unique cloud number
	Cloud_TimeBounds = gu.bool2int(np.diff(IR_Mask_Mask) != 0) + 1
	
	#Check to see if cloud exists along day boundary. If so we add in manual time boundaries
	if IR_Mask_Mask[0] != 0: Cloud_TimeBounds = np.insert(Cloud_TimeBounds, 0, 0)
	if IR_Mask_Mask[-1] != 0: Cloud_TimeBounds = np.append(Cloud_TimeBounds, IR_Mask_Mask.size-1)
	Cloud_TimeBounds = Cloud_TimeBounds.reshape(int(Cloud_TimeBounds.size/2),2)
	
	#Give each cloud a unique integer for each time step 
	for cloud_id, cloud_bounds in enumerate(Cloud_TimeBounds,1):
		IR_Mask_Mask[cloud_bounds[0]:cloud_bounds[1]] = cloud_id
	
	print("IR_Mask_Mask", IR_Mask_Mask.tolist())
	print("No of Detected Clouds", np.unique(IR_Mask_Mask)[-1])
	
	plt.clf()
	plt.hist(IR-IR_Median, bins=50)
	plt.yscale('log')
	#plt.show()
	
	#print("IR", IR)
	print("IR_Mean", IR_Mean)
	print("IR_Median", IR_Median)
	print("IR_Mask", IR_Mask)
	plt.clf()
	plt.plot(IR_Mask, Z, 'o', label='Cloud', linewidth=7.0)
	plt.plot(IR, Z, label='No filter')
	

	plt.legend(loc='upper right', prop={'size': 10}, fancybox=True, framealpha=0.5)
	
	plt.show()
	
	
	sys.exit()
	
def Radiosonde_Ice_Concentration(Radiosonde_File=None, Calibrate=None, Height_Range=None, Sensor_Package=None):
	
	gu.cprint("[INFO] You are running Radiosonde_Ice_Concentration from the DEV release", type='bold')
    
    ############################################################################
	"""Prerequisites"""
   
	Storage_Path    		= '/storage/shared/glusterfs/phd/users/th863480/WC3_InSitu_Electrification/'
	Processed_Data_Path		= 'Processed_Data/Radiosonde/'
	Raw_Data_Path			= 'Raw_Data/'
	Plots_Path      		= 'Plots/Radiosonde/'
    
	t_begin = time.time()
	
    ############################################################################
	"""[Step 1] Check and Import Data"""
    
	#Error check that either Radiosonde_File or Sensor_Package has been specified
	if Radiosonde_File is None and Sensor_Package is None: sys.exit("[Error] You must specify either the Radiosonde_File location or the Sensor_Package number")
	
	#Attempt to find the radiosonde file either directly or from glob
	Radiosonde_File = Storage_Path + Processed_Data_Path + Radiosonde_File if Radiosonde_File is not None else glob.glob(Storage_Path + Processed_Data_Path + 'Radiosonde_Flight_No.' + str(Sensor_Package).rjust(2,'0') + '_*/Radiosonde_Flight_PhD_James_No.' + str(Sensor_Package) + '*a.txt')
	
	#If no radiosonde file was found we end program
	if len(Radiosonde_File) == 0: sys.exit("[Error] Radiosonde package No.%s does not exist. Has the radiosonde been launched yet or has the data been misplaced?" % (Sensor_Package))
	
	#If the radiosonde file was found via glob we need to convert to str from list
	if isinstance(Radiosonde_File, list): Radiosonde_File = Radiosonde_File[0]
	
	#Import all the data
	if Radiosonde_File is not None: Radiosonde_Data = np.genfromtxt(Radiosonde_File, delimiter=None, skip_header=10, dtype=float, comments="#")
	
	Radiosonde_Cal = Radiosonde_Checks(Radiosonde_Data, Calibrate, Sensor_Package, Height_Range, check=1111)
	Radiosonde_Data = Radiosonde_Cal.return_data()
	
	Time = Radiosonde_Data[:,0][Radiosonde_Data[:,9] == 1112]
	PLL = Radiosonde_Data[:,7][Radiosonde_Data[:,9] == 1112]
	
	PLL[PLL == 0] = np.nan
	print(PLL)
	
	print(np.sum(np.isnan(PLL)))	
	
def Radiosonde_ChargeCalibrator(Radiosonde_File=None, Calibrate=None, Sensor_Package=None, Clouds_ID=None, LayerType=None):

	import matplotlib.pyplot as plt

	gu.cprint("[INFO] You are running Radiosonde_ChargeCalibrator from the DEV release", type='bold')
    
    ############################################################################
	"""Prerequisites"""
    
	#Time Controls
	t_begin = time.time()
		
	#Storage Locations
	Storage_Path    		= PhD_Global.Storage_Path_WC3
	Processed_Data_Path		= 'Processed_Data/Radiosonde/'
	Raw_Data_Path			= 'Raw_Data/'
	Plots_Path      		= 'Plots/ChargeCalibrator/'
	
	#Plotting requirements
	plt.style.use('classic') #necessary if Matplotlib version is >= 2.0.0
	
	#Plot Labels
	if Calibrate == "counts":
		Pandora_Labels = ['Current (Counts)',
			'Cloud Sensor\n(Counts)',
			'PLL (counts)']
	elif Calibrate == "volts":
		Pandora_Labels = ['Current (V)',
			'Cloud Sensor\n(V)',
			'PLL (Counts)']
	elif Calibrate == "units":
		Pandora_Labels = ['Charge Density\n$(pC$ $m^{-3})$',
			'Cloud Sensor\n(V)',
			'Vibrating\nWire$(Hz)$']
			
	#Calibration boundaries
	Height_Boundaries = {0 : [],
		1 : [],
		2 : [],
		3 : [],
		4 : [],
		5 : [10.5,12.0],
		6 : [],
		7 : [],
		8 : [],
		9 : [],
		10 : [10.5,12.0]}
		
    ############################################################################
	"""[Step 1] Check and Import Data"""
    
	#Error check that either Radiosonde_File or Sensor_Package has been specified
	if Radiosonde_File is None and Sensor_Package is None: sys.exit("[Error] You must specify either the Radiosonde_File location or the Sensor_Package number")
	
	#Attempt to find the radiosonde file either directly or from glob
	Radiosonde_File = Storage_Path + Processed_Data_Path + Radiosonde_File if Radiosonde_File is not None else glob.glob(Storage_Path + Processed_Data_Path + 'Radiosonde_Flight_No.' + str(Sensor_Package).rjust(2,'0') + '_*/Radiosonde_Flight_PhD_James_No.' + str(Sensor_Package) + '*a.txt')
	
	#If no radiosonde file was found we end program
	if len(Radiosonde_File) == 0: sys.exit("[Error] Radiosonde package No.%s does not exist. Has the radiosonde been launched yet or has the data been misplaced?" % (Sensor_Package))
	
	#If the radiosonde file was found via glob we need to convert to str from list
	if isinstance(Radiosonde_File, list): Radiosonde_File = Radiosonde_File[0]
	
	#Once the radiosonde file is found we can attempt to find the GPS file in the raw file section
	GPS_File = glob.glob(Storage_Path + Raw_Data_Path + 'Radiosonde_Flight_No.' + str(Sensor_Package).rjust(2,'0') + '_*/GPSDCC_RESULT*.tsv')
	
	#Import all the data
	if Radiosonde_File is not None: Radiosonde_Data = np.genfromtxt(Radiosonde_File, delimiter=None, skip_header=10, dtype=float, comments="#")
	if len(GPS_File) != 0: GPS_Data = np.genfromtxt(GPS_File[0], delimiter=None, skip_header=51, dtype=float, comments="#")
	
	############################################################################
	"""[Step 2] Calibrate bespoke sensors"""
    
	#Calibrate Height, Temperature and Convert PANDORA channels from counts to volts if required.
	Radiosonde_Cal = Radiosonde_Checks(Radiosonde_Data, Calibrate, Sensor_Package, Height_Boundaries[Sensor_Package], check=1111, linear=True, log=False)
		
	#Calibrate just the linear charge sensor
	Radiosonde_Cal.charge_calibration(Sensor_Package, type='current', linear=True, log=False)
	
	#Return Data
	Radiosonde_Data = Radiosonde_Cal.return_data()
		
	Linear = gu.moving_average(Radiosonde_Data[:,5], 11)
	Log = gu.moving_average(Radiosonde_Data[:,6], 11)
	
	PosMask = Linear >= 0
	NegMask = Linear < 0
	
	LinearPos = np.log10(Linear[PosMask])
	LogPos = Log[PosMask]
	
	LinearNeg = -np.log10(-Linear[NegMask])
	LogNeg = Log[NegMask]
		
	#Calculate Linear Regressions
	slope_all, intercept_all, r_value_all, p_value_all, std_err_all = sp.stats.linregress(Log, Linear)
	slope_pos, intercept_pos, r_value_pos, p_value_pos, std_err_pos = sp.stats.linregress(LogPos, LinearPos)
	try:
		slope_neg, intercept_neg, r_value_neg, p_value_neg, std_err_neg = sp.stats.linregress(LogNeg, LinearNeg)
	except:
		slope_neg, intercept_neg, r_value_neg, p_value_neg, std_err_neg = (0,0,0,0,0)
	print(slope_all, intercept_all, r_value_all, p_value_all, std_err_all)
	print(slope_pos, intercept_pos, r_value_pos, p_value_pos, std_err_pos)
	print(slope_neg, intercept_neg, r_value_neg, p_value_neg, std_err_neg)
	
	############################################################################
	"""[Step 3] Plot the calibration values for positive and negative linear currents"""
	
	plt.clf()
	plt.close()
	
	f, ax = plt.subplots(1,3)
	ax[0].plot(Log, Linear , 'p', ms=1, marker='o', markeredgecolor='None', markerfacecolor='black', alpha=1, label="Clouds")
	ax[1].plot(LogPos, LinearPos , 'p', ms=1, marker='o', markeredgecolor='None', markerfacecolor='black', alpha=1, label="Clouds")
	ax[2].plot(LogNeg, LinearNeg , 'p', ms=1, marker='o', markeredgecolor='None', markerfacecolor='black', alpha=1, label="Clouds")
	
	ax[0].plot(Log, slope_all*Log+intercept_all, lw=0.5, c='red')
	ax[1].plot(LogPos, slope_pos*LogPos+intercept_pos, lw=0.5, c='red')
	ax[2].plot(LogNeg, slope_neg*LogNeg+intercept_neg, lw=0.5, c='red')
	
	ax[0].set_ylabel("Linear Sensor Current (A)", fontsize=8)
	ax[1].set_ylabel("Linear Sensor Current (log10(pA))", fontsize=8)
	ax[2].set_ylabel("Linear Sensor Current (-log10(-pA))", fontsize=8)

	for subplot in ax: subplot.minorticks_on()
	for subplot in ax: subplot.set_xlabel("Log Sensor Current (Counts)", fontsize=8)
	for subplot in ax: subplot.grid(which='major',axis='both',c='grey')
	for subplot in ax: subplot.tick_params(axis='both', which='major', labelsize=8)
	for subplot in ax: subplot.tick_params(axis='both', which='minor', labelsize=8)
	
	f.suptitle("Linear and Log Charge Sensors for Radiosonde Flight No.5", y=0.90)
	
	ax[0].get_xaxis().get_major_formatter().labelOnlyBase = False
	
	for subplot in ax:
		x0, x1 = subplot.get_xlim()
		y0, y1 = subplot.get_ylim()
		subplot.set_aspect(np.abs((x1-x0)/(y1-y0)))
	
	ax[0].annotate("All Data", xy=(0, 1), xycoords='axes fraction', xytext=(20, -20), textcoords='offset pixels', horizontalalignment='left', verticalalignment='top', fontsize=8)
	ax[1].annotate("Positive Linear Current", xy=(0, 1), xycoords='axes fraction', xytext=(20, -20), textcoords='offset pixels', horizontalalignment='left', verticalalignment='top', fontsize=8)
	ax[2].annotate("Negative Linear Current", xy=(0, 1), xycoords='axes fraction', xytext=(20, -20), textcoords='offset pixels', horizontalalignment='left', verticalalignment='top', fontsize=8)
	
	ax[0].annotate("$R^{2}$ = %.4f\n$Counts$ = %.0f" % (r_value_all**2, Log.size), xy=(1, 1), xycoords='axes fraction', fontsize=8, xytext=(-3, -3), textcoords='offset points', ha='right', va='top')
	ax[1].annotate("$R^{2}$ = %.4f\n$Counts$ = %.0f" % (r_value_pos**2, LogPos.size), xy=(1, 1), xycoords='axes fraction', fontsize=8, xytext=(-3, -3), textcoords='offset points', ha='right', va='top')
	ax[2].annotate("$R^{2}$ = %.4f\n$Counts$ = %.0f" % (r_value_neg**2, LogNeg.size), xy=(1, 1), xycoords='axes fraction', fontsize=8, xytext=(-3, -3), textcoords='offset points', ha='right', va='top')
				
	f.set_size_inches(11.7, 4.3)
	
	############################################################################
	"""[Step 4] Save plot to file"""
	
	#Specify the directory the plots are stored in 
	path = os.path.dirname(Radiosonde_File).replace(Storage_Path + Processed_Data_Path,"")

	#Find any other plots stored in this directory
	previous_plots = glob.glob(Storage_Path + Plots_Path + path + "/*")
	
	#Find the biggest 'v' number in plots
	plot_version = []
	for plots in previous_plots:
		try:
			plot_version.append(int(os.path.basename(plots)[34:37]))
		except ValueError:
			plot_version.append(int(os.path.basename(plots)[34:36]))
	
	plot_version = str(np.max(plot_version)+1) if len(plot_version) != 0 else '1'
	
	#Create full directory and file name
	Save_Location = Storage_Path + Plots_Path + path + '/' + path + '_v' + plot_version.rjust(2,'0') + '_ChargeCalibrator.png'

	#Ensure the directory exists on file system and save to that location
	gu.ensure_dir(os.path.dirname(Save_Location))
	plt.savefig(Save_Location, bbox_inches='tight', pad_inches=0.1, dpi=300)
	
	#Return regression of positive current, regression of negative current and the boundary for counts
	return (slope_all, intercept_all), (slope_pos, intercept_pos), (slope_neg, intercept_neg), (PosMask, NegMask)
	
def Radiosonde_Lightning(Data, Radiosonde_File=None, Calibrate=None, Height_Range=None, Sensor_Package=None):
	"""Compares lightning data from ATDnet with the position of a radiosonde"""
	
	from matplotlib.ticker import MultipleLocator, FormatStrFormatter, MaxNLocator, LogLocator, ScalarFormatter
	from matplotlib.colors import LogNorm, ListedColormap
	from matplotlib.dates import DateFormatter, MinuteLocator, HourLocator, DayLocator
	
	gu.cprint("[INFO] You are running Radiosonde_Lightning from the STABLE release", type='bold')
    
    ############################################################################
	"""Prerequisites"""
    
	#Time Controls
	t_begin = time.time()
		
	#Storage Locations
	Storage_Path    		= PhD_Global.Storage_Path_WC3
	Processed_Data_Path		= 'Processed_Data/Radiosonde/'
	Raw_Data_Path			= 'Raw_Data/'
	Plots_Path      		= 'Plots/Lightning/'
	
	#Plotting requirements
	plt.style.use('classic') #necessary if Matplotlib version is >= 2.0.0
	
	#Set-up data importer
	EPCC_Data = EPCC_Importer()
	
	#Time Offset
	time_offset = 20.5 #s
	
	############################################################################
	"""[Step 1] Check and Import Data"""
    
	t1 = time.time()
	
	#Error check that either Radiosonde_File or Sensor_Package has been specified
	if Radiosonde_File is None and Sensor_Package is None: sys.exit("[Error] You must specify either the Radiosonde_File location or the Sensor_Package number")
	
	#Attempt to find the radiosonde file either directly or from glob
	Radiosonde_File = Storage_Path + Processed_Data_Path + Radiosonde_File if Radiosonde_File is not None else glob.glob(Storage_Path + Processed_Data_Path + 'Radiosonde_Flight_No.' + str(Sensor_Package).rjust(2,'0') + '_*/Radiosonde_Flight_PhD_James_No.' + str(Sensor_Package) + '*a.txt')
	
	#If no radiosonde file was found we end program
	if len(Radiosonde_File) == 0: sys.exit("[Error] Radiosonde package No.%s does not exist. Has the radiosonde been launched yet or has the data been misplaced?" % (Sensor_Package))
	
	#If the radiosonde file was found via glob we need to convert to str from list
	if isinstance(Radiosonde_File, list): Radiosonde_File = Radiosonde_File[0]
	
	#Once the radiosonde file is found we can attempt to find the GPS file in the raw file section
	GPS_File = glob.glob(Storage_Path + Raw_Data_Path + 'Radiosonde_Flight_No.' + str(Sensor_Package).rjust(2,'0') + '_*/GPSDCC_RESULT*.tsv')
	
	#Import all the data
	if Radiosonde_File is not None: Radiosonde_Data = np.genfromtxt(Radiosonde_File, delimiter=None, skip_header=10, dtype=float, comments="#")
	
	#Get Launch Datetime
	Launch_Datetime, _ = Radiosonde_Launch(GPS_File, offset=time_offset)
	
	#Import ATDnet Data
	ID = np.where(Data['Date_ID'] == gu.toDateOnly(Launch_Datetime))[0][0]
	ATDnet_Time, ATDnet_LatLong = EPCC_Data.ATDnet(Data['ATDnet_File'][ID])
	
	if ATDnet_Time is None: raise IOError("No ATDnet data found for this Radiosonde flight")
	
	############################################################################
	"""[Step 2] Calibrate bespoke sensors"""
    
	t2 = time.time()
	
	#Calibrate Height, Temperature and Convert PANDORA channels from counts to volts if required.
	Radiosonde_Cal_Clean = Radiosonde_Checks(Radiosonde_Data.copy(), Calibrate, Sensor_Package, Height_Range, check=1111, enforce_parity=True)
	Radiosonde_Cal_Error = Radiosonde_Checks(Radiosonde_Data.copy(), Calibrate, Sensor_Package, Height_Range, check=1111, enforce_parity=False)
	
	#Return Data. _Clean : Enforce 1111/1112 parity, _Error: leave other parities alone. Used to identify link with lightning
	Radiosonde_Data_Clean = Radiosonde_Cal_Clean.return_data()
	Radiosonde_Data_Error = Radiosonde_Cal_Error.return_data()
	
	#Non 1111/1112 parity bits converted to nan (except time column)
	Radiosonde_Data_Error[~((Radiosonde_Data_Error[:,9] == 1111) ^ (Radiosonde_Data_Error[:,9] == 1112)),1:] = np.nan
	
	print("Launch_Datetime", Launch_Datetime)
	
	############################################################################
	"""[Step 3] Compare ATDnet with Radiosonde"""
	
	t3 = time.time()
	
	#First, convert Radiosonde time in flight to datetime64
	Radiosonde_Time = np.array(Launch_Datetime, dtype='datetime64[s]') + Radiosonde_Data_Clean[:,0].astype('timedelta64[s]')
	Radiosonde_LatLong = Radiosonde_Data_Clean[:,(11,10)]
	
	t4 = time.time()
	
	#Second, subset ATDnet for times when radiosonde was flying
	ATD_Mask = gu.bool2int((ATDnet_Time >= Radiosonde_Time[0]) & (ATDnet_Time <= Radiosonde_Time[-1]))
	ATDnet_Time = ATDnet_Time[ATD_Mask]
	ATDnet_LatLong = ATDnet_LatLong[ATD_Mask]
	
	#Third, join together the ATDnet timestamps and the Radiosonde timestamps
	mask = gu.mask(Radiosonde_Time, ATDnet_Time, approx=True)
	
	Radiosonde_Time = Radiosonde_Time[mask][0]
	Radiosonde_LatLong = Radiosonde_LatLong[mask][0]

	t5 = time.time()
	
	#Fourth, for each lightning detected, calculate the haversine between the latlong of the lightning and latlong of the radiosonde	
	Lightning_Distance = np.array([gu.haversine(tuple(atdnet_latlong), tuple(radiosonde_latlong))[0] for atdnet_latlong, radiosonde_latlong in zip(ATDnet_LatLong, Radiosonde_LatLong)], dtype=np.float64)

	#Fifth, remove nan values from array (N.B can't use gu.antinan as ATDnet has datetime64 which can't be handled)
	nan_mask = np.isnan(Lightning_Distance)
	ATDnet_Time = ATDnet_Time[~nan_mask]
	Lightning_Distance = Lightning_Distance[~nan_mask]
	
	############################################################################
	"""[Step 4]: Plot time series of lightning strikes"""
	
	Error_Mask = gu.contiguous(np.isnan(Radiosonde_Data_Error[:,5]), invalid=False)
	
	Error_Index = [[gu.bool2int(Error_Mask == val)[0], gu.bool2int(Error_Mask == val)[-1] + 0] for val in np.unique(Error_Mask)[1:]]
	Error_Length = np.diff(Error_Index).ravel()
	Radiosonde_Time_Error = np.array(Launch_Datetime, dtype='datetime64[s]') + Radiosonde_Data_Error[:,0].astype('timedelta64[s]')
		
	#Subset lightning distance to closest 100km strikes
	distance_mask = (Lightning_Distance/1000 < 100)
	ATDnet_Time = ATDnet_Time[distance_mask]
	Lightning_Distance = Lightning_Distance[distance_mask]/1000

	#Clear any previous plots
	gu.backend_changer('nbAgg')
	
	plt.clf()
	plt.close()
	
	#Plot lightning data as time-series
	plt.plot(ATDnet_Time, Lightning_Distance, 'p', ms=3, marker='o', markeredgecolor='None', markerfacecolor='blue', alpha=1, label="Lightning", zorder=4)
	
	plt.ylim([0,100])
	plt.grid(which='major',axis='both',c='grey', zorder=2)
	
	#Configure x axis ticks
	gu.date_ticks(plt.gca(), (ATDnet_Time[0], ATDnet_Time[-1]))
	
	#Write Plot Labels
	plt.ylabel("Distance from Radiosonde (km)")
	plt.title("Distance of Lightning Strikes from Radiosonde Flight No." + str(Sensor_Package))
	
	plt.annotate("$Counts$ = %.0f\n$Closest$ $Strike$ = %.2fkm" % (Lightning_Distance.size, np.nanmin(Lightning_Distance)), xy=(1, 1), xycoords='axes fraction', fontsize=12, xytext=(-5, -5), textcoords='offset points', ha='right', va='top')
	
	#Add fill_between highlighting when communication was lost with the Radiosonde
	cmap = plt.cm.Set2
	norm = plt.matplotlib.colors.Normalize(vmin=1, vmax=9)
	for i, (Error, Length) in enumerate(zip(Error_Index, Error_Length)): 
		plt.axvspan(Radiosonde_Time_Error[Error[0]], Radiosonde_Time_Error[Error[1]], alpha=0.5, color=cmap(norm(Length)), zorder=3)
		
	plt.tight_layout()
	
	#Create colour bar for the communication black-out time
	sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
	sm.set_array([])
	cb = plt.colorbar(sm, orientation='vertical', pad=0.01, ticks=[1,2,3,4,5,6,7,8,9])
	cb.set_label('Comms. Blackout ($s$)', labelpad=1, fontsize=10)
	cb.ax.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8','9'])
	cb.ax.tick_params(labelsize=10)
	
	##SAVE PLOT###
	
	#Specify the directory the plots are stored in 
	path = os.path.dirname(Radiosonde_File).replace(Storage_Path + Processed_Data_Path,"")
	
	#Find any other plots stored in this directory
	previous_plots = glob.glob(Storage_Path + Plots_Path + path + "Timeseries/*")
	
	#Find the biggest 'v' number in plots
	plot_version = []
	for plots in previous_plots:
		try:
			plot_version.append(int(os.path.basename(plots)[34:37]))
		except ValueError:
			plot_version.append(int(os.path.basename(plots)[34:36]))
	
	plot_version = str(np.max(plot_version)+1) if len(plot_version) != 0 else '1'
	
	#Create full directory and file name
	Save_Location = Storage_Path + Plots_Path + path + '/Timeseries/' + path + '_v' + plot_version.rjust(2,'0') + '_LightningTimeseries.png'
	
	#Ensure the directory exists on file system and save to that location
	gu.ensure_dir(os.path.dirname(Save_Location))
	plt.savefig(Save_Location, bbox_inches='tight', pad_inches=0.1, dpi=300)
	
	print("Time series", Save_Location)
	
	############################################################################
	"""[Step 5] Plot lightning strikes and position of radiosonde on a map
	
	
	OPTIONS: USE THE fNorth AND fEast VALUES IN GPSDCC_RESULT TO CALCULATE THE 
	RADIOSONDE POSITION MORE ACCURATELY!"""
	
	from mpl_toolkits.basemap import Basemap
	
	from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
	from mpl_toolkits.axes_grid1.inset_locator import inset_axes
	from matplotlib.patches import Polygon
	from matplotlib.dates import date2num, num2date
	from matplotlib.collections import LineCollection
	
	gu.backend_changer()
	
	#Provide dimensions of map
	lonW_Small = -2.0
	lonE_Small = 1.0
	latN_Small = 52.5
	latS_Small = 50.5

	lonW_Large = -11.0
	lonE_Large = 2.5
	latN_Large = 61.0
	latS_Large = 49.0
	
	#Map positioning
	Position = {'upper right' : (lonE_Small-0.55, latN_Small-0.15), 'upper left' : (lonW_Small+0.2, latN_Small-0.15),
		'lower right' : (lonE_Small-0.55, latS_Small+0.15), 'lower left' : (lonW_Small+0.2, latS_Small+0.15)}
	
	#Map Resolution
	map_res = 'f'
	
	#Create base map
	fig = plt.figure()
	ax = fig.add_subplot(111)

	map = Basemap(projection='merc',
		lat_0=51,
		lon_0=-3,
		resolution=map_res,
		llcrnrlon=lonW_Small,
		llcrnrlat=latS_Small,
		urcrnrlon=lonE_Small,
		urcrnrlat=latN_Small,
		ax=ax)
	
	#Define centre of map
	LatLong_Centre = [0.5*(latN_Small + latS_Small), 0.5*(lonE_Small + lonW_Small)]
	
	#Add overlays to map
	map.drawmapboundary(fill_color='LightBlue', zorder=0)
	map.fillcontinents(color='white', lake_color='LightBlue', zorder=0)
	map.drawcoastlines(color='DimGrey', linewidth=1, zorder=0)
	map.drawcountries(color='Grey', linewidth=1, zorder=0)
	map.drawmeridians(np.arange(-15, 5, 1),linewidth=0.5,color='DarkGrey',labels=[0,0,0,1], zorder=0)
	map.drawparallels(np.arange(-50, 70, 1),linewidth=0.5,color='DarkGrey',labels=[1,0,0,0], zorder=0)
	plt.title('Location of Lightning Strikes and Radiosonde Trajectory (Flight No.5)')

	city_labels = True
	if city_labels is True:
		
		# lat/lon coordinates to plot
		lats = [51.441314]
		lons = [-0.937447]
		
		# compute the native map projection coordinates
		x,y = map(lons,lats)
		
		map.scatter(x,y,s=30, edgecolors='DimGrey', marker='s', facecolors='none', alpha=1, zorder=5)
		
		label_txt = ['RUAO']
		
		for lab in range(0,np.size(x)): 
			plt.text(x[lab], y[lab], label_txt[lab], color='black', size=10,
				horizontalalignment='center', verticalalignment='top', zorder=6)
	
	#Create colour bar for plotting time progression
	Radiosonde_Time = date2num(Radiosonde_Time.astype(datetime))
	ATDnet_Time = date2num(ATDnet_Time.astype(datetime))
	
	#Configure colour-map setting
	cmap = plt.cm.rainbow
	norm = plt.matplotlib.colors.Normalize(vmin=Radiosonde_Time[0], vmax=Radiosonde_Time[-1])
	scalarMap = plt.cm.ScalarMappable(norm=norm, cmap='rainbow')	
	scalarMap.set_array([])
	
	print("Radiosonde_Time", Radiosonde_Time.dtype, Radiosonde_Time.shape, Radiosonde_Time.size)
	print("ATDnet_Time", ATDnet_Time.dtype, ATDnet_Time.shape, ATDnet_Time.size)
	
	#Add ATDnet Lightning Strikes
	x, y = map(ATDnet_LatLong[:,1], ATDnet_LatLong[:,0])
	map.scatter(x, y, s=5, marker='o', edgecolors='None', alpha=1, facecolor=cmap(norm(ATDnet_Time)))

	#Add Radiosonde Trajectory
	x, y = map(Radiosonde_LatLong[:,1], Radiosonde_LatLong[:,0])
	xy = np.vstack((x,y)).T
	for rad_time, start, stop in zip(Radiosonde_Time, xy[:-1], xy[1:]):
		xval, yval = zip(start, stop)
		map.plot(xval, yval, '-', lw=1, color=cmap(norm(rad_time)))
		
	#Add scale bar
	map.drawmapscale(*Position['lower right'],
		lat0=LatLong_Centre[0], lon0=LatLong_Centre[1], length=100, units='km', barstyle='fancy', ax=ax)
	
	#Create inset map axes
	axin = inset_axes(map.ax, width="30%", height="30%", loc=3)

	#Create inset map on the same Mercator Geographic Coordinate System
	omap = Basemap(projection='merc',
		lat_0=51,
		lon_0=-3,
		resolution=map_res,
		llcrnrlon=lonW_Large,
		llcrnrlat=latS_Large,
		urcrnrlon=lonE_Large,
		urcrnrlat=latN_Large,
		ax=axin)
		
	#Add overlays to map
	omap.drawmapboundary(fill_color='LightBlue')
	omap.fillcontinents(color='white',lake_color='LightBlue')
	omap.drawcoastlines(color='DimGrey', linewidth=.5)
	omap.drawcountries(color='Grey', linewidth=.5)
	
	#Add Zoomed Box
	bx, by = omap(map.boundarylons, map.boundarylats)
	xy = list(zip(bx,by))
	mapboundary = Polygon(xy,edgecolor='red',linewidth=2,fill=False)
	omap.ax.add_patch(mapboundary)
	
	#Create colour bar for the communication black-out time
	sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
	sm.set_array([])
	myFmt = plt.matplotlib.dates.DateFormatter('%H:%M')
	cb = plt.colorbar(sm, orientation='vertical', pad=0.01, label="Time (UTC)", format=myFmt, fraction=0.04)
	cb.ax.tick_params(labelsize=10)
	
	##SAVE PLOT###
	
	#Specify the directory the plots are stored in 
	path = os.path.dirname(Radiosonde_File).replace(Storage_Path + Processed_Data_Path,"")
	
	#Find any other plots stored in this directory
	previous_plots = glob.glob(Storage_Path + Plots_Path + path + "/Map/*")
	
	print("LOOKING", Storage_Path + Plots_Path + path + "/Map/")
	print("previous_plots", previous_plots)
	
	#Find the biggest 'v' number in plots
	plot_version = []
	for plots in previous_plots:
		try:
			plot_version.append(int(os.path.basename(plots)[34:37]))
		except ValueError:
			plot_version.append(int(os.path.basename(plots)[34:36]))
	
	plot_version = str(np.max(plot_version)+1) if len(plot_version) != 0 else '1'
	
	#Create full directory and file name
	Save_Location = Storage_Path + Plots_Path + path + '/Map/' + path + '_v' + plot_version.rjust(2,'0') + '_LightningMap.png'
	
	#Ensure the directory exists on file system and save to that location
	gu.ensure_dir(os.path.dirname(Save_Location))
	plt.savefig(Save_Location, bbox_inches='tight', pad_inches=0.1, dpi=300)
	
	print("Map", Save_Location)
	
	t6 = time.time()
	
	print("Time t2-t1 = %.2fs, t3-t2 = %.2fs, t4-t3 = %.2fs, t5-t4 = %.2fs, t6-t5 = %.2fs, Total = %.2fs" % 
		(t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, t6-t1))
	
	print("[INFO] Radiosonde_Lightning completed successfully (In %.2fs)" % (time.time()-t_begin))	
	
	sys.exit()
	
if __name__ == "__main__":
	"""Launch the Radiosonde_Analysis.py from the command line. This python script gives command line
	options which can be found using Radiosonde_Analysis.py --help. An example input for a radiosonde
	flight is given as,
	
	>>> python Radiosonde_Analysis.py --sensor 3 --height 0.0 2.0 --calibrate count
	
	if using Radiosonde_ChargeCalibrator use:
	
	>>> python Radiosonde_Analysis.py --sensor x --height zmin zmax --calibrate volts
	
	where x is the radiosonde flight number, zmin and zmax are the height boundaries where you want to 
	perform the calibration and volts or units is required for calibrate to correctly get the linear
	current. Notes for zmin and zmax are to use an area of the ascent where both linear and log charge
	sensors did not saturate. Otherwise, the calibration will have a systematic bias.
	
	"""
	
	gu.cprint("Welcome to Radiosonde Analysis. Plotting the sounding data and calculating profile indices.", type='bold')
		
	############################################################################
	"""Process Command Line Arguments"""
	
	parser = argparse.ArgumentParser(description='Plot the radiosonde data for each flight during my PhD. The calibration from counts to voltage to quantity is applied automatically if found in Radiosonde_Calibration.py')
	
	#Command Line Arguments
	parser.add_argument('-v','--height',
		action="store", dest="Height_Range", nargs='+', type=float,
		help="Specify the minimum height used for plotting the radiosonde data. The format should be '-h 0 18' where 0 and 18 are the lower and upper bound heights respectively.", 
		default=(0,18), required=True)
	
	parser.add_argument('-s','--sensor',
		action="store", dest="Sensor_Package", type=int,
		help="Specify the radiosonde sensor package you want to plot. Ranges from 1+",
		default=1, required=True)

	parser.add_argument('-c', '--calibrate',
		action="store", dest="Calibrate", type=str,
		help="Specify what level of calibration you want to apply to the research channels. Select either 'counts', 'volts', 'units' are available options",
		default='units', required=False)
	
	parser.add_argument('--tephigram',
		action='store_true', dest="plot_tephigram",
		help="Specify if you want to plot the tephigram of the specify radiosonde flight")
		
	parser.add_argument('--camborne',
		action='store_true', dest="plot_camborne",
		help="Specify if you want to plot the Camborne Upper Level Sounding data on top of the radiosonde data")
	
	parser.add_argument('--verbose',
		action='store_true', dest="verbose",
		help="Specify if you want output extra information about the data processing.")
		
	arg = parser.parse_args()
	arg.plot_tephigram = bool(arg.plot_tephigram)
	arg.plot_camborne = bool(arg.plot_camborne)
	
	if not np.any(np.in1d(arg.Calibrate, ['volts', 'units', 'counts'])): sys.exit("[Error] Radiosonde_Analysis requires the Calibrate argument to be specified with either 'counts', 'volts' or 'units")
	
	#Convert Calibrate and Height_Range into tuples
	arg.Height_Range = tuple(arg.Height_Range)
	
	#Initialise Clouds_ID, LayerType
	Clouds_ID = None
	LayerType = None
	
	############################################################################
	"""Start of Main Function"""
	
	tstart_main = time.time()
	
	gu.cprint("Everything was set-up correctly. Let crunch some numbers!", type='okblue')
	
	Data = PhD_Global.Data_CHIL

	
	Rad = Radiosonde(Sensor_Package=arg.Sensor_Package, Height_Range=arg.Height_Range, Calibrate=arg.Calibrate, verbose=arg.verbose)
	Rad.Superplotter()
	Rad.Tephigram(plot_tephigram=arg.plot_tephigram, plot_camborne=arg.plot_camborne)
	Rad.RH_Comparison()
	sys.exit()
	
	#Identify the clouds within the radiosonde data
	Clouds_ID, LayerType = Radiosonde_CloudIdentifier(Height_Range=arg.Height_Range, Sensor_Package=arg.Sensor_Package)
	
	#Determine the charge calibration for the log sensor
	if np.any(np.in1d(arg.Calibrate, ['volts', 'units'])):
		Calibration_Log = Radiosonde_ChargeCalibrator(Calibrate=arg.Calibrate, Sensor_Package=arg.Sensor_Package)
	
	#Plot Radiosonde Data together in Cartesian Coordinates
	if np.any(np.in1d(arg.Calibrate, ['volts', 'units'])):
		Radiosonde_Superplotter(Calibrate=arg.Calibrate, Height_Range=arg.Height_Range, Sensor_Package=arg.Sensor_Package, Clouds_ID=Clouds_ID, LayerType=LayerType, Calibration_Log=Calibration_Log)
	else:
		Radiosonde_Superplotter(Calibrate=arg.Calibrate, Height_Range=arg.Height_Range, Sensor_Package=arg.Sensor_Package, Clouds_ID=Clouds_ID, LayerType=LayerType)
	sys.exit()
	
	
	#Plot Radiosonde Data together in Tephigram Coordinates
	Radiosonde_Tephigram(Height_Range=arg.Height_Range, Sensor_Package=arg.Sensor_Package, plot_tephigram=arg.plot_tephigram, plot_camborne=arg.plot_camborne)
	
	#Plot Lightning maps and comparison with Radiosonde Trajectory
	Radiosonde_Lightning(Data, Calibrate=arg.Calibrate, Height_Range=arg.Height_Range, Sensor_Package=arg.Sensor_Package)
	
	#IN THE FUTURE:
	#Radiosonde_ChargeCalibrator(Calibrate=arg.Calibrate, Height_Range=arg.Height_Range, Sensor_Package=arg.Sensor_Package)
	#Radiosonde_Ice_Concentration(Calibrate=arg.Calibrate, Height_Range=arg.Height_Range, Sensor_Package=arg.Sensor_Package)
	
	gu.cprint("[Radiosonde_Analysis]: All Tasks Completed, Time Taken (s): %.0f" % (time.time()-tstart_main), type='okgreen')