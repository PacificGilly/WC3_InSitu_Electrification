############################################################################
# Project: Electrical Pre-Conditioning of Convective Clouds,
# Title: Plotting Radiosonde Data
# Author: James Gilmore,
# Email: james.gilmore@pgr.reading.ac.uk.
# Version: 1.1
# Date: 24/07/2018
# Status: Stable
# Change: Too many to mention! Save new versions more regularly!!!
############################################################################
from __future__ import absolute_import, division, print_function
import numpy as np
import scipy as sc
import os, sys, time, warnings, glob, argparse
from datetime import datetime, timedelta

sys.path.insert(0, '/home/users/th863480/PhD/Global_Functions')

#User Processing Modules
import Gilly_Utilities as gu

#Data Set-up Modules
from Data_Importer import EPCC_Importer
from Data_Output import SPRadiosonde

#Import Global Variables
import PhD_Config as PhD_Global

#Import Tephigram Plotter
from Tephigram import Tephigram as SPTephigram

from Radiosonde_Calibration import Radiosonde_Checks

def Radiosonde_Superplotter(Radiosonde_File=None, Calibrate=None, Height_Range=None, Sensor_Package=None):
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
	Version : str, optional	
		The version of plots you want to use

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
		Pandora_Labels = ['Charge (Counts)',
			'Cloud Sensor\n(Counts)',
			'PLL (counts)']
	elif Calibrate == "volts":
		Pandora_Labels = ['Charge (V)',
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
	Radiosonde_Cal = Radiosonde_Checks(Radiosonde_Data, Calibrate, Sensor_Package, Height_Range, check=1111)
	
	#Calibrate OMB Sensor
	if Calibrate == "units": Radiosonde_Cal.wire_calibration()
	
	#Calibrate Charge Sensor
	if Calibrate == "units": Radiosonde_Cal.charge_calibration(Sensor_Package)
	
	#Return Data
	Radiosonde_Data = Radiosonde_Cal.return_data()
		
	#Calculate launch datetime
	#print("INDEX", np.arange(len(GPS_Data[:,4]))[GPS_Data[:,4] > 0]+51)
	#sys.exit()
	
	GPS_Data = GPS_Data[GPS_Data[:,4] > 0]
	#GPS_Data = GPS_Data[7718:] #for No.4
	if GPS_File is not None: Launch_Datetime = GPS2UTC(GPS_Data[0,1], GPS_Data[0,2])

	############################################################################
	"""[Step 3] Plot radiosonde data"""

	Title = 'Radiosonde Flight No.' + str(Sensor_Package) + ' (' + Launch_Datetime.strftime("%d/%m/%Y %H%MUTC") + ')' if GPS_File is not None else 'Radiosonde Flight (N/A)'
	Superplotter = SPRadiosonde(8, Title, Height_Range, Radiosonde_Data) if Calibrate == "units" else SPRadiosonde(7, Title, Height_Range, Radiosonde_Data)
	
	#Plot the PANDORA data
	Superplotter.Charge(Linear_Channel=0, Log_Channel=1, XLabel=Pandora_Labels[0])
	Superplotter.Cloud(Cyan_Channel=2, IR_Channel=3, XLabel=Pandora_Labels[1], Cyan_Check=1111)
	Superplotter.PLL(PLL_Channel=2, XLabel=Pandora_Labels[2], PLL_Check=1112, Point=False, Calibrate=Calibrate) if Sensor_Package < 3 else Superplotter.PLL(PLL_Channel=2, XLabel=Pandora_Labels[2], PLL_Check=1112, Point=True, Calibrate=Calibrate)
	
	#Plot the processed PLL data
	#if Calibrate == "units": Superplotter.ch(13, 'dPLLdt $(Hz$ $s^{-1})$', 'dPLL/dt', check=1112, point=True)
	if Calibrate == "units": Superplotter.ch(14, 'SLWC $(g$ $m^{-3})$', 'Supercooled Liquid\nWater Concentration', check=1112, point=True)
	
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
	Superplotter._PlotSave(Save_Location)
	
	print("[INFO] Radiosonde_Superplotter completed successfully (In %.2fs)" % (time.time()-t_begin))

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
	UTC_Time = GPS_Epoch + timedelta(weeks=GPS_Week+(1024*Rollover), seconds=GPS_Second+LeapSeconds)
	
	return UTC_Time

def Radiosonde_Ice_Concentration(Radiosonde_File=None, Calibrate=None, Height_Range=None, Sensor_Package=None):
	
	print("[INFO] You are running Radiosonde_Ice_Concentration from the DEV release")
    
    ############################################################################
	"""Pre-requisities"""
    
	Storage_Path    		= '/glusterfs/phd/users/th863480/WC3_InSitu_Electrification/'
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

def Radiosonde_Tephigram(Radiosonde_File=None, Height_Range=None, Sensor_Package=None, plot_tephigram=False, plot_camborne=False):
	"""The Radiosonde_Tephigram function will plot a tephigram fron the dry bulb temperature,
	T_dry and the Dew point Temperature, T_dew for pressure values, P at each corresponding 
	height. 
	
	Certain tephigram outputs are available from this function including:
	1) Lower Condensation Level (LCL) in m
	2) Level of Free Convection (LFC) in m
	3) Evironmental Level (EL) in m
	4) Convective Available Potential Energy (CAPE) in J/kg
	5) Convective INhibition (CIN) in J/kg

	Parameters
	----------
	
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
	Radiosonde_Cal = Radiosonde_Checks(Radiosonde_Data, calibrate=None, package_no=Sensor_Package, height_range=Height_Range, check=1111)
	
	#Return Data
	Radiosonde_Data = Radiosonde_Cal.return_data()

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
	
	if plot_tephigram is True:
	
		print("[INFO] Plotting Tephigram...")
	
		############################################################################
		"""Create Tephigram"""
        
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
			
			#Determine 
			ULS_File = sorted(glob.glob(PhD_Global.Storage_Path_WC2 + 'Raw_Data/Met_Data/ULS/*'))
			
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
		
		Tephigram.savefig(Save_Location)
		#Tephigram.show()
        

	############################################################################
	"""[Step ??] Calculate Stability Indices"""
	
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
				if (Wind_Mag[P_500]*ms2kn > 15) | (Wind_Mag[P_850]*ms2kn > 15):
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
	"""[Step ??] Calculate Tephigram Indices"""
	
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
	
	print("LCL", LCL)	
	#sys.exit()
	
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
	TEMP = sc.interpolate.interp1d(Pthetaeq,[thetaarr,Tarr], fill_value="extrapolate")(Pres)
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
	
if __name__ == "__main__":
	"""Launch the Radiosonde_Analysis.py from the command line. This python script gives command line
	options which can be found using Radiosonde_Analysis.py --help. An example input for a radiosonde
	flight is given as,
	
	>>> python Radiosonde_Analysis.py --sensor 3 --height 0.0 2.0 --calibrate count
	
	"""
	
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
		action="store", dest="plot_tephigram",
		help="Specify if you want to plot the tephigram of the specify radiosonde flight", 
		default='False')
		
	parser.add_argument('--camborne',
		action="store", dest="plot_camborne",
		help="Specify if you want to plot the Cam borne Upper Level Sounding data on top of the radiosonde data", 
		default='False')
	
	arg = parser.parse_args()
	arg.plot_tephigram = bool(arg.plot_tephigram)
	arg.plot_camborne = bool(arg.plot_camborne)
	
	if not np.any(np.in1d(arg.Calibrate, ['volts', 'units', 'counts'])): sys.exit("[Error] Radiosonde_Analysis requires the Calibrate argument to be specified with either 'counts', 'volts' or 'units")
	
	#Convert Calibrate and Height_Range into tuples
	arg.Height_Range = tuple(arg.Height_Range)
	
	#Plot Radiosonde Data together in Cartesian Coordinates
	#Radiosonde_Superplotter(Calibrate=arg.Calibrate, Height_Range=arg.Height_Range, Sensor_Package=arg.Sensor_Package)
	
	#Plot Radiosonde Data together in Tephigram Coordinates
	Radiosonde_Tephigram(Height_Range=arg.Height_Range, Sensor_Package=arg.Sensor_Package, plot_tephigram=arg.plot_tephigram, plot_camborne=arg.plot_camborne)
	
	sys.exit()
	
	#Radiosonde_Ice_Concentration(Calibrate=arg.Calibrate, Height_Range=arg.Height_Range, Sensor_Package=arg.Sensor_Package)
	
	
	#Radiosonde_File = 'Radiosonde_Flight_No.2_20180302/Radiosonde_Flight_PhD_James_No.2_20180302a.txt'
	#GPS_File = 'Radiosonde_Flight_No.2_20180302/GPSDCC_RESULT20180302c.tsv'
	