############################################################################
# Project: Electrical Pre-Conditioning of Convective Clouds,
# Title: Plotting Radiosonde Data
# Author: James Gilmore,
# Email: james.gilmore@pgr.reading.ac.uk.
# Version: 1.0
# Date: 02/03/2018
# Status: Stable
############################################################################
from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, time, warnings, glob, argparse
from datetime import datetime, timedelta

sys.path.insert(0, '/home/users/th863480/PhD/Global_Functions')
import Gilly_Utilities as gu
from externals import SPRadio

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

	#os.system('cls' if os.name=='nt' else 'clear')
	print("[INFO] You are running Radiosonde_Superplotter from the DEV release")
    
    ############################################################################
	"""Pre-requisities"""
    
	Storage_Path    		= '/glusterfs/phd/users/th863480/WC3_InSitu_Electrification/'
	Processed_Data_Path		= 'Processed_Data/Radiosonde/'
	Raw_Data_Path			= 'Raw_Data/'
	Plots_Path      		= 'Plots/Radiosonde/'
    
	plt.style.use('classic') #neseccary if matplotlib version is >= 2.0.0
	
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
	
	#Once the radiosonde file is found we can attempt to find the GPS file in the raw file section
	GPS_File = glob.glob(Storage_Path + Raw_Data_Path + 'Radiosonde_Flight_No.' + str(Sensor_Package).rjust(2,'0') + '_*/GPSDCC_RESULT*.tsv')
	
	#Import all the data
	if Radiosonde_File is not None: Radiosonde_Data = np.genfromtxt(Radiosonde_File, delimiter=None, skip_header=10, dtype=float, comments="#")
	if len(GPS_File) != 0: GPS_Data = np.genfromtxt(GPS_File[0], delimiter=None, skip_header=51, dtype=float, comments="#")
	
	############################################################################
	"""[Step 2] Calibrate bespoke sensors"""
    
	Radiosonde_Cal = Radiosonde_Checks(Radiosonde_Data, Calibrate, Sensor_Package, Height_Range, check=1111)
	
	Radiosonde_Data = Radiosonde_Cal.wire_calibration()
	
	#Calibrate Charge Sensor
	if Sensor_Package is not None: Radiosonde_Data = Radiosonde_Cal.charge_calibration(Sensor_Package)
	
	#Calculate launch datetime
	GPS_Data = GPS_Data[GPS_Data[:,4] > 0]
	if GPS_File is not None: Launch_Datetime = GPS2UTC(GPS_Data[0,1], GPS_Data[0,2])

	############################################################################
	"""[Step 3] Plot radiosonde data"""

	Title = 'Radiosonde Flight No.' + str(Sensor_Package) + ' (' + Launch_Datetime.strftime("%d/%m/%Y %H%MUTC") + ')' if GPS_File is not None else 'Radiosonde Flight (N/A)'
	Superplotter = SPRadio(11, Title, Height_Range, Radiosonde_Data)
	
	Superplotter.ch(0, 'Charge Linear $(nA)$')
	Superplotter.ch(1, 'Charge Log $(nA)$')
	Superplotter.ch(2, 'Cloud Cyan $(V)$', check=1111)
	Superplotter.ch(3, 'Cloud IR $(V)$')
	Superplotter.ch(2, 'PLL $(Hz)$', check=1112)
	Superplotter.ch(13, 'dPLLdt $(Hz$ $s^{-1})$', check=1112)
	Superplotter.ch(14, 'Ice Concentration $(g$ $m^{-3})$', check=1112)
	
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
			plot_version.append(int(plots[-19:-17]))
		except ValueError:
			plot_version.append(int(plots[-18:-17]))
	
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
	
if __name__ == "__main__":
    
	parser = argparse.ArgumentParser(description='Plot the radiosonde data for each flight during my PhD. The calibration from counts to voltage to quantity is applied automatically if found in Radiosonde_Calibration.py')
	
	#Command Line Arguments
	parser.add_argument('-v','--height',
		action="store", dest="Height_Range", nargs='+', type=int,
		help="Specify the minimum height used for plotting the radiosonde data. The format should be '-h 0 18' where 0 and 18 are the lower and upper bound heights respectively.", 
		default=(0,18), required=True)
	
	parser.add_argument('-s','--sensor',
		action="store", dest="Sensor_Package", type=int,
		help="Specify the radiosonde sensor package you want to plot. Ranges from 1+",
		default=1, required=True)
		
	parser.add_argument('-c','--calibrate',
		action="store", dest="Calibrate", nargs='+', type=int,
		help="Specify the Pandora channels you want to calibrate ranging from 0 to 4. The format should be '-c 0 1 2 3' which calibrates channels 0,1,2,3. The default argument is to calibrate all channels",
		default=(), required=False)	
		
	arg = parser.parse_args()
	
	#Convert Calibrate and Height_Range into tuples
	arg.Height_Range = tuple(arg.Height_Range)
	arg.Calibrate = tuple([val + 5 for val in arg.Calibrate])
	
	#Radiosonde_Ice_Concentration(Calibrate=arg.Calibrate, Height_Range=arg.Height_Range, Sensor_Package=arg.Sensor_Package)
	Radiosonde_Superplotter(Calibrate=arg.Calibrate, Height_Range=arg.Height_Range, Sensor_Package=arg.Sensor_Package)
	
	
	#Radiosonde_File = 'Radiosonde_Flight_No.2_20180302/Radiosonde_Flight_PhD_James_No.2_20180302a.txt'
	#GPS_File = 'Radiosonde_Flight_No.2_20180302/GPSDCC_RESULT20180302c.tsv'
	