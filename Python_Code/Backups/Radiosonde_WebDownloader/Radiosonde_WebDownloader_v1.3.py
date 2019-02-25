############################################################################
# Project: Electrical Pre-Conditioning of Convective Clouds,
# Title: Collecting Web Data
# Author: James Gilmore,
# Email: james.gilmore@pgr.reading.ac.uk.
# Version: 1.3
# Date: 12/12/2018
# Status: Stable
# Change: Added in support for downloading Sat24 and Wetter3 surface pressure images
############################################################################
from __future__ import absolute_import, division, print_function
import numpy as np
import os, urllib, warnings, sys, argparse
import time as systime
from datetime import datetime, timedelta

with warnings.catch_warnings():
	warnings.simplefilter("ignore")

	sys.path.insert(0, '/home/users/th863480/PhD/Global_Functions')
	
	#Import Global Variables
	import PhD_Config as PhD_Global
	
	#User Processing Modules
	import Gilly_Utilities as gu
	
	#Import javascript handler
	sys.path.insert(0,'/home/users/th863480/PhD/Global_Functions/Prerequisites/modules/')
	from selenium import webdriver
	
	eps = sys.float_info.epsilon	
	
def ESTOFEX(Date, Radiosonde_Flight_Num):
	"""Downloads all the GFS model output files from a selected initiation time until 72 hours lead time
	
	Parameters
	----------
	Date : datetime object
		The datetime you want to start collecting the data from. Should have the format 
		datetime(%Y, %m, %d, %H) where %H can be either 0, 6, 12 or 18. Any other value will be rounded 
		to the nearest value defined.
	Radiosonde_Flight_Num : int
		The radiosonde flight number you want to download the images for
		
	Example
	-------
	python Radiosonde_WebDownloader.py --date 31/05/2018_0 --sonde 4
	
	"""
	
	gu.cprint("[Info] You are running ESTOFEX from the STABLE release", type='bold')
		
	############################################################################
	"""Prerequisites"""
	
	#Time Controls
	t_begin = systime.time()
		
	#Storage Locations
	Storage_Path = PhD_Global.Storage_Path_WC3
	Plots_Path = 'Plots/Meteorology/Radiosonde_Flight_No.' + str(Radiosonde_Flight_Num).rjust(2,'0') + "_" + Date.strftime("%Y%m%d") + '/GFS_ESTOFEX/'
		
	#Data Information
	Product_Hour = np.arange(0,75,3).astype(str)
	
	Product_Type = ['0-1shear', '0-3shear', '0-6shear', 
		'300', '500', '700', '850', 
		'cape', 'capeshear', 'el', 
		'inindex', 'ipv320', 'lghail', 
		'li700', 'mixr', 'nconvgust', 
		'precip', 'q-vector', 'srh1', 
		'srh3', 't2m', 'td2m', 'trop']
	
	#Ensure Directory Locations
	for Product in Product_Type:
		gu.ensure_dir(Storage_Path + Plots_Path + Product + '/')
		
	############################################################################
	"""Error Checking"""
	
	#Check Date has been specified correctly
	if not isinstance(Date, datetime): sys.exit("[ERROR] ESTOFEX requires the parameter Date to be in datetime format")
	
	#Check Hour values is of correct type [00, 06, 12, 18]
	Hour_Vals_Str = ['00','06','12','18']
	Hour_Vals_Num = [0,6,12,18]
	if not np.any(np.in1d(Hour_Vals_Str, Date.strftime("%H"))):
		warnings.warn('\n[ESTOFEX]: Date did not have a suitable hour value. Only hour values of 0, 6, 12 and 18 are available. We will round to the nearest interval!', SyntaxWarning, stacklevel=2)
		Date = Date.replace(hour=gu.near(Hour_Vals_Num, Date.hour))
	
	############################################################################
	"""Download all images to file"""
	
	Files_Processed = 0
	t_stage1 = systime.time()
	Total_Files_to_Process = len(Product_Hour) * len(Product_Type)
	for Product in Product_Type:
		for Hour in Product_Hour:
			
			############################################################################
			"""Update user with processing status"""
			
			Files_Processed += 1
			
			#Time Statistics
			t_elapsed = systime.time()-t_stage1
			t_remain  = (t_elapsed/(Files_Processed + eps))*(Total_Files_to_Process - Files_Processed + eps)
			
			#Output information to user.
			sys.stdout.write('Downloading ESTOFEX GFS images for Parameter %s and Time Step %s. Time Elapsed: %.0fs. Time Remaining: %.0fs. \r' %  (Product, Hour, t_elapsed, t_remain))
			sys.stdout.flush()
			
			############################################################################
			"""Download Images"""
			
			urllib.urlretrieve("http://www.estofex.org/modelmaps/maps/" + Date.strftime("%Y%m") + "/" + Date.strftime("%Y%m%d%H") + "/"	+ Hour + "_" + Product + ".png", Storage_Path + Plots_Path + Product + "/GFS_ESTOFEX_" + Date.strftime("%Y%m%d%H") + "_" + Product + "_" + Hour + ".png")
			
			sys.stdout.write("\033[K") #Removes last line in command line
			
	print("[INFO] ESTOFEX has been completed successfully in %.1fs" % (systime.time()-t_begin))
	
def Lightning_Wizard(Date, Radiosonde_Flight_Num):
	"""Downloads all the GFS model output files from a selected PhD_Globaltiation time until 72 hours lead time
	
	Parameters
	----------
	Date : datetime object
		The datetime you want to start collecting the data from. Should have the format 
		datetime(%Y, %m, %d, %H) where %H can be either 0, 6, 12 or 18. Any other value will be rounded 
		to the nearest value defined.
	Radiosonde_Flight_Num : int
		The radiosonde flight number you want to download the images for
		
	Example
	-------
	python Radiosonde_WebDownloader.py --date 31/05/2018_0 --sonde 4
	
	"""
	
	gu.cprint("[Info] You are running Lightning_Wizard from the STABLE release", type='bold')
		
	############################################################################
	"""Prerequisites"""
	
	#Time Controls
	t_begin = systime.time()
		
	#Storage Locations
	Storage_Path = PhD_Global.Storage_Path_WC3
	Plots_Path = 'Plots/Meteorology/Radiosonde_Flight_No.' + str(Radiosonde_Flight_Num).rjust(2,'0') + "_" + Date.strftime("%Y%m%d") + '/GFS_Lightning_Wizard/'
		
	#Data Information
	Product_Hour = np.arange(0,75,3).astype(str)
	
	Product_Type = ['cape', 'mucape', 'icape', 
		'layer', 'lfc', 'mulfc', 
		'el', 'icon10', 'omega', 
		'pvort', 'pvort2', 'difadv', 
		'kili', 'spout', 'lapse', 
		'lapse2', 'the700', 'thetae', 
		'mixr', 'mtv', 'gusts', 
		'stp', 'srh', 'srhl', 
		'pw', 'hail']
	
	#Ensure Directory Locations
	for Product in Product_Type:
		gu.ensure_dir(Storage_Path + Plots_Path + Product + '/')
		
	############################################################################
	"""Error Checking"""
	
	#Check Date has been specified correctly
	if not isinstance(Date, datetime): sys.exit("[ERROR] Lightning_Wizard requires the parameter Date to be in datetime format")
	
	Date += timedelta(hours=12)
	
	#Check Hour values is of correct type [00, 06, 12, 18]
	#Hour_Vals_Str = ['00','06','12','18']
	#Hour_Vals_Num = [0,6,12,18]
	#if not np.any(np.in1d(Hour_Vals_Str, Date.strftime("%H"))):
	#	warnings.warn('\n[Lightning_Wizard]: Date did not have a suitable hour value. Only hour values of 0, 6, 12 and 18 are available. We will round to the nearest interval!', SyntaxWarning, stacklevel=2)
	#	Date = Date.replace(hour=gu.near(Hour_Vals_Num, Date.hour))
	
	############################################################################
	"""Download all images to file"""
	
	Files_Processed = 0
	t_stage1 = systime.time()
	Total_Files_to_Process = len(Product_Hour) * len(Product_Type)
	for Product in Product_Type:
		for Hour in Product_Hour:
			
			############################################################################
			"""Update user with processing status"""
			
			Files_Processed += 1
			
			#Time Statistics
			t_elapsed = systime.time()-t_stage1
			t_remain  = (t_elapsed/(Files_Processed + eps))*(Total_Files_to_Process - Files_Processed + eps)
			
			#Output information to user.
			sys.stdout.write('Downloading Lightning_Wizard GFS images for Parameter %s and Time Step %s. Time Elapsed: %.0fs. Time Remaining: %.0fs. \r' %  (Product, Hour, t_elapsed, t_remain))
			sys.stdout.flush()
			
			############################################################################
			"""Download Images"""
			
			urllib.urlretrieve("http://www.lightningwizard.com/maps/Europe/gfs_" + Product + "_eur" + Hour + ".png", Storage_Path + Plots_Path + Product + "/GFS_Lightning_Wizard_" + Date.strftime("%Y%m%d%H") + "_" + Product + "_" + Hour + ".png")
			
			sys.stdout.write("\033[K") #Removes last line in command line
			
	print("[INFO] Lightning_Wizard has been completed successfully in %.1fs" % (systime.time()-t_begin))

def Sat24(Date, Radiosonde_Flight_Num):
	"""Downloads all the Sat24 images from a selected time until 72 hours lead time
	
	Parameters
	----------
	Date : datetime object
		The datetime you want to start collecting the data from. Should have the format 
		datetime(%Y, %m, %d, %H) where %H can be either 0, 6, 12 or 18. Any other value will be rounded 
		to the nearest value defined.
	Radiosonde_Flight_Num : int
		The radiosonde flight number you want to download the images for
		
	Example
	-------
	python Radiosonde_WebDownloader.py --date 31/05/2018_0 --sonde 4
	
	Reference
	---------
	http://www.sat24.com/h-image.ashx?region=eu&time=201812050100&ir=True
	
	Notes
	-----
	The satellite images does not have a static file location. The image must be requested before
	the download can begin. Hence why there is a lot of code used to fill in the actual drop-down
	boxes that would be required to retrieve the images by hand! Sadly, filling in the data is non
	trivial and care and attention must be given to make sure the correct datetime is being
	requested. Hence the while loops.
	
	"""
	
	gu.cprint("[Info] You are running Sat24 from the STABLE release", type='bold')
		
	############################################################################
	"""Prerequisites"""
	
	#Time Controls
	t_begin = systime.time()
		
	#Storage Locations
	Storage_Path = PhD_Global.Storage_Path_WC3
	Plots_Path = 'Plots/Meteorology/Radiosonde_Flight_No.' + str(Radiosonde_Flight_Num).rjust(2,'0') + "_" + Date.strftime("%Y%m%d") + '/Sat24/'
		
	#Data Information
	Product_Hour = np.arange(1,24,1).astype(str)
	
	Product_Type = ['vis', 'ir']
	
	#Ensure Directory Locations
	for Product in Product_Type:
		gu.ensure_dir(Storage_Path + Plots_Path + Product + '/')
	
	############################################################################
	
	#Check Date has been specified correctly
	if not isinstance(Date, datetime): raise ValueError("[ERROR] Sat24 requires the parameter Date to be in datetime format")
	
	Date += timedelta(hours=12)
	
	############################################################################
	"""Download all images to file"""
	
	driver = webdriver.PhantomJS(executable_path='/home/users/th863480/PhD/Global_Functions/Prerequisites/modules/phantomjs/bin/phantomjs')
	driver.get('http://www2.sat24.com/history.aspx?culture=en')
		
	Files_Processed = 0
	t_stage1 = systime.time()
	Total_Files_to_Process = len(Product_Hour) * len(Product_Type)
	for Product in Product_Type:
		ir = 'True' if Product == 'ir' else 'False'
		for Hour in Product_Hour:
			
			############################################################################
			"""Update user with processing status"""
			
			Files_Processed += 1
			
			#Time Statistics
			t_elapsed = systime.time()-t_stage1
			t_remain  = (t_elapsed/(Files_Processed + eps))*(Total_Files_to_Process - Files_Processed + eps)
			
			#Output information to user.
			sys.stdout.write('Downloading Sat24 images. Time Step %s. Time Elapsed: %.0fs. Time Remaining: %.0fs. \r' %  (Hour, t_elapsed, t_remain))
			sys.stdout.flush()
			
			############################################################################
			"""Download Images"""
			
			#Get web elements
			button = driver.find_element_by_id(id_='ctl00_maincontent_buttonRetrieve')
			day = driver.find_element_by_id(id_='ctl00_maincontent_dropdownListDay')
			month = driver.find_element_by_id(id_='ctl00_maincontent_dropdownListMonth')
			year = driver.find_element_by_id(id_='ctl00_maincontent_dropdownListYear')
			hour = driver.find_element_by_id(id_='ctl00_maincontent_dropdownListHour')
			minute = driver.find_element_by_id(id_='ctl00_maincontent_dropdownListMinute')
			ir_box = driver.find_element_by_id(id_='ctl00_maincontent_checkBoxInfrared')
			
			#Fill in data to web elements
			day.send_keys(int(Date.strftime('%d')))
			month.send_keys(int(Date.strftime('%m')))
			year.send_keys(int(Date.strftime('%Y')))
			hour.send_keys(int(Hour))
			if Product == 'ir': ir_box.click()
			
			#Make sure data has been set correctly
			Driver_Elements = (hour,day,month,year)
			Actual_Elements = (int(Hour), int(Date.strftime('%d')), int(Date.strftime('%m')), int(Date.strftime('%Y')))
			for driver_element, actual_element in zip(Driver_Elements,Actual_Elements):
				iter = 0
				error = True
				while error is True:
					if actual_element != int(driver_element.get_attribute('value')):
						driver_element.send_keys(actual_element)
						iter += 1
					elif iter == 10:
						break
					else:
						error = False
			
			#Check if IR box is in correct state
			iter = 0
			error = True
			if Product == 'ir':
				while error is True:
					if ir_box.is_selected() is False:
						ir_box.click()
						iter += 1
					elif iter == 10:
						break
					else:
						error = False
			else:
				while error is True:
					if ir_box.is_selected() is True:
						ir_box.click()
						iter += 1
					elif iter == 10:
						break
					else:
						error = False

			#Submit data to sat24 servers
			button.click()
			
			#Find image location
			img = driver.find_element_by_id(id_='ctl00_maincontent_imageSat')
			src = img.get_attribute('src')
			
			#Download processed image
			urllib.urlretrieve(src, Storage_Path + Plots_Path + Product + "/sat24_" + Product + "_" + Date.strftime('%Y%m%d') + "_" + str(int(Hour)-1).rjust(2,'0').ljust(4,'0') + "UTC.png")
			
			sys.stdout.write("\033[K") #Removes last line in command line
			
	print("[INFO] Sat24 has been completed successfully in %.1fs" % (systime.time()-t_begin))	

def Surface_Pressure(Date, Radiosonde_Flight_Num):
	"""Downloads all the UKMO surface pressure analysis charts from www1.wetter3.de. Retrieval of the charts from this location
	are better as 6h analysis are given compared with the 12h analysis available directly from the UKMO website.
	
	Parameters
	----------
	Date : datetime object
		The datetime you want to start collecting the data from. Should have the format 
		datetime(%Y, %m, %d, %H) where %H can be either 0, 6, 12 or 18. Any other value will be rounded 
		to the nearest value defined.
	Radiosonde_Flight_Num : int
		The radiosonde flight number you want to download the images for
	
	Notes
	-----
	Please wait until the next day to retrieve the analysis charts for both UKMO to process the data and wetter3 to upload them
	to their server. Fortunately, wetter3 are very reliable and quick at uploading these charts!
	
	Example
	-------
	python Radiosonde_WebDownloader.py --date 31/05/2018_0 --sonde 4
	
	"""
	
	gu.cprint("[Info] You are running Surface_Pressure from the STABLE release", type='bold')
		
	############################################################################
	"""Prerequisites"""
	
	#Time Controls
	t_begin = systime.time()
		
	#Storage Locations
	Storage_Path = PhD_Global.Storage_Path_WC3
	Plots_Path = 'Plots/Meteorology/Radiosonde_Flight_No.' + str(Radiosonde_Flight_Num).rjust(2,'0') + "_" + Date.strftime("%Y%m%d") + '/Surface_Pressure/'
	
	#Data Information
	Product_Hour = np.arange(0,24,6).astype(str)
	
	#Ensure Directory Locations
	gu.ensure_dir(Storage_Path + Plots_Path)
		
	############################################################################
	"""Error Checking"""
	
	#Check Date has been specified correctly
	if not isinstance(Date, datetime): sys.exit("[ERROR] Surface_Pressure requires the parameter Date to be in datetime format")
	
	#Convert Date to correct string format for wetter3 api
	Date = Date.strftime('%y%m%d')
	
	############################################################################
	"""Download all images to file"""
	
	Files_Processed = 0
	t_stage1 = systime.time()
	Total_Files_to_Process = len(Product_Hour)
	for Hour in Product_Hour:
		
		############################################################################
		"""Update user with processing status"""
		
		Files_Processed += 1
		
		#Time Statistics
		t_elapsed = systime.time()-t_stage1
		t_remain  = (t_elapsed/(Files_Processed + eps))*(Total_Files_to_Process - Files_Processed + eps)
		
		#Output information to user.
		sys.stdout.write('Downloading UKMO surface pressure charts. Time Elapsed: %.0fs. Time Remaining: %.0fs. \r' %  (t_elapsed, t_remain))
		sys.stdout.flush()
		
		############################################################################
		"""Download Images"""
		
		urllib.urlretrieve("http://www1.wetter3.de/Archiv/UKMet/" + Date + Hour + "_UKMet_Analyse.gif", Storage_Path + Plots_Path + Date + Hour + "_UKMet_Analyse.png")
		
		sys.stdout.write("\033[K") #Removes last line in command line
			
	print("[INFO] Surface_Pressure has been completed successfully in %.1fs" % (systime.time()-t_begin))
	
if __name__ == "__main__":
	"""Launch the Radiosonde_WebDownloader.py from the command line. This python script gives command line
	options which can be found using Radiosonde_WebDownloader.py --help. An example input for a radiosonde
	flight is given as,
	
	>>> python Radiosonde_WebDownloader.py --date 31/05/2018_0 --sensor 4
	
	72 hours at 3 hour intervals GFS plots are downloaded from the model PhD_Globaltiation date specified
	
	"""

	parser = argparse.ArgumentParser(description='Collect all ESTOFEX GFS plots for all data products \
upto 72 hours forecast time including information on shear, cape and moisture levels. \
You must supply the date and radiosonde sensor package number that relates to your ascent. \
The date relates to the PhD_Globaltation time of the GFS analysis.\n\n\
Example \n\
------- \n\n\
python Radiosonde_WebDownloader.py --date 31/05/2018_0 --sensor 4',
	formatter_class=argparse.RawTextHelpFormatter)
	
	#Command Line Arguments
	parser.add_argument('-d', '--date',
		action="store", dest="Date",
		help="The date and time you want to start collecting GFS data from. Format = %%d/%%m/%%Y_%%H. Go to http://strftime.org/ for reference on datetime formats",
		required=False)

	parser.add_argument('-s','--sensor',
		action="store", dest="Sensor_Package", type=int,
		help="Specify the radiosonde sensor package you want to plot. Ranges from 1+",
		default=1, required=True)

	arg = parser.parse_args()
	
	if not gu.DatetimeFormat(arg.Date, "%d/%m/%Y_%H", check=True): sys.exit("Start date and time has not been formatted correctly. The format is %d/%m/%Y_%H. For example 10/11/2017_12.")
	
	Date_Start = gu.DatetimeFormat(arg.Date, "%d/%m/%Y_%H")
	
	#Download various datasets
	ESTOFEX(Date_Start, arg.Sensor_Package)
	Lightning_Wizard(Date_Start, arg.Sensor_Package)
	Sat24(Date_Start, arg.Sensor_Package)
	Surface_Pressure(Date_Start, arg.Sensor_Package)