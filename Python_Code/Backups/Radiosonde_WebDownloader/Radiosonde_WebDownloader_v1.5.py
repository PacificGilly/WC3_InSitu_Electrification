############################################################################
# Project: Electrical Pre-Conditioning of Convective Clouds,
# Title: Collecting Web Data
# Author: James Gilmore,
# Email: james.gilmore@pgr.reading.ac.uk.
# Version: 1.5
# Date: 14/12/2018
# Status: Stable
# Change: Added support for Disturbance Storm-Time (DST) Index quicklooks.
# N.B. No support will be given for downloading files from Dundee, Manunicast, UKV and UKMO Rainfall Radar.
############################################################################
from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import os, urllib, warnings, sys, argparse, requests
import time as time
from datetime import datetime, timedelta

t_inital = time.time()

with warnings.catch_warnings():
	warnings.simplefilter("ignore")

	sys.path.insert(0, '/home/users/th863480/PhD/Global_Functions')
	
	#Import Global Variables
	import PhD_Config as PhD_Global
	
	#User Processing Modules
	import Gilly_Utilities as gu
	
	#Import PG Plotter
	from PG_Quickplotter import PG_Plotter, PG_Report, _backend_changer
	
	#Import UoW Sounding module
	from PhD_DataDownloader import ULS_Bulk_Downloader
	
	#Import javascript handler
	sys.path.insert(0,'/home/users/th863480/PhD/Global_Functions/Prerequisites/modules/')
	from selenium import webdriver
	
	eps = sys.float_info.epsilon	
	
	#Initalise Selenium driver
	driver = webdriver.PhantomJS(executable_path='/home/users/th863480/PhD/Global_Functions/Prerequisites/modules/phantomjs/bin/phantomjs')

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
	t_begin = time.time()
		
	#Storage Locations
	Storage_Path 		= PhD_Global.Storage_Path_WC3
	Plots_Path_Forecast = 'Plots/Meteorology/Radiosonde_Flight_No.' + str(Radiosonde_Flight_Num).rjust(2,'0') + "_" + Date.strftime("%Y%m%d") + '/ESTOFEX/'
	Plots_Path_GFS 		= 'Plots/Meteorology/Radiosonde_Flight_No.' + str(Radiosonde_Flight_Num).rjust(2,'0') + "_" + Date.strftime("%Y%m%d") + '/GFS_ESTOFEX/'
		
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
		gu.ensure_dir(Storage_Path + Plots_Path_GFS + Product + '/')
	gu.ensure_dir(Storage_Path + Plots_Path_Forecast)
	
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
	"""Download Forecast"""
	
	#Open up webpage for analysis
	driver.get('http://www.estofex.org/cgi-bin/polygon/showforecast.cgi?list=yes&all=yes')

	#Grab information inside table
	table = driver.find_elements_by_xpath("html/body/table/tbody/tr")

	print("[INFO] Querying first 1000 rows of forecast archive...")

	#Only query first 1000 rows of table to get issue, period start and period end of each forecast.
	issued = np.zeros(300, dtype='S28')
	period_start = np.zeros(300, dtype='S21')
	period_end = np.zeros(300, dtype='S21')
	for i in xrange(2,300):
		
		index = table[i].text.index('issued')

		issued[i] = table[i].text[index:index+28]
		period_start[i] = table[i].text[index+29:index+50]
		period_end[i] = table[i].text[index+53:index+74]

	print("[INFO] Converting all data to python datetimes...")

	#Convert all arrays to python datetime (quick method)
	issued = pd.to_datetime(issued[issued != ""], format='issued:%a %d %b %Y %H:%M', errors='coerce').to_pydatetime()
	period_start = pd.to_datetime(period_start[period_start != ""], format='%a %d %b %Y %H:%M', errors='coerce').to_pydatetime()
	period_end = pd.to_datetime(period_end[period_end != ""], format='%a %d %b %Y %H:%M', errors='coerce').to_pydatetime()

	#Sort arrays
	mask = np.argsort(issued)
	issued = issued[mask]
	period_start = period_start[mask]
	period_end = period_end[mask]
	
	#Find any forecast within same period of radiosonde flight
	mask = (period_start <= Date + timedelta(hours=12)) & (period_end >= Date + timedelta(hours=12))
	
	#First, check any forecast was found
	if not np.any(mask): 
		warnings.warn("[Warning] No ESTOFEX forecast found for radiosonde flight :(")
		skip_forecast = True
	else:
		skip_forecast = False
	
	if skip_forecast is False:
	
		#Second remove all other records
		issued = issued[mask][0]
		period_start = period_start[mask][0]
		period_end = period_end[mask][0]

		#Download forecast image
		event_period_start = period_start.strftime('%Y%m%d%H')
		event_period_end = period_end.strftime('%Y%m%d%H')
		event_issue = issued.strftime('%Y%m%d%H%M')
		
		#Check to see if image exists using code 0 else move to next code
		for code in xrange(10):
			request = requests.get("http://www.estofex.org/forecasts/tempmap/" + event_period_end + "_" + event_issue + "_" + str(code) + "_stormforecast.xml.png")
			
			if request.status_code == 200:
				urllib.urlretrieve("http://www.estofex.org/forecasts/tempmap/" + event_period_end + "_" + event_issue + "_" + str(code) + "_stormforecast.xml.png", 
					Storage_Path + Plots_Path_Forecast + "ESTOFEX_StormForecast_" + event_period_start + "_" + event_period_end + "_" + str(code) + ".png")
				
				threat_level = str(code)
				
				break
				
		print("[INFO] Now downloading text forecasts...")

		#Open up webpage for analysis
		driver.get("http://www.estofex.org/cgi-bin/polygon/showforecast.cgi?text=yes&fcstfile=" + event_period_end + "_" + event_issue + "_" + threat_level + "_stormforecast.xml")

		#Grab information inside table
		text = driver.find_elements_by_xpath("html/body/p")

		#Save text to file
		with open(Storage_Path + Plots_Path_Forecast + "ESTOFEX_StormForecast_" + event_period_start + "_" + event_period_end + "_" + threat_level + ".txt", 'w') as f:
			for i in xrange(1,len(text)):
				f.write(text[i].text.replace('\n','\r\n') + '\r\n\r\n')
	
	############################################################################
	"""Download all GFS images to file"""
	
	Files_Processed = 0
	t_stage1 = time.time()
	Total_Files_to_Process = len(Product_Hour) * len(Product_Type)
	for Product in Product_Type:
		for Hour in Product_Hour:
			
			############################################################################
			"""Update user with processing status"""
			
			Files_Processed += 1
			
			#Time Statistics
			t_elapsed = time.time()-t_stage1
			t_remain  = (t_elapsed/(Files_Processed + eps))*(Total_Files_to_Process - Files_Processed + eps)
			
			#Output information to user.
			sys.stdout.write('Downloading ESTOFEX GFS images for Parameter %s and Time Step %s. Time Elapsed: %.0fs. Time Remaining: %.0fs. \r' %  (Product, Hour, t_elapsed, t_remain))
			sys.stdout.flush()
			
			############################################################################
			"""Download Images"""
			
			urllib.urlretrieve("http://www.estofex.org/modelmaps/maps/" + Date.strftime("%Y%m") + "/" + Date.strftime("%Y%m%d%H") + "/"	+ Hour + "_" + Product + ".png", Storage_Path + Plots_Path_GFS + Product + "/GFS_ESTOFEX_" + Date.strftime("%Y%m%d%H") + "_" + Product + "_" + Hour + ".png")
			
			sys.stdout.write("\033[K") #Removes last line in command line
			
	print("[INFO] ESTOFEX has been completed successfully in %.1fs" % (time.time()-t_begin))
	
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
	t_begin = time.time()
		
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
	t_stage1 = time.time()
	Total_Files_to_Process = len(Product_Hour) * len(Product_Type)
	for Product in Product_Type:
		for Hour in Product_Hour:
			
			############################################################################
			"""Update user with processing status"""
			
			Files_Processed += 1
			
			#Time Statistics
			t_elapsed = time.time()-t_stage1
			t_remain  = (t_elapsed/(Files_Processed + eps))*(Total_Files_to_Process - Files_Processed + eps)
			
			#Output information to user.
			sys.stdout.write('Downloading Lightning_Wizard GFS images for Parameter %s and Time Step %s. Time Elapsed: %.0fs. Time Remaining: %.0fs. \r' %  (Product, Hour, t_elapsed, t_remain))
			sys.stdout.flush()
			
			############################################################################
			"""Download Images"""
			
			urllib.urlretrieve("http://www.lightningwizard.com/maps/Europe/gfs_" + Product + "_eur" + Hour + ".png", Storage_Path + Plots_Path + Product + "/GFS_Lightning_Wizard_" + Date.strftime("%Y%m%d%H") + "_" + Product + "_" + Hour + ".png")
			
			sys.stdout.write("\033[K") #Removes last line in command line
			
	print("[INFO] Lightning_Wizard has been completed successfully in %.1fs" % (time.time()-t_begin))

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
	t_begin = time.time()
		
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
	
	#Change website to Sat24
	driver.get('http://www2.sat24.com/history.aspx?culture=en')
		
	Files_Processed = 0
	t_stage1 = time.time()
	Total_Files_to_Process = len(Product_Hour) * len(Product_Type)
	for Product in Product_Type:
		ir = 'True' if Product == 'ir' else 'False'
		for Hour in Product_Hour:
			
			############################################################################
			"""Update user with processing status"""
			
			Files_Processed += 1
			
			#Time Statistics
			t_elapsed = time.time()-t_stage1
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
			
			#Make sure data has been set correctly (don't know why elements don't recieve data correctly on first attempt)
			no_download = False
			Driver_Elements = (hour,day,month,year)
			Actual_Elements = (int(Hour), int(Date.strftime('%d')), int(Date.strftime('%m')), int(Date.strftime('%Y')))
			for driver_element, actual_element in zip(Driver_Elements,Actual_Elements):
				iter = 0
				error = True
				while error is True:
					if actual_element != int(driver_element.get_attribute('value')):
						driver_element.send_keys(actual_element)
						iter += 1
					elif iter == 10: #Give up. Don't bother downloading image if server not working correctly!
						no_download = True
						error = False
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
					elif iter == 10: #Give up. Don't bother downloading image if server not working correctly!
						no_download = True
						error = False
					else:
						error = False
			else:
				while error is True:
					if ir_box.is_selected() is True:
						ir_box.click()
						iter += 1
					elif iter == 10: #Give up. Don't bother downloading image if server not working correctly!
						no_download = True
						error = False
					else:
						error = False
			
			#Check no_download was not triggered
			if no_download is True: continue
			
			#Everything went okay? Submit data to sat24 servers.
			button.click()
			
			#Find image location
			img = driver.find_element_by_id(id_='ctl00_maincontent_imageSat')
			src = img.get_attribute('src')
			
			#Download processed image
			urllib.urlretrieve(src, Storage_Path + Plots_Path + Product + "/sat24_" + Product + "_" + Date.strftime('%Y%m%d') + "_" + str(int(Hour)-1).rjust(2,'0').ljust(4,'0') + "UTC.png")
			
			sys.stdout.write("\033[K") #Removes last line in command line
			
	print("[INFO] Sat24 has been completed successfully in %.1fs" % (time.time()-t_begin))	

def Surface_Analysis(Date, Radiosonde_Flight_Num):
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
	
	gu.cprint("[Info] You are running Surface_Analysis from the STABLE release", type='bold')
		
	############################################################################
	"""Prerequisites"""
	
	#Time Controls
	t_begin = time.time()
		
	#Storage Locations
	Storage_Path = PhD_Global.Storage_Path_WC3
	Plots_Path = 'Plots/Meteorology/Radiosonde_Flight_No.' + str(Radiosonde_Flight_Num).rjust(2,'0') + "_" + Date.strftime("%Y%m%d") + '/Surface_Analysis/'
	
	#Data Information
	Product_Hour = np.arange(0,24,6).astype(str)
	
	#Ensure Directory Locations
	gu.ensure_dir(Storage_Path + Plots_Path)
		
	############################################################################
	"""Error Checking"""
	
	#Check Date has been specified correctly
	if not isinstance(Date, datetime): sys.exit("[ERROR] Surface_Analysis requires the parameter Date to be in datetime format")
	
	#Convert Date to correct string format for wetter3 api
	Date = Date.strftime('%y%m%d')
	
	############################################################################
	"""Download all images to file"""
	
	Files_Processed = 0
	t_stage1 = time.time()
	Total_Files_to_Process = len(Product_Hour)
	for Hour in Product_Hour:
		
		############################################################################
		"""Update user with processing status"""
		
		Files_Processed += 1
		
		#Time Statistics
		t_elapsed = time.time()-t_stage1
		t_remain  = (t_elapsed/(Files_Processed + eps))*(Total_Files_to_Process - Files_Processed + eps)
		
		#Output information to user.
		sys.stdout.write('Downloading UKMO surface pressure charts. Time Elapsed: %.0fs. Time Remaining: %.0fs. \r' %  (t_elapsed, t_remain))
		sys.stdout.flush()
		
		############################################################################
		"""Download Images"""
		
		urllib.urlretrieve("http://www1.wetter3.de/Archiv/UKMet/" + Date + Hour.rjust(2,'0') + "_UKMet_Analyse.gif", Storage_Path + Plots_Path + Date + Hour.rjust(2,'0') + "_UKMet_Analyse.png")
		
		sys.stdout.write("\033[K") #Removes last line in command line
			
	print("[INFO] Surface_Analysis has been completed successfully in %.1fs" % (time.time()-t_begin))

def PG(Date, Radiosonde_Flight_Num, LaunchTime, Window=1):
	"""Plots the potential gradient from the RUAO +- 1 hour around launch time.
	
	Parameters
	----------
	Date : datetime object
		The datetime you want to start collecting the data from. Should have the format 
		datetime(%Y, %m, %d, %H) where %H can be either 0, 6, 12 or 18. Any other value will be rounded 
		to the nearest value defined.
	Radiosonde_Flight_Num : int
		The radiosonde flight number you want to download the images for
	LaunchTime : 
		
	Window : number
		The number of hours to plot either side of the launch time
		
	Notes
	-----
	Please wait until the next day to retrieve the analysis charts for both UKMO to process the data and wetter3 to upload them
	to their server. Fortunately, wetter3 are very reliable and quick at uploading these charts!
	
	Example
	-------
	python Radiosonde_WebDownloader.py --date 31/05/2018_0 --sonde 4
	
	"""
	
	gu.cprint("[Info] You are running PG from the STABLE release", type='bold')
		
	############################################################################
	"""Prerequisites"""
	
	#Time Controls
	t_begin = time.time()
		
	#Storage Locations
	Storage_Path = PhD_Global.Storage_Path_WC3
	Plots_Path = 'Plots/Meteorology/Radiosonde_Flight_No.' + str(Radiosonde_Flight_Num).rjust(2,'0') + "_" + Date.strftime("%Y%m%d") + '/PG/'
	
	#Ensure Directory Locations
	gu.ensure_dir(Storage_Path + Plots_Path)
	
	#Change Matplotlib back-end
	_backend_changer('webagg')
		
	############################################################################
	"""Error Checking"""
	
	#Check Date has been specified correctly
	if not isinstance(Date, datetime): sys.exit("[ERROR] Surface_Pressure requires the parameter Date to be in datetime format")
	
	#Remove time information from Date
	Date = gu.toDateOnly(Date)
	
	############################################################################
	"""Plot Potential Gradient"""
	
	#Set-up parameters
	RUAO_Location = "RUAO"
	Date_Start = datetime.combine(Date, datetime.strptime(LaunchTime, '%H:%M:%S').time()) - timedelta(hours=Window)
	Date_End = datetime.combine(Date, datetime.strptime(LaunchTime, '%H:%M:%S').time()) + timedelta(hours=Window)
	Save_Plot = True
	
	#Plot PG Data from RUAO
	PG_Plotter(RUAO_Location, Date_Start, Date_End, Save_Plot, Save_Dir=Storage_Path+Plots_Path, File_Name=None, High_Grade=False, Force_Plotting=True, Print_Progress=False)
	
	#Create Report for Charged Cloud Activity at the RUAO
	PG_Report(RUAO_Location, Date_Start, Date_End, Save_Plot, Save_Dir=Storage_Path+Plots_Path, File_Name=None, High_Grade=False, Print_Progress=False)
			
	print("[INFO] PG has been completed successfully in %.1fs" % (time.time()-t_begin))

def Lightning_Map(Date, Radiosonde_Flight_Num):
	"""Downloads all the lightning map from Blitzortung
	
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
	
	gu.cprint("[Info] You are running Lightning_Map from the STABLE release", type='bold')
		
	############################################################################
	"""Prerequisites"""
	
	#Time Controls
	t_begin = time.time()
		
	#Storage Locations
	Storage_Path = PhD_Global.Storage_Path_WC3
	Plots_Path = 'Plots/Meteorology/Radiosonde_Flight_No.' + str(Radiosonde_Flight_Num).rjust(2,'0') + "_" + Date.strftime("%Y%m%d") + '/Lightning_Maps/'
	
	#Ensure Directory Locations
	gu.ensure_dir(Storage_Path + Plots_Path)
	
	############################################################################
	"""Download image"""
	
	urllib.urlretrieve("https://images.lightningmaps.org/blitzortung/europe/index.php?map=uk&date=" + Date.strftime('%Y%m%d'), Storage_Path + Plots_Path + "/Lightning_Blitzortung_" + Date.strftime('%Y%m%d') + "_0000+24h.png")
			
	print("[INFO] Lightning_Map has been completed successfully in %.1fs" % (time.time()-t_begin))

def Soundings(Date, Radiosonde_Flight_Num):
	"""Downloads all the upper level sounding data from University of Wyoming
	
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
	
	gu.cprint("[Info] You are running Soundings from the STABLE release", type='bold')
		
	############################################################################
	"""Prerequisites"""
	
	#Time Controls
	t_begin = time.time()
		
	#Storage Locations
	Storage_Path = PhD_Global.Storage_Path_WC3
	Plots_Path = 'Plots/Meteorology/Radiosonde_Flight_No.' + str(Radiosonde_Flight_Num).rjust(2,'0') + "_" + Date.strftime("%Y%m%d") + '/Tephigram/'
	
	#Ensure Directory Locations
	gu.ensure_dir(Storage_Path + Plots_Path)
	
	############################################################################
	"""Download images and data"""
	
	Dates = [Date + timedelta(hours=i) for i in xrange(0,24,3)]
	
	ULS_Bulk_Downloader(Dates, save_location=Storage_Path+Plots_Path, station='03808', data=True, indicies=True, image=True, show_output=False)
	ULS_Bulk_Downloader(Dates, save_location=Storage_Path+Plots_Path, station='03743', data=True, indicies=True, image=True, show_output=False)
	
	############################################################################
	
	print("[INFO] Soundings has been completed successfully in %.1fs" % (time.time()-t_begin))

def Geomagnetic(Date, Radiosonde_Flight_Num):
	"""Downloads all the Disturbance Storm-Time (DST) Index quicklooks from the World Data Center 
	for Geomagnetism, Kyoto
	
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
	
	gu.cprint("[Info] You are running Soundings from the STABLE release", type='bold')
		
	############################################################################
	"""Prerequisites"""
	
	#Time Controls
	t_begin = time.time()
		
	#Storage Locations
	Storage_Path = PhD_Global.Storage_Path_WC3
	Plots_Path = 'Plots/Meteorology/Radiosonde_Flight_No.' + str(Radiosonde_Flight_Num).rjust(2,'0') + "_" + Date.strftime("%Y%m%d") + '/Geomagnetic/'
	
	#Ensure Directory Locations
	gu.ensure_dir(Storage_Path + Plots_Path)
		
	############################################################################
	"""Download image"""
	
	urllib.urlretrieve('http://wdc.kugi.kyoto-u.ac.jp/dst_realtime/' + Date.strftime('%Y%m') + '/dst' + Date.strftime('%y%m') + '.png', 
		Storage_Path + Plots_Path + "/DST_Index_" + Date.strftime('%Y%m') + ".png")

	############################################################################
	
	print("[INFO] Geomagnetic has been completed successfully in %.1fs" % (time.time()-t_begin))
	
if __name__ == "__main__":
	"""Launch the Radiosonde_WebDownloader.py from the command line. This python script gives command line
	options which can be found using Radiosonde_WebDownloader.py --help. An example input for a radiosonde
	flight is given as,
	
	>>> python Radiosonde_WebDownloader.py --date 31/05/2018_0 --sensor 4
	
	72 hours at 3 hour intervals GFS plots are downloaded from the model PhD_Globaltiation date specified
	
	"""
	
	gu.cprint("Welcome to the Radiosonde Web Downloader. Downloading several meterological products from various sources!", type='bold')
		
	#Only used as part of James' PhD radiosonde launches
	Launch_Times = {
		'1' : None, 
		'2': None,
		'3' : None,
		'4' : None, 
		'5' : '15:23:00', 
		'6' : None, 
		'7' : None, 
		'8' : None, 
		'9' : '16:20:00', 
		'10' : '09:22:00'}
		
	############################################################################
	"""Process Command Line Arguments"""
	
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
		help="The date and time you want to start collecting GFS data from. Format = %%d/%%m/%%Y_%%H.\
Go to http://strftime.org/ for reference on datetime formats",
		required=False)

	parser.add_argument('-s','--sensor',
		action="store", dest="Sensor_Package", type=int,
		help="Specify the radiosonde sensor package you want to plot. Ranges from 1+",
		default=1, required=True)
	
	parser.add_argument('-l','--launch',
		action="store", dest="Launch_Time",
		help="Specify the launch time of the radiosonde. Format = %%H:%%M:%%S.\
Go to http://strftime.org/ for reference on datetime formats",
		required=False)
		
	arg = parser.parse_args()
	
	############################################################################
	"""Conditional Formatting Command Line Arguments"""

	if not gu.DatetimeFormat(arg.Date, "%d/%m/%Y_%H", check=True): raise ValueError("Start date and time has not been formatted correctly. The format is %d/%m/%Y_%H. For example 10/11/2017_12.")
	if arg.Launch_Time is None: 
		try:
			arg.Launch_Time = Launch_Times[str(arg.Sensor_Package)]
		except:
			pass #No PG will be downloaded
	else:
		if not gu.DatetimeFormat(arg.Launch_Time, "%H:%M:%S", check=True): raise ValueError("Launch time has not been formatted correctly. The format is %H:%M:%S. For example 12:45:00")
	
	Date_Start = gu.DatetimeFormat(arg.Date, "%d/%m/%Y_%H")
	
	############################################################################
	"""Download various datasets"""
	
	ESTOFEX(Date_Start, arg.Sensor_Package)
	Lightning_Wizard(Date_Start, arg.Sensor_Package)
	Sat24(Date_Start, arg.Sensor_Package)
	Surface_Analysis(Date_Start, arg.Sensor_Package)
	if arg.Launch_Time is not None: PG(Date_Start, arg.Sensor_Package, arg.Launch_Time)
	Lightning_Map(Date_Start, arg.Sensor_Package)
	Soundings(Date_Start, arg.Sensor_Package)
	Geomagnetic(Date_Start, arg.Sensor_Package)
	
	############################################################################
	
	gu.cprint("[Radiosonde_WebDownloader]: All Tasks Completed, Time Taken (s): %.0f" % (time.time()-t_inital), type='okgreen')
