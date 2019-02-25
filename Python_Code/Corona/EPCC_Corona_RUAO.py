# -*- encoding: utf-8 -*-
############################################################################
# Project: Electrical Pre-Conditioning of Convective Clouds,
# Title: Compare the Corona sensor against measured parameters at the RUAO
# Author: James Gilmore,
# Email: james.gilmore@pgr.reading.ac.uk.
# Version: 1.0
# Date: 22/05/17
# Status: Stable
# Change: Added in Corona Functionality
############################################################################
"""Initialising the python script"""

from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import statsmodels.api as sm
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import date2num, num2date, MinuteLocator, HourLocator, DayLocator, DateFormatter
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, MaxNLocator
import csv, os, glob, sys, gzip, warnings
import time as systime
from scipy import stats
from scipy.optimize import curve_fit
eps = sys.float_info.epsilon

sys.path.insert(0, '../../../Global_Functions')
from Gilly_Utilities import running_mean

with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	
	sys.path.insert(0, '../../../Global_Functions')
	from Data_Importer import EPCC_Importer
	from Gilly_Utilities import Excel_to_Python_Date, toHourFraction, list_func, nan_removal_dd, unlink_wrap
	from externals import EPCC_Histogram_Plots
	EPCC_Plots = EPCC_Histogram_Plots()
	from Buffer_Tips import Buffer_Tips_TopDown_v2
	
	sys.path.insert(0, '../Field_Mill')
	from EPCC_PGRR_Ensemble import PGRR_Ensembler

def CoronavsRUAO():
	"""Compare the Corona sensor against measured parameters at the RUAO.
	
	Parameters
	----------
	Time
	Temperature
	Humidity
	Pressure
	Potential Gradient
	
	Statistical Tests
	-----------------
	Linear Regression
		p-value
		r squared value
		AICc
		pearsons correlation
	Distribution Tests
		Anderson-Darling (mean)
		Kolmogorov-Smirnov (mean)
		Kuiper (mean)
		Mann-Whitney U Test (median)
	Variance Tests
		Conover
		Seigel-Tukey
		Quantile 
		
	"""
	
	os.system('cls' if os.name=='nt' else 'clear')
	print("[INFO] You are running Corona vs. RUAO from the DEV release")
	
	############################################################################
	"""Intialise module"""
	
	#File Locations
	RUAO_File = sorted(glob.glob('/net/vina1s/vol/data1_s/meteorology_2/RUAOData/METFiDAS-3a/Level0/2017/*-*-*-Smp1Sec.csv'))
	Storage_Path = '/glusterfs/phd/users/th863480/WC3_Chilbolton_Corona/'
	Plots_Path = 'Plots/Corona_RUAO/'
	Data_Path = 'Processed_Data/Corona_RUAO/'
	
	#Time Ranges
	Sensor_TimeRange = np.array([[datetime(2017,5,22,9,19,0), datetime(2017,9,27,13,00,0)],
		[datetime(2017,9,27,13,20,0), datetime(2019,9,27,13,20,0)]], dtype=object)
		
	#Conditionals
	Step_Detection = True
	Step_Detection_All = False
	Ensembles = True
	All_Save = True
	Daily_Format = True
	Linear_Regression = True
	
	#Other
	EPCC_Data = EPCC_Importer()
	flatten = lambda l: [item for sublist in l for item in sublist]
	Step_Num = 2
	Sensor_Num = np.arange(1, len(Sensor_TimeRange)+1)

	print("Intialised Module Complete")
	
	for x in Sensor_Num:
	
		############################################################################
		"""Represent the dates of each file to be consistent of the form YYYY"M"DDD"""
		
		if x == 1: continue
		
		RUAO_Date = np.zeros(len(RUAO_File), dtype=object)
		for i in xrange(len(RUAO_File)): RUAO_Date[i] = datetime(int(os.path.basename(RUAO_File[i])[0:4]), int(os.path.basename(RUAO_File[i])[5:7]), int(os.path.basename(RUAO_File[i])[8:10]))
		
		start = np.where(np.logical_and(RUAO_Date >= Sensor_TimeRange[x-1,0]-timedelta(days=1), RUAO_Date <= Sensor_TimeRange[x-1,1]))[0]

		Data_All = zip(np.zeros(6))
		Data_FW = zip(np.zeros(5))
		Corona_Step_Date_All = np.array([])
		Corona_Step_Time_All = np.array([])
		Corona_Step_Mag_All = np.array([])
		Corona_Step_Mode_All = np.array([])
		Corona_Step_Stats_All = np.zeros([len(start),8], dtype=object)
		t_begin = systime.time()
		print("Collecting all RUAO Data Files for Corona Sensor Num. %s..." % x)
		for i in xrange(len(start)):

			#Time Statistics
			t_elapsed = systime.time()-t_begin
			t_remain  = (t_elapsed/(i + eps))*len(start)
			
			#Output information to user.
			sys.stdout.write('Collecting data from %s. Time Elapsed: %.0f s. Time Remaining: %.0f s \r' %  (RUAO_Date[start[i]].date(), t_elapsed, t_remain))
			sys.stdout.flush()
		
			############################################################################
			"""Import Data"""

			Time, RH, T, Cor, PG, P, RR = EPCC_Data.RUAO_Calibrate(RUAO_File[start[i]], Col=(0,21,27,44,45,-1,-5), unpack=True)
			
			bob = 0
			
			############################################################################
			"""Calibrate: H and P are already in the correct units"""
					
			if i == 0:
				Cor = np.array(Cor[Time > Sensor_TimeRange[x-1,0]], dtype=float)
				P = np.array(P[Time > Sensor_TimeRange[x-1,0]], dtype=float)
				RH = np.array(RH[Time > Sensor_TimeRange[x-1,0]], dtype=float)*100
				T = (np.array(T[Time > Sensor_TimeRange[x-1,0]], dtype=float)-0.0664051)/0.101
				PG = (np.array(PG[Time > Sensor_TimeRange[x-1,0]], dtype=float)-0.00903)/0.00463
				RR = np.array(RR[Time > Sensor_TimeRange[x-1,0]], dtype=int)
				Time = Time[Time > Sensor_TimeRange[x-1,0]]
				
				Data_All[0] = np.append(Data_All[0], Time)[1:]
				Data_All[1] = np.append(Data_All[1], T)[1:]
				Data_All[2] = np.append(Data_All[2], P)[1:]
				Data_All[3] = np.append(Data_All[3], RH)[1:]
				Data_All[4] = np.append(Data_All[4], PG)[1:]
				Data_All[5] = np.append(Data_All[5], Cor)[1:]

			elif i == len(start)-1:
				Cor = np.array(Cor[Time < Sensor_TimeRange[x-1,1]], dtype=float)
				P = np.array(P[Time < Sensor_TimeRange[x-1,1]], dtype=float)
				RH = np.array(RH[Time < Sensor_TimeRange[x-1,1]], dtype=float)*100
				T = (np.array(T[Time < Sensor_TimeRange[x-1,1]], dtype=float)-0.0664051)/0.101
				PG = (np.array(PG[Time < Sensor_TimeRange[x-1,1]], dtype=float)-0.00903)/0.00463
				RR = np.array(RR[Time < Sensor_TimeRange[x-1,1]], dtype=int)
				Time = Time[Time < Sensor_TimeRange[x-1,1]]
				
				Data_All[0] = np.append(Data_All[0], Time)
				Data_All[1] = np.append(Data_All[1], T)
				Data_All[2] = np.append(Data_All[2], P)
				Data_All[3] = np.append(Data_All[3], RH)
				Data_All[4] = np.append(Data_All[4], PG)
				Data_All[5] = np.append(Data_All[5], Cor)
			
			else:
				Cor = np.array(Cor, dtype=float)
				P = np.array(P, dtype=float)
				RH = np.array(RH, dtype=float)*100
				T = (np.array(T, dtype=float)-0.0664051)/0.101
				PG = (np.array(PG, dtype=float)-0.00903)/0.00463
				
				Data_All[0] = np.append(Data_All[0], Time)
				Data_All[1] = np.append(Data_All[1], T)
				Data_All[2] = np.append(Data_All[2], P)
				Data_All[3] = np.append(Data_All[3], RH)
				Data_All[4] = np.append(Data_All[4], PG)
				Data_All[5] = np.append(Data_All[5], Cor)
			
			#Capture all fair weather days

			if np.sum(RR, dtype=int) == 0: 
				Data_FW[0] = np.append(Data_FW[0], Time)[1:]
				Data_FW[1] = np.append(Data_FW[1], Cor)[1:]
				Data_FW[2] = np.append(Data_FW[2], T)[1:]
				Data_FW[3] = np.append(Data_FW[3], RH)[1:]
				Data_FW[4] = np.append(Data_FW[4], P)[1:]

			if Step_Detection == True:
			
				############################################################################
				"""Run Top-Down Step Algorithm to determine PG variations"""
				
				labels = np.array([])
				modes = [(0,0,1),(0,1,0),(1,0,0)]
				Corona_Step_Time_Ind = zip(np.zeros([len(modes)]))	#Time of step
				Corona_Step_Mag_Ind = zip(np.zeros([len(modes)]))	#Magnitude of step
				Corona_Step_Mode_Ind = zip(np.zeros([len(modes)]))	#Mode used by top down (e.g. forward in time = (1,0,0))

				Time_HF = np.zeros(len(Time))
				for j in xrange(len(Time)): Time_HF[j] = toHourFraction(Time[j])

				
				for j in xrange(len(modes)):
					labels = np.append(labels, str(modes[j]))
					Corona_Step_Time_Ind[j], Corona_Step_Mag_Ind[j] = Buffer_Tips_TopDown_v2(Cor, Time_HF, thres=0.001, abs=True, search=modes[j], output=False)
					Corona_Step_Mode_Ind[j] = list_func(Corona_Step_Mode_Ind[j]).rjust(len(Corona_Step_Time_Ind[j]), j, int)
					Corona_Step_Mode_Ind[j][-1] = j
					
				Corona_Step_Time = np.array(flatten(Corona_Step_Time_Ind), dtype=float)
				Corona_Step_Mag = np.array(flatten(Corona_Step_Mag_Ind), dtype=float)
				Corona_Step_Mode = np.array(flatten(Corona_Step_Mode_Ind), dtype=int)
				
				Corona_Step_Mag = Corona_Step_Mag[np.argsort(Corona_Step_Time)]
				Corona_Step_Mode = Corona_Step_Mode[np.argsort(Corona_Step_Time)]
				Corona_Step_Time = Corona_Step_Time[np.argsort(Corona_Step_Time)]
				
				#Remove nan's
				Corona_Step_Time, Corona_Step_Mag, Corona_Step_Mode = nan_removal_dd(np.array([Corona_Step_Time, Corona_Step_Mag, Corona_Step_Mode]), unpack=True)	
				
				############################################################################
				"""Export Daily Data to File"""
				
				#Sort duplicates by using a binary method on the mode variable (i.e. 2^index)
				Corona_Step_Time_Unique = np.array([], dtype=float)
				Corona_Step_Mag_Unique = np.array([], dtype=float)
				Corona_Step_Mode_Unique = np.array([], dtype=int)
				for j in xrange(len(Corona_Step_Time)-1):
					if Corona_Step_Time[j+1] == Corona_Step_Time[j]:
						if j != 0:
							Corona_Step_Mode_Unique[-1] += 2**Corona_Step_Mode[j]
					else:
						Corona_Step_Time_Unique = np.append(Corona_Step_Time_Unique, Corona_Step_Time[j])
						Corona_Step_Mag_Unique = np.append(Corona_Step_Mag_Unique, Corona_Step_Mag[j])
						Corona_Step_Mode_Unique = np.append(Corona_Step_Mode_Unique, 2**Corona_Step_Mode[j])

				with open(Storage_Path + Data_Path + 'Step_Detection/Daily/Step_Detection_Data_' + RUAO_Date[start[i]].strftime('%Y%m%d') + "_SensorNum_" + str(x) + '.csv', "wb") as output:
					writer = csv.writer(output, lineterminator='\n')
					writer.writerows(zip(Corona_Step_Time_Unique, Corona_Step_Mag_Unique, Corona_Step_Mode_Unique))
				
				# #If there is downtime or skipped data present then we add a np.nan. This will show a gap in the output plots rather than a joined line.
				# Time_Skips = np.arange(len(Time_HF)-1)[(np.roll(Time_HF,-1)-Time_HF)[:-1] > 2/3600]
				# for dd in Time_Skips:
					# Time_HF = np.insert(Time_HF, dd, np.nan)
					# Cor = np.insert(Cor, dd, np.nan)
				
				############################################################################
				"""Plot the significant PG as a 24-hour time series"""

				modes = [1,2,3,4,5,6,7]
				modes_label = ['BTCS','CTCS','BTCS + CTCS', 'FTCS', 'FTCS + BTCS', 'FTCS + CTCS', 'TCS']

				plt.clf()
				fig, ax1 = plt.subplots()

				#Plot Top-Down Output
				colors = ['blue', 'green', 'orange', 'red', 'purple', 'yellow', 'black']
				
				for j in xrange(7):
					try:
						ax1.scatter(Corona_Step_Time_Unique[Corona_Step_Mode_Unique == modes[j]], Corona_Step_Mag_Unique[Corona_Step_Mode_Unique == modes[j]], s=0.1, color=colors[j], label=str(modes_label[j]))
					except:
						continue

				ax1.grid(which='major',axis='both',c='grey')
				axis_temp = ax1.axis()
				ax1.axis([0,24,axis_temp[2],axis_temp[3]])
				ax1.set_xlabel("Time (UTC)")
				ax1.set_ylabel("$dCorona/dt$ $(V$ $s^{-1})$")
				
				#Overlay subtle Corona signal (light grey)
				ax2 = ax1.twinx()
				
				for slc in unlink_wrap(Time_HF[:-1], [0,2/3600]):
					ax2.plot(Time_HF[:-1][slc], Cor[:-1][slc], color='grey', lw=0.5)
					
				ax2.set_ylabel('$Corona$ $(V)$')
				axis_temp = ax2.axis()
				ax2.axis([0,24,axis_temp[2],axis_temp[3]])		

				pg = plt.gca()
				x0, x1 = pg.get_xlim()
				y0, y1 = pg.get_ylim()
				pg.set_aspect(int(np.abs((x1-x0)/(4*(y1-y0)))))
				
				pg.xaxis.set_major_locator(MultipleLocator(4))
				
				fig=plt.gcf()
				fig.set_size_inches(11.7, 11.7/4)

				ax1.legend(bbox_to_anchor=(0., 1., 1., .02), loc=3,
				   ncol=7, mode="expand", borderaxespad=0., prop={'size':12})
				
				ax1.set_title("Sample Test for Significance in dCorona/dt ("+RUAO_Date[start[i]].strftime("%Y/%m/%d")+")" + " (Sensor Num. " + str(x) + ")", y=1.2)
				
				plt.savefig(Storage_Path + Plots_Path + 'Step_Detection/Daily/Top-Down_Detection_' + RUAO_Date[start[i]].strftime("%Y%m%d") + "_SensorNum_" + str(x), dpi=300, bbox_inches='tight',pad_inches=0.1)
				plt.close()
				
				if Corona_Step_Time.shape[0] != 0:
					Corona_Step_Datetime = np.zeros(len(Corona_Step_Time), dtype=object)
					for j in xrange(len(Corona_Step_Time)): Corona_Step_Datetime[j] = RUAO_Date[start[i]] + timedelta(hours=Corona_Step_Time[j])
				else:
					Corona_Step_Datetime = np.array([RUAO_Date[start[i]]], dtype=object)

				#Save to master array
				if Corona_Step_Time.shape[0] == 0:
					Corona_Step_Time = np.array([0])
					Corona_Step_Mag = np.array([0])
					Corona_Step_Mode = np.array([0])
				Corona_Step_Date_All = np.append(Corona_Step_Date_All, Corona_Step_Datetime)
				Corona_Step_Time_All = np.append(Corona_Step_Time_All, Corona_Step_Time)
				Corona_Step_Mag_All = np.append(Corona_Step_Mag_All, Corona_Step_Mag)
				Corona_Step_Mode_All = np.append(Corona_Step_Mode_All, Corona_Step_Mode)

				Corona_Step_Stats = np.array([Corona_Step_Datetime[np.where(Corona_Step_Mag == np.max(Corona_Step_Mag))[0][0]].date(), 
					Corona_Step_Time[np.where(Corona_Step_Mag == np.max(Corona_Step_Mag))[0][0]], 
					np.max(Corona_Step_Mag),
					Corona_Step_Time[np.where(Corona_Step_Mag == np.min(Corona_Step_Mag))[0][0]],
					np.min(Corona_Step_Mag),
					np.median(Corona_Step_Mag),
					np.mean(Corona_Step_Mag),
					np.std(Corona_Step_Mag)], dtype=object)
			
				Corona_Step_Stats_All[i] = Corona_Step_Stats
			
		print("\n[STEP 1]: All Data Collected which took %.2f secs" % (systime.time()-t_begin)) if Step_Detection is False else print("\n[STEP 1]: All Data Collected and Solved Step Detection which took %.2f secs" % (systime.time()-t_begin))
		
		if All_Save == True:
			############################################################################
			"""Output into One File Format"""
			
			t1 = systime.time()
			
			with gzip.open(Storage_Path + Data_Path + "All/CSEA_Corona_Data_RUAO_All_SensorNum_" + str(x) + ".csv.gz", "wb") as output:
				writer = csv.writer(output, lineterminator='\n')
				writer.writerows(zip(*Data_All))
			
			with gzip.open(Storage_Path + Data_Path + "All/CSEA_Corona_Data_RUAO_FW_SensorNum_" + str(x) + ".csv.gz", "wb") as output:
				writer = csv.writer(output, lineterminator='\n')
				writer.writerows(zip(*Data_FW))
			
			print("[STEP " + str(Step_Num) + "]: Saved All Data to File which took %.2f secs" % (systime.time()-t1))

		if Daily_Format == True:
			############################################################################
			"""Output into Daily File Format"""
			
			t2 = systime.time()
			
			#Convert Times to Dates
			Date_All = np.zeros_like(Data_All[0])
			for i in xrange(len(Data_All[0])): Date_All[i] = Data_All[0][i].replace(hour=0, minute=0, second=0, microsecond=0)
			
			#Find Unique Dates
			u = np.unique(Date_All)
			
			Data_Daily = zip(np.zeros(6))
			t_begin = systime.time()
			for i in xrange(len(u)):
				
				#Time Statistics
				t_elapsed = systime.time()-t_begin
				t_remain  = (t_elapsed/(i + eps))*len(u)
				
				#Output information to user.
				sys.stdout.write('Plotting processed data from %s. Time Elapsed: %.0f s. Time Remaining: %.0f s \r' %  (u[i].strftime('%Y%m%d'), t_elapsed, t_remain))
				sys.stdout.flush()
				
				"""Extract the data to file"""
				with gzip.open(Storage_Path + Data_Path + "Daily/CSEA_Corona_Data_RUAO_Daily_" + u[i].strftime('%Y%m%d') + "_SensorNum_" + str(x) + ".csv.gz", "wb") as output:
					writer = csv.writer(output, lineterminator='\n')
					writer.writerows(zip(Data_All[0][Date_All == u[i]], Data_All[-1][Date_All == u[i]], Data_All[1][Date_All == u[i]], Data_All[3][Date_All == u[i]], Data_All[2][Date_All == u[i]], Data_All[4][Date_All == u[i]]))
			
				"""Organise the data ready for plotting"""
				for j in xrange(len(Data_All)):
					Data_Daily[j] = Data_All[j][Date_All == u[i]] #Time
				
				"""Quality Control Data"""
				for j in xrange(len(Data_Daily)):
					Data_Daily[j] = Data_Daily[j][Data_Daily[-1]>-9999]
				
				"""Convert Time_Daily to Hour Frac"""
				Time_Daily_HF = np.zeros(len(Data_Daily[0]))
				for j in xrange(len(Data_Daily[0])):
					Time_Daily_HF[j] = toHourFraction(Data_Daily[0][j])
				
				"""Plot the data and save to file"""
				plt.clf()
				for slc in unlink_wrap(Time_Daily_HF[:-1], [0,2/3600]):
					plt.plot(Time_Daily_HF[slc], Data_Daily[-1][slc], lw=0.5, c='gray')

				plt.ylabel('Corona (V)')
				plt.xlabel('Time (Hours)')
				x1,x2,y1,y2 = plt.axis()
				plt.axis((0,24,y1,y2))
				plt.title('Corona Timeseries at RUAO on ' + u[i].strftime('%Y/%m/%d') + ' (Sensor Num. ' + str(x) + ')')
				plt.grid(which='major',axis='both',c='grey')
				
				pg = plt.gca()
				x0, x1 = pg.get_xlim()
				y0, y1 = pg.get_ylim()
				pg.set_aspect(int(np.abs((x1-x0)/(4*(y1-y0)))))

				pg.xaxis.set_major_locator(MultipleLocator(4))
				
				fig=plt.gcf()
				fig.set_size_inches(11.7, 11.7/4)
				
				out_file = 'Corona_Timeseries_At_Chilbolton_' + u[i].strftime('%Y%m%d') + "_SensorNum_" + str(x)
				plt.savefig(Storage_Path + Plots_Path + 'Daily/' + out_file + '.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
			
			Step_Num += 1
				
			print("[STEP " + str(Step_Num) + "]: Daily Plotting and Saving Completed which took %.2f secs" % (systime.time()-t2))
		
		if Linear_Regression is True:
			############################################################################
			"""Calculate Linear Regression"""
			
			t3 = systime.time()
			
			Linear_Data = np.zeros([5,5])
			Linear_Data[0] = stats.linregress(date2num(Data_All[0])[~np.isnan(Data_All[-1])],Data_All[-1][~np.isnan(Data_All[-1])])

			for i in xrange(1,len(Data_All)-1): Linear_Data[i] = stats.linregress(Data_All[i][(~np.isnan(Data_All[-1])) & (~np.isnan(Data_All[i]))], Data_All[-1][(~np.isnan(Data_All[-1])) & (~np.isnan(Data_All[i]))])
			
			Step_Num += 1
			
			print("[STEP " + str(Step_Num) + "]: Solved Linear Regression which took %.2f secs" % (systime.time()-t3))
		
			############################################################################
			"""Plot Each Linear Regression"""
			
			t4 = systime.time()
			
			xlab = ['Time $(Days)$', 'Temperature $(^\circ C)$', 'Pressure $(hPa)$', 'Relative Humidity (%)', 'Potential Gradient $(V$ $m^{-1})$']
			out = ['Time', 'Temperature', 'Pressure', 'RH', 'PG']
			
			plt.clf()
			for i in xrange(len(xlab)):
				plt.clf()
				fig, ax3 = plt.subplots()
				if i == 0:
					#Delete every 5th element in each series as its too big to plot
					x_data = Data_All[i][~np.isnan(Data_All[-1])]
					y1_data = Data_All[-1][~np.isnan(Data_All[-1])]
					y2_data = date2num(Data_All[i])[~np.isnan(Data_All[-1])]
					x_data = np.delete(x_data, np.arange(0, x_data.size, 5))
					y1_data = np.delete(y1_data, np.arange(0, y1_data.size, 5))
					y2_data = np.delete(y2_data, np.arange(0, y2_data.size, 5))
					
					#Plot
					ax3.plot(x_data, y1_data, lw=0.5, c='gray', alpha=0.8)
					ax3.plot(x_data, Linear_Data[i,0]*y2_data+Linear_Data[i,1], c='red', lw=0.5)
					ax3.axis((x_data.min(), x_data.max(), y1_data.min(), y1_data.max()))
				else:
					#Delete every 5th element in each series as its too big to plot
					x_data = Data_All[i][(~np.isnan(Data_All[-1])) & (~np.isnan(Data_All[i]))]
					y1_data = Data_All[-1][(~np.isnan(Data_All[-1])) & (~np.isnan(Data_All[i]))]
					x_data = np.delete(x_data, np.arange(0, x_data.size, 5))
					y1_data = np.delete(y1_data, np.arange(0, y1_data.size, 5))
					
				#Plot
					ax3.scatter(x_data,y1_data,s=1,c='gray',alpha=0.8,edgecolors=None,lw=0)
					ax3.plot(x_data,Linear_Data[i,0]*x_data+Linear_Data[i,1],c='red',lw=0.5)
					ax3.axis((x_data.min(),x_data.max(),y1_data.min(),y1_data.max()))
				
				ax3.set_ylabel('Corona (V)')
				ax3.set_xlabel(xlab[i])
				ax3.set_title('Corona vs ' + out[i] + ' Linear Regression at RUAO' + " (Sensor Num. " + str(x) + ")")
				ax3.grid(which='major',axis='both',c='grey')
				
				#ax = plt.gca()
				x0, x1 = ax3.get_xlim()
				y0, y1 = ax3.get_ylim()
				print("LIMITS", ax3.get_xlim(), ax3.get_ylim(), np.abs((x1-x0)/((y1-y0))))
				ax3.set_aspect(np.abs((x1-x0)/((y1-y0))))
			
				if i == 0:
					TimeLength = (Data_All[i][-1]-Data_All[i][0]).total_seconds() + 1
					if TimeLength/86400 <= 2:
						"""Short Range: ~Single Day"""
						myFmt = DateFormatter('%H:%M')
						ax3.xaxis.set_major_formatter(myFmt)
						ax3.xaxis.set_major_locator(MinuteLocator(interval=int(round((TimeLength/60)/6))))
						ax3.set_xlabel('Time (local) from '+str(Data_All[i][0].strftime('%d/%m/%Y'))+'')
					elif TimeLength/86400 <= 7:
						"""Medium Range: ~Multiple Days"""
						myFmt = DateFormatter('%Y-%m-%d %H:%M') #Use this when plotting multiple days (e.g. monthly summary)
						ax3.xaxis.set_major_formatter(myFmt)
						ax3.xaxis.set_major_locator(HourLocator(interval=int(round((TimeLength/3600)/6))))
						ax3.set_xlabel('Time (local) from '+str(Data_All[i][0].strftime('%H:%M %d/%m/%Y'))+'')
					else:
						"""Long Range: ~Months"""
						myFmt = DateFormatter('%Y-%m-%d') #Use this when plotting multiple days (e.g. monthly summary)
						ax3.xaxis.set_major_formatter(myFmt)
						ax3.xaxis.set_major_locator(DayLocator(interval=int(round((TimeLength/86400)/6))))
						ax3.set_xlabel('Time (UTC)')
						
					plt.xticks(rotation='vertical')
				
				ax3.annotate('$Counts$: %.0d \n$P-Value$: %.4f \n$R^2$: %.4f \n$SE$: %.4f' % (len(Data_All[i]), Linear_Data[i,3], Linear_Data[i,2], Linear_Data[i,4]), xy=(0, 1), xycoords='axes fraction', xytext=(20, -20), textcoords='offset pixels', horizontalalignment='left', verticalalignment='top', fontsize=10)
						
				out_file = 'Corona_' + out[i] + '_Regression_At_RUAO_All_Data' + "_SensorNum_" + str(x)
				plt.savefig(Storage_Path + Plots_Path + 'Linear_Regression/' + out_file + '.png',bbox_inches='tight',pad_inches=0.1, dpi=300)
				
			Step_Num += 1	
			print("[STEP " + str(Step_Num) + "]: Plotted all Linear Regressions which took %.2f secs" % (systime.time()-t4))
		
			if Ensembles is True:	
				############################################################################
				"""Create Ensemble"""

				for i in xrange(len(xlab)):
				
					t4 = systime.time()
					
					if i == 0:
						Ensemble_Dynamic = PGRR_Ensembler(100, 1, 4, method=None, case=None, PGRR_Loc=None, pertcount=100, dataimport=False, year=None, month=None, time=None, rainrate=date2num(Data_All[i]), pg=Data_All[-1])
						Ensemble_Virtual = PGRR_Ensembler(20, 1, 5, method=None, case=None, PGRR_Loc=None, pertcount=500, dataimport=False, year=None, month=None, time=None, rainrate=date2num(Data_All[i]), pg=Data_All[-1])
					else:
						Ensemble_Dynamic = PGRR_Ensembler(100, 1, 4, method=None, case=None, PGRR_Loc=None, pertcount=100, dataimport=False, year=None, month=None, time=None, rainrate=Data_All[i], pg=Data_All[-1])
						Ensemble_Virtual = PGRR_Ensembler(20, 1, 5, method=None, case=None, PGRR_Loc=None, pertcount=500, dataimport=False, year=None, month=None, time=None, rainrate=Data_All[i], pg=Data_All[-1])
				
					"""Ensemble Plot"""
					EPCC_Plots.Ensemble_Plotter(Ensemble_Dynamic, 'Average relationship between Corona and ' + out[i] + ': Dynamic', xlab[i], 'Corona $(V)$', Storage_Path+Plots_Path+'Ensemble/Chilbolton_Corona_' + out[i] + '_Ensemble_Dynamic_CaseStudy_All_SensorNum_' + str(x) + '.png')
					EPCC_Plots.Ensemble_Plotter(Ensemble_Virtual, 'Average relationship between Corona and ' + out[i] + ': Virtual', xlab[i], 'Corona $(V)$', Storage_Path+Plots_Path+'Ensemble/Chilbolton_Corona_' + out[i] + '_Ensemble_Virtual_CaseStudy_All_SensorNum_' + str(x) + '.png')
				
					"""Ensemble Linear Regression"""
					Linear_Data_Ensemble = np.zeros([2,5])
					Linear_Data_Ensemble[0] = stats.linregress(Ensemble_Dynamic[:,0], Ensemble_Dynamic[:,1])
					Linear_Data_Ensemble[1] = stats.linregress(Ensemble_Virtual[:,0], Ensemble_Virtual[:,1])

					Ensemble_Name = ['Dynamic', 'Virtual']
					
					for j in xrange(2):
						Ensemble_Data = Ensemble_Dynamic if j == 0 else Ensemble_Virtual

						plt.clf()
						if i == 0:
							plt.plot(num2date(Ensemble_Data[:,0]),Ensemble_Data[:,1],lw=0.5,c='gray',alpha=0.8)
							plt.plot(num2date(Ensemble_Data[:,0]),Linear_Data_Ensemble[i,0]*Ensemble_Data[:,0]+Linear_Data_Ensemble[i,1],c='red',lw=0.5)
						else:
							plt.scatter(Ensemble_Data[:,0][~np.isnan(Ensemble_Data[:,1])],Ensemble_Data[:,1][~np.isnan(Ensemble_Data[:,1])],s=4,c='gray',alpha=1,edgecolors=None,lw=0)
							plt.plot(Ensemble_Data[:,0][~np.isnan(Ensemble_Data[:,1])],Linear_Data_Ensemble[j,0]*Ensemble_Data[:,0][~np.isnan(Ensemble_Data[:,1])]+Linear_Data_Ensemble[j,1],c='red',lw=0.5)
							
						plt.ylabel('Corona (V)')
						plt.xlabel(xlab[i])
						plt.title('Corona vs ' + out[i] + ' Linear Ensemble Regression at RUAO' + "(Sensor Num. " + str(x) + ")")
						plt.grid(which='major',axis='both',c='grey')
						plt.axis((Ensemble_Data[:,0].min(),Ensemble_Data[:,0].max(),Ensemble_Data[:,1].min(),Ensemble_Data[:,1].max()))
						
						ax = plt.gca()
						x0, x1 = ax.get_xlim()
						y0, y1 = ax.get_ylim()
						ax.set_aspect(np.abs((x1-x0)/((y1-y0))))
					
						fill_kwargs = {'lw':0.0, 'edgecolor':None}
						plt.fill_between(Ensemble_Data[:,0], Ensemble_Data[:,1] + 1.96*Ensemble_Data[:,2], Ensemble_Data[:,1] - 1.96*Ensemble_Data[:,2], facecolor='black', alpha=0.3, interpolate=True, **fill_kwargs)
						
						if i == 0:
							TimeLength = (num2date(Ensemble_Data[:,0])[-1]-num2date(Ensemble_Data[:,0])[0]).total_seconds() + 1
							if TimeLength/86400 <= 2:
								"""Short Range: ~Single Day"""
								myFmt = DateFormatter('%H:%M')
								ax.xaxis.set_major_formatter(myFmt)
								ax.xaxis.set_major_locator(MinuteLocator(interval=int(round((TimeLength/60)/6))))
								ax.set_xlabel('Time (local) from '+str(Data_All[i][0].strftime('%d/%m/%Y'))+'')
							elif TimeLength/86400 <= 7:
								"""Medium Range: ~Multiple Days"""
								myFmt = DateFormatter('%Y-%m-%d %H:%M') #Use this when plotting multiple days (e.g. monthly summary)
								ax.xaxis.set_major_formatter(myFmt)
								ax.xaxis.set_major_locator(HourLocator(interval=int(round((TimeLength/3600)/6))))
								ax.set_xlabel('Time (local) from '+str(Data_All[i][0].strftime('%H:%M %d/%m/%Y'))+'')
							else:
								"""Long Range: ~Months"""
								myFmt = DateFormatter('%Y-%m-%d') #Use this when plotting multiple days (e.g. monthly summary)
								ax.xaxis.set_major_formatter(myFmt)
								ax.xaxis.set_major_locator(DayLocator(interval=int(round((TimeLength/86400)/6))))
								ax.set_xlabel('Time (UTC)')

							plt.xticks(rotation='vertical')
						
						ax.annotate('$Counts$: %.0d \n$P-Value$: %.4f \n$R^2$: %.4f \n$SE$: %.4f' % (len(Ensemble_Data[:,0]), Linear_Data_Ensemble[j,3], Linear_Data_Ensemble[j,2], Linear_Data_Ensemble[j,4]), xy=(0, 1), xycoords='axes fraction', xytext=(20, -20), textcoords='offset pixels', horizontalalignment='left', verticalalignment='top', fontsize=10)
								
						out_file = 'Corona_' + out[i] + '_Ensemble_' + Ensemble_Name[j] + '_Regression_At_Chilbolton_All_Data' + "_SensorNum_" + str(x)
						plt.savefig(Storage_Path + Plots_Path + 'Linear_Regression/' + out_file + '.png',bbox_inches='tight',pad_inches=0.1, dpi=300)
						
					"""Save Ensemble Data"""
					with open(Storage_Path + "Processed_Data/Corona_RUAO/Ensemble/Corona_" + out[i] + "_Ensemble_Dynamic_SensorNum_" + str(x) + ".csv", "wb") as output:
						writer = csv.writer(output, lineterminator='\n')
						writer.writerows(Ensemble_Dynamic)
					
					with open(Storage_Path + "Processed_Data/Corona_RUAO/Ensemble/Corona_" + out[i] + "_Ensemble_Virtual_SensorNum_" + str(x) + ".csv", "wb") as output:
						writer = csv.writer(output, lineterminator='\n')
						writer.writerows(Ensemble_Virtual)
			
				Step_Num += 1	
				print("[STEP " + str(Step_Num) + "]: Plotted all Ensembles which took %.2f secs" % (systime.time()-t4))
		
			
		if Step_Detection is True:
			############################################################################
			"""Save All Step Detection Data"""
			
			t5 = systime.time()
			
			#Output Data For Step Detection
			with open(Storage_Path + Data_Path + 'Step_Detection/All/Step_Detection_Top-Down_Data_'+str(Data_All[0][0].strftime('%Y%m%d'))+'to'+str(Data_All[0][-1].strftime('%Y%m%d')) + "_SensorNum_" + str(x) +'.csv', "wb") as output:
				writer = csv.writer(output, lineterminator='\n')
				writer.writerows(zip(Corona_Step_Date_All, Corona_Step_Time_All, Corona_Step_Mag_All, Corona_Step_Mode_All))
			
			#Output Statistics For Step Detection
			Corona_Step_Stats_Header = [["Date","Time (min)","Min","Time (Max)","Max","Median","Mean","Std"]]
			with open(Storage_Path + Data_Path + 'Step_Detection/All/Step_Detection_Top-Down_Summary_'+str(Data_All[0][0].strftime('%Y%m%d'))+'to'+str(Data_All[0][-1].strftime('%Y%m%d')) + "_SensorNum_" + str(x) +'.csv', "wb") as output:
				writer = csv.writer(output, lineterminator='\n')
				writer.writerows(Corona_Step_Stats_Header)
				writer.writerows(Corona_Step_Stats_All)
			
			Step_Num += 1	
			print("[STEP " + str(Step_Num) + "]: Saved All Step Detection Data which took %.2f secs" % (systime.time()-t5))
		
		if Step_Detection_All is True and Step_Detection is True:
			
			t6 = systime.time()
			
			############################################################################
			"""Plot overall results of Step Detection"""
			
			#Define model
			thunder_function = lambda x, a, b, c : a*x**2 + b*x + c

			#Fit Model to Data
			datearray2numarray = date2num(Corona_Step_Stats_All.T[0][Corona_Step_Stats_All.T[2] != 0])
			popt_max, pcov_max = curve_fit(thunder_function, np.array(datearray2numarray, dtype=float), np.array(Corona_Step_Stats_All.T[2][Corona_Step_Stats_All.T[2] != 0], dtype=float))
			popt_min, pcov_min = curve_fit(thunder_function, np.array(datearray2numarray, dtype=float), np.array(Corona_Step_Stats_All.T[4][Corona_Step_Stats_All.T[2] != 0], dtype=float))
			
			#Plot All Step Detections
			plt.clf()
			plt.scatter(Corona_Step_Date_All[Corona_Step_Mode_All == 0], Corona_Step_Mag_All[Corona_Step_Mode_All == 0], s=0.1) #, color=colors[0], label=str(labels[0]))
			#plt.scatter(Corona_Step_Date_All[Corona_Step_Mode_All == 1], Corona_Step_Mag_All[Corona_Step_Mode_All == 1], s=0.1, color=colors[1], label=str(labels[1]))
			#plt.scatter(Corona_Step_Date_All[Corona_Step_Mode_All == 2], Corona_Step_Mag_All[Corona_Step_Mode_All == 2], s=0.1, color=colors[2], label=str(labels[2]))
			
			#Plot Fitted Model
			plt.plot(Corona_Step_Stats_All.T[0][Corona_Step_Stats_All.T[2] != 0], thunder_function(datearray2numarray, *popt_max), 'r-', label='fit_max')
			plt.plot(Corona_Step_Stats_All.T[0][Corona_Step_Stats_All.T[2] != 0], thunder_function(datearray2numarray, *popt_min), 'r-', label='fit_min')
			
			#plt.legend(loc='upper left',prop={'size':10})
			plt.grid(which='major',axis='both',c='grey')
			axis_temp = plt.axis()
			plt.axis([axis_temp[0],axis_temp[1],axis_temp[2],axis_temp[3]])
			plt.xlabel("Date (UTC)")
			plt.ylabel("$dCorona/dt$ $(V$ $s^{-1})$")
			plt.title("Time Series of Thunderstorm Activity between " + str(Data_All[0][0].strftime('%Y%m%d')) + " and " + str(Data_All[0][-1].strftime('%Y%m%d')))
			
			#plt.xaxis.set_major_locator(DayLocator(interval=(Corona_Step_Stats_All.T[0][Corona_Step_Stats_All.T[2] != 0][-1]-Corona_Step_Stats_All.T[0][Corona_Step_Stats_All.T[2] != 0][0]).days/10))
			plt.MaxNLocator(nbins=20, min_n_ticks=10)
			plt.xticks(rotation='vertical')
			
			pg = plt.gca()
			x0, x1 = pg.get_xlim()
			y0, y1 = pg.get_ylim()
			pg.set_aspect(int(np.abs((x1-x0)/(4*(y1-y0)))))
			
			fig=plt.gcf()
			fig.set_size_inches(11.7, 11.7/4)
			
			outfile = 'Thunderstorm_Activity_Chilbolton_' + str(Data_All[0][0].strftime('%Y%m%d')) + "_to_" + str(Data_All[0][-1].strftime('%Y%m%d')) + "_SensorNum_" + str(x)
			plt.savefig(Storage_Path + Plots_Path + 'Step_Detection/All/' + outfile, dpi=300, bbox_inches='tight',pad_inches=0.1)
		
			Step_Num += 1	
			print("[STEP " + str(Step_Num) + "]: Saved All Step Detection Data which took %.2f secs" % (systime.time()-t6))
		
	print("[INFO] Corona vs. RUAO completed successfully!")
	
	return
	
	sys.exit()
	# print(P_All)
	# T_AIC = reg_m([PG_All.tolist()], [T_All.tolist()]).aic
	# P_AIC = reg_m([PG_All.tolist()], [P_All.tolist()]).aic
	# RH_AIC = reg_m([PG_All.tolist()], [RH_All.tolist()]).aic
	# TP_AIC = reg_m([PG_All.tolist()], [T_All.tolist(), P_All.tolist()]).aic
	# TRH_AIC = reg_m([PG_All.tolist()], [T_All.tolist(), RH_All.tolist()]).aic
	# PRH_AIC = reg_m([PG_All.tolist()], [P_All.tolist(), RH_All.tolist()]).aic
	# TPRH_AIC = reg_m([PG_All.tolist()], [T_All.tolist(), P_All.tolist(), RH_All.tolist()]).aic
	# print(T_AIC, P_AIC, RH_AIC, TP_AIC, TRH_AIC, PRH_AIC, TPRH_AIC)
		
	############################################################################
	"""Perform Ensemble Plot"""
	
	#Temperature
	PGT_Ensemble_Dynamic = PGRR_Ensembler(100, 1, 4, method=None, case=None, PGRR_Loc=None, pertcount=100, dataimport=False, year=None, month=None, time=None, rainrate=T_All, pg=PG_All)
	PGT_Ensemble_Virtual = PGRR_Ensembler(20, 1, 5, method=None, case=None, PGRR_Loc=None, pertcount=500, dataimport=False, year=None, month=None, time=None, rainrate=T_All, pg=PG_All)
	
	#Relative Humidity
	PGRH_Ensemble_Dynamic = PGRR_Ensembler(100, 1, 4, method=None, case=None, PGRR_Loc=None, pertcount=100, dataimport=False, year=None, month=None, time=None, rainrate=RH_All, pg=PG_All)
	PGRH_Ensemble_Virtual = PGRR_Ensembler(20, 1, 5, method=None, case=None, PGRR_Loc=None, pertcount=500, dataimport=False, year=None, month=None, time=None, rainrate=RH_All, pg=PG_All)
	
	#Pressure
	PGP_Ensemble_Dynamic = PGRR_Ensembler(100, 1, 4, method=None, case=None, PGRR_Loc=None, pertcount=100, dataimport=False, year=None, month=None, time=None, rainrate=P_All[P_All>850], pg=PG_All[P_All>850])
	PGP_Ensemble_Virtual = PGRR_Ensembler(20, 1, 5, method=None, case=None, PGRR_Loc=None, pertcount=500, dataimport=False, year=None, month=None, time=None, rainrate=P_All[P_All>850], pg=PG_All[P_All>850])
	
	EPCC_Plots.Ensemble_Plotter(PGT_Ensemble_Dynamic, 'Average relationship between PG and Temperature: Dynamic', 'Temperature $(^\circ C)$', 'Potential Gradient $(V$ $m^{-1})$', Storage_Path + Plots + 'Chilbolton_TPG_Relationship_Dynamic_CaseStudy_All_SensorNum_' + str(x) + '.png')
	EPCC_Plots.Ensemble_Plotter(PGT_Ensemble_Virtual, 'Average relationship between PG and Temperature: Virtual', 'Temperature $(^\circ C)$', 'Potential Gradient $(V$ $m^{-1})$', Storage_Path + Plots + 'Chilbolton_TPG_Relationship_Virtual_CaseStudy_All_SensorNum_' + str(x) + '.png')
	
	EPCC_Plots.Ensemble_Plotter(PGRH_Ensemble_Dynamic, 'Average relationship between PG and RH: Dynamic', 'Relative Humidity (%)', 'Potential Gradient $(V$ $m^{-1})$', Storage_Path + Plots + 'Chilbolton_RHPG_Relationship_Dynamic_CaseStudy_All_SensorNum_' + str(x) + '.png')
	EPCC_Plots.Ensemble_Plotter(PGRH_Ensemble_Virtual, 'Average relationship between PG and RH: Virtual', 'Relative Humidity (%)', 'Potential Gradient $(V$ $m^{-1})$', Storage_Path + Plots + 'Chilbolton_RHPG_Relationship_Virtual_CaseStudy_All_SensorNum_' + str(x) + '.png')
	
	EPCC_Plots.Ensemble_Plotter(PGP_Ensemble_Dynamic, 'Average relationship between PG and Pressure: Dynamic', 'Pressure $(hPa)$', 'Potential Gradient $(V$ $m^{-1})$', Storage_Path + Plots + 'Chilbolton_PPG_Relationship_Dynamic_CaseStudy_All_SensorNum_' + str(x) + '.png')
	EPCC_Plots.Ensemble_Plotter(PGP_Ensemble_Virtual, 'Average relationship between PG and Pressire: Virtual', 'Pressure $(hPa)$', 'Potential Gradient $(V$ $m^{-1})$', Storage_Path + Plots + 'Chilbolton_PPG_Relationship_Virtual_CaseStudy_All_SensorNum_' + str(x) + '.png')
	
	with open(Storage_Path + "Processed_Data/Corona_RUAO/Ensemble/PGT_Ensemble_Dynamic.csv", "wb") as output:
		writer = csv.writer(output, lineterminator='\n')
		writer.writerows(PGT_Ensemble_Dynamic)
		
	with open(Storage_Path + "Processed_Data/Corona_RUAO/Ensemble/PGT_Ensemble_Virtual.csv", "wb") as output:
		writer = csv.writer(output, lineterminator='\n')
		writer.writerows(PGT_Ensemble_Virtual)
		
	with open(Storage_Path + "Processed_Data/Corona_RUAO/Ensemble/PGRH_Ensemble_Dynamic.csv", "wb") as output:
		writer = csv.writer(output, lineterminator='\n')
		writer.writerows(PGRH_Ensemble_Dynamic)
		
	with open(Storage_Path + "Processed_Data/Corona_RUAO/Ensemble/PGRH_Ensemble_Virtual.csv", "wb") as output:
		writer = csv.writer(output, lineterminator='\n')
		writer.writerows(PGRH_Ensemble_Virtual)
		
	with open(Storage_Path + "Processed_Data/Corona_RUAO/Ensemble/PGP_Ensemble_Dynamic.csv", "wb") as output:
		writer = csv.writer(output, lineterminator='\n')
		writer.writerows(PGP_Ensemble_Dynamic)
		
	with open(Storage_Path + "Processed_Data/Corona_RUAO/Ensemble/PGP_Ensemble_Virtual.csv", "wb") as output:
		writer = csv.writer(output, lineterminator='\n')
		writer.writerows(PGP_Ensemble_Virtual)
	
	print('Ensemble Completed')
	
	t3 = systime.time()
	T_S, T_I, T_R, T_P, std_err = stats.linregress(T_All,PG_All)
	P_S, P_I, P_R, P_P, std_err = stats.linregress(P_All,PG_All)
	RH_S, RH_I, RH_R, RH_P, std_err = stats.linregress(RH_All,PG_All)
	PG_S, PG_I, PG_R, PG_P, std_err = stats.linregress(PG_All,PG_All)
	
	
	############################################################################
	"""Plot Linear Regression"""
	
	t4 = systime.time()	
	plt.clf()
	plt.scatter(T_All,PG_All, s=0.1)
	plt.plot(np.arange(T_All.min(), T_All.max(),0.001), np.arange(T_All.min(), T_All.max(),0.001)*T_S+T_I, lw=0.5)
	plt.xlabel('Temperature $(^\circ C)$')
	plt.ylabel('Potential Gradient $(V$ $m^{-1})$')
	plt.title('Potential Gradient vs Temperature')
	plt.savefig('T vs PG.png',bbox_inches='tight',pad_inches=0.1, dpi=300)
	
	t5 = systime.time()
	plt.clf()
	plt.scatter(P_All,PG_All, s=0.1)
	plt.plot(np.arange(P_All.min(), P_All.max(),0.001), np.arange(P_All.min(), P_All.max(),0.001)*P_S+P_I, lw=0.5)
	plt.xlabel('Pressure $(hPa)$')
	plt.ylabel('Potential Gradient $(V$ $m^{-1})$')
	plt.title('Potential Gradient vs Pressure')
	plt.savefig('P vs PG.png',bbox_inches='tight',pad_inches=0.1, dpi=300)
	
	t6 = systime.time()
	plt.clf()
	plt.scatter(RH_All,PG_All, s=0.1)
	plt.plot(np.arange(RH_All.min(), RH_All.max(),0.001), np.arange(RH_All.min(), RH_All.max(),0.001)*RH_S+RH_I, lw=0.5)
	plt.xlabel('Relative Humidity (%)')
	plt.ylabel('Potential Gradient $(V$ $m^{-1})$')
	plt.title('Potential Gradient vs Relative Humidity')
	plt.savefig('RH vs PG.png',bbox_inches='tight',pad_inches=0.1, dpi=300)
	t7 = systime.time()
	

	print('Time Taken: t2-t1: %.0f,t3-t2: %.0f,t4-t3: %.0f,t5-t4: %.0f,t6-t5: %.0f,t7-t6: %.0f  ' % (t2-t1,t3-t2,t4-t3,t5-t4,t6-t5,t7-t6),'Total %.0f' % (t7-t1))
	print(T_R, T_P)
	print(P_R, P_P)
	print(RH_R, RH_P)
	print(PG_R, PG_P)
	
	############################################################################
	"""Calculate Linear Regression"""
	
	Temp_KS_Stat, Temp_KS_PValue = stats.ks_2samp(T_All,PG_All)
	Temp_KS_Stat, Temp_KS_PValue = stats.ks_2samp(T_All,PG_All)
	Temp_KS_Stat, Temp_KS_PValue = stats.ks_2samp(T_All,PG_All)
	Temp_KS_Stat, Temp_KS_PValue = stats.ks_2samp(T_All,PG_All)

def PGvsRUAO():
	"""Compare the PG against measured parameters at the RUAO.
	
	Parameters
	----------
	Time
	Temperature
	Humidity
	Pressure
	Potential Gradient44
	
	Statistical Tests
	-----------------
	Linear Regression
		p-value
		r squared value
			
	"""
	
	############################################################################
	"""Intialise module"""
	
	Storage_Path = '/glusterfs/phd/users/th863480/WC3_Chilbolton_Corona/'
	Plots_Path = 'Plots/PG_RUAO/'
	Data_Path = 'Processed_Data/PG_RUAO/'
	
	t_start = datetime(2017,05,22,0,0,0) #Inital time for now
	EPCC_Data = EPCC_Importer()
	RUAO_File = glob.glob('/net/vina1s/vol/data1_s/meteorology_2/RUAOData/METFiDAS-3a/Level0/*/*-*-*-Smp1Sec.csv')
	t1 = systime.time()
	############################################################################
	"""Represent the dates of each file to be consistent of the form YYYY"M"DDD"""
	
	Date_Met			= np.array([])
	RUAO_File_Sorted 	= np.array([])

	if RUAO_File > 1:		
		for i in xrange(len(RUAO_File)):
			Year= int(os.path.basename(RUAO_File[i])[:-18])
			Date_Temp = datetime.strptime(os.path.basename(RUAO_File[i])[:-12], '%Y-%m-%d')
			Day = Date_Temp.timetuple().tm_yday
			RUAO_File_Sorted = np.append(RUAO_File_Sorted, RUAO_File[i])
			if int(Day) < 100:
				Date_Met = np.append(Date_Met, str(Year) + "M0" + str(Day))
			else:
				Date_Met = np.append(Date_Met, str(Year) + "M" + str(Day))	
		
	start_ind = np.where(Date_Met >= '2015M100')[0]

	P_All = np.array([])
	T_All = np.array([])
	RH_All = np.array([])
	PG_All = np.array([])	
	t2 = systime.time()
	for i in xrange(len(start_ind)):
		############################################################################
		"""Import Data"""
		print(RUAO_File_Sorted[start_ind[i]])
		Time, RH, T, PG, P = EPCC_Data.RUAO_Calibrate(RUAO_File_Sorted[start_ind[i]], Col=(0,21,27,45,-1), unpack=True)
		
		############################################################################
		"""Calibrate: H and P are already in the correct units"""
		
		P = np.array(P, dtype=float)
		RH = np.array(RH, dtype=float)*100
		T = (np.array(T, dtype=float)-0.0664051)/0.101
		PG = (np.array(PG, dtype=float)-0.00903)/0.00463
				
		P_All = np.append(P_All, P)
		RH_All = np.append(RH_All, RH)
		T_All = np.append(T_All, T)
		PG_All = np.append(PG_All, PG)
		
	############################################################################
	"""Calculate Linear Regression"""
	
	# print(P_All)
	# T_AIC = reg_m([PG_All.tolist()], [T_All.tolist()]).aic
	# P_AIC = reg_m([PG_All.tolist()], [P_All.tolist()]).aic
	# RH_AIC = reg_m([PG_All.tolist()], [RH_All.tolist()]).aic
	# TP_AIC = reg_m([PG_All.tolist()], [T_All.tolist(), P_All.tolist()]).aic
	# TRH_AIC = reg_m([PG_All.tolist()], [T_All.tolist(), RH_All.tolist()]).aic
	# PRH_AIC = reg_m([PG_All.tolist()], [P_All.tolist(), RH_All.tolist()]).aic
	# TPRH_AIC = reg_m([PG_All.tolist()], [T_All.tolist(), P_All.tolist(), RH_All.tolist()]).aic
	# print(T_AIC, P_AIC, RH_AIC, TP_AIC, TRH_AIC, PRH_AIC, TPRH_AIC)
		
	############################################################################
	"""Perform Ensemble Plot"""
	
	#Temperature
	PGT_Ensemble_Dynamic = PGRR_Ensembler(100, 1, 4, method=None, case=None, PGRR_Loc=None, pertcount=100, dataimport=False, year=None, month=None, time=None, rainrate=T_All, pg=PG_All)
	PGT_Ensemble_Virtual = PGRR_Ensembler(20, 1, 5, method=None, case=None, PGRR_Loc=None, pertcount=500, dataimport=False, year=None, month=None, time=None, rainrate=T_All, pg=PG_All)
	
	#Relative Humidity
	PGRH_Ensemble_Dynamic = PGRR_Ensembler(100, 1, 4, method=None, case=None, PGRR_Loc=None, pertcount=100, dataimport=False, year=None, month=None, time=None, rainrate=RH_All, pg=PG_All)
	PGRH_Ensemble_Virtual = PGRR_Ensembler(20, 1, 5, method=None, case=None, PGRR_Loc=None, pertcount=500, dataimport=False, year=None, month=None, time=None, rainrate=RH_All, pg=PG_All)
	
	#Pressure
	PGP_Ensemble_Dynamic = PGRR_Ensembler(100, 1, 4, method=None, case=None, PGRR_Loc=None, pertcount=100, dataimport=False, year=None, month=None, time=None, rainrate=P_All[P_All>850], pg=PG_All[P_All>850])
	PGP_Ensemble_Virtual = PGRR_Ensembler(20, 1, 5, method=None, case=None, PGRR_Loc=None, pertcount=500, dataimport=False, year=None, month=None, time=None, rainrate=P_All[P_All>850], pg=PG_All[P_All>850])
	
	EPCC_Plots.Ensemble_Plotter(PGT_Ensemble_Dynamic, 'Average relationship between PG and Temperature: Dynamic', 'Temperature $(^\circ C)$', 'Potential Gradient $(V$ $m^{-1})$', Storage_Path+Plots_Path+'Chilbolton_TPG_Relationship_Dynamic_CaseStudy_All.png')
	EPCC_Plots.Ensemble_Plotter(PGT_Ensemble_Virtual, 'Average relationship between PG and Temperature: Virtual', 'Temperature $(^\circ C)$', 'Potential Gradient $(V$ $m^{-1})$', Storage_Path+Plots_Path+'Chilbolton_TPG_Relationship_Virtual_CaseStudy_All.png')
	
	EPCC_Plots.Ensemble_Plotter(PGRH_Ensemble_Dynamic, 'Average relationship between PG and RH: Dynamic', 'Relative Humidity (%)', 'Potential Gradient $(V$ $m^{-1})$', Storage_Path+Plots_Path+'Chilbolton_RHPG_Relationship_Dynamic_CaseStudy_All.png')
	EPCC_Plots.Ensemble_Plotter(PGRH_Ensemble_Virtual, 'Average relationship between PG and RH: Virtual', 'Relative Humidity (%)', 'Potential Gradient $(V$ $m^{-1})$', Storage_Path+Plots_Path+'Chilbolton_RHPG_Relationship_Virtual_CaseStudy_All.png')
	
	EPCC_Plots.Ensemble_Plotter(PGP_Ensemble_Dynamic, 'Average relationship between PG and Pressure: Dynamic', 'Pressure $(hPa)$', 'Potential Gradient $(V$ $m^{-1})$', Storage_Path+Plots_Path+'Chilbolton_PPG_Relationship_Dynamic_CaseStudy_All.png')
	EPCC_Plots.Ensemble_Plotter(PGP_Ensemble_Virtual, 'Average relationship between PG and Pressire: Virtual', 'Pressure $(hPa)$', 'Potential Gradient $(V$ $m^{-1})$', Storage_Path+Plots_Path+'Chilbolton_PPG_Relationship_Virtual_CaseStudy_All.png')
	
	with open(Storage_Path + Data_Path + "Ensemble/PGT_Ensemble_Dynamic.csv", "wb") as output:
		writer = csv.writer(output, lineterminator='\n')
		writer.writerows(PGT_Ensemble_Dynamic)
		
	with open(Storage_Path + Data_Path + "Ensemble/PGT_Ensemble_Virtual.csv", "wb") as output:
		writer = csv.writer(output, lineterminator='\n')
		writer.writerows(PGT_Ensemble_Virtual)
		
	with open(Storage_Path + Data_Path + "Ensemble/PGRH_Ensemble_Dynamic.csv", "wb") as output:
		writer = csv.writer(output, lineterminator='\n')
		writer.writerows(PGRH_Ensemble_Dynamic)
		
	with open(Storage_Path + Data_Path + "Ensemble/PGRH_Ensemble_Virtual.csv", "wb") as output:
		writer = csv.writer(output, lineterminator='\n')
		writer.writerows(PGRH_Ensemble_Virtual)
		
	with open(Storage_Path + Data_Path + "Ensemble/PGP_Ensemble_Dynamic.csv", "wb") as output:
		writer = csv.writer(output, lineterminator='\n')
		writer.writerows(PGP_Ensemble_Dynamic)
		
	with open(Storage_Path + Data_Path + "Ensemble/PGP_Ensemble_Virtual.csv", "wb") as output:
		writer = csv.writer(output, lineterminator='\n')
		writer.writerows(PGP_Ensemble_Virtual)
	
	print('Ensemble Completed')

	t3 = systime.time()
	T_S, T_I, T_R, T_P, std_err = stats.linregress(T_All,PG_All)
	P_S, P_I, P_R, P_P, std_err = stats.linregress(P_All,PG_All)
	RH_S, RH_I, RH_R, RH_P, std_err = stats.linregress(RH_All,PG_All)
	PG_S, PG_I, PG_R, PG_P, std_err = stats.linregress(PG_All,PG_All)
	
	
	############################################################################
	"""Plot Linear Regression"""
	
	t4 = systime.time()	
	plt.clf()
	plt.scatter(T_All,PG_All, s=0.1)
	plt.plot(np.arange(T_All.min(), T_All.max(),0.001), np.arange(T_All.min(), T_All.max(),0.001)*T_S+T_I, lw=0.5)
	plt.xlabel('Temperature $(^\circ C)$')
	plt.ylabel('Potential Gradient $(V$ $m^{-1})$')
	plt.title('Potential Gradient vs Temperature')
	plt.savefig('T vs PG.png',bbox_inches='tight',pad_inches=0.1, dpi=300)
	
	t5 = systime.time()
	plt.clf()
	plt.scatter(P_All,PG_All, s=0.1)
	plt.plot(np.arange(P_All.min(), P_All.max(),0.001), np.arange(P_All.min(), P_All.max(),0.001)*P_S+P_I, lw=0.5)
	plt.xlabel('Pressure $(hPa)$')
	plt.ylabel('Potential Gradient $(V$ $m^{-1})$')
	plt.title('Potential Gradient vs Pressure')
	plt.savefig('P vs PG.png',bbox_inches='tight',pad_inches=0.1, dpi=300)
	
	t6 = systime.time()
	plt.clf()
	plt.scatter(RH_All,PG_All, s=0.1)
	plt.plot(np.arange(RH_All.min(), RH_All.max(),0.001), np.arange(RH_All.min(), RH_All.max(),0.001)*RH_S+RH_I, lw=0.5)
	plt.xlabel('Relative Humidity (%)')
	plt.ylabel('Potential Gradient $(V$ $m^{-1})$')
	plt.title('Potential Gradient vs Relative Humidity')
	plt.savefig('RH vs PG.png',bbox_inches='tight',pad_inches=0.1, dpi=300)
	t7 = systime.time()
	

	print('Time Taken: t2-t1: %.0f,t3-t2: %.0f,t4-t3: %.0f,t5-t4: %.0f,t6-t5: %.0f,t7-t6: %.0f  ' % (t2-t1,t3-t2,t4-t3,t5-t4,t6-t5,t7-t6),'Total %.0f' % (t7-t1))
	print(T_R, T_P)
	print(P_R, P_P)
	print(RH_R, RH_P)
	print(PG_R, PG_P)
	
	############################################################################
	"""Calculate Linear Regression"""
	
	Temp_KS_Stat, Temp_KS_PValue = stats.ks_2samp(T_All,PG_All)
	Temp_KS_Stat, Temp_KS_PValue = stats.ks_2samp(T_All,PG_All)
	Temp_KS_Stat, Temp_KS_PValue = stats.ks_2samp(T_All,PG_All)
	Temp_KS_Stat, Temp_KS_PValue = stats.ks_2samp(T_All,PG_All)	

def CoronaFW():

	os.system('cls' if os.name=='nt' else 'clear')
	print("[INFO] You are running Corona FW from the DEV release")

	############################################################################
	"""Intialise module"""
	
	Storage_Path = '/glusterfs/phd/users/th863480/WC3_Chilbolton_Corona/'
	Plots_Path = 'Plots/Corona_RUAO/'
	Data_Path = 'Processed_Data/Corona_RUAO/'
	t1_start = datetime(2017,5,22,9,19,0) #Inital time for first corona sensor
	t1_end   = datetime(2017,9,26,13,30,0) #End time for first corona sensor
	t2_start = datetime(2017,9,26,13,30,0) #Inital time for second corona sensor
	#t2_end   = datetime(2017,09,26,14,00,0) #End time for second corona sensor
	EPCC_Data = EPCC_Importer()
	flatten = lambda l: [item for sublist in l for item in sublist]
	plot_all_step_detection = False
	RUAO_File = glob.glob('/net/vina1s/vol/data1_s/meteorology_2/RUAOData/METFiDAS-3a/Level0/2017/*-*-*-Smp1Sec.csv')
	Data_FW = zip(np.zeros(5))
	print("Intialised Module Complete")

	for x in xrange(1,3): #Loop over each Corona sensor
		
		############################################################################
		"""Import and quality control FW Data"""
		
		if x == 1: continue
		
		Data_FW[0], Data_FW[1], Data_FW[2], Data_FW[3], Data_FW[4] = np.loadtxt(Storage_Path + Data_Path + "All/CSEA_Corona_Data_RUAO_FW_SensorNum_" + str(x) + ".csv.gz", delimiter=",", dtype=str, unpack=True)
		
		print("Imported Data")
		
		Data_FW[0] = np.array([datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in Data_FW[0]], dtype=object)
		for i in xrange(1,len(Data_FW)): Data_FW[i] = np.array(Data_FW[i], dtype=float)
		
		#If there is downtime or skipped data present then we add a np.nan. This will show a gap in the output plots rather than a joined line.
		Time_Skips = np.arange(len(Data_FW[0])-1)[(np.roll(Data_FW[0],-1)-Data_FW[0])[:-1] > timedelta(seconds=1)]
		for i in Time_Skips:
			for j in xrange(len(Data_FW)):
				Data_FW[j] = np.insert(Data_FW[j], i, np.nan)
				Data_FW[j] = np.insert(Data_FW[j], i, np.nan)
		
		############################################################################
		"""Plot all fair weather days of Corona"""
		
		#Convert Times to Dates
		Date_FW = np.zeros(len(Data_FW[0]), dtype=object)
		Time_FW = np.zeros(len(Data_FW[0]), dtype=float)
		for i in xrange(len(Data_FW[0])):
			try:
				Date_FW[i] = Data_FW[0][i].replace(hour=0, minute=0, second=0, microsecond=0)
				Time_FW[i] = toHourFraction(Data_FW[0][i])
			except:
				Date_FW[i] = np.nan
				Time_FW[i] = np.nan
			
		#Find Unique Dates
		Date_FW_Num = np.array([date2num(date) if isinstance(date, datetime) else np.nan for date in Date_FW], dtype=float); Date_FW_Num = Date_FW_Num[~np.isnan(Date_FW_Num)]
		u = num2date(np.unique(Date_FW_Num), tz=None)
		
		#Remove timezones
		u = np.array([datetime.strptime(dates.strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S') for dates in u], dtype=object)
		
		print("No. of Fair Weather Days:", len(u))
		print("Fair Weather Days:", u)

		############################################################################
		"""Plot Corona Voltage and Temperature"""
		
		
		Data_Daily_FW = zip(np.zeros(5))
		for i in xrange(len(u)):
			print('Processing: ' + u[i].strftime('%Y/%m/%d'))
		
			#Organise the data ready for plotting
			for j in xrange(len(Data_FW)):
				Data_Daily_FW[j] = Data_FW[j][Date_FW_Num == u[i]] #Time
			
			#Quality Control Data
			for j in xrange(len(Data_Daily_FW)):
				Data_Daily_FW[j] = Data_Daily_FW[j][(Data_Daily_FW[1]>-9999)&(Data_Daily_FW[2]>-9999)]
			
			#Convert Time_Daily to Hour Frac
			Time_Daily_HF = np.zeros(len(Data_Daily_FW[0]))
			for j in xrange(len(Data_Daily_FW[0])):
				Time_Daily_HF[j] = toHourFraction(Data_Daily_FW[0][j])
					
		XLab = ['Corona $(V)$', 'Temperature $(^\circ C)$', 'Relative Humidity (%)', 'Pressure $(hPa)$']
		YAxis = [[0.02,0.10],[5,20],[30,100],[1000,1030]] if x == 2 else [[0.1,0.16],[10,35],[30,100],[1000,1030]]
		Label = ['Corona', 'Temperature','Relative Humidity','Pressure']
		for k in xrange(4):
			#Calculate Ensemble
			PGFW_Ensemble_Dynamic = PGRR_Ensembler(86400, 1, 4, method=None, case=None, PGRR_Loc=None, pertcount=100, dataimport=False, year=None, month=None, time=None, rainrate=Time_FW, pg=Data_FW[k+1])
			#PGFW_Ensemble_Virtual = PGRR_Ensembler(20, 1, 2, method=None, case=None, PGRR_Loc=None, pertcount=500, dataimport=False, year=None, month=None, time=None, rainrate=Time_FW, pg=Data_FW[1])
			
			plt.clf()
			
			#Plot the data and save to file
			print("LENGTH:",len(Data_FW[k+1]))
			print("LENGHT:",len(Data_FW[k+1][0::10]))
			plt.plot(running_mean(Time_FW,60), running_mean(Data_FW[k+1],60), lw=0.1, c='gray', alpha=0.4)
			
			plt.ylabel(XLab[k])
			plt.xlabel('Time UTC (Hr)')
			x1,x2,y1,y2 = plt.axis()
			plt.axis((0,24,YAxis[k][0],YAxis[k][1]))
			plt.title(Label[k] + ' Timeseries at RUAO during Fair Weather (' + u[0].strftime('%Y/%m/%d') + " - " + u[-1].strftime('%Y/%m/%d') + "), Sensor" + str(x))
			plt.grid(which='major',axis='both',c='grey')
			
			pg = plt.gca()
			x0, x1 = pg.get_xlim()
			y0, y1 = pg.get_ylim()
			pg.set_aspect(np.abs((x1-x0)/(4*(y1-y0))))

			plt.plot(PGFW_Ensemble_Dynamic[:,0],PGFW_Ensemble_Dynamic[:,1],lw=0.5,c='black')
			
			ax=plt.gca()
			ax.xaxis.set_major_locator(MultipleLocator(4))
			plt.minorticks_on()
			
			#Save to file
			out_file = Label[k] + '_Timeseries_Fair Weather_At_Chilbolton_' + u[0].strftime('%Y%m%d') + "_" + u[-1].strftime('%Y%m%d') + "_SensorNum_" + str(x)
			plt.savefig(Storage_Path + Plots_Path + 'Cases/' + out_file + '.png',bbox_inches='tight',pad_inches=0.1, dpi=300)

	print("Fair Weather Plotting and Saving Completed")

def Corona_Bulk():
	"""This function will analyse the corona data in bulk and determine basic statistics after the
	major processing has been completed by CoronavsRUAO function
	
	Data_All[0] = Time
	Data_All[1] = T
	Data_All[2] = P
	Data_All[3] = RH
	Data_All[4] = PG
	Data_All[5] = Cor
	"""
	
	os.system('cls' if os.name=='nt' else 'clear')
	print("[INFO] You are running Corona_Bulk from the DEV release")
	
	############################################################################
	"""Intialise module"""
	
	#File Locations
	Storage_Path = '/glusterfs/phd/users/th863480/WC3_Chilbolton_Corona/'
	Plots_Path = 'Plots/Corona_RUAO/'
	Data_Path = 'Processed_Data/Corona_RUAO/'
	
	#Other
	EPCC_Data = EPCC_Importer()
	flatten = lambda l: [item for sublist in l for item in sublist]
	Data_All = zip(np.zeros(5))
	str2date = False
	
	print("Intialised Module Complete")
	
	for x in xrange(1,3): #Loop over each corona sensor
		
		if x == 1: continue
		
		############################################################################
		"""Import and quality control FW Data"""

		if str2date == True:
			dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
			datatemp = pd.read_csv(Storage_Path + Data_Path + "All/CSEA_Corona_Data_RUAO_All_SensorNum_" + str(x) + ".csv.gz", sep=',', compression = 'gzip', error_bad_lines=False, parse_dates=[0], date_parser=dateparse)
		else:
			datatemp = pd.read_csv(Storage_Path + Data_Path + "All/CSEA_Corona_Data_RUAO_All_SensorNum_" + str(x) + ".csv.gz", sep=',', compression = 'gzip', error_bad_lines=False)
		Data_All = datatemp.as_matrix().T

		print("Imported Data")
		
		#Data_All[0] = np.array([datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in Data_All[0]], dtype=object)
		#for i in xrange(1,len(Data_All)): Data_All[i] = np.array(Data_All[i], dtype=float)
		
		print("Converted to Datetime")
		
		Cor_SD = np.std(Data_All[5], ddof=1)
		Cor_SE = np.median(Data_All[5])/np.std(Data_All[5], ddof=1)
		
		print("Standard Deviation = %.3f, Standard Error = %.3f" % (Cor_SD, Cor_SE))
		
	sys.exit()
	
def reg_m(y, x):
    ones = np.ones(len(x[0]))
    X = sm.add_constant(np.column_stack((x[0], ones)))
    for ele in x[1:]:
        X = sm.add_constant(np.column_stack((ele, X)))
    results = sm.OLS(y, X).fit()
    return results

	
if __name__ == "__main__":
	CoronavsRUAO()
	sys.exit()
	CoronavsRUAO()
	Corona_Bulk()
	PGvsRUAO()
	CoronaFW()