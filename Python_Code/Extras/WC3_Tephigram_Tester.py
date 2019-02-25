############################################################################
# Project: Electrical Pre-Conditioning of Convective Clouds,
# Title: University of Wyoming Upper Level Sounding Tester
# Author: James Gilmore,
# Email: james.gilmore@pgr.reading.ac.uk.
# Version: 0.1
# Date: 24/07/2018
# Status: Alpha
# Change: TBC
############################################################################
from __future__ import absolute_import, division, print_function
import numpy as np
import scipy as sp
import os, sys, time, warnings, glob
from datetime import datetime, timedelta

sys.path.insert(0, '/home/users/th863480/PhD/Global_Functions')

#User Processing Modules
import Gilly_Utilities as gu

#Data Set-up Modules
from Data_Importer import EPCC_Importer
from Data_Output import SPRadiosonde

#Import Global Variables
import PhD_Config as PhD_Global

def Wyoming_Tephigram(Launch_Datetime):
	"""This function will use the University of Wyoming Upper Level Sounding data set
	to calculate the radiosonde indices. The aim is to cross-compare the values 
	calculated here with the values the University of Wyoming states"""
	
	gu.cprint("[INFO] You are running Wyoming_Tephigram from the STABLE release", type='bold')
	
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
	"""Find and import Wyoming ULS data"""
	
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
	
	#Import Camborne Radiosonde Data
	print("Importing ULS File @", ULS_File[ID])
	Pres, Z, Tdry, Tdew, Wind_Dir, Wind_Mag = EPCC_Data.ULS_Calibrate(ULS_File[ID], unpack=True, HGHT=True, PRES=True, TEMP=True, DWPT=True, SKNT=True, DRCT=True)
	
	# print("Z", Z)
	# print("Tdry", Tdry)
	# print("Tdew", Tdew)
	# print("Pres", Pres)
	# print("Wind_Mag", Wind_Mag)
	# print("Wind_Dir", Wind_Dir)
	# sys.exit()
	
	############################################################################
	"""Calculate indices from Wyoming ULS"""
		
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
	ms2kn = 1	#Conversion between m/s to knots
		
	SW_1 = 20*(TT-49)
	SW_2 = 12*Tdew[P_850]
	SW_3 = 2*Wind_Mag[P_850]*ms2kn
	SW_4 = Wind_Mag[P_500]*ms2kn
	SW_5 = 125*(np.sin(Wind_Dir[P_500]-Wind_Dir[P_850]) + 0.2)
	
	print("SWEAT Index", SW_1, SW_2, SW_3, SW_4, SW_5)
	
	#Condition SWEAT Term 1 from several conditions
	SW_1 = 0 if SW_1 < 49 else SW_1

	print("Wind_Dir[P_850]", Wind_Dir[P_850])
	print("Wind_Dir[P_500]", Wind_Dir[P_500])
	print("Wind_Dir[P_500]-Wind_Dir[P_850]", Wind_Dir[P_500]-Wind_Dir[P_850])
	print("Wind_Mag[P_500]", Wind_Mag[P_500], "Wind_Mag[P_850]", Wind_Mag[P_850])
	
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
	#Z *= 1000
	
	#Constants
	over27 = 2/7 # Value used for calculating potential temperature 2/7
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
	
	#Find first location where Pqs_base > theta_base
	y1 = np.arange(Pqs_base.size)[Pqs_base > theta_base][0]
	
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
	print("LCL = %.2fm (%.1fhPa)" % (LCL, Pres[gu.argnear(Z, LCL)]))
	print("LFC = %.2fm (%.1fhPa)" % (LFC, Pres[gu.argnear(Z, LFC)]))
	print("EL = %.2fm (%.1fhPa)" % (EL, Pres[gu.argnear(Z, EL)]))
	print("CAPE %.2f J/kg" % CAPE)
	print("CIN %.2f J/kg" % CIN)
	print("\n")
	
	print("[INFO] Radiosonde_Tephigram has been completed successfully (In %.2fs)" % (time.time()-t_begin))
	
if __name__ == "__main__":
	
	gu.cprint("Welcome to Tephigram Tester. Testing the codes ability to calculate the stability indices, LCL, LFC, EL, CAPE and CIN.", type='bold')
	
	Date2Test = datetime(2017,10,12,0)
	
	Wyoming_Tephigram(Date2Test)