from __future__ import print_function
import numpy as np
import os, shutil, threading, errno, signal, logging

from .extras import cprint

class DelayedKeyboardInterrupt(object):
	"""Forces an operation to complete when this class function is called before an 
	operation. Handles Ctrl+C, Ctrl+Z and Ctrl+\ keyboard interrupts.
	
	Example
	-------
	
	with DelayedKeyboardInterrupt():
		for i in xrange(10):
			print(i)
			systime.sleep(1)
			
	>>> 0
	>>> 1
	>>> ^C[Warning] Please wait until the current operation has finish
	>>> ^Z[Warning] Please wait until the current operation has finish
	>>> ^\[Warning] Please wait until the current operation has finish
	>>> 2
	...
	>>> 10
	>>> KeyboardInterrupt
	
	Reference
	---------
	https://stackoverflow.com/a/21919644/8765762
	"""	
	
	def __enter__(self):
		self.signal_received = False
		self.ctrlc_handler = signal.signal(signal.SIGINT, self.handler)
		self.ctrlz_handler = signal.signal(signal.SIGTSTP, self.handler)
		self.ctrlslash_handler = signal.signal(signal.SIGQUIT, self.handler)
		
	def handler(self, sig, frame):
		self.signal_received = (sig, frame)
		logging.debug('SIGINT received. Delaying KeyboardInterrupt.')
		cprint("[Warning] Please wait until the current operation has finished", type='warning')
		
	def __exit__(self, type, value, traceback):
		signal.signal(signal.SIGINT, self.ctrlc_handler)
		if self.signal_received:
			self.ctrlc_handler(*self.signal_received)
			
		signal.signal(signal.SIGTSTP, self.ctrlz_handler)
		if self.signal_received:
			self.ctrlz_handler(*self.signal_received)
			
		signal.signal(signal.SIGQUIT, self.ctrlslash_handler)
		if self.signal_received:
			self.ctrlslash_handler(*self.signal_received)
			
def ensure_dir(dirname):
    """Ensure that a named directory exists; if it does not, attempt to create it.
    
	Sources
	-------
	http://stackoverflow.com/a/21349806
	"""

    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def del_folder(folder, sub=False): 
    """Deletes a folder. Option sub to make recursive"""

    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path) and sub == True: 
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)
			
def Print_Output(delay_time, t_start, it_tot, it):
  t = threading.Timer(delay_time, Print_Output(delay_time, t_start, it_tot, it)).start()
  os.system('cls' if os.name=='nt' else 'clear')
  print("Time Left (s): ", "%.0f" % ((it_tot-it)*((time.time()-t_start)/it)))
  print("Events Left: ", it_tot-it)

def File_Checker(files, dates, date_range, file_size_min=0):
	"""Checks the availability of files
	
	Parameters
	----------
	files : numpy array
		A list of file locations to be tested
	dates : numpy object array
		An array of datetimes corresponding to each file location in files.
	date_range : nump object array
		An array over the list of dates you want to test.
	file_size_min : int, optional
		The minimum file size (in bytes) required to be valid. This is useful for files that
		exist but don't contain any data.

	Returns
	-------
	File_Availability : list
		A boolean list with 1 for file exists and is above the minimum file size threshold
		and zero for either file doesn't exist or file is below or equal to minimum file
		threshold.		
	"""
	
	File_Availability = []
	for i in xrange(date_range.shape[0]):
		if np.sum(dates == date_range[i]) == 1:			
			if os.stat(files[dates == date_range[i]][0]).st_size > file_size_min:
				File_Availability.append(1)
			else:
				File_Availability.append(2)
		else:
			File_Availability.append(0)
			
	return np.array(File_Availability, dtype=float)

def File_Aligner_NoSize(files, dates, date_range):
	"""Checks the availability of files
	
	Parameters
	----------
	files : numpy array
		A list of file locations to be tested
	dates : numpy object array
		An array of datetimes corresponding to each file location in files.
	date_range : nump object array
		An array over the list of dates you want to test.
	file_size_min : int, optional
		The minimum file size (in bytes) required to be valid. This is useful for files that
		exist but don't contain any data.

	Returns
	-------
	File_Availability : list
		A boolean list with 1 for file exists and is above the minimum file size threshold
		and zero for either file doesn't exist or file is below or equal to minimum file
		threshold.		
	"""
	
	File_Date = np.zeros(date_range.shape[0], dtype=object)
	File_Loc = np.zeros(date_range.shape[0], dtype='S256')
	for i in xrange(date_range.shape[0]):
		if np.sum(dates == date_range[i]) >= 1:			
			File_Date[i] = date_range[i]
			File_Loc[i] = files[dates == date_range[i]][0]
			
	return np.array(File_Loc, dtype='S256'), np.array(File_Date, dtype=object)
	
def File_Aligner(files, dates, date_range, file_size_min=0):
	"""Checks the availability of files
	
	Parameters
	----------
	files : numpy array
		A list of file locations to be tested
	dates : numpy object array
		An array of datetimes corresponding to each file location in files.
	date_range : nump object array
		An array over the list of dates you want to test.
	file_size_min : int, optional
		The minimum file size (in bytes) required to be valid. This is useful for files that
		exist but don't contain any data.

	Returns
	-------
	File_Availability : list
		A boolean list with 1 for file exists and is above the minimum file size threshold
		and zero for either file doesn't exist or file is below or equal to minimum file
		threshold.		
	"""
	
	File_Date = np.zeros(date_range.shape[0], dtype=object)
	File_Loc = np.zeros(date_range.shape[0], dtype='S256')
	for i in xrange(date_range.shape[0]):
		if np.sum(dates == date_range[i]) == 1:	
			if os.stat(files[dates == date_range[i]][0]).st_size > file_size_min:
				File_Date[i] = date_range[i]
				File_Loc[i] = files[dates == date_range[i]][0]
			else:
				File_Date[i] = 2
				File_Loc[i] = 2
			
	return np.array(File_Loc, dtype='S256'), np.array(File_Date, dtype=object)
	
