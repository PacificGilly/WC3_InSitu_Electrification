from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
from matplotlib import transforms
from itertools import takewhile

#Define alphabet
alphabet = list('abcdefghijklmnopqrstuvwxyz')

def inheritors(klass):
    subclasses = set()
    work = [klass]
    while work:
        parent = work.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)
                work.append(child)
    return list(subclasses)

class prettyfloat(object):
	"""returns a truncated float of 2 decimal places.
	
	Example
	-------
	use: map(prettyfloat, x) for a list, x
	
	Reference
	---------
	https://stackoverflow.com/a/1567630
	"""
	
	def __init__(self, float, dec=6):
		"""dec = precision of float"""
		
		self.float=float
		self.dec=dec

	def __repr__(self):
		return "%0.*f" % (self.dec, self.float)
	
def rainbow_text(x, y, strings, colors, ax=None, **kw):
	"""
    Take a list of ``strings`` and ``colors`` and place them next to each
    other, with text strings[i] being shown in colors[i].

    This example shows how to do both vertical and horizontal text, and will
    pass all keyword arguments to plt.text, so you can set the font size,
    family, etc.

	The text will get added to the ``ax`` axes, if provided, otherwise the
	currently active axes will be used.
	"""

	if ax is None:
		ax = plt.gca()
	t = ax.transData
	canvas = ax.figure.canvas

    # horizontal version
	for s, c in zip(strings, colors):
		text = ax.text(x, y, s + " ", color=c, transform=t, **kw)
		text.draw(canvas.get_renderer())
		ex = text.get_window_extent()
		t = transforms.offset_copy(text._transform, x=ex.width, units='dots')

	# vertical version
	for s, c in zip(strings, colors):
		text = ax.text(x, y, s + " ", color=c, transform=t,
						rotation=90, va='bottom', ha='center', **kw)
		text.draw(canvas.get_renderer())
		ex = text.get_window_extent()
		t = transforms.offset_copy(text._transform, y=ex.height, units='dots')
		
def cprint(*args, **kwargs):
	"""Prints colourful messages under a standard style set.
	
	Update: Now handles in the same as does print, meaning you
	can give print as many input strings as you want. e.g.
		
	>>> cprint("Hello", "Brian", "Simon")
	Hello Brian Simon	#but in bold
	
	>>> print("Hello", "Brian", "Simon")
	Hello Brian Simon
	
	No need to specifiy kind anymore. Now defaults to bold.
	Also give kwargs to the print statement now such as 'file'.
	N.B. supplying end in cprint will cause an error as that is
	already being used in this function when print is called."""
	
	#Console Colours
	bcolours = {
		"header" : '\033[95m',
		"okblue" : '\033[94m',
		"okgreen" : '\033[92m',
		"warning" : '\033[31m',
		"fail" : '\033[91m',
		"endc" : '\033[0m',
		"bold" : '\033[1m',
		"underline" : '\033[4m'}
	
	type = kwargs.pop('type', "bold")
	
	for message in args:
		print(bcolours[type] + message + bcolours['endc'], end=' ', **kwargs)
	print("\r")
	
def rgb2hex(r,g,b):
	"""Converts RGB values into Hexadecimal
	
	Reference
	---------
	
	https://stackoverflow.com/a/43572620/8765762
	"""
	
	return "#{:02x}{:02x}{:02x}".format(r,g,b)		

def readcomments(filename, comment='#'):
	"""
	Reads in filename and extracts only data that contains the comments
	parameter.
	
	Reference
	---------
	https://stackoverflow.com/a/39724905/8765762
	"""
			
	comments_all = []
	with open(filename,'r') as cmt_file:    # open file
		for line in cmt_file:    # read each line
			if line[0] == comment:    # check the first character
				comments_all.append(line[1:])    # remove first '#'

	return comments_all	

