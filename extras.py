from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
from matplotlib import transforms

#Define alphabet
alphabet = list('abcdefghijklmnopqrstuvwxyz')

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
		
def cprint(message, type):
	"""Prints colourful messages under a standard style set"""
	
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
		
	print(bcolours[type] + message + bcolours['endc'])
	
def rgb2hex(r,g,b):
	"""Converts RGB values into Hexadecimal
	
	Reference
	---------
	
	https://stackoverflow.com/a/43572620/8765762
	"""
	
	return "#{:02x}{:02x}{:02x}".format(r,g,b)		


