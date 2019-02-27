from __future__ import absolute_import, division, print_function
import warnings

__version__ = 2.2
__doc__ = "Gilly Utilities is a module which adds extra support for common data analysis problems. The functions and classes in this module are mainly extensions of the numpy package with some extensions for scipy and pandas."

with warnings.catch_warnings():
	warnings.simplefilter("ignore")

	from .datetime64 import *
	from .nanfunctions import *
	from .math import *
	from .stats import *
	from .meteorology import *
	from .system import *
	from .plotting import *
	from .manipulation import *
	from .geospatial import *
	from .extras import *
	