from __future__ import absolute_import, division, print_function
import warnings

__version__ = 2.0

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
	