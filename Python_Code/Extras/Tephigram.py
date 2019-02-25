from __future__ import absolute_import, division, print_function
from collections import Iterable, namedtuple
from functools import partial
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.grid_helper_curvelinear import GridHelperCurveLinear
from mpl_toolkits.axisartist import Subplot
import numbers
import numpy as np
import os.path
import matplotlib as mpl
from matplotlib.transforms import Transform
import types
import math
import sys, warnings
from matplotlib.collections import PathCollection
import matplotlib.transforms as mtransforms
from matplotlib.path import Path
from scipy.interpolate import interp1d
from matplotlib.transforms import Affine2D, IdentityTransform

with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	
	sys.path.insert(0, '/home/users/th863480/PhD/Global_Functions')
	
	#User Processing Modules
	import Gilly_Utilities as gu
	
DEFAULT_WIDTH = 700    # in pixels

ISOBAR_SPEC = [(25, .03), (50, .10), (100, .25), (200, 1.5)]
ISOBAR_LINE = {'color':'blue', 'linewidth':0.5, 'clip_on':True}
ISOBAR_TEXT = {'size':8, 'color':'blue', 'clip_on':True, 'va':'bottom', 'ha':'right'}
ISOBAR_FIXED = [50, 1000]

WET_ADIABAT_SPEC = [(1, .05), (2, .15), (4, 1.5)]
WET_ADIABAT_LINE = {'color':'orange', 'linewidth':0.5, 'clip_on':True}
WET_ADIABAT_TEXT = {'size':8, 'color':'orange', 'clip_on':True, 'va':'bottom', 'ha':'left'}
WET_ADIABAT_FIXED = None

MIXING_RATIO_SPEC = [(1, .05), (2, .18), (4, .3), (8, 1.5)]
MIXING_RATIO_LINE = {'color':'green', 'linewidth':0.5, 'clip_on':True}
MIXING_RATIO_TEXT = {'size':8, 'color':'green', 'clip_on':True, 'va':'bottom', 'ha':'right'}
MIXING_RATIOS = [.001, .002, .005, .01, .02, .03, .05, .1, .15, .2, .3, .4, .5, .6, .8,
                  1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0,
                  18.0, 20.0, 24.0, 28.0, 32.0, 36.0, 40.0, 44.0, 48.0, 52.0, 56.0, 60.0, 68.0, 80.0]
MIXING_RATIO_FIXED = None

MIN_PRESSURE = 50     # mb = hPa
MAX_PRESSURE = 1000   # mb = hPa
MIN_THETA = 0         # degC
MAX_THETA = 250       # degC
MIN_WET_ADIABAT = 1   # degC
MAX_WET_ADIABAT = 60  # degC
MIN_TEMPERATURE = -50 # degC

CONST_CP = 1.01e3
CONST_K = 0.286
CONST_KELVIN = 273.15  # celsius to kelvin offset.
CONST_L = 2.5e6
CONST_MA = 300.0
CONST_RD = 287.0
CONST_RV = 461.0

class _FormatterTheta(object):
    def __call__(self, direction, factor, values):
        return [r"$\theta=%s$" % str(value) for value in values]


class _FormatterIsotherm(object):
    def __call__(self, direction, factor, values):
        return [r"  $T=%s$" % str(value) for value in values]


class Locator(object):
    def __init__(self, step):
        self.step = int(step)

    def __call__(self, start, stop):
        step = self.step
        start = (int(start) / step) * step
        stop = (int(stop) / step) * step
        ticks = np.arange(start, stop + step, step)

        return ticks, len(ticks), 1


def _refresh_isopleths(axes):
    changed = False

    # Determine the current zoom level.
    xlim = axes.get_xlim()
    delta_xlim = xlim[1] - xlim[0]
    ylim = axes.get_ylim()
    zoom = delta_xlim / axes.tephigram_original_delta_xlim

    # Determine the display mid-point.
    x_point = xlim[0] + delta_xlim * 0.5
    y_point = ylim[0] + (ylim[1] - ylim[0]) * 0.5
    xy_point = axes.tephigram_inverse.transform(np.array([[x_point, y_point]]))[0]

    for profile in axes.tephigram_profiles:
        profile.refresh()

    for isopleth in axes.tephigram_isopleths:
        changed = isopleth.refresh(zoom, xy_point) or changed

    return changed


def _handler(event):
    for axes in event.canvas.figure.axes:
        if hasattr(axes, 'tephigram'):
            if _refresh_isopleths(axes):
                event.canvas.figure.show()


class _PlotGroup(dict):

    def __init__(self, axes, plot_func, text_kwargs, step, zoom, tags, fixed=None, xfocus=None):
        self.axes = axes
        self.text_kwargs = text_kwargs
        self.step = step
        self.zoom = zoom

        pairs = []
        for tag in tags:
            text = plt.text(0, 0, str(tag), **text_kwargs)
            text.set_bbox(dict(boxstyle='Round,pad=0.3', facecolor='white',
                               edgecolor='white', alpha=0.5, clip_on=True,
                               clip_box=self.axes.bbox))
            pairs.append((tag, [plot_func(tag), text]))

        dict.__init__(self, pairs)
        for line, text in self.itervalues():
            line.set_visible(True)
            text.set_visible(True)
        self._visible = True

        if fixed is None:
            fixed = []

        if not isinstance(fixed, Iterable):
            fixed = [fixed]

        if zoom is None:
            self.fixed = set(tags)
        else:
            self.fixed = set(tags) & set(fixed)

        self.xfocus = xfocus

    def __setitem__(self, tag, item):
        raise ValueError('Cannot add or set an item into the plot group %r' % self.step)

    def __getitem__(self, tag):
        if tag not in self.keys():
            raise KeyError('Tag item %r is not a member of the plot group %r' % (tag, self.step))
        return dict.__getitem__(self, tag)

    def refresh(self, zoom, xy_point):
        if self.zoom is None or zoom <= self.zoom:
            changed = self._item_on()
        else:
            changed = self._item_off()
        self._refresh_text(xy_point)
        return changed

    def _item_on(self, zoom=None):
        changed = False
        if zoom is None or self.zoom is None or zoom <= self.zoom:
            if not self._visible:
                for line, text in self.itervalues():
                    line.set_visible(True)
                    text.set_visible(True)
                changed = True
                self._visible = True
        return changed

    def _item_off(self, zoom=None):
        changed = False
        if self.zoom is not None and (zoom is None or zoom > self.zoom):
            if self._visible:
                for tag, (line, text) in self.iteritems():
                    if tag not in self.fixed:
                        line.set_visible(False)
                        text.set_visible(False)
                        changed = True
                        self._visible = False
        return changed

    def _generate_text(self, tag, xy_point):
        line, text = self[tag]
        x_data = line.get_xdata()
        y_data = line.get_ydata()

        if self.xfocus:
            delta = np.power(x_data - xy_point[0], 2)
        else:
            delta = np.power(x_data - xy_point[0], 2) + np.power(y_data - xy_point[1], 2)
        index = np.argmin(delta)
        text.set_position((x_data[index], y_data[index]))

    def _refresh_text(self, xy_point):
        if self._visible:
            for tag in self:
                self._generate_text(tag, xy_point)
        elif self.fixed:
            for tag in self.fixed:
                self._generate_text(tag, xy_point)


class _PlotCollection(object):
    def __init__(self, axes, spec, stop, plot_func, text_kwargs, fixed=None, minimum=None, xfocus=None):
        if isinstance(stop, Iterable):
            if minimum and minimum > max(stop):
                raise ValueError('Minimum value of %r exceeds all other values' % minimum)

            items = [[step, zoom, set(stop[step - 1::step])] for step, zoom in sorted(spec, reverse=True)]
        else:
            if minimum and minimum > stop:
                raise ValueError('Minimum value of %r exceeds maximum threshold %r' % (minimum, stop))

            items = [[step, zoom, set(range(step, stop + step, step))] for step, zoom in sorted(spec, reverse=True)]

        for index, item in enumerate(items):
            if minimum:
                item[2] = set([value for value in item[2] if value >= minimum])

            for subitem in items[index + 1:]:
                subitem[2] -= item[2]

        self.groups = {item[0]:
                       _PlotGroup(axes, plot_func, text_kwargs, *item, fixed=fixed, xfocus=xfocus) for item in items if item[2]}

        if not self.groups:
            raise ValueError('The plot collection failed to generate any plot groups')

    def refresh(self, zoom, xy_point):

        changed = False

        for group in self.groups.itervalues():
            changed = group.refresh(zoom, xy_point) or changed

        return changed

def savefig(Location, bbox_inches='tight', pad_inches=0.1, dpi=300):
    """Saves plots specified by Location and closes all plots to reduce the memory usage"""
    
    plt.savefig(Location, bbox_inches=bbox_inches, pad_inches=pad_inches, dpi=dpi)		
    plt.close()
    plt.clf()	
 
def show():
    
    plt.show()
 
class Tephigram(object):
    def __init__(self, figure=None, isotherm_locator=None,
                 dry_adiabat_locator=None, anchor=None):
               
        if not figure:
            # Create a default figure.
            self.figure = plt.figure(0, figsize=(9, 9))
        else:
            self.figure = figure

        # Configure the locators.
        if isotherm_locator and not isinstance(isotherm_locator, Locator):
            if not isinstance(isotherm_locator, numbers.Number):
                raise ValueError('Invalid isotherm locator')
            locator_isotherm = Locator(isotherm_locator)
        else:
            locator_isotherm = isotherm_locator

        if dry_adiabat_locator and not isinstance(dry_adiabat_locator, Locator):
            if not isinstance(dry_adiabat_locator, numbers.Number):
                raise ValueError('Invalid dry adiabat locator')
            locator_theta = Locator(dry_adiabat_locator)
        else:
            locator_theta = dry_adiabat_locator
		
        # Define the tephigram coordinate-system transformation.
        self.tephi_transform = TephiTransform()
        grid_helper1 = GridHelperCurveLinear(self.tephi_transform,
                                             tick_formatter1=_FormatterIsotherm(),
                                             grid_locator1=locator_isotherm,
                                             tick_formatter2=_FormatterTheta(),
                                             grid_locator2=locator_theta)
        self.axes = Subplot(self.figure, 1, 1, 1, grid_helper=grid_helper1)
        self.transform = self.tephi_transform + self.axes.transData
        self.axes.axis['isotherm'] = self.axes.new_floating_axis(1, 0)
        self.axes.axis['theta'] = self.axes.new_floating_axis(0, 0)
        self.axes.axis['left'].get_helper().nth_coord_ticks = 0
        self.axes.axis['left'].toggle(all=True)
        self.axes.axis['bottom'].get_helper().nth_coord_ticks = 1
        self.axes.axis['bottom'].toggle(all=True)
        self.axes.axis['top'].get_helper().nth_coord_ticks = 0
        self.axes.axis['top'].toggle(all=False)
        self.axes.axis['right'].get_helper().nth_coord_ticks = 1
        self.axes.axis['right'].toggle(all=True)
        self.axes.gridlines.set_linestyle('solid')

        self.figure.add_subplot(self.axes)

        # Configure default axes.
        axis = self.axes.axis['left']
        axis.major_ticklabels.set_fontsize(10)
        axis.major_ticklabels.set_va('baseline')
        axis.major_ticklabels.set_rotation(135)
        axis = self.axes.axis['right']
        axis.major_ticklabels.set_fontsize(10)
        axis.major_ticklabels.set_va('baseline')
        axis.major_ticklabels.set_rotation(-135)
        self.axes.axis['top'].major_ticklabels.set_fontsize(10)
        axis = self.axes.axis['bottom']
        axis.major_ticklabels.set_fontsize(10)
        axis.major_ticklabels.set_ha('left')
        axis.major_ticklabels.set_va('top')
        axis.major_ticklabels.set_rotation(-45)

        # Isotherms: lines of constant temperature (degC).
        axis = self.axes.axis['isotherm']
        axis.set_axis_direction('right')
        axis.set_axislabel_direction('-')
        axis.major_ticklabels.set_rotation(90)
        axis.major_ticklabels.set_fontsize(10)
        axis.major_ticklabels.set_va('bottom')
        axis.major_ticklabels.set_color('grey')
        axis.major_ticklabels.set_visible(False)  # turned-off

        # Dry adiabats: lines of constant potential temperature (degC).
        axis = self.axes.axis['theta']
        axis.set_axis_direction('right')
        axis.set_axislabel_direction('+')
        axis.major_ticklabels.set_fontsize(10)
        axis.major_ticklabels.set_va('bottom')
        axis.major_ticklabels.set_color('grey')
        axis.major_ticklabels.set_visible(False)  # turned-off
        axis.line.set_linewidth(3)
        axis.line.set_linestyle('--')

        # Lock down the aspect ratio.
        self.axes.set_aspect(1.)
        self.axes.grid(True)

        # Initialise the text formatter for the navigation status bar.
        self.axes.format_coord = self._status_bar

        # Factor in the tephigram transform.
        ISOBAR_TEXT['transform'] = self.transform
        WET_ADIABAT_TEXT['transform'] = self.transform
        MIXING_RATIO_TEXT['transform'] = self.transform

        # Create plot collections for the tephigram 
        func = partial(isobar, MIN_THETA, MAX_THETA, self.axes, self.transform, ISOBAR_LINE)
        self._isobars = _PlotCollection(self.axes, ISOBAR_SPEC, MAX_PRESSURE, func, ISOBAR_TEXT,
                                        fixed=ISOBAR_FIXED, minimum=MIN_PRESSURE)

        func = partial(wet_adiabat, MAX_PRESSURE, MIN_TEMPERATURE, self.axes, self.transform, WET_ADIABAT_LINE)
        self._wet_adiabats = _PlotCollection(self.axes, WET_ADIABAT_SPEC, MAX_WET_ADIABAT, func, WET_ADIABAT_TEXT,
                                             fixed=WET_ADIABAT_FIXED, minimum=MIN_WET_ADIABAT, xfocus=True)

        func = partial(mixing_ratio, MIN_PRESSURE, MAX_PRESSURE, self.axes, self.transform, MIXING_RATIO_LINE)
        self._mixing_ratios = _PlotCollection(self.axes, MIXING_RATIO_SPEC, MIXING_RATIOS, func, MIXING_RATIO_TEXT,
                                              fixed=MIXING_RATIO_FIXED)

        # Initialise for the tephigram plot event handler.
        plt.connect('motion_notify_event', _handler)
        self.axes.tephigram = True
        self.axes.tephigram_original_delta_xlim = self.original_delta_xlim = DEFAULT_WIDTH
        self.axes.tephigram_transform = self.tephi_transform
        self.axes.tephigram_inverse = self.tephi_transform.inverted()
        self.axes.tephigram_isopleths = [self._isobars, self._wet_adiabats, self._mixing_ratios]

       # The tephigram profiles.
        self._profiles = []
        self.axes.tephigram_profiles = self._profiles

        # Center the plot around the anchor extent.
        self._anchor = anchor
        if self._anchor is not None:
            self._anchor = np.asarray(anchor)
            if self._anchor.ndim != 2 or self._anchor.shape[-1] != 2 or \
              len(self._anchor) != 2:
                msg = 'Invalid anchor, expecting [(bottom-left-pressure, ' \
                'bottom-left-temperature), (top-right-pressure, ' \
                'top-right-temperature)]'
                raise ValueError(msg)
            (bottom_pressure, bottom_temp), \
              (top_pressure, top_temp) = self._anchor

            if (bottom_pressure - top_pressure) < 0:
                raise ValueError('Invalid anchor pressure range')
            if (bottom_temp - top_temp) < 0:
                raise ValueError('Invalid anchor temperature range')

            self._anchor = Profile(anchor, self.axes)
            self._anchor.plot(visible=False)
            xlim, ylim = self._calculate_extents()
            self.axes.set_xlim(xlim)
            self.axes.set_ylim(ylim)
    
        #Specify the function to save the tephigram
        self.savefig = savefig
        
        #Specify the function to show the tephigram to screen
        self.show = show
        
        self.legend = legend
		
    def plot(self, data, **kwargs):
        profile = Profile(data, self.axes)
        profile.plot(**kwargs)
        self._profiles.append(profile)

        # Centre the tephigram plot around all the profiles.
        if self._anchor is None:
            xlim, ylim = self._calculate_extents(xfactor=.25, yfactor=.05)
            self.axes.set_xlim(xlim)
            self.axes.set_ylim(ylim)

        # Refresh the tephigram plot 
        _refresh_isopleths(self.axes)

        # Show the plot legend.
        if 'label' in kwargs:
            font_properties = FontProperties(size='x-small')
            plt.legend(loc='upper left', fancybox=True, shadow=True, prop=font_properties)

        return profile
	
    def _status_bar(self, x_point, y_point):
        """Generate text for the interactive backend navigation status bar."""

        temperature, theta = xy_to_temperature_theta(x_point, y_point)
        pressure, _ = temperature_theta_to_pressure_temperature(temperature, theta)
        xlim = self.axes.get_xlim()
        zoom = (xlim[1] - xlim[0]) / self.original_delta_xlim
        text = "T:%.2f, theta:%.2f, phi:%.2f (zoom:%.3f)" % (float(temperature), float(theta), float(pressure), zoom)

        return text

    def _calculate_extents(self, xfactor=None, yfactor=None):
        min_x = min_y = 1e10
        max_x = max_y = -1e-10
        profiles = self._profiles

        if self._anchor is not None:
            profiles = [self._anchor]

        for profile in profiles:
            xy_points = self.tephi_transform.transform(np.concatenate((profile.temperature.reshape(-1, 1),
                                                                       profile.theta.reshape(-1, 1)),
                                                                       axis=1))
            x_points = xy_points[:, 0]
            y_points = xy_points[:, 1]
            min_x, min_y = np.min([min_x, np.min(x_points)]), np.min([min_y, np.min(y_points)])
            max_x, max_y = np.max([max_x, np.max(x_points)]), np.max([max_y, np.max(y_points)])

        if xfactor is not None:
            delta_x = max_x - min_x
            min_x, max_x = min_x - xfactor * delta_x, max_x + xfactor * delta_x

        if yfactor is not None:
            delta_y = max_y - min_y
            min_y, max_y = min_y - yfactor * delta_y, max_y + yfactor * delta_y

        return ([min_x, max_x], [min_y, max_y])
 
def legend():
    """Adds a legend to the plot"""
    
    font_properties = FontProperties(size='x-small')
    plt.legend(loc='upper left', fancybox=True, shadow=True, prop=font_properties)
        
def dewpt(tarr,rarr):
	dewarr =243.04*(np.log(rarr/100.)+((17.625*tarr)/(243.04+tarr)))/(17.625-np.log(rarr/100)-((17.625*tarr)/(243.04+tarr)))
	return dewarr

_BARB_BINS = np.arange(20) * 5 + 3
_BARB_GUTTER = 0.1
_BARB_DTYPE = np.dtype(dict(names=('speed', 'angle', 'pressure', 'barb'),
                            formats=('f4', 'f4', 'f4', np.object)))

#
# Reference: http://www-nwp/~hadaa/tephigram/tephi_plot.html; https://github.com/SciTools/tephi
#


def mixing_ratio(min_pressure, max_pressure, axes,
                 transform, kwargs, mixing_ratio_value):
    
    pressures = np.linspace(min_pressure, max_pressure, 100)
    temps = pressure_mixing_ratio_to_temperature(pressures,
                                                            mixing_ratio_value)
    _, thetas = pressure_temperature_to_temperature_theta(pressures,
                                                                     temps)
    line, = axes.plot(temps, thetas, transform=transform, **kwargs)

    return line


def isobar(min_theta, max_theta, axes, transform, kwargs, pressure):
    steps = 100
    thetas = np.linspace(min_theta, max_theta, steps)
    _, temps = pressure_theta_to_pressure_temperature([pressure] * steps, thetas)
    line, = axes.plot(temps, thetas, transform=transform, **kwargs)

    return line


def _wet_adiabat_gradient(min_temperature, pressure, temperature, dp):
    kelvin = temperature + CONST_KELVIN
    lsbc = (CONST_L / CONST_RV) * ((1.0 / CONST_KELVIN) - (1.0 / kelvin))
    rw = 6.11 * np.exp(lsbc) * (0.622 / pressure)
    lrwbt = (CONST_L * rw) / (CONST_RD * kelvin)
    nume = ((CONST_RD * kelvin) / (CONST_CP * pressure)) * (1.0 + lrwbt)
    deno = 1.0 + (lrwbt * ((0.622 * CONST_L) / (CONST_CP * kelvin)))
    gradi = nume / deno
    dt = dp * gradi

    if (temperature + dt) < min_temperature:
        dt = min_temperature - temperature
        dp = dt / gradi

    return dp, dt


def wet_adiabat(max_pressure, min_temperature, axes,
                transform, kwargs, temperature):
    temps = [temperature]
    pressures = [max_pressure]
    dp = -5.0

    for i in xrange(200):
        dp, dt = _wet_adiabat_gradient(min_temperature, pressures[i],
                                       temps[i], dp)
        temps.append(temps[i] + dt)
        pressures.append(pressures[i] + dp)

    _, thetas = pressure_temperature_to_temperature_theta(pressures,
                                                                     temps)
    line, = axes.plot(temps, thetas, transform=transform, **kwargs)

    return line


class Barbs(object):
    def __init__(self, axes):
        self.axes = axes
        self.barbs = None
        self._gutter = None
        self._transform = axes.tephigram_transform + axes.transData
        self._kwargs = None
        self._custom_kwargs = None
        self._custom = dict(color=['barbcolor', 'color', 'edgecolor', 'facecolor'],
                            linewidth=['lw', 'linewidth'],
                            linestyle=['ls', 'linestyle'])

    @staticmethod
    def _uv(magnitude, angle):
        angle = angle % 360
        u = v = 0
        magnitude = np.searchsorted(_BARB_BINS, magnitude, side='right') * 5
        modulus = angle % 90
        if modulus:
            quadrant = int(angle / 90)
            radians = math.radians(modulus)
            y = math.cos(radians) * magnitude
            x = math.sin(radians) * magnitude
            if quadrant == 0:
                u, v = -x, -y
            elif quadrant == 1:
                u, v = -y, x
            elif quadrant == 2:
                u, v = x, y
            else:
                u, v = y, -x
        else:
            angle = int(angle)
            if angle == 0:
                v = -magnitude
            elif angle == 90:
                u = -magnitude
            elif angle == 180:
                v = magnitude
            else:
                u = magnitude
        return u, v

    def _make_barb(self, temperature, theta, speed, angle):
        u, v = self._uv(speed, angle)
        if 0 < speed < _BARB_BINS[0]:
            # Plot the missing barbless 1-2 knots line.
            length = self._kwargs['length']
            pivot_points = dict(tip=0.0, middle=-length / 2.)
            pivot = self._kwargs.get('pivot', 'tip')
            offset = pivot_points[pivot]
            verts = [(0.0, offset), (0.0, length + offset)]
            verts = Affine2D().rotate(math.radians(-angle)).transform(verts)
            codes = [Path.MOVETO, Path.LINETO]
            path = Path(verts, codes)
            size = length ** 2 / 4
            xy = np.array([[temperature, theta]])
            barb = PathCollection([path], (size,), offsets=xy,
                                  transOffset=self._transform, **self._custom_kwargs)
            barb.set_transform(IdentityTransform())
            self.axes.add_collection(barb)
        else:
            barb = plt.barbs(temperature, theta, u, v,
                             transform=self._transform, **self._kwargs)
        return barb
    
    def refresh(self):
        if self.barbs is not None:
            xlim = self.axes.get_xlim()
            ylim = self.axes.get_ylim()
            y = np.linspace(*ylim)[::-1]
            xdelta = xlim[1] - xlim[0]
            x = np.asarray([xlim[1] - (xdelta * self._gutter)] * y.size)
            points = self.axes.tephigram_inverse.transform(np.asarray(zip(x, y)))
            temperature, theta = points[:, 0], points[:, 1]
            pressure, _ = temperature_theta_to_pressure_temperature(temperature,
                                                                               theta)
            min_pressure, max_pressure = np.min(pressure), np.max(pressure)
            func = interp1d(pressure, temperature)
            for i, (speed, angle, pressure, barb) in enumerate(self.barbs):
                if min_pressure < pressure < max_pressure:
                    temperature, theta = pressure_temperature_to_temperature_theta(pressure,
                                                                                              func(pressure))
                    if barb is None:
                        self.barbs[i]['barb'] = self._make_barb(temperature, theta, speed, angle)
                    else:
                        barb.set_offsets(np.array([[temperature, theta]]))
                        barb.set_visible(True)
                else:
                    if barb is not None:
                        barb.set_visible(False)

    def plot(self, barbs, **kwargs):
        self._gutter = kwargs.pop('gutter', _BARB_GUTTER)
        self._kwargs = dict(length=7, zorder=10)
        self._kwargs.update(kwargs)
        self._custom_kwargs = dict(color=None, linewidth=1.5,
                                   zorder=self._kwargs['zorder'])
        for key, values in self._custom.iteritems():
            common = set(values).intersection(kwargs)
            if common:
                self._custom_kwargs[key] = kwargs[sorted(common)[0]]
        barbs = np.asarray(barbs)
        if barbs.ndim != 2 or barbs.shape[-1] != 3:
            msg = 'The barbs require to be a sequence of wind speed, ' \
              'wind direction and pressure value triples.'
            raise ValueError(msg)
        self.barbs = np.empty(barbs.shape[0], dtype=_BARB_DTYPE)
        for i, barb in enumerate(barbs):
            self.barbs[i] = tuple(barb) + (None,)
        self.refresh()


class Profile(object):
    def __init__(self, data, axes):
        self.data = np.asarray(data)
        if self.data.ndim != 2 or self.data.shape[-1] != 2:
            msg = 'The environment profile data requires to be a sequence ' \
              'of pressure, temperature value pairs.'
            raise ValueError(msg)
        self.axes = axes
        self._transform = axes.tephigram_transform + axes.transData
        self.pressure = self.data[:, 0]
        self.temperature = self.data[:, 1]
        _, self.theta = pressure_temperature_to_temperature_theta(self.pressure,
                                                                             self.temperature)
        self.line = None
        self._barbs = Barbs(axes)
				
    def plot(self, **kwargs):
        if self.line is not None and line in self.axes.lines:
            self.axes.lines.remove(line)

        if 'zorder' not in kwargs:
            kwargs['zorder'] = 10

        self.line, = self.axes.plot(self.temperature, self.theta,
                                    transform=self._transform, **kwargs)
        return self.line

    def refresh(self):
        self._barbs.refresh()

    def barbs(self, barbs, **kwargs):
        colors = ['color', 'barbcolor', 'edgecolor', 'facecolor']
        if not set(colors).intersection(kwargs):
            kwargs['color'] = self.line.get_color()
        self._barbs.plot(barbs, **kwargs)   

def temperature_theta_to_pressure_temperature(temperature, theta):
    temperature, theta = np.asarray(temperature), np.asarray(theta)

    # Convert temperature and theta from degC to kelvin.
    kelvin = temperature + CONST_KELVIN
    theta = theta + CONST_KELVIN

    # Calculate the associated pressure given the temperature and
    # potential temperature.
    pressure = 1000.0 * np.power(kelvin / theta, 1 / CONST_K)

    return pressure, temperature


def pressure_temperature_to_temperature_theta(pressure, temperature):
    pressure, temperature = np.asarray(pressure), np.asarray(temperature)

    # Convert temperature from degC to kelvin.
    kelvin = temperature + CONST_KELVIN

    # Calculate the potential temperature given the pressure and temperature.
    theta = kelvin * gu.power((1000.0 / pressure), CONST_K)

    # Convert potential temperature from kelvin to degC.
    return temperature, theta - CONST_KELVIN


def pressure_theta_to_pressure_temperature(pressure, theta):
    pressure, theta = np.asarray(pressure), np.asarray(theta)

    # Convert potential temperature from degC to kelvin.
    theta = theta + CONST_KELVIN

    # Calculate the temperature given the pressure and
    # potential temperature.
    kelvin = theta * (pressure ** CONST_K) / (1000.0 ** CONST_K)

    # Convert temperature from kelvin to degC.
    return pressure, kelvin - CONST_KELVIN


def temperature_theta_to_xy(temperature, theta):
    temperature, theta = np.asarray(temperature), np.asarray(theta)

    # Convert potential temperature from degC to kelvin.
    theta = theta + CONST_KELVIN
    theta = np.clip(theta, 1, 1e10)

    phi = np.log(theta)

    x_data = phi * CONST_MA + temperature
    y_data = phi * CONST_MA - temperature

    return x_data, y_data


def xy_to_temperature_theta(x_data, y_data):
    x_data, y_data = np.asarray(x_data), np.asarray(y_data)

    phi = (x_data + y_data) / (2 * CONST_MA)
    temperature = (x_data - y_data) / 2.

    theta = np.exp(phi) - CONST_KELVIN

    return temperature, theta


def pressure_mixing_ratio_to_temperature(pressure, mixing_ratio):
    pressure = np.array(pressure)

    # Calculate the dew-point.
    vapp = pressure * (8.0 / 5.0) * (mixing_ratio / 1000.0)
    temp = 1.0 / ((1.0 / CONST_KELVIN) - ((CONST_RV / CONST_L) * np.log(vapp / 6.11)))

    return temp - CONST_KELVIN


class TephiTransform(Transform):
    input_dims = 2
    output_dims = 2
    is_separable = False
    has_inverse = True

    def transform_non_affine(self, values):
        return np.concatenate(temperature_theta_to_xy(values[:, 0:1], values[:, 1:2]), axis=1)

    def inverted(self):
        return TephiTransformInverted()


class TephiTransformInverted(Transform):
    input_dims = 2
    output_dims = 2
    is_separable = False
    has_inverse = True

    def transform_non_affine(self, values):
        return np.concatenate(xy_to_temperature_theta(values[:, 0:1], values[:, 1:2]), axis=1)

    def inverted(self):
        return TephiTransform()  	

if __name__ == "__main__":		
	'''####PLOTTING SUBROUTINE#####'''

	pressure = np.array([ 1000.,   975.,   950.,   925.,   900.,   875.,   850.,   825.,
			 800.,   775.,   750.,   700.,   650.,   600.,   550.,   500.,
			 450.,   400.,   350.,   300.,   250.,   225.,   200.,   175.,
			 150.,   125.,   100.])
	temperature = np.array([ 28.13538345,  26.66628428,  25.16955946,  23.86828005,
			22.65900604,  21.46101683,  20.22887875,  18.98980924,
			17.75413103,  16.52913328,  15.31474142,  12.76488613,
			 9.86460882,   6.35658798,   2.44785581,  -1.05558403,
			-5.272435  , -10.58404989, -16.98981033, -25.11008589,
		   -35.18328257, -41.1803993 , -47.89660411, -55.30133102,
		   -63.38163261, -72.12679684, -79.67241396])
	rel_hum = np.array([ 89.27313978,  90.21353687,  91.98861606,  93.0377063 ,
			93.61489972,  93.62926463,  93.34088101,  92.56918629,
			91.44150499,  90.04826896,  88.30413316,  84.38214649,
			80.62636976,  78.65866032,  78.123495  ,  73.93630546,
			68.84032143,  64.64213722,  62.82134388,  63.66245727,
			66.85679341,  69.67181515,  74.15342369,  81.26106606,
			87.92477937,  91.82851447,  94.35841179])
	dewpoint = dewpt(temperature,rel_hum)

	wind_pressures = np.array([ 1000.,   900.,   850.,
			 800.,   700.,   600.,   550.,   500.,  400.,   350.,   300.,   250.,   200.,    150.,   100.])
	wind_speed = np.array([  0.,   1.,   5.,   5.,   7.,  10.,  12.,  15.,  25.,  35.,  40.,
			43.,  45.,  50.,  55.])
	wind_direction = np.array([   0.,   15.,   25.,   30.,   60.,   90.,  105.,  120.,  180.,
			240.,  270.,  285.,  300.,  330.,  359.])

	print("pressure", type(pressure), pressure)
	print("temperature", type(temperature), temperature)
	print("dewpoint", type(dewpoint), dewpoint)
	print("wind_speed", type(wind_speed), wind_speed)
	print("wind_direction", type(wind_direction), wind_direction)
	
	dews = zip(pressure, dewpoint)
	temps = zip(pressure, temperature)
	barb_vals = zip(wind_speed,wind_direction,pressure)
	tpg = Tephigram()
	profile_t1 = tpg.plot(temps,color="red",linewidth=2)
	profile_d1 = tpg.plot(dews,color="blue",linewidth=2)
	profile_t1.barbs(barb_vals)
	plt.show()







    

