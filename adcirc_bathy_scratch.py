

import netCDF4 as nc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import os
import numpy as np
import datetime as DT
from prepdata import inputOutput
import matplotlib.pyplot as plt
import plotting.dailyPlotsADCIRC as adcplots
from prepdata import prepDataLib
from getdatatestbed import getDataFRF
from getdatatestbed import getOutsideData
import subprocess
import netCDF4 as ncattempt
from types import SimpleNamespace
import pygrib
from siphon.catalog import TDSCatalog
import siphon.ncss as ncss

from frontback.frontBackADCIRC import ADCIRCsimSetup

from prepdata import prepDataLib as STPD

#
# ncFile = nc.Dataset('https://chlthredds.erdc.dren.mil/thredds/dodsC/cmtb/integratedBathyProduct/backgroundDEM.nc')
#
# latbounds = [33, 36]
# lonbounds = [-73, -76]
#
#
# #lat = ncFile['latitude']
# index1 = np.arange(0, 22699, 100)
# index2 = np.arange(0, 8999, 100)
# #latIndex = np.nonzero(lat>30)
# lats = ncFile.variables['latitude'][index1, index2]
# lons = ncFile.variables['longitude'][index1, index2]
#
# # # latitude lower and upper index
# # latli = np.argmin(np.abs(lats - latbounds[0]))
# # latui = np.argmin(np.abs(lats - latbounds[1]))
# #
# # # longitude lower and upper index
# # lonli = np.argmin(np.abs(lons - lonbounds[0]))
# # lonui = np.argmin(np.abs(lons - lonbounds[1]))
#
# # Bathy (time, latitude, longitude)
# bathySubset = ncFile.variables['bottomElevation'][index1, index2]
#
# plt.scatter(lons, lats, c=bathySubset, vmin=-200, vmax=10)
# plt.colorbar()
# plt.show()
#




version_prefix = 'base'
model = 'adcirc'

stations = {}
stations['x'] = [-75.7467, -75.748719, -75.739155]
stations['y'] = [36.1833, 36.186769, 36.189244]
stations['name'] = ['Station 8651370 Duck NC', 'AWAC-4.5m', 'AWAC-11m']


# lets go get some recent observations
start = DT.datetime(2018, 5, 2, 0, 0, 0)
end = DT.datetime(2018, 5, 3, 0, 0, 0)
date_str = start.strftime('%Y-%m-%dT%H%M%SZ')
path_prefix = "/home/dylananderson/cmtb/data/{}/{}/".format(model, version_prefix)

pdl = prepDataLib.PrepDataTools()
gdtb = getDataFRF.getDataTestBed(start, end, THREDDS='FRF')
go = getDataFRF.getObs(start, end, THREDDS='FRF')

#
#
# winds = go.getWind()
# pres = go.getBarom()
#


#
#
#
# # fig, ax = plt.subplots(4, 1, figsize=(10,10))
# # plt1 = ax[0].plot(windTime, u)
# # plt2 = ax[1].plot(windTime, v)
# # plt3 = ax[2].plot(windTime, windSpeed)
# # plt4 = ax[3].plot(windTime, windDirection)
# #
# # plt.show
#
#
#
# #
# # #forecast = getOutsideData.forecastData(start)
# # #meteo = forecast.getNamMeteo()
# #
class objectview(object):
    def __init__(self, d):
        self.__dict__ = d
# # #
adcircio = inputOutput.adcircIO(RUNDES='Duck', NOLIBF=1, NOLIFA=0, NOLICA=0, NOLICAT=0, RNDAY=10, DRAMP=5,
                                SLAM0=-75.748712, SFEA0=36.200989, iopath=path_prefix, NSTAE2=3, obs=stations)
# # #
# # #
# # # ############ bathy #########################################
bathy = gdtb.getBathyIntegratedTransect()       # get new bathy
meshDict = adcircio.read_ADCIRCmesh()          # load old bathy
x = np.array(meshDict['x'])
y = np.array(meshDict['y'])
z = np.array(meshDict['values']) #/3.28
points = np.transpose(np.vstack((x, y, z)))
meshDict['points'] = points
gridNodes = objectview(meshDict)



from scipy.spatial import Delaunay

pts = np.zeros((26305, 2))
pts[:, 0] = x
pts[:, 1] = y

tri = Delaunay(pts)


centers = np.sum(pts[tri.simplices], axis=1, dtype='int')/3.0
#colors = np.array([ (x-w/2.)**2 + (y-h/2.)**2 for x,y in centers])
plt.triplot(pts[:, 0], pts[:, 1], tri.simplices.copy())

plt.gca().set_aspect('equal')
plt.show()



import pyproj
from testbedutils import geoprocess

myProj = pyproj.Proj("+proj=utm +zone=18S, +north +ellps=WGS84 +datum=WGS84 +untis=m +no_defs")

ncSP = pyproj.Proj(init='epsg:3358')

#surNCx, surNCy = geoprocess.LatLon2ncsp(bathy['lon'], bathy['lat'])
#meshNCx, meshNCy = geoprocess.LatLon2ncsp(x, y)
surNCx, surNCy = ncSP(bathy['lon'], bathy['lat'])
meshNCx, meshNCy = ncSP(x, y)

FRF = geoprocess.FRFcoord(meshNCx, meshNCy)
#FRF = geoprocess.FRFcoord(surNCx, surNCy)

UTMx, UTMy = myProj(bathy['lon'], bathy['lat'])

#plt.scatter(meshNCx, meshNCy, c=z)



#plt.scatter(FRF['xFRF'], FRF['yFRF'], c=z)#bathy['elevation'])
#plt.scatter(FRF['xFRF'], FRF['yFRF'], bathy['elevation'])
gridx, gridy = np.meshgrid(bathy['xFRF'], bathy['yFRF'])

from matplotlib import tri

triObj = tri.Triangulation(bathy['lon'].flatten(), bathy['lat'].flatten())  # making the triangles
ftri = tri.LinearTriInterpolator(triObj, bathy['elevation'].flatten())  # making the interpolation function
newZs = ftri(gridNodes.points[:, 0], gridNodes.points[:, 1])  # the actual interpolation
oldZs = np.copy(gridNodes.points[:, 2])  ## for plotting
gridNodes.points[~newZs.mask, 2] = newZs[~newZs.mask]  # replacing values in total data structure

fig, ax = plt.subplots(3, 1, figsize=(12, 10))
plt1 = ax[0].scatter(gridx, gridy, c=bathy['elevation'], vmin=-36, vmax=4)
cbar = plt.colorbar(plt1, ax=ax[0])
plt2 = ax[1].scatter(FRF['xFRF'], FRF['yFRF'], c=z, vmin=-36, vmax=4)
cbar2 = plt.colorbar(plt2, ax=ax[1])

subx = FRF['xFRF'][~newZs.mask]
suby = FRF['yFRF'][~newZs.mask]
subz = gridNodes.points[~newZs.mask, 2]
oldz = oldZs[~newZs.mask]
diffz = (oldz/subz)
# subx = gridNodes.points[~newZs.mask, 0]
# suby = gridNodes.points[~newZs.mask, 1]
# subz = gridNodes.points[~newZs.mask, 2]
#plt3 = ax[2].scatter(FRF['xFRF'], FRF['yFRF'], c=gridNodes.points[:, 2], vmin=-36, vmax=4)
plt3 = ax[2].scatter(subx, suby, c=diffz, vmin=2.5, vmax=4.1, cmap='bwr')

# plt3 = ax[2].scatter(bathy['lon'].flatten(), bathy['lat'].flatten(), c=bathy['elevation'].flatten(), vmin=-12, vmax=4)
cbar3 = plt.colorbar(plt3, ax=ax[2])
cbar3.set_label('ratio (b)/(a)')
cbar2.set_label('Mesh depth ()')
cbar.set_label('Survey depth (m)')
ax[0].set_ylim([-500, 5500])
ax[0].set_xlim([-500, 1400])
ax[1].set_ylim([-500, 5500])
ax[1].set_xlim([-500, 1400])
ax[2].set_ylim([-500, 5500])
ax[2].set_xlim([-500, 1400])
ax[0].title.set_text('a) Recent Survey (in FRF coordinates)')
ax[1].title.set_text('b) Mesh grid in same lat/lon box as (a)')
ax[2].title.set_text('c) Ratio: (Mesh grid) / (survey) ~ 3.3')
ax[0].set_ylabel('alongshore (m)')
ax[1].set_ylabel('alongshore (m)')
ax[2].set_ylabel('alongshore (m)')
ax[2].set_xlabel('cross-shore (m)')

#plt.show()

#plt.savefig('bathy_interpolation.png')
# ax[1].set_ylim([bathy['lat'].min(), bathy['lat'].max()])
# # ax[1].set_xlim([bathy['lon'].min(), bathy['lon'].max()])
# # ax[2].set_ylim([bathy['lat'].min(), bathy['lat'].max()])
# # ax[2].set_xlim([bathy['lon'].min(), bathy['lon'].max()])



from mpl_toolkits.basemap import Basemap

import matplotlib.cm as mpl_cm
import matplotlib.colors as mpl_col
# plt.figure(figsize=(8, 8))
# m = Basemap(projection='ortho', resolution=None, lat_0=50, lon_0=-100)
# m.bluemarble(scale=0.5);
# plt.show

from itertools import chain


def draw_map(m, scale=0.2):
    # draw a shaded-relief image
    m.shadedrelief(scale=scale)

    # lats and longs are returned as a dictionary
    lats = m.drawparallels(np.linspace(-90, 90, 13))
    lons = m.drawmeridians(np.linspace(-180, 180, 13))

    # keys contain the plt.Line2D instances
    lat_lines = chain(*(tup[1][0] for tup in lats.items()))
    lon_lines = chain(*(tup[1][0] for tup in lons.items()))
    all_lines = chain(lat_lines, lon_lines)

    # cycle through these lines and set the desired style
    for line in all_lines:
        line.set(linestyle='-', alpha=0.3, color='w')


# fig = plt.figure(figsize=(8, 8))
# m = Basemap(projection='lcc', resolution=None,
#             width=4E6, height=2E6,
#             lat_0=32, lon_0=-123,)
# m.etopo(scale=0.7, alpha=0.4)

# Map (long, lat) to (x, y) for plotting
#x, y = m(-122.3, 47.6)
#plt.plot(x, y, 'ok', markersize=5)
#plt.text(x, y, ' Seattle', fontsize=12);

def plot_map(service='World_Physical_Map', epsg=4269, xpixels=6000):
    # note, you need change the epsg for different region,
    # US is 4269, and you can google the region you want
    #fig1, ax2 = plt.subplots(constrained_layout=True)
    plt.figure(figsize=(8, 6))
#    m = Basemap(projection='mill', llcrnrlon=-123., llcrnrlat=37,
#                urcrnrlon=-121, urcrnrlat=39, resolution='l', epsg=epsg)

    # southern cali
    m = Basemap(projection='mill', llcrnrlon=-76, llcrnrlat=35.5,
                urcrnrlon=-74.5, urcrnrlat=37, resolution='l', epsg=epsg)

    # southern cali
#    m = Basemap(projection='mill', llcrnrlon=-122., llcrnrlat=30,
#                urcrnrlon=-115.5, urcrnrlat=36, resolution='l', epsg=epsg)


    # xpixels controls the pixels in x direction, and if you leave ypixels
    # None, it will choose ypixels based on the aspect ratio
    m.arcgisimage(service=service, xpixels=xpixels, verbose=False)

    plt.show()


plot_map(service='Ocean_Basemap', epsg=4269)
#plt.scatter(bathy['lon'].flatten(), bathy['lat'].flatten(), c=bathy['elevation'].flatten())#, vmin=-12, vmax=4)
sc = plt.scatter(x, y, c=z, vmin=-100, vmax=8)
#sc.cmap.set_under('k')
cbar5 = plt.colorbar() #set_over=80, color='k')
cbar5.set_label('Mesh units ()')
sc2 = plt.scatter(-75.592666, 36.25816, c='red', label='Wave Buoy')
plt.legend()
plt.title('Closest Mesh node to 26 m wave buoy = 85.6 depth')


stn_lat=36.25816
stn_lon=-75.592666

abslat = np.abs(y-stn_lat)
abslon= np.abs(x-stn_lon)


c = np.maximum(abslon, abslat)
latlon_idx = np.argmin(c)
grid_temp = z[latlon_idx]
print(grid_temp)
# from math import cos, asin, sqrt
#
# def distance(lat1, lon1, lat2, lon2):
#     p = 0.017453292519943295
#     a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p)*cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
#     return 12742 * asin(sqrt(a))


# #def closest(data, v):
# #    return min(data, key=lambda p: distance(v['lat'], v['lon'], p['lat'], p['lon']))

# def closest(x, y, v):
#     d1 = []
#     for p in x:
#         dist = distance(x[p], y[p], v['lon'], v, ['lat'])
#         d1.append(dist)
#
#     return min(d1)

#
#     return min(data, key=lambda p: distance(v['lat'], v['lon'], p['lat'], p['lon']))
#
# tempDataList = [{'lat': 39.7612992, 'lon': -86.1519681},
#                 {'lat': 39.762241,  'lon': -86.158436 },
#                 {'lat': 39.7622292, 'lon': -86.1578917}]
#
# tempData = dict()
# tempData['x'] = x
# tempData['y'] = y
#
#
# v = {'lat': 39.7622290, 'lon': -86.1519750}
# print(closest(tempDataList, v))



# # fig = plt.figure(figsize=(8, 8))
# # m = Basemap(projection='ortho', resolution=None,
# #             lat_0=50, lon_0=0)
# # draw_map(m);
#
# # 1. Draw the map background
# fig = plt.figure(figsize=(8, 8))
# m = Basemap(projection='lcc', resolution='i',
#             lat_0=33, lon_0=-120,
#             width=1.0E6, height=0.5E6,
#             epsg=2771)
# #m.shadedrelief()
# m.arcgisimage(service="World_Shaded_Relief", xpixels=1000)
# #m.bluemarble()
# #m.drawcoastlines(color='gray')
# #m.drawmapboundary(color='blue')
# #m.drawcountries(color='gray')
# #m.drawstates(color='gray')
#



