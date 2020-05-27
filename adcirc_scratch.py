'Scratch ADCIRC file for CMTB development'

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

import netCDF4 as nc



version_prefix = 'base'
model = 'adcirc'

stations = {}
stations['x'] = [-75.7467, -75.748719, -75.739155]
stations['y'] = [36.1833, 36.186769, 36.189244]
stations['name'] = ['Station 8651370 Duck NC', 'AWAC-4.5m', 'AWAC-11m']


# lets go get some recent observations
start = DT.datetime(2018, 9, 1, 0, 0, 0)
end = DT.datetime(2018, 9, 30, 0, 0, 0)
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
wl = go.getWL()

#
#
#        #temp1= np.transpose(A[0][:,num])
        #nmseEst[num] = temp1.dot((dcovA2[:,1]-2*dgcov[:,1]))
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
z = np.array(meshDict['values'])
points = np.transpose(np.vstack((x, y, z)))
meshDict['points'] = points
gridNodes = objectview(meshDict)
# prepdata = STPD.PrepDataTools()
#
# meteoPacket = prepdata.prep_ADCIRCmeteo(winds, pres, gridNodes)
#



path = '/home/dylananderson/cmtb/data/adcirc/base/2018-09-01T000000Z/'


output = adcircio.read_fort61(path=path, file_name='fort.61', ihot=False, timeSteps=None, getTime=True)
timemodel = output['timeObs']
elevmodel = output['timeSeriesData'][0, :]

piermodel = output['timeSeriesData'][0, :]
awac4model = output['timeSeriesData'][1, :]
awac11model = output['timeSeriesData'][2, :]

from plotting.operationalPlots import obs_V_mod_TS



# path = '/media/dylananderson/fattyBoBatty/2019-08-20T000000Zb/'
# output2 = adcircio.read_fort61(path=path, file_name='fort.61', ihot=False, timeSteps=None, getTime=True)
# timemodel2 = output['timeObs']
# elevmodel2 = output['timeSeriesData'][0, :]#+.3317-.13



import time
import matplotlib.dates as mdate
#d='2015-08-24 00:00:00'
d='2018-08-30 00:00:00'

p='%Y-%m-%d %H:%M:%S'
epoch = int(time.mktime(time.strptime(d, p)))

epochtime = epoch+timemodel#-(60*60*4)
#realtime = np.zeros((len(epochtime),))
#for i in range(len(epochtime)):
#    realtime[i] = DT.datetime.fromtimestamp(epochtime[i-1])
obs_time = wl['epochtime']+(60*60*4)
obs_secs = mdate.epoch2num(obs_time)
model_secs = mdate.epoch2num(epochtime)


overlap_time = np.intersect1d(model_secs, obs_secs)
overlap = np.in1d(model_secs, obs_secs)
indices = np.arange(model_secs.shape[0])[np.in1d(model_secs, obs_secs)]
indices2 = np.arange(obs_secs.shape[0])[np.in1d(obs_secs, model_secs)]
overlap_time2 = epochtime[indices2]

modelpred = elevmodel[indices]
noaapred = wl['WL']
#noaapred = wl['predictedWL']

ticks = overlap_time2.astype('datetime64[s]').tolist()

#plot_time = np.empty((len(overlap_time2),))
#for num, i in enumerate(overlap_time2):
#plot_time = DT.datetime.utcfromtimestamp(overlap_time2[0])
#plot_time = nc.num2date(overlap_time2, p)

#    t=overlap_time).strftime('%Y-%m-%d %H:%M:%S')
#


p_dict = dict()
p_dict['time'] = ticks
p_dict['obs'] = noaapred
p_dict['model'] = modelpred
p_dict['var_name'] = 'Water Level'
p_dict['units'] = 'm'
p_dict['p_title'] = 'testing operational plot scripts'
#/home/dylananderson/cmtb/data/adcirc/base/2018-09-01T000000Z
testing = obs_V_mod_TS(ofname='testing.png', p_dict=p_dict, logo_path='ArchiveFolder/CHL_logo.png')


fig = plt.figure(figsize=(6, 6))

grid = plt.GridSpec(2, 2, wspace=0.6, hspace=0.6)

main_ax = fig.add_subplot(grid[:-1, 0:])


main_ax.plot_date(model_secs, elevmodel, 'k-', label='ADCIRC tidal forcing')
#ax.plot_date(model_secs, elevmodel2, 'k-', label='ADCIRC tides + met')
main_ax.plot_date(obs_secs, wl['WL'], 'r-', label='Pier observations')
main_ax.plot_date(obs_secs, wl['predictedWL'], 'b-', label='NOAA prediction')
date_fmt = '%d-%m-%y %H:%M:%S'
date_formatter = mdate.DateFormatter(date_fmt)
main_ax.xaxis.set_major_formatter(date_formatter)
main_ax.title.set_text('Simulation for Sep. 2018 (UTC)')
main_ax.legend()
# Sets the tick labels diagonal so they fit easier.
fig.autofmt_xdate()


ll_ax = fig.add_subplot(grid[1:, :-1])

#overlap = list(set(model_secs) & set(obs_secs))
#overlap_time = np.intersect1d(model_secs, obs_secs)
#overlap = np.in1d(model_secs, obs_secs)
#indices = np.arange(model_secs.shape[0])[np.in1d(model_secs, obs_secs)]

ll_ax.scatter(wl['predictedWL'], elevmodel[indices], marker='.')
ll_ax.plot([-1, 1], [-1, 1], 'k-', label='1-to-1')


#m, c = np.linalg.lstsq(wl['predictedWL'], elevmodel[indices], rcond=None)[0]
from scipy import stats

slope, intercept, r_value, p_value, std_err = stats.linregress(wl['predictedWL'], elevmodel[indices])

xfit = np.array([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
yfit = slope*xfit + intercept

ll_ax.plot(xfit, yfit, 'r--', label='Regression fit intercept: {:<.3f} cm'.format(intercept))

plt.xlim([-1.1, 1.1])
plt.ylim([-1.1, 1.1])
ll_ax.legend()
plt.title('End of Pier: Comparison at dt = 6 minutes')
plt.ylabel('ADCIRC simulation')
plt.xlabel('NOAA Prediction')


lr_ax = fig.add_subplot(grid[1:, 1:])


lr_ax.scatter(wl['WL'], elevmodel[indices], color='r', marker='.')
lr_ax.plot([-1.5, 1.5], [-1.5, 1.5], 'k-', label='1-to-1')


#m, c = np.linalg.lstsq(wl['predictedWL'], elevmodel[indices], rcond=None)[0]

slope, intercept, r_value, p_value, std_err = stats.linregress(wl['WL'], elevmodel[indices])

xfit = np.array([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
yfit = slope*xfit + intercept

lr_ax.plot(xfit, yfit, 'r--', label='Regression fit intercept: {:<.3f} cm'.format(intercept))

plt.xlim([-1.1, 1.1])
plt.ylim([-1.1, 1.1])
lr_ax.legend()
plt.title('End of Pier: Comparison at dt = 6 minutes')
plt.ylabel('ADCIRC simulation')
plt.xlabel('FRF Water Level Observation')












fig, ax = plt.subplots()
ax.plot_date(model_secs, awac4model,'k-', label='AWAC in 4.5m')
ax.plot_date(model_secs, piermodel,'b-', label='Pier in 7.5m')
ax.plot_date(model_secs, awac11model,'r-', label='AWAC in 11m')
date_fmt = '%d-%m-%y %H:%M:%S'
date_formatter = mdate.DateFormatter(date_fmt)
ax.xaxis.set_major_formatter(date_formatter)
ax.title.set_text('Simulation for Thursday Aug. 15, 2019 (UTC)')
ax.legend()
# Sets the tick labels diagonal so they fit easier.
fig.autofmt_xdate()




#plt.scatter(epochtime, elevmodel)
#plt.plot(wl['epochtime'], wl['WL'], color='red')
plt.show


#
#
# #
# # # do interpolation
#
# #gridNodes = pdl.prep_Bathy(bathy, gridNodes=gridNodes, unstructured=True, positiveDown=False, plotFname=os.path.join(path_prefix, date_str, 'bathy'+date_str))
# #
# #
# # #adcircio.tidalForcing()
# #
# # #namMeteo = pdl.prep_NAMmeteo(meteo, gridNodes)
# #
# # #adcircio.write_fort22(namMeteo, gridNodes)
# #
# # #fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# # #ax.scatter(namWinds['lon'], namWinds['lat'], c=namWinds['values'], vmin=1, vmax=3)
# #
# # #test = adcplots.make_AdcircMeshPlot(gridNodes, vmin=-105, vmax=10)
# #
# # #adcircio.write_fort14(gridNodes, meshDict)
# # #adcircio.write_fort15()
# # #adcircio.write_fort151s()
# # #adcircio.write_fort13()
# #
# #
# #
#
#
#
#
#
#
#
#
#
#
#
# #from AdcircPy import Model
# #from AdcircPy.Model import AdcircMesh
# #from datetime import datetime, timedelta
#
# #fort14 = '/home/dylananderson/cmtb/grids/adcirc/fort.14'
#
# #mesh = AdcircMesh(fort14, SpatialReference=4326, vertical_datum='LSML')
# #start_date = datetime.now()
# #end_date = start_date + timedelta(days=5)
#
# #adicrc_run = mesh.write_fort14(path='/home/dylananderson/projects/ADCIRC_ERDC/test_adcirc_gen')
#
# #adcirc_run = mesh.TidalRun(start_date, end_date, spinup_days=7)
# #adcirc_run.dump(output_dir='/home/dylananderson/projects/ADCIRC_ERDC/test_adcirc_gen')  # will write to stdout if no output path is given.
#
#
# #mesh.make_plot(show=True, figsize=(8.0,10.0), vmin=-105, vmax=8)
#
#
# # import datetime as DT
# # from getdatatestbed import getDataFRF
# # import matplotlib.pyplot as plt
# #
# # start = DT.datetime(2018, 1, 1, 0, 0, 0)
# # end = DT.datetime(2018, 1, 2, 0, 0, 0)
# #
# # data = getDataFRF.getObs(start, end, THREDDS='FRF')
# # winds = data.getWind()
# # windtime = winds['time']
# # windspeed = winds['windspeed']
# # plt.plot(windtime, windspeed)
# # plt.show
# #
# #
