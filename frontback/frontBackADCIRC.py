
import datetime as DT
import os
import numpy as np
from prepdata import prepDataLib
from getdatatestbed import getDataFRF
from getdatatestbed import getOutsideData
from prepdata import inputOutput
from prepdata import prepDataLib as STPD
import subprocess



def ADCIRCsimSetup(startTime, inputDict):
    """This Function is the master call for the  data preparation for the Coastal Model
    Test Bed (CMTB) and the ADCIRC flow model

    Designed for unstructured grids

    NOTE: input to the function is the end of the duration.  All Files are labeled by this convention
    all time stamps otherwise are top of the data collection

    Args:
        startTime (str): this is a string of format YYYY-mm-ddTHH:MM:SSZ (or YYYY-mm-dd) in UTC time
        inputDict (dict): this is a dictionary that is read from the yaml read function

    """

    # begin by setting up input parameters

    timerun = int(inputDict.get('simulationDuration', 24))
    plotFlag = inputDict.get('plotFlag', True)
    version_prefix = inputDict['version_prefix'].lower()
    server = inputDict.get('THREDDS', 'CHL')
    model = inputDict['model'].lower()
    path_prefix = inputDict.get('path_prefix').lower()
    modelExecutable = inputDict.get('modelExecutable')
    numOfCores = inputDict.get('numberOfCores')
    TOD = 0  # hour of day simulation to start (UTC)

    # ______________________________________________________________________________
    # define version parameters
    versionlist = ['base']
    assert version_prefix.lower() in versionlist, 'Please check your version Prefix'
    #simFnameBackground = inputDict.get('gridSIM', None)
    #backgroundDepFname = inputDict.get('gridDEP', None)
    # do versioning stuff here
    if version_prefix.lower() in ['hp', 'untuned']:
        full = False
    else:
        full = True
        pass

    # _______________________________________________________________________________
    # set times
    try:
        d1 = DT.datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%SZ') + DT.timedelta(TOD / 24., 0, 0)
        d2 = d1 + DT.timedelta(0, timerun * 3600, 0)
        date_str = d1.strftime('%Y-%m-%dT%H%M%SZ')  # used to be endtime

    except ValueError:
        assert len(startTime) == 10, 'Your Time does not fit convention, check T/Z and input format'
        d1 = DT.datetime.strptime(startTime, '%Y-%m-%d') + DT.timedelta(TOD / 24., 0, 0)
        d2 = d1 + DT.timedelta(0, timerun * 3600, 0)
        date_str = d1.strftime('%Y-%m-%d')  # used to be endtime
        assert int(timerun) >= 24, 'Running Simulations with less than 24 Hours of simulation time require end ' \
                                   'Time format in type: %Y-%m-%dT%H:%M:%SZ'

    # __________________Make Directories_____________________________________________
    if not os.path.exists(os.path.join(path_prefix, date_str)):                    # if it doesn't exist
        os.makedirs(os.path.join(path_prefix, date_str))                           # make the directory
    if not os.path.exists(os.path.join(path_prefix, date_str, 'figures')):
        os.makedirs(os.path.join(path_prefix, date_str, "figures"))

    outputDir = os.path.join(path_prefix, date_str)


    # _________________ choosing model save (gauge) locations ____________________________________
    # TODO build a lookup table to search for stations that have changes location at the FRF over the years
    stations = {}
    stations['x'] = [-75.7467, -75.748719, -75.739155]
    stations['y'] = [36.1833, 36.186769, 36.189244]
    stations['name'] = ['Station 8651370 Duck NC', 'AWAC-4.5m', 'AWAC-11m']

    # __________________________________________________________________________________________________________________
    # ____________________________ begin model data gathering __________________________________________________________
    # __________________________________________________________________________________________________________________
    gdTB = getDataFRF.getDataTestBed(d1, d2, THREDDS=server)

    # ______________ Get latest grid from thredds to determine its time stamp and hot/cold start _______________________
    print('_________________ Getting Bathymetry Data _________________')
    bathy = gdTB.getBathyIntegratedTransect(method=1)  # , ForcedSurveyDate=ForcedSurveyDate)

    # ______________ Two different calls to input/output depending on the hot/cold designation _________________________
    timeElapsed = d2 - bathy['time']
    if timeElapsed.days <= 1:
        print('  There is a new survey, time to run a cold start with ramp function')
        # Which means we need to reset the start date to 2(?) days further back in time
        d1 = d1-DT.timedelta(days=2)
        adcircio = inputOutput.adcircIO(RUNDES='Duck', RNDAY=8, DRAMP=1, NWS=5, iopath=outputDir, NSTAE2=3, obs=stations, bathy=bathy)

    else:
        print('  At %s days old lets see if there is a hot start available to run...' % (timeElapsed.days))
        # if a hot start really exist then we need to identify which file is "newer", the 67 or 68 version and then
        # give an ihot=67/68 to the IO
        # This also means we just want the model to start 24 hours ago
        d1 = d1-DT.timedelta(days=2)

        adcircio = inputOutput.adcircIO(RUNDES='Duck', RNDAY=8, DRAMP=1, NWS=5, iopath=outputDir, NSTAE2=3, obs=stations, bathy=bathy)
    prepdata = STPD.PrepDataTools()

    meshDict = adcircio.read_ADCIRCmesh()  # load old bathy
    gridNodes = adcircio.prep_ADCIRCbathy(bathy=bathy, meshDict=meshDict)
    # do interpolation
    gridNodes = prepdata.prep_Bathy(bathy, gridNodes=gridNodes, unstructured=True, positiveDown=False,
                               plotFname=os.path.join(path_prefix, date_str, 'bathy' + date_str))


    # _____________METEOROLOGY ______________________
    print('_________________ Getting Met Data _________________')

    timeDifference = DT.datetime.now()-DT.datetime.strptime(inputDict['endTime'], '%Y-%m-%dT%H:%M:%SZ')
    # print(timeDifference.days)
    if timeDifference.days < 0:
        print('You are asking this simulation to occur in the future..')
        print('Assuming this is a forecast for the next 1 to 3 days')
        forecast = getOutsideData.forecastData(d1=d1)
        namMeteo = forecast.getNamMeteo()
        meteoPacket = prepdata.prep_NAMmeteo(namMeteo, gridNodes)
        print('Retrieved latest NAM meteorology forecast')

    else:
        try:
            d2 = d2 + DT.timedelta(days=2)
            go = getDataFRF.getObs(d1, d2)
            rawwind = go.getWind()                                                # get wind data
            rawpres = go.getBarom()                                                  # get atmospheric pressure
            meteoPacket = prepdata.prep_ADCIRCmeteo(rawwind, rawpres, gridNodes)
            print('  -- number of met records {} '.format(np.size(meteoPacket['time'])))
        except (RuntimeError, TypeError):
            meteoPacket = None
            print('     NO MET ON RECORD')

    # ___________TIDAL FORCING__________________
    print('_________________ Getting Tidal Constituents _________________')
    adcircio.tidalForcing(d1=d1, durationInDays=timerun/24)

    # __________________________________________________________________________________________________________________
    # ____________________________ begin writing model Input ___________________________________________________________
    # __________________________________________________________________________________________________________________
    print('_________________ writing output _________________')
    print('writing out fort.15 (parameter controller')
    print('            fort.14 (mesh nodes and boundaries)')
    print('            fort.13 (mesh attributes)')
    print('            fort.22 (met forcing)')
    print('            ###_stat.151 (save locations for each variable)')

    adcircio.write_fort14(gridNodes)
    adcircio.write_fort15()
    adcircio.write_fort151s()
    adcircio.write_fort13()
    adcircio.write_fort22(meteoPacket, gridNodes)


    # __________  Time to prepare for parallel processing ______________________________________________________________
    # ___________ Need to split the files into sub folders _____________________________________________________________
    #input = '{:0.0f}\n'.format(durationInDays) + '{:0.0f} {:0.0f} {:0.0f} {:0.0f}\n\n\n'.format(hour, day, month, year)
    #self.tideOutput = subprocess.check_output([self.tide_fac_dir], input=input,
    #                                          universal_newlines=True, shell=True)
    # first need to run adcprep
    # tell it 25 cores
    # tell it option 1
    # tell it the name of the fort.14 file
    # run adcprep again?
    # tell it 25 cores
    # tell it option 2
    print(' Preparing mesh partition ')
    adcprep_dir = modelExecutable #'/home/dylananderson/projects/ADCIRC_ERDC/CSTORM_MS_Phase1_Build/adcirc/work/adcprep'
    adcprep_dir = '/home/dylananderson/projects/ADCIRC_ERDC/CSTORM_MS_Phase1_Build/adcirc/work/adcprep'

    option = 1
    input = '{:0.0f}\n'.format(numOfCores) + '{:0.0f}\n'.format(option) + 'fort.14\n\n\n'
    os.chdir(outputDir)
    partition = subprocess.check_output([adcprep_dir], input=input, universal_newlines=True, shell=True)

    print(' Preparing separate folders for each mesh partition ')
    option = 2
    input = '{:0.0f}\n'.format(numOfCores) + '{:0.0f}\n\n\n'.format(option)
    os.chdir(outputDir)
    foldersmaking = subprocess.check_output([adcprep_dir], input=input, universal_newlines=True, shell=True)







#    print("Model Time Start : %s  Model Time End:  %s" % (d1, d2))
 #    print("OPERATIONAL files will be place in {0} folder".format(outputDir))

 #


 #    # _______________________________ meteorological inputs ____________________________________________________________
 #
 #    if modelMode == 'forecast':
 #        print('lets get NAM winds for tomorrow')
 # #       forecast = getOutsideData.forecastData(d1=d1)
 # #       namMeteo = forecast.getNamMeteo()
 # #       meteo = pdl.prep_NAMmeteo(namMeteo, gridNodes)
 # #       adcircio.write_fort22(meteo, gridNodes)
 #
 #    elif modelMode == 'base':
 #        print('lets get our observations for this')
 #
 #    elif modelMode == 'hindcast':
 #        print('lets get a high res reanalysis for this?')
 #
 #    # _______________________________________ tidal forcing ____________________________________________________________
 #    adcircio.tidalForcing(d1=d1, durationInDays=timerun/24)
