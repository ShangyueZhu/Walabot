from __future__ import print_function # WalabotAPI works on both Python 2 an 3.
from sys import platform
from os import system
from imp import load_source
from os.path import join
import time
import csv
import numpy as np
import matplotlib.pyplot as plt

if platform == 'win32':
	modulePath = join('C:/', 'Program Files', 'Walabot', 'WalabotSDK',
		'python', 'WalabotAPI.py')
elif platform.startswith('linux'):
    modulePath = join('/usr', 'share', 'walabot', 'python', 'WalabotAPI.py')

wlbt = load_source('WalabotAPI', modulePath)
wlbt.Init()
codeData = []
codeDataTime = []
def PrintSensorTargets(targets):
    system('cls' if platform == 'win32' else 'clear')
    if targets:
        for i, target in enumerate(targets):
            print('Target #{}:\nx: {}\ny: {}\nz: {}\namplitude: {}\n'.format(
                i + 1, target.xPosCm, target.yPosCm, target.zPosCm,
                target.amplitude))
    else:
        print('No Target Detected')

def savesignalfile(a):
    with open("Data.csv", "w") as datafile:
        datafilewirte = csv.writer(datafile)
        for signalx in zip(a):
            # data = '{}\n'.format(signalx)
            datafilewirte.writerow(a)

def PrintResults(targets):
    system('cls' if platform == 'win32' else 'clear')
    if targets:
        for i, target in enumerate(targets):
            print('Target #{}:\nx: {}\ny: {}\nz: {}\namplitude: {}\n'.format(
                i + 1, target.xPosCm, target.yPosCm, target.zPosCm,
                target.amplitude))
    else:
        print('No Target Detected')

def SensorApp():
    # wlbt.SetArenaR - input parameters
    minInCm, maxInCm, resInCm = 30, 200, 3
    # wlbt.SetArenaTheta - input parameters
    minIndegrees, maxIndegrees, resIndegrees = -15, 15, 3
    # wlbt.SetArenaPhi - input parameters
    minPhiInDegrees, maxPhiInDegrees, resPhiInDegrees = -45, 45, 5
    # Set MTI mode
    mtiMode = False
    # Configure Walabot database install location (for windows)
    wlbt.SetSettingsFolder()
    # 1) Connect : Establish communication with walabot.
    wlbt.ConnectAny()
    # 2) Configure: Set scan profile and arena
    # Set Profile - to Sensor.
    wlbt.SetProfile(wlbt.PROF_SENSOR)
    # Setup arena - specify it by Cartesian coordinates.
    wlbt.SetArenaR(minInCm, maxInCm, resInCm)
    # Sets polar range and resolution of arena (parameters in degrees).
    wlbt.SetArenaTheta(minIndegrees, maxIndegrees, resIndegrees)
    # Sets azimuth range and resolution of arena.(parameters in degrees).
    wlbt.SetArenaPhi(minPhiInDegrees, maxPhiInDegrees, resPhiInDegrees)
    # Moving Target Identification: standard dynamic-imaging filter
    filterType = wlbt.FILTER_TYPE_MTI if mtiMode else wlbt.FILTER_TYPE_NONE
    # filterType = wlbt.FILTER_TYPE_NONE
    wlbt.SetDynamicImageFilter(filterType)
    # wlbt.SetThreshold(100)
    incrementer = 0
    # 3) Start: Start the system in preparation for scanning.
    wlbt.Start()
    t0 = time.time()
    if not mtiMode: # if MTI mode is not set - start calibrartion
        # calibrates scanning to ignore or reduce the signals
        wlbt.StartCalibration()
        while wlbt.GetStatus()[0] == wlbt.STATUS_CALIBRATING:
            wlbt.Trigger()
    while int(time.time()-t0) <=20:
        appStatus, calibrationProcess = wlbt.GetStatus()
        # 5) Trigger: Scan(sense) according to profile and record signals
        # to be available for processing and retrieval.
        wlbt.Trigger()
        # 6) Get action: retrieve the last completed triggered recording
        targets = wlbt.GetSensorTargets()
        rasterImage, _, _, sliceDepth, power = wlbt.GetRawImageSlice()

        # PrintSensorTargets(targets)
        Antenna = wlbt.GetAntennaPairs()
        I = Antenna[11]
        print(Antenna[11])
        signal = wlbt.GetSignal(I)
        print(len(signal[0]))
        codeData.append(signal[0])
        codeDataTime.append(signal[1])
        incrementer+=1
    # 7) Stop and Disconnect.
    wlbt.Stop()
    wlbt.Disconnect()
    print ('pulse',incrementer)
    print('Terminate successfully')
    print(len(codeData))
    # savesignalfile(codeData)
    np.savetxt('Data/Data.out', codeData, delimiter=',', fmt='%f')
    np.savetxt('Data/DataTime.out', codeDataTime, delimiter=',', fmt='%1.9e')

if __name__ == '__main__':
    SensorApp()