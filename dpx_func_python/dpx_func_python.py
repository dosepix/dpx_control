#!/usr/bin/env python
import numpy as np
import time
import serial
import textwrap
from collections import namedtuple
import os
import os.path
import sys
import configparser 
import yaml
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.signal
import scipy.optimize
import scipy.constants
import scipy.special
import scipy.interpolate
import cPickle
import hickle

# GUI 
GUI = False

# Global Flags
DEBUG = False

# Important files
THL_CALIB_FILES = ['THLCalibration/THLCalib_%d.p' % slot for slot in [22, 6, 109]]
BIN_EDGES_FILE = None # 'Dennis1_binEdges.hck'
PARAMS_FILE = 'energyConversion/paramsDict_DPX22_6_109.hck' # 'testCalibFactors.p'

GEN_BIN_EDGES = False
GEN_BIN_EDGES_RANDOM = False
GEN_BIN_EDGES_UNIFORM = True
# BIN_EDGES_RANDOM_FN = 'binEdgesRandom_DPX22_6_109_v2.hck'
BIN_EDGES_RANDOM_FN = 'binEdgesUniform_DPX22_6_109_v2.hck'

if GEN_BIN_EDGES:
    import bin_edges_random as ber

    if PARAMS_FILE.endswith('.p'):
        paramsDict = cPickle.load(open(PARAMS_FILE, 'rb'))
    else:
        paramsDict = hickle.load(PARAMS_FILE)

    binEdgesDict = {}
    for slot in range(1, 3 + 1):
        if GEN_BIN_EDGES_RANDOM:
            binEdges = ber.getBinEdgesRandom(NPixels=256, edgeMin=12, edgeMax=100, edgeOvfw=430, uniform=False)
        elif GEN_BIN_EDGES_UNIFORM:
	    # Generate edges for multiple energy regions
            binEdges = []
	    energy_start, energy_range = 10, 90
	    for idx in range(4):
	    	edges = ber.getBinEdgesUniform(NPixels=256, edgeMin=energy_start + idx*energy_range, edgeMax=energy_start + (idx + 1)*energy_range, edgeOvfw=430)
		binEdges.append( edges )
        binEdgesDict['Slot%d' % slot] = binEdges
    hickle.dump(binEdgesDict, BIN_EDGES_RANDOM_FN)

# BIN_EDGES = hickle.load(BIN_EDGES_RANDOM_FN)
BIN_EDGES = {'Slot1': [12, 18, 21, 24.5, 33.5, 43, 53.5, 66.5, 81.5, 97, 113, 131.5, 151.5, 173, 200.5, 236, 430],
                'Slot2': [12, 17, 31, 40, 45.5, 50.5, 60.5, 68, 91, 102.5, 133, 148, 163, 196, 220, 257, 430],
                'Slot3': [32, 37, 47, 57.6, 68.5, 80, 91.5, 104, 117, 131, 145, 163.5, 183.5, 207.5, 234.5, 269.5, 430]}

def main():
    # Create object of class and establish connection
    dpx = Dosepix('/dev/ttyUSB0', 2e6, 'DPXConfig_22_6_109.conf')
    # dpx = Dosepix('/dev/ttyUSB0', 2e6, 'Configurations/DPXConfig_Dennis1_THL20.conf')
    # dpx = Dosepix('/dev/ttyUSB0', 2e6, 'DPXConfig_22_wCable.conf')
    # dpx.measureTestPulses(1, 0)

    # Change DAC values
    '''
    d = dpx.splitPerihperyDACs(dpx.peripherys + dpx.THLs[0], perc=False)
    d['I_krum'] = 80
    code = dpx.periheryDACsDictToCode(d, perc=False)
    dpx.peripherys = code[:-4]
    dpx.DPXWritePeripheryDACCommand(1, code)
    '''

    # dpx.TPtoToT(slot=1, column='all')
    # dpx.thresholdEqualizationConfig('DPXConfig_22_wLight.conf', I_pixeldac=None, reps=1, intPlot=False, resPlot=True)

    '''
    while True:
        for item in np.reshape( list( dpx.DPXReadDigitalThresholdsCommand(1) ), (256, 16*4) ):
            print ''.join(item)
        print
    '''

    # dpx.getPixelSlopes(1, I_pixeldac=0.)

    # dpx.meanSigmaMatrixPlot('pixelDAC_3.p', (0, 1., 15), (0, 63, 15))
    # dpx.meanSigmaLinePlot('pixelDAC_3.p', (0, 1., 15), (0, 63, 15))

    #while True:
        # print dpx.DPXReadPixelDACCommand(1)
        # dpx.DPXWriteOMRCommand(1, '0000')
        # print dpx.DPXReadPeripheryDACCommand(1)
        # print dpx.DPXReadOMRCommand(1)

    # dpx.ADCWatch(1, ['I_krum', 'Temperature'], cnt=0)
    # dpx.measureADC(1, AnalogOut='V_fbk', perc=False, ADChigh=8191, ADClow=0, ADCstep=1, N=1, fn=None, plot=True)
    # return
    # dpx.ADCWatch(1, ['V_ThA', 'V_TPref_fine', 'V_casc_preamp', 'V_fbk', 'V_TPref_coarse', 'V_gnd', 'I_preamp', 'I_disc1', 'I_disc2', 'V_TPbufout', 'V_TPbufin', 'I_krum', 'I_dac_pixel', 'V_bandgap', 'V_casc_krum', 'V_per_bias', 'V_cascode_bias', 'Temperature', 'I_preamp'], cnt=0)
    # dpx.energySpectrumTHL(1, THLhigh=8000, THLlow=int(dpx.THLs[0], 16) - 500, THLstep=25, timestep=1, intPlot=True)
    # dpx.ADCWatch(1, ['V_fbk', 'V_ThA', 'V_gnd', 'I_krum', 'V_casc_krum', 'I_preamp', 'V_casc_preamp', 'Temperature'], cnt=0)

    # dpx.measureADC(1, AnalogOut='V_cascode_bias', perc=True, ADChigh=0.06, ADClow=0., ADCstep=0.00001, N=1)
    # dpx.ToTtoTHL_pixelDAC(slot=1, THLstep=1, I_pixeldac=0.0001, valueLow=200, valueHigh=200, valueCount=1, energy=False, plot=False)
    # dpx.ToTtoTHL(slot=1, column='all', THLstep=1, valueLow=1.5e3, valueHigh=30e3, valueCount=20, energy=True, plot=False, outFn='ToTtoTHL.p')

    # dpx.energySpectrumTHL(1)
    paramsDict = None # hickle.load(PARAMS_FILE)
    # dpx.measureToT(slot=1, intPlot=True, storeEmpty=False, logTemp=False, paramsDict=paramsDict)

    # while True:
    #     dpx.TPtoToT(slot=1, column=0)

    # dpx.testPulseDosi(1, column='all')
    # dpx.testPulseSigma(1)
    # dpx.testPulseToT(1, 10, column=0, DAC='I_krum', DACRange=range(3, 25) + range(25, 50, 5) + range(50, 120, 10), perc=False)

    # dpx.measureDose(slot=2, measurement_time=0, freq=False, frames=1000, logTemp=False, intPlot=True, conversion_factors=None) # ('conversion_factors/conversionFit_wSim_10.csv', 'conversion_factors/conversionFit_wSim_10.csv')) 

    # dpx.measureIntegration()
    # dpx.temperatureWatch(slot=1, column='all', frames=1000, energyRange=(50.e3, 50.e3), fn='TemperatureToT_DPX22_Ikrum_50keV.hck', intplot=True)

    # dpx.measureTHL(1, fn='THLCalib_6.p', plot=False)
    # dpx.thresholdEqualizationConfig('DPXConfig_22_wCable.conf', I_pixeldac=None, reps=1, intPlot=False, resPlot=True)

    dpx.measurePulseShape(1, column=1)

    # Close connection
    dpx.close()
    return

class Dosepix:
    # === FLAGS ===
    USE_GUI = False

    # === CONSTANTS ===
    __startOfTransmission   = unichr(0x02)
    __endOfTransmission     = unichr(0x03) 

    # = Receiver =
    __receiverDPX1 = '01'   # Dosepix
    __receiverDPX2 = '02'
    __receiverDPX3 = '03'
    __receiverMC = '04'     # Microcontroller
    __receiverVC = '09'     # Voltage Controller
    __receiverHV = '11'     # High Voltage

    # = Subreceiber =
    __subReceiverNone = '000'

    # = Sender =
    __senderPC = '10'

    # = Commands =
    __commandNone = ''
    __commandNoneLength = '0' * 6

    # = HV Commands =
    __HVenable = '000'
    __HVdisable = '001'
    __HVisEnabled = '002'
    __HVsetDAC = '003'
    __HVgetDAC = '004'

    # = VC Commands =
    __VCset3V3 = '000'
    __VCset1V8 = '001'
    __VCgetVoltage = '002'

    # = MC Commands =
    __MCgetVersion = '000'
    __MCLEDenable = '001'
    __MCLEDdisable = '002'
    __MCgetADCvalue = '003'
    __MCsetSPIclock1 = '004'
    __MCsetSPIclock2 = '005'
    __MCsetSPIclock3 = '006'

    # = DPX Commands =
    __DPXwriteOMRCommand = '001' 
    __DPXwriteConfigurationCommand = '002'
    __DPXwriteSingleThresholdCommand = '003'
    __DPXwritePixelDACCommand = '004'
    __DPXwritePeripheryDACCommand = '005'
    __DPXwriteColSelCommand = '006'
    __DPXburnSingleFuseCommand = '007'
    __DPXreadOMRCommand = '008'
    __DPXreadConfigurationCommand = '009'
    __DPXreadDigitalThresholdsCommand = '010'
    __DPXreadPixelDACCommand = '011'
    __DPXreadPeripheryDACCommand = '012'
    __DPXreadColumnTestPulseCommand = '013'
    __DPXreadColSelCommand = '014'
    __DPXreadChipIdCommand = '015'

    __DPXglobalResetCommand = '020'
    __DPXdataResetCommand = '021'

    __DPXreadToTDataDosiModeCommand = '050'
    __DPXreadBinDataDosiModeCommand = '051'
    __DPXreadToTDatakVpModeCommand = '053'
    __DPXreadToTDataIntegrationModeCommand = '054'

    __DPXgeneralTestPulse = '057'
    __DPXreadToTDataInkVpModeWithFixedFrameSizeCommand = '066'
    __DPXgeneralMultiTestPulse = '067'

    __DPXstartStreamingReadout = '068'
    __DPXstopStreamingReadout ='069'

    # = Custom Functions =
    __DPXreadToTDataDosiModeMultiCommand = '090'

    # = CRC =
    __CRC = 'FFFF'

    # = OMR =
    OMROperationModeType = namedtuple("OMROperationMode", "DosiMode TestWakeUp PCMode IntegrationMode")
    __OMROperationMode = OMROperationModeType(
        DosiMode = 0b00 << 22,
        TestWakeUp = 0b01 << 22,
        PCMode = 0b10 << 22,
        IntegrationMode = 0b11 << 22)

    OMRGlobalShutterType = namedtuple("OMRGlobalShutter", "ClosedShutter OpenShutter")
    __OMRGlobalShutter = OMRGlobalShutterType(
        ClosedShutter = 0b0 << 21,
        OpenShutter = 0b1 << 21)

    # Do not use 200 MHz!
    OMRPLLType = namedtuple("OMRPLL", "Direct f16_6MHz f20MHz f25MHz f33_2MHz f50MHz f100MHz")
    __OMRPLL = OMRPLLType(Direct = 0b000 << 18,
        f16_6MHz = 0b001 << 18,
        f20MHz = 0b010 << 18,
        f25MHz = 0b011 << 18,
        f33_2MHz = 0b100 << 18,
        f50MHz = 0b101 << 18,
        f100MHz = 0b110 << 18)

    OMRPolarityType = namedtuple("OMRPolarity", "electron hole")
    __OMRPolarity = OMRPolarityType(
        hole = 0b0 << 17,
        electron = 0b1 << 17)

    OMRAnalogOutSelType = namedtuple("OMRAnalogOutSel", "V_ThA V_TPref_fine V_casc_preamp V_fbk V_TPref_coarse V_gnd I_preamp I_disc1 I_disc2 V_TPbufout V_TPbufin I_krum I_dac_pixel V_bandgap V_casc_krum Temperature V_per_bias V_cascode_bias High_Z")
    __OMRAnalogOutSel = OMRAnalogOutSelType(
        V_ThA = 0b00001 << 12,
        V_TPref_fine = 0b00010 << 12, 
        V_casc_preamp = 0b00011 << 12,
        V_fbk = 0b00100 << 12,
        V_TPref_coarse = 0b00101 << 12,
        V_gnd = 0b00110 << 12,
        I_preamp = 0b00111 << 12,
        I_disc1 = 0b01000 << 12,
        I_disc2 = 0b01001 << 12,
        V_TPbufout = 0b01010 << 12,
        V_TPbufin = 0b01011 << 12,
        I_krum = 0b01100 << 12,
        I_dac_pixel = 0b01101 << 12,
        V_bandgap = 0b01110 << 12,
        V_casc_krum = 0b01111 << 12,
        Temperature = 0b11011 << 12,
        V_per_bias = 0b11100 << 12,
        V_cascode_bias = 0b11101 << 12,
        High_Z = 0b11111 << 12)

    OMRAnalogInSelType = namedtuple("OMRAnalogInSel", "V_ThA V_TPref_fine V_casc_preamp V_fbk V_TPref_coarse V_gnd I_preamp I_disc1 I_disc2 V_TPbufout V_TPbufin I_krum I_dac_pixel V_bandgap V_casc_krum Temperature V_per_bias V_cascode_bias V_no")
    __OMRAnalogInSel = OMRAnalogInSelType(
        V_ThA = 0b00001 << 7,
        V_TPref_fine = 0b00010 << 7, 
        V_casc_preamp = 0b00011 << 7,
        V_fbk = 0b00100 << 7,
        V_TPref_coarse = 0b00101 << 7,
        V_gnd = 0b00110 << 7,
        I_preamp = 0b00111 << 7,
        I_disc1 = 0b01000 << 7,
        I_disc2 = 0b01001 << 7,
        V_TPbufout = 0b01010 << 7,
        V_TPbufin = 0b01011 << 7,
        I_krum = 0b01100 << 7,
        I_dac_pixel = 0b01101 << 7,
        V_bandgap = 0b01110 << 7,
        V_casc_krum = 0b01111 << 7,
        Temperature = 0b11011 << 7,
        V_per_bias = 0b11100 << 7,
        V_cascode_bias = 0b11101 << 7,
        V_no = 0b11111 << 7)

    OMRDisableColClkGateType = namedtuple("OMRDisableColClkGate", "Enabled Disabled")
    __OMRDisableColClkGate = OMRDisableColClkGateType(
        Enabled = 0b0 << 6,
        Disabled = 0b1 << 6)

    # = ConfBits =
    ConfBitsType = namedtuple("ConfBits", "MaskBit TestBit_Analog TestBit_Digital")
    __ConfBits = ConfBitsType(
        MaskBit = 0b1 << 2,
        TestBit_Analog = 0b1 << 1,
        TestBit_Digital = 0b1 << 0)

    def __init__(self, portName, baudRate, configFn):
        self.portName = portName
        self.baudRate = baudRate
        self.configFn = configFn

        if GUI:
            self.getSettingsGUI()
            self.portName, self.baudRate, self.configFn = setSettingsGUI()
        else:
            self.unsetGUI()

        self.getConfigDPX(self.portName, self.baudRate, self.configFn)

    def getSettingsGUI(self):
        serialPorts = getSerialPorts(self)

    def setSettingsGUI(self):
        return None, None, None

    def setGUI(self):
        self.USE_GUI = True

    def unsetGUI(self):
        self.USE_GUI = False

    def getConfigDPX(self, portName, baudRate, configFn):
        self.ser = serial.Serial(self.portName, self.baudRate)
        assert self.ser.is_open, 'Error: Could not establish serial connection!'

        # Read config
        self.peripherys = ''
        self.OMR = ''
        self.THLs = [[]] * 3
        self.confBits = [[]] * 3
        self.pixelDAC = [[]] * 3
        self.binEdges = [[]] * 3
 
        if not os.path.isfile(self.configFn):
            print 'Config file not found. Please run THL equalization first. Using standard values.'
            self.peripherys = 'dc310bc864508230768080ff0064'
            self.OMR = '39ffc0'
            self.THLs = ['153b'] * 3
            self.confBits = [['0']*512] * 3
            self.pixelDAC = [['0']*512] * 3
            self.binEdges = [['0']*1024] * 3
        else:
            self.readConfig(self.configFn)

        self.initDPX()

        # If no THL calibration file is present
        if THL_CALIB_FILES is None:
            print 'Warning: No THL calibration file set! Functions using the THL edges won\'t be usuable'
            return

        # Load THL calibration data
        self.voltCalib, self.THLCalib = [], []
        self.THLEdgesLow, self.THLEdgesHigh = [], []
        self.THLFitParams = []
        self.THLEdges = []
        for THL_CALIB_FILE in THL_CALIB_FILES:
            if not os.path.isfile(THL_CALIB_FILE):
                print 'Need the specified THL Calibration file %s!' % THL_CALIB_FILE
                self.THLEdgesLow.append( None ) # [   0,  662, 1175, 1686, 2193, 2703, 3215, 3728, 4241, 4751, 5263, 5777, 6284, 6784, 7312]
                self.THLEdgesHigh.append( None ) # [ 388,  889, 1397, 1904, 2416, 2927, 3440, 3952, 4464, 4976, 5488, 5991, 6488, 7009, 8190]
                self.THLEdges.append( None )

            else:
                if THL_CALIB_FILE.endswith('.p'):
                    d = cPickle.load(open(THL_CALIB_FILE, 'rb'))
                else:
                    d = hickle.load(THL_CALIB_FILE)

                self.voltCalib.append( np.asarray(d['Volt']) / max(d['Volt']) )
                self.THLCalib.append( np.asarray(d['ADC']) )

                THLLow, THLHigh, THLFitParams = self.THLCalibToEdges(d)
                self.THLEdgesLow.append(THLLow), self.THLEdgesHigh.append(THLHigh), self.THLFitParams.append(THLFitParams)
                print self.THLEdgesLow[-1], self.THLEdgesHigh[-1]
                
                # Combine
                THLEdges = []
                for i in range(len(self.THLEdgesLow[-1])):
                    THLEdges += list( np.arange(self.THLEdgesLow[-1][i], self.THLEdgesHigh[-1][i] + 1) )
                self.THLEdges.append( THLEdges ) 

    def __del__(self):
        self.close()

    # === MAIN FUNCTIONS ===
    def initDPX(self):
        # Start HV
        self.HVSetDac('0000')
        print 'HV DAC set to %s' % self.HVGetDac()

        self.HVActivate()
        # Set voltage to 3.3V
        self.VCVoltageSet3V3()

        # Check if HV is enabled
        print 'Check if HV is activated...',
        # Try five times
        for i in range(5):
            if self.HVGetState():
                print 'done!'
                break
            else:
                self.HVActivate()
        else:
            assert 'HV could not be activated!'

        print 'Voltage set to %s' % self.VCGetVoltage()

        # Disable LED
        self.MCLEDdisable()

        # Wait
        time.sleep(0.5)

        # Global reset
        for i in range(1, 3 + 1):
            # Do three times
            for j in range(3):
                self.DPXGlobalReset(i)
        time.sleep(0.5)

        # = Write Settings =
        for i in range(1, 3 + 1):
            self.DPXWriteConfigurationCommand(i, self.confBits[i-1])
            self.DPXWriteOMRCommand(i, self.OMR)

            # Merge peripheryDACcode and THL value
            self.DPXWritePeripheryDACCommand(i, self.peripherys + self.THLs[i-1])
            print 'Periphery DAC on Slot %d set to: %s' % (i, self.DPXReadPeripheryDACCommand(i))

            self.DPXWritePixelDACCommand(i, self.pixelDAC[i-1])
            print 'Pixel DAC on Slot %d set to: %s' % (i, self.DPXReadPixelDACCommand(i))
        print
        time.sleep(0.5)

        # = Data Reset =
        for i in range(1, 3 + 1):
            self.DPXDataResetCommand(i)

        # = Dummy Readout =
        for i in range(1, 3 + 1):
            self.DPXReadToTDataDosiModeCommand(i)

        # = Bin Edges =
        if BIN_EDGES_FILE is not None:
            print BIN_EDGES_FILE
            if BIN_EDGES_FILE.endswith('.p'):
                binEdgesDict = cPickle.load(open(BIN_EDGES_FILE, 'rb'))
            else:
                binEdgesDict = hickle.load(BIN_EDGES_FILE)

            for slot in range(1, 3 + 1):
                for idx in range(16):
                    self.DPXWriteSingleThresholdCommand(slot, binEdgesDict['Slot%d' % slot][idx])

        else:
            if PARAMS_FILE is None:
                print 'Warning: No parameters for the bin edges specified. Using default values.'

                gray = [0, 1, 3, 2, 6, 4, 5, 7, 15, 13, 12, 14, 10, 11, 9, 8]
                for i in range(1, 3 + 1):
                    for binEdge in range(16):
                        gc = gray[binEdge]
                        binEdges = ('%01x' % gc + '%03x' % (20*binEdge + 15)) * 256
                        self.DPXWriteSingleThresholdCommand(i, binEdges)

            else:
                # TODO: Add energy bins to config file
                if PARAMS_FILE.endswith('.p'):
                    paramsDict = cPickle.load(open(PARAMS_FILE, 'rb'))
                else:
                    paramsDict = hickle.load(PARAMS_FILE)

                self.binEdges = {}
                for slot in range(1, 3 + 1):
                    binEdges = BIN_EDGES['Slot%d' % slot] # [0]
                    self.setBinEdges(slot, paramsDict['Slot%d' % slot], binEdges)

        # = Empty Bins =
        for i in range(1, 3 + 1):
            # Loop over bins
            for col in range(1, 16 + 1):
                self.DPXWriteColSelCommand(i, 16 - col)
                # Dummy readout
                self.DPXReadBinDataDosiModeCommand(i)

    def fillParamDict(self, paramDict):
        # Get mean factors
        meanDict = {}

        if 'h' in paramDict[paramDict.keys()[0]].keys():
            keyList = ['a', 'c', 'b', 'h', 'k', 't']
        else:
            keyList = ['a', 'c', 'b', 't']

        for val in keyList:
            valList = []
            for idx in paramDict.keys():
                valList.append( paramDict[idx][val] )
            valList = np.asarray(valList)
            valList[abs(valList) == np.inf] = np.nan
            meanDict[val] = np.nanmean(valList)

        # Find non existent entries and loop
        for pixel in set(np.arange(256)) - set(paramDict.keys()):
            paramDict[pixel] = {val: meanDict[val] for val in keyList}

        return paramDict

    # Convert the bin edges from energy to ToT and write them to the designated registers
    def setBinEdges(self, slot, paramDict, binEdgesEnergyList):
        # assert len(paramDict) == 256, 'getBinEdges: Number of pixels in paramDict differs from 256!'
        if len(paramDict) != 256:
            paramDict = self.fillParamDict(paramDict)

        # The indices of the bins are specified via the following gray code
        gray = [0, 1, 3, 2, 6, 4, 5, 7, 15, 13, 12, 14, 10, 11, 9, 8]

        # Check if paramDict was made using THL calibration.
        # If so, additional parameters h and k are present in the dict
        if 'h' in paramDict[paramDict.keys()[0]].keys():
            fEnergyConv = True
        else:
            fEnergyConv = False

        binEdgesTotal = []
        binEdgesEnergy = np.asarray( binEdgesEnergyList )

        binEdgesList = []
        nanCnt = 0
        for pixel in sorted(paramDict.keys()):
            params = paramDict[pixel]
            a, b, c, t = params['a'], params['b'], params['c'], params['t']
            if fEnergyConv:
                h, k = params['h'], params['k']
            else:
                h, k = 1, 0

            if len(binEdgesEnergy) > 17: # binEdgesEnergy.shape[1] > 0:
                beEnergy = binEdgesEnergy[pixel]
            else:
                beEnergy = binEdgesEnergy

            # Convert energy to ToT
            if h == 1 and k == 0:
                binEdgesToT = self.energyToToTFitHyp(beEnergy, a, b, c, t)
            else:
                binEdgesToT = self.EnergyToToTSimple(beEnergy, a, b, c, t, h, k)

            # Round the values - do not use floor function as this leads to bias
            binEdgesToT = np.around( binEdgesToT )
            binEdgesToT = np.nan_to_num(binEdgesToT)
            binEdgesToT[binEdgesToT < 0] = 0
            binEdgesToT[binEdgesToT > 4095] = 4095
            binEdgesList.append( binEdgesToT )

        # Transpose matrix to get pixel values
        binEdgesList = np.asarray(binEdgesList).T
        cmdTotal = ''
        for idx, gc in enumerate(gray):
            # print binEdgesList[idx]
            # Construct command
            cmd = ''.join( [('%01x' % gc) + ('%03x' % be) for be in binEdgesList[idx]] )
            self.DPXWriteSingleThresholdCommand(slot, cmd)
            cmdTotal += cmd
        print

	self.binEdges['Slot%d' % slot] = binEdgesList
        # binEdgesTotal.append( binEdgesTotal )
        # self.binEdges = binEdgesTotal

    def ToTtoEnergySimple(self, x, a, b, c, t, h=1, k=0):
        return h * (b + 1./(4 * a) * (2*x + np.pi*c + np.sqrt(16 * a * c * t + (2 * x + np.pi * c)**2))) + k

    def EnergyToToTSimple(self, x, a, b, c, t, h=1, k=0):
        res = np.where(x < b, a*((x - k)/h - b) - c * (np.pi / 2 + t / ((x - k)/h - b)), 0)
        res[res <= 0] = 0
        return res
        
        # Old?
        '''
        res = np.where(x < b*h + k, a*((x - k)/h - b) - c * (np.pi / 2 + t / ((x - k)/h - b)), 0)
        res[res < 0] = 0
        return res
        '''

    # TODO: Correction factor of 16/15 necessary?
    def measureDose(self, slot=1, measurement_time=120, frames=10, freq=False, outFn='doseMeasurement.p', logTemp=False, intPlot=False, conversion_factors=None):
        # Set Dosi Mode in OMR
        # If OMR code is list
        OMRCode = self.OMR
        if not isinstance(OMRCode, basestring):
            OMRCode[0] = 'DosiMode'
        else:
            OMRCode = int(OMRCode, 16) & ~((0b11) << 22)

        # If slot is no list, transform it into one
        if not isinstance(slot, (list,)):
            slot = [slot]

        # Set ADC out to temperature if requested
        if logTemp:
            tempDict = {'temp': [], 'time': []}

            if type(self.OMR) is list:
                OMRCode_ = self.OMRListToHex(self.OMR)
            else:
                OMRCode_ = self.OMR
            OMRCode_ = int(OMRCode_, 16)

            OMRCode_ &= ~(0b11111 << 12)
            OMRCode_ |= getattr(self.__OMRAnalogOutSel, 'Temperature')
            self.DPXWriteOMRCommand(1, hex(OMRCode_).split('0x')[-1])

        # Initial reset 
        for sl in slot:
            self.DPXWriteOMRCommand(sl, OMRCode)
            self.DPXDataResetCommand(sl)
        self.clearBins(slot)

        # Load conversion factors if requested
        if conversion_factors is not None:
            if len(slot) < 3:
                print 'WARNING: Need three detectors to determine total dose!'
            cvLarge, cvSmall = np.asarray( pd.read_csv(conversion_factors[0], header=None) ), np.asarray( pd.read_csv(conversion_factors[1], header=None) )
            doseDict = {'Slot%d' % sl: [] for sl in slot}
            isLarge = np.asarray([self.isLarge(pixel) for pixel in range(256)])

        # Data storage
        outDict = {'Slot%d' % sl: [] for sl in slot}

        # Interactive plot
        if intPlot:
            print 'Warning: plotting takes time, therefore data should be stored as frequency instead of counts!'
            fig, ax = plt.subplots(1, len(slot), figsize=(5*len(slot), 5))
            plt.ion()
            imList = {}

            if len(slot) == 1:
                ax = [ax]

            for idx, a in enumerate(ax):
                imList[slot[idx]] = a.imshow(np.zeros((12, 16)))
                a.set_title('Slot%d' % slot[idx])
                a.set_xlabel('x (px)')
                a.set_ylabel('y (px)')
            plt.show()

        # = START MEASUREMENT =
        try:
            print 'Starting Dose Measurement!'
            print '========================='
            measStart = time.time()
            for c in range(frames):
                startTime = time.time()

                # Measure temperature?
                if logTemp:
                    temp = float(int(self.MCGetADCvalue(), 16))
                    tempDict['temp'].append( temp )
                    tempDict['time'].append( time.time() - measStart )

                # for sl in slot:
                #    self.DPXDataResetCommand(sl)
                #    self.clearBins([sl])
                time.sleep(measurement_time)

                # = Readout =
                doseList = []
                for sl in slot:
                    outList = []
                    showList = []

                    # Loop over columns
                    for col in range(16):
                        measTime = float(time.time() - startTime)
                        # self.DPXDataResetCommand(sl)
                        # self.clearBins([sl])

                        self.DPXWriteColSelCommand(sl, 16 - col)
                        out = np.asarray( self.DPXReadBinDataDosiModeCommand(sl), dtype=float )
                        # out[out >= 2**15] = np.nan

                        # Calculate frequency if requested
                        if freq:
                            out = out / measTime

                        # Bug: discard strange values in matrix
                        # print np.nanmean(out), np.nanstd(out), np.nansum(out)
                        # out[abs(out - np.nanmean(out)) > 3 * np.nanstd(out)] = 0
                        
                        # plt.imshow(out.T[2:-2].T)
                        # plt.show()
                        # print out

                        outList.append( out )
                        showList.append( np.nansum(out, axis=1)[2:-2] )

                    # Append to outDict
                    data = np.asarray(outList)
                    outDict['Slot%d' % sl].append( data )

                    # plt.imshow(showList)
                    # plt.show()

                    if conversion_factors is not None:
                        dLarge, dSmall = np.sum(np.asarray(data).reshape((256, 16))[isLarge], axis=0), np.sum(np.asarray(data).reshape((256, 16))[~isLarge], axis=0)
                        doseLarge = np.nan_to_num( np.dot(dLarge, cvLarge[:,(sl-1)]) / (time.time() - measStart) )
                        doseSmall = np.nan_to_num( np.dot(dSmall, cvSmall[:,(sl-1)]) / (time.time() - measStart) )
                        print 'Slot %d: %.2f uSv/s (large), %.2f uSv/s (small)' % (sl, doseLarge, doseSmall)
                        doseDict['Slot%d' % sl].append( (doseLarge, doseSmall) )

                    if intPlot:
                        print showList
                        imList[sl].set_data( showList )
                        # fig.canvas.flush_events()

                        plt.imshow(showList)
                        plt.show()
                if conversion_factors is not None:
                    print 'Total dose: %.2f uSv/s' % (np.sum([np.sum(doseDict['Slot%d' % sl]) for sl in slot]))
                    print

                if intPlot:
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    # plt.show()

            # Loop finished
            self.pickleDump(outDict, outFn)
            if logTemp:
                self.pickleDump(tempDict, '%s_temp' % outFn.split('.p')[0] + '.p')
            if conversion_factors:
                self.pickleDump(doseDict, '%s_dose' % outFn.split('.p')[0] + '.p')

        except (KeyboardInterrupt, SystemExit):
            # Store data and plot in files
            print 'KeyboardInterrupt-Exception: Storing data!'
            print 'Measurement time: %.2f min' % ((time.time() - measStart) / 60.)
            self.pickleDump(outDict, outFn)
            if logTemp:
                self.pickleDump(tempDict, '%s_temp' % outFn.split('.p')[0] + '.p')
            if conversion_factors:
                self.pickleDump(doseDict, '%s_dose' % outFn.split('.p')[0] + '.p')
            raise

        # Reset OMR
        for sl in slot:
            self.DPXWriteOMRCommand(sl, self.OMR)

    # If binEdgesDict is provided, perform shifting of energy regions between
    # frames. I.e. the measured region is shifted in order to increase the
    # total energy region.
    def measureDoseEnergyShift(self, slot=1, measurement_time=120, frames=10, freq=False, outFn='doseMeasurement.p', logTemp=False, intPlot=False, binEdgesDict=None, paramsDict=None):
        # Set Dosi Mode in OMR
        # If OMR code is list
        OMRCode = self.OMR
        if not isinstance(OMRCode, basestring):
            OMRCode[0] = 'DosiMode'
        else:
            OMRCode = int(OMRCode, 16) & ~((0b11) << 22)

        # If slot is no list, transform it into one
        if not isinstance(slot, (list,)):
            slot = [slot]

        # Set ADC out to temperature if requested
        if logTemp:
            tempDict = {'temp': [], 'time': []}

            if type(self.OMR) is list:
                OMRCode_ = self.OMRListToHex(self.OMR)
            else:
                OMRCode_ = self.OMR
            OMRCode_ = int(OMRCode_, 16)

            OMRCode_ &= ~(0b11111 << 12)
            OMRCode_ |= getattr(self.__OMRAnalogOutSel, 'Temperature')
            self.DPXWriteOMRCommand(1, hex(OMRCode_).split('0x')[-1])

        for sl in slot:
            self.DPXWriteOMRCommand(sl, OMRCode)
            self.DPXDataResetCommand(sl)

        # Initial reset
        self.clearBins(slot)

        # Data storage
        outDict = {'Slot%d' % sl: {'Region%d' % reg: [] for reg in range(4)} for sl in slot}

        # = START MEASUREMENT =
        try:
            print 'Starting Dose Measurement!'
            print '========================='
            region_stack = {sl: [0] for sl in slot}
            region_idx = 0
            measStart = time.time()
            for c in range(frames):
                # Measure temperature?
                if logTemp:
                    temp = float(int(self.MCGetADCvalue(), 16))
                    tempDict['temp'].append( temp )
                    tempDict['time'].append( time.time() - measStart )

                time.sleep(measurement_time)

                for sl in slot:
                    outList = []
                    # Loop over columns
                    for col in range(16):
                        self.DPXWriteColSelCommand(sl, 16 - col)
                        out = np.asarray( self.DPXReadBinDataDosiModeCommand(sl), dtype=float )
                        print np.nanmean(out), np.nanstd(out), np.nansum(out)
                        outList.append( out )
                    print

                    # Append to outDict
                    data = np.asarray(outList)
                    outDict['Slot%d' % sl]['Region%d' % region_idx].append( data )

                    # = Measurement time handler =
                    if len(region_stack[sl]) == 0:
                        region_idx = 0
                    else:
                        # Get current energy region
                        region_idx = region_stack[sl].pop(0)

                    if region_idx == 3:
                        continue

                    # Get number of counts in frame and overflow bins
                    N_frame = np.sum(data[1:])
                    N_ovfw = np.sum(data[0])

                    if N_ovfw == 0 or N_frame == 0:
                        # region_stack[sl].append( region_idx )
                        continue

                    # Calculate fraction
                    p_next = N_ovfw / N_frame
                    print 'p_next =', p_next

                    if p_next <= 0.01:
                        print 'Continue'
                        continue

                    # Select next frames
                    if p_next < 1:
                        if int(1. / p_next) == 0:
                            region_stack[sl].append( region_idx )
                        region_stack[sl] += [region_idx + 1] * int(np.around(1. / p_next))
                    else:
                        continue
                        # region_stack[sl] += [region_idx] * int(np.around(p_next))

                    print region_stack[sl]
                    print region_stack[sl][0], region_idx
                    if region_stack[sl][0] != region_idx:
                        self.setBinEdges(sl, paramsDict['Slot%d' % sl], binEdgesDict['Slot%d' % sl][region_stack[sl][-1]])
                        self.clearBins(slot)

            # Loop finished
            if logTemp:
                self.pickleDump(tempDict, '%s_temp' % outFn.split('.p')[0] + '.p')

            self.pickleDump(outDict, outFn)
            self.pickleDump(self.binEdges, '%s_binEdges' % outFn.split('.hck')[0] + '.hck')

        except (KeyboardInterrupt, SystemExit):
            # Store data and plot in files
            print 'KeyboardInterrupt-Exception: Storing data!'
            print 'Measurement time: %.2f min' % ((time.time() - measStart) / 60.)
            if logTemp:
                self.pickleDump(tempDict, '%s_temp' % outFn.split('.p')[0] + '.p')

            self.pickleDump(outDict, outFn)
            self.pickleDump(self.binEdges, '%s_binEdges' % outFn.split('.p')[0] + '.p')
            raise

        # Reset OMR
        for sl in slot:
            self.DPXWriteOMRCommand(sl, self.OMR)

    def clearBins(self, slot):
        # Clear bins
        # Only call this at start since it takes a long time
        for i in slot:
            for k in range(3):
                self.DPXDataResetCommand(i)
                self.DPXReadToTDataDosiModeCommand(i)

            for col in range(16):
                self.DPXWriteColSelCommand(i, 16 - col)
                self.DPXReadBinDataDosiModeCommand(i)

    def measureIntegration(self, slot=1, measurement_time=10, frames=10, outFn='integrationMeasurement.p'):
        # Set Integration Mode in OMR
        # If OMR code is list
        OMRCode = self.OMR
        if not isinstance(OMRCode, basestring):
            OMRCode[0] = 'IntegrationMode'
        else:
            OMRCode = int(OMRCode, 16) & ((0b11) << 22)

        # If slot is no list, transform it into one
        if not isinstance(slot, (list,)):
            slot = [slot]

        for sl in slot:
            self.DPXWriteOMRCommand(sl, OMRCode)

        outDict = {'Slot%d' % sl: [] for sl in slot}

        # = START MEASUREMENT =
        print 'Starting ToT Integral Measurement!'
        print '=================================='
        for c in range(frames):
            # Start frame 
            OMRCode[1] = 'ClosedShutter'
            for sl in slot:
                # Reset data registers
                self.DPXDataResetCommand(sl)
                self.DPXWriteOMRCommand(slot, OMRCode)

            # Wait specified time
            time.sleep(measurement_time)

            # Read data and end frame
            OMRCode[1] = 'OpenShutter'
            for sl in slot:
                data = self.DPXReadToTDataIntegrationModeCommand(slot)
                # print data
                outDict['Slot%d' % slot].append( data.flatten() )
                self.DPXWriteOMRCommand(sl, OMRCode)

        if outFn:
            self.pickleDump(outDict, outFn)

    # ToT is measured in endless loop. The data is written to file after 'cnt' frames 
    # were processed. Additionally, Keyboard Interrupts are caught in order
    # to store data afterwards. 
    # If storeEmpty flag is set, empty frames are stored, otherwise discarded.
    # If logTemp flag is set, current chip temperature is stored with each frame.
    def measureToT(self, slot=1, outDir='ToTMeasurement/', cnt=10000, storeEmpty=False, logTemp=False, paramsDict=None, intPlot=False):
        # Set Dosi Mode in OMR
        # If OMR code is list
        OMRCode = self.OMR
        print self.OMRListToHex(OMRCode)
        if not isinstance(OMRCode, basestring):
            OMRCode[0] = 'DosiMode'
        else:
            OMRCode = '%04x' % ((int(OMRCode, 16) & ~((0b11) << 22)) | (0b10 << 22))
        print self.OMRListToHex(OMRCode)

        # Check which slots to read out
        if isinstance(slot, int):
            slotList = [slot]
        elif not isinstance(slot, basestring):
            slotList = slot

        # Set mode in slots
        for slot in slotList:
            self.DPXWriteOMRCommand(slot, OMRCode)

        # Set ADC out to temperature if requested
        if logTemp:
            tempDict = {'temp': [], 'time': []}

            if type(self.OMR) is list:
                OMRCode_ = self.OMRListToHex(self.OMR)
            else:
                OMRCode_ = self.OMR
            OMRCode_ = int(OMRCode_, 16)

            OMRCode_ &= ~(0b11111 << 12)
            OMRCode_ |= getattr(self.__OMRAnalogOutSel, 'Temperature')
            self.DPXWriteOMRCommand(slot, hex(OMRCode_).split('0x')[-1])

        # GUI
        '''
        if self.USE_GUI:
            bins = np.arange(0, 400, 1)
            histData = np.zeros(len(bins)-1)
            yield bins, histData
        '''

        # Init plot
        if intPlot:
            plt.ion()

            fig, ax = plt.subplots()

            # Create empty axis
            line, = ax.plot(np.nan, np.nan, color='k') # , where='post')
            ax.set_xlabel('ToT')
            ax.set_ylabel('Counts')
            # ax.set_yscale("log", nonposy='clip')
            plt.grid()

            # Init bins and histogram data
            if paramsDict is not None:
                bins = np.linspace(20, 100, 300)
            else:
                bins = np.arange(0, 400, 1)
            histData = np.zeros(len(bins)-1)

            ax.set_xlim(min(bins), max(bins))

        # Check if output directory exists
        outDir = self.makeDirectory(outDir)
        outFn = outDir.split('/')[0] + '.p'

        # Initial reset 
        for slot in slotList:
            self.DPXDataResetCommand(slot)

        # For KeyboardInterrupt exception
        print 'Starting ToT Measurement!'
        print '========================='
        measStart = time.time()
        try:
            while True:
                ToTDict = {'Slot%d' % slot: [] for slot in slotList}
                # if paramsDict is not None:
                #     energyDict = {'Slot%d' % slot: [] for slot in slotList}

                c = 0
                startTime = time.time()
                if intPlot or self.USE_GUI:
                    dataPlot = []

                while c <= cnt:
                    for slot in slotList:
                        # Read data
                        data = self.DPXReadToTDataDosiModeCommand(slot)

                        # Reset data registers
                        self.DPXDataResetCommand(slot)

                        # print np.median( data )
                        # data = self.DPXReadToTDataDosiModeMultiCommand(slot)
                        # print data[256:]
                        # print data[:256]
                        # print
                        # data = data[256:]

                        data = data.flatten()
                        # Discard empty frame?
                        if not np.any(data) and not storeEmpty:
                            continue

                        if paramsDict is not None:
                            energyData = []
                            p = paramsDict['Slot%d' % slot]
                            for pixel in range(256):
                                if pixel not in p.keys():
                                    energyData.append(np.nan)
                                    continue

                                pPixel = p[pixel]
                                if len( pPixel.keys() ) == 6:
                                    energyData.append( self.ToTtoEnergySimple(data[pixel], 
                                        pPixel['a'], pPixel['b'], pPixel['c'],
                                        pPixel['t'], pPixel['h'], pPixel['k']) )
                                else:
                                    energyData.append( self.ToTtoEnergySimple(data[pixel], 
                                        pPixel['a'], pPixel['b'], pPixel['c'],
                                        pPixel['t']) )
                            data = np.asarray(energyData)
                            print data

                        # Remove overflow
                        # data[data >= 4096] -= 4096

                        if intPlot or self.USE_GUI:
                            data_ = np.asarray(data[[self.isLarge(pixel) for pixel in range(256)]])
                            data_ = data_[~np.isnan(data_)]
                            dataPlot += data_.tolist()

                        ToTDict['Slot%d' % slot].append( np.nan_to_num(data).tolist() )

                    # Measure temperature?
                    if logTemp:
                        temp = float(int(self.MCGetADCvalue(), 16))
                        tempDict['temp'].append( temp )
                        tempDict['time'].append( time.time() - measStart )

                    if c > 0 and not c % 100:
                        print '%.2f Hz' % (100./(time.time() - startTime))
                        startTime = time.time()

                    # Increment loop counter
                    c += 1

                    # Update plot every 100 iterations
                    if intPlot or self.USE_GUI:
                        if not c % 10:
                            dataPlot = np.asarray(dataPlot)
                            # Remove empty entries
                            dataPlot = dataPlot[dataPlot > 0]

                            hist, bins_ = np.histogram(dataPlot, bins=bins)
                            histData += hist
                            dataPlot = []

                            '''
                            if self.USE_GUI:
                                yield bins_, histData
                            '''

                            if intPlot:
                                line.set_xdata(bins_[:-1])
                                line.set_ydata(histData)

                                # Update plot scale
                                ax.set_ylim(1, 1.1 * max(histData))
                                fig.canvas.draw()

                # Loop finished
                self.pickleDump(ToTDict, '%s/%s' % (outDir, outFn))
                if logTemp:
                    self.pickleDump(tempDict, '%s/%s_temp' % (outDir, outFn.split('.p')[0]) + '.p')
                print 'Measurement time: %.2f min' % ((time.time() - measStart) / 60.)
                for key in ToTDict.keys():
                    print 'Slot%d: %d events' % (slot, len(np.asarray(ToTDict[key]).flatten()) / 256.)

        except (KeyboardInterrupt, SystemExit):
            # Store data and plot in files
            print 'KeyboardInterrupt-Exception: Storing data!'
            print 'Measurement time: %.2f min' % ((time.time() - measStart) / 60.)
            for key in ToTDict.keys():
                print 'Slot%d: %d events' % (slot, len(np.asarray(ToTDict[key]).flatten()) / 256.)

            self.pickleDump(ToTDict, '%s/%s' % (outDir, outFn))
            if logTemp:
                self.pickleDump(tempDict, '%s/%s_temp' % (outDir, outFn.split('.p')[0]) + '.p')
            raise

        # Reset OMR
        for slot in slotList:
            self.DPXWriteOMRCommand(slot, self.OMR)

    def testPulseInit(self, slot, column=0):
        # Set Polarity to hole and Photon Counting Mode in OMR
        OMRCode = self.OMR
        if not isinstance(self.OMR, basestring):
            OMRCode[3] = 'hole'
        else:
            OMRCode = '%04x' % (int(self.OMR, 16) & ~(1 << 17))

        if not isinstance(self.OMR, basestring):
            OMRCode[0] = 'DosiMode'
        else:
            OMRCode = '%04x' % ((int(self.OMR, 16) & ~((0b11) << 22)) | (0b10 << 22))
        
        self.DPXWriteOMRCommand(slot, OMRCode)

        if column == 'all':
            columnRange = range(16)
        elif not isinstance(column, basestring):
            if isinstance(column, int):
                columnRange = [column]
            else:
                columnRange = column

        return columnRange

    def testPulseClose(self, slot):
        # Reset ConfBits
        self.DPXWriteConfigurationCommand(slot, self.confBits[slot-1])

        # Reset peripheryDACs
        self.DPXWritePeripheryDACCommand(slot, self.peripherys + self.THLs[slot-1])

    def maskBitsColumn(self, slot, column):
        # Set ConfBits to use test charge on preamp input if pixel is enabled
        # Only select one column at max
        confBits = np.zeros((16, 16))
        confBits.fill(getattr(self.__ConfBits, 'MaskBit'))
        confBits[column] = [getattr(self.__ConfBits, 'TestBit_Analog')] * 16
        # print confBits

        # confBits = np.asarray( [int(num, 16) for num in textwrap.wrap(self.confBits[slot-1], 2)] )
        # confBits[confBits != getattr(self.__ConfBits, 'MaskBit')] = getattr(self.__ConfBits, 'TestBit_Analog')

        self.DPXWriteConfigurationCommand(slot, ''.join(['%02x' % conf for conf in confBits.flatten()]))

    def measureTestPulses(self, slot, column='all'):
        columnRange = self.testPulseInit(slot, column=column)
        for column in columnRange:
            self.maskBitsColumn(slot, column)

            while True:
                self.DPXDataResetCommand(slot)
                # self.DPXGeneralTestPulse(slot, 1000)
                print self.DPXReadToTDataDosiModeCommand(slot)[column]

    def measurePulseShape(self, slot, column='all'):
        columnRange = self.testPulseInit(slot, column=column)
        N = 10

        # Set energy
        energy = 10.e3
        DACVal = self.getTestPulseVoltageDAC(slot, energy, True)
        print 'Energy DAC:', DACVal
        # self.DPXWritePeripheryDACCommand(slot, DACval)

        # Define THLRange
        THLhigh = int(self.THLs[slot-1], 16)
        THLstep = 1
        THLstart = THLhigh - 1000
        THLstop = THLhigh

        THLRange = np.asarray(self.THLEdges[slot - 1])
        THLstop_ = THLhigh if THLstop > THLhigh else THLstop
        THLRange = THLRange[np.logical_and(THLRange >= THLstart, THLRange <= THLstop_)]

        for column in columnRange:
            self.maskBitsColumn(slot, column)

            ToTList, voltList = [], []
            for cnt, THL in enumerate(THLRange[::THLstep]):

                # Set new THL
                self.DPXWritePeripheryDACCommand(slot, DACVal[:-4] + '%04x' % THL)
                self.statusBar(float(cnt) / len(THLRange[::THLstep]) * 100 + 1)

                dataList = []
                for n in range(N):
                    self.DPXDataResetCommand(slot)
                    self.DPXGeneralTestPulse(slot, 1000)
                    dataList.append( self.DPXReadToTDataDosiModeCommand(slot)[column] )

                mean = np.mean( dataList, axis=0 )
                if np.any(mean):
                    print mean
                    ToTList.append( mean )
                    voltList.append( self.getVoltFromTHLFit(THL, slot) )

            plt.plot(ToTList, -np.asarray(voltList))
            plt.show()

        self.testPulseClose(slot)

    def testPulseSigma(self, slot, column=[0, 1, 2], N=100):
        columnRange = self.testPulseInit(slot, column=column)

        fig, ax = plt.subplots()

        energyRange = np.linspace(5e3, 100e3, 50)
        meanListEnergy, sigmaListEnergy = [], []

        for energy in energyRange:
            DACval = self.getTestPulseVoltageDAC(slot, energy, True)
            print 'Energy DAC:', DACVal
            self.DPXWritePeripheryDACCommand(slot, DACval)

            meanList, sigmaList = [], []
            print 'E = %.2f keV' % (energy / 1000.)
            for k, column in enumerate(columnRange):
                self.maskBitsColumn(slot, column)

                # Record ToT spectrum of pulse
                dataList = []
                for n in range(N):
                    self.statusBar(float(k*N + n)/(len(columnRange) * N) * 100 + 1)
                    self.DPXDataResetCommand(slot)
                    self.DPXGeneralTestPulse(slot, 100)
                    dataList.append( self.DPXReadToTDataDosiModeCommand(slot)[column] )

                dataList = np.asarray(dataList)
                for pixel in range(16):
                    # ax.hist(dataList[:,pixel], bins=30)
                    meanList.append( np.mean(dataList[:,pixel]) )
                    sigmaList.append( np.std(dataList[:,pixel]) )
                    # plt.show()

            print
            print meanList
            meanListEnergy.append( meanList )
            sigmaListEnergy.append( sigmaList )

        meanListEnergy = np.asarray(meanListEnergy)
        sigmaListEnergy = np.asarray(sigmaListEnergy)
        # for pixel in range(len(meanListEnergy[0])):

        plt.errorbar(energyRange/1000., [np.mean(item) for item in sigmaListEnergy], xerr=energyRange/1000.*0.18, yerr=np.asarray([np.std(item) for item in sigmaListEnergy])/np.sqrt(len(columnRange)*16), marker='x', ls='')

        # Fit curve
        #popt, pcov = scipy.optimize.curve_fit(self.energyToToTFitAtan, energyRange/1000., [np.mean(item) for item in meanListEnergy], p0=(3, 8, 50, 1))
        #perr = np.sqrt( np.diag(pcov) )
        #print popt, perr/popt*100

        #plt.plot(energyRange/float(1000), self.energyToToTFitAtan(energyRange/float(1000), *popt))

        plt.xlabel('Energy (keV)')
        plt.ylabel('Standard deviation (ToT)')
        plt.show()

        self.testPulseClose(slot)

    def testPulseToT(self, slot, length, column='all', paramOutFn='testPulseParams.p', DAC=None, DACRange=None, perc=True):
        columnRange = self.testPulseInit(slot, column=column)

        # Measure multiple test pulses and return the average ToT
        energyRange = np.asarray(list(np.linspace(1.e3, 20.e3, 30)) + list(np.linspace(20e3, 100e3, 20)))
        energyRangeFit = np.linspace(1.e3, 100e3, 1000)
        # energyRange = np.arange(509, 0, -10)

        outDict = {'energy': energyRange, 'DAC': DACRange}

        # Set DAC value
        if DAC is not None:
            if DACRange is not None:
                outList = []
                for DACVal in DACRange:
                    d = self.splitPerihperyDACs(self.peripherys + self.THLs[0], perc=perc)
                    d[DAC] = DACVal
                    code = self.periheryDACsDictToCode(d, perc=perc)
                    self.peripherys = code[:-4]
                    self.DPXWritePeripheryDACCommand(slot, code)

                    dataArray = []
                    for column in  columnRange:
                        self.maskBitsColumn(slot, column)

                        ToTValues = []
                        for energy in energyRange:
                            # DACval = self.getTestPulseVoltageDAC(slot, energy)
                            DACval = self.getTestPulseVoltageDAC(slot, energy, True)
                            self.DPXWritePeripheryDACCommand(slot, DACval)

                            data = []
                            for i in range(10):
                                self.DPXDataResetCommand(slot)
                                self.DPXGeneralTestPulse(slot, 1000)
                                data.append( self.DPXReadToTDataDosiModeCommand(slot)[column] )
                            data = np.mean(data, axis=0)

                            # Filter zero entries
                            # data = data[data != 0]

                            ToTValues.append( data )

                        ToTValues = np.asarray( ToTValues )
                        print ToTValues
                        dataArray.append( ToTValues.T )
                        print dataArray

                        if False:
                            x = energyRange/float(1000)
                            for row in range(16):
                                y = []
                                for ToTValue in ToTValues:
                                    # print ToTValue
                                    y.append( ToTValue[row])

                                # Fit curve
                                try:
                                    popt, pcov = scipy.optimize.curve_fit(self.energyToToTFitAtan, x, y, p0=(3, 8, 50, 1))
                                    perr = np.sqrt( np.diag(pcov) )
                                except:
                                    popt = (3, 8, 50, 1)
                                    perr = np.zeros(len(popt))

                                # Store in dictionary
                                a, b, c, t = popt
                                paramOutDict['a'].append( a )
                                paramOutDict['b'].append( b )
                                paramOutDict['c'].append( c )
                                paramOutDict['t'].append( t )

                                print popt, perr/popt*100

                                plt.plot(x, y, marker='x')
                                plt.plot(energyRangeFit/float(1000), self.energyToToTFitAtan(energyRangeFit/float(1000), *popt))
                                plt.show()

                    outList.append(np.vstack(dataArray))

        outDict['data'] = outList

        # Dump to file
        if paramOutFn:
            cPickle.dump(outDict, open(paramOutFn, 'wb'))
        return

        # Plot
        plt.xlabel('Energy (keV)')
        plt.ylabel('ToT')
        plt.grid()
        plt.show()

        self.testPulseClose(slot)
        return

    def testPulseDosi(self, slot=1, column='all'):
        columnRange = self.testPulseInit(slot, column=column)

        # Select DosiMode
        '''
        OMRCode = self.OMR
        if not isinstance(OMRCode, basestring):
            OMRCode[0] = 'DosiMode'
        else:
            OMRCode = '%04x' % (int(OMRCode, 16) & ~((0b11) << 22))

        self.DPXWriteOMRCommand(slot, OMRCode)
        '''

        # Specify energy range
        energyRange = [25.e3] # list(np.linspace(20e3, 100e3, 10))

        ToTValues = []
        # Loop over energies
        for energy in energyRange:
            DACval = self.getTestPulseVoltageDAC(slot, energy, True)
            self.DPXWritePeripheryDACCommand(slot, DACval)

            # Clear data
            self.clearBins([slot])

            # Loop over columns
            outList = []
            for column in columnRange:
                self.maskBitsColumn(slot, 15 - column)
                # self.DPXWriteColSelCommand(slot, column)

                # Dummy readouts
                # self.DPXReadBinDataDosiModeCommand(slot)

                # Generate pulses
                for i in range(10):
                    # time.sleep(1)
                    self.DPXGeneralTestPulse(slot, 1000)

                # Read from column
                self.DPXWriteColSelCommand(slot, column)
                out = self.DPXReadBinDataDosiModeCommand(slot)
                print np.mean(out), np.std(out)
                print out
                outList.append( out )

            dataMatrix = np.rec.fromarrays( outList )
            for i in range(len(dataMatrix)):
                print np.asarray([list(entry) for entry in dataMatrix[i]])
                plt.imshow(np.asarray([list(entry) for entry in dataMatrix[i]]))
                plt.show()

        self.testPulseClose(slot)
        return

    def energyToToTFitAtan(self, x, a, b, c, d):
        return np.where(x > b, a*(x - b) + c*np.arctan((x - b)/d), 0)

    def energyToToTFitHyp(self, x, a, b, c, d):
        return np.where(x > d, a*x + b + float(c)/(x - d), 0)

    def ToTtoTHL_pixelDAC(self, slot=1, THLstep=1, I_pixeldac=0.1, valueLow=0, valueHigh=460, valueCount=460, energy=False, plot=False):

        pixelDACs = ['00', '3f']

        # Set I_pixeldac
        if I_pixeldac:
            dPeripherys = self.splitPerihperyDACs(self.peripherys + self.THLs[0], perc=True)
            dPeripherys['I_pixeldac'] = I_pixeldac
            code = self.periheryDACsDictToCode(dPeripherys, perc=True)
            self.peripherys = code[:-4]
            self.DPXWritePeripheryDACCommand(slot, code)

        dMeas = {}
        slopeDict = {'slope': []}

        for column in range(16):
            for pixelDAC in pixelDACs:
                # Set pixel DAC values
                self.DPXWritePixelDACCommand(slot, pixelDAC*256, file=False)

                d = self.ToTtoTHL(slot=slot, column=column, THLstep=THLstep, valueLow=valueLow, valueHigh=valueHigh, valueCount=valueCount, energy=energy, plot=plot)

                # Get THL positions
                dMeas[pixelDAC] = d['THL']

            # Get slope
            slope = (dMeas['00'] - dMeas['3f']) / 63.

            print 'Pixel slope for column %d:' % column
            print slope
            print 'Mean slope: %.2f +/- %.2f' % (np.mean(slope), np.std(slope))

            slopeDict['slope'].append( slope )

        self.pickleDump(slopeDict, 'pixelSlopes.p')

        return slope

    def TPtoToT(self, slot=1, column=0, outFn='TPtoToT.p'):
        # Description: Generate test pulses and measure ToT afterwards
        #              to match test pulse voltage and ToT value.

        # Measure temperature
        if type(self.OMR) is list:
            OMRCode_ = self.OMRListToHex(self.OMR)
        else:
            OMRCode_ = self.OMR
        OMRCode_ = int(OMRCode_, 16)

        OMRCode_ &= ~(0b11111 << 12)
        OMRCode_ |= getattr(self.__OMRAnalogOutSel, 'Temperature')
        self.DPXWriteOMRCommand(slot, hex(OMRCode_).split('0x')[-1])

        TList = []
        for i in range(100):
            TList.append( float(int(self.MCGetADCvalue(), 16)) )
        T = np.mean(TList)

        # Number of test pulses for ToT measurement
        NToT = 10

        # Test pulse voltage range
        # TPvoltageRange = list(reversed(np.arange(300, 490, 1))) # list(reversed(np.arange(490, 512, 1)))
        TPvoltageRange = list(reversed(np.arange(440, 512, 1))) + list(reversed(np.arange(250, 440, 5)))

        # Store results per column in dict
        resDict = {}

        # Activate DosiMode and select columns
        columnRange = self.testPulseInit(slot, column=column)

        # Loop over columns
        for column in columnRange:
            # Select column
            self.maskBitsColumn(slot, column)

            # Store results in lists
            ToTListTotal, ToTErrListTotal = [], []

            # Loop over test pulse voltages
            for val in TPvoltageRange:
                # Set test pulse voltage
                DACval = self.getTestPulseVoltageDAC(slot, val, energy=False)
                self.DPXWritePeripheryDACCommand(slot, DACval)

                # Generate test pulses
                dataList = []
                for i in range(NToT):
                    self.DPXDataResetCommand(slot)
                    self.DPXGeneralTestPulse(slot, 1000)
                    data = self.DPXReadToTDataDosiModeCommand(slot)[column]
                    dataList.append( data )

                # Determine corresponding ToT value
                testPulseToT = np.mean(dataList , axis=0)
                testPulseToTErr = np.std(dataList, axis=0) / np.sqrt( NToT )
                print 'Test Pulse ToT:'
                print testPulseToT, np.std(dataList, axis=0)

                # Store in lists
                ToTListTotal.append( testPulseToT )
                ToTErrListTotal.append( testPulseToTErr )

            resDict[column] = {'ToT': ToTListTotal, 'ToTErr': ToTErrListTotal, 'volt': TPvoltageRange}

            '''
            for i in range(16):
                plt.errorbar(TPvoltageRange, np.asarray(ToTListTotal).T[i], yerr=np.asarray(ToTErrListTotal).T[i], marker='x')
            plt.show()
            '''

        outFn = outFn.split('.')[0] + '_T%d.hck' % T
        self.pickleDump(resDict, outFn)

    def ToTtoTHL(self, slot=1, column=0, THLstep=1, valueLow=0, valueHigh=460, valueCount=460, energy=False, plot=False, outFn='ToTtoTHL.p'):
        # Description: Generate test pulses and measure their ToT
        #              values. Afterwards, do a THL-scan in order to
        #              find the corresponding THL value. Repeat 
        #              multiple times to find the correlation between
        #              ToT and THL
        # Parameters: 
        #   - energy: if set, iteration is performed over energies.
        #             Else, Test Pulse DAC values are modified directly.

        # Set low level to zero and high level to noise limit
        # for the coarse measurement
        THLlow = 0
        THLhigh = int(self.THLs[slot-1], 16)

        # Set AnalogOut to V_ThA
        OMRCode = self.OMR
        if not isinstance(OMRCode, basestring):
            OMRCode[4] = 'V_ThA'
        else:
            OMRCode &= ~(0b11111 << 12)
            OMRCode |= getattr(self.__OMRAnalogOutSel, 'V_ThA')
        self.DPXWriteOMRCommand(slot, OMRCode)

        valueRange = np.linspace(valueLow, valueHigh, valueCount)

        # Number of test pulses for ToT measurement
        NToT = 100

        # Store results per column in dict
        resDict = {}

        # Activate DosiMode and select columns
        columnRange = self.testPulseInit(slot, column=column)

        for column in columnRange:
            THLstart = THLhigh - 400
            THLstop = THLhigh

            # Select column
            self.maskBitsColumn(slot, column)

            # Store results in lists
            ToTListTotal, ToTErrListTotal = [], []
            THLListTotal, THLErrListTotal = [], []

            # Loop over test pulse energies
            for val in valueRange:
                # Set test pulse energy
                if energy:
                    print 'Energy: %.2f keV' % (val/1000.)
                else:
                    val = int(val)
                    print 'DAC: %04x' % val

                # Activates DosiMode and selects columns
                columnRange = self.testPulseInit(slot, column=column)

                # Activate DosiMode
                '''
                if not isinstance(OMRCode, basestring):
                    OMRCode[0] = 'DosiMode'
                else:
                    OMRCode = (int(OMRCode, 16) & ((0b11) << 22))
                self.DPXWriteOMRCommand(slot, OMRCode)
                '''
                
                # Set test pulse energy
                DACval = self.getTestPulseVoltageDAC(slot, val, energy)
                self.DPXWritePeripheryDACCommand(slot, DACval)

                # Generate test pulses
                dataList = []
                for i in range(NToT):
                    self.DPXDataResetCommand(slot)
                    self.DPXGeneralTestPulse(slot, 1000)
                    data = self.DPXReadToTDataDosiModeCommand(slot)[column]
                    dataList.append( data )

                # Determine corresponding ToT value
                testPulseToT = np.mean( dataList , axis=0)
                testPulseToTErr = np.std(dataList, axis=0) / np.sqrt( NToT )
                print 'Test Pulse ToT:'
                print testPulseToT

                # Store in lists
                ToTListTotal.append( testPulseToT )
                ToTErrListTotal.append( testPulseToTErr )

                # Set PCMode
                # WARNING: 8bit counter only!
                if not isinstance(OMRCode, basestring):
                    OMRCode[0] = 'PCMode'
                else:
                    OMRCode = '%04x' % ((int(OMRCode, 16) | ((0b11) << 22)))
                self.DPXWriteOMRCommand(slot, OMRCode)

                # Loop over THLs
                # Init result lists
                lastTHL = 0
                THLList = []
                dataList = []
                THLMeasList = []

                # Dummy readout of AnalogOut
                # self.MCGetADCvalue()
                # THLRange = np.arange(THLlow, THLhigh, THLstep)
                # THLRange = self.THLCalib[np.logical_and(self.THLCalib > THLlow, self.THLCalib < THLhigh)]
                THLRange = np.asarray(self.THLEdges[slot - 1])
                THLstop_ = THLhigh if THLstop > THLhigh else THLstop
                THLRange = THLRange[np.logical_and(THLRange >= THLstart, THLRange <= THLstop_)]

                # Do a fast measurement to estimate the position of the edge for each pixel
                THLFastList = []
                dataFastList = []
                THLFastStep = 5
                NPulses = 30        # Has to be less than 256

                for cnt, THL in enumerate(THLRange[::THLFastStep]): # :THLFastStep]):
                    # Set new THL
                    self.DPXWritePeripheryDACCommand(slot, DACval[:-4] + '%04x' % THL)

                    # Start measurement
                    self.DPXDataResetCommand(slot)
                    for i in range(NPulses):
                        self.DPXGeneralTestPulse(slot, 1000)
                    data = self.DPXReadToTDatakVpModeCommand(slot)[column]
                    data[data > NPulses] = NPulses

                    # Store data in lists
                    THLFastList.append( THL )
                    dataFastList.append( data )

                if not np.any(THLFastList):
                    continue

                print 'THLFastList'
                print THLFastList
                print np.asarray(dataFastList).T

                # Check if array is empty
                if not np.count_nonzero(dataFastList):
                    # If so, energy is set too low
                    valueCount -= 1
                    ToTListTotal.pop()
                    ToTErrListTotal.pop()
                    continue

                # Get derivative and find maximum for every pixel
                xFastDer = np.asarray(THLFastList[:-1]) + THLFastStep / 2.
                dataFastDer = [xFastDer[np.argmax(np.diff(data) / float(THLFastStep))] for data in np.asarray(dataFastList).T if np.count_nonzero(data)]

                print 'dataFastDer'
                print dataFastDer

                # Get mean THL value
                meanTHL = np.mean( dataFastDer )

                # Set THL range to new estimated region
                regSel = np.logical_and(THLRange >= min(dataFastDer) - THLFastStep, THLRange <= max(dataFastDer) + THLFastStep)
                THLRange = THLRange[regSel]
                print 'THLRange'
                print THLRange
                # THLRange = THLRange[abs(THLRange - meanTHL) < 3 * THLFastStep]

                # Do a slow but precise measurement with small THL steps
                for cnt, THL in enumerate(THLRange[::THLstep]):
                    self.statusBar(float(cnt)/len(THLRange[::THLstep]) * 100 + 1)
                    # for V in np.linspace(0.34, 0.36, 1000):
                    # THL = self.getTHLfromVolt(V)
                    # THLmeas = np.mean( [float(int(self.MCGetADCvalue(), 16)) for i in range(3) ] )
                    # THLMeasList.append( THLmeas )
                    # print THL, '%04x' % THL

                    # Set new THL
                    self.DPXWritePeripheryDACCommand(slot, DACval[:-4] + '%04x' % THL)
                    # print DACval[:-4] + '%04x' % THL
                    # print self.peripherys + '%04x' % THL

                    # Generate test pulse and measure ToT in integrationMode
                    # dataTemp = np.zeros(16)
                    self.DPXDataResetCommand(slot)
                    for i in range(NPulses):
                        self.DPXGeneralTestPulse(slot, 1000)
                    data = self.DPXReadToTDatakVpModeCommand(slot)[column]
                    # print data

                    # Store data in lists
                    THLList.append( THL )
                    dataList.append( data )

                    # lastTHL = THLmeas
                print

                # Calculate derivative of THL spectrum
                xDer = np.asarray(THLList[:-1]) + 0.5*THLstep
                dataDer = [np.diff(data) / float(THLstep) for data in np.asarray(dataList).T]

                peakList = []
                peakErrList = []
                # Transpose to access data of each pixel
                for data in np.asarray(dataList).T:
                    # Convert to corrected THL
                    THLListCorr = [self.getVoltFromTHLFit(elm, slot) for elm in THLList]
                    THLListFit = np.linspace(min(THLListCorr), max(THLListCorr), 1000)

                    # Perform erf-Fit
                    p0 = [THLListCorr[int(len(THLListCorr) / 2.)], 3.]
                    try:
                        popt, pcov = scipy.optimize.curve_fit(lambda x, b, c: self.erfFit(x, NPulses, b, c), THLListCorr, data, p0=p0)
                        perr = np.sqrt(np.diag(pcov))
                        print popt
                    except:
                        popt = p0
                        perr = len(p0) * [0]
                        print 'Fit failed!'
                        pass

                    # Return fit parameters
                    # a: Amplitude
                    # b: x-offset
                    # c: scale
                    b, c = popt
                    peakList.append( b )
                    # Convert to sigma
                    peakErrList.append( perr[1] / np.sqrt(2) )

                    # Savitzky-Golay filter the data
                    # Ensure odd window length
                    windowLength = int(len(THLRange[::THLstep]) / 10.)
                    if not windowLength % 2:
                        windowLength += 1

                    # Plots
                    if plot:
                        try:
                            dataFilt = scipy.signal.savgol_filter(data, windowLength, 3)
                            plt.plot(THLListCorr, dataFilt)
                            plt.plot(*self.getDerivative(THLListCorr, dataFilt))
                        except:
                            pass

                        plt.plot(THLListFit, self.erfFit(THLListFit, NPulses, b, c), ls='-')
                        plt.plot(THLListFit, self.normalErf(THLListFit, NPulses, b, c), ls='-')
                        plt.plot(THLListCorr, data, marker='x', ls='')
                        plt.show()

                peakList, peakErrList = np.asarray(peakList), np.asarray(peakErrList)
                peakList[peakErrList > 50] = np.nan
                peakErrList[peakErrList > 50] = np.nan
                print peakList, peakErrList

                # Set start value for next scan
                print 'Start, Stop'
                THLstart, THLstop = np.nanmin(THLList) - 400, np.nanmax(THLList) + 400
                print THLstart, THLstop

                # Convert back to THL
                # peakList = [THLRange[int(peak + 0.5)] + (peak % 1) for peak in peakList]

                THLListTotal.append( peakList )
                THLErrListTotal.append( peakErrList )
                print

            # Transform to arrays
            THLListTotal = np.asarray(THLListTotal).T
            THLErrListTotal = np.asarray(THLErrListTotal).T
            ToTListTotal = np.asarray(ToTListTotal).T
            ToTErrListTotal = np.asarray(ToTErrListTotal).T

            print THLListTotal
            print THLErrListTotal
            print ToTListTotal
            print ToTErrListTotal

            resDict[column] = {'THL': THLListTotal, 'THLErr': THLErrListTotal, 'ToT': ToTListTotal, 'ToTErr': ToTErrListTotal}

        if plot:
            # Show results in plot for last accessed column
            fig, ax = plt.subplots()
            for i in range(len(THLListTotal)):
                ax.errorbar(THLListTotal[i], ToTListTotal[i], xerr=THLErrListTotal[i], yerr=ToTErrListTotal[i], color=self.getColor('Blues', len(ToTListTotal), i), marker='x')

            ax.grid()
            plt.xlabel('THL (DAC)')
            plt.ylabel('ToT')

            # TODO: Perform fit
            plt.show()

        # Save to pickle file
        if outFn:
            # Create dictionary
            # d = {'THL': THLListTotal, 'THLErr': THLErrListTotal, 'ToT': ToTListTotal, 'ToTErr': ToTErrListTotal}

            self.pickleDump(resDict, outFn)

        return resDict

    def getDerivative(self, x, y):
        deltaX = x[1] - x[0]
        der = np.diff(y) / deltaX
        return x[:-1] + deltaX, der

    def erfFit(self, x, a, b, c):
        return a*(0.5 * scipy.special.erf((x - b)/c) + 0.5)

    def normalErf(self, x, a, b, c):
        return 0.56419*a*np.exp(-(x-b)**2/(c**2)) + 0.5

    def energySpectrumTHL(self, slot=1, THLhigh=4975, THLlow=2000, THLstep=1, timestep=0.1, intPlot=True, outFn='energySpectrumTHL.p', slopeFn='pixelSlopes.p'):
        # Description: Measure cumulative energy spectrum 
        #              by performing a threshold scan. The
        #              derivative of this spectrum resembles
        #              the energy spectrum.

        THLhigh = int(self.THLs[slot-1], 16)

        assert THLhigh <= 8191, "energySpectrumTHL: THLHigh value set too high!"

        # Take data in integration mode: 
        #     Sum of deposited energy in ToT
        OMRCode = self.OMR
        if not isinstance(OMRCode, basestring):
            OMRCode[0] = 'IntegrationMode'
            # Select AnalogOut
            OMRCode[4] = 'V_ThA'

        else:
            OMRCode = '%04x' % ((int(OMRCode, 16) | ((0b11) << 22)))

            # Select AnalogOut
            OMRCode &= ~(0b11111 << 12)
            OMRCode |= getattr(self.__OMRAnalogOutSel, 'V_ThA')

        self.DPXWriteOMRCommand(slot, OMRCode)
        self.DPXWriteColSelCommand(slot, 0)

        lastTHL = 0
        if intPlot:
            plt.ion()
            fig, ax = plt.subplots()
            axDer = ax.twinx()

            ax.set_yscale("log", nonposy='clip')
            # axDer.set_yscale("log", nonposy='clip')

            # Empty plot
            lineCum, = ax.plot(np.nan, np.nan, label='Cumulative', color='cornflowerblue')
            lineDer, = axDer.plot(np.nan, np.nan, label='Derivative', color='crimson')
            # lineTHL, = axTHL.plot(np.nan, np.nan, label='THL', color='orange')

            # Settings
            plt.xlabel('THL (DAC)')
            ax.set_ylabel('Counts / s')
            axDer.set_ylabel('Derivative (a.u.)')
            # axTHL.set_ylabel('THL (V)')

            plt.grid()

        # Also use pixelDACs?
        if slopeFn and THLstep == 1:
            if os.path.isfile(slopeFn):
                if slopeFn.endswith('.p'):
                    slopeDict = cPickle.load( open(slopeFn, 'rb') )
                else:
                    slopeDict = hickle.load(slopeFn)

                slopes = np.reshape(slopeDict['slope'], (16, 16))
                # Remove values with large slope...
                slopes[slopes > 3] = np.nan
                # ...and replace with mean
                slopes[np.isnan(slopes)] = np.nanmean( slopes )

                # Mean slope
                slopeMean = np.mean( slopes.flatten() )
                print 'SlopeMean:'
                print slopeMean

                # pixelDAC commands
                # Start each pixel at 0x1f
                pixelDACs = ['1f' * 256]
                THLOffsets = [[0]*256]
                for m in reversed(range(1, int(1./slopeMean) + 1)):
                    pixelDAC = np.asarray( [0x1f + 1./slope * 1./m for slope in slopes.flatten()], dtype=int )

                    offset = np.reshape([(pixelDAC[i] - 0x1f) * slopes.flatten()[i] for i in range(len(pixelDAC))], (16, 16))
                    THLOffsets.append( offset )

                    # Check if pixel is within range
                    pixelDAC[pixelDAC < 0] = 0
                    pixelDAC[pixelDAC > 63] = 63
                    # Convert to string
                    pixelDACs.append( ''.join(['{:02x}'.format(item) for item in pixelDAC]) )

                print pixelDACs

                # THLOffsets = [ np.reshape([(int(1./slope * m) - 0x1f) * slope for slope in slopes.flatten()], (16, 16)) for m in range(int(1./slopeMean)) ]

                print 'THLOffsets:'
                print THLOffsets
                print

            else:
                pixelDACs = [None]
        else:
            pixelDACs = [None]

        # Loop over DAC values
        THLList = []
        THLVList = []
        dataList = []

        THLRange = np.asarray(self.THLEdges[slot - 1])
        THLRange = THLRange[np.logical_and(THLRange > THLlow, THLRange < THLhigh)]

        # Savitzky-Golay filter the data
        # Ensure odd window length
        windowLength = int(len(THLRange[::THLstep]) / 10.)
        if not windowLength % 2:
            windowLength += 1

        # Empty bins
        '''
        self.DPXDataResetCommand(slot)
        for col in range(16):
            self.DPXWriteColSelCommand(slot, col)
            self.DPXReadBinDataDosiModeCommand(slot)
        '''

        # Create dictionary to store data
        pixelDataList = []

        # Catch keyboardInterrupt exception if necessary
        measStart = time.time()

        try:
            # Measure time
            startTime = time.time()
            for THL in THLRange[::THLstep]:
                # Set threshold
                self.DPXWritePeripheryDACCommand(slot, self.peripherys + '%04x' % THL)

                # Convert THL to corrected THL value
                THL = self.getVoltFromTHLFit(THL, slot)

                for i, pixelDAC in enumerate(pixelDACs):
                    # THL matrix for all pixels
                    THLArray = np.zeros((16, 16))
                    print THL
                    THLArray.fill( THL )

                    if pixelDAC:
                        self.DPXWritePixelDACCommand(slot, pixelDAC)
                        THLArray = np.asarray( THLArray ) + np.reshape(THLOffsets[i] , (16, 16))

                    THLList.append( THLArray.flatten() )

                    # Start frame 
                    # OMRCode[1] = 'ClosedShutter'
                    # self.DPXWriteOMRCommand(slot, OMRCode)
                    
                    self.DPXDataResetCommand(slot)

                    # Wait to accumulate data
                    time.sleep(timestep)

                    pixelData = []

                    # Loop over pixel columns
                    '''
                    for col in range(16):
                        self.DPXWriteColSelCommand(slot, col) 
                        data = np.sum(self.DPXReadBinDataDosiModeCommand(slot), axis=1)

                        # Gather in pixel matrix
                        pixelData.append( data )
                    '''

                    # Read out in integration mode
                    pixelData = self.DPXReadToTDataIntegrationModeCommand(slot)

                    # Scale data with readout time
                    pixelData = np.asarray(pixelData) / float(time.time() - startTime)

                    if intPlot:
                        # Large pixels only
                        plotData = pixelData[:,range(2, 14)]

                    # After readout, new measurement begins
                    startTime = time.time()

                    # Flatten and sum data to get total number of events
                    data = pixelData.flatten()
                    print data
                    pixelDataList.append( data )

                    # OMRCode[1] = 'OpenShutter'
                    # self.DPXWriteOMRCommand(slot, OMRCode)
                    # End frame

                    try:
                        # Update plot
                        pixel = -55
                        if intPlot:
                            # Plot only one pixel
                            dataList.append( plotData.flatten()[pixel] )

                            if len(THLList[1:]) < windowLength:
                                continue

                            dataFilt = scipy.signal.savgol_filter(dataList[1:], windowLength, 3)

                            THLList_ = np.arange(len(np.asarray(THLList[1:])[:,pixel]))

                            # Cumulative
                            lineCum.set_xdata( THLList_ )
                            lineCum.set_ydata( dataFilt )
                            
                            # Derivative
                            dataDer = np.diff( dataFilt )
                            lineDer.set_xdata( np.asarray(THLList_[:-1]) + 0.5)
                            lineDer.set_ydata( dataDer )

                            # Measure threshold voltage
                            # lineTHL.set_xdata( THLList )
                            # lineTHL.set_ydata( THLVList )

                            ax.set_xlim(min(THLList_), max(THLList_))
                            ax.set_ylim(0.9*min(dataFilt), 1.1*max(dataFilt))

                            # axTHL.set_xlim(min(THLList), max(THLList))
                            # axTHL.set_ylim(0.9*min(THLVList), 1.1*max(THLVList))

                            if dataDer.size:
                                dataDer *= max(dataFilt)/max(dataDer)
                                axDer.set_xlim(min(THLList_), max(THLList_))
                                axDer.set_ylim(0.9*min(dataDer[1:]), 1.1*max(dataDer[1:]))

                            fig.canvas.draw()
                    except:
                        pass

            # PixelDataList to dictionary
            pixelDict = {i: {'THL': np.asarray(THLList)[:,i][1:], 'data': np.asarray(pixelDataList)[:,i]} for i in range(256)}

            # Save to file
            self.pickleDump(pixelDict, outFn)
            raw_input('')

        except (KeyboardInterrupt, SystemExit):
            # Store data and plot in files
            print 'KeyboardInterrupt-Exception: Storing data!'
            print 'Measurement time: %.2f min' % ((time.time() - measStart) / 60.)

            # PixelDataList to dictionary
            pixelDict = {i: {'THL': np.asarray(THLList)[:,i], 'data': np.asarray(pixelDataList)[:,i]} for i in range(256)}

            self.pickleDump(pixelDict, outFn)
            raise

        # Reset OMR and peripherys
        self.DPXWriteOMRCommand(slot, self.OMR)
        self.DPXWritePeripheryDACCommand(slot, self.peripherys + self.THLs[slot-1])

    def temperatureWatch(self, slot, column='all', frames=10, energyRange=(15.e3, 125.e3, 10), fn='TemperatureToT.p', intplot=True):
        if intplot:
            # Interactive plot
            plt.ion()
            fig, ax = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw = {'width_ratios': [1, 1]})

            # ax0 label
            ax[0].set_xlabel('Time (s)')
            ax[0].set_ylabel('Temperature (DAC)')

            # ax1 label
            ax[1].set_xlabel(r'$\mu_\mathrm{ToT}$')
            ax[1].set_ylabel('Temperature (DAC)')

            # Temperature over time
            line, = ax[0].plot(np.nan, np.nan)
            # Correlation plot
            lineCorrList = []
            for i in range(256):
                lineCorr, = ax[1].plot(np.nan, np.nan, color=self.getColor('viridis', 256, i))
                lineCorrList.append( lineCorr )

        if type(self.OMR) is list:
            OMRCode_ = self.OMRListToHex(self.OMR)
        else:
            OMRCode_ = self.OMR
        OMRCode_ = int(OMRCode_, 16)

        OMRCode_ &= ~(0b11111 << 12)
        OMRCode_ |= getattr(self.__OMRAnalogOutSel, 'Temperature')
        self.DPXWriteOMRCommand(slot, hex(OMRCode_).split('0x')[-1])

        # Init test pulses
        columnList = self.testPulseInit(slot, column=column)
        # self.maskBitsColumn(slot, column)

        # Get energy within specified range
        energyRange = np.linspace(*energyRange)

        # Number of test pulses per measurement
        Npulses = 5

        try:
            # Loop over measurements
            startTime = time.time()
            energyDict = {}

            logDict = {'time': [], 'temp': [], 'tempErr': [], 'ToT': [], 'ToTErr': [], 'energy': []}

            for energy in energyRange:
                DACval = self.getTestPulseVoltageDAC(slot, energy, True)
                self.DPXWritePeripheryDACCommand(slot, DACval)

                if intplot:
                    fig.suptitle('Test Pulse Energy: %.2f keV' % (energy*1.e-3))

                for c in range( frames ):
                    # Time
                    logDict['time'].append( time.time() - startTime )

                    # Energy
                    logDict['energy'].append( energy )

                    # Measure temperature at start
                    TList = []
                    for i in range(100):
                        TList.append( float(int(self.MCGetADCvalue(), 16)) )

                    # Temperature
                    logDict['temp'].append( np.mean(TList) )
                    logDict['tempErr'].append( np.std(TList) / np.sqrt(100) )

                    # Plot
                    if intplot:
                        line.set_xdata(logDict['time'])
                        line.set_ydata(logDict['temp'])
                        ax[0].set_xlim(min(logDict['time']), max(logDict['time']))
                        ax[0].set_ylim(0.95*min(logDict['temp']), 1.05*max(logDict['temp']))

                    # Test pulses
                    dataList = []
                    for i in range(Npulses):
                        self.DPXDataResetCommand(slot)
                        for col in columnList:
                            self.maskBitsColumn(slot, col)
                            self.DPXGeneralTestPulse(slot, 1000)

                        data = np.asarray(self.DPXReadToTDataDosiModeCommand(slot), dtype=float)[columnList].flatten()
                        # Get rid of outliers
                        data[abs(data - np.mean(data)) > 3 * np.std(data)] = np.nan
                        dataList.append( np.asarray(self.DPXReadToTDataDosiModeCommand(slot))[columnList].flatten() )

                    dataMean = np.mean(dataList, axis=0)
                    dataErr = np.std(dataList, axis=0) / np.sqrt( Npulses )

                    logDict['ToT'].append( dataMean )
                    logDict['ToTErr'].append( dataErr )

                    if intplot:
                        # Correlation plot
                        for i in range(256):
                            lineCorrList[i].set_xdata(np.asarray(logDict['ToT'])[:,i])
                            lineCorrList[i].set_ydata(logDict['temp'])

                        print logDict['ToT']
                        ax[1].set_xlim(0.99*np.min(logDict['ToT']), 1.01*np.max(logDict['ToT']))
                        ax[1].set_ylim(0.95*min(logDict['temp']), 1.05*max(logDict['temp']))

                        fig.canvas.draw()

            if fn:
                self.pickleDump(energyDict, fn)

        except (KeyboardInterrupt, SystemExit):
            # Store data and plot in files
            print 'KeyboardInterrupt'
            if fn:
                self.pickleDump(logDict, fn)
            raise

    # If cnt = 0, loop infinitely
    def ADCWatch(self, slot, OMRAnalogOutList, cnt=0, fn='ADCWatch.p'):
        # Interactive plot
        plt.ion()
        fig, ax = plt.subplots()

        # Plot settings
        plt.xlabel('Time (s)')

        # Init plot
        OMRAnalogOutDict = {}

        lineDict = {}   # Lines of plots
        startDict = {}  # Used to normalize the ADC values

        startTime = time.time()

        for OMRAnalogOut in OMRAnalogOutList:
            dataDict = {}

            # Select AnalogOut
            print 'OMR Manipulation:'
            if type(self.OMR) is list:
                OMRCode_ = self.OMRListToHex(self.OMR)
            else:
                OMRCode_ = self.OMR
            OMRCode_ = int(OMRCode_, 16)

            OMRCode_ &= ~(0b11111 << 12)
            OMRCode_ |= getattr(self.__OMRAnalogOutSel, OMRAnalogOut)
            # OMRCode_ &= ~(0b11111 << 7)
            # OMRCode_ |= getattr(self.__OMRAnalogInSel, OMRAnalogOut)
            self.DPXWriteOMRCommand(slot, hex(OMRCode_).split('0x')[-1])

            # Data
            # Take mean of 10 samples for first point
            meanList = []
            for i in range(10):
                val = float(int(self.MCGetADCvalue(), 16))
                meanList.append( val )
            print meanList

            startVal = float(np.mean(meanList))
            startDict[OMRAnalogOut] = startVal
            dataDict['data'] = [startVal]

            # Time
            currTime = time.time() - startTime
            dataDict['time'] = [currTime]

            OMRAnalogOutDict[OMRAnalogOut] = dataDict

            line, = ax.plot([currTime], [1.], label=OMRAnalogOut)
            lineDict[OMRAnalogOut] = line

        plt.legend()
        plt.grid()
        fig.canvas.draw()

        print OMRAnalogOutDict

        # Run measurement forever if cnt == 0
        loopCnt = 0
        while (True if cnt == 0 else loopCnt < cnt):
            try:
                for i, OMRAnalogOut in enumerate(OMRAnalogOutList):

                    # Select AnalogOut
                    if type(self.OMR) is list:
                        OMRCode_ = self.OMRListToHex(self.OMR)
                    else:
                        OMRCode_ = self.OMR
                    OMRCode_ = int(OMRCode_, 16)
                    OMRCode_ = OMRCode_ & ~(0b11111 << 12)
                    OMRCode_ |= getattr(self.__OMRAnalogOutSel, OMRAnalogOut)
                    print hex(OMRCode_).split('0x')[-1]
                    # OMRCode_ &= ~(0b11111 << 7)
                    # OMRCode_ |= getattr(self.__OMRAnalogInSel, OMRAnalogOut)
                    self.DPXWriteOMRCommand(slot, hex(OMRCode_).split('0x')[-1])

                    # Time 
                    currTime = time.time() - startTime

                    # Data
                    currVal = float(int(self.MCGetADCvalue(), 16))
                    print currVal, startDict[OMRAnalogOut]

                    # Set values in dict
                    OMRAnalogOutDict[OMRAnalogOut]['time'].append(currTime)

                    OMRAnalogOutDict[OMRAnalogOut]['data'].append(currVal)

                    # Refresh line
                    lineDict[OMRAnalogOut].set_xdata(OMRAnalogOutDict[OMRAnalogOut]['time'])

                    if len(OMRAnalogOutDict[OMRAnalogOut]['data']) > 21:
                        lineDict[OMRAnalogOut].set_ydata(scipy.signal.savgol_filter(np.asarray(OMRAnalogOutDict[OMRAnalogOut]['data'])/startDict[OMRAnalogOut], 21, 5))
                    else: 
                        lineDict[OMRAnalogOut].set_ydata(np.asarray(OMRAnalogOutDict[OMRAnalogOut]['data'])/startDict[OMRAnalogOut])

                # Scale plot axes
                plt.xlim(min([min(OMRAnalogOutDict[key]['time']) for key in OMRAnalogOutDict.keys()]), max([max(OMRAnalogOutDict[key]['time']) for key in OMRAnalogOutDict.keys()]))

                plt.ylim(min([min(np.asarray(OMRAnalogOutDict[key]['data'])/startDict[OMRAnalogOut]) for key in OMRAnalogOutDict.keys()]), max([max(np.asarray(OMRAnalogOutDict[key]['data'])/startDict[OMRAnalogOut]) for key in OMRAnalogOutDict.keys()]))

                fig.canvas.draw()
                loopCnt += 1

            except (KeyboardInterrupt, SystemExit):
                # Store data and plot in files
                print 'KeyboardInterrupt'
                if fn:
                    self.pickleDump(OMRAnalogOutDict, fn)
                raise

        if fn:
            self.pickleDump(OMRAnalogOutDict, fn)

    def measureTHL(self, slot, fn=None, plot=False):
        if not fn:
            fn = THL_CALIB_FILE
        self.measureADC(slot, AnalogOut='V_ThA', perc=False, ADChigh=8191, ADClow=0, ADCstep=1, N=1, fn=fn, plot=plot)

    def measureADC(self, slot, AnalogOut='V_ThA', perc=False, ADChigh=8191, ADClow=0, ADCstep=1, N=1, fn=None, plot=False):
        # Display execution time at the end
        startTime = time.time()

        # Select AnalogOut
        if type(self.OMR) is list:
            OMRCode_ = self.OMRListToHex(self.OMR)
        else:
            OMRCode_ = self.OMR
        OMRCode_ = int(OMRCode_, 16)

        OMRCode_ &= ~(0b11111 << 12)
        OMRCode_ |= getattr(self.__OMRAnalogOutSel, AnalogOut)
        self.DPXWriteOMRCommand(slot, hex(OMRCode_).split('0x')[-1])

        # Get peripherys
        dPeripherys = self.splitPerihperyDACs(self.peripherys + self.THLs[0], perc=perc)

        if AnalogOut == 'V_cascode_bias':
            AnalogOut = 'V_casc_reset'
        elif AnalogOut == 'V_per_bias':
            AnalogOut = 'V_casc_preamp'

        ADCList = np.arange(ADClow, ADChigh, ADCstep)
        ADCVoltMean = []
        ADCVoltErr = []
        print 'Measuring ADC!'
        for cnt, ADC in enumerate(ADCList):
            self.statusBar(float(cnt)/len(ADCList) * 100 + 1)
            
            # Set threshold
            # self.DPXWritePeripheryDACCommand(slot, self.peripherys + '%04x' % ADC)

            # Set value in peripherys
            dPeripherys[AnalogOut] = ADC
            code = self.periheryDACsDictToCode(dPeripherys, perc=perc)
            self.peripherys = code[:-4]
            self.DPXWritePeripheryDACCommand(slot, code)

            # Measure multiple times
            ADCValList = []
            for i in range(N):
                ADCVal = float(int(self.MCGetADCvalue(), 16))
                ADCValList.append( ADCVal )

            ADCVoltMean.append( np.mean(ADCValList) )
            ADCVoltErr.append( np.std(ADCValList)/np.sqrt(N) )

        if plot:
            plt.errorbar(ADCList, ADCVoltMean, yerr=ADCVoltErr, marker='x')
            plt.show()

        # Sort lists
        ADCVoltMeanSort, ADCListSort = zip(*sorted(zip(ADCVoltMean, ADCList)))
        # print THLListSort
        if plot:
            plt.plot(ADCVoltMeanSort, ADCListSort)
            plt.show()

        d = {'Volt': ADCVoltMean, 'ADC': ADCList}
        if fn:
            self.pickleDump(d, fn)
        else:
            self.pickleDump(d, 'ADCCalib.p')

        print 'Execution time: %.2f min' % ((time.time() - startTime)/60.)

    def THLCalibToEdges(self, THLDict):
        volt, thl = THLDict['Volt'], THLDict['ADC']

        # Sort by THL
        thl, volt = zip(*sorted(zip(thl, volt)))
        diff = abs(np.diff(volt))
        edges = np.argwhere(diff > 200).flatten() + 1

        # Store fit results in dict
        d = {}

        edges = list(edges)
        edges.insert(0, 0)
        edges.append( 8190 )

        THLEdgesLow, THLEdgesHigh = [0], []

        x1 = np.asarray( thl[edges[0]:edges[1]] )
        y1 = np.asarray( volt[edges[0]:edges[1]] )
        popt1, pcov1 = scipy.optimize.curve_fit(self.erfStdFit, x1, y1)
        d[0] = popt1

        for i in range(1, len(edges) - 2):
            # Succeeding section
            x2 = np.asarray( thl[edges[i]:edges[i+1]] )
            y2 = np.asarray( volt[edges[i]:edges[i+1]] )

            popt2, pcov2 = scipy.optimize.curve_fit(self.linearFit, x2, y2)
            m1, m2, t1, t2 = popt1[0], popt2[0], popt1[1], popt2[1]
            d[i] = popt2

            # Get central position
            # Calculate intersection to get edges
            if i == 1:
                    Vcenter = 0.5*(self.erfStdFit(edges[i], *popt1) + self.linearFit(edges[i], m2, t2))
                    THLEdgesHigh.append( scipy.optimize.fsolve(lambda x: self.erfStdFit(x, *popt1) - Vcenter, 100)[0] )
            else:
                    Vcenter = 1./(m1 + m2) * (2*edges[i]*m1*m2 + t1*m1 + t2*m2)
                    THLEdgesHigh.append( (Vcenter - t1)/m1 )

            THLEdgesLow.append( (Vcenter - t2)/m2 )
            popt1, pcov1 = popt2, pcov2

        THLEdgesHigh.append( 8190 )

        return THLEdgesLow, THLEdgesHigh, d

    def linearFit(self, x, m, t):
        return m*x + t

    def erfStdFit(self, x, a, b, c, d):
        return a * (scipy.special.erf((x - b)/c) + 1) + d

    def thresholdEqualizationConfig(self, configFn, reps=1, I_pixeldac=0.21, intPlot=False, resPlot=True):
        for i in range(1, 3 + 1):
            pixelDAC, THL, confMask = self.thresholdEqualization(slot=i, reps=reps, I_pixeldac=I_pixeldac, intPlot=intPlot, resPlot=resPlot)

            # Set values
            self.pixelDAC[i-1] = pixelDAC
            self.THLs[i-1] = THL
            self.confBits[i-1] = confMask

        self.writeConfig(configFn)

    def thresholdEqualization(self, slot, reps=1, I_pixeldac=0.5, intPlot=False, resPlot=True):
        # THLlow, THLhigh = 5120, 5630
        THLlow, THLhigh = 4000, 6000
        # THLRange = np.arange(THLlow, THLhigh, 1) 

        if self.THLEdges[slot - 1] is None:
            THLRange = np.arange(THLlow, THLhigh)
        else:
            THLRange = np.asarray(self.THLEdges[slot - 1])
            THLRange = np.around(THLRange[np.logical_and(THLRange >= THLlow, THLRange <= THLhigh)])

        THLstep = 1
        noiseLimit = 3
        spacing = 2
        existAdjust = True

        print '== Threshold equalization of detector %d ==' % slot

        # Set PC Mode in OMR in order to read kVp values
        # If OMR code is list
        if isinstance(self.OMR, (list,)):
            OMRCode = self.OMR
            OMRCode[0] = 'PCMode'
            self.DPXWriteOMRCommand(slot, OMRCode)
        else:
            self.DPXWriteOMRCommand(slot, '%04x' % ((int(self.OMR, 16) & ~((0b11) << 22)) | (0b10 << 22)))

        # Linear dependence:
        # Start and end points are sufficient
        pixelDACs = ['00', '3f']

        # Set I_pixeldac
        if I_pixeldac is not None:
            d = self.splitPerihperyDACs(self.peripherys + self.THLs[0], perc=True)
            d['I_pixeldac'] = I_pixeldac
            code = self.periheryDACsDictToCode(d, perc=True)
            self.peripherys = code[:-4]
            self.DPXWritePeripheryDACCommand(slot, code)

            # Perform equalization
            if I_pixeldac > 0.2:
                # Nonlinear dependence
                pixelDACs = ['%02x' % int(num) for num in np.linspace(0, 63, 9)]

        countsDict = self.getTHLLevel(slot, THLRange, pixelDACs, reps, intPlot)
        print countsDict['00'].keys(), THLRange
        gaussDict, noiseTHL = self.getNoiseLevel(countsDict, THLRange, pixelDACs, noiseLimit)
        print 'noiseTHL1:', noiseTHL

        # Transform values to indices and get meanDict
        meanDict, noiseTHL = self.valToIdx(slot, pixelDACs, THLRange, gaussDict, noiseTHL)

        if len(pixelDACs) > 2:
            def slopeFit(x, m, t):
                return m*x + t
            slope = np.zeros((16, 16))
            offset = np.zeros((16, 16))

        else: 
            slope = (noiseTHL['00'] - noiseTHL['3f']) / 63.
            offset = noiseTHL['00']

        x = [int(key, 16) for key in pixelDACs]
        if len(pixelDACs) > 2:
            # Store fit functions in list
            polyCoeffList = []

        for pixelX in range(16):
            for pixelY in range(16):
                y = []
                for pixelDAC in pixelDACs:
                    y.append(noiseTHL[pixelDAC][pixelX, pixelY])

                if len(pixelDACs) > 2:
                    x, y = np.asarray(x), np.asarray(y)
                    # Remove nan values
                    notNanIdx = np.isfinite(x) & np.isfinite(y)

                    # Perform polyfit
                    polyCoeff = np.polynomial.polynomial.polyfit(x[notNanIdx], y[notNanIdx], 3)
                    polyCoeffList.append( polyCoeff )

                    polyFit = np.polynomial.polynomial.Polynomial(polyCoeff)

                # plt.plot(adjust[pixelX, pixelY], mean, marker='x')
                if resPlot:
                    # Add offset to getColor to get rid of bright colors
                    plt.plot(x, y, alpha=.5, color=self.getColor('Blues', 5 + 256, pixelX * 16 + pixelY + 5))

                    if len(pixelDACs) > 2:
                        plt.plot(x, y, alpha=.5, color=self.getColor('Blues', 5 + 256, pixelX * 16 + pixelY + 5), ls='', marker='x')

                        xFit = np.linspace(min(x), max(x), 1000)
                        plt.plot(xFit, polyFit(xFit), ls='-', color=self.getColor('Blues', 5 + 256, pixelX * 16 + pixelY + 5))

        mean = 0.5 * (meanDict['00'] + meanDict['3f'])
        print meanDict['00'], meanDict['3f'], mean

        if resPlot:
            plt.xlabel('DAC')
            plt.ylabel('THL')
            plt.axhline(y=mean, ls='--')
            plt.grid()
            plt.show()

        if len(pixelDACs) == 2:
            # Get adjustment value for each pixel
            adjust = np.asarray( (offset - mean) / slope + 0.5 )
        else:
            # Get intersection of fit function and mean value
            # Describe mean in terms of a polynomial of 3rd grade
            polyMean = [mean, 0., 0., 0.]
            adjust = np.zeros(256)

            for pixel in range(256):
                polyCoeff = polyCoeffList[pixel]

                if np.any(np.isnan(polyCoeff)):
                    adjust[pixel] = 0
                else:
                    roots = np.asarray( np.polynomial.polynomial.polyroots(polyCoeff - polyMean) )
                    roots = roots[np.logical_and(roots <= 63, roots >= 0)]
                    if list(roots):
                        adjust[pixel] = roots[0]
                    else:
                        adjust[pixel] = 0

                    print adjust[pixel]
                
            adjust = np.reshape(adjust, (16, 16))

        # Consider extreme values
        adjust[np.isnan(adjust)] = 0
        adjust[adjust > 63] = 63
        adjust[adjust < 0] = 0

        # Convert to integer
        adjust = adjust.astype(dtype=int)

        # Set new pixelDAC values
        pixelDACNew = ''.join(['%02x' % entry for entry in adjust.flatten()])

        # Repeat procedure to get noise levels
        countsDictNew = self.getTHLLevel(slot, THLRange, pixelDACNew, reps, intPlot)

        gaussDictNew, noiseTHLNew = self.getNoiseLevel(countsDictNew, THLRange, pixelDACNew, noiseLimit)

        # Transform values to indices
        meanDictNew, noiseTHLNew = self.valToIdx(slot, [pixelDACNew], THLRange, gaussDictNew, noiseTHLNew)

        # Plot the results of the equalization
        if resPlot:
            bins = np.linspace(min(gaussDict['3f']), max(gaussDict['00']), 100)

            for pixelDAC in ['00', '3f']:
                plt.hist(gaussDict[pixelDAC], bins=bins, label='%s' % pixelDAC, alpha=0.5)

            plt.hist(gaussDictNew[pixelDACNew], bins=bins, label='After equalization', alpha=0.5)

            plt.legend()

            plt.xlabel('THL')
            plt.ylabel('Counts')

            plt.show()

        # Create confBits
        confMask = np.zeros((16, 16)).astype(str)
        confMask.fill('00')
        confMask[abs(noiseTHLNew[pixelDACNew] - mean) > 10] = '%02x' % getattr(self.__ConfBits, 'MaskBit')
        confMask = ''.join(confMask.flatten())

        '''
        # Find center point of next sawtooth in THL curve
        # in negative direction of THL
        THLNew = THLRange[int(mean)]

        # Calculate centers of edge intervals
        edgesCenter = np.asarray([ 0.5 * (self.THLEdgesHigh[k] + self.THLEdgesLow[k]) for k in range(len(self.THLEdgesLow)) ])

        # Eliminate values larger than THLNew
        edgesCenter = edgesCenter[edgesCenter <= THLNew]

        # Get closest center value to THLNew
        THLNew = min(edgesCenter, key=lambda x:abs(x-THLNew))
        '''
        # THLNew = self.THLEdges[list(THLRange).index(int(self.getTHLfromVolt(mean))) - 20]
        THLNew = int(np.mean(gaussDictNew[pixelDACNew]) - 20)

        print
        print 'Summary:'
        print 'pixelDACs:', pixelDACNew
        print 'confMask:', confMask
        print 'Bad pixels:', np.argwhere((abs(noiseTHLNew[pixelDACNew] - mean) > 10) == True)
        print 'THL:', '%04x' % THLNew

        # Restore OMR values
        self.DPXWriteOMRCommand(slot, self.OMR)

        # Subtract value from THLMin to guarantee robustness
        return pixelDACNew, '%04x' % THLNew, confMask

    def THSEqualization(self, slot):
        # Set PC Mode in OMR in order to read kVp values
        # If OMR code is list
        if isinstance(self.OMR, (list,)):
            OMRCode = self.OMR
            OMRCode[0] = 'PCMode'
            self.DPXWriteOMRCommand(slot, OMRCode)
        else:
            self.DPXWriteOMRCommand(slot, (int(self.OMR, 16) & ~((0b11) << 22)) | (0b10 << 22))

        # Get THLRange
        THLlow, THLhigh = 4000, 8000
        # THLRange = np.arange(THLlow, THLhigh, 1) 
        THLRange = np.asarray(self.THLEdges[slot - 1])
        THLRange = THLRange[np.logical_and(THLRange >= THLlow, THLRange <= THLhigh)]

        # Transform peripheryDAC values to dictionary
        print self.peripherys + self.THLs[0]
        d = self.splitPerihperyDACs(self.peripherys + self.THLs[0], perc=True)

        minIpixeldac, maxIpixeldac, NIpixeldac = 0, 1., 15
        minDAC, maxDAC, NDAC = 0, 63, 15

        I_pixeldacList = np.linspace(minIpixeldac, maxIpixeldac, NIpixeldac)
        pixelDACs = ['%02x' % elm for elm in np.asarray(np.linspace(minDAC, maxDAC, NDAC), dtype=int)]
        stdList = []

        # Store results in dicts
        gaussDicts = {}

        for idx, I_pixeldac in enumerate( I_pixeldacList ):
            # Change pixelDAC step value
            d['I_pixeldac'] = I_pixeldac
            code = self.periheryDACsDictToCode(d, perc=True)
            print code

            # Set peripheryDACs
            self.peripherys = code[:-4]
            self.DPXWritePeripheryDACCommand(slot, code)

            # Set pixelDACs to maximum value
            # pixelDACs = ['00']
            countsDict = self.getTHLLevel(slot, THLRange, pixelDACs, reps, intPlot)
            gaussDict, noiseTHL = self.getNoiseLevel(countsDict, THLRange, pixelDACs, noiseLimit)
            meanDict, noiseTHL = self.valToIdx(slot, pixelDACs, THLRange, gaussDict, noiseTHL)
            gaussCorrDict = {pixelDAC: [self.getVoltFromTHLFit(elm, slot) if elm else np.nan for elm in gaussDict[pixelDAC]] for pixelDAC in pixelDACs}

            # Add to I_pixeldac-dict
            gaussDicts[I_pixeldac] = gaussCorrDict

            # Add standard deviation to list
            # stdList.append( np.std(dic) )

        # Save to file
        self.pickleDump({'mean': meanMatrix, 'sigma': sigmaMatrix}, 'pixelDAC.p')

        return

    def meanSigmaLinePlot(self, fn, pIpixeldac, pPixelDAC):
        if fn.endswith('.p'):
            d = cPickle.load( open(fn, 'rb') )
        else:
            d = hickle.load(fn)
        meanMatrix, sigmaMatrix = d['mean'], d['sigma']
        meanMatrix, sigmaMatrix = meanMatrix, sigmaMatrix

        # Ipixeldac params
        minIpixeldac, maxIpixeldac, NIpixeldac = pIpixeldac
        # pixelDAC params
        minDAC, maxDAC, NDAC = pPixelDAC

        # Create x-axis
        pixelDAC = np.linspace(minDAC, maxDAC, NDAC)

        figMean = plt.figure(figsize=(7, 7))
        axMean = figMean.add_axes([0.1, 0.14, 0.7, 0.8])
        axCBarMean = figMean.add_axes([0.85, 0.1, 0.05, 0.8])
        figSigma = plt.figure(figsize=(7, 7))
        axSigma = figSigma.add_axes([0.1, 0.14, 0.7, 0.8])
        axCBarSigma = figSigma.add_axes([0.85, 0.1, 0.05, 0.8])

        for row in range(len(meanMatrix)):
            axMean.plot(pixelDAC, meanMatrix[row], color=self.getColor('viridis', len(meanMatrix), row))
            axSigma.plot(pixelDAC, sigmaMatrix[row], color=self.getColor('viridis', len(meanMatrix), row))

        self.getColorBar(axCBarMean, minIpixeldac*100, maxIpixeldac*100, NIpixeldac, label=r'$I_{\mathrm{pixelDAC}}$ (%)', rotation=0)
        self.getColorBar(axCBarSigma, minIpixeldac*100, maxIpixeldac*100, NIpixeldac, label=r'$I_{\mathrm{pixelDAC}}$ (%)', rotation=0)

        # Labels
        axMean.set_xlabel('PixelDAC')
        axMean.set_ylabel(r'$\mu_{\mathrm{THL}}$ (THL)')

        axSigma.set_xlabel('PixelDAC')
        axSigma.set_ylabel(r'$\sigma_{\mathrm{THL}}$ (THL)')

        figMean.show()
        figSigma.show()

        # Init plot
        # fig = plt.figure(figsize=(7, 4))
        # ax = fig.add_axes([0.1, 0.14, 0.7, 0.8])
        # Add colorbar later
        # axCBar = fig.add_axes([0.85, 0.1, 0.05, 0.8])

        # ax.hist(gaussCorrDict[pixelDAC], label=str(I_pixeldac), bins=np.arange(0, 1500, 5), alpha=0.5, color=self.getColor('viridis', len(I_pixeldacList), idx))

        # Add colorbar
        # self.getColorBar(axCBar, I_pixeldacList[0]*100, I_pixeldacList[-1]*100, N=len(I_pixeldacList), label=r'I_pixelDAC (\%)')

        # Standard deviation plot
        # figStd, axStd = plt.subplots()
        # axStd.plot(I_pixeldacList, stdList)

        # figStd.show()
        raw_input('')

        return

    def meanSigmaMatrixPlot(self, fn, pIpixeldac, pPixelDAC):
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        if fn.endswith('.p'):
            d = cPickle.load( open(fn, 'rb') )
        else:
            d = hickle.load(fn)
        meanMatrix = d['mean']
        print meanMatrix
        sigmaMatrix = d['sigma']

        figMean, axMean = plt.subplots()
        figSigma, axSigma = plt.subplots()

        # Ipixeldac params
        minIpixeldac, maxIpixeldac, NIpixeldac = pIpixeldac
        # pixelDAC params
        minDAC, maxDAC, NDAC = pPixelDAC

        # Calculate bin widths
        wIpixeldac = float(maxIpixeldac - minIpixeldac) / NIpixeldac
        wDAC = float(maxDAC - minDAC) / NDAC

        # Sigma plot
        im = axSigma.imshow(sigmaMatrix.T, extent=[minIpixeldac - 0.5*wIpixeldac, maxIpixeldac + 0.5*wIpixeldac, minDAC - 0.5*wDAC, maxDAC + 0.5*wDAC], aspect='auto')
        divider = make_axes_locatable(axSigma)
        caxSigma = divider.append_axes('right', size='5%', pad=0.1)
        figSigma.colorbar(im, cax=caxSigma)
        axSigma.set_xlabel('I_pixeldac'), axSigma.set_ylabel('pixelDAC')
        figSigma.show()

        # Mean plot
        im = axMean.imshow(meanMatrix.T, extent=[minIpixeldac - 0.5*wIpixeldac, maxIpixeldac + 0.5*wIpixeldac, minDAC - 0.5*wDAC, maxDAC + 0.5*wDAC], aspect='auto')
        divider = make_axes_locatable(axMean)
        caxMean = divider.append_axes('right', size='5%', pad=0.1)
        figMean.colorbar(im, cax=caxMean)
        axMean.set_xlabel('I_pixeldac'), axMean.set_ylabel('pixelDAC')
        figMean.show()

        raw_input('')

    # Get information about the slope THL/pixelDAC for each pixel.
    # Notice that the slope is only linear for I_pixeldac values 
    # about less than 20%. 
    def getPixelSlopes(self, slot, I_pixeldac=0.2):
        if not isinstance(I_pixeldac, list, ):
            I_pixeldacList = [I_pixeldac]
        else:
            I_pixeldacList = I_pixeldac

        assert max(I_pixeldacList) <= 0.2, "I_pixeldac has to be set less than or equal 20% to ensure linearity of the pixels."

        # Set PC Mode in OMR in order to read kVp values
        # If OMR code is list
        if not isinstance(self.OMR, (list,)):
            OMRCode = self.OMR
            OMRCode[0] = 'PCMode'
            self.DPXWriteOMRCommand(slot, OMRCode)
        else:
            self.DPXWriteOMRCommand(slot, (int(self.OMR, 16) & ~((0b11) << 22)) | (0b10 << 22))

        # Get THLRange
        THLlow, THLhigh = 4000, 8000
        # THLRange = np.arange(THLlow, THLhigh, 1) 
        THLRange = np.asarray(self.THLEdges[slot - 1])
        THLRange = THLRange[np.logical_and(THLRange >= THLlow, THLRange <= THLhigh)]

        # Only use first and last pixelDAC values
        pixelDACs = ['00', '3f']

        # Transform peripheryDAC values to dictionary
        print self.peripherys + self.THLs[0]
        dPeripherys = self.splitPerihperyDACs(self.peripherys + self.THLs[0], perc=True)

        # Init plot
        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_axes([0.1, 0.14, 0.7, 0.8])
        # Add colorbar later
        axCBar = fig.add_axes([0.85, 0.1, 0.05, 0.8])

        slopeList = []
        # Loop over I_pixeldacs
        for idx, I_pixeldac in enumerate( I_pixeldacList ):
            # Set I_pixeldac in peripherys
            dPeripherys['I_pixeldac'] = I_pixeldac
            code = self.periheryDACsDictToCode(dPeripherys, perc=True)
            self.peripherys = code[:-4]
            self.DPXWritePeripheryDACCommand(slot, code)

            # Get noise levels for each pixel
            countsDict = self.getTHLLevel(slot, THLRange, pixelDACs, 1, False)
            gaussDict, noiseTHL = self.getNoiseLevel(countsDict, THLRange, pixelDACs, 1)
            meanDict, noiseTHL = self.valToIdx(slot, pixelDACs, THLRange, gaussDict, noiseTHL)

            # Get slope for each pixel
            slope = (noiseTHL['00'] - noiseTHL['3f']) / 63.
            slopeList.append( slope )
            offset = noiseTHL['00']

            slope = np.nan_to_num( slope )
            print slope

            # Show histogram of slopes
            color = self.getColor('viridis', len(I_pixeldacList), idx)
            hist, bins = np.histogram(slope) # , bins=np.linspace(0., .15, 30))
            ax.plot(bins[:-1], hist, drawstyle='steps', color=color, alpha=0.5)
            ax.axvline(x=np.nanmean(slope))
            ax.axvline(x=np.nanmean(slope)+np.nanstd(slope), ls='--', color=color)
            ax.axvline(x=np.nanmean(slope)-np.nanstd(slope), ls='--', color=color)

        # Add colorbar
        self.getColorBar(axCBar, I_pixeldacList[0]*100, I_pixeldacList[-1]*100, N=len(I_pixeldacList), label=r'$I_\mathrm{pixelDAC}$ (%)')

        plt.show()
        if len(slopeList) == 1:
            return slopeList[0]
        else:
            return slopeList

    def valToIdx(self, slot, pixelDACs, THLRange, gaussDict, noiseTHL):
        # Transform values to indices
        meanDict = {}
        for pixelDAC in pixelDACs:
            d = np.asarray([self.getVoltFromTHLFit(elm, slot) if elm else np.nan for elm in gaussDict[pixelDAC] ], dtype=np.float)
            meanDict[pixelDAC] = np.nanmean(d)

            for pixelX in range(16):
                for pixelY in range(16):
                    elm = noiseTHL[pixelDAC][pixelX, pixelY]
                    if elm:
                        noiseTHL[pixelDAC][pixelX, pixelY] = self.getVoltFromTHLFit(elm, slot)
                    else:
                        noiseTHL[pixelDAC][pixelX, pixelY] = np.nan

        return meanDict, noiseTHL

    def getTHLLevel(self, slot, THLRange, pixelDACs=['00', '3f'], reps=1, intPlot=False):
        countsDict = {}

        # Interactive plot showing pixel noise
        if intPlot:
            plt.ion()
            fig, ax = plt.subplots()

            plt.xlabel('x (pixel)')
            plt.ylabel('y (pixel)')
            im = ax.imshow(np.zeros((16, 16)), vmin=0, vmax=255)

        if isinstance(pixelDACs, basestring):
            pixelDACs = [pixelDACs]

        # Loop over pixelDAC values
        for pixelDAC in pixelDACs:
            countsDict[pixelDAC] = {}
            print 'Set pixel DACs to %s' % pixelDAC
            
            # Set pixel DAC values
            if len(pixelDAC) > 2:
                pixelCode = pixelDAC
            else:
                pixelCode = pixelDAC*256
            self.DPXWritePixelDACCommand(slot, pixelCode, file=False)

            '''
            resp = ''
            while resp != pixelCode:
                resp = self.DPXReadPixelDACCommand(slot)
            ''' 

            # Dummy readout
            for j in range(3):
                self.DPXReadToTDatakVpModeCommand(slot)
                time.sleep(0.2)

            # Noise measurement
            # Loop over THL values
            print 'Loop over THLs'

            # Fast loop
            countsList = []
            THLRangeFast = THLRange[::10]
            for cnt, THL in enumerate(THLRangeFast):
                self.DPXWritePeripheryDACCommand(slot, self.peripherys + ('%04x' % THL))
                self.DPXDataResetCommand(slot)
                time.sleep(0.001)

                # Read ToT values into matrix
                countsList.append( self.DPXReadToTDatakVpModeCommand(slot).flatten() )

            countsList = np.asarray( countsList ).T
            THLRangeFast = [ THLRangeFast[item[0][0]] if np.any(item) else np.nan for item in [np.argwhere(counts > 3) for counts in countsList] ]

            # Precise loop
            THLRangeSlow = np.around(THLRange[np.logical_and(THLRange >= (np.nanmin(THLRangeFast) - 10), THLRange <= np.nanmax(THLRangeFast))])

            NTHL = len(THLRangeSlow)
            for cnt, THL in enumerate( THLRangeSlow ):
                self.statusBar(float(cnt)/NTHL * 100 + 1)

                # Repeat multiple times since data is noisy
                counts = np.zeros((16, 16))
                for lp in range(reps):
                    self.DPXWritePeripheryDACCommand(slot, self.peripherys + ('%04x' % THL))
                    self.DPXDataResetCommand(slot)
                    time.sleep(0.001)

                    # Read ToT values into matrix
                    counts += self.DPXReadToTDatakVpModeCommand(slot)

                    if intPlot:
                        im.set_data(counts)
                        ax.set_title('THL: ' + hex(THL))
                        fig.canvas.draw()

                counts /= float(reps)
                countsDict[pixelDAC][int(THL)] = counts
            print 
            print
        return countsDict

    def getNoiseLevel(self, countsDict, THLRange, pixelDACs=['00', '3f'], noiseLimit=3):
        if isinstance(pixelDACs, basestring):
            pixelDACs = [pixelDACs]

        # Get noise THL for each pixel
        noiseTHL = {key: np.zeros((16, 16)) for key in pixelDACs}

        gaussDict, gaussSmallDict, gaussLargeDict = {key: [] for key in pixelDACs}, {key: [] for key in pixelDACs}, {key: [] for key in pixelDACs}

        # Loop over each pixel in countsDict
        for pixelDAC in pixelDACs:
            for pixelX in range(16):
                for pixelY in range(16):
                    for idx, THL in enumerate(THLRange):
                        if not THL in countsDict[pixelDAC].keys():
                            continue

                        if countsDict[pixelDAC][THL][pixelX, pixelY] >= noiseLimit and noiseTHL[pixelDAC][pixelX, pixelY] == 0:
                                noiseTHL[pixelDAC][pixelX, pixelY] = THL

                                gaussDict[pixelDAC].append(THL)
                                if pixelY in [0, 1, 14, 15]:
                                    gaussSmallDict[pixelDAC].append(THL)
                                else:
                                    gaussLargeDict[pixelDAC].append(THL)

        return gaussDict, noiseTHL

    def close(self):
        # = Shut down =
        self.HVDeactivate()
        print 'Check if HV is deactivated...',
        for i in range(5):
            if not self.HVGetState():
                print 'done!'
                break
            else:
                self.HVDeactivate()
        else:
            assert 'HV could not be deactivated'

        print 'Measurement finished.'

        self.ser.close()

    # === initializeDPX ===
    # Establish serial connection to board
    def initializeDPX(self, portName, baudRate):
        # TODO: Add autodetection of correct com-port
        return 

    # === HV SECTION ===
    def HVSetDac(self, DAC):
        assert len(DAC) == 4, 'Error: DAC command has to be of size 4!'
 
        self.sendCmd([self.__receiverHV, self.__subReceiverNone, self.__senderPC, self.__HVsetDAC, '%06d' % len(DAC), DAC, self.__CRC])

    def HVGetDac(self):
        self.sendCmd([self.__receiverHV, self.__subReceiverNone, self.__senderPC, self.__HVgetDAC, self.__commandNoneLength, self.__commandNone, self.__CRC])

        return self.getDPXResponse()

    def HVActivate(self):
        self.sendCmd([self.__receiverHV, self.__subReceiverNone, self.__senderPC, self.__HVenable, self.__commandNoneLength, self.__commandNone, self.__CRC])

    def HVDeactivate(self):
        self.sendCmd([self.__receiverHV, self.__subReceiverNone, self.__senderPC, self.__HVdisable, self.__commandNoneLength, self.__commandNone, self.__CRC])

    def HVGetState(self):
        self.sendCmd([self.__receiverHV, self.__subReceiverNone, self.__senderPC, self.__HVisEnabled, self.__commandNoneLength, self.__CRC])

        res = int(self.getDPXResponse())

        if res:
            return True
        else:
            return False

    # === VC SECTION ===
    def VCVoltageSet3V3(self):
        self.sendCmd([self.__receiverVC, self.__subReceiverNone, self.__senderPC, self.__VCset3V3, self.__commandNoneLength, self.__commandNone, self.__CRC])

    def VCVoltageSet1V8(self):
        self.sendCmd([self.__receiverVC, self.__subReceiverNone, self.__senderPC, self.__VCset1V8, self.__commandNoneLength, self.__commandNone, self.__CRC])

    def VCGetVoltage(self):
        self.sendCmd([self.__receiverVC, self.__subReceiverNone, self.__senderPC, self.__VCgetVoltage, self.__commandNoneLength, self.__commandNone, self.__CRC])

        res = int(self.getDPXResponse())
        if res:
            return '3.3V'
        else:
            return '1.8V'

    # === MC SECTION ===
    def MCLEDenable(self):
        self.sendCmd([self.__receiverMC, self.__subReceiverNone, self.__senderPC, self.__MCLEDenable, self.__commandNoneLength, self.__commandNone, self.__CRC])

    def MCLEDdisable(self):
        self.sendCmd([self.__receiverMC, self.__subReceiverNone, self.__senderPC, self.__MCLEDdisable, self.__commandNoneLength, self.__commandNone, self.__CRC])

    def MCGetADCvalue(self):
        self.sendCmd([self.__receiverMC, self.__subReceiverNone, self.__senderPC, self.__MCgetADCvalue, self.__commandNoneLength, self.__commandNone, self.__CRC])

        return self.getDPXResponse()

    def MCGetFirmwareVersion(self):
        self.sendCmd([self.receiverMC, self.__subReceiverNone, self.__senderPC, self.__MCgetVersion, self.__commandNoneLength, self.__commandNone, self.__CRC])

        return self.getDPXResponse()

    # === DPX SECTION ===
    def OMRListToHex(self, OMRCode):
        OMRCodeList = OMRCode
        OMRTypeList = [self.__OMROperationMode,
        self.__OMRGlobalShutter,
        self.__OMRPLL,
        self.__OMRPolarity,
        self.__OMRAnalogOutSel,
        self.__OMRAnalogInSel,
        self.__OMRDisableColClkGate]

        OMRCode = 0x000000
        for i, OMR in enumerate(OMRCodeList):
            OMRCode |= getattr(OMRTypeList[i], OMR)

        OMRCode = '%04x' % (OMRCode) # hex(OMRCode).split('0x')[-1]

        return OMRCode

    def DPXWriteOMRCommand(self, slot, OMRCode):
        if type(OMRCode) is list:
            OMRCode = self.OMRListToHex(OMRCode)

        self.sendCmd([self.getReceiverFromSlot(slot), self.__subReceiverNone, self.__senderPC, self.__DPXwriteOMRCommand, '%06d' % len(OMRCode), OMRCode, self.__CRC])

        return self.getDPXResponse()

    def DPXReadOMRCommand(self, slot):
        self.sendCmd([self.getReceiverFromSlot(slot), self.__subReceiverNone, self.__senderPC, self.__DPXreadOMRCommand, self.__commandNoneLength, self.__commandNone, self.__CRC])

        return self.getDPXResponse()

    def DPXReadDigitalThresholdsCommand(self, slot):
        self.sendCmd([self.getReceiverFromSlot(slot), self.__subReceiverNone, self.__senderPC, self.__DPXreadDigitalThresholdsCommand, self.__commandNoneLength, self.__commandNone, self.__CRC])

        return self.getDPXResponse()

    def DPXGlobalReset(self, slot):
        self.sendCmd([self.getReceiverFromSlot(slot), self.__subReceiverNone, self.__senderPC, self.__DPXglobalResetCommand, self.__commandNoneLength, self.__commandNone, self.__CRC])

        return self.getDPXResponse()

    def DPXWriteConfigurationCommand(self, slot, confBitsFn, file=False):
        if file:
            with open(confBitsFn, 'r') as f:
                confBits = f.read()
            confBits = confBits.split('\n')
            assert len(confBits) == 1 or (len(confBits) == 2 and confBits[1] == ''), "Conf-Bits file must contain only one line!"
            confBits = confBits[0]
        else:
            confBits = confBitsFn

        self.sendCmd([self.getReceiverFromSlot(slot), self.__subReceiverNone, self.__senderPC, self.__DPXwriteConfigurationCommand, '%06d' % len(confBits), confBits, self.__CRC])

        return self.getDPXResponse()

    def DPXWriteSingleThresholdCommand(self, slot, THFn, file=False):
        if file:
            with open(THFn, 'r') as f:
                TH = f.read()
        else:
            TH = THFn

        self.sendCmd([self.getReceiverFromSlot(slot), self.__subReceiverNone, self.__senderPC, self.__DPXwriteSingleThresholdCommand, '%06d' % len(TH), TH, self.__CRC])

        return self.getDPXResponse()

    def DPXWriteColSelCommand(self, slot, col):
        colCode = '%02x' % col
        self.sendCmd([self.getReceiverFromSlot(slot), self.__subReceiverNone, self.__senderPC, self.__DPXwriteColSelCommand, '%06d' % len(colCode), colCode, self.__CRC])

        return self.getDPXResponse()

    def DPXWritePeripheryDACCommand(self, slot, code):
        self.sendCmd([self.getReceiverFromSlot(slot), self.__subReceiverNone, self.__senderPC, self.__DPXwritePeripheryDACCommand, '%06d' % len(code), code, self.__CRC])

        return self.getDPXResponse()

    def DPXReadPeripheryDACCommand(self, slot):
        self.sendCmd([self.getReceiverFromSlot(slot), self.__subReceiverNone, self.__senderPC, self.__DPXreadPeripheryDACCommand, self.__commandNoneLength, self.__commandNone, self.__CRC])

        return self.getDPXResponse()

    def DPXWritePixelDACCommand(self, slot, code, file=False):
        if file:
            with open(code, 'r') as f:
                code = f.read().split('\n')[0]

        # else: use code string

        self.sendCmd([self.getReceiverFromSlot(slot), self.__subReceiverNone, self.__senderPC, self.__DPXwritePixelDACCommand, '%06d' % len(code), code, self.__CRC])

        return self.getDPXResponse()

    def DPXReadPixelDACCommand(self, slot):
        self.sendCmd([self.getReceiverFromSlot(slot), self.__subReceiverNone, self.__senderPC, self.__DPXreadPixelDACCommand, self.__commandNoneLength, self.__commandNone, self.__CRC])

        return self.getDPXResponse()

    def DPXDataResetCommand(self, slot):
        self.sendCmd([self.getReceiverFromSlot(slot), self.__subReceiverNone, self.__senderPC, self.__DPXdataResetCommand, self.__commandNoneLength, self.__commandNone, self.__CRC])

        return self.getDPXResponse()

    def DPXReadBinDataDosiModeCommand(self, slot):
        self.sendCmd([self.getReceiverFromSlot(slot), self.__subReceiverNone, self.__senderPC, self.__DPXreadBinDataDosiModeCommand, self.__commandNoneLength, self.__commandNone, self.__CRC])

        return self.convertToDecimal(self.getDPXResponse())

    def DPXReadToTDataDosiModeCommand(self, slot):
        self.sendCmd([self.getReceiverFromSlot(slot), self.__subReceiverNone, self.__senderPC, self.__DPXreadToTDataDosiModeCommand, self.__commandNoneLength, self.__commandNone, self.__CRC])

        return self.convertToDecimal(self.getDPXResponse())

    def DPXReadToTDataDosiModeMultiCommand(self, slot):
        self.sendCmd([self.getReceiverFromSlot(slot), self.__subReceiverNone, self.__senderPC, self.__DPXreadToTDataDosiModeMultiCommand, self.__commandNoneLength, self.__commandNone, self.__CRC])

        res = self.getDPXResponse()

        # Read response and 
        x = [ord(char) for char in res]
        x1 = np.asarray( x[::2] )
        x2 = np.asarray( x[1::2] )
        # print [bin(x) for x in x1]
        # print [bin(x) for x in x2]

        x1 -= 32
        x2[x1 >= 128] -= 100
        x1[x1 >= 128] -= 128
        # print x1
        # print x2

        x1 <<= 8
        x = x1 + x2
        # print 
        # print [bin(x_) for x_ in x]

        return np.asarray(x)

    def DPXReadToTDataIntegrationModeCommand(self, slot):
        self.sendCmd([self.getReceiverFromSlot(slot), self.__subReceiverNone, self.__senderPC, self.__DPXreadToTDataIntegrationModeCommand, self.__commandNoneLength, self.__commandNone, self.__CRC])

        return self.convertToDecimal(self.getDPXResponse(), 6)

    def DPXReadToTDatakVpModeCommand(self, slot):
        self.sendCmd([self.getReceiverFromSlot(slot), self.__subReceiverNone, self.__senderPC, self.__DPXreadToTDatakVpModeCommand, self.__commandNoneLength, self.__commandNone, self.__CRC])

        return self.convertToDecimal(self.getDPXResponse(), 2)

    def DPXGeneralTestPulse(self, slot, length):
        lengthHex = '%04x' % length

        self.sendCmd([self.getReceiverFromSlot(slot), self.__subReceiverNone, self.__senderPC, self.__DPXgeneralTestPulse, '%06d' % len(lengthHex), lengthHex, self.__CRC])

        return self.getDPXResponse()

    def DPXGeneralMultiTestPulse(self, slot, length):
        lengthHex = '%04x' % length

        self.sendCmd([self.getReceiverFromSlot(slot), self.__subReceiverNone, self.__senderPC, self.__DPXgeneralMultiTestPulse, '%06d' % len(lengthHex), lengthHex, self.__CRC])

        return self.getDPXResponse()

    # == SUPPORT ==
    def readConfig(self, configFn):
        config = configparser.ConfigParser()
        config.read(configFn)

        # Mandatory sections
        sectionList = ['General', 'OMR', 'Slot1', 'Slot2', 'Slot3']

        # Check if set, else throw error
        for section in config.sections():
            assert section in sectionList, 'Config: %s is a mandatory section and has to be specified' % section

        # Read General
        if 'peripheryDAC' in config['General']:
            self.peripherys = config['General']['peripheryDAC']
        else:
            for i in range(1, 3 + 1):
                assert 'peripheryDAC' in config['Slot%d' % i], 'Config: peripheryDAC has to be set in either General or the Slots!'

        # Read OMR
        if 'code' in config['OMR']:
            self.OMR = config['OMR']['code']
        else:
            OMRList = []

            OMRCodeList = ['OperationMode', 'GlobalShutter', 'PLL', 'Polarity', 'AnalogOutSel', 'AnalogInSel', 'OMRDisableColClkGate']
            for OMRCode in OMRCodeList:
                assert OMRCode in config['OMR'], 'Config: %s has to be specified in OMR section!' % OMRCode

                OMRList.append(config['OMR'][OMRCode])

            self.OMR = OMRList

        # Read slot specific data
        for i in range(1, 3 + 1):
            assert 'Slot%d' % i in config.sections(), 'Config: Slot %d is a mandatory section!' % i

            # THL
            assert 'THL' in config['Slot%d' % i], 'Config: THL has to be specified in Slot%d section!' % i
            self.THLs[i-1] = config['Slot%d' % i]['THL']

            # confBits - optional field
            if 'confBits' in config['Slot%d' % i]:
                self.confBits[i-1] = config['Slot%d' % i]['confBits']
            else:
                # Use all pixels
                self.confBits[i-1] = '00' * 256

            # pixelDAC
            assert 'pixelDAC' in config['Slot%d' % i], 'Config: pixelDAC has to be specified in Slot%d section!' % i
            self.pixelDAC[i-1] = config['Slot%d' % i]['pixelDAC']

            # binEdges
            assert 'binEdges' in config['Slot%d' % i], 'Config: binEdges has to be specified in Slot%d section!' % i
            self.binEdges[i-1] = config['Slot%d' % i]['binEdges']

        return

    def writeConfig(self, configFn):
        config = configparser.ConfigParser()
        config['General'] = {'peripheryDAC': self.peripherys}

        if not isinstance(self.OMR, basestring):
            OMRCodeList = ['OperationMode', 'GlobalShutter', 'PLL', 'Polarity', 'AnalogOutSel', 'AnalogInSel', 'OMRDisableColClkGate']
            config['OMR'] = {OMRCode: self.OMR[i] for i, OMRCode in enumerate(OMRCodeList)}
        else:
            config['OMR'] = {'code': self.OMR}

        for i in range(1, 3 + 1):
            config['Slot%d' % i] = {'pixelDAC': self.pixelDAC[i-1], 'binEdges': self.binEdges[i-1], 'confBits': self.confBits[i-1], 'binEdges': self.binEdges[i-1], 'THL': self.THLs[i-1]}

        with open(configFn, 'w') as configFile:
            config.write(configFile)

    def sendCmd(self, cmdList):
        # Typical command string:
        # RRrrrSSSssCCCllllllcccc
        # R - receiver
        # r - subreceiver
        # s - sender
        # C - command
        # l - command length
        # c - CRC (unused, usually set to FFFF)

        # self.ser.write(self.__startOfTransmission.encode())
        cmdOut = [self.__startOfTransmission.encode()]

        if DEBUG:
            print self.__startOfTransmission.encode(),
        for cmd in cmdList:
            if not cmd:
                continue
            for c in cmd:
                if DEBUG:
                    print unichr(ord(c)),

                cmdOut.append( unichr(ord(c)).encode() )
                # self.ser.write(unichr(ord(c)).encode())
            if DEBUG:
                print ' ',

        if DEBUG:
            print self.__endOfTransmission.encode()

        cmdOut.append( self.__endOfTransmission.encode() )
        # self.ser.write(self.__endOfTransmission.encode())
        self.ser.write(cmdOut)

    def getBinEdges(self, slot, energyDict, paramDict, transposePixelMatrix=False):
        a, b, c, t = paramDict['a'], paramDict['b'], paramDict['c'], paramDict['t']
        grayCode = [0, 1, 3, 2, 6, 4, 5, 7, 15, 13, 12, 14, 10, 11, 9, 8]

        if slot == 0:
            energyType = 'free'
        elif slot == 1:
            energyType = 'Al'
        else:
            energyType = 'Sn'

        binEdgeString = ''
        for pixel in range(256):
            if self.isBig(pixel):
                energyList = energyDict['large'][energyType]
            else:
                energyList = energyDict['large'][energyType]

            energyList = np.asarray()

            # Convert to ToT
            ToTList = self.energyToToT(energyList, a[pixel], b[pixel], c[pixel], d[pixel])

            for binEdge in range(16):
                grayC = grayCode[binEdge]
                ToT = int( ToTList[binEdge] )

                binEdgeString += ('%01x' % grayC)
                binEdgeString += ('%03x' % ToT)

        return binEdgeString

    def isLarge(self, pixel):
        if ( (pixel - 1) % 16 == 0 ) or ( pixel % 16 == 0 ) or ( (pixel + 1) % 16 == 0 ) or ( (pixel + 2) % 16 == 0 ):
            return False
        return True

    def energyToToTAtan(self, x, a, b, c, t):
        return np.where(x > b, a*(x - b) + c*np.arctan((x - b)/t), np.nan)

    def energyToToT(self, x, a, b, c, t, atan=False):
        if atan:
            return EnergyToToTAtan(x, a, b, c, t)
        else:
            return np.where(x > t, a*x + b + float(c)/(x - t), np.nan)

    def getResponse(self):
        res = self.ser.readline()
        while res[0] != '\x02':
            res = self.ser.readline()

        if DEBUG:
            print res
        return res

    def getDPXResponse(self):
        # res = ''
        # while not res:
        res = self.getResponse()

        if DEBUG:
            print 'Length:', res[11:17]
        cmdLength = int( res[11:17] )

        if DEBUG:
            print 'CmdData:', res[17:17+cmdLength]
        cmdData = res[17:17+cmdLength]

        return cmdData

    def getReceiverFromSlot(self, slot):
        if slot == 1:
            receiver = self.__receiverDPX1
        elif slot == 2:
            receiver = self.__receiverDPX2
        elif slot == 3:
            receiver = self.__receiverDPX3
        else:
            assert 'Error: Function needs to access one of the three slots.'

        return receiver

    def convertToDecimal(self, res, length=4):
        # Convert 4 hexadecimal characters each to int
        hexNumList = np.asarray( [int(hexNum, 16) for hexNum in textwrap.wrap(res, length)] )
        return hexNumList.reshape((16, 16))

    def statusBar(self, perc, width='screen'):
        if width == 'screen':
            width = int(os.popen('tput cols', 'r').read()) - 8

        p = int(perc * float(width)/100)
        sys.stdout.write('\r')
        sys.stdout.write('[%-*s] %d%%' % (int(width), '='*p, perc))
        sys.stdout.flush()

    def pickleDump(self, outDict, outFn, hck=True):
        ending = '.p'
        if hck and outFn.endswith('.p'):
            outFn = outFn.split('.p')[0] + '.hck'
            ending = '.hck'

        # Check if file already exists
        while os.path.isfile(outFn):
            # Split dir and file
            outFnSplit = outFn.split('/')
            if len(outFnSplit) >= 2:
                directory, outFn = outFnSplit[:-1], outFnSplit[-1]
            else:
                directory = None

            outFnFront = outFn.split('.')[0]
            outFnFrontSplit = outFnFront.split('_')
            if len(outFnFrontSplit) >= 2:
                if outFnFrontSplit[-1].isdigit():
                    fnNum = int( outFnFrontSplit[-1] ) + 1
                    outFn = ''.join(outFnFrontSplit[:-1]) + '_' + str(fnNum) + ending
                else:
                    outFn = outFnFront + '_1' + ending
            else:
                outFn = outFnFront + '_1' + ending

            # Reattach directory
            if directory:
                outFn = '/'.join(directory + [outFn])

        with open(outFn, 'w') as f:
            if hck:
                hickle.dump(outDict, f, compression='lzf')
            else:
                cPickle.dump(outDict, f)

    def makeDirectory(self, directory):
        while os.path.isdir(directory):
            dir_front = directory.split('/')[0]
            dir_front_split = dir_front.split('_')
            if len(dir_front_split) >= 2:
                if dir_front_split[-1].isdigit():
                    dir_num = int(dir_front_split[-1]) + 1
                    directory = ''.join(dir_front_split[:-1]) + '_' + str(dir_num) + '/'
                else:
                    directory = dir_front + '_1/'
            else:
                directory = dir_front + '_1/'

        os.makedirs(directory)
        return directory

    def getTestPulseVoltageDAC(self, slot, DACVal, energy=False):
        # Set coarse and fine voltage of test pulse
        peripheryDACcode = int(self.peripherys + self.THLs[slot-1], 16)

        if energy:
            # Use nominal value of test capacitor
            # and DACVal as energy
            C = 5.14e-15

            deltaV = DACVal * scipy.constants.e / (C * 3.62)

            assert deltaV < 1.275, "TestPulse Voltage: The energy of the test pulse was set too high! Has to be less than or equal to 148 keV."

            # Set coarse voltage to 0
            voltageDiv = 2.5e-3
            DACVal = int((1.275 - deltaV) / voltageDiv)
        else:
            assert DACVal >= 0, 'Minimum THL value must be at least 0'
            assert DACVal <= 0x1ff, 'Maximum THL value mustn\'t be greater than %d' % 0x1ff

        # Delete current values
        peripheryDACcode &= ~(0xff << 32)   # coarse
        peripheryDACcode &= ~(0x1ff << 16)  # fine

        # Adjust fine voltage only
        peripheryDACcode |= (DACVal << 16)
        peripheryDACcode |= (0xff << 32)
        print '%032x' % peripheryDACcode
        print DACVal

        return '%032x' % peripheryDACcode

    def splitPerihperyDACs(self, code, perc=True):
        if perc:
            percEightBit = float( 2**8 )
            percNineBit = float( 2**9 )
            percThirteenBit = float( 2**13 )
        else:
            percEightBit, percNineBit, percThirteenBit = 1, 1, 1


        code = bin(int(code, 16)).split('0b')[-1]
        print code
        d = {'V_ThA': int(code[115:], 2) / percThirteenBit,
            'V_tpref_fine': int(code[103:112], 2) / percNineBit,
            'V_tpref_coarse': int(code[88:96], 2) / percEightBit, 
            'I_tpbufout': int(code[80:88], 2) / percEightBit, 
            'I_tpbufin': int(code[72:80], 2) / percEightBit, 
            'I_disc2': int(code[64:72], 2) / percEightBit, 
            'I_disc1': int(code[56:64], 2) / percEightBit, 
            'V_casc_preamp': int(code[48:56], 2) / percEightBit, 
            'V_gnd': int(code[40:48], 2) / percEightBit, 
            'I_preamp': int(code[32:40], 2) / percEightBit, 
            'V_fbk': int(code[24:32], 2) / percEightBit, 
            'I_krum': int(code[16:24], 2) / percEightBit, 
            'I_pixeldac': int(code[8:16], 2) / percEightBit, 
            'V_casc_reset': int(code[:8], 2) / percEightBit}

        print 'PeripheryDAC values in',
        if perc:
            print 'percent:'
        else:
            print 'DAC:'
        print yaml.dump(d, indent=4, default_flow_style=False)
        print 

        return d

    def periheryDACsDictToCode(self, d, perc=True):
        if perc:
            percEightBit = float( 2**8 )
            percNineBit = float( 2**9 )
            percThirteenBit = float( 2**13 )
        else:
            percEightBit, percNineBit, percThirteenBit = 1, 1, 1

        code = 0
        code |= int(d['V_ThA'] * percThirteenBit) 
        code |= (int(d['V_tpref_fine'] * percNineBit) << 25 - 9)
        code |= (int(d['V_tpref_coarse'] * percEightBit) << 40 - 8)
        code |= (int(d['I_tpbufout'] * percEightBit) << 48 - 8)
        code |= (int(d['I_tpbufin'] * percEightBit) << 56 - 8)
        code |= (int(d['I_disc2'] * percEightBit) << 64 - 8)
        code |= (int(d['I_disc1'] * percEightBit) << 72 - 8)
        code |= (int(d['V_casc_preamp'] * percEightBit) << 80 - 8)
        code |= (int(d['V_gnd'] * percEightBit) << 88 - 8)
        code |= (int(d['I_preamp'] * percEightBit) << 96 - 8)
        code |= (int(d['V_fbk'] * percEightBit) << 104 - 8)
        code |= (int(d['I_krum'] * percEightBit) << 112 - 8)
        code |= (int(d['I_pixeldac'] * percEightBit) << 120 - 8)
        code |= (int(d['V_casc_reset'] * percEightBit) << 128 - 8)

        return hex(code).split('0x')[-1][:-1]

    def getTHLfromVolt(self, V, slot):
        return self.THLCalib[slot - 1][abs(V - self.voltCalib).argmin()]

    def getVoltFromTHL(self, THL, slot):
        return self.voltCalib[slot - 1][np.argwhere(self.THLCalib == THL)]

    def getVoltFromTHLFit(self, THL, slot):
        if self.THLEdgesLow[slot - 1] is None:
            return THL

        edges = zip(self.THLEdgesLow[slot - 1], self.THLEdgesHigh[slot - 1])
        for i, edge in enumerate(edges):
            if THL >= edge[0] and THL <= edge[1]:
                break
        # else:
        #     return None

        params = self.THLFitParams[slot - 1][i]
        if i == 0:
            return self.erfStdFit(THL, *params)
        else:
            return self.linearFit(THL, *params)

    def normal(self, x, A, mu, sigma):
        return A * np.exp(-(x - mu)/(2 * sigma**2))

    def getColor(self, c, N, idx):
        import matplotlib as mpl
        cmap = mpl.cm.get_cmap(c)
        norm = mpl.colors.Normalize(vmin=0.0, vmax=N - 1)
        return cmap(norm(idx))

    def getColorBar(self, ax, cbMin, cbMax, N=20, label=None, rotation=90):
        # Plot colorbar
        from matplotlib.colors import ListedColormap
        cmap = mpl.cm.get_cmap('viridis', N)
        norm = mpl.colors.Normalize(vmin=cbMin, vmax=cbMax)
        cBar = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical')

        # cBar.ax.invert_yaxis()
        cBar.formatter.set_powerlimits((0, 0))
        cBar.ax.yaxis.set_offset_position('right')
        cBar.update_ticks()

        labels = np.linspace(cbMin, cbMax, N + 1)
        locLabels = np.linspace(cbMin, cbMax, N)
        loc = labels + abs(labels[1] - labels[0])*.5
        cBar.set_ticks(loc)
        cBar.ax.set_yticklabels(['%.1f' % loc for loc in locLabels], rotation=rotation, verticalalignment='center')
        cBar.outline.set_visible(False)
        cBar.set_label(label)

    # https://stackoverflow.com/questions/12090503/listing-available-com-ports-with-python
    def getSerialPorts(self):
        """ Lists serial port names

            :raises EnvironmentError:
                On unsupported or unknown platforms
            :returns:
                A list of the serial ports available on the system
        """
        if sys.platform.startswith('win'):
            ports = ['COM%s' % (i + 1) for i in range(256)]
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            # this excludes your current terminal "/dev/tty"
            ports = glob.glob('/dev/tty[A-Za-z]*')
        elif sys.platform.startswith('darwin'):
            ports = glob.glob('/dev/tty.*')
        else:
            raise EnvironmentError('Unsupported platform')

        result = []
        for port in ports:
            try:
                s = serial.Serial(port)
                s.close()
                result.append(port)
            except (OSError, serial.SerialException):
                pass
        return result

if __name__ == '__main__':
    main()
