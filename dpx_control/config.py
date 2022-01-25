from __future__ import print_function

import serial
import os
import os.path
import numpy as np
import json
import configparser

try:
  basestring
except NameError:
  basestring = str

class Config(object):
    def getConfigDPX(self, portName, baudRate, configFn, Ikrum=None, eye_lens=False):
        self.ser = serial.Serial(self.portName, self.baudRate)
        assert self.ser.is_open, 'Error: Could not establish serial connection!'

        # Read config
        if self.eye_lens:
            slot_fac = 1
        else:
            slot_fac = 3
        
        # Standard values
        self.peripherys = ['dc310bc864508230768080ff0064'] * slot_fac
        self.OMR = ['39ffc0'] * slot_fac
        self.THLs = ['153b'] * slot_fac
        self.confBits = ['0' * 512] * slot_fac
        self.pixelDAC = ['0' * 512] * slot_fac
        self.binEdges = ['0' * 1024] * slot_fac

        # Set standard Ikrum
        if Ikrum is not None:
            for p_idx, p in enumerate( self.peripherys ):
                d = self.splitPerihperyDACs(p + self.THLs[p_idx], perc=False, show=False)
                d['I_krum'] = Ikrum[p_idx]
                code = self.periheryDACsDictToCode(d, perc=False)
                self.peripherys[p_idx] = code[:-4]

        if self.configFn is None:
            print('Config file not found. Please run THL equalization first. Using standard values.')
        else:
            if isinstance(self.configFn, basestring):
                if not os.path.isfile(conf):
                    print('Config file not found. Please run THL equalization first. Using standard values.')
                else:
                    print('Only one config file specified. Using same file for all slots.')
                    for slot in self.slot_range:
                        self.readConfig(self.configFn, slot=slot)
            else:
                for idx, conf in enumerate(self.configFn):
                    if conf is not None:
                        if os.path.isfile(conf):
                            self.readConfig(conf, slot=idx+1)
                        else:
                            print('Config file not found. Please run THL equalization first. Using standard values.')

        self.initDPX()

        # If no THL calibration file is present
        if self.thl_calib_files is None:
            print('Warning: No THL calibration file set! Functions using THL edges won\'t be usuable')
            return

        # Load THL calibration data
        for thl_calib_file in self.thl_calib_files:
            if thl_calib_file is None or not os.path.isfile(thl_calib_file):
                print('Need the specified THL Calibration file %s!' % thl_calib_file)
                self.THLEdgesLow.append( None ) 
                self.THLEdgesHigh.append( None )
                self.THLEdges.append( None )
                self.THLFitParams.append( None )
            else:
                if thl_calib_file.endswith('.json'):
                    # JSON
                    print(thl_calib_file)
                    with open(thl_calib_file, 'r') as f:
                        d = json.load(f)
                else:
                    print('Cannot open THL calibration file with extension %s' % thl_calib_file.split('.')[-1])
                    self.THLEdgesLow.append( None ) 
                    self.THLEdgesHigh.append( None )
                    self.THLEdges.append( None )
                    self.THLFitParams.append( None )

                self.load_THLEdges(d)
    
    def load_THLEdges(self, d):
        self.voltCalib.append( np.asarray(d['Volt']) / max(d['Volt']) )
        self.THLCalib.append( np.asarray(d['ADC']) )

        THLLow, THLHigh, THLFitParams = self.THLCalibToEdges(d)

        # Rounding
        THLLow = np.ceil(THLLow)
        THLHigh = np.floor(THLHigh)

        self.THLEdgesLow.append(THLLow), self.THLEdgesHigh.append(THLHigh), self.THLFitParams.append(THLFitParams)
        print(self.THLEdgesLow[-1], self.THLEdgesHigh[-1])
        
        # Combine
        THLEdges = []
        for i in range(len(self.THLEdgesLow[-1])):
            THLEdges += list( np.arange(self.THLEdgesLow[-1][i], self.THLEdgesHigh[-1][i] + 1) )
        self.THLEdges.append( THLEdges ) 

    def splitPerihperyDACs(self, code, perc=False, show=False):
        if perc:
            percEightBit = float( 2**8 )
            percNineBit = float( 2**9 )
            percThirteenBit = float( 2**13 )
        else:
            percEightBit, percNineBit, percThirteenBit = 1, 1, 1

        if show:
            print(code)
        code = format(int(code, 16), 'b')
        if show:
            print(code)
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

        if show:
            print('PeripheryDAC values in', end='')
            if perc:
                print('percent:')
            else:
                print('DAC:')

        return d

    def periheryDACsDictToCode(self, d, perc=False):
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

        return '%04x' % code

    def readConfig(self, configFn, slot=1):
        config = configparser.ConfigParser()
        config.read(configFn)

        # Mandatory sections
        sectionList = ['Peripherydac', 'OMR', 'Equalisation']

        # Check if set, else throw error
        for section in config.sections():
            assert section in sectionList, 'Config: %s is a mandatory section and has to be specified' % section

        # Read Peripherydac
        if 'code' in config['Peripherydac']:
            self.peripherys = config['Peripherydac']
        else:
            PeripherydacDict = {}
            PeripherydacCodeList = ['V_ThA', 'V_tpref_fine', 'V_tpref_coarse', 'I_tpbufout', 'I_tpbufin', 'I_disc1', 'I_disc2', 'V_casc_preamp', 'V_gnd', 'I_preamp', 'V_fbk', 'I_krum', 'I_pixeldac', 'V_casc_reset']
            for PeripherydacCode in PeripherydacCodeList:
                assert PeripherydacCode in config['Peripherydac'], 'Config: %s has to be specified in OMR section!' % PeripherydacCode

                PeripherydacDict[PeripherydacCode] = int(float( config['Peripherydac'][PeripherydacCode] ))
            self.peripherys[slot - 1] = self.periheryDACsDictToCode(PeripherydacDict)[:-4]
            self.THLs[slot - 1] = '%04x' % PeripherydacDict['V_ThA']

        # Read OMR
        if 'code' in config['OMR']:
            self.OMR[slot - 1] = config['OMR']['code']
        else:
            OMRList = []

            OMRCodeList = ['OperationMode', 'GlobalShutter', 'PLL', 'Polarity', 'AnalogOutSel', 'AnalogInSel', 'OMRDisableColClkGate']
            for OMRCode in OMRCodeList:
                assert OMRCode in config['OMR'], 'Config: %s has to be specified in OMR section!' % OMRCode

                OMRList.append(config['OMR'][OMRCode])

            self.OMR[slot - 1] = OMRList

        # Equalisation
        # confBits - optional field
        if 'confBits' in config['Equalisation']:
            self.confBits[slot - 1] = config['Equalisation']['confBits']
        else:
            # Use all pixels
            self.confBits[slot - 1] = '00' * 256

        # pixelDAC
        assert 'pixelDAC' in config['Equalisation'], 'Config: pixelDAC has to be specified in Equalisation section!'
        self.pixelDAC[slot - 1] = config['Equalisation']['pixelDAC']

        # binEdges
        assert 'binEdges' in config['Equalisation'], 'Config: binEdges has to be specified in Equalisation section!'
        self.binEdges[slot - 1] = config['Equalisation']['binEdges']

        return

    def setConfig_gui(self, config):
        self.THLs[0] = config['v_tha']
        self.confBits[0] = config['confbits']
        self.pixelDAC[0] = config['pixeldac']
        return

    def writeConfig(self, configFn, slot=1):
        config = configparser.ConfigParser()
        d = self.splitPerihperyDACs(self.peripherys[slot-1] + self.THLs[slot-1])
        d['V_ThA'] = int(self.THLs[slot-1], 16)
        config['Peripherydac'] = d

        if not isinstance(self.OMR[slot-1], basestring):
            OMRCodeList = ['OperationMode', 'GlobalShutter', 'PLL', 'Polarity', 'AnalogOutSel', 'AnalogInSel', 'OMRDisableColClkGate']
            config['OMR'] = {OMRCode: self.OMR[slot-1][i] for i, OMRCode in enumerate(OMRCodeList)}
        else:
            config['OMR'] = {'code': self.OMR[slot-1]}

        config['Equalisation'] = {'pixelDAC': self.pixelDAC[slot-1], 'confBits': self.confBits[slot-1], 'binEdges': ''.join(self.binEdges['Slot%d' % slot])}

        with open(configFn, 'w') as configFile:
            config.write(configFile)

