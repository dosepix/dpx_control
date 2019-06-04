import serial
import os
import os.path
import yaml
import numpy as np
import hickle
import cPickle
import configparser

class Config(object):
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

        if self.configFn is None or not os.path.isfile(self.configFn):
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
        if self.thl_calib_files is None:
            print 'Warning: No THL calibration file set! Functions using the THL edges won\'t be usuable'
            return

        # Load THL calibration data
        for thl_calib_file in self.thl_calib_files:
            if thl_calib_file is None or not os.path.isfile(thl_calib_file):
                print 'Need the specified THL Calibration file %s!' % thl_calib_file
                self.THLEdgesLow.append( None ) # [   0,  662, 1175, 1686, 2193, 2703, 3215, 3728, 4241, 4751, 5263, 5777, 6284, 6784, 7312]
                self.THLEdgesHigh.append( None ) # [ 388,  889, 1397, 1904, 2416, 2927, 3440, 3952, 4464, 4976, 5488, 5991, 6488, 7009, 8190]
                self.THLEdges.append( None )
                self.THLFitParams.append( None )

            else:
                if thl_calib_file.endswith('.p'):
                    d = cPickle.load(open(thl_calib_file, 'rb'))
                else:
                    d = hickle.load(thl_calib_file)

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
            config['Slot%d' % i] = {'pixelDAC': self.pixelDAC[i-1], 'binEdges': self.binEdges[i-1], 'confBits': self.confBits[i-1], 'binEdges': ''.join(self.binEdges[i-1]), 'THL': self.THLs[i-1]}

        with open(configFn, 'w') as configFile:
            config.write(configFile)

