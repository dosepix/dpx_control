from __future__ import print_function

import numpy as np
import time
from collections import namedtuple
import json

DEBUG = False
import dpx_control.dpx_settings as ds

class Control():
    def initDPX(self):
        # Not required for eye lens dosimetry hardware
        if not self.eye_lens:
            # Start HV
            self.HVSetDac('0000')
            print('HV DAC set to %s' % self.HVGetDac())

            self.HVActivate()
            # Set voltage to 3.3V
            self.VCVoltageSet3V3()

            # Check if HV is enabled
            print('Check if HV is activated...', end='')
            # Try five times
            for i in range(5):
                if self.HVGetState():
                    print('done!')
                    break
                else:
                    self.HVActivate()
            else:
                assert 'HV could not be activated!'

            print('Voltage set to %s' % self.VCGetVoltage())

            # Disable LED
            self.MCLEDdisable()

        # Wait
        time.sleep(0.5)

        # Global reset
        for i in self.slot_range:
            # Do three times
            for j in range(3):
                self.DPXGlobalReset(i)
        time.sleep(0.5)

        # = Write Settings =
        for i in self.slot_range:
            self.DPXWriteConfigurationCommand(i, self.confBits[i-1])
            self.DPXWriteOMRCommand(i, self.OMR[i-1])

            # Merge peripheryDACcode and THL value
            self.DPXWritePeripheryDACCommand(i, self.peripherys[i-1] + self.THLs[i-1])
            print('Periphery DAC on Slot %d set to: %s' % (i, self.DPXReadPeripheryDACCommand(i)))

            self.DPXWritePixelDACCommand(i, self.pixelDAC[i-1])
            print('Pixel DAC on Slot %d set to: %s' % (i, self.DPXReadPixelDACCommand(i)))
        print()
        time.sleep(0.5)

        # = Data Reset =
        for i in self.slot_range:
            self.DPXDataResetCommand(i)

        # = Dummy Readout =
        for i in self.slot_range:
            self.DPXReadToTDataDosiModeCommand(i)

        # = Calibration parameters =
        if self.params_file is not None:
            self.paramsDict = {}
            for slot in self.slot_range:
                if self.params_file[slot - 1] is None:
                    self.paramsDict['Slot%d' % slot] = None
                    continue

                if self.params_file[slot - 1].endswith('.json'):
                    with open(self.params_file[slot - 1], 'r') as f:
                        paramsDict = json.load(f)
                        paramsDict = {int(key): paramsDict[key] for key in paramsDict.keys()}
                        self.paramsDict['Slot%d' % slot] = paramsDict
                else:
                    print('Warning: No parameters for the bin edges specified. Using default values.')
                    self.paramsDict = None
        else:
            print('Warning: No parameters for the bin edges specified. Using default values.')
            self.paramsDict = None

        # = Bin Edges =
        # Check if bin edges are given as dictionary or file
        self.binEdges = {'Slot%d' % slot: [] for slot in self.slot_range}
        if self.bin_edges_file is not None:
            # Dictionary
            if isinstance(self.bin_edges_file, dict):
                # Using ToT values
                if self.paramsDict is None:
                    self.binEdges = self.bin_edges_file
                    for slot in self.slot_range:
                        self.setBinEdgesToT(slot, self.binEdges['Slot%d' % slot])
                # Convert to energy
                else:
                    for slot in self.slot_range:
                        binEdgesList = self.setBinEdges(slot, self.paramsDict['Slot%d' % slot], self.bin_edges_file['Slot%d' % slot])
                        self.binEdges['Slot%d' % slot].insert(0, binEdgesList)
            
            # File
            else:
                for slot in self.slot_range:
                    be_fn = self.bin_edges_file[slot - 1]
                    if be_fn is None:
                        continue

                    if be_fn.endswith('.json'):
                        with open(be_fn, 'r') as f:
                            binEdges = json.load(f)
                    else:
                        continue

                    # If shape is larger than 2, bin edges are used for shifted
                    # bin edges with more than one region!
                    if len(np.asarray(binEdges).shape) <= 2:
                        binEdges = np.asarray([binEdges])

                    # bin edges are specified for a shifted dose measurement
                    # idx = 0
                    for idx in reversed(range(len(binEdges))):
                        # Convert to energy
                        if self.paramsDict['Slot%d' % slot] is not None:
                            binEdgesList = self.setBinEdges(slot, self.paramsDict['Slot%d' % slot], binEdges[idx])
                            self.binEdges['Slot%d' % slot].insert(0, binEdgesList)
                        # Using ToT values
                        else:
                            self.binEdges['Slot%d' % slot] = binEdges['Slot%d' % slot]
                            self.setBinEdgesToT(slot, self.binEdges['Slot%d' % slot])
        else:
            gray = [0, 1, 3, 2, 6, 4, 5, 7, 15, 13, 12, 14, 10, 11, 9, 8]
            for i in self.slot_range:
                for binEdge in range(16):
                    gc = gray[binEdge]
                    binEdges = ('%01x' % gc + '%03x' % (20*binEdge + 15)) * 256
                    self.DPXWriteSingleThresholdCommand(i, binEdges)

        # = Empty Bins =
        for i in self.slot_range:
            # Loop over bins
            for col in range(1, 16 + 1):
                self.DPXWriteColSelCommand(i, 16 - col)
                # Dummy readout
                self.DPXReadBinDataDosiModeCommand(i)

    def getResponse(self):
        res = self.ser.readline()
        '''
        while res[0] != '\x02':
            res = self.ser.readline()
        '''
        if DEBUG:
            print(res)
        return res

    def getDPXResponse(self):
        # res = ''
        # while not res:
        res = self.getResponse()

        if DEBUG:
            print('Length:', res[11:17])
        cmdLength = int( res[11:17] )

        if DEBUG:
            print('CmdData:', res[17:17+cmdLength])
        cmdData = res[17:17+cmdLength]

        return cmdData

    def getReceiverFromSlot(self, slot):
        if slot == 1:
            receiver = ds._receiverDPX1
        elif slot == 2:
            receiver = ds._receiverDPX2
        elif slot == 3:
            receiver = ds._receiverDPX3
        else:
            assert 'Error: Function needs to access one of the three slots.'

        return receiver

    # Convert the bin edges from energy to ToT and write them to the designated registers
    def setBinEdges(self, slot, paramDict, binEdgesEnergyList):
        # assert len(paramDict) == 256, 'getBinEdges: Number of pixels in paramDict differs from 256!'
        if len(paramDict) != 256:
            paramDict = self.fillParamDict(paramDict)

        # Check if paramDict was made using THL calibration.
        # If so, additional parameters h and k are present in the dict
        if 'h' in paramDict[list(paramDict.keys())[0]].keys():
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
            # if h == 1 and k == 0:
            #     binEdgesToT = self.energyToToTFitHyp(beEnergy, a, b, c, t)
            # else:
            binEdgesToT = self.EnergyToToTSimple(beEnergy, a, b, c, t, h, k)
            print( binEdgesToT )

            # Round the values - do not use floor function as this leads to bias
            binEdgesToT = np.around( binEdgesToT )
            binEdgesToT = np.nan_to_num(binEdgesToT)
            binEdgesToT[binEdgesToT < 0] = 0
            binEdgesToT[binEdgesToT > 4095] = 4095
            binEdgesList.append( binEdgesToT )

        # Transpose matrix to get pixel values
        binEdgesList = np.asarray(binEdgesList).T
        self.setBinEdgesToT(slot, binEdgesList)

        # self.binEdges['Slot%d' % slot] = binEdgesList
        return binEdgesList
        # binEdgesTotal.append( binEdgesTotal )
        # self.binEdges = binEdgesTotal

    def setBinEdgesToT(self, slot, binEdgesList):
        # The indices of the bins are specified via the following gray code
        gray = [0, 1, 3, 2, 6, 4, 5, 7, 15, 13, 12, 14, 10, 11, 9, 8]

        cmdTotal = ''
        for idx, gc in enumerate(gray):
            # Construct command
            cmd = ''.join( [('%01x' % gc) + ('%03x' % be) for be in np.asarray(binEdgesList[idx], dtype=int)] )
            self.DPXWriteSingleThresholdCommand(slot, cmd)
            cmdTotal += cmd

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

    # === HV SECTION ===
    def HVSetDac(self, DAC):
        assert len(DAC) == 4, 'Error: DAC command has to be of size 4!'

        self.sendCmd([ds._receiverHV, ds._subReceiverNone, ds._senderPC, ds._HVsetDAC, '%06d' % len(DAC), DAC, ds._CRC])

    def HVGetDac(self):
        self.sendCmd([ds._receiverHV, ds._subReceiverNone, ds._senderPC, ds._HVgetDAC, ds._commandNoneLength, ds._commandNone, ds._CRC])

        return self.getDPXResponse()

    def HVActivate(self):
        self.sendCmd([ds._receiverHV, ds._subReceiverNone, ds._senderPC, ds._HVenable, ds._commandNoneLength, ds._commandNone, ds._CRC])

    def HVDeactivate(self):
        self.sendCmd([ds._receiverHV, ds._subReceiverNone, ds._senderPC, ds._HVdisable, ds._commandNoneLength, ds._commandNone, ds._CRC])

    def HVGetState(self):
        self.sendCmd([ds._receiverHV, ds._subReceiverNone, ds._senderPC, ds._HVisEnabled, ds._commandNoneLength, ds._CRC])

        res = int(self.getDPXResponse())

        if res:
            return True
        else:
            return False

    # === VC SECTION ===
    def VCVoltageSet3V3(self):
        self.sendCmd([ds._receiverVC, ds._subReceiverNone, ds._senderPC, ds._VCset3V3, ds._commandNoneLength, ds._commandNone, ds._CRC])

    def VCVoltageSet1V8(self):
        self.sendCmd([ds._receiverVC, ds._subReceiverNone, ds._senderPC, ds._VCset1V8, ds._commandNoneLength, ds._commandNone, ds._CRC])

    def VCGetVoltage(self):
        self.sendCmd([ds._receiverVC, ds._subReceiverNone, ds._senderPC, ds._VCgetVoltage, ds._commandNoneLength, ds._commandNone, ds._CRC])

        res = int(self.getDPXResponse())
        if res:
            return '3.3V'
        else:
            return '1.8V'

    # === MC SECTION ===
    def MCLEDenable(self):
        self.sendCmd([ds._receiverMC, ds._subReceiverNone, ds._senderPC, ds._MCLEDenable, ds._commandNoneLength, ds._commandNone, ds._CRC])

    def MCLEDdisable(self):
        self.sendCmd([ds._receiverMC, ds._subReceiverNone, ds._senderPC, ds._MCLEDdisable, ds._commandNoneLength, ds._commandNone, ds._CRC])

    def MCGetADCvalue(self):
        self.sendCmd([ds._receiverMC, ds._subReceiverNone, ds._senderPC, ds._MCgetADCvalue, ds._commandNoneLength, ds._commandNone, ds._CRC])

        return self.getDPXResponse()

    def MCGetFirmwareVersion(self):
        self.sendCmd([self.receiverMC, ds._subReceiverNone, ds._senderPC, ds._MCgetVersion, ds._commandNoneLength, ds._commandNone, ds._CRC])

        return self.getDPXResponse()

    # === DPX SECTION ===
    def OMRListToHex(self, OMRCode):
        OMRCodeList = OMRCode
        OMRTypeList = [ds._OMROperationMode,
        ds._OMRGlobalShutter,
        ds._OMRPLL,
        ds._OMRPolarity,
        ds._OMRAnalogOutSel,
        ds._OMRAnalogInSel,
        ds._OMRDisableColClkGate]

        OMRCode = 0x000000
        for i, OMR in enumerate(OMRCodeList):
            OMRCode |= getattr(OMRTypeList[i], OMR)

        OMRCode = '%04x' % (OMRCode) # hex(OMRCode).split('0x')[-1]

        return OMRCode

    def DPXWriteOMRCommand(self, slot, OMRCode):
        if type(OMRCode) is list:
            OMRCode = self.OMRListToHex(OMRCode)
        if type(OMRCode) is int:
            OMRCode = '%x' % OMRCode

        self.sendCmd([self.getReceiverFromSlot(slot), ds._subReceiverNone, ds._senderPC, ds._DPXwriteOMRCommand, '%06d' % len(OMRCode), OMRCode, ds._CRC])

        return self.getDPXResponse()

    def DPXReadOMRCommand(self, slot):
        self.sendCmd([self.getReceiverFromSlot(slot), ds._subReceiverNone, ds._senderPC, ds._DPXreadOMRCommand, ds._commandNoneLength, ds._commandNone, ds._CRC])

        return self.getDPXResponse()

    def DPXReadDigitalThresholdsCommand(self, slot):
        self.sendCmd([self.getReceiverFromSlot(slot), ds._subReceiverNone, ds._senderPC, ds._DPXreadDigitalThresholdsCommand, ds._commandNoneLength, ds._commandNone, ds._CRC])

        return self.getDPXResponse()

    def DPXGlobalReset(self, slot):
        self.sendCmd([self.getReceiverFromSlot(slot), ds._subReceiverNone, ds._senderPC, ds._DPXglobalResetCommand, ds._commandNoneLength, ds._commandNone, ds._CRC])

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

        self.sendCmd([self.getReceiverFromSlot(slot), ds._subReceiverNone, ds._senderPC, ds._DPXwriteConfigurationCommand, '%06d' % len(confBits), confBits, ds._CRC])

        return self.getDPXResponse()

    def DPXWriteSingleThresholdCommand(self, slot, THFn, file=False):
        if file:
            with open(THFn, 'r') as f:
                TH = f.read()
        else:
            TH = THFn

        self.sendCmd([self.getReceiverFromSlot(slot), ds._subReceiverNone, ds._senderPC, ds._DPXwriteSingleThresholdCommand, '%06d' % len(TH), TH, ds._CRC])

        return self.getDPXResponse()

    def DPXWriteColSelCommand(self, slot, col):
        colCode = '%02x' % col
        self.sendCmd([self.getReceiverFromSlot(slot), ds._subReceiverNone, ds._senderPC, ds._DPXwriteColSelCommand, '%06d' % len(colCode), colCode, ds._CRC])

        return self.getDPXResponse()

    def DPXWritePeripheryDACCommand(self, slot, code):
        self.sendCmd([self.getReceiverFromSlot(slot), ds._subReceiverNone, ds._senderPC, ds._DPXwritePeripheryDACCommand, '%06d' % len(code), code, ds._CRC])

        return self.getDPXResponse()

    def DPXReadPeripheryDACCommand(self, slot):
        self.sendCmd([self.getReceiverFromSlot(slot), ds._subReceiverNone, ds._senderPC, ds._DPXreadPeripheryDACCommand, ds._commandNoneLength, ds._commandNone, ds._CRC])

        return self.getDPXResponse()

    def DPXWritePixelDACCommand(self, slot, code, file=False):
        if file:
            with open(code, 'r') as f:
                code = f.read().split('\n')[0]

        # else: use code string

        self.sendCmd([self.getReceiverFromSlot(slot), ds._subReceiverNone, ds._senderPC, ds._DPXwritePixelDACCommand, '%06d' % len(code), code, ds._CRC])

        return self.getDPXResponse()

    def DPXReadPixelDACCommand(self, slot):
        self.sendCmd([self.getReceiverFromSlot(slot), ds._subReceiverNone, ds._senderPC, ds._DPXreadPixelDACCommand, ds._commandNoneLength, ds._commandNone, ds._CRC])

        return self.getDPXResponse()

    def DPXDataResetCommand(self, slot):
        self.sendCmd([self.getReceiverFromSlot(slot), ds._subReceiverNone, ds._senderPC, ds._DPXdataResetCommand, ds._commandNoneLength, ds._commandNone, ds._CRC])

        return self.getDPXResponse()

    def DPXReadBinDataDosiModeCommand(self, slot):
        self.sendCmd([self.getReceiverFromSlot(slot), ds._subReceiverNone, ds._senderPC, ds._DPXreadBinDataDosiModeCommand, ds._commandNoneLength, ds._commandNone, ds._CRC])

        res = self.getDPXResponse()
        return self.convertToDecimal( res )

    def DPXReadToTDataDosiModeCommand(self, slot):
        self.sendCmd([self.getReceiverFromSlot(slot), ds._subReceiverNone, ds._senderPC, ds._DPXreadToTDataDosiModeCommand, ds._commandNoneLength, ds._commandNone, ds._CRC])

        return self.convertToDecimal(self.getDPXResponse())

    def DPXReadToTDataDosiModeMultiCommand(self, slot):
        self.sendCmd([self.getReceiverFromSlot(slot), ds._subReceiverNone, ds._senderPC, ds._DPXreadToTDataDosiModeMultiCommand, ds._commandNoneLength, ds._commandNone, ds._CRC])

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
        self.sendCmd([self.getReceiverFromSlot(slot), ds._subReceiverNone, ds._senderPC, ds._DPXreadToTDataIntegrationModeCommand, ds._commandNoneLength, ds._commandNone, ds._CRC])

        return self.convertToDecimal(self.getDPXResponse(), 6)

    def DPXReadToTDatakVpModeCommand(self, slot):
        self.sendCmd([self.getReceiverFromSlot(slot), ds._subReceiverNone, ds._senderPC, ds._DPXreadToTDatakVpModeCommand, ds._commandNoneLength, ds._commandNone, ds._CRC])

        return self.convertToDecimal(self.getDPXResponse(), 2)

    def DPXGeneralTestPulse(self, slot, length):
        lengthHex = '%04x' % length

        self.sendCmd([self.getReceiverFromSlot(slot), ds._subReceiverNone, ds._senderPC, ds._DPXgeneralTestPulse, '%06d' % len(lengthHex), lengthHex, ds._CRC])

        return self.getDPXResponse()

    def DPXGeneralMultiTestPulse(self, slot, length):
        lengthHex = '%04x' % length

        self.sendCmd([self.getReceiverFromSlot(slot), ds._subReceiverNone, ds._senderPC, ds._DPXgeneralMultiTestPulse, '%06d' % len(lengthHex), lengthHex, ds._CRC])

        return self.getDPXResponse()

    def sendCmd(self, cmdList):
        # Typical command string:
        # RRrrrSSSssCCCllllllcccc
        # R - receiver
        # r - subreceiver
        # s - sender
        # C - command
        # l - command length
        # c - CRC (unused, usually set to FFFF)

        cmdOut = [str.encode(str(cmd)) for cmd in cmdList]
        cmdOut.insert(0, ds._startOfTransmission)
        cmdOut.append( ds._endOfTransmission )
        cmdOut = b''.join(cmdOut)
        self.ser.write(cmdOut)
        return

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

