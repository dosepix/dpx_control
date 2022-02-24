from __future__ import print_function
import dpx_control.dpx_settings as ds
from dpx_support import DPXsupport as dps

import numpy as np
import time
from collections import namedtuple
import json

DEBUG = False

class Control():
    def __init__(self):
        return

    def init_DPX(self, config, params_file, bin_edges_file,
            eye_lens=False):

        # Get from config
        slot_range, conf_bits = config.slot_range, config.conf_bits
        OMR, THLs = config.OMR, config.THLs
        pixel_DAC = config.pixel_DAC
        peripherys = config.perypheries

        # Not required for eye lens dosimetry hardware
        if not eye_lens:
            # Start HV
            self.HV_set_Dac('0000')
            print('HV DAC set to %s' % self.HVGetDac())

            self.HV_activate()
            # Set voltage to 3.3V
            self.VC_voltage_set_3V3()

            # Check if HV is enabled
            print('Check if HV is activated...', end='')
            # Try five times
            for i in range(5):
                if self.HV_get_state():
                    print('done!')
                    break
                self.HV_activate()
            else:
                assert False, 'HV could not be activated!'

            print('Voltage set to %s' % self.VC_get_voltage())

            # Disable LED
            self.MC_LED_disable()

        # Wait
        time.sleep(0.5)

        # Global reset
        for slot in slot_range:
            # Do three times
            for j in range(3):
                self.DPX_global_reset(slot)
        time.sleep(0.5)

        # = Write Settings =
        for i in slot_range:
            self.DPX_write_configuration_command(i, conf_bits[i - 1])
            self.DPX_write_OMR_command(i, OMR[i - 1])

            # Merge peripheryDACcode and THL value
            self.DPX_write_periphery_DAC_command(
                i, peripherys[i - 1] + THLs[i - 1])
            print('Periphery DAC on Slot %d set to: %s' %
                  (i, self.DPX_read_periphery_DAC_command(i)))

            self.DPX_write_pixel_DAC_command(i, pixel_DAC[i - 1])
            print('Pixel DAC on Slot %d set to: %s' %
                  (i, self.DPX_read_pixel_DAC_command(i)))
        print()
        time.sleep(0.5)

        # = Data Reset =
        for i in slot_range:
            self.DPX_data_reset_command(i)

        # = Dummy Readout =
        for i in slot_range:
            self.DPX_read_ToT_data_dosimode_command(i)

        # = Calibration parameters =
        params_dict = {}
        if params_file is not None:
            for slot in slot_range:
                if params_file[slot - 1] is None:
                    params_dict['Slot%d' % slot] = None
                    continue

                if params_file[slot - 1].endswith('.json'):
                    with open(params_file[slot - 1], 'r') as f:
                        params_dict_ = json.load(f)
                        params_dict_ = {
                            int(key): params_dict_[key] for key in params_dict_.keys()}
                        params_dict['Slot%d' % slot] = params_dict_
                else:
                    print(
                        'Warning: No parameters for the bin edges specified. Using default values.')
                    params_dict = None
        else:
            print(
                'Warning: No parameters for the bin edges specified. Using default values.')
            params_dict = None

        # = Bin Edges =
        # Check if bin edges are given as dictionary or file
        bin_edges = {'Slot%d' % slot: [] for slot in slot_range}
        if bin_edges_file is not None:
            # Dictionary
            if isinstance(bin_edges_file, dict):
                # Using ToT values
                if params_dict is None:
                    bin_edges = bin_edges_file
                    for slot in slot_range:
                        self.set_bin_edges_ToT(
                            slot, bin_edges['Slot%d' % slot])
                # Convert to energy
                else:
                    for slot in slot_range:
                        bin_edges_list = self.set_bin_edges(
                            slot, params_dict['Slot%d' % slot], bin_edges_file['Slot%d' % slot])
                        bin_edges['Slot%d' % slot].insert(0, bin_edges_list)

            # File
            else:
                for slot in slot_range:
                    be_fn = bin_edges_file[slot - 1]
                    if be_fn is None:
                        continue

                    if be_fn.endswith('.json'):
                        with open(be_fn, 'r') as f:
                            bin_edges = json.load(f)
                    else:
                        continue

                    # If shape is larger than 2, bin edges are used for shifted
                    # bin edges with more than one region!
                    if len(np.asarray(bin_edges).shape) <= 2:
                        bin_edges = np.asarray([bin_edges])

                    # bin edges are specified for a shifted dose measurement
                    # idx = 0
                    for idx in reversed(range(len(bin_edges))):
                        # Convert to energy
                        if params_dict['Slot%d' % slot] is not None:
                            bin_edges_list = self.set_bin_edges(
                                slot, params_dict['Slot%d' % slot], bin_edges[idx])
                            bin_edges['Slot%d' % slot].insert(0, bin_edges_list)
                        # Using ToT values
                        else:
                            bin_edges['Slot%d' %
                                          slot] = bin_edges['Slot%d' % slot]
                            self.set_bin_edges_ToT(
                                slot, bin_edges['Slot%d' % slot])
        else:
            gray = [0, 1, 3, 2, 6, 4, 5, 7, 15, 13, 12, 14, 10, 11, 9, 8]
            for i in slot_range:
                for binEdge in range(16):
                    gc = gray[binEdge]
                    bin_edges = (
                        '%01x' %
                        gc + '%03x' %
                        (20 * binEdge + 15)) * 256
                    self.DPX_write_single_threshold_command(i, bin_edges)

        # = Empty Bins =
        for i in slot_range:
            # Loop over bins
            for col in range(1, 16 + 1):
                self.DPX_write_col_sel_command(i, 16 - col)
                # Dummy readout
                self.DPX_read_bin_data_dosi_mode_command(i)

        return params_dict, bin_edges

    def get_response(self):
        res = self.ser.readline()
        '''
        while res[0] != '\x02':
            res = self.ser.readline()
        '''
        if DEBUG:
            print(res)
        return res

    def get_DPX_response(self):
        # res = ''
        # while not res:
        res = self.get_response()

        if DEBUG:
            print('Length:', res[11:17])
        cmdLength = int(res[11:17])

        if DEBUG:
            print('CmdData:', res[17:17 + cmdLength])
        cmdData = res[17:17 + cmdLength]

        return cmdData

    def get_receiver_from_slot(self, slot):
        if slot == 1:
            receiver = ds._receiverDPX1
        elif slot == 2:
            receiver = ds._receiverDPX2
        elif slot == 3:
            receiver = ds._receiverDPX3
        else:
            assert 'Error: Function needs to access one of the three slots.'

        return receiver

    # Convert the bin edges from energy to ToT and write them to the
    # designated registers
    def set_bin_edges(self, slot, param_dict, bin_edgesEnergyList):
        # assert len(param_dict) == 256, 'getBinEdges: Number of pixels in param_dict differs from 256!'
        if len(param_dict) != 256:
            param_dict = dps.fill_param_dict(param_dict)

        # Check if param_dict was made using THL calibration.
        # If so, additional parameters h and k are present in the dict
        if 'h' in param_dict[list(param_dict.keys())[0]].keys():
            fEnergyConv = True
        else:
            fEnergyConv = False

        bin_edgesTotal = []
        bin_edgesEnergy = np.asarray(bin_edgesEnergyList)

        bin_edges_list = []
        nanCnt = 0
        for pixel in sorted(param_dict.keys()):
            params = param_dict[pixel]
            a, b, c, t = params['a'], params['b'], params['c'], params['t']
            if fEnergyConv:
                h, k = params['h'], params['k']
            else:
                h, k = 1, 0

            if len(bin_edgesEnergy) > 17:  # bin_edgesEnergy.shape[1] > 0:
                beEnergy = bin_edgesEnergy[pixel]
            else:
                beEnergy = bin_edgesEnergy

            # Convert energy to ToT
            # if h == 1 and k == 0:
            #     bin_edgesToT = self.energyToToTFitHyp(beEnergy, a, b, c, t)
            # else:
            bin_edgesToT = self.energy_to_ToT_simple(beEnergy, a, b, c, t, h, k)
            print(bin_edgesToT)

            # Round the values - do not use floor function as this leads to
            # bias
            bin_edgesToT = np.around(bin_edgesToT)
            bin_edgesToT = np.nan_to_num(bin_edgesToT)
            bin_edgesToT[bin_edgesToT < 0] = 0
            bin_edgesToT[bin_edgesToT > 4095] = 4095
            bin_edges_list.append(bin_edgesToT)

        # Transpose matrix to get pixel values
        bin_edges_list = np.asarray(bin_edges_list).T
        self.set_bin_edges_ToT(slot, bin_edges_list)

        # self.bin_edges['Slot%d' % slot] = bin_edges_list
        return bin_edges_list
        # bin_edgesTotal.append( bin_edgesTotal )
        # self.bin_edges = bin_edgesTotal

    def set_bin_edges_ToT(self, slot, bin_edges_list):
        # The indices of the bins are specified via the following gray code
        gray = [0, 1, 3, 2, 6, 4, 5, 7, 15, 13, 12, 14, 10, 11, 9, 8]

        cmdTotal = ''
        for idx, gc in enumerate(gray):
            # Construct command
            cmd = ''.join([('%01x' % gc) + ('%03x' % be)
                          for be in np.asarray(bin_edges_list[idx], dtype=int)])
            self.DPX_write_single_threshold_command(slot, cmd)
            cmdTotal += cmd

    def clear_bins(self, slot):
        # Clear bins
        # Only call this at start since it takes a long time
        for i in slot:
            for k in range(3):
                self.DPX_data_reset_command(i)
                self.DPX_read_ToT_data_dosimode_command(i)

            for col in range(16):
                self.DPX_write_col_sel_command(i, 16 - col)
                self.DPX_read_bin_data_dosi_mode_command(i)

    # === HV SECTION ===
    def HV_set_Dac(self, DAC):
        assert len(DAC) == 4, 'Error: DAC command has to be of size 4!'

        self.sendCmd([ds._receiverHV,
                      ds._subReceiverNone,
                      ds._senderPC,
                      ds._HVsetDAC,
                      '%06d' % len(DAC),
                      DAC,
                      ds._CRC])

    def HVGetDac(self):
        self.sendCmd([ds._receiverHV,
                      ds._subReceiverNone,
                      ds._senderPC,
                      ds._HVgetDAC,
                      ds._commandNoneLength,
                      ds._commandNone,
                      ds._CRC])

        return self.get_DPX_response()

    def HV_activate(self):
        self.sendCmd([ds._receiverHV,
                      ds._subReceiverNone,
                      ds._senderPC,
                      ds._HVenable,
                      ds._commandNoneLength,
                      ds._commandNone,
                      ds._CRC])

    def HV_deactivate(self):
        self.sendCmd([ds._receiverHV,
                      ds._subReceiverNone,
                      ds._senderPC,
                      ds._HVdisable,
                      ds._commandNoneLength,
                      ds._commandNone,
                      ds._CRC])

    def HV_get_state(self):
        self.sendCmd([ds._receiverHV,
                      ds._subReceiverNone,
                      ds._senderPC,
                      ds._HVisEnabled,
                      ds._commandNoneLength,
                      ds._CRC])

        res = int(self.get_DPX_response())

        if res:
            return True
        else:
            return False

    # === VC SECTION ===
    def VC_voltage_set_3V3(self):
        self.sendCmd([ds._receiverVC,
                      ds._subReceiverNone,
                      ds._senderPC,
                      ds._VCset3V3,
                      ds._commandNoneLength,
                      ds._commandNone,
                      ds._CRC])

    def VCVoltageSet1V8(self):
        self.sendCmd([ds._receiverVC,
                      ds._subReceiverNone,
                      ds._senderPC,
                      ds._VCset1V8,
                      ds._commandNoneLength,
                      ds._commandNone,
                      ds._CRC])

    def VC_get_voltage(self):
        self.sendCmd([ds._receiverVC,
                      ds._subReceiverNone,
                      ds._senderPC,
                      ds._VCgetVoltage,
                      ds._commandNoneLength,
                      ds._commandNone,
                      ds._CRC])

        res = int(self.get_DPX_response())
        if res:
            return '3.3V'
        else:
            return '1.8V'

    # === MC SECTION ===
    def MCLEDenable(self):
        self.sendCmd([ds._receiverMC,
                      ds._subReceiverNone,
                      ds._senderPC,
                      ds._MCLEDenable,
                      ds._commandNoneLength,
                      ds._commandNone,
                      ds._CRC])

    def MC_LED_disable(self):
        self.sendCmd([ds._receiverMC,
                      ds._subReceiverNone,
                      ds._senderPC,
                      ds._MC_LED_disable,
                      ds._commandNoneLength,
                      ds._commandNone,
                      ds._CRC])

    def MCGetADCvalue(self):
        self.sendCmd([ds._receiverMC,
                      ds._subReceiverNone,
                      ds._senderPC,
                      ds._MCgetADCvalue,
                      ds._commandNoneLength,
                      ds._commandNone,
                      ds._CRC])

        return self.get_DPX_response()

    def MCGetFirmwareVersion(self):
        self.sendCmd([self.receiverMC,
                      ds._subReceiverNone,
                      ds._senderPC,
                      ds._MCgetVersion,
                      ds._commandNoneLength,
                      ds._commandNone,
                      ds._CRC])

        return self.get_DPX_response()

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

        OMRCode = '%04x' % (OMRCode)  # hex(OMRCode).split('0x')[-1]

        return OMRCode

    def DPX_write_OMR_command(self, slot, OMRCode):
        if isinstance(OMRCode, list):
            OMRCode = self.OMRListToHex(OMRCode)
        if isinstance(OMRCode, int):
            OMRCode = '%x' % OMRCode

        self.sendCmd([self.get_receiver_from_slot(slot),
                      ds._subReceiverNone,
                      ds._senderPC,
                      ds._DPXwriteOMRCommand,
                      '%06d' % len(OMRCode),
                      OMRCode,
                      ds._CRC])

        return self.get_DPX_response()

    def DPXReadOMRCommand(self, slot):
        self.sendCmd([self.get_receiver_from_slot(slot),
                      ds._subReceiverNone,
                      ds._senderPC,
                      ds._DPXreadOMRCommand,
                      ds._commandNoneLength,
                      ds._commandNone,
                      ds._CRC])

        return self.get_DPX_response()

    def DPXReadDigitalThresholdsCommand(self, slot):
        self.sendCmd([self.get_receiver_from_slot(slot),
                      ds._subReceiverNone,
                      ds._senderPC,
                      ds._DPXreadDigitalThresholdsCommand,
                      ds._commandNoneLength,
                      ds._commandNone,
                      ds._CRC])

        return self.get_DPX_response()

    def DPX_global_reset(self, slot):
        self.sendCmd([self.get_receiver_from_slot(slot),
                      ds._subReceiverNone,
                      ds._senderPC,
                      ds._DPXglobalResetCommand,
                      ds._commandNoneLength,
                      ds._commandNone,
                      ds._CRC])

        return self.get_DPX_response()

    def DPX_write_configuration_command(self, slot, conf_bitsFn, file=False):
        if file:
            with open(conf_bitsFn, 'r') as f:
                conf_bits = f.read()
            conf_bits = conf_bits.split('\n')
            assert len(conf_bits) == 1 or (
                len(conf_bits) == 2 and conf_bits[1] == ''), "Conf-Bits file must contain only one line!"
            conf_bits = conf_bits[0]
        else:
            conf_bits = conf_bitsFn

        self.sendCmd([self.get_receiver_from_slot(slot),
                      ds._subReceiverNone,
                      ds._senderPC,
                      ds._DPXwriteConfigurationCommand,
                      '%06d' % len(conf_bits),
                      conf_bits,
                      ds._CRC])

        return self.get_DPX_response()

    def DPX_write_single_threshold_command(self, slot, THFn, file=False):
        if file:
            with open(THFn, 'r') as f:
                TH = f.read()
        else:
            TH = THFn

        self.sendCmd([self.get_receiver_from_slot(slot),
                      ds._subReceiverNone,
                      ds._senderPC,
                      ds._DPXwriteSingleThresholdCommand,
                      '%06d' % len(TH),
                      TH,
                      ds._CRC])

        return self.get_DPX_response()

    def DPX_write_col_sel_command(self, slot, col):
        colCode = '%02x' % col
        self.sendCmd([self.get_receiver_from_slot(slot),
                      ds._subReceiverNone,
                      ds._senderPC,
                      ds._DPXwriteColSelCommand,
                      '%06d' % len(colCode),
                      colCode,
                      ds._CRC])

        return self.get_DPX_response()

    def DPX_write_periphery_DAC_command(self, slot, code):
        self.sendCmd([self.get_receiver_from_slot(slot),
                      ds._subReceiverNone,
                      ds._senderPC,
                      ds._DPXwritePeripheryDACCommand,
                      '%06d' % len(code),
                      code,
                      ds._CRC])

        return self.get_DPX_response()

    def DPX_read_periphery_DAC_command(self, slot):
        self.sendCmd([self.get_receiver_from_slot(slot),
                      ds._subReceiverNone,
                      ds._senderPC,
                      ds._DPXreadPeripheryDACCommand,
                      ds._commandNoneLength,
                      ds._commandNone,
                      ds._CRC])

        return self.get_DPX_response()

    def DPX_write_pixel_DAC_command(self, slot, code, file=False):
        if file:
            with open(code, 'r') as f:
                code = f.read().split('\n')[0]

        # else: use code string

        self.sendCmd([self.get_receiver_from_slot(slot),
                      ds._subReceiverNone,
                      ds._senderPC,
                      ds._DPXwritePixelDACCommand,
                      '%06d' % len(code),
                      code,
                      ds._CRC])

        return self.get_DPX_response()

    def DPX_read_pixel_DAC_command(self, slot):
        self.sendCmd([self.get_receiver_from_slot(slot),
                      ds._subReceiverNone,
                      ds._senderPC,
                      ds._DPXreadPixelDACCommand,
                      ds._commandNoneLength,
                      ds._commandNone,
                      ds._CRC])

        return self.get_DPX_response()

    def DPX_data_reset_command(self, slot):
        self.sendCmd([self.get_receiver_from_slot(slot),
                      ds._subReceiverNone,
                      ds._senderPC,
                      ds._DPXdataResetCommand,
                      ds._commandNoneLength,
                      ds._commandNone,
                      ds._CRC])

        return self.get_DPX_response()

    def DPX_read_bin_data_dosi_mode_command(self, slot):
        self.sendCmd([self.get_receiver_from_slot(slot),
                      ds._subReceiverNone,
                      ds._senderPC,
                      ds._DPXreadBinDataDosiModeCommand,
                      ds._commandNoneLength,
                      ds._commandNone,
                      ds._CRC])

        res = self.get_DPX_response()
        return self.convertToDecimal(res)

    def DPX_read_ToT_data_dosimode_command(self, slot):
        self.sendCmd([self.get_receiver_from_slot(slot),
                      ds._subReceiverNone,
                      ds._senderPC,
                      ds._DPXreadToTDataDosiModeCommand,
                      ds._commandNoneLength,
                      ds._commandNone,
                      ds._CRC])

        return self.convertToDecimal(self.get_DPX_response())

    def DPXReadToTDataDosiModeMultiCommand(self, slot):
        self.sendCmd([self.get_receiver_from_slot(slot),
                      ds._subReceiverNone,
                      ds._senderPC,
                      ds._DPXreadToTDataDosiModeMultiCommand,
                      ds._commandNoneLength,
                      ds._commandNone,
                      ds._CRC])

        res = self.get_DPX_response()

        # Read response and
        x = [ord(char) for char in res]
        x1 = np.asarray(x[::2])
        x2 = np.asarray(x[1::2])
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
        self.sendCmd([self.get_receiver_from_slot(slot),
                      ds._subReceiverNone,
                      ds._senderPC,
                      ds._DPXreadToTDataIntegrationModeCommand,
                      ds._commandNoneLength,
                      ds._commandNone,
                      ds._CRC])

        return self.convertToDecimal(self.get_DPX_response(), 6)

    def DPXReadToTDatakVpModeCommand(self, slot):
        self.sendCmd([self.get_receiver_from_slot(slot),
                      ds._subReceiverNone,
                      ds._senderPC,
                      ds._DPXreadToTDatakVpModeCommand,
                      ds._commandNoneLength,
                      ds._commandNone,
                      ds._CRC])

        return self.convertToDecimal(self.get_DPX_response(), 2)

    def DPXGeneralTestPulse(self, slot, length):
        lengthHex = '%04x' % length

        self.sendCmd([self.get_receiver_from_slot(slot),
                      ds._subReceiverNone,
                      ds._senderPC,
                      ds._DPXgeneralTestPulse,
                      '%06d' % len(lengthHex),
                      lengthHex,
                      ds._CRC])

        return self.get_DPX_response()

    def DPXGeneralMultiTestPulse(self, slot, length):
        lengthHex = '%04x' % length

        self.sendCmd([self.get_receiver_from_slot(slot),
                      ds._subReceiverNone,
                      ds._senderPC,
                      ds._DPXgeneralMultiTestPulse,
                      '%06d' % len(lengthHex),
                      lengthHex,
                      ds._CRC])

        return self.get_DPX_response()

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
        cmdOut.append(ds._endOfTransmission)
        cmdOut = b''.join(cmdOut)
        self.ser.write(cmdOut)
        return

    def getBinEdges(
            self,
            slot,
            energyDict,
            param_dict,
            transposePixelMatrix=False):
        a, b, c, t = param_dict['a'], param_dict['b'], param_dict['c'], param_dict['t']
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
            ToTList = self.energyToToT(
                energyList, a[pixel], b[pixel], c[pixel], d[pixel])

            for binEdge in range(16):
                grayC = grayCode[binEdge]
                ToT = int(ToTList[binEdge])

                binEdgeString += ('%01x' % grayC)
                binEdgeString += ('%03x' % ToT)

        return binEdgeString
