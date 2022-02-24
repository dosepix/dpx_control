from __future__ import print_function

import os
import os.path
import numpy as np
import json
import configparser
from control import control
from dpx_support import DPXSupport as dps

try:
    basestring
except NameError:
    basestring = str

class Config(object):
    def __init__(self,
            port_name,
            baud_rate,
            config_fn,
            eye_lens=False,
            Ikrum=None):
        self.eye_lens = eye_lens

        # Set config parameters
        self.port_name = port_name
        self.baud_rate = baud_rate
        self.config_fn = config_fn
        self.Ikrum = Ikrum

        # Read config
        if self.eye_lens:
            slot_fac = 1
        else:
            slot_fac = 3

        # Eye lens board only uses slot1
        if self.eye_lens:
            self.slot_range = [1]
        else:
            if config_fn is None:
                self.slot_range = [1, 2, 3]
            else:
                self.slot_range = [idx for idx in range(
                    1, len(self.config_fn) + 1) if self.config_fn[idx - 1] is not None]

        print('Slots')
        print(self.slot_range)

        # Standard config values
        self.peripherys = ['dc310bc864508230768080ff0064'] * slot_fac
        self.OMR = ['39ffc0'] * slot_fac
        self.THLs = ['153b'] * slot_fac
        self.conf_bits = ['0' * 512] * slot_fac
        self.pixel_DAC = ['0' * 512] * slot_fac
        self.bin_edges = ['0' * 1024] * slot_fac

        self.volt_calib = []
        self.THL_calib = []

    # Read config and return THL calibration
    def get_config_DPX(self, thl_calib_files):
        THL_edges_low, THL_edges_high = [], []
        THL_edges = [] * 3
        THL_fit_params = []

        # Set standard Ikrum
        if self.Ikrum is not None:
            for p_idx, p in enumerate(self.peripherys):
                d = self.split_perihpery_DACs(
                    p + self.THLs[p_idx], perc=False, show=False)
                d['I_krum'] = self.Ikrum[p_idx]
                code = self.perihery_DACs_dict_to_code(d, perc=False)
                self.peripherys[p_idx] = code[:-4]

        if self.config_fn is None:
            print(
                'Config file not found. Please run THL equalization first. Using standard values.')
        else:
            if isinstance(self.config_fn, basestring):
                if not os.path.isfile(conf):
                    print(
                        'Config file not found. Please run THL equalization first. Using standard values.')
                else:
                    print(
                        'Only one config file specified. Using same file for all slots.')
                    for slot in self.slot_range:
                        self.read_config(self.config_fn, slot=slot)
            else:
                for idx, conf in enumerate(self.config_fn):
                    if conf is not None:
                        if os.path.isfile(conf):
                            self.read_config(conf, slot=idx + 1)
                        else:
                            print(
                                'Config file not found. Please run THL equalization first. Using standard values.')

        # self.dosepix.control.init_DPX()

        # If no THL calibration file is present
        if thl_calib_files is None:
            print(
                'Warning: No THL calibration file set! Functions using THL edges won\'t be usuable')
            return THL_edges_low, THL_edges_high, THL_edges, THL_fit_params

        # Load THL calibration data
        for thl_calib_file in thl_calib_files:
            if thl_calib_file is None or not os.path.isfile(thl_calib_file):
                print(
                    'Need the specified THL Calibration file %s!' %
                    thl_calib_file)
                THL_edges_low.append(None)
                THL_edges_high.append(None)
                THL_edges.append(None)
                THL_fit_params.append(None)
            else:
                if thl_calib_file.endswith('.json'):
                    # JSON
                    print(thl_calib_file)
                    with open(thl_calib_file, 'r') as f:
                        d = json.load(f)
                else:
                    print('Cannot open THL calibration file with extension %s' %
                          thl_calib_file.split('.')[-1])
                    THL_edges_low.append(None)
                    THL_edges_high.append(None)
                    THL_edges.append(None)
                    THL_fit_params.append(None)

                self.load_THL_edges(d)

        return THL_edges_low, THL_edges_high, THL_edges, THL_fit_params

    def load_THL_edges(self, d):
        self.volt_calib.append(np.asarray(d['Volt']) / max(d['Volt']))
        self.THL_calib.append(np.asarray(d['ADC']))

        THLLow, THLHigh, THL_fit_params = dps.THL_calib_to_edges(d, self.eye_lens)

        # Rounding
        THLLow = np.ceil(THLLow)
        THLHigh = np.floor(THLHigh)

        self.THL_edges_low.append(THLLow), self.THL_edges_high.append(
            THLHigh), self.THL_fit_params.append(THL_fit_params)
        print(self.THL_edges_low[-1], self.THL_edges_high[-1])

        # Combine
        THL_edges = []
        for i in range(len(self.THL_edges_low[-1])):
            THL_edges += list(np.arange(self.THL_edges_low[-1]
                             [i], self.THL_edges_high[-1][i] + 1))
        self.THL_edges.append(THL_edges)

    def split_perihpery_DACs(self, code, perc=False, show=False):
        if perc:
            perc_eight_bit = float(2**8)
            perc_nine_bit = float(2**9)
            perc_thirteen_bit = float(2**13)
        else:
            perc_eight_bit, perc_nine_bit, perc_thirteen_bit = 1, 1, 1

        if show:
            print(code)
        code = format(int(code, 16), 'b')
        if show:
            print(code)
        d = {'V_ThA': int(code[115:], 2) / perc_thirteen_bit,
             'V_tpref_fine': int(code[103:112], 2) / perc_nine_bit,
             'V_tpref_coarse': int(code[88:96], 2) / perc_eight_bit,
             'I_tpbufout': int(code[80:88], 2) / perc_eight_bit,
             'I_tpbufin': int(code[72:80], 2) / perc_eight_bit,
             'I_disc2': int(code[64:72], 2) / perc_eight_bit,
             'I_disc1': int(code[56:64], 2) / perc_eight_bit,
             'V_casc_preamp': int(code[48:56], 2) / perc_eight_bit,
             'V_gnd': int(code[40:48], 2) / perc_eight_bit,
             'I_preamp': int(code[32:40], 2) / perc_eight_bit,
             'V_fbk': int(code[24:32], 2) / perc_eight_bit,
             'I_krum': int(code[16:24], 2) / perc_eight_bit,
             'I_pixeldac': int(code[8:16], 2) / perc_eight_bit,
             'V_casc_reset': int(code[:8], 2) / perc_eight_bit}

        if show:
            print('PeripheryDAC values in', end='')
            if perc:
                print('percent:')
            else:
                print('DAC:')

        return d

    def perihery_DACs_dict_to_code(self, d, perc=False):
        if perc:
            perc_eight_bit = float(2**8)
            perc_nine_bit = float(2**9)
            perc_thirteen_bit = float(2**13)
        else:
            perc_eight_bit, perc_nine_bit, perc_thirteen_bit = 1, 1, 1

        code = 0
        code |= int(d['V_ThA'] * perc_thirteen_bit)
        code |= (int(d['V_tpref_fine'] * perc_nine_bit) << 25 - 9)
        code |= (int(d['V_tpref_coarse'] * perc_eight_bit) << 40 - 8)
        code |= (int(d['I_tpbufout'] * perc_eight_bit) << 48 - 8)
        code |= (int(d['I_tpbufin'] * perc_eight_bit) << 56 - 8)
        code |= (int(d['I_disc2'] * perc_eight_bit) << 64 - 8)
        code |= (int(d['I_disc1'] * perc_eight_bit) << 72 - 8)
        code |= (int(d['V_casc_preamp'] * perc_eight_bit) << 80 - 8)
        code |= (int(d['V_gnd'] * perc_eight_bit) << 88 - 8)
        code |= (int(d['I_preamp'] * perc_eight_bit) << 96 - 8)
        code |= (int(d['V_fbk'] * perc_eight_bit) << 104 - 8)
        code |= (int(d['I_krum'] * perc_eight_bit) << 112 - 8)
        code |= (int(d['I_pixeldac'] * perc_eight_bit) << 120 - 8)
        code |= (int(d['V_casc_reset'] * perc_eight_bit) << 128 - 8)

        return '%04x' % code

    def read_config(self, config_fn, slot=1):
        config = configparser.ConfigParser()
        config.read(config_fn)

        # Mandatory sections
        section_list = ['Peripherydac', 'OMR', 'Equalisation']

        # Check if set, else throw error
        for section in config.sections():
            assert section in section_list, 'Config: %s is a mandatory section and has to be specified' % section

        # Read Peripherydac
        if 'code' in config['Peripherydac']:
            self.peripherys = config['Peripherydac']
        else:
            PeripherydacDict = {}
            PeripherydacCodeList = [
                'V_ThA',
                'V_tpref_fine',
                'V_tpref_coarse',
                'I_tpbufout',
                'I_tpbufin',
                'I_disc1',
                'I_disc2',
                'V_casc_preamp',
                'V_gnd',
                'I_preamp',
                'V_fbk',
                'I_krum',
                'I_pixeldac',
                'V_casc_reset']
            for PeripherydacCode in PeripherydacCodeList:
                assert PeripherydacCode in config['Peripherydac'], 'Config: %s has to be specified in OMR section!' % PeripherydacCode

                PeripherydacDict[PeripherydacCode] = int(
                    float(config['Peripherydac'][PeripherydacCode]))
            self.peripherys[slot - 1] = self.perihery_DACs_dict_to_code(PeripherydacDict)[:-4]
            self.THLs[slot - 1] = '%04x' % PeripherydacDict['V_ThA']

        # Read OMR
        if 'code' in config['OMR']:
            self.OMR[slot - 1] = config['OMR']['code']
        else:
            OMRList = []

            OMRCodeList = [
                'OperationMode',
                'GlobalShutter',
                'PLL',
                'Polarity',
                'AnalogOutSel',
                'AnalogInSel',
                'OMRDisableColClkGate']
            for OMRCode in OMRCodeList:
                assert OMRCode in config['OMR'], 'Config: %s has to be specified in OMR section!' % OMRCode
                OMRList.append(config['OMR'][OMRCode])

            self.OMR[slot - 1] = OMRList

        # Equalisation
        # conf_bits - optional field
        if 'conf_bits' in config['Equalisation']:
            self.conf_bits[slot - 1] = config['Equalisation']['conf_bits']
        else:
            # Use all pixels
            self.conf_bits[slot - 1] = '00' * 256

        # pixel_DAC
        assert 'pixel_DAC' in config['Equalisation'], 'Config: pixel_DAC has to be specified in Equalisation section!'
        self.pixel_DAC[slot - 1] = config['Equalisation']['pixel_DAC']

        # bin_edges
        assert 'bin_edges' in config['Equalisation'], 'Config: bin_edges has to be specified in Equalisation section!'
        self.bin_edges[slot - 1] = config['Equalisation']['bin_edges']

        return

    def setConfig_gui(self, config):
        self.THLs[0] = config['v_tha']
        self.conf_bits[0] = config['confbits']
        self.pixel_DAC[0] = config['pixeldac']
        return

    def writeConfig(self, config_fn, slot=1):
        config = configparser.ConfigParser()
        d = self.split_perihpery_DACs(
            self.peripherys[slot - 1] + self.THLs[slot - 1])
        d['V_ThA'] = int(self.THLs[slot - 1], 16)
        config['Peripherydac'] = d

        if not isinstance(self.OMR[slot - 1], basestring):
            OMRCodeList = [
                'OperationMode',
                'GlobalShutter',
                'PLL',
                'Polarity',
                'AnalogOutSel',
                'AnalogInSel',
                'OMRDisableColClkGate']
            config['OMR'] = {OMRCode: self.OMR[slot - 1][i]
                             for i, OMRCode in enumerate(OMRCodeList)}
        else:
            config['OMR'] = {'code': self.OMR[slot - 1]}

        config['Equalisation'] = {'pixel_DAC': self.pixel_DAC[slot - 1],
                                  'conf_bits': self.conf_bits[slot - 1],
                                  'bin_edges': ''.join(self.bin_edges['Slot%d' % slot])}

        with open(config_fn, 'w') as configFile:
            config.write(configFile)
