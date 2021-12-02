#!/usr/bin/env python3
from __future__ import print_function
import dpx_control

PORT = '/dev/ttyUSB0'
PORT = '/dev/ttyUSB0'
CHIP_NUMS = [22, 6, 109]
CONFIG_DIR = 'config/'
CONFIG_FN = [CONFIG_DIR + '/DPXConfig_%d.conf' % c for c in CHIP_NUMS]
IKRUM = [10, 30, 50] 
THL_SHIFT = 0

def main():
    # Establish connection
    thl_calib_files = [CONFIG_DIR + '/THLCalib_%d.hck' % CHIP for CHIP in CHIP_NUMS] 
    dpx = dpx_control.Dosepix(PORT, 2e6, CONFIG_FN, thl_calib_files=thl_calib_files)

    # Change Ikrum values
    new_peripherys = []
    for chip_idx in range(3):
        d = dpx.splitPerihperyDACs(dpx.peripherys[chip_idx] + dpx.THLs[chip_idx], perc=False, show=True)
        d['I_krum'] = IKRUM[chip_idx]
        code = dpx.periheryDACsDictToCode(d, perc=False)
        code = code[:-4] + '%04x' % (int(dpx.THLs[chip_idx], 16) - THL_SHIFT)
        dpx.THLs[chip_idx] = '%04x' % (int(dpx.THLs[chip_idx], 16) - THL_SHIFT)

        dpx.peripherys[chip_idx] = code[:-4]
        new_peripherys.append( code[:-4] )
        dpx.DPXWritePeripheryDACCommand(chip_idx + 1, code)

    for slot in range(1, 4):
        print(dpx.DPXReadPeripheryDACCommand(slot))

    dpx.measureToT(slot=[1], intPlot=True, cnt=10, storeEmpty=False, logTemp=True, meas_time=None, external_THL=False) # , paramsDict=hck.load('config/paramsDict_22_6_109_Ikrum_10_30_50_THLShift_10_slot2_fail.hck'))

    # Close connection
    dpx.close()

if __name__ == '__main__':
    main()

