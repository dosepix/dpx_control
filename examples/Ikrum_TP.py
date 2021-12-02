#!/usr/bin/env python
import dpx_control

PORT = '/dev/ttyUSB0'
CONFIG_FN = 'DPXConfig_22_6_109.conf'
CONFIG_DIR = 'config/'
CHIP_NUMS = [22, 6, 109]
IKRUM = [10, 30, 50] 
THL_SHIFT = 10 

def main():
    # Establish connection
    thl_calib_files = None # [CONFIG_DIR + '/THLCalib_%d.hck' % CHIP for CHIP in CHIP_NUMS] 
    dpx = dpx_control.Dosepix(PORT, 2e6, CONFIG_DIR + '/' + CONFIG_FN, thl_calib_files=thl_calib_files)

    # Change Ikrum values
    for chip_idx in range(3):
        # Type
        d = dpx.splitPerihperyDACs(dpx.peripherys + dpx.THLs[chip_idx], perc=False)
        d['I_krum'] = IKRUM[chip_idx]
        # d['I_tpbufin'] = 255
        # d['I_tpbufout'] = 255
        code = dpx.periheryDACsDictToCode(d, perc=False)
        code = code[:-4] + '%04x' % (int(dpx.THLs[chip_idx], 16) - THL_SHIFT)
        dpx.THLs[chip_idx] = '%04x' % (int(dpx.THLs[chip_idx], 16) - THL_SHIFT)
        print code

        dpx.peripherys = code[:-4]
        dpx.DPXWritePeripheryDACCommand(chip_idx + 1, code)
        dpx.splitPerihperyDACs(dpx.DPXReadPeripheryDACCommand(chip_idx + 1), perc=False)
        print dpx.DPXReadOMRCommand(chip_idx + 1)

        # Measure Test Pulses
        # import time
        # import numpy as np
        # dpx.TPtoToT(slot=chip_idx+1, column=0, low=460, high=505, step=2, outFn='TPpixel_THL%d.hck' % THL_SHIFT) 
        dpx.TPtoToT(slot=chip_idx+1, column='all', low=250, high=512, step=5, outFn='TP_Slot%d.hck' % (chip_idx + 1)) # outFn='TPpixelTemperature/TPpixelTemperature_t%d.hck' % time.time())
        # dpx.TPfindMax(slot=chip_idx + 1, column=0)

        # for voltage in [100, 200, 300, 400]:
        #     dpx.TPTime(slot=chip_idx + 1, column=0, voltage=voltage, timeRange=np.linspace(0, 100, 100), outFn='TPTime/TPTime%d_Slot%d.hck' % (voltage, chip_idx + 1))
        # time.sleep(60)
       #  dpx.ToTtoTHL(slot=1, column=0, THLstep=1, valueLow=460, valueHigh=512, valueStep=1, energy=False, plot=False, outFn='ToTtoTHL_slot1_test.hck')

    # Close connection
    dpx.close()

if __name__ == '__main__':
    main()

