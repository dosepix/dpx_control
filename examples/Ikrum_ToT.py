#!/usr/bin/env python
import dpx_func_python

PORT = '/dev/ttyUSB0'
CONFIG_FN = 'DPXConfig.conf'
CONFIG_DIR = 'config/'
CHIP_NUMS = [22, 6, 109]
IKRUM = [5, 10, 20] 

def main():
    # Establish connection
    thl_calib_files = [CONFIG_DIR + '/THLCalib_%d.hck' % CHIP for CHIP in CHIP_NUMS] 
    dpx = dpx_func_python.Dosepix(PORT, 2e6, CONFIG_DIR + '/' + CONFIG_FN, thl_calib_files=thl_calib_files)

    # Change Ikrum values
    for chip_idx in range(3):
        d = dpx.splitPerihperyDACs(dpx.peripherys + dpx.THLs[chip_idx], perc=False)
        d['I_krum'] = IKRUM[chip_idx]
        code = dpx.periheryDACsDictToCode(d, perc=False)
        dpx.peripherys = code[:-4]
        dpx.DPXWritePeripheryDACCommand(chip_idx + 1, code)

    for slot in range(1, 4):
        print dpx.DPXReadPeripheryDACCommand(slot)

    # Measure ToT
    dpx.measureToT(slot=[1, 2, 3], intPlot=True, storeEmpty=True, logTemp=True)

    # Close connection
    dpx.close()

if __name__ == '__main__':
    main()

