#!/usr/bin/env python
import dpx_control
import numpy as np

REUSE_CONFIG = False
PORT = '/dev/ttyUSB0'
CONFIG_DIR = 'config/'
CHIP_NUMS = [22, None, None] # 101, 109]

if REUSE_CONFIG:
    IKRUMS = np.arange(37, 51) 
else:
    IKRUMS = [20, 40, 50] # [10, 20, 30, 40, 50]

PARAMS_FILES = None
BIN_EDGES_FILES = None
MEAS_TIME = 60 * 30

def main():
    if REUSE_CONFIG:
        CONFIG_FN = [CONFIG_DIR + '%d/DPXConfig_%d_Ikrum%d.conf' % (CHIP, CHIP, 20) if CHIP is not None else None for CHIP in CHIP_NUMS]

    # Loop over different Ikrum values and perform measurements
    for Ikrum in IKRUMS:
        if not REUSE_CONFIG:
            CONFIG_FN = [CONFIG_DIR + '%d/DPXConfig_%d_Ikrum%d.conf' % (CHIP, CHIP, Ikrum) if CHIP is not None else None for CHIP in CHIP_NUMS]

        print('=== Starting ToT Measurement for Ikrum %d ===' % Ikrum)
        # Establish connection
        thl_calib_files = None 
        dpx = dpx_control.Dosepix(PORT, 2e6, CONFIG_FN, thl_calib_files=thl_calib_files, 
                params_file=PARAMS_FILES, bin_edges_file=BIN_EDGES_FILES)
        
        # Change Ikrum
        if REUSE_CONFIG:
            # Ikrum is changed in this section, else it is changed via configuration file
            for chip_idx in range(3):
                d = dpx.splitPerihperyDACs(dpx.peripherys[chip_idx] + dpx.THLs[chip_idx], perc=False, show=True)
                d['I_krum'] = Ikrum
                code = dpx.periheryDACsDictToCode(d, perc=False)
                code = code[:-4] + '%04x' % (int(dpx.THLs[chip_idx], 16))

                dpx.peripherys[chip_idx] = code[:-4]
                dpx.DPXWritePeripheryDACCommand(chip_idx + 1, code)

        # Measure ToT
        dpx.measureToT(slot=[1, 2, 3], intPlot=False, cnt=10000, storeEmpty=False, logTemp=True, meas_time=MEAS_TIME,
                        outDir='ToTMeasurement_%s_Ikrum%d/' % ('_'.join([str(chip) for chip in CHIP_NUMS]), Ikrum))

        # Close connection
        dpx.close()
        print()

if __name__ == '__main__':
    main()

