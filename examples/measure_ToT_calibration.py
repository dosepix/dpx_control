#!/usr/bin/env python
import dpx_func_python

PORT = '/dev/ttyUSB0'
CONFIG_DIR = 'config/'
CHIP_NUMS = [22, 6, 109]
IKRUMS = [10, 20, 30, 40, 50]
PARAMS_FILES = None
BIN_EDGES_FILES = None
MEAS_TIME = 30 * 60

def main():
    # Loop over different Ikrum values and perform measurements
    for Ikrum in IKRUMS:
        print('=== Starting ToT Measurement for Ikrum %d ===' % Ikrum)
        CONFIG_FN = [CONFIG_DIR + '%d/DPXConfig_%d_Ikrum%d.conf' % (CHIP, CHIP, Ikrum) if CHIP is not None else None for CHIP in CHIP_NUMS]
        # Establish connection
        thl_calib_files = None 
        dpx = dpx_func_python.Dosepix(PORT, 2e6, CONFIG_FN, thl_calib_files=thl_calib_files, 
                params_file=PARAMS_FILES, bin_edges_file=BIN_EDGES_FILES)

        # Measure ToT
        dpx.measureToT(slot=[1, 2, 3], intPlot=False, cnt=10000, storeEmpty=False, logTemp=True, meas_time=MEAS_TIME,
                        outDir='ToTMeasurement_%s_Ikrum%d/' % ('_'.join([str(chip) for chip in CHIP_NUMS]), Ikrum))

        # Close connection
        dpx.close()
        print()

if __name__ == '__main__':
    main()

