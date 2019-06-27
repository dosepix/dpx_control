#!/usr/bin/env python
import dpx_func_python

PORT = '/dev/ttyUSB0'
CONFIG_FN = 'DPXConfig.conf'
CONFIG_DIR = 'config/'
CHIP_NUMS = [22, 6, 109]
IKRUM = [5, 10, 20] 

def main():
    # Establish connection
    thl_calib_files = None # [CONFIG_DIR + '/THLCalib_%d.hck' % CHIP for CHIP in CHIP_NUMS] 
    dpx = dpx_func_python.Dosepix(PORT, 2e6, CONFIG_DIR + '/' + CONFIG_FN, thl_calib_files=thl_calib_files)

    # Measure ToT
    dpx.measureToT(slot=[1, 2, 3], intPlot=True, cnt=10000, storeEmpty=True, logTemp=True)

    # Close connection
    dpx.close()

if __name__ == '__main__':
    main()

