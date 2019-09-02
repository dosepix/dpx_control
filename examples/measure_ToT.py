#!/usr/bin/env python
import dpx_func_python

PORT = '/dev/ttyUSB0'
CONFIG_FN = 'DPXConfig_Robot.conf'
CONFIG_DIR = 'config/'
CHIP_NUMS = [999]

def main():
    # Establish connection
    thl_calib_files = None 
    dpx = dpx_func_python.Dosepix(PORT, 2e6, CONFIG_DIR + '/' + CONFIG_FN, thl_calib_files=thl_calib_files)

    # Measure ToT
    # dpx.measureToT(slot=[1, 2, 3], intPlot=True, cnt=10000, storeEmpty=False, logTemp=True)
    dpx.measurePC(slot=2, measurement_time=0, frames=10000, intPlot=False)

    # Close connection
    dpx.close()

if __name__ == '__main__':
    main()

