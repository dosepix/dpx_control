#!/usr/bin/env python
import dpx_control

PORT = '/dev/ttyUSB0'
CONFIG_DIR = 'config/'
CHIP_NUMS = [22, 6, 109]
CONFIG_FN = [CONFIG_DIR + 'DPXConfig_%d.conf' % CHIP if CHIP is not None else None for CHIP in CHIP_NUMS]

def main():
    # Establish connection
    thl_calib_files = None 
    dpx = dpx_control.Dosepix(PORT, 2e6, CONFIG_FN, thl_calib_files=thl_calib_files)

    # Measure ToT
    dpx.measurePC(slot=[1, 2, 3], measurement_time=0, frames=10000, intPlot=True)

    # Close connection
    dpx.close()

if __name__ == '__main__':
    main()

