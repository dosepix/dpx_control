#!/usr/bin/env python
import dpx_func_python
import os

PORT = '/dev/ttyUSB0'
CONFIG_FN = 'DPXConfig.conf'
CONFIG_DIR = 'config/'
CHIP_NUMS = [22, 6, 109]
CALIB_THL = False

def main():
    # Create config dir if not already existing
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)

    # THL measurements
    if CALIB_THL:
        print('Ensure that detector %d is inserted at slot 1' % CHIP_NUMS[0]) 
        raw_input('Press any key to proceed')
        for chip_idx in enumerate(CHIP_NUMS):
            print 'Measuring THL of chip %d' % CHIP_NUMS[chip_idx]
            dpx = dpx_func_python.Dosepix(PORT, 2e6)
            dpx.measureTHL(1, fn=CONFIG_DIR + 'THLCalib_%d.hck' % CHIP_NUMS[chip_idx], plot=False)
            dpx.close()

            if chip_idx != len(CHIP_NUMS):
                print('Please disconnect board and insert detector %d into slot 1' % CHIP_NUMS[chip_idx+1])
                raw_input('Reconnect and press any key to proceed')

    # Threshold equalization
    print('Please disconnect board and insert detectors in their specified slots')
    raw_input('Reconnect and press any key to proceed')
    if CALIB_THL:
        thl_calib_files = [CONFIG_DIR + '/THLCalib_%d.hck' % CHIP for CHIP in CHIP_NUMS] 
    else:
        thl_calib_files = None
    dpx = dpx_func_python.Dosepix(PORT, 2e6, thl_calib_files=thl_calib_files)
    dpx.thresholdEqualizationConfig(CONFIG_DIR + '/' + CONFIG_FN, I_pixeldac=None, reps=1, intPlot=False, resPlot=True)
    dpx.close()

if __name__ == '__main__':
    main()

