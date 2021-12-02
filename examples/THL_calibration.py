#!/usr/bin/env python
import dpx_control
import os

PORT = '/dev/ttyUSB0'
CONFIG_FN = 'DPXConfig.conf'
CONFIG_DIR = 'config/'
CHIP_NUMS = [22, 6, 109]

def main():
    # Create config dir if not already existing
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)

    # THL measurements
    print('Ensure that detector %d is inserted at slot 1' % CHIP_NUMS[0]) 
    raw_input('Press any key to proceed')
    for chip_idx in range(len(CHIP_NUMS)):
        print 'Measuring THL of chip %d' % CHIP_NUMS[chip_idx]
        dpx = dpx_control.Dosepix(PORT, 2e6, CONFIG_DIR + CONFIG_FN)
        dpx.measureTHL(1, fn=CONFIG_DIR + 'THLCalib_%d.hck' % CHIP_NUMS[chip_idx], plot=False)
        dpx.close()

        if chip_idx != len(CHIP_NUMS):
            print('Please disconnect board and insert detector %d into slot 1' % CHIP_NUMS[chip_idx+1])
            raw_input('Reconnect and press any key to proceed')

    dpx.close()

if __name__ == '__main__':
    main()

