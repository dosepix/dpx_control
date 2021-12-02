#!/usr/bin/env python
import dpx_control
import dpx_control.multi as mu

PORT = ['/dev/ttyUSB0']
CONFIG_DIR = ['config/']
CHIP_NUMS = [[22, 6, 109]]
# PORT = ['/dev/ttyUSB0', '/dev/ttyUSB1']
# CONFIG_DIR = ['config/', 'config/']
# CHIP_NUMS = [[22, 6, 109], [101, 154, None]]
CONFIG_FN = [[CONFIG_DIR[idx] + 'DPXConfig_%d.conf' % CHIP if CHIP is not None else None for CHIP in CHIP_NUMS[idx]] for idx in range(len(CONFIG_DIR))]
N_DET = len(PORT)
thl_calib_files = None 

def main():
    # Establish connection
    dpxObjects = []
    for idx in range(N_DET): 
        print(CONFIG_FN[idx])
        dpxObjects.append( dpx_control.Dosepix(PORT[idx], 2e6, CONFIG_FN[idx], thl_calib_files=thl_calib_files) )

    dpx_multi = mu.DosepixMulti(dpxObjects, [[1, 2, 3], [1, 2]])
    dpx_multi.measureToT(frames=None, sync=True)

    # Close connection
    for dpx in dpxObjects:
        dpx.close()

if __name__ == '__main__':
    main()

