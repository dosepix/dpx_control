#!/usr/bin/env python
import dpx_func_python

PORT = '/dev/ttyUSB0'
CONFIG_FN = 'DPXConfig_22_6_109.conf'
CONFIG_DIR = 'config/'
CHIP_NUMS = [22, 6, 109]
IKRUM = [10, 30, 50] 
THL_SHIFT = 880 # 10

def main():
    # Establish connection
    thl_calib_files = [CONFIG_DIR + '/THLCalib_%d.hck' % CHIP for CHIP in CHIP_NUMS] 
    dpx = dpx_func_python.Dosepix(PORT, 2e6, CONFIG_DIR + '/' + CONFIG_FN, thl_calib_files=thl_calib_files)

    # Change Ikrum values
    new_peripherys = []
    for chip_idx in range(3):
        d = dpx.splitPerihperyDACs(dpx.peripherys + dpx.THLs[chip_idx], perc=False)
        d['I_krum'] = IKRUM[chip_idx]
        code = dpx.periheryDACsDictToCode(d, perc=False)
        code = code[:-4] + '%04x' % (int(dpx.THLs[chip_idx], 16) - THL_SHIFT)
        dpx.THLs[chip_idx] = '%04x' % (int(dpx.THLs[chip_idx], 16) - THL_SHIFT)

        dpx.peripherys = code[:-4]
        new_peripherys.append( code[:-4] )
        dpx.DPXWritePeripheryDACCommand(chip_idx + 1, code)

    for slot in range(1, 4):
        print dpx.DPXReadPeripheryDACCommand(slot)

<<<<<<< HEAD
    # import hickle as hck
    # Measure ToT 
    # for THLshift in range(-1000, -100, 10):
    #     print THLshift
    #     for slot in range(3):
    #         THL = int(dpx.THLs[slot], 16) - THL_SHIFT
    #         THL_idx = list(dpx.THLEdges[slot]).index(THL)
    #         THL = '%04x' % (dpx.THLEdges[slot][THL_idx + THLshift])
    #         dpx.DPXWritePeripheryDACCommand(slot + 1, new_peripherys[slot] + THL)
    #     dpx.measureToT(slot=[1, 2, 3], intPlot=False, cnt=50000, storeEmpty=False, logTemp=True, meas_time=60, paramsDict=hck.load('config/paramsDict_22_6_109_Ikrum_10_30_50_THLShift_10_slot2_fail.hck'))
    # dpx.measurePC(slot=2, measurement_time=0, frames=1000, intPlot=True)

    dpx.measureToT(slot=[1], intPlot=True, cnt=10000, storeEmpty=False, logTemp=True, meas_time=None, external_THL=False) # , paramsDict=hck.load('config/paramsDict_22_6_109_Ikrum_10_30_50_THLShift_10_slot2_fail.hck'))
=======
    # Measure ToT
    dpx.measureToT(slot=[1, 2, 3], intPlot=True, storeEmpty=True, logTemp=True)
>>>>>>> 7e2a9b2c979288ec1985b0b193cab92e1810443c

    # Close connection
    dpx.close()

if __name__ == '__main__':
    main()

