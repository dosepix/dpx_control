#!/usr/bin/env python
import dpx_control
import json

# Important files
THL_CALIB_FILES = ['THLCalibration/THLCalib_%d.p' % slot for slot in [22, 6, 109]]
BIN_EDGES_FILE = None # 'Dennis1_binEdges.hck'
PARAMS_FILE = 'energyConversion/paramsDict_DPX22_6_109.hck'

GEN_BIN_EDGES = False
GEN_BIN_EDGES_RANDOM = False
GEN_BIN_EDGES_UNIFORM = True
# BIN_EDGES_RANDOM_FN = 'binEdgesRandom_DPX22_6_109_v2.hck'
BIN_EDGES_RANDOM_FN = 'binEdgesUniform_DPX22_6_109_v2.hck'

if GEN_BIN_EDGES:
    import bin_edges_random as ber

    if PARAMS_FILE.endswith('.p'):
        paramsDict = json.load(open(PARAMS_FILE, 'rb'))

    binEdgesDict = {}
    for slot in range(1, 3 + 1):
        if GEN_BIN_EDGES_RANDOM:
            binEdges = ber.getBinEdgesRandom(NPixels=256, edgeMin=12, edgeMax=100, edgeOvfw=430, uniform=False)
        elif GEN_BIN_EDGES_UNIFORM:
	    # Generate edges for multiple energy regions
            binEdges = []
	    energy_start, energy_range = 10, 90
	    for idx in range(4):
	    	edges = ber.getBinEdgesUniform(NPixels=256, edgeMin=energy_start + idx*energy_range, edgeMax=energy_start + (idx + 1)*energy_range, edgeOvfw=430)
		binEdges.append( edges )
        binEdgesDict['Slot%d' % slot] = binEdges
    json.dump(binEdgesDict, BIN_EDGES_RANDOM_FN)

BIN_EDGES = {'Slot1': [12, 18, 21, 24.5, 33.5, 43, 53.5, 66.5, 81.5, 97, 113, 131.5, 151.5, 173, 200.5, 236, 430],
                'Slot2': [12, 17, 31, 40, 45.5, 50.5, 60.5, 68, 91, 102.5, 133, 148, 163, 196, 220, 257, 430],
                'Slot3': [32, 37, 47, 57.6, 68.5, 80, 91.5, 104, 117, 131, 145, 163.5, 183.5, 207.5, 234.5, 269.5, 430]}
PORT = '/dev/tty.usbserial-A907PD5F'
CONFIG_FN = 'DPXConfig.conf'

def main():
    dpx = dpx_control.Dosepix(PORT, 2e6, CONFIG_FN, bin_edges_file=BIN_EDGES, params_file=PARAMS_FILE, thl_calib_files=THL_CALIB_FILES)
    dpx.measureDose(slot=1, measurement_time=0, freq=False, frames=1000, logTemp=False, intPlot=True, conversion_factors=None)
    dpx.close()

if __name__ == '__main__':
    main()

