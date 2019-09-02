#!/usr/bin/env python
import numpy as np
import hickle
import sys
sys.path.insert(0, '../dpx_func_python/')
import bin_edges_random as ber

CONFIG_DIR = 'config/'
BIN_EDGES_RANDOM_FN = CONFIG_DIR + 'binEdgesUniform_DPX22_6_109_Ikrum_10_40.hck'
GEN_BIN_EDGES_RANDOM = False
ENERGY_START, ENERGY_RANGE = 10, 40

binEdgesDict = {}
for slot in range(1, 3 + 1):
    if GEN_BIN_EDGES_RANDOM:
        binEdges = ber.getBinEdgesRandom(NPixels=256, edgeMin=12, edgeMax=100, edgeOvfw=430, uniform=False)
    else:
        # Generate edges for multiple energy regions
        binEdges = []
        for idx in range(4):
            edges = ber.getBinEdgesUniform(NPixels=256, edgeMin=ENERGY_START + idx*ENERGY_RANGE, edgeMax=ENERGY_START + (idx + 1)*ENERGY_RANGE, edgeOvfw=430)
            binEdges.append( edges )
    binEdgesDict['Slot%d' % slot] = binEdges
hickle.dump(binEdgesDict, BIN_EDGES_RANDOM_FN)

