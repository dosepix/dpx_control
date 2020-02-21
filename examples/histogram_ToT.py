#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import tqdm

MULTI = False
# indir = 'measureToTMulti_Ikrum100/'
indir = 'calibration_measurements/ToTMeasurement_22_6_109_Ikrum%d/'
# BINS = np.arange(4096)
# DNN calibration
BINS = np.arange(1, 402)

def main():
    if MULTI:
        histogramDataMulti(indir)
    else:
        for Ikrum in [10, 20, 30, 40, 50]:
            histogramData(indir % Ikrum)
    # plotData(indir[:-1] + '.json')

def plotData(inFile):
    with open(inFile, 'r') as f:
        d = json.load(f)

    for board in d.keys():
        for slot in d[board].keys():
            for pixel in range(256):
                plt.step(BINS[:-1], d[board][slot][pixel])
            plt.show()

def histogramData(indir):
    dataDict = {}
    for fn in tqdm.tqdm(os.listdir(indir)):
        if 'ToTMeasurement' in fn and not 'temp' in fn:
            with open(indir + fn, 'r') as f:
                try:
                    d = json.load( f )
                except:
                    continue

            for slot_key in d.keys():
                slot = int(''.join(filter(str.isdigit, str(slot_key))))
                if not slot in dataDict.keys():
                    dataDict[slot] = np.zeros((256, len(BINS) - 1))

                x_slot = np.asarray( d[slot_key] ).T
                for pixel in range(256):
                    x_pixel = x_slot[pixel]
                    x_pixel = x_pixel[x_pixel > 0]
                    h, b = np.histogram(x_pixel, bins=BINS)
                    dataDict[slot][pixel] += h

    outFn = indir[:-1] + '.json'
    with open(outFn, 'w') as f:
        json.dump(dataDict, f, cls=NumpyEncoder)

def histogramDataMulti(indir):
    dataDict = {}
    for fn in os.listdir(indir):
        if 'data' in fn:
            print(fn)
            with open(indir + fn, 'r') as f:
                try:
                    d = json.load( f )
                except:
                    continue
                print( d.keys() )

            for board in d.keys():
                if not board in dataDict.keys():
                    dataDict[int(board)] = {}

                print( np.asarray(d[board]).shape )
                for slot in range(np.asarray(d[board]).shape[1]):
                    if not slot in dataDict[int(board)].keys():
                        dataDict[int(board)][slot] = np.zeros((256, len(BINS) - 1))

                    x_slot = np.asarray( d[board] )[:,slot]
                    for pixel in range(256):
                        x_pixel = x_slot[:,pixel]
                        x_pixel = x_pixel[x_pixel > 0]
                        h, b = np.histogram(x_pixel, bins=BINS)
                        dataDict[int(board)][slot][pixel] += h

    for board in d.keys():
        outFn = indir[:-1] + 'board%d.json' % int( board )
        with open(outFn, 'w') as f:
            json.dump(dataDict[int(board)], f, cls=NumpyEncoder)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

if __name__ == '__main__':
    main()

