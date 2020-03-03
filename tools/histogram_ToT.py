#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import tqdm
import argparse

def main():
    indir, outDir, p1, p2, p3, bins_min, bins_max, bins, multi = parse_args()
    if not outDir:
        outDir = inDir
    bins = np.linspace(bins_min, bins_max, bins + 1)

    # Get energy calibration parameters
    paramsDict = {}
    params = [p1, p2, p3]
    for p_idx, p in enumerate(params):
        if not p:
            paramsDict['Slot%d' % (p_idx + 1)] = None
            continue
        with open(p, 'r') as f:
            paramsDict['Slot%d' % (p_idx + 1)] = json.load(f)

    # Call histogram routine
    if multi:
        histogramDataMulti(indir, outDir, bins, paramsDict)
    else:
        histogramData(indir, outDir, bins, paramsDict)
        # histGetMax(indir, 100)
    # plotData(indir[:-1] + '.json')

def plotDataMulti(inFile):
    with open(inFile, 'r') as f:
        d = json.load(f)

    for board in d.keys():
        for slot in d[board].keys():
            for pixel in range(256):
                plt.step(BINS[:-1], d[board][slot][pixel])
            plt.show()

# Divide dataset into chunks and get maximum of spectrum for each chunk. 
# Afterwards, mean and standard deviation over all chunks is stored for 
# each pixel separately.
def histGetMax(indir, chunk_size, params):
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
                    dataDict[slot] = {pixel: [] for pixel in np.arange(256)}
            
                x_slot = np.asarray( d[slot_key] ).T
                for pixel in range(256):
                    x_pixel = x_slot[pixel]

                    chunk_max = []
                    for chunk in np.array_split(x_pixel, len(x_pixel) // chunk_size):
                        h, b = np.histogram(chunk, bins=np.arange(1, 401))
                        chunk_max.append( b[np.argmax(h)] )
                    dataDict[slot][pixel] += chunk_max

    maxDict = {key: [[]] * 256 for key in dataDict.keys()}
    for slot in dataDict.keys():
        for pixel in range(256):
            x = dataDict[slot][pixel]
            maxDict[int(slot)][int(pixel)] = (np.nanmean(x), np.nanstd(x))

    outFn = indir[:-1] + '_max.json'
    with open(outFn, 'w') as f:
        json.dump(maxDict, f, cls=NumpyEncoder)

# Read in ToT measurement and yield binned file
def histogramData(indir, outdir, bins, params):
    dataDict = {}
    dataDict['bins'] = bins
    for fn in tqdm.tqdm(os.listdir(indir)):
        if not 'temp' in fn:
            with open(indir + fn, 'r') as f:
                d = json.load( f )

            for slot_key in d.keys():
                slot = int(''.join(filter(str.isdigit, str(slot_key))))
                if not slot in dataDict.keys():
                    dataDict['Slot%d' % slot] = np.zeros((256, len(bins) - 1))

                x_slot = np.asarray( d[slot_key] ).T
                for pixel in range(256):
                    x_pixel = x_slot[pixel]
                    x_pixel = x_pixel[x_pixel > 0]
                    p = params['Slot%d' % slot]
                    if p is not None:
                        x_pixel = ToTtoEnergySimple(x_pixel, p[str(pixel)]['a'], p[str(pixel)]['b'], 
                                                    p[str(pixel)]['c'], p[str(pixel)]['t'])
                    h, b = np.histogram(x_pixel, bins=bins)
                    dataDict['Slot%d' % slot][pixel] += h

    outFn = outdir + '/' + [s for s in indir.split('/') if s][-1] + '.json'
    with open(outFn, 'w') as f:
        json.dump(dataDict, f, cls=NumpyEncoder)

# Histogram data which was measured with multiple boards at once. The
# function is similar to histogramData but some details are different
# due to the different structure of data.
def histogramDataMulti(indir, outdir, bins):
    dataDict = {}
    dataDict['bins'] = bins
    for fn in os.listdir(indir):
        if 'data' in fn:
            print(fn)
            with open(indir + fn, 'r') as f:
                d = json.load( f )
                print( d.keys() )

            for board in d.keys():
                if not board in dataDict.keys():
                    dataDict[int(board)] = {}

                print( np.asarray(d[board]).shape )
                for slot in range(np.asarray(d[board]).shape[1]):
                    if not slot in dataDict[int(board)].keys():
                        dataDict[int(board)]['Slot%d' % slot] = np.zeros((256, len(bins) - 1))

                    x_slot = np.asarray( d[board] )[:,slot]
                    for pixel in range(256):
                        x_pixel = x_slot[:,pixel]
                        x_pixel = x_pixel[x_pixel > 0]
                        p = params['Slot%d' % slot]
                        if p is not None:
                            x_pixel = ToTtoEnergySimple(x_pixel, p[str(pixel)]['a'], p[str(pixel)]['b'], 
                                                        p[str(pixel)]['c'], p[str(pixel)]['t'])
                        h, b = np.histogram(x_pixel, bins=bins)
                        dataDict[int(board)]['Slot%d' % slot][pixel] += h

    for board in d.keys():
        outFn = outdir + '/' + [s for s in indir.split('/') if s][-1] + '_board%d.json' % board
        with open(outFn, 'w') as f:
            json.dump(dataDict[int(board)], f, cls=NumpyEncoder)

# === ETC ===
def ToTtoEnergySimple(x, a, b, c, t, h=1, k=0):
    return h * (b + 1./(4 * a) * (2*x + np.pi*c + np.sqrt(16 * a * c * t + (2 * x + np.pi * c)**2))) + k

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# === ARGPARSE ===
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', '--directory', action='store', required=True, type=str)
    parser.add_argument('-out', '--outdir', action='store', required=False, type=str)
    parser.add_argument('-p1', '--params1', action='store', required=False, type=str, default='')
    parser.add_argument('-p2', '--params2', action='store', required=False, type=str, default='')
    parser.add_argument('-p3', '--params3', action='store', required=False, type=str, default='')
    parser.add_argument('-bmin', '--bins_min', action='store', required=False, type=int, default=10)
    parser.add_argument('-bmax', '--bins_max', action='store', required=False, type=int, default=63)
    parser.add_argument('-b', '--bins', action='store', required=False, type=int, default=300)
    parser.add_argument('-m', '--multi', action='store_true', required=False)
    args = parser.parse_args()
    return args.directory, args.outdir, args.params1, args.params2, args.params3, args.bins_min, args.bins_max, args.bins, args.multi

if __name__ == '__main__':
    main()

