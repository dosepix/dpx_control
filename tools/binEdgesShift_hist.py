#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import tqdm
import argparse
import DPXhistSupport as dhs

def main():
    dose_fn, time_fn, p1, p2, p3, b1, b2, b3, rb, bw, multi, split, out = parse_args()

    # === Loading ===
    # Dose
    doseDict = json.load(open(dose_fn, 'r'))

    # Time
    timeDict = json.load(open(time_fn, 'r'))

    histogram_data(doseDict, timeDict, (p1, p2, p3), (b1, b2, b3), rb, bw, multi, split, out)

def histogram_data(doseDict, timeDict, p, b, rb, bw, multi=False, split=1, out='', plot=''):
    p1, p2, p3 = p
    b1, b2, b3 = b

    # Used slots
    SLOT = sorted([int(key[-1]) for key in doseDict.keys()])

    # Bin edges
    binEdgesDict = {}
    for idx, binEdgesFile in enumerate([b1, b2, b3]):
        if not binEdgesFile:
            assert idx + 1 not in SLOT, 'Need bin edges for slot %d!' % (idx + 1)
            binEdgesDict['Slot%d' % (idx + 1)] = None
        else:
            binEdgesRandom = np.asarray( json.load(open(binEdgesFile, 'r')) )

            # if len(np.asarray(binEdgesRandom).shape) > 2:
            if not multi:
                binEdgesRandom = binEdgesRandom[0]
            binEdgesDict['Slot%d' % (idx + 1)] = binEdgesRandom

    if multi:
        print(binEdgesDict['Slot1'].shape)
        doseDict, binEdgesDict = dhs.loadMultiRegionSingle(doseDict, timeDict, binEdgesDict, split=split)

    # Filter noisy pixels
    for slot in doseDict.keys():
        dd = doseDict[slot].reshape((256, -1))
        doseSum = np.sum(dd, axis=0)
        doseMedian, doseStd = np.median( doseSum ), np.std( doseSum )
        noiseFilt = np.argwhere(abs(dd - doseMedian) > 3 * doseStd)
        dd[noiseFilt] = np.zeros(dd.shape[-1])
        doseDict[slot] = dd.reshape(doseDict[slot].shape)

    # Get energy calibration parameters
    paramsDict = {}
    params = [p1, p2, p3]
    for p_idx, p in enumerate(params):
        if not p:
            assert p_idx + 1 not in SLOT, 'Need calibration parameter for slot %d!' % (p_idx + 1)
            paramsDict['Slot%d' % (p_idx + 1)] = None
            continue
        with open(p, 'r') as f:
            paramsDict['Slot%d' % (p_idx + 1)] = json.load(f)

    # Plot data distribution
    if plot is not None:
        fig, ax = plt.subplots(len(SLOT), 1, figsize=(16, 5), sharex=True)
        largePixels = np.asarray([pixel for pixel in np.arange(256) if pixel % 16 not in [0, 1, 14, 15]])
        if type(ax) != np.ndarray:
            ax = [ax]

        for idx, slot in enumerate( SLOT ):
            ax[idx].imshow( np.reshape(np.sum(doseDict['Slot%d' % slot], axis=0), (256, -1))[largePixels].T, aspect='auto' )
        ax[-1].set_xlabel('Pixel')
        if len(ax) > 1:
            ax[1].set_ylabel('Energy bin index')
        else:
            ax[0].set_ylabel('Energy bin index')
        if plot:
            plt.savefig(plot)
            plt.clf()
        else:
            plt.show()

    # Rebinning
    if out:
        save_data = True
    else:
        save_data = False

    if plot is not None:
        if plot:
            plot_split = plot.split('.')
            plot_fn = plot_split[0] + '_spectrum.' + plot_split[-1]
        else:
            plot_fn = None
    else:
        plot_fn = None
    slotList, save_data_dict = dhs.rebinSpectrum(SLOT, doseDict, binEdgesDict, bw, rb, save_data=save_data, save_data_fn=out, plot_fn=plot_fn)
    if plot is not None:
        plt.show()

# === ETC ===
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# === ARGPARSE ===
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-df', '--dose_file', action='store', required=True, type=str)
    parser.add_argument('-tf', '--time_file', action='store', required=True, type=str)
    parser.add_argument('-p1', '--params1', action='store', required=False, type=str, default='')
    parser.add_argument('-p2', '--params2', action='store', required=False, type=str, default='')
    parser.add_argument('-p3', '--params3', action='store', required=False, type=str, default='')
    parser.add_argument('-b1', '--binedges1', action='store', required=False, type=str, default='')
    parser.add_argument('-b2', '--binedges2', action='store', required=False, type=str, default='')
    parser.add_argument('-b3', '--binedges3', action='store', required=False, type=str, default='')
    parser.add_argument('-rb', '--rebins', action='store', required=False, type=int, default=500)
    parser.add_argument('-bw', '--binwidth', action='store', required=False, type=float, default=0.2)
    parser.add_argument('-m', '--multi', action='store_true', required=False)
    parser.add_argument('-s', '--split', action='store', required=False, type=int, default=1)
    parser.add_argument('-o', '--out', action='store', required=False, type=str, default='')
    args = parser.parse_args()
    return args.dose_file, args.time_file, args.params1, args.params2, args.params3, \
            args.binedges1, args.binedges2, args.binedges3, args.rebins, args.binwidth, args.multi, args.split, args.out

if __name__ == '__main__':
    main()

