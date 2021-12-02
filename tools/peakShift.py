#!/usr/bin/env python
'''
Get the relation between energy and THL by shifting THL and observing the resulting shift in energy.
This is useful to later correct the movement of the baseline under conditions of high flux.
Here, measurements with an Am-source were performed. The maximum value in the spectrum is found and
a Gaussian is fitted around the region of the maximum. Afterwards, the linear relation of THL and 
energy shift can be estimated by fit.
'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import os
import json
import argparse
import re
import dpx_control.support as dfps
sup = dfps.Support()

plotFn = 'peakShift_22_5_109'
THLRange = [-100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

def main():
    directory, PLOT, shift_index = parse_args()
    
    # Get THL range from files
    THLRange = []
    files = []
    # Use only .json files
    for fn in [fn for fn in os.listdir(directory) if fn.endswith('.json')]:
        # Get numbers from string
        fn_nums = [int(num) for num in re.findall(r'[+-]?\d+', fn)]
        if not len(fn_nums):
            continue
        THL_shift = fn_nums[shift_index]
        THLRange.append( THL_shift )

        # Append file
        files.append( fn )

    # Convert to arrays
    THLRange, files = np.asarray(THLRange), np.asarray(files)

    # Sort by THLRange
    sort_idx = np.argsort( THLRange )
    THLRange, files = THLRange[sort_idx], files[sort_idx]
    print(THLRange)

    energy = np.linspace(15, 100, 200)
    muDict = {'Slot%d' % slot: [] for slot in range(1, 3 + 1)}
    muErrDict = {'Slot%d' % slot: [] for slot in range(1, 3 + 1)}

    if PLOT:
        fig, ax = plt.subplots(1, 3, figsize=(15, 15), sharey=True)
        axCBar = fig.add_axes([0.85, 0.1, 0.05, 0.8])
        plt.subplots_adjust(left=0.1, right=0.83, wspace=0.1)
    for THL_idx, THL in enumerate(THLRange):
        x = json.load(open(directory + '/' + files[THL_idx], 'r'))
        bins = np.asarray(x['bins'])
        for slot in [1]: # range(1, 3 + 1):
            # Sum over pixels
            hist = np.sum(x['Slot%d' % slot], axis=0)

            # Get maximum
            max_idx = np.argmax(np.where(bins[:-1] > 30, hist, 0))
            p0 = (hist[max_idx], bins[:-1][max_idx], 1.2)
            try:
                popt, pcov = scipy.optimize.curve_fit(normal, bins[:-1], hist, p0=p0)
                perr = np.sqrt(np.diag(pcov))
            except:
                popt = np.full(len(p0), np.nan)
                perr = popt # np.zeros(len(p0))
            if PLOT:
                norm = float(np.max( hist ))
                color = sup.getColor('viridis', len(THLRange), THL_idx)
                ax[slot-1].step(bins[:-1], 0.2 * THL_idx + hist / norm, where='post', color=color, zorder=len(THLRange)-THL_idx)
                ax[slot-1].plot(energy, 0.2 * THL_idx + normal(energy, *popt) / norm, color=color, ls='--', zorder=len(THLRange)-THL_idx)
            muDict['Slot%d' % slot].append( popt[1] )
            muErrDict['Slot%d' % slot].append( perr[1] )
        continue

    if PLOT:
        sup.getColorBar(axCBar, THLRange[0], THLRange[-1], N=len(THLRange), label='THL shift', rotation=0)
        for a in ax:
            a.grid()
            a.set_xlabel('Deposited energy (keV)')
            a.set_ylim(-0.05, len(THLRange) * 0.2 + 1.05)
        ax[0].get_yaxis().set_ticks([])
        plt.savefig(plotFn + '_spectra.svg')
        plt.show()

    for slot in [1]: # range(1, 3 + 1):
        diff = abs( np.diff(muDict['Slot%d' % slot] - np.max(muDict['Slot%d' % slot])) )
        try:
            xlim = np.argwhere(diff > 10).flatten()[0]
        except:
            xlim = -1

        x = np.asarray( muDict['Slot%d' % slot][:xlim] - np.min(muDict['Slot%d' % slot][np.argwhere(np.asarray(THLRange) == 0).flatten()[0]]) )
        y = np.asarray( THLRange[:xlim] ) # - THLRange[0]
        xerr = np.asarray( muErrDict['Slot%d' % slot][:xlim] )
        xerr[np.isnan(xerr)] = 0
        xerr[np.isinf(xerr)] = 0
        print(x.shape, y.shape)
        print(xerr.shape)

        xFit = x[~np.isnan(x)]
        yFit = y[~np.isnan(x)]
        xerrFit = xerr[~np.isnan(x)]
        if len(xFit):
            popt, pcov = scipy.optimize.curve_fit(linear, xFit, yFit, sigma=xerrFit)
            perr = np.sqrt(np.diag(pcov))
            print('Slot%d:' % slot, popt, perr)

            plt.errorbar(x, y, xerr=xerr, label='Slot%d' % slot, marker='x', ls='', color='C%d' % (slot - 1))
            plt.plot(x, linear(x, *popt), color='C%d' % (slot - 1))
        else:
            print('Fit for slot %d failed!' % slot)
    plt.legend()
    plt.grid()
    plt.xlabel('Energy shift (keV)')
    plt.ylabel('THL shift (DAC)')
    plt.savefig(plotFn + '_relation.svg')
    plt.show()

# === ETC ===
def linear(x, m, t):
    return m * x + t

def normal(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

# === ARGPARSE ===
def parse_args():
    parser = argparse.ArgumentParser(description='Analysis of measurements performed via measure_THL_energy_relation. \
            Note that the ToT measurements have to be binned and converted to energy via histogram_ToT before they can \
            be used by this program.')
    parser.add_argument('-dir', '--directory', action='store', required=True, type=str)
    parser.add_argument('-p', '--plot', action='store_true', required=False)
    parser.add_argument('-si', '--shift_index', action='store', required=False, type=int, default=0)
    args = parser.parse_args()
    return args.directory, args.plot, args.shift_index

if __name__ == '__main__':
    main()

