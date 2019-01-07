#!/usr/bin/env python
import numpy as np
import cPickle
import matplotlib.pyplot as plt
import seaborn as sns

INFILE = 'THL_1394/ToTMeasurement_Sn.p'
OUTFILE = '../../energy_calibration/spectraBlackCalib/Sn_total.p'
# INFILE = 'ToTMeasurement_41.p'
PARAMSFILE = 'ToTtoEnergy.p'
SLOT = 1

def main():
    paramsDict = cPickle.load( open(PARAMSFILE, 'rb') )
    data = np.asarray( cPickle.load( open(INFILE, 'rb') )['Slot%d' % slot] ).T

    totalData = ToTtoEnergy(data, paramsDict, slot=SLOT)
    if OUTFILE:
        cPickle.dump(totalData, open(OUTFILE, 'wb'))

def getToTtoEnergy(data, params, slot=1, use_hist=False, plot=True, save=None):
    binsTotal = np.arange(4095)
    totalData = []

    # plt.hist(data.flatten(), bins=300)
    # plt.show()

    for pixel in params.keys():
        p = params[pixel]
        a, b, c, t, h, k = p['a'], p['b'], p['c'], p['t'], p['h'], p['k']
        # print h, k

        if not use_hist:
            pixelData = data[pixel]
            pixelData = pixelData[pixelData > 0]
            # pixelData = pixelData[pixelData < 250]

            hist, bins = np.histogram(pixelData, bins=binsTotal) # int(max(pixelData) - min(pixelData)))

            # Get rid of empty entries
            bins = bins[:-1][hist > 0]
            hist = hist[hist > 0]
        else:
            bins, hist = data['bins'][pixel], data['hist'][pixel]
            if bins is None:
                continue
            
            # Remove zeros
            if bins[0] == 0:
                bins = bins[1:]
                hist = hist[1:]
                
            cond = (hist > 0)
            cond = np.append(cond, True)
            bins = bins[cond]
            hist = hist[hist > 0]
            
            bins, hist = np.asarray(bins, dtype=int), np.asarray(hist, dtype=int)

        # plt.step(bins[:-1], hist, where='post')
        # plt.show()

        # Convert bins to energy
        binsEnergy = ToTtoEnergy(bins, a, b, c, t, h, k)

        # Convert single entries
        if not use_hist:
            totalData += list( ToTtoEnergy(pixelData, a, b, c, t, h, k) )
        else:
            histEnergy = []
            for b in range(len(binsEnergy[:-1])):
                if hist[b]:
                    histEnergy += [binsEnergy[b]] * hist[b]
            totalData += list(histEnergy)

        if plot:
            plt.step(binsEnergy[:-1], hist, where='post')
            plt.title('Pixel #%d' % pixel)
            plt.xlabel('Energy (keV)')
            plt.ylabel('Counts')
            plt.xlim(0, 200)
            if False: # save is not None:
                outFn = ''
                if '.' in save:
                    outFn = save.split('.')[0]
                else:
                    outFn = save
                outFn += '_pixel%d.png' % pixel
                plt.savefig(outFn)
            plt.show()

    totalData = np.asarray( totalData )
    print totalData
    totalData = totalData[np.logical_and(totalData > 0, totalData < 1000)]
    hist, bins = np.histogram(totalData, bins=np.linspace(5, 1000, 1000))
    # Get rid of empty entries
    bins = bins[:-1][hist > 10]
    hist = hist[hist > 10]

    plt.step(bins, hist, where='post')
    plt.xlabel('Energy (keV)')
    plt.ylabel('Counts')
    sns.despine(top=True, right=True, offset=0, trim=False)
    
    if save is not None:
        plt.savefig(save)
    # plt.show()
    return totalData
        
def ToTtoEnergy(x, a, b, c, t, h=1, k=0):
    return h * (b + 1./(4 * a) * (2*x + np.pi*c + np.sqrt(16 * a * c * t + (2 * x + np.pi * c)**2))) + k

def EnergyToToT(x, a, b, c, t, h=1, k=0):
    res = np.where(x < b, a*((x - k)/h - b) - c * (np.pi / 2 + t / ((x - k)/h - b)), 0)
    idx = np.argwhere(res <= 0)
    
    if list(idx):
        res[np.arange(idx[-1]+1)] = 0
        # res[res < 0] = 0
    return res

if __name__ == '__main__':
    main()
