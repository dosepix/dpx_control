#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import json
import asciichartpy

MATPLOTLIB = True

def main():
    # with open('calibration_measurements/ToTMeasurement_109_6_22_Ikrum25/ToTMeasurement.json', 'r') as f:
    # with open('ToTMeasurement_22_6_109_Ikrum25/ToTMeasurement_2.json', 'r') as f:

    idx = 22
    for i in range(idx, idx + 4):
        with open('ToTMeasurement_%d/ToTMeasurement_%d.json' % (i, i), 'r') as f:
            data = json.load( f )
        bins = np.arange(10, 800)
        # bins = np.linspace(10, 63, 130)
        plotData(data, bins, params=None)
    plt.show()
    # plotDataPixels(data, bins, params=None)
    return

    paramsDict = {}
    titles = [22, 6, 109]
    for idx, title in enumerate(titles):
        with open('calibration_parameters/params_%d_Ikrum25.json' % title, 'r') as f:
            paramsDict['Slot%d' % (idx + 1)] = json.load(f)

    bins = np.linspace(10, 63, 130)
    # bins = np.linspace(10, 200, 130)
    # plotData(data, bins, paramsDict)
    plotDataPixels(data, bins, paramsDict)

def plotDataPixels(data, bins, params=None):
    large_pixels = np.arange(256)[[pixel % 16 not in [0, 1, 14, 15] for pixel in range(256)]]
    for slot in range(1, 3 + 1):
        d = np.asarray(data['Slot%d' % slot])
        print( d.shape )

        for pixel in large_pixels:
            if params is not None:
                p = params['Slot%d' % slot]
                d_ = np.asarray(ToTtoEnergySimple(d.T[pixel], p[str(pixel)]['a'], p[str(pixel)]['b'], p[str(pixel)]['c'], p[str(pixel)]['t']))
            else:
                d_ = d[:,pixel]
                d_ = d_[d_ > 0]

            h, b = np.histogram(d_, bins=bins)
            plt.step(b[:-1], h, where='post')
        plt.show()

def plotData(data, bins, params=None):
    large_pixels = [pixel % 16 not in [0, 1, 14, 15] for pixel in range(256)]
    for slot in range(1, 3 + 1):
        d = np.asarray(data['Slot%d' % slot])

        if params is not None:
            p = params['Slot%d' % slot]
            d = np.asarray([ToTtoEnergySimple(d.T[pixel], p[str(pixel)]['a'], p[str(pixel)]['b'], p[str(pixel)]['c'], p[str(pixel)]['t']) for pixel in range(256)])
            d = d[large_pixels].flatten()
        else:
            d = d[:,large_pixels].flatten()
            d = d[d > 0]

        h, b = np.histogram(d, bins=bins)

        # Ascii plot
        res = asciichartpy.plot( h / np.max(h) * 30 )
        print(res)

        # Matplotlib
        if MATPLOTLIB:
            plt.step(b[:-1], h, where='post', color='C%d' % (slot - 1))
    # if MATPLOTLIB:
    #    plt.show()

def ToTtoEnergySimple(x, a, b, c, t, h=1, k=0):
    return h * (b + 1./(4 * a) * (2*x + np.pi*c + np.sqrt(16 * a * c * t + (2 * x + np.pi * c)**2))) + k

if __name__ == '__main__':
    main()

