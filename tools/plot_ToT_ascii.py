#!/usr/bin/env python
MATPLOTLIB = True
try:
    import matplotlib.pyplot as plt
except:
    MATPLOTLIB = False
import numpy as np
import json
import asciichartpy
import argparse

def main():
    fn, p1, p2, p3, bmin, bmax, bN = parse_args()
    with open(fn, 'r') as f:
            data = json.load( f )

    paramsDict = {}
    params = [p1, p2, p3]
    for p_idx, p in enumerate(params):
        if not p:
            paramsDict['Slot%d' % (p_idx + 1)] = None
            continue
        with open(p, 'r') as f:
            paramsDict['Slot%d' % (p_idx + 1)] = json.load(f)

    bins = np.linspace(bmin, bmax, bN)
    plotData(data, bins, paramsDict)
    # plotDataPixels(data, bins, paramsDict)

# Plot distribution for every pixel. Only works with matplotlib.
def plotDataPixels(data, bins, params):
    large_pixels = np.arange(256)[[pixel % 16 not in [0, 1, 14, 15] for pixel in range(256)]]
    for slot in range(1, 3 + 1):
        d = np.asarray(data['Slot%d' % slot])
        print( d.shape )

        for pixel in large_pixels:
            p = params['Slot%d' % slot]
            if p is None:
                d_ = d[:,pixel]
                d_ = d_[d_ > 0]
            else:
                d_ = np.asarray(ToTtoEnergySimple(d.T[pixel], p[str(pixel)]['a'], p[str(pixel)]['b'], p[str(pixel)]['c'], p[str(pixel)]['t']))

            h, b = np.histogram(d_, bins=bins)
            plt.step(b[:-1], h, where='post')
        plt.show()

# Plot distribution over all pixels. Can specifiy energy calibration
# parameters to directly convert ToT into energy values.
def plotData(data, bins, params):
    large_pixels = [pixel % 16 not in [0, 1, 14, 15] for pixel in range(256)]
    print(params)
    for slot in range(1, 3 + 1):
        d = np.asarray(data['Slot%d' % slot])

        p = params['Slot%d' % slot]
        if p is None:
            d = d[:,large_pixels].flatten()
            d = d[d > 0]
        else:
            d = np.asarray([ToTtoEnergySimple(d.T[pixel], p[str(pixel)]['a'], p[str(pixel)]['b'], p[str(pixel)]['c'], p[str(pixel)]['t']) for pixel in range(256)])
            d = d[large_pixels].flatten()
        h, b = np.histogram(d, bins=bins)

        # Ascii plot
        res = asciichartpy.plot( h / np.max(h) * 30 )
        print(res)

        # Matplotlib
        if MATPLOTLIB:
            plt.step(b[:-1], h, where='post', color='C%d' % (slot - 1), label='Slot%d' % slot)
    if MATPLOTLIB:
        plt.legend()
        plt.xlabel('Energy (keV)')
        plt.ylabel('Counts')
        plt.grid()
        plt.show()

# === ETC ===
def ToTtoEnergySimple(x, a, b, c, t, h=1, k=0):
    return h * (b + 1./(4 * a) * (2*x + np.pi*c + np.sqrt(16 * a * c * t + (2 * x + np.pi * c)**2))) + k

# === ARGPARSE ===
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-fn', '--file', action='store', required=True, type=str)
    parser.add_argument('-p1', '--params1', action='store', required=False, type=str, default='')
    parser.add_argument('-p2', '--params2', action='store', required=False, type=str, default='')
    parser.add_argument('-p3', '--params3', action='store', required=False, type=str, default='')
    parser.add_argument('-bmin', '--bins_min', action='store', required=False, type=int, default=10)
    parser.add_argument('-bmax', '--bins_max', action='store', required=False, type=int, default=63)
    parser.add_argument('-b', '--bins', action='store', required=False, type=int, default=300)
    args = parser.parse_args()
    return args.file, args.params1, args.params2, args.params3, args.bins_min, args.bins_max, args.bins

if __name__ == '__main__':
    main()

