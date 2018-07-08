#!/usr/bin/env python
import numpy as np
import cPickle
import matplotlib.pyplot as plt

INFILE = 'THL_1394/ToTMeasurement_Am.p'
PARAMSFILE = 'ToTtoEnergy.p'
SLOT = 1

def main():
	data = np.asarray( cPickle.load( open(INFILE, 'rb') )['Slot%d' % SLOT] ).T
	params = cPickle.load( open(PARAMSFILE, 'rb') )

        binsTotal = np.arange(4095)
        totalData = []

	for pixel in params.keys():
		p = params[pixel]
		a, b, c, t, h, k = p['a'], p['b'], p['c'], p['t'], p['h'], p['k']
		print h, k

                pixelData = data[pixel]
                pixelData = pixelData[pixelData > 0]

		hist, bins = np.histogram(pixelData, bins=binsTotal) # int(max(pixelData) - min(pixelData)))

                # Get rid of empty entries
                bins = bins[:-1][hist > 0]
                hist = hist[hist > 0]

		# Convert bins to energy
		binsEnergy = ToTtoEnergy(bins, a, b, c, t, h, k)

                # Convert single entries
                totalData += list( ToTtoEnergy(pixelData, a, b, c, t, h, k) )

		# plt.step(binsEnergy, hist, where='post')
		# plt.show()

        hist, bins = np.histogram(totalData, bins=300)
        # Get rid of empty entries
        bins = bins[:-1][hist > 0]
        hist = hist[hist > 0]

        plt.step(bins, hist, where='post')
        plt.show()

def ToTtoEnergy(x, a, b, c, t, h, k):
	return h * (b + 1./(4 * a) * (2*x + np.pi*c + np.sqrt(16 * a * c * t + (2 * x + np.pi * c)**2))) + k

if __name__ == '__main__':
	main()

