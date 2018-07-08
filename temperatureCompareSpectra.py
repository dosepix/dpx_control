#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import cPickle
import scipy.optimize

INDIR = 'temperatureSpectra'
FNCOLD = INDIR + '/ToTMeasurement_Am_cool.p'
FNWARM = INDIR + '/ToTMeasurement_Am_warm.p'

def main():
	dCool, dWarm = cPickle.load(open(FNCOLD, 'rb')), cPickle.load(open(FNWARM, 'rb'))

	for slot in dCool.keys():
		print 'Accesssing %s...' % slot

		# Transpose to access single pixels
		dataCool, dataWarm = np.asarray(dCool[slot]).T, np.asarray(dWarm[slot]).T

		for pixel in range(256):
			dataCoolPixel, dataWarmPixel = np.asarray(dataCool[pixel]), np.asarray(dataWarm[pixel])
			dataCoolPixel = dataCoolPixel[dataCoolPixel > 0]
			dataWarmPixel = dataWarmPixel[dataWarmPixel > 0]

			# Print statistics
			print np.mean(dataCoolPixel), np.mean(dataWarmPixel)

			# Bin the data
			histCool, binsCool = np.histogram(dataCoolPixel, bins=np.arange(min(dataCoolPixel), max(dataCoolPixel), 1))
			histWarm, binsWarm = np.histogram(dataWarmPixel, bins=np.arange(min(dataWarmPixel), max(dataWarmPixel), 1))

			plt.plot(binsCool[:-1], histCool, color='cornflowerblue', label='cold')
			plt.plot(binsWarm[:-1], histWarm, color='crimson', label='warm')
			plt.legend()
			plt.show()

if __name__ == '__main__':
	main()

