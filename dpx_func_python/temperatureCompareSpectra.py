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

                histSmallWarm, histSmallCool = np.zeros(4094), np.zeros(4094)
                histLargeWarm, histLargeCool = np.zeros(4094), np.zeros(4094)
		for pixel in range(256):
			dataCoolPixel, dataWarmPixel = np.asarray(dataCool[pixel]), np.asarray(dataWarm[pixel])
			dataCoolPixel = dataCoolPixel[dataCoolPixel > 0]
			dataWarmPixel = dataWarmPixel[dataWarmPixel > 0]

			# Print statistics
			print np.mean(dataCoolPixel), np.mean(dataWarmPixel)

			# Bin the data
			histCool, binsCool = np.histogram(dataCoolPixel, bins=np.arange(0, 4095))
			histWarm, binsWarm = np.histogram(dataWarmPixel, bins=np.arange(0, 4095))
                        if not isBig(pixel + 1):
                                histSmallWarm = histSmallWarm + histWarm
                                histSmallCool = histSmallCool + histCool
                        else:
                                histLargeWarm = histLargeWarm + histWarm
                                histLargeCool = histLargeCool + histCool

                        # Get rid of zeros
                        binsWarm, binsCool = binsWarm[:-1][histWarm > 0], binsCool[:-1][histCool > 0]
                        histWarm, histCool = histWarm[histWarm > 0], histCool[histCool > 0]

                        # Plot
			plt.step(binsCool, histCool, color='cornflowerblue', label='cold', where='post')
			plt.step(binsWarm, histWarm, color='crimson', label='warm', where='post')
			plt.legend()
			plt.show()

                bins = np.arange(4094)

                # Small pixels summary
                binsCool = bins[histSmallCool > 1]
                histSmallCool = histSmallCool[histSmallCool > 1]
                binsWarm = bins[histSmallWarm > 1]
                histSmallWarm = histSmallWarm[histSmallWarm > 1]

                plt.step(binsWarm, histSmallWarm, where='post', color='crimson')
                plt.step(binsCool, histSmallCool, where='post', color='cornflowerblue')
                plt.title('Small pixels')
                plt.show()

                # Large pixels summary
                binsCool = bins[histLargeCool > 15]
                histLargeCool = histLargeCool[histLargeCool > 15]
                binsWarm = bins[histLargeWarm > 15]
                histLargeWarm = histLargeWarm[histLargeWarm > 15]

                plt.step(binsWarm, histLargeWarm, where='post', color='crimson')
                plt.step(binsCool, histLargeCool, where='post', color='cornflowerblue')
                plt.title('Large pixels')
                plt.show()

def isBig(pixel):
	smallIdx = [1, 2]
	bigIdx = []

	while max(smallIdx) < 256:
		maxSmall = max(smallIdx)
		bigIdx += range(maxSmall+1, maxSmall+13)
		maxBig = max(bigIdx)
		smallIdx += range(maxBig+1, maxBig+5)

	if pixel in bigIdx:
		return True
	else:
		return False

if __name__ == '__main__':
	main()

