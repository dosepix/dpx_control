#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import cPickle

INFILE = 'ADCWatch_5.p'
OUTDIR = 'plotADCWatch/'
WINDOW = 200

def main():
	# Load dictionary from file
	d = cPickle.load( open(INFILE, 'rb') )

	tempData = np.asarray( running_mean(d['Temperature']['data'], WINDOW) )

	# Loop over entries
	for key in d.keys():
		time, data = d[key]['time'], d[key]['data']

		fig, ax = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw = {'width_ratios':[1, 1]}, sharex=False)

		ax[0].plot(time, data)

		timeRun = time[int(0.5*WINDOW) - 1:-int(0.5*WINDOW)]
		dataRun = np.asarray( running_mean(data, WINDOW) )
		ax[0].plot(timeRun, dataRun)
		ax[0].set_xlabel('Time (s)')
		ax[0].set_ylabel(key + ' (ADC)')
		# ax[1].plot(timeRun, (dataRun - tempData) / tempData)

		length = min(len(dataRun), len(tempData))
		ax[1].plot(tempData[:length]/tempData[0], dataRun[:length]/dataRun[0])

		# Fit correlation
		popt, pcov = scipy.optimize.curve_fit(linear, tempData[:length]/tempData[0], dataRun[:length]/dataRun[0])

		m, t = popt
		print 'Fit results:'
		print '============'
		print 'm =', m
		print 't =', t
		print

		minX, maxX = min(tempData[:length]/tempData[0]), max(tempData[:length]/tempData[0])
		ax[1].plot((minX, maxX), linear(np.asarray([minX, maxX]), *popt))

		ax[1].set_xlabel('Temperature (normalized)')
		ax[1].set_ylabel(key + ' (normalized)')

		plt.tight_layout()
		plt.savefig(OUTDIR + '%s.pdf' % key)
		plt.show()

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def linear(x, m, t):
	return m*x + t

if __name__ == '__main__':
	main()
