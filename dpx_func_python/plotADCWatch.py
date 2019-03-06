#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import cPickle, hickle

INFILE = '../../dpx_data/ADCWatch_15.hck'
OUTDIR = 'plotADCWatch/'
WINDOW = 200

def main():
	# Load dictionary from file
	if INFILE.split('.')[-1] == 'p':
		d = cPickle.load( open(INFILE, 'rb') )
	else:
		d = hickle.load(INFILE)

	tempData = np.asarray( running_mean(d['Temperature']['data'], WINDOW) )

	# Loop over entries
	# fig, ax = plt.subplots(len(d.keys()), 2, figsize=(13, 5), gridspec_kw = {'width_ratios':[1, 1]}, sharex=False)

	skipList = ['V_casc_krum', 'V_casc_preamp', 'I_preamp', 'Temperature']
	fig, ax = plt.subplots(len(d.keys()) - len(skipList), 2, sharex=False)

	idx = 0
	for key in d.keys():
		if key in skipList:
			continue
		time, data = d[key]['time'], d[key]['data']
		ax[idx][0].plot(time, data)

		timeRun = time[int(0.5*WINDOW) - 1:-int(0.5*WINDOW)]
		dataRun = np.asarray( running_mean(data, WINDOW) )
		ax[idx][0].plot(timeRun, dataRun)
		# ax[1].plot(timeRun, (dataRun - tempData) / tempData)

		length = min(len(dataRun), len(tempData))
		ax[idx][1].plot(tempData[:length]/tempData[0], dataRun[:length]/dataRun[0])

		# Fit correlation
		popt, pcov = scipy.optimize.curve_fit(linear, tempData[:length]/tempData[0], dataRun[:length]/dataRun[0])

		m, t = popt
		print 'Fit results:'
		print '============'
		print 'm =', m
		print 't =', t
		print

		minX, maxX = min(tempData[:length]/tempData[0]), max(tempData[:length]/tempData[0])
		ax[idx][1].plot((minX, maxX), linear(np.asarray([minX, maxX]), *popt))

		ax[idx][0].set_ylabel(key)
		# ax[idx][1].set_ylabel(key + ' (normalized)')
		idx += 1

	ax[0][0].set_title('Parameter measurement')
	ax[0][1].set_title('Relative temperature correlation')
	ax[idx-1][0].set_xlabel('Time (s)')
	ax[idx-1][1].set_xlabel('Temperature (normalized)')

	# plt.tight_layout()
	plt.savefig(OUTDIR + '%s.pdf' % key)
	plt.show()

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def linear(x, m, t):
	return m*x + t

if __name__ == '__main__':
	main()
