#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import cPickle
import scipy.signal
from itertools import groupby
from operator import itemgetter

def main():
	fn = 'energySpectrumTHL_13.p'
	d = cPickle.load(open(fn, 'rb'))

	fig, ax = plt.subplots(2, 1)

	# Loop over pixels
	for key in d.keys():
		thl = d[key]['THL']
		data = d[key]['data']

		# Take mean of double THL values
		dataCorr = [(k, np.mean(list(list(zip(*g))[1]))) for k, g in groupby(zip(thl, data), itemgetter(0))]
		
		# Savgol filter the data
		try:
			dataFilt = scipy.signal.savgol_filter(data, 51, 3)
			ax[0].plot(data)
		except:
			dataFilt = data
			
		ax[0].plot(dataFilt)

		# Calculate deviation
		dataDiff = np.diff(dataFilt)
		ax[1].plot(np.arange(len(dataDiff)) + 0.5, dataDiff)

		fig.show()
		raw_input('')
		ax[0].clear()
		ax[1].clear()

if __name__ == '__main__':
	main()

