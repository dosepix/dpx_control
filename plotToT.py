#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import cPickle
import argparse

def main():
	ap = argparse.ArgumentParser(description='Process some integers.')
	ap.add_argument('-fn', '--filename', type=str, help='File to plot', required=True)
	ap.add_argument('-sl', '--slot', type=int, help='Slot to plot', required=False)

	args = ap.parse_args()
	fn = args.filename
	if args.slot:
		slot = args.slot
	else:
		slot = 1

	# Load yaml file
	print fn
	x = cPickle.load( open(fn, 'r') )

	# plotAnim(x['Slot%d' % slot])
	# return

	# Get slot from data
	x = np.asarray(x['Slot%d' % slot]).flatten()

	# Remove zeros
	x = x[x > 0]
	print len(x)

	plt.hist(x, bins=np.linspace(0, 2000, 1000))
	plt.xlabel('ToT')
	plt.ylabel('Counts')
	plt.xlim(0, 1200)
	plt.show()

def plotAnim(x, window=100, step=50):
	bins = np.linspace(0, 2000, 1000)
	print len(x), len(x) - window
	for i in range(0, len(x) - window, step):
		print i, i + window
		xFlat = np.asarray(x[i:i + window]).flatten()

		# Remove zeros
		xFlat = xFlat[xFlat > 0]

		plt.clf()
		plt.hist(xFlat, bins=bins)
		plt.pause(0.1)

if __name__ == '__main__':
	main()

