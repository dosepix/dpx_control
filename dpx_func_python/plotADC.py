#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import cPickle

INFILE = 'ADCCalib.p'

def main():
	# Load dictionary from file
	d = cPickle.load( open(INFILE, 'rb') )

	x, y = d['ADC'], d['Volt']
	plt.plot(x, y)

	plt.xlabel('ADC (%)')
	plt.ylabel(r'$V_{\mathrm{casc, krum}}$')

	plt.show()

if __name__ == '__main__':
	main()

