#!/usr/bin/env python
import numpy as np
import cPickle
import matplotlib.pyplot as plt

def main():
	folderList = ['ToTMeasurement_70', 'ToTMeasurement_55', 'LEDBlue_2', 'LEDBlue_RedFoil', 'LEDBlue_wKapton']

	mFig, mAx = plt.subplots(figsize=(14, 3))
	tFig, tAx = plt.subplots(figsize=(14, 3))
	devFig, devAx = plt.subplots(figsize=(14, 3))

	for folder in folderList:
		d = cPickle.load(open(folder + '/ToTCorrectionParams.p', 'rb'))

		mList, tList = [], []
		devList = []
		for key in d.keys():
			mList.append( d[key]['m'] )
			tList.append( d[key]['t'] )

			m, t = d[key]['m'], d[key]['t']

			devList.append( linear(4095, m, t) - linear(0, m, t) )

		m = np.asarray( mList )  / np.mean( mList )
		t = np.asarray( tList ) / np.mean( tList )

		mAx.plot(m, label=folder)
		tAx.plot(t, label=folder)
		devAx.plot(devList, label=folder)

	mAx.legend(loc='best')
	mAx.set_xlabel('Pixel index')
	mAx.set_title('')
	tAx.legend(loc='best')
	tAx.set_xlabel('Pixel index')
	devAx.legend(loc='best')
	devAx.set_xlabel('Pixel index')

	mFig.tight_layout()
	tFig.tight_layout()
	devFig.tight_layout()

	mFig.show()
	tFig.show()
	devFig.show()
	
	raw_input('')

def linear(x, m, t):
	return m*x + t

if __name__ == '__main__':
	main()

