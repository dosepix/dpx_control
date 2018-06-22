#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import cPickle
import argparse
from scipy.optimize import fsolve
import scipy.optimize
import peakutils
from scipy import signal
import os

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

    # Load file
    print fn
    x = cPickle.load( open(fn, 'r') )

    data = np.asarray( x['Slot%d' % slot] )
    sumSpectrumAnalysis(data, fn.split('.')[0])
    pixelAnalysis(data, fn.split('.')[0])
    return

    plotCorrected(data)
    print getPixelShift(data)
    return 

def pixelAnalysis(data, saveDir=None):
    if saveDir:
        checkIfDirExists(saveDir)

    bins = np.arange(0, 4095, 1)
    data = np.asarray(data).T

    # Store fit data
    dLinear = {}

    pixelDataCorrected = []
    Npixel = 0
    peakMin, peakMax = [], []

    for pixel in range(256):
        if not isLarge(pixel):
            continue
        pixelData = data[pixel].flatten()
        pixelData = pixelData[pixelData > 0]

        hist, bins = np.histogram(pixelData, bins=bins)

        #plt.plot(bins[:-1], hist)
        histLow = hist[bins[:-1] < 1000]
        binsLow = bins[bins < 1000]
        dataFiltLow = np.asarray( scipy.signal.savgol_filter(histLow, 11, 3) )
        dataFiltLow[dataFiltLow < 1] = 0

        # Get high energy region and fit background
        histHigh = np.asarray( hist[bins[:-1] >= 1000] )
        binsHigh = bins[bins >= 1000][:-1]
        popt, pcov = scipy.optimize.curve_fit(normal, binsHigh, histHigh, p0=(30., 3000., 300., 1.))
        histHigh = histHigh - np.asarray( normal(binsHigh, *popt) )

        # Filter high energy region again and only select 3sigma-region
        # of previous background fit
        Amp, mu, sigma, off = popt
        histHigh = histHigh[abs(binsHigh - mu) <= 3*sigma]
        binsHigh = binsHigh[abs(binsHigh - mu) <= 3*sigma]

        try:
            dataFiltHigh = np.asarray( scipy.signal.savgol_filter(histHigh, 43, 3) )
        except:
            dataFiltHigh = np.asarray(histHigh)

        # Plot of filtering results
        '''
        plt.plot(binsHigh, histHigh, label='raw')
        plt.plot(binsHigh, dataFiltHigh, label='filtered')
        plt.xlabel('ToT')
        plt.ylabel('Counts')
        plt.savefig('filterData.pdf')
        plt.show()
        '''

        # dataFiltHigh[dataFiltHigh < 3] = 0

        # Combine low and high energy filtered data
        print len(binsLow), len(dataFiltLow)
        print len(binsHigh), len(dataFiltHigh)
        binsFilt = np.hstack((binsLow, binsHigh))
        dataFilt = np.hstack((dataFiltLow, dataFiltHigh))

        # Find all peaks
        peakIdx = peakutils.indexes(dataFilt, thres=.0015, min_dist=40)

        # plt.plot(binsFilt, dataFilt)
        # plt.plot(binsFilt[peakIdx], dataFilt[peakIdx], marker='x', ls='')
        # plt.show()

        peakDiff = np.diff(binsFilt[peakIdx])
        meanPeakDiff = np.mean( peakDiff[peakDiff < 75] )
        print meanPeakDiff

        peakCorrected = [0] + list( np.cumsum( [int(peak/meanPeakDiff + 0.5) for peak in peakDiff] ) )
        popt, pcov = scipy.optimize.curve_fit(linear, binsFilt[peakIdx], peakCorrected)
        m, t = popt

        dLinear[pixel] = {'m': m, 't': t}

        # Plot linear relation between peak position and index
        '''
        plt.plot(binsFilt[peakIdx], peakCorrected, marker='x', ls='')
        plt.plot(binsFilt[peakIdx], linear(np.asarray(binsFilt[peakIdx]), *popt))
        plt.show()
        '''

        # Scale axis
        axisCorrected = linear(binsFilt, *popt)
        pixelDataCorrected = np.hstack( (pixelDataCorrected, linear(pixelData, *popt)) )

        # plt.plot(axisCorrected, dataFilt)
        # plt.plot(axisCorrected[peakIdx], dataFilt[peakIdx], marker='x', ls='')
        # plt.plot(np.diff(binsFilt[peakIdx]) / meanPeakDiff)
        # plt.show()

        Npixel += 1
        peakMax.append(linear(4095, *popt))
        peakMin.append(linear(0, *popt))

    # Dump fit data
    if saveDir:
        cPickle.dump(dLinear, open(saveDir + '/ToTCorrectionParams.p', 'wb'))

    # = Plots =
    peakMin = np.mean( peakMin )
    peakMax = np.mean( peakMax )

    print 'Number of processed pixels:', Npixel
    hist, bins = np.histogram(pixelDataCorrected, bins=np.linspace(peakMin, peakMax, 4095))
    histFilt = np.asarray( scipy.signal.savgol_filter(hist, 21, 3) )

    fig, ax = plt.subplots()

    ax.set_yscale("log", nonposy='clip')
    ax.step(bins[:-1], hist, where='post')
    # ax.step(bins[:-1], histFilt, where='post')

    ax.set_xlabel('Peak index')
    ax.set_ylabel('Counts')

    ax.grid(which='both')

    if saveDir:
        fig.savefig(saveDir + '/energySpectrumCorrected_total.pdf')

        # Plot low and high energy regions only
        ax.set_xlim(peakMin, (peakMax - peakMin) * 1000/4095.)
        fig.savefig(saveDir + '/energySpectrumCorrected_low.pdf')
        ax.set_xlim((peakMax - peakMin) * 1000/4095., peakMax)
        fig.savefig(saveDir + '/energySpectrumCorrected_high.pdf')
    else:
        fig.show()
        raw_input()

def sumSpectrumAnalysis(data, saveDir=None):
    if saveDir:
        checkIfDirExists(saveDir)
        labelList = ['Small pixels', 'Large pixels', 'All pixels']
        fnList = ['small', 'large', 'total']

    data = np.asarray(data).T

    dataLarge = np.asarray( [data[i] for i in range(256) if isLarge(i)] )
    dataSmall = np.asarray( [data[i] for i in range(256) if not isLarge(i)] )
    dataTotal = data

    dataList = [dataSmall, dataLarge, dataTotal]

    for idx, data_ in enumerate(dataList):
        data_ = data_.flatten()
        data_ = data_[data_ > 0]

        hist, bins = np.histogram(data_, bins=np.arange(0, 4095, 1))
        # plt.semilogy(bins[:-1], hist)
        # plt.show()
        # hist = getFFT(bins[:-1], hist, bins[1] - bins[0], plot=True)

        # Find all peaks
        peakIdx = peakutils.indexes(hist, thres=.01, min_dist=40)

        # Difference of peaks
        print np.diff( bins[peakIdx] )

        figPeak, axPeak = plt.subplots()
        # axPeak.set_yscale("log", nonposy='clip')

        xPeak = np.arange(1, len(peakIdx) + 1)
        axPeak.plot(xPeak, bins[peakIdx], marker='x', ls='', color='cornflowerblue')
        popt, pcov = scipy.optimize.curve_fit(linear, xPeak, bins[peakIdx])
        print popt, np.sqrt( np.diag(pcov) )
        plt.plot(xPeak, linear(xPeak, *popt), color='cornflowerblue')

        axPeak.set_xlabel('Peak Index')
        axPeak.set_ylabel('ToT')
        axPeak.grid()
        axPeak.set_title(labelList[idx])
        if saveDir:
            figPeak.savefig(saveDir + '/peakDetect_%s.pdf' % fnList[idx])
        else:
            figPeak.show()

        # Logarithmic energy spectrum
        fig, ax = plt.subplots()
        ax.set_yscale("log", nonposy='clip')
        ax.step(bins[:-1], hist, where='post')
        # Peak positions
        ax.plot(bins[peakIdx], hist[peakIdx], marker='x', ls='')

        ax.set_xlabel('ToT')
        ax.set_ylabel('Counts')
        ax.set_title(labelList[idx])

        ax.grid(which='both')

        if saveDir:
            fig.savefig(saveDir + '/energySpectrum_%s.pdf' % fnList[idx])

            # Plot low and high energy regions only
            ax.set_xlim(0, 1000)
            fig.savefig(saveDir + '/energySpectrum_%s_low.pdf' % fnList[idx])
            ax.set_xlim(1000, 4095)
            fig.savefig(saveDir + '/energySpectrum_%s_high.pdf' % fnList[idx])
        else:
            fig.show()
            raw_input('')

    plt.close(figPeak)
    plt.close(fig)

    return

def plotCorrected(data):
    d = cPickle.load(open('testPulseParams.p', 'r'))
    aList, bList, cList, tList = d['a'], d['b'], d['c'], d['t']

    # Transpose to get pixel info
    data = np.asarray( data ).T

    peakCorrection = []
    binsEnergyList = []
    histList = []
    peak = []
    pixelList = []

    for pixel in range(16):
        if not isLarge(pixel):
            continue

        pixelData = data[pixel]
        pixelData = pixelData[pixelData > 0]

        a, b, c, t = aList[pixel], bList[pixel], cList[pixel], tList[pixel]

        # Make histogram
        hist, bins = np.histogram(pixelData, bins=np.arange(0, 1000, 1))

        # Convert to energy
        binsEnergy = []
        for bi in bins:
            binsEnergy.append( ToTtoEnergy([bi], a, b, c, t) )
        binsEnergy = np.asarray( binsEnergy )

        binsEnergyList.append( binsEnergy )
        histList.append( hist )

        # Find all peaks
        peakIdx = peakutils.indexes(hist, thres=.005, min_dist=40)
        peakCorrection.append( binsEnergy[peakIdx][0] )

        if pixel == 6:
            peak = binsEnergy[peakIdx]

        # plt.plot(np.diff( binsEnergy[peakIdx] ))
        # plt.show()

        # plt.plot(binsEnergy[peakIdx], hist[peakIdx], marker='x', ls='')
        # plt.step(binsEnergy[:-1], hist)
        # plt.show()

        pixelList.append( pixel )

    print 'Uncorrected energy spectrum'
    plt.semilogy(binsEnergyList[0][:-1], np.sum(histList, axis=0))
    plt.show()

    peakCorrectionFirst = np.mean( peakCorrection ) - np.asarray( peakCorrection )
    peak = np.asarray( peak ) - peak[0] + np.mean(peakCorrection)
    eventListTotal = []

    for i, binsEnergy in enumerate( binsEnergyList ):
        binsEnergy = np.asarray( binsEnergy[:-1] ) + peakCorrectionFirst[i]

        # Find all peaks
        peakIdx = peakutils.indexes(histList[i], thres=.005, min_dist=40)
        binsDiff = binsEnergy[peakIdx]

        plt.plot(peak[:len(peakIdx)], binsEnergy[peakIdx][:len(peak)] / peak[:len(peakIdx)] ) 
        print binsEnergy[peakIdx][:len(peak)]
        print peak[:len(peakIdx)]/binsEnergy[peakIdx][:len(peak)]
        popt, pcov = scipy.optimize.curve_fit(linear, binsEnergy[peakIdx][:len(peak)].flatten(), (peak[:len(peakIdx)] / binsEnergy[peakIdx][:len(peak)]).flatten() )
        perr = np.sqrt( np.diag(pcov) )

        # plt.plot( binsEnergy[peakIdx], linear(binsEnergy[peakIdx], *popt))

        pixelData = data[pixelList[i]]
        pixelData = pixelData[pixelData > 0]

        a, b, c, t = aList[pixelList[i]], bList[pixelList[i]], cList[pixelList[i]], tList[pixelList[i]]

        eventList = []
        for event in pixelData:
            eventList.append( ToTtoEnergy([event], a, b, c, t) )

        eventList = np.asarray( eventList ) 
        eventList += peakCorrectionFirst[i]
        eventList *= linear(eventList, *popt)

        eventListTotal += list( eventList )
    plt.show()

    # Make histogram
    print 'Corrected energy spectrum'
    hist, bins = np.histogram(eventListTotal, bins=np.arange(0, 4095, 1))
    plt.semilogy(bins[:-1], hist)

    plt.show()

def linear(x, m, t):
    return m*x + t

def getPixelShift(data):
    pixelData = []
    muListTotal = []
    peakListTotal = []

    slopeDict = {'pixel%d' % i: {} for i in range(256)}
    dataCorrected = []

    # energyHigh = (2765, 3500)
    energyHigh = (2000, 3500)

    for pixel in range(16):
        if not isLarge(pixel):
            continue
        pixelData = data[:,pixel].flatten()
        pixelData = pixelData[pixelData > 0]
        pixelData = list( pixelData )
        
        # Get slot from data
        # fig, ax = plt.subplots()
        # ax.set_yscale("log", nonposy='clip')
        hist, bins = np.histogram(pixelData, bins=np.arange(0, 4095, 1))

        # Filter region between low and high energy region
        bins[~np.logical_or(bins <= 500, np.logical_and(bins > energyHigh[0], bins < energyHigh[1]))] = 0
        hist[~np.logical_or(bins[:-1] <= 500, np.logical_and(bins[:-1] > energyHigh[0], bins[:-1] < energyHigh[1]))] = 0

        print bins, hist

        # Get peaks of low energy region
        binsLow_, histLow_ = bins, hist
        binsLow = binsLow_[binsLow_ < 500]
        histLow = histLow_[binsLow_[:-1] < 500]

        # Find low energy peaks
        peakIdxLow = peakutils.indexes(histLow, thres=.01, min_dist=40)

        # Fit low energy peak positions
        #plt.plot(binsLow[peakIdxLow], np.arange(len(peakIdxLow)), marker='x')
        print(np.arange(len(peakIdxLow)))
        popt, pcov = scipy.optimize.curve_fit(linear, binsLow[peakIdxLow], np.arange(len(peakIdxLow)))
        m1, t1 = popt
        print binsLow[peakIdxLow], linear(binsLow[peakIdxLow], *popt)

        # plt.plot(binsLow[peakIdxLow], linear(binsLow[peakIdxLow], *popt))
        plt.plot(binsLow[peakIdxLow], np.arange(len(peakIdxLow)), marker='x', ls='')

        #plt.plot(binsLow[peakIdxLow], linear(binsLow[peakIdxLow], *popt))
        #plt.show()

        # Get peaks of high energy region
        binsHigh_, histHigh_ = bins, hist
        binsHigh = binsHigh_[np.logical_and(binsHigh_ > energyHigh[0], binsHigh_ < energyHigh[1])]
        histHigh = histHigh_[np.logical_and(binsHigh_[:-1] > energyHigh[0], binsHigh_[:-1] < energyHigh[1])]

        # Find high energy peaks
        peakIdxHigh = np.asarray( peakutils.indexes(histHigh, thres=.1, min_dist=40) )
        print peakIdxHigh

        '''
        plt.plot(binsHigh, histHigh)
        plt.plot(binsHigh[peakIdxHigh], histHigh[peakIdxHigh], marker='x')
        plt.show()
        '''

        # Get position of first high energy peak
        peakHighX = binsHigh[peakIdxHigh[0]]
    
        # Insert in low energy fit to get corresponding peak index
        peakIdxCorrHigh = int( linear(peakHighX, *popt) + 0.5 ) + 1
        print peakIdxCorrHigh

        # plt.plot(binsHigh[peakIdxHigh], linear(binsHigh[peakIdxHigh], *popt))
        plt.plot(binsHigh[peakIdxHigh], list(peakIdxCorrHigh + np.arange(len(peakIdxHigh))), marker='x', ls='')
        # plt.show()

        xPeak = list( binsLow[peakIdxLow] ) + list( binsHigh[peakIdxHigh] )

        # Fit corrected peaks
        popt, pcov = scipy.optimize.curve_fit(linear, binsHigh[peakIdxHigh], list(peakIdxCorrHigh + np.arange(len(peakIdxHigh))))
        m2, t2 = popt

        # Shift high peak indices by correction
        peakIdxHigh += peakIdxCorrHigh

        # Combine peak information of low and high energy regions
        peakIdx = list(range(len(peakIdxLow))) + list(peakIdxCorrHigh + np.arange(len(peakIdxHigh))) # np.asarray( list( peakIdxLow ) + list( peakIdxHigh ) )

        plt.plot(xPeak, doubleLinear(np.asarray(xPeak), m1, m2, t1, t2))
    
        # Do FFT of signal
        # getFFT(bins[:-1], hist, bins[1] - bins[0])

        # Find all peaks
        # peakIdx = peakutils.indexes(hist, thres=.001, min_dist=40)
        print peakIdx

                # plt.plot(bins[:-1], hist)
        # plt.plot(bins[peakIdx], hist[peakIdx], marker='x', ls='')
                # plt.plot(xPeak, [0]*len(xPeak), marker='x', ls='')
        # plt.show()
        # plt.clf()

        # xPeak = bins[peakIdx]
        muList = []
        sigmaList = []
        muErrList = []
        sigmaErrList = []
        peakList = []
        peakCnt = 0
        for xP in xPeak:
            # Filter data
            x_ = bins[:-1][abs(bins[:-1] - xP) <= 25]
            y_ = hist[abs(bins[:-1] - xP) <= 25]
 
            p0 = (max(y_), xP, 10., 0.1)
            try:
                popt, pcov = scipy.optimize.curve_fit(normal, x_, y_, p0=p0)
                perr = np.sqrt( np.diag(pcov) )

                A, mu, sigma, off = popt
                if abs(sigma) > 20:
                    continue

                if perr[1] > 200:
                    continue

                muList.append( mu )
                sigmaList.append( np.abs(sigma) )
                muErrList.append( perr[1] )
                sigmaErrList.append( perr[2] )

                # plt.plot(x_, normal(x_, *popt))
                # plt.plot(x_, y_)
                # plt.show()
            except: 
                '''
                plt.plot(x_, normal(x_, *p0))
                plt.plot(x_, y_)
                plt.show()
                '''

                muList.append( np.nan )
                pass

            peakList.append( peakIdx[peakCnt] )
            peakCnt += 1

            muListTotal.append( muList )
            peakListTotal.append( peakList )

            # plt.step(bins[:-1], hist, where='post')
            # plt.show() 

            # sigma vs. Index
            '''
            plt.errorbar(peakList, sigmaList, yerr=sigmaErrList, marker='x', ls='', color='cornflowerblue')
            popt, pcov = scipy.optimize.curve_fit(lambda x, m, t: m*x + t, peakList, sigmaList)
            plt.plot(peakList, popt[0]*np.asarray(peakList) + popt[1], ls='-', color='cornflowerblue')
            plt.xlabel('Index')
            plt.ylabel(r'$\sigma$ (ToT)')
            plt.grid()
            plt.show()
            '''

            # mu vs. Index
            print peakList
            print muList
            print
            
                    # Add a peak at zero
                    # peakList.insert(0, 0)
                    # muList.insert(0, 0)
                    # muErrList.insert(0, 0)

            peakList = np.asarray(peakList)
            muList = np.asarray(muList)
            # Remove nans
            peakList = peakList[~np.isnan(muList)]
            muList = muList[~np.isnan(muList)]

            plt.errorbar(peakList, muList, yerr=muErrList, marker='x', ls='', color=getColor('Blues', 16, pixel))
            try:
                # Linear fit
                # popt, pcov = scipy.optimize.curve_fit(lambda x, m, t: m*x + t, peakList, muList)
                # slopeDict['pixel%d' % pixel]['m'] = popt[0]
                # slopeDict['pixel%d' % pixel]['t'] = popt[1]

                # popt, pcov = scipy.optimize.curve_fit(EnergyToToT, muList, peakList, p0=(5., 0., 1., 1.))
                popt = (m1, m2, t1, t2)
                # popt, pcov = scipy.optimize.curve_fit(doubleLinear, muList, peakList, p0=(m1, m2, t1, t2))

            except:
                popt = (m1, m2, t1, t2)
                continue

            print 'Diff'
            print np.diff(muList)

            # plt.plot(peakList, popt[0]*np.asarray(peakList) + popt[1], ls='-', color=getColor('Blues', 256, pixel))
            muListFit = np.linspace(min(muList), max(muList), 1000)
            plt.plot(doubleLinear(muListFit, *popt), muListFit, ls='-', color=getColor('Blues', 16, pixel))

            plt.xlabel('Index')
            plt.ylabel(r'$\mu$ (ToT)')
            plt.grid()

            # Plot peak corrected spectrum
            # dataCorrected += list( (np.asarray(pixelData) - popt[1]) / popt[0] )
            dataCorrected = doubleLinear(np.asarray(pixelData), *popt)

            # plt.hist(dataCorrected, bins=np.linspace(0, (500-popt[1])/popt[0], 500))
            # plt.show()

            # Scale bins
            # bins = (np.asarray(bins) - popt[1]) / popt[0]
            # plt.plot(bins[:-1], hist)
            # plt.show()

    # getFFT(bins[:-1], hist, bins[1] - bins[0])

    # Mean mu
    max_len = np.max([len(a) for a in muListTotal])
    muListTotal = np.asarray([np.pad(a, (0, max_len - len(a)), 'constant', constant_values=np.nan) for a in muListTotal])
    print muListTotal

    muTotal = np.nanmean(np.asarray(muListTotal), axis=0)
    plt.plot(np.arange(len(muTotal)), muTotal, ls='--', c='k')
    plt.show()

    # Plot corrected spectrum
    hist, bins = np.histogram(dataCorrected, bins=np.linspace(-1, 80, 1000))
    plt.semilogy(bins[:-1], hist)
    plt.show()

    # Shift spectra onto first peak of mean
    muListTotal = np.asarray([ np.asarray(muListTotal[i])- (muListTotal - muTotal)[:, 0][i] for i in range(len(muListTotal)) ])

    # Calculate mean correction
    for pixel in range(len(muListTotal)):
        print 'Pixel %d: ' % pixel,
        muDiff = muTotal/muListTotal[pixel]
        print muListTotal[pixel]
        plt.plot(np.arange(muDiff.size), muDiff)
        plt.show()

    return slopeDict

def doubleLinear(x, m1, m2, t1, t2):
    t = float(t2 - t1) / (m1 - m2)
    return np.where(x <= t, m1*x + t1, m2*x + t2)

def Idontknow():
    for i in range(2, 15):
        x_ = np.asarray(x['Slot%d' % slot])[:,i].flatten()

        # Remove zeros
        x_ = x_[x_ > 0]
        print len(x_)

        a = 2.7501806945155627
        b = 8.482372245673544
        c = 74.51948122705083
        t = 2.609729787968528

        hist, bins = np.histogram(x_, bins=np.linspace(0, 2000, 1000))
        ax.step(bins[:-1], hist, where='post')
    plt.show()
    return

    bins_ = ToTtoEnergy(bins, a, b, c, t)
    # plt.plot(hist, hist_, marker='x', ls='')
    plt.step(bins_[:-1], hist, where='post')

    ax.set_xlabel('ToT')
    ax.set_ylabel('Counts')
    ax.set_xlim(0, 1200)
    plt.show()

def getFFT(x, y, deltaX, plot=True):
    # Calculate FFT
    freq = np.fft.rfftfreq(len(x), deltaX)
    freqAmplitude = np.fft.rfft(y)

    bandpassH = (1., np.infty)
    bandpassV = (-np.infty, -np.infty)

    if plot:
        plt.semilogy(freq, np.abs(np.real(freqAmplitude)))
        plt.semilogy(freq, np.abs(np.imag(freqAmplitude)))
        plt.axvspan(bandpassH[0], bandpassH[1], color='gray', alpha=.5)
        plt.axhspan(bandpassV[0], bandpassV[1], color='gray', alpha=.7)
        plt.xlabel('Frequency (a.u.)')
        plt.ylabel('FFT (a.u.)')
        plt.grid(which='both')

        plt.xlim(min(freq), max(freq))
        plt.ylim(min(freqAmplitude), max(freqAmplitude))
        plt.show()

    freqAmplitude[freqAmplitude < 5] = 0
    # freqAmplitude[~np.logical_or(freq < bandpassH[0], freq > bandpassH[1])] = 0
    # freqAmplitude[~np.logical_or(freqAmplitude < bandpassV[0], freqAmplitude > bandpassV[1])] = 0

    # Inverse FFT
    amplitude = np.fft.irfft( freqAmplitude )
    
    if plot:
        plt.semilogy(np.arange(len(amplitude)), amplitude)
        plt.xlabel('ToT')
        plt.ylabel('Counts (a.u.)')
        plt.grid(which='both')
        plt.show()
    
    return amplitude

# p: Period
def multiGauss(x, p):
    return

def normal(x, A, mu, sigma, off):
    return A*np.exp(-(x-mu)**2 / (2*sigma**2) ) + off

def EnergyToToT(x, a, b, c, t):
    return a*(x - b) + c*np.arctan((x - b)/t)

def ToTtoEnergy(x, a, b, c, t):
    f = lambda xVal : a*(xVal - b) + c*np.arctan((xVal - b)/t) - x
    return fsolve(f, np.full(len(x), 19))

def isLarge(pixel):
    if ( (pixel - 1) % 16 == 0 ) or ( pixel % 16 == 0 ) or ( (pixel + 1) % 16 == 0 ) or ( (pixel + 2) % 16 == 0 ):
        return False
    return True

def getColor(c, N, idx):
    import matplotlib as mpl
    cmap = mpl.cm.get_cmap(c)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=N - 1)
    return cmap(norm(idx))

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

def checkIfDirExists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == '__main__':
    main()

