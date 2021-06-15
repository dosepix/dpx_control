import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import json
import sys
import rebin as rb

# === Load Data ===
def loadSingleRegion(dataDir, doseFile, binEdgesFile):
    if doseFile.endswith('.hck'):
        doseDict = hck.load(dataDir + doseFile)
    elif doseFile.endswith('.json'):
        doseDict = json.load(open(dataDir + doseFile, 'r'))
    doseDict = {slot: np.asarray(doseDict[slot])[:,:,:,1:] for slot in doseDict.keys()}
    
    if binEdgesFile.endswith('.hck'):
        binEdgesRandomDict = hck.load(binEdgesFile)
    elif binEdgesFile.endswith('.json'):
        binEdgesRandomDict = json.load(open(binEdgesFile, 'r'))
        
    if len(np.asarray(binEdgesRandomDict).shape) > 2:
        binEdgesRandomDict = binEdgesRandomDict[0]
    '''
    if paramsDictFile.endswith('.hck'):
        paramsDict = hck.load(paramsDictFile)
    '''

    binEdgesCorrDict = {'Slot%d' % slot: binEdgesRandomDict['Slot%d' % slot][0] for slot in range(1, 3 + 1)}
    return doseDict, binEdgesCorrDict

def loadMultiRegionSingle(doseDict_, timeDict_, binEdgesRandomDict, split=1):
    doseDict = {}
    regionRange = list( range(len(doseDict_[list(doseDict_.keys())[0]].keys())) )
    for slot in doseDict_.keys():
        # Minimum number of frames
        minLength = min([len(doseDict_[slot]['Region%d' % region]) for region in regionRange])
        sumList = []
        for region in reversed(regionRange):
            d = doseDict_[slot]['Region%d' % region][:minLength]
            if not d:
                continue
            # Remove overflow bin and sum over frames
            data = np.sum(d, axis=0)[:,:,1:] / (float(len(d)) * np.sum(timeDict_[slot]['Region%d' % region][:minLength]))
            print( np.sum(data[:,:,-1]), data.shape )

            if split > 1:
                splitList = []
                for s in reversed(range(split)):
                    data_split = data.reshape((256, -1))[256 // split * s: 256 // split * (s + 1)]
                    data_split_pad = np.pad(data_split, ((0, 256 - 256 // split), (0, 0)), 'constant', constant_values=(0, 0))
                    splitList.append( data_split_pad.reshape((16, 16, -1)) )
                data = np.dstack( splitList )
                data.reshape([1] + list(data.shape))

            # plt.imshow(data.reshape((256, -1)).T, aspect='auto')
            # plt.show()
            sumList.append( data )

        temp = np.dstack(sumList)
        doseDict[slot] = temp.reshape([1] + list(temp.shape))
        
    # bin edges
    newEdgesDict = {}
    for slot in binEdgesRandomDict.keys():
        if binEdgesRandomDict[slot] is None:
            newEdgesDict[slot] = None
            continue

        newEdges = np.hstack(np.asarray(binEdgesRandomDict[slot][:len(regionRange)])[:,:,:-2])
        if split > 1:
            edge_split_sum = []
            for s in range(split):
                edge_split = newEdges[256 // split * s: 256 // split * (s + 1)]
                edge_split_sum.append( edge_split )
            newEdgesSplit = np.hstack( edge_split_sum )
            newEdges = np.vstack([newEdgesSplit] * split)
        newEdgesDict[slot] = newEdges
        
    return doseDict, newEdgesDict

# === Rebinning ===
def rebinEnergyData(edges, data):
    isLarge = np.asarray([pixel for pixel in np.arange(len(data)) if pixel % 16 not in [0, 1, 14, 15]])
    return rebinEnergyHistExec(np.asarray(edges)[isLarge], np.asarray(data)[isLarge])

def rebinEnergyHist(edges, events, NPixel=192, plot=False):
    binsList, histList = [], []
    dataList = []

    events = np.asarray( events )

    # Total number of events
    NEvents = len(events)

    if plot:
        for i in range(len(edges)):
            plt.hist(events[events > edges[i][0]], bins=edges[i], alpha=0.5, density=False)

    for i in range(NPixel):
        # Get bin edges
        binEdges = edges[i]

        # Select data
        data = events

        # Discard empty events
        data = np.asarray( data )
        data = data[data > 0]
        dataList += list( data )

        # Calculate histogram
        try:
            hist, bins = np.histogram(data, bins=binEdges, density=False)
            hist += np.asarray(np.random.normal(0, np.sqrt(hist)), dtype=int)
            histList.append( hist ), binsList.append( bins )

            binsNew, histNew = rebinEnergyHistExec(binsList, histList)
        except:
            continue
            
        # Plot
        if plot:
            plt.step(binsNew[:-1], histNew, where='post') 
            hist, bins = np.histogram(dataList, bins=100)
            plt.step(bins[:-1], hist, where='post')
            plt.show()
            
    return binsNew, histNew

def rebinEnergyHistExec(binsList, histList):
    binsNew = np.sort(np.hstack(binsList))
    histNewList = []
    NPixel = len(binsList)

    for i in range(NPixel):
        bins, hist = binsList[i], histList[i]
        # plt.step(bins[:-1], hist, where='post')

        h = np.zeros(len(binsNew) - 1)
        for j in range(len(bins) - 2):
            if not hist[j]:
                continue

            b1, b2 = bins[j:j+2]
            # print b1, b2
            bw = float(b2 - b1)

            bCond = np.logical_and(binsNew[:-1] >= b1, binsNew[:-1] <= b2)
            h[bCond] = list(np.diff(binsNew[:-1][bCond]) / bw * hist[j]) + [0]

        h *= (np.sum(hist) / np.sum(h))
        # Last bin
        b1 = bins[-2]
        h[binsNew[:-1] >= b1] = hist[-1] / (bins[-1] + b1)
        histNewList.append( h )
    # plt.show()

    histNew = np.nan_to_num(np.nansum(histNewList, axis=0))

    # Remove duplicates
    histComb = sorted(list(set(zip(binsNew[:-1], histNew))))
    binsNew_, histNew = zip(*histComb)
    binsNew = list(binsNew_) + [binsNew[-1]]
    if histNew[0] == 0:
        histNew = histNew[1:]
        binsNew = binsNew[1:]
    histNew = np.asarray(histNew) / NPixel

    return binsNew, histNew

def rebinSpectrum(slots, doseDict, binEdgesCorrDict, bin_width, rebin, save_data, save_data_fn=None, plot_fn=None):
    save_data_dict = {}
    titles = ['Slot%d' % slot for slot in range(1, 3 + 1)] # ['vac', 'al', 'sn']
    
    energy_max = 250
    slotList = []
    for slot in slots:
        # Get minimum and maximum energy values
        flatEdges = np.asarray(binEdgesCorrDict['Slot%d' % slot]).T[:-1].flatten()
        minE, maxE = min(flatEdges), max(flatEdges)
        print( minE, maxE )

        # Sum over frames and flatten
        dose = np.flip( np.reshape( np.sum(np.asarray(doseDict['Slot%d' % slot]), axis=0), (256, -1) ), axis=1)

        # Rebin data
        print( binEdgesCorrDict['Slot%d' % slot].shape )
        binsNew, histNew = rebinEnergyData(binEdgesCorrDict['Slot%d' % slot], dose)
        # xNew = np.arange(minE, maxE + bin_width, bin_width) 
        binsNew = np.asarray(binsNew) - bin_width
        minE, maxE = 10, 120
        xNew = np.linspace(minE, maxE, rebin + 1)
        yNew = np.nan_to_num(rb.rebin(binsNew, histNew, xNew, interp_kind='piecewise_constant'))

        if len(slots) > 1 and slot == 1:
            slot1max = np.nanmax(yNew)
        elif len(slots) == 1:
            slot1max = np.nanmax(yNew)

        x = xNew[:-1][np.logical_and(xNew[:-1] >= 10, xNew[:-1] < energy_max)]
        y = yNew[np.logical_and(xNew[:-1] >= 10, xNew[:-1] < energy_max)]
        # x, y = xNew[:-1], yNew
        print( yNew.shape )
        print( x.shape, y.shape )
        y /= slot1max

        save_data_dict[titles[slot - 1]] = y / np.sum(y) * np.sum(doseDict['Slot%d' % slot])
        # Plot
        plt.step(x, y / np.sum(y), where='post', label=titles[slot-1])

        slotList.append( yNew )
        
    save_data_dict['bins'] = x

    if save_data:
        with open(save_data_fn, 'w') as f:
            json.dump({key: save_data_dict[key].tolist() for key in save_data_dict.keys()}, f)

    if len(slots) > 1:
        plt.legend()

    # plt.yscale('log')
    # plt.xlim(minE, maxE)
    _ = plt.xlabel('Energy (keV)')
    plt.grid()
    plt.ylabel('Normalized counts')
    if plot_fn is not None:
        plt.savefig(plot_fn)
        plt.clf()
    
    return slotList, save_data_dict

# === Deconvolution ===
def wiener_deconvolution(signal, kernel, lambd):
    "lambd is the SNR"
    kernel = np.hstack((kernel, np.zeros(len(signal) - len(kernel)))) # zero pad the kernel to same length
    H = np.fft.fft(kernel)
    deconvolved = np.real(np.fft.ifft(np.fft.fft(signal)*np.conj(H)/(H*np.conj(H) + lambd**2)))
    return deconvolved

def water_level_decon(y_meas, window, eps=0.1):
    padded = np.zeros_like(y_meas)
    padded[:window.size] = window

    yfreq = np.fft.fft(y_meas)
    winfreq = np.fft.fft(padded)

    winfreq[winfreq < eps] = eps
    newfreq = yfreq / winfreq 

    return np.fft.ifft(newfreq)

# === Bin edges generation ===
def getBinEdgesRandom(NPixels, edgeMin, edgeMax, edgeOvfw, uniform=False, paramDict=None):
    edgeList = []
    for pixel in range(NPixels):
        if uniform:
            # Calculate bin edges
            binEdges = np.sort(np.random.uniform(edgeMin, edgeMax, 15))
            binEdges = np.insert(binEdges, 0, edgeMin)
            binEdges = np.append(binEdges, edgeOvfw)
        else:
            pixelOffset = 2
            if paramDict is not None:
                while pixel + pixelOffset not in paramDict.keys():
                    pixelOffset += 4
                params = paramDict[pixel + pixelOffset]

                a, b, c, t = params['a'], params['b'], params['c'], params['t']
                if 'h' in params.keys():
                    h, k = params['h'], params['k']
                else:
                    h, k = 1, 0

                # print( a, b, c, t, h, k )
                # Get min, max and overflow edges
                edgeMin_ = EnergyToToTSimple(edgeMin, a, b, c, t, h, k)
                edgeMax_ = EnergyToToTSimple(edgeMax, a, b, c, t, h, k)
                edgeOvfw_ = EnergyToToTSimple(edgeOvfw, a, b, c, t, h, k)
                if edgeMin_ > edgeMax_:
                    edgeMin_, edgeMax_ = edgeMax_, edgeMin_
                # print (edgeMin_, edgeMax_, edgeOvfw_)

                # Get bin edges with noisy evenly spaced distances
                binEdges = np.around(getBinEdgesRandomEvenSpace(edgeMin_, edgeMax_, edgeOvfw_))

                # Convert back to energy
                binEdges = ToTtoEnergySimple(np.asarray(binEdges), a, b, c, t, h, k)
                if any(np.isnan(binEdges)):
                    binEdges = getBinEdgesRandomEvenSpace(edgeMin, edgeMax, edgeOvfw)
                # print( binEdges )
                # print

            else:
                binEdges = getBinEdgesRandomEvenSpace(edgeMin, edgeMax, edgeOvfw)

        edgeList.append( binEdges )
    return edgeList

def getBinEdgesRandomEvenSpace(edgeMin, edgeMax, edgeOvfw):
    # Mean difference 
    diff = float(edgeMax - edgeMin) / 15
    binDiff = np.random.normal(diff, diff / 3., 15)
    binDiff -= (np.sum(binDiff) - (edgeMax - edgeMin)) / 15
    binEdges = np.cumsum(binDiff) + edgeMin
    binEdges = np.insert(binEdges, 0, 1.5 * edgeMin + abs(np.random.normal(0, 0.5*diff)))
    binEdges = np.append(binEdges, edgeOvfw)
    binEdges = np.sort( binEdges )

    return binEdges

def getBinEdgesUniform(NPixels, edgeMin, edgeMax, edgeOvfw):
    edgeList = []
    xInit = np.linspace(edgeMin, edgeMax, 16)
    bw = xInit[1] - xInit[0]
    pixelIdx = 0
    for pixel in range(NPixels):
        if not isLarge(pixel):
            edgeList.append(np.append(xInit, edgeOvfw))
            continue

        offset = bw / 192. * pixelIdx
        edgeList.append(np.append(xInit + offset, edgeOvfw))
        pixelIdx += 1
    return edgeList

# === Window estimation ===
def getRebinedHist(binEdges, data, plot=False):
    bins, hist = rebinEnergyHist(binEdges, data, NPixel=192, plot=False)
    bins, hist = np.asarray(bins), np.asarray(hist)
    binsLast = bins[-1]
    bins = bins[:-1][hist > 0]
    bins = np.append(bins, binsLast)
    hist = hist[hist > 0]
    
    if plot:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.step(bins[:-1], hist, where='post')
        ax.set_xlabel('Energy (keV)')
        ax.set_ylabel('Counts')
        plt.show()

    return bins, hist

def estimateWindow(x, binEdges, windowFunc, meanRange=np.arange(100, 150, 5), NEntries=10000, plot=False):
    windowTauList = []
    for idx, mean in enumerate(meanRange):
        data = [mean] * NEntries
        bins, hist = getRebinedHist(binEdges, data)
        y = np.nan_to_num(rb.rebin(bins, hist, x, interp_kind='piecewise_constant'))
        # y = scipy.signal.savgol_filter(y, 31, 3)

        p0 = (max(y), mean, 5)
        popt, pcov = scipy.optimize.curve_fit(windowFunc, x[:-1], y, p0=p0)
        print( p0, popt )
        windowTau = popt[-1]
        windowTauList.append( windowTau )
        
        if plot:
            color = 'C%d' % idx
            plt.plot(bins[:-1], np.asarray(hist, dtype=float)/np.max(hist), color=color)
            plt.plot(x[:-1], y/np.max(y), ls='--', color=color)

            yFit = windowFunc(bins[:-1], *popt)
            plt.plot(bins[:-1], yFit / np.max(yFit), ls='-.', color=color)
            plt.xlabel('Energy (keV)')
            plt.ylabel('Normalized counts')
            plt.xlim(20, 80)

    windowTau = np.mean(windowTauList)
    print(windowTau)
    return windowTau

# === Window functions ===
def expWindow(x, A, mu, tau):
    return A * np.exp(-np.abs(x - mu) / float(tau))

def triangleWindow(x, A, mu, tau):
    return np.where(np.logical_or(x < (mu - tau), x > (mu + tau)), 0, A * (1 - np.abs(x - mu) / float(tau)))

# === Temperature Correction ===
def correctTemperature(binEdgesDict, tempDict, tempCalibDict, slot=1):
    T = np.mean(tempDict['temp'])
    Toffset = tempCalibDict['Toffset']
    slope, offset = tempCalibDict['slope'], tempCalibDict['offset']

    tempCalibDict.keys()
    binEdgesCorrDict = {'Slot%d' % slot: []}
    for pixel in range(256):
        b = binEdgesRandomDict['Slot%d' % slot][pixel]
        if pixel not in paramsDict.keys():
            binEdgesDict['Slot%d' % slot].append( b )
            continue

        p = paramsDict[pixel]
        bToT = tte.EnergyToToT(b, p['a'], p['b'], p['c'], p['t'], p['h'], p['k'])
        b_new = pttt.getDataAtTSingle(bToT, T, slope[pixel], offset[pixel], Toffset, paramsDict[pixel], energy=True)
        if np.any(np.isnan(b_new)):
            binEdgesDict['Slot%d' % slot].append( b )
        else:
            binEdgesDict['Slot%d' % slot].append( b_new )
            
    return binEdgesCorrDict

# === Support ===
def getColor(c, N, idx):
    import matplotlib as mpl
    cmap = mpl.cm.get_cmap(c)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=N - 1)
    return cmap(norm(idx))

def getColorBar(ax, cbMin, cbMax, N=20, label=None, rotation=90):
    import matplotlib as mpl
    
    # Plot colorbar
    from matplotlib.colors import ListedColormap
    cmap = mpl.cm.get_cmap('viridis', N)
    norm = mpl.colors.Normalize(vmin=cbMin, vmax=cbMax)
    cBar = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical')

    # cBar.ax.invert_yaxis()
    cBar.formatter.set_powerlimits((0, 0))
    cBar.ax.yaxis.set_offset_position('right')
    cBar.update_ticks()

    labels = np.linspace(cbMin, cbMax, N + 1)
    locLabels = np.linspace(cbMin, cbMax, N)
    loc = labels + abs(labels[1] - labels[0])*.5
    cBar.set_ticks(loc)
    cBar.ax.set_yticklabels(['%.1f' % loc for loc in locLabels], rotation=rotation, verticalalignment='center')
    cBar.outline.set_visible(False)
    cBar.set_label(label)
    
def linear(x, m, t):
    return m * x + t

