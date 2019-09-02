#!/usr/bin/env python
import numpy as np
import hickle as hck
import matplotlib.pyplot as plt
import scipy.signal
import scipy.optimize
import os

# directory = 'noiseShift_20_45/'
directory = 'noiseShift/'
pixel = 8

def getColor(c, N, idx):
    import matplotlib as mpl
    cmap = mpl.cm.get_cmap(c)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=N - 1)
    return cmap(norm(idx))

def rolling_std(x, win=10):
    roll_std = []
    for idx in range(len(x) - win):
        roll_std.append( np.std(x[idx:idx+win]) )
    return roll_std

def makeNoiseDict(directory):
    x, y = {}, {} 
    for fn in os.listdir(directory):
        if fn.endswith('.p') or fn.endswith('.hck'):
            print fn
            try:
                num = int( fn.split('.')[0].split('_')[-1] )
                data = hck.load(directory + fn)
                x[num] = data['THL']
                y[num] = np.asarray(data['data']).reshape(-1, 256).T[pixel]
            except:
                continue

    return {'x': x, 'y': y}

def main():
    fn = directory.split('/')[0] + '.hck'
    # noiseDict = makeNoiseDict(directory)
    # hck.dump(noiseDict, fn)
    # return
    
    noiseDict = hck.load(fn)
    x, y = noiseDict['x'], noiseDict['y']

    win = 30
    slope = []
    for num in sorted(x.keys()):
        a, b = np.asarray(x[num]), y[num]
        a = a[b < 50]
        b = b[b < 50]

        # plt.plot(a, b)
        a, b = a, scipy.signal.savgol_filter(b, 11, 3)
        a = a[b < 20]
        b = b[b < 20]

        # Group
        dist = abs(np.diff(a))
        groupTotal = []
        groupX, groupY = [], []
        for idx, d in enumerate(dist):
            if d == 1:
                groupX.append(a[idx])
                groupY.append(b[idx])
            else:
                if len(groupX):
                    groupTotal.append([groupX, groupY])
                groupX, groupY = [], []

        if not groupTotal:
            continue
        if num > 1:
            a, b = groupTotal[0][0], scipy.signal.savgol_filter(groupTotal[0][1], 11, 3)
        else:
            a, b = groupTotal[0][0], groupTotal[0][1]
        
        def linear(x, m, t):
            return m * (x - t)
        def expFit(x, a, b, c):
            return a * np.exp(-(x - b) / c)

        # p0 = (1., a[-1], 3.)
        p0 = (-1., a[-1])
        try:
            # popt, pcov = scipy.optimize.curve_fit(expFit, a[:-10], b[:-10], p0=p0)
            popt, pcov = scipy.optimize.curve_fit(linear, a[-20:], b[-20:], p0=p0)
        except:
            popt = p0

        slope.append(popt[-1])

        print popt[-1]
        color = getColor('viridis', len(x.keys()), num - 1)
        plt.plot(a, linear(a, *popt), ls='--', color=color)
        plt.plot(a[-20:], b[-20:], color=color)
        # plt.plot(a[:-1], scipy.signal.savgol_filter(np.diff(scipy.signal.savgol_filter(b, 31, 3)), 31, 3), label=num)
        continue

        maxima = np.argsort(-np.diff(scipy.signal.savgol_filter(b, 11, 3))).flatten()
        m1, m2 = np.sort(maxima[:2])
        a = a[m1:m2]
        b = b[m1:m2]
        print len(a), len(b)

        plt.plot(a, b)
        #plt.plot(a, scipy.signal.savgol_filter(b, 11, 3), label=num)
        # plt.plot(a[:-1], np.diff(scipy.signal.savgol_filter(b, 11, 3)), label=num)

        # plt.plot(x[num][:-win], rolling_std(y[num], win), label=num)
        # plt.plot(x[num][:-win], scipy.signal.savgol_filter(rolling_std(y[num], win), 11, 3), label=num)
        # plt.plot(x[num][:-1][:-win], np.diff(rolling_std(y[num], win)), label=num)

    plt.ylim(-2, 22)
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(range(11) + range(12, 22, 2), slope, marker='x')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()

