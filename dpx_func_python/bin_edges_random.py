#!/usr/bin/env python
import numpy as np

def getBinEdgesRandom(NPixels, edgeMin, edgeMax, edgeOvfw, uniform=False):
    edgeList = []
    for pixel in range(NPixels):
        if uniform:
            # Calculate bin edges
            binEdges = np.sort(np.random.uniform(edgeMin, edgeMax, 15))
            binEdges = np.insert(binEdges, 0, edgeMin)
            binEdges = np.append(binEdges, edgeOvfw)
        else:
            # Get bin edges with noisy evenly spaced distances
            binEdges = getBinEdgesRandomEvenSpace(edgeMin, edgeMax, edgeOvfw)

            if any(np.isnan(binEdges)):
                binEdges = getBinEdgesRandomEvenSpace(edgeMin, edgeMax, edgeOvfw)

        # Convert to ToT
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

def getBinEdgesUniform(pixels, edgeMin, edgeMax, edgeOvfw):
    edgeList = []
    xInit = np.linspace(edgeMin, edgeMax, 16)
    bw = xInit[1] - xInit[0]
    pixelIdx = 0

    pixels_filt = [pixel for pixel in pixels if isLarge(pixel)]
    for pixel in pixels:
        if not isLarge(pixel):
            edgeList.append(np.append(xInit, edgeOvfw))
            continue

        offset = bw / float(len(pixels_filt)) * pixelIdx
        edgeList.append(np.append(xInit + offset, edgeOvfw))
        pixelIdx += 1
    return edgeList

def isLarge(pixel):
	if pixel % 16 in [0, 1, 14, 15]:
		return False
	else:
		return True

# Conversion functions
# Energy to ToT and vice versa
def ToTtoEnergySimple(self, x, a, b, c, t, h=1, k=0):
    return h * (b + 1./(4 * a) * (2*x + np.pi*c + np.sqrt(16 * a * c * t + (2 * x + np.pi * c)**2))) + k

def EnergyToToTSimple(x, a, b, c, t, h=1, k=0):
    res = np.where(x < b, a*((x - k)/h - b) - c * (np.pi / 2 + t / ((x - k)/h - b)), 0)
    res[res < 0] = 0
    return res
