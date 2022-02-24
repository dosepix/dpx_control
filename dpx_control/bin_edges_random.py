#!/usr/bin/env python
import numpy as np

def get_bin_edges_random(NPixels, edgeMin, edgeMax, edgeOvfw, uniform=False):
    edge_list = []
    for pixel in range(NPixels):
        if uniform:
            # Calculate bin edges
            bin_edges = np.sort(np.random.uniform(edgeMin, edgeMax, 15))
            bin_edges = np.insert(bin_edges, 0, edgeMin)
            bin_edges = np.append(bin_edges, edgeOvfw)
        else:
            # Get bin edges with noisy evenly spaced distances
            bin_edges = get_bin_edges_randomEvenSpace(edgeMin, edgeMax, edgeOvfw)

            if any(np.isnan(bin_edges)):
                bin_edges = get_bin_edges_randomEvenSpace(
                    edgeMin, edgeMax, edgeOvfw)

        # Convert to ToT
        edge_list.append(bin_edges)
    return edge_list

def get_bin_edges_randomEvenSpace(edgeMin, edgeMax, edgeOvfw):
    # Mean difference
    diff = float(edgeMax - edgeMin) / 15
    bin_diff = np.random.normal(diff, diff / 3., 15)
    bin_diff -= (np.sum(bin_diff) - (edgeMax - edgeMin)) / 15
    bin_edges = np.cumsum(bin_diff) + edgeMin
    bin_edges = np.insert(bin_edges, 0, 1.5 * edgeMin +
                         abs(np.random.normal(0, 0.5 * diff)))
    bin_edges = np.append(bin_edges, edgeOvfw)
    bin_edges = np.sort(bin_edges)

    return bin_edges

def getBinEdgesUniform(pixels, edgeMin, edgeMax, edgeOvfw):
    edge_list = []
    x_init = np.linspace(edgeMin, edgeMax, 16)
    bw = x_init[1] - x_init[0]
    pixel_idx = 0

    pixels_filt = [pixel for pixel in pixels if isLarge(pixel)]
    for pixel in pixels:
        if not isLarge(pixel):
            edge_list.append(np.append(x_init, edgeOvfw))
            continue

        offset = bw / float(len(pixels_filt)) * pixel_idx
        edge_list.append(np.append(x_init + offset, edgeOvfw))
        pixel_idx += 1
    return edge_list

def isLarge(pixel):
    if pixel % 16 in [0, 1, 14, 15]:
        return False
    return True

# Conversion functions
# Energy to ToT and vice versa
def ToT_to_energy_simple(self, x, a, b, c, t, h=1, k=0):
    return h * (b + 1. / (4 * a) * (2 * x + np.pi * c +
                np.sqrt(16 * a * c * t + (2 * x + np.pi * c)**2))) + k


def energy_to_ToT_simple(x, a, b, c, t, h=1, k=0):
    res = np.where(x < b, a * ((x - k) / h - b) - c *
                   (np.pi / 2 + t / ((x - k) / h - b)), 0)
    res[res < 0] = 0
    return res
