#!/usr/bin/env python
import numpy as np
import json
import argparse
import sys
sys.path.insert(0, '../dpx_func_python/')
import bin_edges_random as ber

def main():
    out, shifted, minenergy, maxenergy, regions, split = parse_args()
    generate_bin_edges(out, shifted, minenergy, maxenergy, regions, split)

def generate_bin_edges(out=None, shifted=False, minenergy=10, maxenergy=120, regions=1, split=4):
    # Generate edges
    if not shifted:
        binEdges = ber.getBinEdgesRandom(NPixels=256, edgeMin=minenergy, edgeMax=maxenergy, edgeOvfw=430, uniform=False)
        binEdges = [list(e) for e in binEdges]
    else:
        # Generate edges for multiple energy regions
        binEdges = []
        energyWidth = (maxenergy - minenergy) / float(regions)
        for idx in range(regions):
            edgeMin = minenergy + idx * energyWidth
            edgeMax = minenergy + (idx + 1) * energyWidth
            edgeWidth = (edgeMax - edgeMin) / float(split)
            edgeMax += edgeWidth

            split_edges = []
            for s in range(split):
                splitMin = edgeMin + s * edgeWidth
                splitMax = edgeMin + (s + 1) * edgeWidth # + edgeWidth / 15.

                # Generate edges
                pixels = np.arange(256 // split * s, 256 // split * (1 + s))
                edges = ber.getBinEdgesUniform(pixels=pixels, edgeMin=splitMin, edgeMax=splitMax, edgeOvfw=430)
                split_edges += list( edges )

            binEdges.append( [list(e) for e in split_edges] )

            # edges = ber.getBinEdgesUniform(pixels=np.arange(256), edgeMin=minenergy + idx*(maxenergy - minenergy), edgeMax=minenergy + (idx + 1) * (maxenergy - minenergy), edgeOvfw=430)
            # binEdges.append( [list(e) for e in edges] )

    if out is not None:
        with open(out, 'w') as f:
            json.dump(binEdges, f)
    
    return binEdges

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-out', '--outfile', action='store', required=True, help='output-file')
    parser.add_argument('-sh', '--shifted', action='store_true', required=False, help='Create shifted edges')
    parser.add_argument('-min_e', '--minenergy', action='store', type=int, required=True, help='Minimum energy')
    parser.add_argument('-max_e', '--maxenergy', action='store', type=int, required=True, help='Maximum energy')
    parser.add_argument('-r', '--regions', action='store', type=int, required=False, default=1, help='Number of regions')
    parser.add_argument('-s', '--split', action='store', type=int, required=False, default=1, help='Number of energy splits')
    args = parser.parse_args()
    return args.outfile, args.shifted, args.minenergy, args.maxenergy, args.regions, args.split

if __name__ == '__main__':
    main()
