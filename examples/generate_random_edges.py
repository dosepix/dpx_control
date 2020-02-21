#!/usr/bin/env python
import numpy as np
import hickle
import json
import argparse
import sys
sys.path.insert(0, '../dpx_func_python/')
import bin_edges_random as ber

def main():
    out, shifted, minenergy, maxenergy = parse_args()

    # Generate edges
    if not shifted:
        binEdges = ber.getBinEdgesRandom(NPixels=256, edgeMin=minenergy, edgeMax=maxenergy, edgeOvfw=430, uniform=False)
        binEdges = [list(e) for e in binEdges]
    else:
        # Generate edges for multiple energy regions
        binEdges = []
        for idx in range(4):
            edges = ber.getBinEdgesUniform(NPixels=256, edgeMin=minenergy + idx*(maxenergy - minenergy), edgeMax=minenergy + (idx + 1)*(maxenergy - minenergy), edgeOvfw=430)
            binEdges.append( [list(e) for e in edges] )

    # hickle.dump(binEdgesDict, BIN_EDGES_RANDOM_FN)
    with open(out, 'w') as f:
        json.dump(binEdges, f)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-out', '--outfile', action='store', required=True, help='output-file')
    parser.add_argument('-sh', '--shifted', action='store_true', required=False, help='Create shifted edges')
    parser.add_argument('-min_e', '--minenergy', action='store', type=int, required=True, help='Minimum energy')
    parser.add_argument('-max_e', '--maxenergy', action='store', type=int, required=True, help='Maximum energy')
    args = parser.parse_args()
    return args.outfile, args.shifted, args.minenergy, args.maxenergy

if __name__ == '__main__':
    main()

