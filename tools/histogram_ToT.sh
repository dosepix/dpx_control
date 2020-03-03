#!/usr/bin/env bash
INDIR="../examples/calibration_measurements4/"
OUTDIR="$INDIR"
BMIN=1
BMAX=401
BINS=400

for d in "$INDIR"/*/; do
	python3 histogram_ToT.py -dir "$d" -bmin "$BMIN" -bmax "$BMAX" -b "$BINS" -out "$OUTDIR"
done

