#!/usr/bin/env bash
IKRUM="50"
INDIR="../examples/THLShift_Ikrum_${IKRUM}_det_22/"
OUTDIR="./THLShift_Ikrum_${IKRUM}_det_22/"
CALIB_DIR="../examples/calibration_parameters/"
P1="${CALIB_DIR}params_22_Ikrum${IKRUM}.json"
P2="${CALIB_DIR}params_6_Ikrum${IKRUM}.json"
P3="${CALIB_DIR}params_109_Ikrum${IKRUM}.json"
BMIN=10
BMAX=100
BINS=301

for d in "$INDIR"/*/; do
	python3 histogram_ToT.py -dir "$d" -p1 "$P1" -p2 "$P2" -p3 "$P3" -bmin "$BMIN" -bmax "$BMAX" -b "$BINS" -out "$OUTDIR"
done

