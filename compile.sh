#!/bin/bash

set -xe

OPTARG="-O3"

JOB_PARTITION="${1}"

CODE_FN="main.cu"

EXE_FN="main.exe"

HOSTNAME="$(hostname)"
PARTITION="${HOSTNAME%%.*}"
PARTITION="${JOB_PARTITION%-*}"

module load system/${PARTITION} nvhpc

nvcc -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90,code=sm_90 -Xcompiler -fopenmp -std=c++17 "${OPTARG}" "${CODE_FN}" -o "${EXE_FN}"
