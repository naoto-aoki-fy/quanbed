#!/bin/bash

set -xe

OPTARG="-O3"

JOB_PARTITION="${1}"

CODE_FN="main.cu"

EXE_FN="/dev/null"

nvcc -Xcompiler -fopenmp -std=c++17 "${OPTARG}" "${CODE_FN}" -o "${EXE_FN}"
