#!/bin/bash

DATENOW="$(date +%Y%m%d_%H%M_%S)"

OPTARG="-O3"
JOB_PARTITION="${1}"

ORIGINAL_CODE_FN="main.cu"

FN_EXT="${ORIGINAL_CODE_FN##*.}"
PROGRAM_NAME="${ORIGINAL_CODE_FN%.*}_${DATENOW}"
CODE_FN="${PROGRAM_NAME}.${FN_EXT}"
EXE_FN="${PROGRAM_NAME}.exe"

cp "${ORIGINAL_CODE_FN}" "${CODE_FN}"

JOB_FN="job_${DATENOW}.sh"
JOBNAME="${DATENOW}"

OUTPUT_FN="output_${DATENOW}"
OUTPUT_STDOUT="${OUTPUT_FN}.out"
OUTPUT_STDERR="${OUTPUT_FN}.err"

exec 3>&1 4>&2

exec >"${OUTPUT_FN}.out" 2>"${OUTPUT_FN}.err"

python3 -c "import os, json, sys; print(json.dumps({k: v for k, v in os.environ.items() if k.startswith(\"SLURM\")}));" 1>"${OUTPUT_STDERR}" 1>&2

module load system/${JOB_PARTITION} nvhpc

nvcc -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90,code=sm_90 -Xcompiler -fopenmp -std=c++17 "${OPTARG}" "${CODE_FN}" -o "${EXE_FN}"

./"${EXE_FN}"

echo "[info] the end of job card" 1>&2

exec 1>&3 2>&4