#!/bin/bash
#SBATCH --job-name=job
#SBATCH --output=output_%j.out
#SBATCH --error=output_%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:00:00

# declare -p | grep -E '^declare -x' 1>&2

python3 -c "import os, json, sys; print(json.dumps({k: v for k, v in os.environ.items() if k.startswith(\"SLURM\")}));" 1>&2

module load system/${SLURM_JOB_PARTITION} nvhpc

set -xe

ORIGINAL_CODE_FN="main.cu"

FN_EXT="${ORIGINAL_CODE_FN##*.}"
PROGRAM_NAME="${ORIGINAL_CODE_FN%.*}_${SLURM_JOBID}"
CODE_FN="${PROGRAM_NAME}.${FN_EXT}"
EXE_FN="${PROGRAM_NAME}_${SLURM_JOBID}.exe"

cp "${ORIGINAL_CODE_FN}" "${CODE_FN}"

# mpicxx -O3 "${CODE_FN}" -o "${EXE_FN}" -cudalib=nccl

nvcc -std=c++17 -O3 "${CODE_FN}" -o "${EXE_FN}"

./"${EXE_FN}"