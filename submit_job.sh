#!/bin/bash

DATENOW="$(date +%Y%m%d_%H%M_%S)"

set -xe

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

cat <<EOF > "${JOB_FN}"
#!/bin/bash
#SBATCH --job-name="${JOBNAME}"
#SBATCH --partition="${JOB_PARTITION}"
#SBATCH --output="${OUTPUT_FN}.out"
#SBATCH --error="${OUTPUT_FN}.err"
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:00:00

# declare -p | grep -E '^declare -x' 1>&2
python3 -c "import os, json, sys; print(json.dumps({k: v for k, v in os.environ.items() if k.startswith(\"SLURM\")}));" 1>&2

module load system/${JOB_PARTITION} nvhpc

set -xe

nvcc -Xcompiler -fopenmp -std=c++17 "${OPTARG}" "${CODE_FN}" -o "${EXE_FN}"

./"${EXE_FN}"

echo "[info] the end of job card" 1>&2
EOF

sbatch "${JOB_FN}"

echo "[info] date: ${DATENOW}" 1>&2