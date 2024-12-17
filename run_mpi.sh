#!/bin/bash

DATENOW="$(date +%Y%m%d_%H%M_%S)"

OPTARG="-O3"

HOSTNAME="$(hostname)"
JOB_PARTITION="${HOSTNAME%%.*}"
JOB_PARTITION="${JOB_PARTITION%-*}"
# JOB_PARTITION="${1}"

ORIGINAL_CODE_FN="${1:-"main.cu"}"

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

exec > >(tee "${OUTPUT_STDOUT}" >&3) 2> >(tee "${OUTPUT_STDERR}" >&4)
# exec >"${OUTPUT_STDOUT}" 2> >(tee "${OUTPUT_STDERR}" >&4)

python3 -c "import os, json, sys; print(json.dumps({k: v for k, v in os.environ.items() if k.startswith(\"SLURM\")}));" 1>"${OUTPUT_STDERR}" 1>&2

module load system/${JOB_PARTITION} nvhpc

set -e

mpicxx -std=c++17 ${OPTARG} "${CODE_FN}" -cudalib=curand,nccl -o "${EXE_FN}"

# mpicxx_output=$(mpicxx -show)
# linker_options=$(<<< "$mpicxx_output" grep -oP '(?<=-Wl,)[^ ]+' | sed 's/^/-Xlinker /' | tr '\n' ' ')
# non_wl_options=$(<<< "$mpicxx_output" tr ' ' '\n' | grep -Ev '^(nvc\+\+|-pthread|-Wl,)' | tr '\n' ' ')

# nvcc \
# -I/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/12.6/hpcx/hpcx-2.20/ompi/include -I/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/12.6/hpcx/hpcx-2.20/ompi/include/openmpi -I/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/12.6/hpcx/hpcx-2.20/ompi/include/openmpi/opal/mca/hwloc/hwloc201/hwloc/include -I/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/12.6/hpcx/hpcx-2.20/ompi/include/openmpi/opal/mca/event/libevent2022/libevent -I/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/12.6/hpcx/hpcx-2.20/ompi/include/openmpi/opal/mca/event/libevent2022/libevent/include -L/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/12.6/hpcx/hpcx-2.20/ompi/lib \
#   -Xlinker -rpath -Xlinker /opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/12.6/hpcx/hpcx-2.20/ompi/lib \
#   -Xlinker --enable-new-dtags \
#   -lmpi \
#   -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90,code=sm_90 \
#   -std=c++17 ${OPTARG} "${CODE_FN}" -lcurand -lnccl -o "${EXE_FN}"

# export NCCL_DEBUG=TRACE
mpirun --oversubscribe -np 8 ./"${EXE_FN}"
# mpirun -np 1 ./"${EXE_FN}"

echo "[info] the end of job card" 1>&2

exec 1>&3 2>&4