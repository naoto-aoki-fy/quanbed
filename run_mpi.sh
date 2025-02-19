#!/bin/bash

set -e

WORKDIR="20250219_1026_a100_openssl"
WORKDIR="${WORKDIR%/}"
mkdir -p "${WORKDIR}"

DATENOW="$(date +%Y%m%d_%H%M_%S)"

# OPTARG="-O3"
OPTARG=("-O3" "-Xcompiler" "-fopenmp" "-std=c++17" "-lnccl" "-lcurand" "-lssl" "-lcrypto")

HOSTNAME_FQDN="$(hostname)"
HOSTNAME="${HOSTNAME_FQDN%%.*}"
JOB_PARTITION="${HOSTNAME%-*}"
# JOB_PARTITION="${1}"

ORIGINAL_CODE_FN="${1:-"main.cu"}"

FN_EXT="${ORIGINAL_CODE_FN##*.}"
PROGRAM_NAME="${WORKDIR}/${ORIGINAL_CODE_FN%.*}_${DATENOW}"
CODE_FN="${PROGRAM_NAME}.${FN_EXT}"
EXE_FN="${PROGRAM_NAME}.exe"

cp "${ORIGINAL_CODE_FN}" "${CODE_FN}"

OUTPUT_FN="${WORKDIR}/output_${DATENOW}"
OUTPUT_STDOUT="${OUTPUT_FN}.out"
OUTPUT_STDERR="${OUTPUT_FN}.err"

exec 3>&1 4>&2

exec > >(tee "${OUTPUT_STDOUT}" >&3) 2> >(tee "${OUTPUT_STDERR}" >&4)
# exec >"${OUTPUT_STDOUT}" 2> >(tee "${OUTPUT_STDERR}" >&4)

python3 -c "import os, json, sys; print(json.dumps({k: v for k, v in os.environ.items() if k.startswith(\"SLURM\")}));" 1>"${OUTPUT_STDERR}" 1>&2

module load system/${JOB_PARTITION} nvhpc

echo "[info] getting mpicxx params" 1>&2

# `mpicxx -show` の出力を配列として取得
read -r -a mpicxx_output < <(mpicxx -show)

# nvcc用のオプションを格納する配列
nvcc_options=()

# 配列をループして処理
for word in "${mpicxx_output[@]}"; do
    case $word in
        # 除外条件
        nvc++|*-pthread) 
            continue ;;  # スキップ
        # -Wl,で始まる場合は-Xlinkerに変換
        -Wl,*) 
            IFS=',' read -ra parts <<< "$word"
            for part in "${parts[@]:1}"; do  # 最初の -Wl, をスキップ
                nvcc_options+=("-Xlinker" "$part")  # 配列に追加
            done
            ;;
        # その他はそのまま追加
        *) 
            nvcc_options+=("$word") ;;  # 配列に追加
    esac
done

set -x
nvcc \
  "${nvcc_options[@]}" \
  -gencode=arch=compute_80,code=sm_80 \
  -gencode=arch=compute_90,code=sm_90 \
  "${OPTARG[@]}" "${ORIGINAL_CODE_FN}" -o "${EXE_FN}"

# mpicxx -std=c++17 ${OPTARG} "${CODE_FN}" -cudalib=curand,nccl -o "${EXE_FN}"

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG=TRACE
# -host "${HOSTNAME_FQDN%%.*}"
mpirun --oversubscribe -np 8 -host "${HOSTNAME_FQDN%%.*}" ./"${EXE_FN}"
# mpirun -np 1 ./"${EXE_FN}"

set +x
set +e

echo "[info] the end of job card" 1>&2

exec 1>&3 2>&4