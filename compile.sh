#!/bin/bash

set -xe

OPTARG="-O3"

CODE_FN="main.cu"

EXE_FN="main.exe"

HOSTNAME="$(hostname)"
PARTITION="${HOSTNAME%%.*}"
PARTITION="${PARTITION%-*}"

set +x
module load system/${PARTITION} nvhpc
set -x

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

nvcc "${nvcc_options[@]}" \
    -gencode=arch=compute_80,code=sm_80 \
    -gencode=arch=compute_90,code=sm_90 \
    -Xcompiler -fopenmp \
    -std=c++17 \
    -lnccl -lcurand \
    "${OPTARG}" "${CODE_FN}" -o "${EXE_FN}"
