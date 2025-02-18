#!/usr/bin/bash

set -e
set -x

for i in {24..37}; do
    echo "[info] num_qubits: $i" 1>&2
    sed -i -e 's/int const num_qubits = \([0-9]*\);/int const num_qubits = '"${i}"';/g' main.cu
    ./run_mpi.sh
done

