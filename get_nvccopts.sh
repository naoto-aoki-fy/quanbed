#!/bin/bash

nvcc_options_fn=.nvcc_options

if [[ -f .nvcc_options ]]; then
    cat "$nvcc_options_fn"
    exit
fi

read -r -a mpicxx_output < <(mpicxx -show)

# Array to store options for nvcc
nvcc_options=()

# Loop through the array and process
for word in "${mpicxx_output[@]}"; do
    case $word in
        # Exclusion conditions
        nvc++|*-pthread)
            continue ;;  # Skip
        # If it starts with -Wl,, convert to -Xlinker
        -Wl,*)
            IFS=',' read -ra parts <<< "$word"
            for part in "${parts[@]:1}"; do  # Skip the first -Wl,
                nvcc_options+=("-Xlinker" "$part")  # Add to array
            done
            ;;
        # Other cases: add as is
        *)
            nvcc_options+=("$word") ;;  # Add to array
    esac
done

echo "${nvcc_options[@]}" | tee "$nvcc_options_fn"
