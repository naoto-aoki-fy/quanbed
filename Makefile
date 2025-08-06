SM_VER ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | awk '{print $$1*10;}')
NVCC = nvcc
NVCCFLAGS = $(shell ./get_nvccopts.sh) -Xcompiler -Wformat=2 -I./atlc/include -lcurand -lnccl -lssl -lcrypto --cudart=shared -g -O3 -Xcompiler -fopenmp -std=c++11 -rdc=true -Wno-deprecated-gpu-targets -gencode=arch=compute_$(SM_VER),code=sm_$(SM_VER)
MPIRUN = mpirun

.PHONY: target
target: qcs

qcs: qcs.cu qcs.hpp
	$(NVCC) $(NVCCFLAGS) $< -o $@

.PHONY: run
run: qcs
	$(MPIRUN) $(MPIRUN_FLAGS) ./$<

.PHONY: clean
clean:
	$(RM) qcs
