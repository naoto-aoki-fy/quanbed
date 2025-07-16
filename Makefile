SM_VER ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | awk '{print $$1*10;}')
NVCC = nvcc
NVCCFLAGS = $(shell ./get_nvccopts.sh) -I./atlc/include -lcurand -lnccl -lssl -lcrypto --cudart=shared -O3 -Xcompiler -fopenmp -std=c++11 -rdc=true -Wno-deprecated-gpu-targets -gencode=arch=compute_$(SM_VER),code=sm_$(SM_VER)

.Phony: target
target: qcs

qcs: main.cu
	nvcc $(NVCCFLAGS) main.cu -o qcs

.Phony: run
run: qcs
	mpirun $(MPIRUN_FLAGS) ./qcs

.Phony: clean
clean:
	$(RM) qcs
