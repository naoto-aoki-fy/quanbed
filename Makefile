SM_VER ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | awk '{print $$1*10;}')
NVCC = nvcc
NVCCFLAGS = $(shell ./nvccoptions/get_nvccopts.sh) -Xcompiler -Wformat=2 -I./atlc/include -lcurand -lnccl -lssl -lcrypto --cudart=shared -O3 -Xcompiler -fopenmp -Xcompiler -rdynamic -std=c++11 -rdc=true -Wno-deprecated-gpu-targets -gencode=arch=compute_$(SM_VER),code=sm_$(SM_VER)
MPIRUN = mpirun

.PHONY: target
target: lib/libqcs.so

qcs: qcs.cu qcs.hpp
	$(NVCC) -DQCS_BUILD_STANDALONE $(NVCCFLAGS) $< -o $@

lib/libqcs.so: qcs.cu qcs.hpp
	mkdir -p lib
	$(NVCC) -Xcompiler -fPIC -shared $(NVCCFLAGS) $< -o $@


.PHONY: run
run: qcs
	$(MPIRUN) $(MPIRUN_FLAGS) ./$<

.PHONY: clean
clean:
	$(RM) qcs lib/libqcs.so
