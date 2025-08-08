# Quanbed

A Testbed for Quantum computer simulator.

Quanbed is a GPU-accelerated testbed for quantum circuit simulation. The code provides a `simulator` API (see `qcs.hpp`) implementing common gate operations, state preparation, and measurement using CUDA, MPI, and NCCL. The Makefile builds either a standalone `qcs` executable or the shared library `libqcs.so`.

## Building

The project requires NVIDIA's CUDA toolkit as well as NCCL, and MPI. Example build targets:

```
make qcs      # build standalone simulator
make target   # build shared library
```

## Running

You can execute the built simulator via:

```
make run
```

