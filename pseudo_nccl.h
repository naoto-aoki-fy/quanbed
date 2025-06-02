#pragma once
#ifndef NCCL_H_
#define NCCL_H_

#include <stdint.h>
#include <mpi.h>
#include <cuda_runtime.h>

#include "check_mpi.hpp"
#include "check_cuda.hpp"

#include <cdl.h>

typedef int ncclUniqueId;

typedef enum  {
    ncclSuccess = 0,
    ncclUnhandledCudaError = 1,
    ncclSystemError = 2,
    ncclInternalError = 3,
    ncclInvalidArgument = 4,
    ncclInvalidUsage = 5,
    ncclRemoteError = 6,
    ncclInProgress = 7,
} ncclResult_t;

inline static char const* ncclGetErrorString(ncclResult_t error) {
    switch (error) {
        case ncclSuccess: return "success";
        default: return "error";
    }
    return "<unknown>";
}

#include "check_x.hpp"

typedef enum {
    ncclChar = 0,
    ncclInt8 = 1,
    ncclUint8 = 2,
    ncclInt32 = 3,
    ncclUint32 = 4,
    ncclInt64 = 5,
    ncclUint64 = 6,
    ncclFloat16 = 7,
    ncclFloat32 = 8,
    ncclFloat64 = 9,
    ncclDouble = 9,
    ncclBfloat16 = 10,
    ncclHalf = 11,
    ncclBool = 12,
} ncclDataType_t;

inline uint64_t sizeof_ncclDataType_t(int datatype) {
    switch (datatype) {
        case ncclChar: return sizeof(char);
        case ncclInt8: return sizeof(int8_t);
        case ncclUint8: return sizeof(uint8_t);
        case ncclInt32: return sizeof(int32_t);
        case ncclUint32: return sizeof(uint32_t);
        case ncclInt64: return sizeof(int64_t);
        case ncclUint64: return sizeof(uint64_t);
        case ncclFloat16: return sizeof(__half);
        case ncclFloat32: return sizeof(float);
        case ncclFloat64: return sizeof(double);
        case ncclBfloat16: return sizeof(__nv_bfloat16);
        case ncclHalf: return sizeof(__half);
        case ncclBool: return sizeof(bool);
        default: return 0;
    }
}

inline ncclResult_t ncclGetUniqueId(ncclUniqueId* nccl_id) {
    *nccl_id = 1;
    return ncclSuccess;
}

typedef struct {
    void* buff;
    uint64_t count;
    int datatype;
    int peer;
    cudaStream_t stream;

} pnccl_sendrecv_args_t;

typedef struct {
    cudaIpcMemHandle_t handle;
    uint64_t offset;
} pnccl_handle_offset;

struct pnccl_comm_struct {
    bool in_group;
    std::vector<pnccl_sendrecv_args_t> send_args;
    std::vector<pnccl_sendrecv_args_t> recv_args;
    // std::vector<cudaIpcMemHandle_t> send_buff_handle_list;
    std::vector<pnccl_handle_offset> src_buff_handle_list;
    std::vector<void*> src_buff_list;
    std::vector<MPI_Request> mpi_request_list;
    std::vector<uint64_t> pointer_list;
    struct cdl_jmp_patch jmp_patch;
};

typedef pnccl_comm_struct* ncclComm_t;

pnccl_comm_struct nccl_comm_struct_private;

void* pncclGetClosestPointer(void* pointer_input, uint64_t* offset) {
    uint64_t num_ptrs = nccl_comm_struct_private.pointer_list.size();
    uint64_t distance_closest = (uint64_t)(-1);
    uint64_t pointer_closest = 0;
    for(uint64_t ptr_num = 0; ptr_num < num_ptrs; ptr_num++) {
        uint64_t pointer = nccl_comm_struct_private.pointer_list[ptr_num];
        // fprintf(stderr, "[%d] pointer=%p pointer_input=%p\n", __LINE__, pointer, pointer_input);
        if ((uint64_t)pointer_input < pointer) {
            continue;
        }
        uint64_t const distance = (uint64_t)pointer_input - pointer;
        if (distance < distance_closest) {
            distance_closest = distance;
            pointer_closest = pointer;
        }
    }
    if (pointer_closest != 0 && offset != 0) {
        *offset = distance_closest;
    }
    return (void*)pointer_closest;
}


// template<class T>
// inline cudaError_t pncclCudaMalloc(T **devPtr, size_t size) {
//     cudaError_t const ret = cudaMalloc(devPtr, size);
//     nccl_comm_struct_private.pointer_list.push_back((uint64_t)*devPtr);
//     // fprintf(stderr, "[%d] *devPtr=%p", __LINE__, *devPtr);
//     return ret;
// }

static cudaError_t (*orig_cudaMalloc)(void **, size_t) = NULL;

cudaError_t pncclCudaMalloc(void **devPtr, size_t size) {
    cudaError_t const ret = orig_cudaMalloc(devPtr, size);
    nccl_comm_struct_private.pointer_list.push_back((uint64_t)*devPtr);
    return ret;
}



inline ncclResult_t ncclCommInitRank(ncclComm_t* comm, int ndev, ncclUniqueId nccl_id, int rank) {
    *comm = &nccl_comm_struct_private;
    nccl_comm_struct_private.in_group = false;

    // plthook_t *plthook;
    // if (plthook_open(&plthook, NULL) != 0) {
    //     fprintf(stderr, "plthook_open error: %s\n", plthook_error());
    //     return ncclSystemError;
    // }

    // if (plthook_replace(plthook, "cudaMalloc", (void *)pncclCudaMalloc, (void **)&orig_cudaMalloc) != 0) {
    //     fprintf(stderr, "plthook_replace error: %s\n", plthook_error());
    //     return ncclSystemError;
    // }

    // struct cdl_jmp_patch jmp_patch = {};
    orig_cudaMalloc = ::cudaMalloc;
    nccl_comm_struct_private.jmp_patch = cdl_jmp_attach((void**)&orig_cudaMalloc, (void**)pncclCudaMalloc);

    return ncclSuccess;
}

inline ncclResult_t ncclGroupStart() {
    nccl_comm_struct_private.send_args.clear();
    nccl_comm_struct_private.recv_args.clear();
    nccl_comm_struct_private.src_buff_handle_list.clear();
    nccl_comm_struct_private.src_buff_list.clear();
    nccl_comm_struct_private.mpi_request_list.clear();
    nccl_comm_struct_private.in_group = true;
    return ncclSuccess;
}



inline ncclResult_t ncclGroupEnd() {

    if (!nccl_comm_struct_private.in_group) {
        return ncclInvalidUsage;
    }

    nccl_comm_struct_private.mpi_request_list.resize(nccl_comm_struct_private.send_args.size() + nccl_comm_struct_private.recv_args.size());
    int mpi_request_idx = 0;

    /* get memhandle and send it to peer */
    for (size_t i = 0; i < nccl_comm_struct_private.send_args.size(); ++i) {
        // fprintf(stderr, "[debug] nccl_comm_struct_private.send_args[i].buff=%p\n", nccl_comm_struct_private.send_args[i].buff);
        pnccl_handle_offset handle_offset;
        void* const buffer = pncclGetClosestPointer(nccl_comm_struct_private.send_args[i].buff, &handle_offset.offset);
        // fprintf(stderr, "[debug] pncclGetClosestPointer=%p\n", buffer);
        CHECK_CUDA(cudaIpcGetMemHandle, &handle_offset.handle, buffer);
        CHECK_MPI(MPI_Isend, &handle_offset, sizeof(pnccl_handle_offset), MPI_BYTE, nccl_comm_struct_private.send_args[i].peer, 0, MPI_COMM_WORLD, &nccl_comm_struct_private.mpi_request_list[mpi_request_idx]);
        mpi_request_idx++;
    }

    /* recv memhandle from peer */
    nccl_comm_struct_private.src_buff_handle_list.resize(nccl_comm_struct_private.recv_args.size());
    for (size_t i = 0; i < nccl_comm_struct_private.send_args.size(); ++i) {
        CHECK_MPI(MPI_Irecv, &nccl_comm_struct_private.src_buff_handle_list[i], sizeof(pnccl_handle_offset), MPI_BYTE, nccl_comm_struct_private.recv_args[i].peer, 0, MPI_COMM_WORLD, &nccl_comm_struct_private.mpi_request_list[mpi_request_idx]);
        mpi_request_idx++;
    }

    /* wait for all send/recv requests to finish */
    CHECK_MPI(MPI_Waitall, nccl_comm_struct_private.mpi_request_list.size(), nccl_comm_struct_private.mpi_request_list.data(), MPI_STATUSES_IGNORE);

    /* open memhandle and copy data */
    nccl_comm_struct_private.src_buff_list.resize(nccl_comm_struct_private.src_buff_handle_list.size());
    for (size_t i = 0; i < nccl_comm_struct_private.src_buff_handle_list.size(); ++i) {
        cudaIpcOpenMemHandle(&nccl_comm_struct_private.src_buff_list[i], nccl_comm_struct_private.src_buff_handle_list[i].handle, cudaIpcMemLazyEnablePeerAccess);
        CHECK_CUDA(
            cudaMemcpyAsync,
            nccl_comm_struct_private.recv_args[i].buff,
            /* nccl_comm_struct_private.src_buff_list[i] */
            (void*)((uint64_t)nccl_comm_struct_private.src_buff_list[i] + nccl_comm_struct_private.src_buff_handle_list[i].offset),
            nccl_comm_struct_private.recv_args[i].count * sizeof_ncclDataType_t(nccl_comm_struct_private.recv_args[i].datatype),
            cudaMemcpyDeviceToDevice,
            nccl_comm_struct_private.recv_args[i].stream
        );
    }

    /* synchronize streams */
    CHECK_CUDA(cudaStreamSynchronize, nccl_comm_struct_private.recv_args[0].stream);
    for (size_t i = 1; i < nccl_comm_struct_private.send_args.size(); ++i) {
        if (nccl_comm_struct_private.recv_args[i].stream != nccl_comm_struct_private.recv_args[i - 1].stream) {
            CHECK_CUDA(cudaStreamSynchronize, nccl_comm_struct_private.recv_args[i].stream);
        }
    }

    /* close memhandle */
    for (size_t i = 0; i < nccl_comm_struct_private.src_buff_list.size(); ++i) {
        CHECK_CUDA(cudaIpcCloseMemHandle, nccl_comm_struct_private.src_buff_list[i]);
    }

    /* free send/recv args */

    /* synchronize */
    CHECK_MPI(MPI_Barrier, MPI_COMM_WORLD);

    nccl_comm_struct_private.in_group = false;
    return ncclSuccess;
}

inline ncclResult_t ncclSend(void* sendbuff, uint64_t count, int datatype, int peer, ncclComm_t comm, cudaStream_t stream) {
    if (comm->in_group) {
        comm->send_args.push_back({sendbuff, count, datatype, peer, stream});
    } else {
        cudaIpcMemHandle_t handle;
        cudaIpcGetMemHandle(&handle, sendbuff);
        MPI_Send(&handle, sizeof(handle), MPI_BYTE, peer, 0, MPI_COMM_WORLD);
    }
    return ncclSuccess;
}

inline ncclResult_t ncclRecv(void* recvbuff, uint64_t count, int datatype, int peer, ncclComm_t comm, cudaStream_t stream) {
    if (comm->in_group) {
        comm->recv_args.push_back({recvbuff, count, datatype, peer, stream});
    } else {
        cudaIpcMemHandle_t handle;
        CHECK_MPI(MPI_Recv, &handle, sizeof(handle), MPI_BYTE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        void* sendbuff;
        cudaIpcOpenMemHandle(&sendbuff, handle, cudaIpcMemLazyEnablePeerAccess);
        CHECK_CUDA(cudaMemcpy, recvbuff, sendbuff, count * sizeof_ncclDataType_t(datatype), cudaMemcpyDeviceToDevice);
        CHECK_CUDA(cudaIpcCloseMemHandle, sendbuff);
    }
    return ncclSuccess;
}





#endif /* NCCL_H_ */