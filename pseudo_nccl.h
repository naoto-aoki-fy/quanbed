#pragma once
#ifndef NCCL_H_
#define NCCL_H_

#include <stdint.h>
#include <cuda_runtime.h>

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

} nccl_sendrecv_args_t;

struct ncclComm_struct {
    bool in_group;
    std::vector<nccl_sendrecv_args_t> send_args;
    std::vector<nccl_sendrecv_args_t> recv_args;
    // std::vector<cudaIpcMemHandle_t> send_buff_handle_list;
    std::vector<cudaIpcMemHandle_t> src_buff_handle_list;
    std::vector<void*> src_buff_list;
    std::vector<MPI_Request> mpi_request_list;
};

typedef ncclComm_struct* ncclComm_t;

ncclComm_struct nccl_comm_struct_private;

inline ncclResult_t ncclCommInitRank(ncclComm_t* comm, int ndev, ncclUniqueId nccl_id, int rank) {
    *comm = &nccl_comm_struct_private;
    nccl_comm_struct_private.in_group = false;
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
        cudaIpcMemHandle_t send_buff_handle;
        cudaIpcGetMemHandle(&send_buff_handle, nccl_comm_struct_private.send_args[i].buff);
        CHECK_MPI(MPI_Isend, &send_buff_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, nccl_comm_struct_private.send_args[i].peer, 0, MPI_COMM_WORLD, &nccl_comm_struct_private.mpi_request_list[mpi_request_idx]);
        mpi_request_idx++;
    }

    /* recv memhandle from peer */
    nccl_comm_struct_private.src_buff_handle_list.resize(nccl_comm_struct_private.recv_args.size());
    for (size_t i = 0; i < nccl_comm_struct_private.send_args.size(); ++i) {
        CHECK_MPI(MPI_Irecv, &nccl_comm_struct_private.src_buff_handle_list[i], sizeof(cudaIpcMemHandle_t), MPI_BYTE, nccl_comm_struct_private.recv_args[i].peer, 0, MPI_COMM_WORLD, &nccl_comm_struct_private.mpi_request_list[mpi_request_idx]);
        mpi_request_idx++;
    }

    /* wait for all send/recv requests to finish */
    CHECK_MPI(MPI_Waitall, nccl_comm_struct_private.mpi_request_list.size(), nccl_comm_struct_private.mpi_request_list.data(), MPI_STATUSES_IGNORE);

    /* open memhandle and copy data */
    nccl_comm_struct_private.src_buff_list.resize(nccl_comm_struct_private.src_buff_handle_list.size());
    for (size_t i = 0; i < nccl_comm_struct_private.src_buff_handle_list.size(); ++i) {
        cudaIpcOpenMemHandle(&nccl_comm_struct_private.src_buff_list[i], nccl_comm_struct_private.src_buff_handle_list[i], cudaIpcMemLazyEnablePeerAccess);
        CHECK_CUDA(cudaMemcpyAsync, nccl_comm_struct_private.recv_args[i].buff, nccl_comm_struct_private.src_buff_list[i], nccl_comm_struct_private.recv_args[i].count * sizeof_ncclDataType_t(nccl_comm_struct_private.recv_args[i].datatype), cudaMemcpyDeviceToDevice, nccl_comm_struct_private.recv_args[i].stream);
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