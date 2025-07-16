// nvcc -shared -Xcompiler -fPIC -std=c++17 -lmpi -lnccl -rdc=true -I./cdl86 ./mynccl.cpp ./cdl86/cdl.c -o mynccl.so

#include "mynccl.h"

#include <nccl.h>

#include <stdint.h>
#include <mpi.h>
#include <cuda_runtime.h>

#include "check_mpi.hpp"
#include "check_cuda.hpp"

#include <cdl.h>

#include "check_x.hpp"

typedef struct {
    cudaIpcMemHandle_t handle;
    uint64_t offset;
} myncclHandleOffset;

typedef struct {
    void* buff;
    uint64_t count;
    int datatype;
    int peer;
    cudaStream_t stream;
} myncclSendRecvArgs_t;

struct myncclCommStruct {
    bool in_group;
    std::vector<myncclSendRecvArgs_t> send_args;
    std::vector<myncclSendRecvArgs_t> recv_args;
    std::vector<myncclHandleOffset> src_buff_handle_list;
    std::vector<void*> src_buff_list;
    std::vector<MPI_Request> mpi_request_list;
    std::vector<uint64_t> pointer_list;

    struct cdl_jmp_patch jmpPatchCudaMalloc;
    cudaError_t (*origCudaMalloc)(void **, size_t);

    struct cdl_jmp_patch jmpPatchCudaSetDevice;
    cudaError_t (*origCudaSetDevice)(int);

    void* origNcclGetUniqueId;
    struct cdl_jmp_patch jmpPatchNcclGetUniqueId;

    void* origNcclCommInitRank;
    struct cdl_jmp_patch jmpPatchNcclCommInitRank;

    void* origNcclGroupStart;
    struct cdl_jmp_patch jmpPatchNcclGroupStart;

    void* origNcclGroupEnd;
    struct cdl_jmp_patch jmpPatchNcclGroupEnd;

    void* origNcclSend;
    struct cdl_jmp_patch jmpPatchNcclSend;

    void* origNcclRecv;
    struct cdl_jmp_patch jmpPatchNcclRecv;
};

// typedef myncclCommStruct* ncclComm_t;

static myncclCommStruct myncclCommStructPrivate;

static cudaError_t myncclCudaMalloc(void **devPtr, size_t size) {
    cudaError_t const ret = myncclCommStructPrivate.origCudaMalloc(devPtr, size);
    myncclCommStructPrivate.pointer_list.push_back((uint64_t)*devPtr);
    return ret;
}

static cudaError_t myncclCudaSetDevice(int device) {
    cudaError_t const ret = myncclCommStructPrivate.origCudaSetDevice(0);
    return ret;

}

static inline uint64_t myncclSizeofNcclDataType(int datatype) {
    switch (datatype) {
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
        default:
            throw datatype;
            return 0;
    }
    return 0;
}

static ncclResult_t myncclGetUniqueId(ncclUniqueId* nccl_id) {
    *(int*)nccl_id = 1;
    return ncclSuccess;
}

void* myncclGetClosestPointer(void* pointer_input, uint64_t* offset) {
    uint64_t num_ptrs = myncclCommStructPrivate.pointer_list.size();
    uint64_t distance_closest = (uint64_t)(-1);
    uint64_t pointer_closest = 0;
    for (uint64_t ptr_num = 0; ptr_num < num_ptrs; ptr_num++) {
        uint64_t pointer = myncclCommStructPrivate.pointer_list[ptr_num];
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

static ncclResult_t myncclCommInitRank(ncclComm_t* comm, int ndev, ncclUniqueId nccl_id, int rank) {
    *comm = (ncclComm_t)(void*)&myncclCommStructPrivate;
    myncclCommStructPrivate.in_group = false;
    return ncclSuccess;
}

static ncclResult_t myncclGroupStart() {
    myncclCommStructPrivate.send_args.clear();
    myncclCommStructPrivate.recv_args.clear();
    myncclCommStructPrivate.src_buff_handle_list.clear();
    myncclCommStructPrivate.src_buff_list.clear();
    myncclCommStructPrivate.mpi_request_list.clear();
    myncclCommStructPrivate.in_group = true;
    return ncclSuccess;
}

static ncclResult_t myncclGroupEnd() {

    if (!myncclCommStructPrivate.in_group) {
        return ncclInvalidUsage;
    }

    myncclCommStructPrivate.mpi_request_list.resize(myncclCommStructPrivate.send_args.size() + myncclCommStructPrivate.recv_args.size());
    int mpi_request_idx = 0;

    /* get memhandle and send it to peer */
    for (size_t i = 0; i < myncclCommStructPrivate.send_args.size(); ++i) {
        // fprintf(stderr, "[debug] myncclCommStructPrivate.send_args[i].buff=%p\n", myncclCommStructPrivate.send_args[i].buff);
        myncclHandleOffset handle_offset;
        void* const buffer = myncclGetClosestPointer(myncclCommStructPrivate.send_args[i].buff, &handle_offset.offset);
        // fprintf(stderr, "[debug] myncclGetClosestPointer=%p offset=%p\n", buffer, handle_offset.offset);
        ATLC_CHECK_CUDA(cudaIpcGetMemHandle, &handle_offset.handle, buffer);
        ATLC_CHECK_MPI(MPI_Isend, &handle_offset, sizeof(myncclHandleOffset), MPI_BYTE, myncclCommStructPrivate.send_args[i].peer, 0, MPI_COMM_WORLD, &myncclCommStructPrivate.mpi_request_list[mpi_request_idx]);
        mpi_request_idx++;
    }

    /* recv memhandle from peer */
    myncclCommStructPrivate.src_buff_handle_list.resize(myncclCommStructPrivate.recv_args.size());
    for (size_t i = 0; i < myncclCommStructPrivate.send_args.size(); ++i) {
        ATLC_CHECK_MPI(MPI_Irecv, &myncclCommStructPrivate.src_buff_handle_list[i], sizeof(myncclHandleOffset), MPI_BYTE, myncclCommStructPrivate.recv_args[i].peer, 0, MPI_COMM_WORLD, &myncclCommStructPrivate.mpi_request_list[mpi_request_idx]);
        mpi_request_idx++;
    }

    /* wait for all send/recv requests to finish */
    ATLC_CHECK_MPI(MPI_Waitall, myncclCommStructPrivate.mpi_request_list.size(), myncclCommStructPrivate.mpi_request_list.data(), MPI_STATUSES_IGNORE);

    /* open memhandle and copy data */
    myncclCommStructPrivate.src_buff_list.resize(myncclCommStructPrivate.src_buff_handle_list.size());
    for (size_t i = 0; i < myncclCommStructPrivate.src_buff_handle_list.size(); ++i) {
        cudaIpcOpenMemHandle(&myncclCommStructPrivate.src_buff_list[i], myncclCommStructPrivate.src_buff_handle_list[i].handle, cudaIpcMemLazyEnablePeerAccess);
        ATLC_CHECK_CUDA(
            cudaMemcpyAsync,
            myncclCommStructPrivate.recv_args[i].buff,
            /* myncclCommStructPrivate.src_buff_list[i] */
            (void*)((uint64_t)myncclCommStructPrivate.src_buff_list[i] + myncclCommStructPrivate.src_buff_handle_list[i].offset),
            myncclCommStructPrivate.recv_args[i].count * myncclSizeofNcclDataType(myncclCommStructPrivate.recv_args[i].datatype),
            cudaMemcpyDeviceToDevice,
            myncclCommStructPrivate.recv_args[i].stream
        );
    }

    /* synchronize streams */
    ATLC_CHECK_CUDA(cudaStreamSynchronize, myncclCommStructPrivate.recv_args[0].stream);
    for (size_t i = 1; i < myncclCommStructPrivate.send_args.size(); ++i) {
        if (myncclCommStructPrivate.recv_args[i].stream != myncclCommStructPrivate.recv_args[i - 1].stream) {
            ATLC_CHECK_CUDA(cudaStreamSynchronize, myncclCommStructPrivate.recv_args[i].stream);
        }
    }

    /* close memhandle */
    for (size_t i = 0; i < myncclCommStructPrivate.src_buff_list.size(); ++i) {
        ATLC_CHECK_CUDA(cudaIpcCloseMemHandle, myncclCommStructPrivate.src_buff_list[i]);
    }

    /* free send/recv args */

    /* synchronize */
    ATLC_CHECK_MPI(MPI_Barrier, MPI_COMM_WORLD);

    myncclCommStructPrivate.in_group = false;
    return ncclSuccess;
}

static ncclResult_t myncclSend(void* sendbuff, uint64_t count, int datatype, int peer, myncclCommStruct* comm, cudaStream_t stream) {
    if (comm->in_group) {
        comm->send_args.push_back({sendbuff, count, datatype, peer, stream});
    } else {
        // not yet tested
        myncclHandleOffset handle_offset;
        void* const buffer = myncclGetClosestPointer(sendbuff, &handle_offset.offset);
        cudaIpcMemHandle_t handle;
        cudaIpcGetMemHandle(&handle, buffer);
        MPI_Send(&handle_offset, sizeof(myncclHandleOffset), MPI_BYTE, peer, 0, MPI_COMM_WORLD);
    }
    return ncclSuccess;
}

static ncclResult_t myncclRecv(void* recvbuff, uint64_t count, int datatype, int peer, myncclCommStruct* comm, cudaStream_t stream) {
    if (comm->in_group) {
        comm->recv_args.push_back({recvbuff, count, datatype, peer, stream});
    } else {
        // not yet tested
        myncclHandleOffset handle_offset;
        ATLC_CHECK_MPI(MPI_Recv, &handle_offset, sizeof(myncclHandleOffset), MPI_BYTE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        void* sendbuff;
        cudaIpcOpenMemHandle(&sendbuff, handle_offset.handle, cudaIpcMemLazyEnablePeerAccess);
        ATLC_CHECK_CUDA(cudaMemcpy, recvbuff, (void*)((uint64_t)sendbuff + handle_offset.offset), count * myncclSizeofNcclDataType(datatype), cudaMemcpyDeviceToDevice);
        ATLC_CHECK_CUDA(cudaIpcCloseMemHandle, sendbuff);
    }
    return ncclSuccess;
}

// __attribute__((constructor))
// static void hoge() {
//     fprintf(stderr, "debug: hello\n");
// }

extern "C"
void myncclPatch() {

    // fprintf(stderr, "debug: ::cudaMalloc=%p\n", (cudaError_t (*)(void **, size_t))::cudaMalloc);
    myncclCommStructPrivate.origCudaMalloc = (cudaError_t (*)(void **, size_t))::cudaMalloc;
    myncclCommStructPrivate.jmpPatchCudaMalloc = cdl_jmp_attach((void**)&myncclCommStructPrivate.origCudaMalloc, (void**)myncclCudaMalloc);

    myncclCommStructPrivate.origCudaSetDevice = (cudaError_t (*)(int))::cudaSetDevice;
    myncclCommStructPrivate.jmpPatchCudaSetDevice = cdl_jmp_attach((void**)&myncclCommStructPrivate.origCudaSetDevice, (void**)myncclCudaSetDevice);

    myncclCommStructPrivate.origNcclGetUniqueId = (void*)ncclGetUniqueId;
    myncclCommStructPrivate.jmpPatchNcclGetUniqueId = cdl_jmp_attach((void**)&myncclCommStructPrivate.origNcclGetUniqueId, (void**)myncclGetUniqueId);

    myncclCommStructPrivate.origNcclCommInitRank = (void*)ncclCommInitRank;
    myncclCommStructPrivate.jmpPatchNcclCommInitRank = cdl_jmp_attach((void**)&myncclCommStructPrivate.origNcclCommInitRank, (void**)myncclCommInitRank);

    myncclCommStructPrivate.origNcclGroupStart = (void*)ncclGroupStart;
    myncclCommStructPrivate.jmpPatchNcclGroupStart = cdl_jmp_attach((void**)&myncclCommStructPrivate.origNcclGroupStart, (void**)myncclGroupStart);

    myncclCommStructPrivate.origNcclGroupEnd = (void*)ncclGroupEnd;
    myncclCommStructPrivate.jmpPatchNcclGroupEnd = cdl_jmp_attach((void**)&myncclCommStructPrivate.origNcclGroupEnd, (void**)myncclGroupEnd);

    myncclCommStructPrivate.origNcclSend = (void*)ncclSend;
    myncclCommStructPrivate.jmpPatchNcclSend = cdl_jmp_attach((void**)&myncclCommStructPrivate.origNcclSend, (void**)myncclSend);

    myncclCommStructPrivate.origNcclRecv = (void*)ncclRecv;
    myncclCommStructPrivate.jmpPatchNcclRecv = cdl_jmp_attach((void**)&myncclCommStructPrivate.origNcclRecv, (void**)myncclRecv);
}