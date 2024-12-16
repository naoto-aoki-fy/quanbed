#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <stdint.h>

#include <stdexcept>
#include <string>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <random>
#include <utility>
#include <unordered_set>

#include <mpi.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cuda/std/complex>
#include <nccl.h>

#include "pipe3.hpp"

#define SQRT2 (1.41421356237309504880168872420969807856967187537694)
#define INV_SQRT2 (1.0/SQRT2)

int log2_int(int arg) {
    if(arg<=0) return -1;
    int value = 0;
    while(arg>1) {
        value += 1;
        arg = arg >> 1;
    }
    return value;
}

typedef double my_float_t;
typedef cuda::std::complex<my_float_t> my_complex_t;

// 任意のCUDA API関数とその引数を受け取る
template <typename Func, typename... Args>
void check_cuda(char const* const filename_abs, int const lineno, char const* const funcname, Func func, Args&&... args)
{
    char const* const strrchr_result = strrchr(filename_abs, '/');
    char const* const filename = strrchr_result? strrchr_result + 1 : filename_abs;
    // 引数を文字列化するためのostringstream
    std::ostringstream oss;
    ((oss << args << ", "), ...);  // C++17の折り返し式を使って引数を順番に追加
    std::string args_str = oss.str();
    if (!args_str.empty()) {
        args_str.pop_back();  // 最後の", "を削除
        args_str.pop_back();
    }

    // 実際にCUDA関数を呼び出し
    cudaError_t err = func(std::forward<Args>(args)...);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "[debug] %s:%d call:%s args:%s error:%s\n", filename, lineno, funcname, args_str.c_str(), cudaGetErrorString(err));
        exit(1);
    }
}

// マクロで簡単に呼び出せるようにラップ
#define CHECK_CUDA(func, ...) check_cuda(__FILE__, __LINE__, #func, func, __VA_ARGS__)

#define CASE_RETURN(code) case code: return #code

static const char *curandGetErrorString(curandStatus_t error) {
    switch (error) {
        CASE_RETURN(CURAND_STATUS_SUCCESS);
        CASE_RETURN(CURAND_STATUS_VERSION_MISMATCH);
        CASE_RETURN(CURAND_STATUS_NOT_INITIALIZED);
        CASE_RETURN(CURAND_STATUS_ALLOCATION_FAILED);
        CASE_RETURN(CURAND_STATUS_TYPE_ERROR);
        CASE_RETURN(CURAND_STATUS_OUT_OF_RANGE);
        CASE_RETURN(CURAND_STATUS_LENGTH_NOT_MULTIPLE);
        CASE_RETURN(CURAND_STATUS_DOUBLE_PRECISION_REQUIRED);
        CASE_RETURN(CURAND_STATUS_LAUNCH_FAILURE);
        CASE_RETURN(CURAND_STATUS_PREEXISTING_FAILURE);
        CASE_RETURN(CURAND_STATUS_INITIALIZATION_FAILED);
        CASE_RETURN(CURAND_STATUS_ARCH_MISMATCH);
        CASE_RETURN(CURAND_STATUS_INTERNAL_ERROR);
    }
    return "<unknown>";
}

template <typename Func, typename... Args>
void check_curand(char const* const filename_abs, int const lineno, char const* const funcname, Func func, Args&&... args)
{
    char const* const strrchr_result = strrchr(filename_abs, '/');
    char const* const filename = strrchr_result? strrchr_result + 1 : filename_abs;
    std::ostringstream oss;
    ((oss << args << ", "), ...);
    std::string args_str = oss.str();
    if (!args_str.empty()) {
        args_str.pop_back();
        args_str.pop_back();
    }

    curandStatus_t err = func(std::forward<Args>(args)...);
    if (err != CURAND_STATUS_SUCCESS)
    {
        fprintf(stderr, "[debug] %s:%d call:%s args:%s error:%s\n", filename, lineno, funcname, args_str.c_str(), curandGetErrorString(err));
        exit(1);
    }
}

#define CHECK_CURAND(func, ...) check_curand(__FILE__, __LINE__, #func, func, __VA_ARGS__)


template <typename Func, typename... Args>
void check_nccl(char const* const filename_abs, int const lineno, char const* const funcname, Func func, Args&&... args)
{
    char const* const strrchr_result = strrchr(filename_abs, '/');
    char const* const filename = strrchr_result? strrchr_result + 1 : filename_abs;

    ncclResult_t err = func(std::forward<Args>(args)...);
    if (err != ncclSuccess)
    {
        fprintf(stderr, "[debug] %s:%d call:%s error:%s\n", filename, lineno, funcname, ncclGetErrorString(err));
        exit(1);
    }
}

#define CHECK_NCCL(func, ...) check_nccl(__FILE__, __LINE__, #func, func, __VA_ARGS__)

// 可変長引数を取る関数ポインタをラップするテンプレート
template <typename Func, typename... Args>
class Defer {
public:
    // デフォルトコンストラクタ
    Defer() : valid_(false) {}

    // コンストラクタで関数ポインタと引数を受け取る
    Defer(Func func, Args... args)
        : func_(func), args_(args...), valid_(true) {}

    // ムーブ代入演算子
    Defer& operator=(Defer&& other) noexcept {
        if (this != &other) {
            func_ = other.func_;
            args_ = other.args_;
            valid_ = other.valid_;
            other.valid_ = false;
        }
        return *this;
    }

    // ムーブコンストラクタ
    Defer(Defer&& other) noexcept
        : func_(other.func_), args_(other.args_), valid_(other.valid_) {
        other.valid_ = false;
    }

    // デストラクタで関数ポインタを呼び出す
    ~Defer() {
        if (valid_) {
            call(std::index_sequence_for<Args...>{});
        }
    }

private:
    // 関数ポインタ
    Func func_;
    // 関数の引数（可変長引数をタプルで保持）
    std::tuple<Args...> args_;
    bool valid_;

    // 引数を展開して関数を呼び出す
    template <std::size_t... I>
    void call(std::index_sequence<I...>) {
        func_(std::get<I>(args_)...);
    }
};

__global__ void norm_sum_reduce_kernel(my_complex_t const* const input_global, my_float_t* const output_global)
{
    extern __shared__ my_float_t sum_shared[];
    int64_t const idx =  blockDim.x * blockIdx.x + threadIdx.x;
    sum_shared[threadIdx.x] = cuda::std::norm(input_global[idx]);

    my_float_t sum_cached;
    sum_cached = sum_shared[threadIdx.x];
    for(int active_threads = blockDim.x; active_threads > 1;) {
        int const half_active_threads = active_threads >> 1;
        active_threads = (active_threads + 1) >> 1;
        __syncthreads();
        if(threadIdx.x < half_active_threads){
            sum_cached += sum_shared[threadIdx.x + active_threads];
            sum_shared[threadIdx.x] = sum_cached;
        }
    }
    if (threadIdx.x == 0) {
        output_global[blockIdx.x] = sum_shared[0];
    }
}

__global__ void sum_reduce_kernel(my_float_t const* const input_global, my_float_t* const output_global)
{
    extern __shared__ my_float_t sum_shared[];
    int64_t const idx =  blockDim.x * blockIdx.x + threadIdx.x;
    sum_shared[threadIdx.x] = input_global[idx];

    my_float_t sum_cached;
    sum_cached = sum_shared[threadIdx.x];
    for(int active_threads = blockDim.x; active_threads > 1;) {
        int const half_active_threads = active_threads >> 1;
        active_threads = (active_threads + 1) >> 1;
        __syncthreads();
        if(threadIdx.x < half_active_threads){
            sum_cached += sum_shared[threadIdx.x + active_threads];
            sum_shared[threadIdx.x] = sum_cached;
        }
    }
    if (threadIdx.x == 0) {
        output_global[blockIdx.x] = sum_shared[0];
    }
}

__global__ void normalize_kernel(my_float_t* const data_global, my_float_t const factor)
{
    int64_t const idx = blockDim.x * blockIdx.x + threadIdx.x;
    data_global[idx] *= factor;
}

class hadamard { public:
    static __device__ __host__ void apply(int64_t const thread_num, int64_t const num_qubits, int64_t const target_qubit_num, my_complex_t* const state_data_device) {

        uint64_t const lower_mask = (((uint64_t)1)<<target_qubit_num) - (uint64_t)1;

        int64_t const index_state_lower = thread_num & lower_mask;
        int64_t const index_state_higher = (thread_num & ~lower_mask) << ((int64_t)1);

        int64_t const index_state_0 = index_state_lower | index_state_higher;
        int64_t const index_state_1 = index_state_0 | (((int64_t)1)<<target_qubit_num);

        my_complex_t const amp_state_0 = state_data_device[index_state_0];
        my_complex_t const amp_state_1 = state_data_device[index_state_1];

        state_data_device[index_state_0] = (amp_state_0 + amp_state_1) * INV_SQRT2;
        state_data_device[index_state_0] = (amp_state_0 - amp_state_1) * INV_SQRT2;

    }

};

template<class Gate>
__global__ void cuda_gate(int64_t const num_qubits, int64_t const target_qubit_num, my_complex_t* const state_data_device) {
    int64_t const thread_num = threadIdx.x + blockIdx.x * blockDim.x;
    Gate::apply(thread_num, num_qubits, target_qubit_num, state_data_device);
}

int main(int argc, char** argv) {

    float elapsed_ms, elapsed_ms_2;

    setvbuf(stdout, NULL, _IOLBF, 1024 * 512);

    MPI_Init(&argc, &argv);

    int num_procs, proc_num;

    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_num);

    if (proc_num==0) {
        fprintf(stderr, "[info] num_procs=%d\n", num_procs);
    }

    int const gpu_id = proc_num;
    // int const gpu_id = 0;
    CHECK_CUDA(cudaSetDevice, gpu_id);

    ncclUniqueId nccl_id;
    if (proc_num == 0) {
        CHECK_NCCL(ncclGetUniqueId, &nccl_id);
    }

    ncclComm_t nccl_comm;
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
    int nccl_rank = proc_num;
    CHECK_NCCL(ncclCommInitRank, &nccl_comm, num_procs, nccl_id, nccl_rank);

    int const num_qubits = 14;
    if (proc_num == 0) { fprintf(stderr, "[info] num_qubits=%d\n", num_qubits); }

    uint64_t const swap_buffer_length = UINT64_C(1) << 27;

    std::vector<int> perm_p2l(num_qubits);
    std::vector<int> perm_l2p(num_qubits);

    for(int qubit_num=0; qubit_num<num_qubits; qubit_num++) {
        perm_p2l[qubit_num] = qubit_num;
        perm_l2p[qubit_num] = qubit_num;
    }

    int const num_samples = 16;
    int const rng_seed = 12345;

    int const log_num_procs = log2_int(num_procs);

    int const log_block_size = 8;
    int const target_qubit_num_begin = 0;
    int const target_qubit_num_end = num_qubits;

    if (proc_num == 0) { fprintf(stderr, "[info] log_block_size=%d\n", log_block_size); }

    cudaStream_t stream;
    cudaEvent_t event_1;
    cudaEvent_t event_2;

    CHECK_CUDA(cudaStreamCreate, &stream);
    decltype(Defer(cudaStreamDestroy, stream)) defer_destroy_stream(cudaStreamDestroy, stream);

    CHECK_CUDA(cudaEventCreateWithFlags, &event_1, cudaEventDefault);
    decltype(Defer(cudaEventDestroy, event_1)) defer_destroy_event_1(cudaEventDestroy, event_1);

    CHECK_CUDA(cudaEventCreateWithFlags, &event_2, cudaEventDefault);
    decltype(Defer(cudaEventDestroy, event_2)) defer_destroy_event_2(cudaEventDestroy, event_2);

    int64_t const num_states = ((int64_t)1) << ((int64_t)num_qubits);

    int const num_qubits_local = num_qubits - log_num_procs;
    int64_t const num_states_local = ((int64_t)1) << ((int64_t)num_qubits_local);
    int const block_size = 1 << log_block_size;
    int64_t const num_blocks = ((int64_t)1) << ((int64_t)(num_qubits_local - 1 - log_block_size));

    if (proc_num == 0) { fprintf(stderr, "[info] malloc device memory\n"); }

    my_complex_t* state_data_device;
    CHECK_CUDA(cudaMalloc<void>, (void**)&state_data_device, num_states_local * sizeof(*state_data_device));
    decltype(Defer(cudaFree, (void*)0)) defer_free_state_data(cudaFree, (void*)state_data_device);

    my_complex_t* swap_buffer;
    CHECK_CUDA(cudaMalloc<void>, (void**)&swap_buffer, swap_buffer_length * sizeof(my_complex_t));
    decltype(Defer(cudaFree, (void*)0)) defer_free_swap_buffer(cudaFree, (void*)swap_buffer);


    my_float_t* norm_sum_device;
    CHECK_CUDA(cudaMalloc<void>, (void**)&norm_sum_device, (num_states_local>>log_block_size) * sizeof(my_float_t));
    decltype(Defer(cudaFree, (void*)0)) defer_free_norm_sum_device(cudaFree, (void*)norm_sum_device);

    if (proc_num == 0) { fprintf(stderr, "[info] generating random state\n"); }
    curandGenerator_t rng_device;

    CHECK_CURAND(curandCreateGenerator, &rng_device, CURAND_RNG_PSEUDO_DEFAULT);
    CHECK_CURAND(curandSetStream, rng_device, stream);
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed, rng_device, rng_seed + proc_num);

    if (proc_num == 0) { fprintf(stderr, "[info] gpu reduce\n"); } 
    CHECK_CUDA(cudaEventRecord, event_1, stream);

    CHECK_CURAND(curandGenerateNormalDouble, rng_device, (my_float_t*)(void*)state_data_device, num_states_local * 2, 0.0, 1.0);

    // CHECK_CURAND(curandGenerateNormalDouble, rng_device, (my_float_t*)(void*)state_data_device, num_states_local, 0.0, 1.0);

    // curandGenerator_t rng_device_2;

    // CHECK_CURAND(curandCreateGenerator, &rng_device_2, CURAND_RNG_PSEUDO_DEFAULT);
    // CHECK_CURAND(curandSetStream, rng_device_2, stream);
    // CHECK_CURAND(curandSetPseudoRandomGeneratorSeed, rng_device_2, rng_seed + 1);

    // CHECK_CURAND(curandGenerateNormalDouble, rng_device_2, &((my_float_t*)(void*)state_data_device)[num_states_local], num_states_local, 0.0, 1.0);

    {
        int64_t data_length = num_states_local;
        int64_t num_blocks_reduce;
        int block_size_reduce;

        if (data_length > block_size) {
            block_size_reduce = block_size;
            num_blocks_reduce = data_length >> log_block_size;
        } else {
            block_size_reduce = data_length;
            num_blocks_reduce = 1;
        }

        norm_sum_reduce_kernel<<<num_blocks_reduce, block_size_reduce, sizeof(my_float_t) * block_size_reduce, stream>>>(state_data_device, norm_sum_device);

        data_length = num_blocks_reduce;

        while (data_length > 1) {
            if (data_length > block_size) {
                block_size_reduce = block_size;
                num_blocks_reduce = data_length >> log_block_size;
            } else {
                block_size_reduce = data_length;
                num_blocks_reduce = 1;
            }

            sum_reduce_kernel<<<num_blocks_reduce, block_size_reduce, sizeof(my_float_t) * block_size_reduce, stream>>>(norm_sum_device, norm_sum_device);

            data_length = num_blocks_reduce;
        }
    }

    // fprintf(stderr, "[debug] line=%d\n", __LINE__);

    my_float_t norm_sum_local;
    CHECK_CUDA(cudaMemcpyAsync, &norm_sum_local, norm_sum_device, sizeof(my_float_t), cudaMemcpyDeviceToHost, stream);

    // fprintf(stderr, "[debug] line=%d\n", __LINE__);

    CHECK_CUDA(cudaFree, (void*)norm_sum_device);

    // fprintf(stderr, "[debug] line=%d\n", __LINE__);

    CHECK_CUDA(cudaStreamSynchronize, stream);

    // fprintf(stderr, "[debug] line=%d\n", __LINE__);

    my_float_t norm_sum_global;
    MPI_Allreduce(&norm_sum_local, &norm_sum_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (proc_num == 0) { fprintf(stderr, "[info] norm_sum_global=%lf\n", norm_sum_global); }

    if (proc_num == 0) { fprintf(stderr, "[info] normalize\n"); }

    my_float_t const normalize_factor = 1.0 / sqrt(norm_sum_global);
    fprintf(stderr, "[debug] normalize_factor=%.20e\n", normalize_factor);

    // fprintf(stderr, "[debug] line=%d\n", __LINE__);

    normalize_kernel<<<1ULL<<(num_qubits_local+1-log_block_size), block_size, 0, stream>>>((my_float_t*)(void*)state_data_device, normalize_factor);

    // fprintf(stderr, "[debug] line=%d\n", __LINE__);

    CHECK_CUDA(cudaEventRecord, event_2, stream);

    // fprintf(stderr, "[debug] line=%d\n", __LINE__);

    CHECK_CUDA(cudaStreamSynchronize, stream);

    // fprintf(stderr, "[debug] line=%d\n", __LINE__);

    CHECK_CUDA(cudaEventElapsedTime, &elapsed_ms, event_1, event_2);

    // fprintf(stderr, "[debug] line=%d\n", __LINE__);

    MPI_Reduce(&elapsed_ms, &elapsed_ms_2, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    elapsed_ms = elapsed_ms_2;

    // fprintf(stderr, "[debug] line=%d\n", __LINE__);

    if(proc_num==0) {
        fprintf(stderr, "[info] rng elapsed=%lf\n", elapsed_ms * 1e-3);
    }

    if (proc_num == 0) { fprintf(stderr, "[info] normalize done\n"); }

    if (proc_num == 0) { fprintf(stderr, "[info] gpu_hadamard\n"); }

    MPI_Barrier(MPI_COMM_WORLD);

    for(int sample_num=0; sample_num < num_samples; ++sample_num) {

        CHECK_CUDA(cudaEventRecord, event_1, stream);

        for(int target_qubit_num_logical = target_qubit_num_begin; target_qubit_num_logical < target_qubit_num_end; target_qubit_num_logical++) {

            int target_qubit_num_physical = perm_l2p[target_qubit_num_logical];

            /* target qubits is global */
            if (target_qubit_num_physical >= num_qubits - log_num_procs) {
                int64_t const swap_width = UINT64_C(1)<<(target_qubit_num_physical - log_num_procs);
                int64_t const num_swap_areas = UINT64_C(1)<<(num_qubits - target_qubit_num_physical - 1);
                // int64_t const pow2_nt_lnp = UINT64_C(1)<<(target_qubit_num_physical - log_num_procs);
                for(int64_t swap_area_num = 0; swap_area_num < num_swap_areas; swap_area_num++) {
                    int const fp = (proc_num >> (log_num_procs - num_qubits + target_qubit_num_physical))&1;

                    // fprintf(stderr, "[debug] nccl_rank=%d fp=%d\n", nccl_rank, fp);

                    int64_t const swap_area_begin = (2*swap_area_num+((fp)?0:1)) * swap_width;
                    // int64_t const swap_area_end = swap_area_begin + swap_width;
                    int const peer_gpu_num = (proc_num + ((fp)?1:-1) * (UINT64_C(1)<< (target_qubit_num_physical + log_num_procs - num_qubits)) + num_procs) & (num_procs-1);

                    // fprintf(stderr, "[debug] nccl_rank=%d peer_gpu_num=%d\n", nccl_rank, peer_gpu_num);

                    int64_t swap_area_dispos = 0;
                    while (true) {
                        int64_t swap_length = swap_width - swap_area_dispos;
                        if (swap_length > swap_buffer_length) {
                            swap_length = swap_buffer_length;
                        }
                        int64_t const swap_area_pos = swap_area_begin + swap_area_dispos;

                        /* implement swap here*/
                        if (fp) {
                            CHECK_NCCL(ncclSend, (void*)&state_data_device[swap_area_pos], swap_length*2, ncclDouble, peer_gpu_num, nccl_comm, stream);
                            CHECK_NCCL(ncclRecv, (void*)swap_buffer, swap_length * 2, ncclDouble, peer_gpu_num, nccl_comm, stream);
                        } else {
                            CHECK_NCCL(ncclRecv, (void*)swap_buffer, swap_length * 2, ncclDouble, peer_gpu_num, nccl_comm, stream);
                            CHECK_NCCL(ncclSend, (void*)&state_data_device[swap_area_pos], swap_length*2, ncclDouble, peer_gpu_num, nccl_comm, stream);
                        }
                        CHECK_CUDA(cudaMemcpyAsync, (void*)&state_data_device[swap_area_pos], (void*)swap_buffer, swap_length*sizeof(my_complex_t), cudaMemcpyDeviceToDevice, stream);

                        swap_area_dispos += swap_length;
                        if (swap_area_dispos==swap_width) {
                            break;
                        }
                    }
                }

                int const swap_qubit_num_logical = perm_p2l[target_qubit_num_physical - log_num_procs];

                // swap p2l
                perm_p2l[target_qubit_num_physical] = swap_qubit_num_logical;
                perm_p2l[target_qubit_num_physical - log_num_procs] = target_qubit_num_logical;

                // update l2p
                perm_l2p[target_qubit_num_logical] = target_qubit_num_physical - log_num_procs;
                perm_l2p[swap_qubit_num_logical] = target_qubit_num_physical;

                target_qubit_num_physical -= log_num_procs;
            }

            cuda_gate<hadamard><<<num_blocks, block_size, 0, stream>>>(num_qubits, target_qubit_num_physical, state_data_device);
        }

        CHECK_CUDA(cudaEventRecord, event_2, stream);

        CHECK_CUDA(cudaStreamSynchronize, stream);

        CHECK_CUDA(cudaEventElapsedTime, &elapsed_ms, event_1, event_2);
        MPI_Reduce(&elapsed_ms, &elapsed_ms_2, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
        elapsed_ms = elapsed_ms_2;
        if(proc_num==0) {
            fprintf(stderr, "[info] elapsed_gpu=%f\n", elapsed_ms * 1e-3);
            fprintf(stdout, "%lf\n", elapsed_ms);
        }

    }

    // uint64_t const cksumbuf_data_count = 1ULL < 24;
    // if(proc_num==0) {
    //     process cksumproc;
    //     char* const cksumproc_argv[] = {"openssl", "sha256", NULL};
    //     if (popen3(&cksumproc, cksumproc_argv, true, false, false) != 0) {
    //         fprintf(stderr, "[errpr] popen3 failed\n");
    //         exit(1);
    //     }

    //     void* cksumbuf = malloc(cksumbuf_data_count * sizeof(my_complex_t));
    //     decltype(Defer(free, (void*)0)) defer_free_cksum_buffer(free, (void*)cksumbuf);

    //     {
    //         int64_t data_pos = 0;
    //         while(true) {
    //             int64_t cksumbuf_copy_count = cksumbuf_data_count;
    //             if (cksumbuf_copy_count > num_states_local) { cksumbuf_copy_count = num_states_local; }
    //             CHECK_CUDA(cudaMemcpyAsync, cksumbuf, &state_data_device[data_pos], cksumbuf_copy_count * sizeof(my_complex_t), cudaMemcpyDeviceToHost, stream);
    //             CHECK_CUDA(cudaStreamSynchronize, stream);
    //             fwrite(cksumbuf, 1, cksumbuf_data_count, cksumproc.stdin);
    //             data_pos += cksumbuf_data_count;
    //             if(data_pos == num_states_local) { break; }
    //         }
    //     }
    //     for(int peer_proc_num=1; peer_proc_num<num_procs; peer_proc_num++) {
    //         int64_t data_pos = 0;
    //         while(true) {
    //             int64_t cksumbuf_copy_count = cksumbuf_data_count;
    //             if (cksumbuf_copy_count > num_states_local) { cksumbuf_copy_count = num_states_local; }
    //             // CHECK_NCCL(ncclRecv, cksumbuf, cksumbuf_copy_count * 2, ncclDouble, peer_proc_num, nccl_comm, stream);
    //             // CHECK_CUDA(cudaStreamSynchronize, stream);
    //             fwrite(cksumbuf, 1, cksumbuf_data_count, cksumproc.stdin);
    //             data_pos += cksumbuf_data_count;
    //             if(data_pos == num_states_local) { break; }
    //         }
    //     }
    //     fclose(cksumproc.stdin);

    //     int cksumproc_status;
    //     waitpid(cksumproc.pid, &cksumproc_status, 0);
    //     if (cksumproc_status!=0) {
    //         fprintf(stderr, "[warn] cksumproc_status=%d\n", cksumproc_status);
    //     }

    // } else {
    //     for(int proc_num = 1; proc_num < num_procs; proc_num++) {
    //         int64_t data_pos = 0;
    //         while(true) {
    //             int64_t cksumbuf_copy_count = cksumbuf_data_count;
    //             if (cksumbuf_copy_count > num_states_local) { cksumbuf_copy_count = num_states_local; }
    //             CHECK_NCCL(ncclSend, (void*)&state_data_device[data_pos], cksumbuf_copy_count * 2, ncclDouble, 0, nccl_comm, stream);
    //             CHECK_CUDA(cudaStreamSynchronize, stream);
    //             data_pos += cksumbuf_data_count;
    //             if(data_pos == num_states_local) { break; }
    //         }
    //     }
    // }

    if(proc_num==0) {
        fprintf(stderr, "[info] gathering state data\n");

        process cksumproc;
        char* const cksumproc_argv[] = {"openssl", "sha256", NULL};
        if (popen3(&cksumproc, cksumproc_argv, true, false, false) != 0) {
            fprintf(stderr, "[errpr] popen3 failed\n");
            exit(1);
        }

        my_complex_t* state_data_host = (my_complex_t*)malloc(num_states * sizeof(my_complex_t));
        decltype(Defer(free, (void*)0)) defer_free_state_data_host(free, (void*)state_data_host);

        CHECK_CUDA(cudaMemcpyAsync, state_data_host, state_data_device, num_states_local * sizeof(my_complex_t), cudaMemcpyDeviceToHost, stream);
        for(int peer_proc_num=1; peer_proc_num<num_procs; peer_proc_num++) {
            MPI_Status mpi_status;
            MPI_Recv(&state_data_host[peer_proc_num * num_states_local], num_states_local * 2, MPI_DOUBLE, peer_proc_num, 0, MPI_COMM_WORLD, &mpi_status);
        }
        CHECK_CUDA(cudaStreamSynchronize, stream);

        for(int64_t state_num_logical = 0; state_num_logical < num_states; state_num_logical++) {
            int64_t state_num_physical = 0;
            for(int qubit_num_logical = 0; qubit_num_logical < num_qubits; qubit_num_logical++) {
                int qubit_num_physical = perm_l2p[qubit_num_logical];
                state_num_physical = state_num_physical | (((state_num_logical >> qubit_num_logical) & 1) << qubit_num_physical);
            }
            fwrite(&state_data_host[state_num_physical], sizeof(my_complex_t), 1, cksumproc.stdin);
        }
        // for(int64_t state_num_logical = 0; state_num_logical < num_states; state_num_logical++) {
        //     int64_t state_num_physical = state_num_logical;
        //     fwrite(&state_data_host[state_num_physical], sizeof(my_complex_t), 1, cksumproc.stdin);
        // }
        // fprintf(stderr, "[debug] line=%d\n", __LINE__);

        fclose(cksumproc.stdin);

        // fprintf(stderr, "[debug] line=%d\n", __LINE__);

        int cksumproc_status;
        waitpid(cksumproc.pid, &cksumproc_status, 0);

        // fprintf(stderr, "[debug] line=%d\n", __LINE__);

        if (cksumproc_status!=0) {
            fprintf(stderr, "[warn] cksumproc_status=%d\n", cksumproc_status);
        }

        // fprintf(stderr, "[debug] line=%d\n", __LINE__);

    } else {
        // for(int proc_num = 1; proc_num < num_procs; proc_num++) {
        MPI_Send(state_data_device, num_states_local * 2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        // }
    }

    MPI_Finalize();

    return 0;

}