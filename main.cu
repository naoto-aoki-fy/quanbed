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

#include <cuda_runtime.h>
#include <curand.h>
#include <cuda/std/complex>

#define SQRT2 (1.41421356237309504880168872420969807856967187537694)
#define INV_SQRT2 (1.0/SQRT2)

const int max_num_gpus = 8;

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

__constant__ my_complex_t* state_data_device_list_constmem[max_num_gpus];

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
    int const idx =  blockDim.x * blockIdx.x + threadIdx.x;
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
    int const idx =  blockDim.x * blockIdx.x + threadIdx.x;
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
    int const idx =  blockDim.x * blockIdx.x + threadIdx.x;
    data_global[idx] *= factor;
}

class hadamard { public:
    static __device__ __host__ void apply(int const num_split_areas, int const log_num_split_areas, int64_t const thread_num, int64_t const num_qubits, int64_t const target_qubit_num, my_complex_t** const state_data) {

        uint64_t const lower_mask = (((uint64_t)1)<<target_qubit_num) - (uint64_t)1;
        uint64_t const split_mask = (((uint64_t)1)<<((uint64_t)(num_qubits - log_num_split_areas))) - (uint64_t)1;

        int64_t const index_state_lower = thread_num & lower_mask;
        int64_t const index_state_higher = (thread_num & ~lower_mask) << ((int64_t)1);

        int64_t const index_state_0 = index_state_lower | index_state_higher;
        int64_t const index_state_1 = index_state_0 | (((int64_t)1)<<target_qubit_num);

        int64_t const index_state_0_split_num = index_state_0 >> (num_qubits - log_num_split_areas);
        int64_t const index_state_0_split_address = index_state_0 & split_mask;

        int64_t const index_state_1_split_num = index_state_1 >> (num_qubits - log_num_split_areas);
        int64_t const index_state_1_split_address = index_state_1 & split_mask;

        my_complex_t const amp_state_0 = state_data[index_state_0_split_num][index_state_0_split_address];
        my_complex_t const amp_state_1 = state_data[index_state_1_split_num][index_state_1_split_address];

        state_data[index_state_0_split_num][index_state_0_split_address] = (amp_state_0 + amp_state_1) * INV_SQRT2;
        state_data[index_state_1_split_num][index_state_1_split_address] = (amp_state_0 - amp_state_1) * INV_SQRT2;

    }

};

template<class Gate>
__global__ void cuda_gate(int const num_split_areas, int const log_num_split_areas, int64_t const split_num, int64_t const num_qubits, int64_t const target_qubit_num) {
    int64_t const num_qubits_local = num_qubits - log_num_split_areas;
    int64_t const num_threads_local = ((int64_t)1) << (num_qubits_local-1);

    int64_t const thread_num = threadIdx.x + blockIdx.x * blockDim.x + num_threads_local * split_num;
    Gate::apply(num_split_areas, log_num_split_areas, thread_num, num_qubits, target_qubit_num, state_data_device_list_constmem);
}

int main() {

    setvbuf(stdout, NULL, _IOLBF, 1024 * 512);

    int const num_qubits = 31;
    fprintf(stderr, "[info] num_qubits=%d\n", num_qubits);

    std::vector<int> perm_p2l(num_qubits);
    std::vector<int> perm_l2p(num_qubits);

    for(int qubit_num=0; qubit_num<num_qubits; qubit_num++) {
        perm_p2l[qubit_num] = qubit_num;
        perm_l2p[qubit_num] = qubit_num;
    }

    int const num_samples = 128;
    int const rng_seed = 12345;
    // std::vector<int> gpu_list{0, 1, 2, 3, 4, 5, 6, 7};
    // std::vector<int> gpu_list{0, 1, 2, 3};
    // std::vector<int> gpu_list{3, 2, 1, 0};
    // std::vector<int> gpu_list{0, 1};
    std::vector<int> gpu_list{2, 3};
    // std::vector<int> gpu_list{1};
    int const num_gpus = gpu_list.size();
    int const log_num_gpus = log2_int(num_gpus);

    int const log_block_size = 8;
    int const target_qubit_num_begin = 0;
    int const target_qubit_num_end = num_qubits;

    fprintf(stderr, "[info] num_gpus=%d (", num_gpus);
    for(int i=0; i<num_gpus; i++) {
        fprintf(stderr, "%d, ", gpu_list[i]);
    }
    fprintf(stderr, ")\n");
    fprintf(stderr, "[info] log_block_size=%d\n", log_block_size);

    auto gpu_list_dedup = [](auto& gpu_list){
        std::unordered_set<int> gpu_set{gpu_list.begin(), gpu_list.end()};
        // std::vector<int> gpu_list_dedup{gpu_set.begin(), gpu_set.end()};
        return std::vector<int>{gpu_set.begin(), gpu_set.end()};
    }(gpu_list);

    std::vector<cudaStream_t> stream(num_gpus);
    std::vector<cudaEvent_t> event_1(num_gpus);
    std::vector<cudaEvent_t> event_2(num_gpus);

    std::vector<decltype(Defer(cudaStreamDestroy, stream[0]))> defer_destroy_streams(num_gpus);
    std::vector<decltype(Defer(cudaEventDestroy, event_1[0]))> defer_destroy_event_1(num_gpus);
    std::vector<decltype(Defer(cudaEventDestroy, event_2[0]))> defer_destroy_event_2(num_gpus);

    for(int gpu_num=0; gpu_num<num_gpus; gpu_num++) {

        int const gpu_id = gpu_list[gpu_num]; 
        CHECK_CUDA(cudaSetDevice, gpu_id);

        CHECK_CUDA(cudaStreamCreate, &stream[gpu_num]);
        defer_destroy_streams[gpu_num] = {cudaStreamDestroy, stream[gpu_num]};

        CHECK_CUDA(cudaEventCreateWithFlags, &event_1[gpu_num], cudaEventDefault);
        defer_destroy_event_1[gpu_num] = {cudaEventDestroy, event_1[gpu_num]};

        CHECK_CUDA(cudaEventCreateWithFlags, &event_2[gpu_num], cudaEventDefault);
        defer_destroy_event_2[gpu_num] = {cudaEventDestroy, event_2[gpu_num]};

    }

    std::vector<my_complex_t**> state_data_device_list_constmem_addr(num_gpus);
    for(int gpu_num=0; gpu_num<num_gpus; gpu_num++) {
        int const gpu_id = gpu_list[gpu_num];
        CHECK_CUDA(cudaSetDevice, gpu_id);
        my_complex_t** addr;
        CHECK_CUDA(cudaGetSymbolAddress<decltype(state_data_device_list_constmem)>, (void**)&addr, state_data_device_list_constmem);
        state_data_device_list_constmem_addr[gpu_num] = addr;
    }

    int64_t const num_states = ((int64_t)1) << ((int64_t)num_qubits);

    int const num_qubits_local = num_qubits - log_num_gpus;
    int64_t const num_states_local = ((int64_t)1) << ((int64_t)num_qubits_local);
    int const block_size = 1 << log_block_size;
    int64_t const num_blocks = ((int64_t)1) << ((int64_t)(num_qubits_local - 1 - log_block_size));

    my_complex_t* state_data_host;
    // fprintf(stderr, "[info] cudaMallocHost state_data_host\n");
    // CHECK_CUDA(cudaMallocHost<void>, (void**)&state_data_host, num_states * sizeof(*state_data_host), 0);
    // Defer defer_free_state_data_host(cudaFreeHost, (void*)state_data_host);
    fprintf(stderr, "[info] malloc host memory\n");
    state_data_host = (my_complex_t*)malloc(num_states * sizeof(*state_data_host));
    Defer defer_free_state_data_host(free, (void*)state_data_host);

    fprintf(stderr, "[info] malloc device memory\n");
    std::vector<my_complex_t*> state_data_device_list(num_gpus);
    std::vector<decltype(Defer(cudaFree, (void*)0))> defer_free_device_mem(num_gpus);
    std::vector<my_float_t*> norm_sum_device_list(num_gpus);
    std::vector<decltype(Defer(cudaFree, (void*)0))> defer_free_norm_sum_device(num_gpus);

    for(int i=0; i<num_gpus; i++) {

        int const gpu_i = gpu_list[i]; 
        CHECK_CUDA(cudaSetDevice, gpu_i);

        my_complex_t* state_data_device;
        CHECK_CUDA(cudaMalloc<void>, (void**)&state_data_device, num_states_local * sizeof(*state_data_device));
        state_data_device_list[i] = state_data_device;
        defer_free_device_mem[i] = {cudaFree, (void*)state_data_device};

        my_float_t* norm_sum_device;
        CHECK_CUDA(cudaMalloc<void>, (void**)&norm_sum_device, (num_states_local>>log_block_size) * sizeof(my_float_t));
        norm_sum_device_list[i] = norm_sum_device;
        defer_free_norm_sum_device[i] = {cudaFree, (void*)norm_sum_device};
    }

    fprintf(stderr, "[info] update constant memory\n");
    for(int i=0; i<num_gpus; i++) {
        int const gpu_i = gpu_list[i];
        CHECK_CUDA(cudaSetDevice, gpu_i);
        CHECK_CUDA(cudaMemcpyAsync, state_data_device_list_constmem_addr[i], &state_data_device_list[0], state_data_device_list.size() * sizeof(state_data_device_list[0]), cudaMemcpyHostToDevice, stream[i]);
    }

    fprintf(stderr, "[info] enable peer access\n");
    for(int i=0; i<gpu_list_dedup.size(); i++) {
        int const gpu_i = gpu_list_dedup[i]; 
        for(int j=0; j<gpu_list_dedup.size(); j++) {
            if(i==j) continue;
            int const gpu_j = gpu_list_dedup[j]; 
            CHECK_CUDA(cudaSetDevice, gpu_i);
            CHECK_CUDA(cudaDeviceEnablePeerAccess, gpu_j, 0);
        }
    }

    for(int i=0; i<gpu_list_dedup.size(); i++) {
        int const gpu_i = gpu_list_dedup[i]; 
        for(int j=0; j<gpu_list_dedup.size(); j++) {
            if(i==j) continue;
            int const gpu_j = gpu_list_dedup[j]; 
            int canAccessPeer;
            CHECK_CUDA(cudaDeviceCanAccessPeer, &canAccessPeer, gpu_i, gpu_j);
            if (!canAccessPeer) {
                fprintf(stderr, "[error] GPU%d can not access GPU%d\n", gpu_i, gpu_j);
            }
        }
    }

    for(int i=0; i<num_gpus; i++) {
        CHECK_CUDA(cudaStreamSynchronize, stream[i]);
    }

    fprintf(stderr, "[info] generating random state\n");
    std::vector<curandGenerator_t> rng_device_list(num_gpus);
    for(int gpu_num = 0; gpu_num < num_gpus; gpu_num++) {
        int const gpu_id = gpu_list[gpu_num];
        CHECK_CUDA(cudaSetDevice, gpu_id);
        CHECK_CURAND(curandCreateGenerator, &rng_device_list[gpu_num], CURAND_RNG_PSEUDO_DEFAULT);
        CHECK_CURAND(curandSetStream, rng_device_list[gpu_num], stream[gpu_num]);
        CHECK_CURAND(curandSetPseudoRandomGeneratorSeed, rng_device_list[gpu_num], rng_seed + gpu_num);
    }

    fprintf(stderr, "[info] gpu reduce\n");
    for(int gpu_num = 0; gpu_num < num_gpus; gpu_num++) {
        int const gpu_id = gpu_list[gpu_num];
        CHECK_CUDA(cudaSetDevice, gpu_id);
        CHECK_CUDA(cudaEventRecord, event_1[gpu_num], stream[gpu_num]);
    }

    std::vector<my_float_t> norm_sum_list(num_gpus);
    for(int gpu_num = 0; gpu_num < num_gpus; gpu_num++) {
        int const gpu_id = gpu_list[gpu_num];
        CHECK_CUDA(cudaSetDevice, gpu_id);

        CHECK_CURAND(curandGenerateNormalDouble, rng_device_list[gpu_num], (my_float_t*)(void*)state_data_device_list[gpu_num], num_states_local * 2, 0.0, 1.0);

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

        norm_sum_reduce_kernel<<<num_blocks_reduce, block_size_reduce, sizeof(my_float_t) * block_size_reduce, stream[gpu_num]>>>(state_data_device_list[gpu_num], norm_sum_device_list[gpu_num]);

        data_length = num_blocks_reduce;

        while (data_length > 1) {
            if (data_length > block_size) {
                block_size_reduce = block_size;
                num_blocks_reduce = data_length >> log_block_size;
            } else {
                block_size_reduce = data_length;
                num_blocks_reduce = 1;
            }

            sum_reduce_kernel<<<num_blocks_reduce, block_size_reduce, sizeof(my_float_t) * block_size_reduce, stream[gpu_num]>>>(norm_sum_device_list[gpu_num], norm_sum_device_list[gpu_num]);

            data_length = num_blocks_reduce;
        }

        CHECK_CUDA(cudaMemcpyAsync, &norm_sum_list[gpu_num], norm_sum_device_list[gpu_num], sizeof(my_float_t), cudaMemcpyDeviceToHost, stream[gpu_num]);
    }

    for(int gpu_num=0; gpu_num<num_gpus; gpu_num++) {
        int const gpu_id = gpu_list[gpu_num];
        CHECK_CUDA(cudaSetDevice, gpu_id);
        CHECK_CUDA(cudaStreamSynchronize, stream[gpu_num]);
    }

    my_float_t norm_sum_gpu = 0;
    for(int gpu_num=0; gpu_num<num_gpus; gpu_num++) {
        int const gpu_id = gpu_list[gpu_num];
        CHECK_CUDA(cudaSetDevice, gpu_id);
        CHECK_CUDA(cudaStreamSynchronize, stream[gpu_num]);
        norm_sum_gpu += norm_sum_list[gpu_num];
    }
    fprintf(stderr, "[info] norm_sum_gpu=%lf\n", norm_sum_gpu);

    fprintf(stderr, "[info] normalize\n");
    my_float_t const normalize_factor = 1.0 / sqrt(norm_sum_gpu);
    for(int gpu_num=0; gpu_num<num_gpus; gpu_num++) {
        int const gpu_id = gpu_list[gpu_num];
        CHECK_CUDA(cudaSetDevice, gpu_id);

        int64_t data_length = num_states_local * 2;
        int64_t num_blocks_reduce;
        int block_size_reduce;

        if (data_length > block_size) {
            block_size_reduce = block_size;
            num_blocks_reduce = data_length >> log_block_size;
        } else {
            block_size_reduce = data_length;
            num_blocks_reduce = 1;
        }

        normalize_kernel<<<num_blocks_reduce, block_size_reduce, 0, stream[gpu_num]>>>((my_float_t*)(void*)state_data_device_list[gpu_num], normalize_factor);

        CHECK_CUDA(cudaEventRecord, event_2[gpu_num], stream[gpu_num]);
    }

    for(int gpu_num=0; gpu_num<num_gpus; gpu_num++) {
        int const gpu_id = gpu_list[gpu_num];
        CHECK_CUDA(cudaSetDevice, gpu_id);
        CHECK_CUDA(cudaStreamSynchronize, stream[gpu_num]);
    }

    double elapsed_rng = 0;
    for(int i=0; i<num_gpus; i++) {
        int const gpu_i = gpu_list[i]; 
        CHECK_CUDA(cudaSetDevice, gpu_i);

        float elapsed_i_ms;
        CHECK_CUDA(cudaEventElapsedTime, &elapsed_i_ms, event_1[i], event_2[i]);
        double const elapsed_i = elapsed_i_ms * 1e-3;

        if(elapsed_i>elapsed_rng) {
            elapsed_rng = elapsed_i;
        }
    }

    fprintf(stderr, "[info] normalize done\n");

    fprintf(stderr, "[info] elapsed_rng=%lf\n", elapsed_rng);
    fprintf(stdout, "%lf\n", elapsed_rng);

    fprintf(stderr, "[info] gpu_hadamard\n");

    for(int sample_num=0; sample_num < num_samples; ++sample_num) {

        for(int i=0; i<num_gpus; i++) {
            CHECK_CUDA(cudaEventRecord, event_1[i], stream[i]);
        }

        for(int target_qubit_num = target_qubit_num_begin; target_qubit_num < target_qubit_num_end; target_qubit_num++) {

            int const target_qubit_num_physical = perm_l2p[target_qubit_num];

            for(int i = 0; i < num_gpus; i++) {

                int const gpu_i = gpu_list[i]; 
                CHECK_CUDA(cudaSetDevice, gpu_i);

                cuda_gate<hadamard><<<num_blocks, block_size, 0, stream[i]>>>(num_gpus, log_num_gpus, i, num_qubits, target_qubit_num_physical);

            }

            if (target_qubit_num_physical >= num_qubits - log_num_gpus) {
                for(int i=0; i<num_gpus; i++) {
                    CHECK_CUDA(cudaStreamSynchronize, stream[i]);
                }
            }

        }

        for(int i=0; i<num_gpus; i++) {
            CHECK_CUDA(cudaEventRecord, event_2[i], stream[i]);
        }

        for(int i=0; i<num_gpus; i++) {
            CHECK_CUDA(cudaStreamSynchronize, stream[i]);
        }

        double elapsed_gpu = 0;
        for(int i=0; i<num_gpus; i++) {
            int const gpu_i = gpu_list[i]; 
            CHECK_CUDA(cudaSetDevice, gpu_i);

            float elapsed_i_ms;
            CHECK_CUDA(cudaEventElapsedTime, &elapsed_i_ms, event_1[i], event_2[i]);
            double const elapsed_i = elapsed_i_ms * 1e-3;

            if(elapsed_i>elapsed_gpu) {
                elapsed_gpu = elapsed_i;
            }
        }
        fprintf(stderr, "[info] elapsed_gpu=%lf\n", elapsed_gpu);
        fprintf(stdout, "%lf\n", elapsed_gpu);

    }

    // fprintf(stderr, "[info] transfer device data to host memory\n");
    // for(int i=0; i<num_gpus; i++) {
    //     int const gpu_i = gpu_list[i]; 
    //     CHECK_CUDA(cudaSetDevice, gpu_i);

    //     CHECK_CUDA(cudaMemcpyAsync, &state_data_host[num_states_local * i], state_data_device_list[i], num_states_local * sizeof(*state_data_device_list[0]), cudaMemcpyDeviceToHost, stream[i]);
    // }

    for(int i=0; i<num_gpus; i++) {
        CHECK_CUDA(cudaStreamSynchronize, stream[i]);
    }

    return 0;

}