#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <stdint.h>

#include <stdexcept>
#include <string>
// #include <iostream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <random>
#include <utility>

#include <numaif.h>
#include <sys/mman.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <numa.h>
#include <sched.h>



#define ARRAY_SIZE(array) (sizeof(array)/sizeof(*array))

__host__ __device__ int log2_int(int arg) {
    if(arg<=0) return -1;
    int value = 0;
    while(arg>1) {
        value += 1;
        arg = arg >> 1;
    }
    return value;
}

// template<typename INT, int arg> class log2_compile_time { public: static INT const value = 1 + log2_compile_time<INT, arg/2>::value; };
// template<typename INT> class log2_compile_time<INT, 1> { public: static INT const value = 0; };

int const gpu_list[] = {0, 1, 2, 3, 4, 5, 6, 7};
// int const gpu_list[] = {0};
// int const gpu_list[] = {0, 1};
// int const gpu_list[] = {0};
// int const gpu_list[] = {0, 0};
// int const gpu_list[] = {0, 1};
// int const gpu_list[] = {0, 0, 0, 0, 0, 0, 0, 0};
// int const gpu_list[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

bool const do_numa_sort = false;
// int const num_gpus = 1 << log_num_gpus;
int const num_gpus = ARRAY_SIZE(gpu_list);
// int const log_num_gpus = log2_compile_time<int, num_gpus>::value;
int const log_num_gpus = log2_int(num_gpus);

int const num_omp_threads = 256;
// int const log_num_omp_threads = log2_compile_time<int, num_omp_threads>::value;
int const log_num_omp_threads = log2_int(num_omp_threads);


int const num_qubits = 30;
int const log_block_size = 8;

typedef double my_float_t;

my_float_t const sqrt2 = 1.41421356237309504880168872420969807856967187537694;
my_float_t const inv_sqrt2 = 1 / sqrt2;

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
    }
}

// マクロで簡単に呼び出せるようにラップ
#define CHECK_CUDA(func, ...) check_cuda(__FILE__, __LINE__, #func, func, __VA_ARGS__)

// 可変長引数を取る関数ポインタをラップするテンプレート
template <typename Func, typename... Args>
class Defer {
public:
    // デフォルトコンストラクタ
    Defer() : valid_(false) {}

    // コンストラクタで関数ポインタと引数を受け取る
    // Defer(Func func, Args&&... args)
    //     : func_(func), args_(std::forward<Args>(args)...), valid_(true) {}
    Defer(Func func, Args... args)
        : func_(func), args_(args...), valid_(true) {}

    // ムーブ代入演算子
    Defer& operator=(Defer&& other) noexcept {
        if (this != &other) {
            func_ = other.func_;
            // args_ = std::move(other.args_);
            args_ = other.args_;
            valid_ = other.valid_;
            other.valid_ = false;
        }
        return *this;
    }

    // ムーブコンストラクタ
    Defer(Defer&& other) noexcept
        // : func_(other.func_), args_(std::move(other.args_)), valid_(other.valid_) {
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

__constant__ my_float_t* state_data_device_list_constmem[16];

// template<int num_split_areas>
class hadamard { public:
    static __device__ __host__ void apply(int num_split_areas, int64_t thread_num, int64_t const num_qubits, int64_t const target_qubit_num, my_float_t** const state_data_arg) {

        #ifdef __CUDA_ARCH__
            my_float_t** state_data = state_data_device_list_constmem;
        #else
            my_float_t** state_data = state_data_arg;
        #endif

        // int const log_num_split_areas = log2_compile_time<int, num_split_areas>::value;
        int const log_num_split_areas = log2_int(num_split_areas);

        int64_t const lower_mask = (((int64_t)1)<<target_qubit_num) - (int64_t)1;
        int64_t const split_mask = (((int64_t)1)<<((int64_t)(num_qubits - log_num_split_areas))) - (int64_t)1;

        int64_t const index_state_lower = thread_num & lower_mask;
        int64_t const index_state_higher = (thread_num & ~lower_mask) << ((int64_t)1);

        int64_t const index_state_0 = index_state_lower | index_state_higher;
        int64_t const index_state_1 = index_state_0 | (((int64_t)1)<<target_qubit_num);

        int64_t const index_state_0_split_num = index_state_0 >> (num_qubits - log_num_split_areas);
        int64_t const index_state_0_split_address = index_state_0 & split_mask;

        int64_t const index_state_1_split_num = index_state_1 >> (num_qubits - log_num_split_areas);
        int64_t const index_state_1_split_address = index_state_1 & split_mask;

        my_float_t const amp_state_0 = state_data[index_state_0_split_num][index_state_0_split_address];
        my_float_t const amp_state_1 = state_data[index_state_1_split_num][index_state_1_split_address];

        state_data[index_state_0_split_num][index_state_0_split_address] = (amp_state_0 + amp_state_1) * inv_sqrt2;
        state_data[index_state_1_split_num][index_state_1_split_address] = (amp_state_0 - amp_state_1) * inv_sqrt2;

    }

};

template<class Gate>
__global__ void cuda_gate(int num_split_areas, int64_t split_num, int64_t const num_qubits, int64_t const target_qubit_num, my_float_t** const) {
    // int const log_num_split_areas = log2_compile_time<int, num_split_areas>::value;
    int const log_num_split_areas = log2_int(num_split_areas);
    int64_t const num_qubits_local = num_qubits - log_num_split_areas;
    int64_t const num_threads_local = ((int64_t)1) << (num_qubits_local-1);

    int64_t const thread_num = threadIdx.x + blockIdx.x * blockDim.x + num_threads_local * split_num;
    Gate::apply(num_split_areas, thread_num, num_qubits, target_qubit_num, 0);
}

template<class Gate>
void omp_gate(int num_split_areas, int64_t split_num, int64_t const num_qubits, int64_t const target_qubit_num, my_float_t** const state_data) {

    // my_float_t* state_data_array[1];
    // state_data_array[0] = state_data;

    int const log_num_split_areas = log2_int(num_split_areas);
    int64_t const num_qubits_local = num_qubits - log_num_split_areas;
    int64_t const num_threads_local = ((int64_t)1) << (num_qubits_local-1);

    // int64_t const num_threads_local = ((int64_t)1) << ((int64_t)(num_qubits-1));
    // #pragma omp parallel for 

    for(int64_t thread_num_local = 0; thread_num_local < num_threads_local; thread_num_local++) {
        int64_t thread_num = thread_num_local + num_threads_local * split_num;
        Gate::apply(num_split_areas, thread_num, num_qubits, target_qubit_num, state_data);
    }
}



int main() {

    setvbuf(stdout, NULL, _IOLBF, 1024 * 512);

    fprintf(stderr, "[info] num_qubits=%d\n", num_qubits);
    fprintf(stderr, "[info] num_gpus=%d (", num_gpus);
    for(int i=0; i<num_gpus; i++) {
        fprintf(stderr, "%d, ", gpu_list[i]);
    }
    fprintf(stderr, ")\n");
    fprintf(stderr, "[info] log_block_size=%d\n", log_block_size);

    fprintf(stderr, "[info] num_omp_threads=%d\n", num_omp_threads);

    fprintf(stderr, "[info] do_numa_sort=%d\n", do_numa_sort);

    // int const dev_list[] = {0, 1, 2, 3};
    // int const num_gpus = ARRAY_SIZE(dev_list);
    // int const log_num_gpus = log2(num_gpus);

    std::vector<cudaStream_t> stream(num_gpus);
    std::vector<cudaEvent_t> event_1(num_gpus);
    std::vector<cudaEvent_t> event_2(num_gpus);

    std::vector<my_float_t**> state_data_device_list_constmem_addr(num_gpus);
    for(int i=0; i<num_gpus; i++) {
        CHECK_CUDA(cudaSetDevice, i);

        my_float_t** addr;
        CHECK_CUDA(cudaGetSymbolAddress<decltype(state_data_device_list_constmem)>, (void**)&addr, state_data_device_list_constmem);
        // err = cudaGetSymbolAddress(&addr, state_data_device_list_constmem);
        // if (err != cudaSuccess)
        // {
        //     fprintf(stderr, "[debug] l%d d%d error:%s\n", __LINE__, i, cudaGetErrorString(err));
        // }
        // fprintf(stderr, "[info] l%d d%d addr=%p\n", __LINE__, i, addr);

        state_data_device_list_constmem_addr[i] = (my_float_t**)addr;

        // std::vector<double*> dummy(8); // = {0, 1, 2, 3, 4, 5};

        // err = cudaMemcpy(addr, &dummy[0], dummy.size() * sizeof(dummy[0]), cudaMemcpyHostToDevice);
        // if (err != cudaSuccess)
        // {
        //     fprintf(stderr, "[debug] l%d d%d error:%s\n", __LINE__, i, cudaGetErrorString(err));
        // }

    }

    // Defer<void(*)()> defer_destroy_streams[num_gpus];
    std::vector<decltype(Defer(cudaStreamDestroy, stream[0]))> defer_destroy_streams(num_gpus);
    // Defer<decltype(&cudaStreamDestroy), decltype(stream[0])> defer_destroy_streams[num_gpus];
    std::vector<decltype(Defer(cudaEventDestroy, event_1[0]))> defer_destroy_event_1(num_gpus);
    std::vector<decltype(Defer(cudaEventDestroy, event_2[0]))> defer_destroy_event_2(num_gpus);

    for(int i=0; i<num_gpus; i++) {
        int const gpu_i = gpu_list[i]; 
        CHECK_CUDA(cudaSetDevice, gpu_i);

        CHECK_CUDA(cudaStreamCreate, &stream[i]);
        // defer_destroy_streams[i] = decltype(defer_destroy_streams)({cudaStreamDestroy, stream[i]});
        defer_destroy_streams[i] = {cudaStreamDestroy, stream[i]};
        // defer_destroy_streams[i] = decltype(defer_destroy_streams[0])({&cudaStreamDestroy, stream[i]});
        
        CHECK_CUDA(cudaEventCreateWithFlags, &event_1[i], cudaEventDefault);
        // CHECK_CUDA((cudaError_t(*)(cudaEvent_t*))cudaEventCreate, &event_1);
        defer_destroy_event_1[i] = {cudaEventDestroy, event_1[i]};
        
        CHECK_CUDA(cudaEventCreateWithFlags, &event_2[i], cudaEventDefault);
        // CHECK_CUDA((cudaError_t(*)(cudaEvent_t*))cudaEventCreate, &event_2);
        defer_destroy_event_2[i] = {cudaEventDestroy, event_2[i]};
    }

    // struct defer_destroy_streams_class {
    //     cudaStream_t* stream; int length;
    //     defer_destroy_streams_class(cudaStream_t* stream, int length) : stream(stream), length(length) {}
    //     ~defer_destroy_streams_class() {
    //         for (int i = 0; i < length; ++i) {
    //             cudaStreamDestroy(stream[i]);
    //         }
    //     }
    // } defer_destroy_streams_(stream, num_gpus);

    int64_t const num_states = ((int64_t)1) << ((int64_t)num_qubits);

    int const num_qubits_local = num_qubits - log_num_gpus;
    int64_t const num_states_local = ((int64_t)1) << ((int64_t)num_qubits_local);
    int const block_size = 1 << log_block_size;
    int64_t const num_blocks = ((int64_t)1) << ((int64_t)(num_qubits_local-1-log_block_size));
    // int64_t const num_threads = ((int64_t)1) << ((int64_t)(num_qubits-1));

    int64_t const num_qubits_local_omp = num_qubits - log_num_omp_threads;
    int64_t const num_states_local_omp = ((int64_t)1) << num_qubits_local_omp;

    // int const target_qubit_num = num_qubits - 1;
    int const target_qubit_num = 0;
    fprintf(stderr, "[info] target_qubit_num=%d\n", target_qubit_num);

    fprintf(stderr, "[info] cudaMallocHost state_data_host\n", target_qubit_num);
    my_float_t* state_data_host;
    // CHECK_CUDA((cudaError_t (*)(void**, size_t))&cudaMallocHost, (void**)&state_data_host, num_states * sizeof(*state_data_host));
    CHECK_CUDA(cudaMallocHost<void>, (void**)&state_data_host, num_states * sizeof(*state_data_host), 0);
    // CHECK_CUDA(cudaMallocHost<void>, &state_data_host, num_states * sizeof(*state_data_host), 0);
    // CHECK_CUDA(cudaMallocHost<my_float_t>, &state_data_host, num_states * sizeof(*state_data_host), 0);
    // CHECK_CUDA([&state_data_host, num_states](){ return cudaMallocHost(&state_data_host, num_states * sizeof(*state_data_host), 0); });

    Defer defer_free_state_data_host(cudaFreeHost, (void*)state_data_host);

    fprintf(stderr, "[info] generating random state\n");
    my_float_t sum_pow2 = 0;
    std::vector<my_float_t> sum_pow2_list(num_omp_threads);
    std::vector<std::chrono::_V2::system_clock::time_point>
        time_rng_begin(num_omp_threads),
        time_rng_end(num_omp_threads);

    #pragma omp parallel
    {
        int const omp_thread_num = omp_get_thread_num();

        time_rng_begin[omp_thread_num] = std::chrono::high_resolution_clock::now();

        std::mt19937_64 mt(12345 + omp_thread_num);
        std::uniform_real_distribution<double> uni01(0.0,1.0);

        my_float_t sum_pow2_local = 0;
        for(int64_t state_num = num_states_local_omp * omp_thread_num;
            state_num < num_states_local_omp * (omp_thread_num + 1);
            state_num++) {
            // my_float_t const amp = static_cast<my_float_t>(rand()) / RAND_MAX;
            my_float_t const amp = uni01(mt);
            state_data_host[state_num] = amp;
            sum_pow2_local += amp * amp;
        }

        sum_pow2_list[omp_thread_num] = sum_pow2_local;

        #pragma omp barrier

        #pragma omp single
        for(int i=0; i<num_omp_threads; i++) {
            sum_pow2 += sum_pow2_list[i];
        }

        #pragma omp barrier

        for(int64_t state_num = num_states_local_omp * omp_thread_num;
            state_num < num_states_local_omp * (omp_thread_num + 1);
            state_num++) {
            state_data_host[state_num] /= sum_pow2;
        }

        time_rng_end[omp_thread_num] = std::chrono::high_resolution_clock::now();
    }

    double elapsed_rng = 0;
    for(int i=0; i<num_omp_threads; i++) {
        double const elapsed_rng_i = std::chrono::duration<double>(time_rng_end[i] - time_rng_begin[i]).count();
        if(elapsed_rng_i>elapsed_rng) {
            elapsed_rng = elapsed_rng_i;
        }
    }
    fprintf(stderr, "[info] elapsed_rng=%lf\n", elapsed_rng);

    // for(int64_t state_num=0; state_num<num_states; state_num++) {
    //     state_data_host[state_num] /= sum_pow2;
    // }


    // my_float_t* state_data_host_2;
    // CHECK_CUDA((cudaError_t (*)(void**, size_t))&cudaMallocHost, (void**)&state_data_host_2, num_states * sizeof(*state_data_host_2));
    // CHECK_CUDA(cudaMallocHost<void>, (void**)&state_data_host_2, num_states * sizeof(*state_data_host_2), 0);
    my_float_t* const state_data_host_2 = (my_float_t*)malloc(num_states * sizeof(*state_data_host_2));
    // Defer defer_free_state_data_host_2(cudaFreeHost, (void*)state_data_host_2);
    Defer defer_free_state_data_host_2(free, (void*)state_data_host_2);
    memcpy(state_data_host_2, state_data_host, num_states * sizeof(*state_data_host_2));

    std::vector<my_float_t*> state_data_device_list(num_gpus);
    // my_array<my_float_t*, num_gpus> state_data_device_list;
    // std::vector<decltype(Defer(cudaFree, (void*)0))> defer_free_device_mem;
    std::vector<decltype(Defer(cudaFree, (void*)0))> defer_free_device_mem(num_gpus);
    // Defer<decltype(&cudaFree), void*> defer_free_device_mem[num_gpus];

    for(int i=0; i<num_gpus; i++) {
    // for(int i=num_gpus-1; i>=0; i--) {

        int const gpu_i = gpu_list[i]; 
        CHECK_CUDA(cudaSetDevice, gpu_i);
        // CHECK_CUDA(cudaSetDevice, 1);

        // CHECK_CUDA((cudaError_t (*)(void**, size_t))&cudaMalloc, (void**)&state_data_device, num_states * sizeof(*state_data_device));
        my_float_t* state_data_device;
        // CHECK_CUDA(cudaMalloc<void>, (void**)&state_data_device, num_states_local * sizeof(*state_data_device));
        CHECK_CUDA(cudaMalloc<void>, (void**)&state_data_device, num_states_local * sizeof(*state_data_device));
        state_data_device_list[i] = state_data_device;

        CHECK_CUDA(cudaMemcpyAsync, state_data_device, &state_data_host[num_states_local * i], num_states_local * sizeof(*state_data_device), cudaMemcpyHostToDevice, stream[i]);
        // defer_free_device_mem.push_back({cudaFree, (void*)state_data_device});
        defer_free_device_mem[i] = {cudaFree, (void*)state_data_device};

        // cudaEvent_t event_cudaMemcpyAsync;
        // CHECK_CUDA(cudaEventCreateWithFlags, &event_cudaMemcpyAsync, cudaEventDefault);

        // CHECK_CUDA(cudaEventRecord, event_cudaMemcpyAsync, stream);

        // // CHECK_CUDA((cudaError_t ()(cudaStream_t, ))cudaStreamWaitEvent , stream, event_cudaMemcpyAsync, 0);
        // CHECK_CUDA(cudaStreamWaitEvent , stream, event_cudaMemcpyAsync, 0);

        // CHECK_CUDA(cudaStreamSynchronize, stream);
    }
    // Defer defer_free_state_data_device(cudaFree, (void*)state_data_device);

    for(int i=0; i<num_gpus; i++) {

        int const gpu_i = gpu_list[i]; 
        // CHECK_CUDA(cudaSetDevice, gpu_i);
        CHECK_CUDA(cudaSetDevice, gpu_i);

        // CHECK_CUDA(cudaMemcpyToSymbolAsync<void*>, state_data_device_list_constmem, &state_data_device_list[0], state_data_device_list.size() * sizeof(state_data_device_list[0]), 0, cudaMemcpyHostToDevice, stream);
        // cudaMemcpyToSymbolAsync<void*>(state_data_device_list_constmem, &state_data_device_list[0], state_data_device_list.size() * sizeof(state_data_device_list[0]), 0, cudaMemcpyHostToDevice, stream[i]);
        // CHECK_CUDA(cudaMemcpyToSymbolAsync, "state_data_device_list_constmem", (const void*)&state_data_device_list[0], state_data_device_list.size() * sizeof(state_data_device_list[0]), 0, cudaMemcpyHostToDevice, stream[i]);
        // cudaMemcpyToSymbolAsync("state_data_device_list_constmema", &state_data_device_list[0], state_data_device_list.size() * sizeof(state_data_device_list[0]), 0, cudaMemcpyHostToDevice, stream[i]);
        // CHECK_CUDA(cudaMemcpyToSymbolAsync<void*>, state_data_device_list_constmem, &state_data_device_list[0], state_data_device_list.size() * sizeof(state_data_device_list[0]), 0, cudaMemcpyHostToDevice, stream[i]);
        CHECK_CUDA(cudaMemcpyAsync, state_data_device_list_constmem_addr[i], &state_data_device_list[0], state_data_device_list.size() * sizeof(state_data_device_list[0]), cudaMemcpyHostToDevice, stream[i]);
    }

    for(int i=0; i<num_gpus; i++) {
        int const gpu_i = gpu_list[i]; 
        for(int j=0; j<num_gpus; j++) {
            int const gpu_j = gpu_list[j]; 
            if (gpu_i == gpu_j) continue;
            CHECK_CUDA(cudaSetDevice, gpu_i);
            CHECK_CUDA(cudaDeviceEnablePeerAccess, gpu_j, 0);
        }
    }

    for(int i=0; i<num_gpus; i++) {
        int const gpu_i = gpu_list[i]; 
        for(int j=0; j<num_gpus; j++) {
            int const gpu_j = gpu_list[j]; 
            if (gpu_i == gpu_j) continue;
            int canAccessPeer;
            CHECK_CUDA(cudaDeviceCanAccessPeer, &canAccessPeer, gpu_i, gpu_j);
            if (!canAccessPeer) {
                fprintf(stderr, "[error] GPU%d can not access GPU%d\n", gpu_i, gpu_j);
            }
        }
    }

    // CHECK_CUDA(cudaStreamSynchronize, stream[]);
    // for(int state_num=0; state_num<num_states; state_num++) {
    //     state_data_host[state_num] = 0.0;
    // }

    for(int i=0; i<num_gpus; i++) {
        CHECK_CUDA(cudaStreamSynchronize, stream[i]);
    }

    fprintf(stderr, "[info] gpu_hadamard\n");

    for(int i=0; i<num_gpus; i++) {
        CHECK_CUDA(cudaEventRecord, event_1[i], stream[i]);
    }

    // for(int i=num_gpus-1; i>=0; i--) {
    for(int i=0; i<num_gpus; i++) {

        int const gpu_i = gpu_list[i]; 
        // CHECK_CUDA(cudaSetDevice, gpu_i);
        CHECK_CUDA(cudaSetDevice, gpu_i);

        // CHECK_CUDA(cudaMemcpyToSymbolAsync<void*>, state_data_device_list_constmem, &state_data_device_list[0], state_data_device_list.size() * sizeof(state_data_device_list[0]), 0, cudaMemcpyHostToDevice, stream);

        cuda_gate<hadamard><<<num_blocks, block_size, 0, stream[i]>>>(num_gpus, i, num_qubits, target_qubit_num, 0);

        // CHECK_CUDA(cudaStreamSynchronize, stream[i]);

    }

    // my_float_t* state_data_host_split[1];
    // for(int i=0; i<1; i++) {
    //     state_data_host_split[i] = state_data_host;
    // }
    // cpu_gate<hadamard, 1>(num_qubits, target_qubit_num, state_data_host_split);

    for(int i=0; i<num_gpus; i++) {
        CHECK_CUDA(cudaEventRecord, event_2[i], stream[i]);
    }

    fprintf(stderr, "[info] cpu_hadamard\n");

    std::vector<my_float_t*> state_data_host_2_split(num_omp_threads);
    std::vector<std::chrono::_V2::system_clock::time_point>
        time_cpu_begin(num_omp_threads),
        time_cpu_end(num_omp_threads);

    omp_set_num_threads(num_omp_threads);

    std::vector<int> numa_node_list(num_omp_threads);
    std::vector<int> thread_num_sort(num_omp_threads);
    std::vector<int> thread_index_list(num_omp_threads);

    #pragma omp parallel
    {
        // int const num_omp_threads_runtime = omp_get_num_threads();
        // if(num_omp_threads!=num_omp_threads_runtime) {
        //     fprintf(stderr, "[error] num_omp_threads=%f num_omp_threads_runtime=%d\n", num_omp_threads, num_omp_threads_runtime);
        //     // return -1;
        // }

        int const omp_thread_num = omp_get_thread_num();

        /* ---- begin of NUMA sort ---- */

        if (numa_available() < 0) {
            fprintf(stderr, "[error] NUMA is not available on this system.\n");
            // return 1;
        }

        // 現在のスレッドがバインドされているノードの取得
        int current_node = numa_node_of_cpu(sched_getcpu());
        if (current_node == -1) {
            fprintf(stderr, "[error] %d Failed to get the current CPU's NUMA node.\n", omp_thread_num);
            // return 1;
        }

        numa_node_list[omp_thread_num] = current_node;

        #pragma omp barrier

        #pragma omp single
        {
            for (int i = 0; i < num_omp_threads; ++i) {
                thread_num_sort[i] = i;
            }

            // ソート：元の配列の値を基にインデックスをソート
            if(do_numa_sort) {
                std::sort(thread_num_sort.begin(), thread_num_sort.end(), [&numa_node_list](auto i1, auto i2) { return numa_node_list[i1] < numa_node_list[i2]; /* 昇順 */ });
            }
            
            for (int thread_index_for = 0; thread_index_for < num_omp_threads; ++thread_index_for) {
                int const thread_num_for = thread_num_sort[thread_index_for];
                thread_index_list[thread_num_for] = thread_index_for;
            }           

            // for (int i = 0; i < thread_num_sort.size(); ++i) {
            //     fprintf(stderr, "[info] thread_num=%d thread_index=%d numa_node=%d\n", thread_num_sort[i], i, numa_node_list[thread_num_sort[i]]);
            // }

        }

        #pragma omp barrier

        // for (int thread_index_for = 0; thread_index_for < num_omp_threads; ++thread_index_for) {
        // auto begin = state_data_host_2 + thread_index_for * num_states_local_omp;
        // auto end = state_data_host_2 + (thread_index_for+1) * num_states_local_omp;
        int const thread_index = thread_index_list[omp_thread_num];
        // int const numa_node = numa_node_list[thread_num_sort[thread_index]];

        // unsigned long nodemask = 1UL << numa_node;

        // auto split_address = state_data_host_2 + thread_index * num_states_local_omp;
        // if (mbind(split_address, num_states_local_omp * sizeof(my_float_t), MPOL_BIND, &nodemask, sizeof(nodemask) * 8, 0) != 0) {
        //     fprintf(stderr, "[error] mbind for thread_index %d failed\n", thread_index);
        //     // free(ptr);
        //     // return 1;
        // }

        my_float_t* state_data_host_2_split_i;
        if(do_numa_sort) {
            state_data_host_2_split_i = (my_float_t*)numa_alloc_onnode(num_states_local_omp * sizeof(*state_data_host_2_split_i), current_node);
        } else {
            state_data_host_2_split_i = (my_float_t*)malloc(num_states_local_omp * sizeof(*state_data_host_2_split_i));
        }

        state_data_host_2_split[thread_index] = state_data_host_2_split_i;
        // }

        /* ---- end of NUMA sort ---- */

        // if (omp_thread_num==0) {
        // my_float_t* state_data_host_2_split[num_gpus];
        // for(int i=0; i<num_gpus; i++) {
        // state_data_host_2_split[omp_thread_num] = (my_float_t*)malloc(num_states_local_omp * sizeof(state_data_host_2_split[0][0]));
        // state_data_host_2_split[i] = &state_data_host_2[i * num_states_local];
        // }
        //}

        memcpy(state_data_host_2_split[thread_index], state_data_host_2 + num_states_local_omp * thread_index, num_states_local_omp * sizeof(*state_data_host_2));

        #pragma omp barrier

        time_cpu_begin[omp_thread_num] = std::chrono::high_resolution_clock::now();

        // cpu_gate<hadamard, 1>(0, num_qubits, target_qubit_num, &state_data_host_2);

        // for(int i=0; i<num_gpus; i++) {
        // state_data_host_2_split[i] = &state_data_host_2[i * num_states_local];
        // int thread_index = thread_index_list[omp_thread_num];
        omp_gate<hadamard>(num_omp_threads, thread_index, num_qubits, target_qubit_num, &state_data_host_2_split[0]);
        // cpu_gate<hadamard, num_omp_threads>(omp_thread_num, num_qubits, target_qubit_num, state_data_host_2_split);
        //}

        // auto end = std::chrono::high_resolution_clock::now();
        time_cpu_end[omp_thread_num] = std::chrono::high_resolution_clock::now();

        #pragma omp barrier

        memcpy(state_data_host_2 + num_states_local_omp * thread_index, state_data_host_2_split[thread_index], num_states_local_omp * sizeof(*state_data_host_2));

    }

    double elapsed_cpu = 0;
    for(int i=0; i<num_omp_threads; i++) {
        double const elapsed_cpu_i = std::chrono::duration<double>(time_cpu_end[i] - time_cpu_begin[i]).count();
        if(elapsed_cpu_i>elapsed_cpu) {
            elapsed_cpu = elapsed_cpu_i;
        }
    }

    fprintf(stderr, "[info] elapsed_cpu=%lf\n", elapsed_cpu);

    fprintf(stderr, "[info] wait for GPU kernel completion...\n");
    for(int i=0; i<num_gpus; i++) {
        CHECK_CUDA(cudaStreamSynchronize, stream[i]);
    }

    // double const elapsed = [](auto event_1, auto event_2) {
    //     float _elapsed_fp32;
    //     CHECK_CUDA(cudaEventElapsedTime, &_elapsed_fp32, event_1, event_2);
    //     return _elapsed_fp32 * 1e-3;
    // } (event_1, event_2);

    // double time_1_min = 1e300;
    // double time_2_max = -1e300;
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


    // CHECK_CUDA(cudaStreamWaitEvent , stream, event_2, 0);

    // CHECK_CUDA(cudaMemcpyAsync, state_data_host, state_data_device, num_states * sizeof(*state_data_host), cudaMemcpyDeviceToHost, stream);
    for(int i=0; i<num_gpus; i++) {
    // for(int i=num_gpus-1; i>=0; i--) {
        int const gpu_i = gpu_list[i]; 
        CHECK_CUDA(cudaSetDevice, gpu_i);

        CHECK_CUDA(cudaMemcpyAsync, &state_data_host[num_states_local * i], state_data_device_list[i], num_states_local * sizeof(*state_data_device_list[0]), cudaMemcpyDeviceToHost, stream[i]);

        // CHECK_CUDA(cudaStreamSynchronize, stream);

        // CHECK_CUDA(cudaStreamSynchronize, stream);
        // cudaEvent_t event_cudaMemcpyAsync;
        // CHECK_CUDA(cudaEventCreateWithFlags, &event_cudaMemcpyAsync, cudaEventDefault);

        // CHECK_CUDA(cudaEventRecord, event_cudaMemcpyAsync, stream);

        // // CHECK_CUDA((cudaError_t ()(cudaStream_t, ))cudaStreamWaitEvent , stream, event_cudaMemcpyAsync, 0);
        // CHECK_CUDA(cudaStreamWaitEvent , stream, event_cudaMemcpyAsync, 0);

    }

    // if (cudaSuccess != cudaStreamQuery(stream)) {
    //     fprintf(stderr, "[info] wait for GPU job completion...\n");
    // }

    fprintf(stderr, "[info] wait for GPU memory transfer completion...\n");
    for(int i=0; i<num_gpus; i++) {
        CHECK_CUDA(cudaStreamSynchronize, stream[i]);
    }

    // cudaDeviceSynchronize();


    my_float_t max_diff = 0;
    for(int i=0; i<num_states; i++) {
        my_float_t diff = std::fabs(state_data_host[i] - state_data_host_2[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    fprintf(stderr, "[info] max_diff=%.16e\n", max_diff);

    my_float_t max_rel_diff = 0;
    for(int i=0; i<num_states; i++) {
        my_float_t const state_data_host_i = state_data_host[i];
        my_float_t const state_data_host_2_i = state_data_host_2[i];
        if (state_data_host_i == 0 || state_data_host_2_i == 0) {
            continue;
        }
        my_float_t rel_diff = (state_data_host_i!=0)? std::fabs(state_data_host_i / state_data_host_2_i - 1) : std::fabs(state_data_host_2_i / state_data_host_i - 1);
        if (rel_diff > max_rel_diff) {
            max_rel_diff = rel_diff;
        }
    }
    fprintf(stderr, "[info] max_rel_diff=%.16e\n", max_rel_diff);

    // for(int i=0; i<num_states; i++) {
    //     fprintf(stdout, "%lld,%.16e,%.16e,%d\n", i, state_data_host[i], state_data_host_2[i], state_data_host[i]==state_data_host_2[i]);
    //     // fprintf(stdout, "%lld,%.16e\n", i, state_data_host[i]);
    // }

    return 0;
}