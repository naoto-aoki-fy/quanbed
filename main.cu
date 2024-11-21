#include <cstdlib>
#include <cstdio>
#include <stdint.h>
#include <stdexcept>
#include <string>
#include <iostream>
#include <sstream>
#include <string.h>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <random>

#include <omp.h>
#include <cuda_runtime.h>

#define ARRAY_SIZE(array) (sizeof(array)/sizeof(*array))

template<typename INT, int arg> class log2_compile_time { public: static INT const value = 1 + log2_compile_time<INT, arg/2>::value; };
template<typename INT> class log2_compile_time<INT, 1> { public: static INT const value = 0; };

// int const gpu_list[] = {0, 1, 2, 3, 4, 5, 6, 7};
int const gpu_list[] = {0, 1, 2, 3};
// int const gpu_list[] = {0, 1};
// int const gpu_list[] = {0};
// int const gpu_list[] = {0, 0};
// int const gpu_list[] = {0, 1};
// int const gpu_list[] = {0, 0, 0, 0, 0, 0, 0, 0};
// int const gpu_list[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

// int const num_gpus = 1 << log_num_gpus;
int const num_gpus = ARRAY_SIZE(gpu_list);
int const log_num_gpus = log2_compile_time<int, num_gpus>::value;

int const num_omp_threads = 256;
int const log_num_omp_threads = log2_compile_time<int, num_omp_threads>::value;

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

// 任意の型の引数を取る関数をラップするテンプレート
template <typename Func, typename Arg>
class Defer {
public:
    Defer(): valid_(false) {}

    // コンストラクタで関数ポインタと引数を受け取る
    Defer(Func func, Arg&& arg)
        : func_(func), arg_(std::forward<Arg>(arg)), valid_(true) {}

    // ムーブ代入演算子
    Defer& operator=(Defer&& other) noexcept {
        // fprintf(stderr, "[info] move substitute\n");
        if (this != &other) {
            func_ = std::move(other.func_);
            arg_ = std::move(other.arg_);
            valid_ = other.valid_;
            other.valid_ = false;  // 元のオブジェクトの処理を無効化
        }
        return *this;
    }

    // ムーブコンストラクタ
    Defer(Defer&& other) noexcept
        : func_(std::move(other.func_)), arg_(std::move(other.arg_)), valid_(other.valid_) {
        // fprintf(stderr, "[info] move constructor\n");
        other.valid_ = false;  // 元のオブジェクトの処理を無効化
    }

    // デストラクタで関数ポインタを呼び出す
    ~Defer() {
        if (valid_) {
            // fprintf(stderr, "[debug] deconstructor called\n");
            func_(arg_);
        }
    }

private:
    Func func_; // 関数ポインタ
    Arg arg_; // 関数の引数
    bool valid_;
};

template<int num_split_areas>
class hadamard {
public:
    static __device__ __host__ void apply(int64_t thread_num, int64_t const num_qubits, int64_t const target_qubit_num, my_float_t* const state_data[num_split_areas]) {

        int const log_num_split_areas = log2_compile_time<int, num_split_areas>::value;

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

template<template<int> class Gate, int num_split_areas>
__global__ void gpu_gate(int64_t split_num, int64_t const num_qubits, int64_t const target_qubit_num, my_float_t* const state_data[num_split_areas]) {
    int const log_num_split_areas = log2_compile_time<int, num_split_areas>::value;
    int64_t const num_qubits_local = num_qubits - log_num_split_areas;
    int64_t const num_threads_local = ((int64_t)1) << (num_qubits_local-1);

    int64_t const thread_num = threadIdx.x + blockIdx.x * blockDim.x + num_threads_local * split_num;
    Gate<num_split_areas>::apply(thread_num, num_qubits, target_qubit_num, state_data);
}

template<template<int> class Gate, int num_split_areas>
void cpu_gate(int64_t split_num, int64_t const num_qubits, int64_t const target_qubit_num, my_float_t* const state_data[num_split_areas]) {
    
    // my_float_t* state_data_array[1];
    // state_data_array[0] = state_data;

    int const log_num_split_areas = log2_compile_time<int, num_split_areas>::value;
    int64_t const num_qubits_local = num_qubits - log_num_split_areas;
    int64_t const num_threads_local = ((int64_t)1) << (num_qubits_local-1);
    // int64_t const num_threads_local = ((int64_t)1) << ((int64_t)(num_qubits-1));
    // #pragma omp parallel for 
    for(int64_t thread_num_local = 0; thread_num_local < num_threads_local; thread_num_local++) {
        int64_t thread_num = thread_num_local + num_threads_local * split_num;
        Gate<num_split_areas>::apply(thread_num, num_qubits, target_qubit_num, state_data);
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

    // int const dev_list[] = {0, 1, 2, 3};
    // int const num_gpus = ARRAY_SIZE(dev_list);
    // int const log_num_gpus = log2(num_gpus);

    cudaStream_t stream[num_gpus];
    cudaEvent_t event_1[num_gpus];
    cudaEvent_t event_2[num_gpus];

    for(int i=0; i<num_gpus; i++) {
        int const gpu_i = gpu_list[i]; 
        CHECK_CUDA(cudaSetDevice, gpu_i);

        CHECK_CUDA(cudaStreamCreate, &stream[i]);
        
        CHECK_CUDA(cudaEventCreateWithFlags, &event_1[i], cudaEventDefault);
        // CHECK_CUDA((cudaError_t(*)(cudaEvent_t*))cudaEventCreate, &event_1);
        
        CHECK_CUDA(cudaEventCreateWithFlags, &event_2[i], cudaEventDefault);
        // CHECK_CUDA((cudaError_t(*)(cudaEvent_t*))cudaEventCreate, &event_2);
    }

    CHECK_CUDA(cudaSetDevice, gpu_list[0]);
    cudaEvent_t ref_event;
    CHECK_CUDA(cudaEventCreateWithFlags, &ref_event, cudaEventDefault);
    CHECK_CUDA(cudaEventRecord, ref_event, stream[0]);

    int64_t const num_states = ((int64_t)1) << ((int64_t)num_qubits);

    int const num_qubits_local = num_qubits - log_num_gpus;
    int64_t const num_states_local = ((int64_t)1) << ((int64_t)num_qubits_local);
    int const block_size = 1 << log_block_size;
    int64_t const num_blocks = ((int64_t)1) << ((int64_t)(num_qubits_local-1-log_block_size));
    // int64_t const num_threads = ((int64_t)1) << ((int64_t)(num_qubits-1));

    int64_t const num_qubits_local_omp = num_qubits - log_num_omp_threads;
    int64_t const num_states_local_omp = ((int64_t)1) << num_qubits_local_omp;

    int const target_qubit_num = num_qubits - 1;
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
    my_float_t sum_pow2_list[num_omp_threads];
    std::chrono::_V2::system_clock::time_point
        time_rng_begin[num_omp_threads],
        time_rng_end[num_omp_threads];

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

    my_float_t* state_data_device_list[num_gpus];
    // my_array<my_float_t*, num_gpus> state_data_device_list;
    // std::vector<decltype(Defer(cudaFree, (void*)0))> defer_free_device_mem;
    decltype(Defer(cudaFree, (void*)0)) defer_free_device_mem[num_gpus];
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

        gpu_gate<hadamard, num_gpus><<<num_blocks, block_size, 0, stream[i]>>>(i, num_qubits, target_qubit_num, state_data_device_list);

        CHECK_CUDA(cudaStreamSynchronize, stream[i]);

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

    my_float_t* state_data_host_2_split[num_omp_threads];
    std::chrono::_V2::system_clock::time_point
        time_cpu_begin[num_omp_threads],
        time_cpu_end[num_omp_threads];

    omp_set_num_threads(num_omp_threads);

    #pragma omp parallel
    {
        // int const num_omp_threads_runtime = omp_get_num_threads();
        // if(num_omp_threads!=num_omp_threads_runtime) {
        //     fprintf(stderr, "[error] num_omp_threads=%f num_omp_threads_runtime=%d\n", num_omp_threads, num_omp_threads_runtime);
        //     // return -1;
        // }

        int const omp_thread_num = omp_get_thread_num();
        // if (omp_thread_num==0) {
        // my_float_t* state_data_host_2_split[num_gpus];
        // for(int i=0; i<num_gpus; i++) {
        state_data_host_2_split[omp_thread_num] = (my_float_t*)malloc(num_states_local_omp * sizeof(state_data_host_2_split[0][0]));
        // state_data_host_2_split[i] = &state_data_host_2[i * num_states_local];
        // }
        //}

        memcpy(state_data_host_2_split[omp_thread_num], state_data_host_2 + num_states_local_omp * omp_thread_num, num_states_local_omp * sizeof(*state_data_host_2));

        #pragma omp barrier

        time_cpu_begin[omp_thread_num] = std::chrono::high_resolution_clock::now();

        // cpu_gate<hadamard, 1>(0, num_qubits, target_qubit_num, &state_data_host_2);

        // for(int i=0; i<num_gpus; i++) {
        // state_data_host_2_split[i] = &state_data_host_2[i * num_states_local];
        cpu_gate<hadamard, num_omp_threads>(omp_thread_num, num_qubits, target_qubit_num, state_data_host_2_split);
        //}

        // auto end = std::chrono::high_resolution_clock::now();
        time_cpu_end[omp_thread_num] = std::chrono::high_resolution_clock::now();

        #pragma omp barrier

        memcpy(state_data_host_2 + num_states_local_omp * omp_thread_num, state_data_host_2_split[omp_thread_num], num_states_local_omp * sizeof(*state_data_host_2));

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