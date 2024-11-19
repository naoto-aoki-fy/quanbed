// #include <iostream>
#include <cstdlib>
#include <cstdio>
#include <stdint.h>
#include <stdexcept>
#include <string>
#include <iostream>
#include <sstream>
#include <string.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <cmath>

#define ARRAY_SIZE(array) (sizeof(array)/sizeof(*array))

template<typename INT, int arg> class log2_compile_time { public: static INT const value = 1 + log2_compile_time<INT, arg/2>::value; };
template<typename INT> class log2_compile_time<INT, 1> { public: static INT const value = 0; };

// int const gpu_list[] = {0, 1, 2, 3, 4, 5, 6, 7};
// int const gpu_list[] = {0, 1, 2, 3};
// int const gpu_list[] = {0};
// int const gpu_list[] = {0, 0};
// int const gpu_list[] = {0, 0, 0, 0};
// int const gpu_list[] = {0, 0, 0, 0, 0, 0, 0, 0};
int const gpu_list[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

// int const num_gpus = 1 << log_num_gpus;
int const num_gpus = ARRAY_SIZE(gpu_list);
int const log_num_gpus = log2_compile_time<int, num_gpus>::value;

int const num_qubits = 16;
int const log_block_size = 8;

typedef double my_float_t;

my_float_t const sqrt2 = 1.41421356237309504880168872420969807856967187537694;
my_float_t const inv_sqrt2 = 1 / sqrt2;

// 任意のCUDA API関数とその引数を受け取る
template <typename Func, typename... Args>
void check_cuda(char const* const filename_abs, int const lineno, char const* const funcname, Func func, Args&&... args)
{
    auto const strrchr_result = strrchr(filename_abs, '/');
    auto const filename = strrchr_result? strrchr_result + 1 : filename_abs;
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

typedef struct /* hadamard_impl */ {
    static __device__ __host__ void apply(int64_t thread_num, int64_t const num_qubits, int64_t const target_qubit_num, my_float_t* const state_data[num_gpus]) {

        int64_t const lower_mask = (((int64_t)1)<<target_qubit_num) - (int64_t)1;
        int64_t const local_mask = (((int64_t)1)<<((int64_t)(num_qubits - log_num_gpus))) - (int64_t)1;

        int64_t const index_state_local = thread_num & lower_mask;
        int64_t const index_state_global = (thread_num & ~lower_mask) << ((int64_t)1);

        int64_t const index_state_0 = index_state_local | index_state_global;
        int64_t const index_state_1 = index_state_0 | (((int64_t)1)<<target_qubit_num);

        int64_t const index_state_0_gpu_num = index_state_0 >> (num_qubits - log_num_gpus);
        int64_t const index_state_0_address = index_state_0 & local_mask;

        int64_t const index_state_1_gpu_num = index_state_1 >> (num_qubits - log_num_gpus);
        int64_t const index_state_1_address = index_state_1 & local_mask;

        my_float_t const amp_state_0 = state_data[index_state_0_gpu_num][index_state_0_address];
        my_float_t const amp_state_1 = state_data[index_state_1_gpu_num][index_state_1_address];

        state_data[index_state_0_gpu_num][index_state_0_address] = (amp_state_0 + amp_state_1) * inv_sqrt2;
        state_data[index_state_1_gpu_num][index_state_1_address] = (amp_state_0 - amp_state_1) * inv_sqrt2;
    }
} hadamard_impl;

template<class Functor>
__global__ void gpu_gate(int64_t gpu_num, int64_t const num_qubits, int64_t const target_qubit_num, my_float_t* const state_data[num_gpus]) {
    int64_t const num_qubits_local = num_qubits - log_num_gpus;
    int64_t const num_threads_local = ((int64_t)1) << (num_qubits_local-1);

    int64_t const thread_num = threadIdx.x + blockIdx.x * blockDim.x + num_threads_local * gpu_num;
    Functor::apply(thread_num, num_qubits, target_qubit_num, state_data);
}

// template<class Functor>
// void cpu_gate(int64_t const num_qubits, int64_t const target_qubit_num, my_float_t* const state_data) {
    
//     int64_t const num_threads = ((int64_t)1) << ((int64_t)(num_qubits-1));
//     // #pragma omp parallel for 
//     for(int64_t thread_num=0; thread_num<num_threads; thread_num++) {
//         Functor::apply(thread_num, num_qubits, target_qubit_num, state_data);
//     }
// }

int main() {

    setvbuf(stdout, NULL, _IOLBF, 1024 * 512);

    // int const dev_list[] = {0, 1, 2, 3};
    // int const num_gpus = ARRAY_SIZE(dev_list);
    // int const log_num_gpus = log2(num_gpus);

    for(int i=0; i<num_gpus; i++) {
        int const gpu_i = gpu_list[i]; 
        CHECK_CUDA(cudaSetDevice, gpu_i);

        for(int j=0; j<num_gpus; j++) {
            int const gpu_j = gpu_list[j]; 
            if (gpu_i == gpu_j) continue;
            CHECK_CUDA(cudaDeviceEnablePeerAccess, gpu_j, 0);
        }
    }

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate, &stream);

    cudaEvent_t event_1;
    CHECK_CUDA(cudaEventCreateWithFlags, &event_1, cudaEventDefault);
    // CHECK_CUDA((cudaError_t(*)(cudaEvent_t*))cudaEventCreate, &event_1);

    cudaEvent_t event_2;
    CHECK_CUDA(cudaEventCreateWithFlags, &event_2, cudaEventDefault);
    // CHECK_CUDA((cudaError_t(*)(cudaEvent_t*))cudaEventCreate, &event_2);

    int64_t const num_states = ((int64_t)1) << ((int64_t)num_qubits);

    int64_t const num_qubits_local = num_qubits - log_num_gpus;
    int64_t const num_states_local = ((int64_t)1) << num_qubits_local;
    int const block_size = 1 << log_block_size;
    int64_t const num_blocks = ((int64_t)1) << ((int64_t)(num_qubits_local-1-log_block_size));
    // int64_t const num_threads = ((int64_t)1) << ((int64_t)(num_qubits-1));

    my_float_t* state_data_host;
    // CHECK_CUDA((cudaError_t (*)(void**, size_t))&cudaMallocHost, (void**)&state_data_host, num_states * sizeof(*state_data_host));
    CHECK_CUDA(cudaMallocHost<void>, (void**)&state_data_host, num_states * sizeof(*state_data_host), 0);
    // CHECK_CUDA(cudaMallocHost<void>, &state_data_host, num_states * sizeof(*state_data_host), 0);
    // CHECK_CUDA(cudaMallocHost<my_float_t>, &state_data_host, num_states * sizeof(*state_data_host), 0);
    // CHECK_CUDA([&state_data_host, num_states](){ return cudaMallocHost(&state_data_host, num_states * sizeof(*state_data_host), 0); });

    Defer defer_free_state_data_host(cudaFreeHost, (void*)state_data_host);

    fprintf(stderr, "[info] generating random state\n");

    {
        double sum_pow2 = 0;
        for(int state_num=0; state_num<num_states; state_num++) {
            my_float_t const amp = static_cast<my_float_t>(rand()) / RAND_MAX;
            state_data_host[state_num] = amp;
            sum_pow2 += amp * amp;
        }
        for(int state_num=0; state_num<num_states; state_num++) {
            state_data_host[state_num] /= sum_pow2;
        }
    }

    my_float_t* state_data_host_2;
    // CHECK_CUDA((cudaError_t (*)(void**, size_t))&cudaMallocHost, (void**)&state_data_host_2, num_states * sizeof(*state_data_host_2));
    CHECK_CUDA(cudaMallocHost<void>, (void**)&state_data_host_2, num_states * sizeof(*state_data_host_2), 0);
    Defer defer_free_state_data_host_2(cudaFreeHost, (void*)state_data_host_2);
    memcpy(state_data_host_2, state_data_host, num_states * sizeof(*state_data_host_2));

    my_float_t* state_data_device_list[num_gpus];
    // my_array<my_float_t*, num_gpus> state_data_device_list;
    std::vector<decltype(Defer(cudaFree, (void*)0))> defer_free_device_mem;

    for(int i=0; i<num_gpus; i++) {

        int const gpu_i = gpu_list[i]; 
        CHECK_CUDA(cudaSetDevice, gpu_i);

        // CHECK_CUDA((cudaError_t (*)(void**, size_t))&cudaMalloc, (void**)&state_data_device, num_states * sizeof(*state_data_device));
        my_float_t* state_data_device;
        // CHECK_CUDA(cudaMalloc<void>, (void**)&state_data_device, num_states_local * sizeof(*state_data_device));
        CHECK_CUDA(cudaMalloc<void>, (void**)&state_data_device, num_states * sizeof(*state_data_device));
        state_data_device_list[i] = state_data_device;

        CHECK_CUDA(cudaMemcpyAsync, state_data_device, state_data_host + num_states_local * i, num_states_local * sizeof(*state_data_device), cudaMemcpyHostToDevice, stream);
        defer_free_device_mem.push_back({cudaFree, (void*)state_data_device});
    }
    // Defer defer_free_state_data_device(cudaFree, (void*)state_data_device);

    fprintf(stderr, "[info] gpu_hadamard\n");

    CHECK_CUDA(cudaEventRecord, event_1, stream);

    for(int i=0; i<num_gpus; i++) {

        int const gpu_i = gpu_list[i]; 
        CHECK_CUDA(cudaSetDevice, gpu_i);

        gpu_gate<hadamard_impl><<<num_blocks, block_size, 0, stream>>>(i, num_qubits, 3, state_data_device_list);
        // gpu_gate<hadamard_impl><<<1, 1, 0, stream>>>(i, num_qubits, 3, state_data_device_list);

    }

    CHECK_CUDA(cudaEventRecord, event_2, stream);

    CHECK_CUDA(cudaStreamSynchronize, stream);

    // CHECK_CUDA(cudaMemcpyAsync, state_data_host, state_data_device, num_states * sizeof(*state_data_host), cudaMemcpyDeviceToHost, stream);
    for(int i=0; i<num_gpus; i++) {
        int const gpu_i = gpu_list[i]; 
        CHECK_CUDA(cudaSetDevice, gpu_i);

        CHECK_CUDA(cudaMemcpyAsync, state_data_host + num_states_local * i, state_data_device_list[i], num_states_local * sizeof(*state_data_device_list[0]), cudaMemcpyDeviceToHost, stream);
    }

    // fprintf(stderr, "[info] cpu_hadamard\n");
    // cpu_gate<hadamard_impl>(num_qubits, 3, state_data_host_2);

    if (cudaSuccess != cudaStreamQuery(stream)) {
        fprintf(stderr, "[info] wait for GPU job completion...\n");
    }

    CHECK_CUDA(cudaStreamSynchronize, stream);

    double const elapsed = [](cudaEvent_t event_1, cudaEvent_t event_2) {
        float _elapsed_fp32;
        CHECK_CUDA(cudaEventElapsedTime, &_elapsed_fp32, event_1, event_2);
        return _elapsed_fp32 * 1e-3;
    } (event_1, event_2);

    fprintf(stderr, "[info] elapsed=%lf\n", elapsed);

    // my_float_t max_diff = 0;
    // for(int i=0; i<num_states; i++) {
    //     my_float_t diff = std::fabs(state_data_host[i]-state_data_host_2[i]);
    //     if(diff>max_diff) {
    //         max_diff = diff;
    //     }
    // }
    // fprintf(stderr, "[info] max_diff=%.16e\n", max_diff);

    for(int i=0; i<num_states; i++) {
        // fprintf(stdout, "%lld,%.16e,%.16e\n", i, state_data_host[i], state_data_host_2[i]);
        fprintf(stdout, "%lld,%.16e\n", i, state_data_host[i]);
    }

    return 0;
}