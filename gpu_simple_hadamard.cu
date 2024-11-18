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
// #include <cooperative_groups.h>

typedef double my_float_t;

/* https://stackoverflow.com/questions/8487986/file-macro-shows-full-path */
// #define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

my_float_t const sqrt2 = 1.41421356237309504880168872420969807856967187537694;
my_float_t const inv_sqrt2 = 1 / sqrt2;

// #define CHECK_CUDA(call) { \
//     cudaError_t err = call; \
//     if (err != cudaSuccess) \
//     { \
//         fprintf(stderr, "[error] %s:%d call=" #call " error=%s\n", __FILENAME__, __LINE__, cudaGetErrorString(err)); \
//         exit(1); \
//     } \
// }

// 任意のCUDA API関数とその引数を受け取る
template <typename Func, typename... Args>
void check_cuda(char const* const filename_abs, int const lineno, char const* const funcname, Func func, Args&&... args)
{
    auto const strrchr_result = strrchr(filename_abs, '/');
    auto const filename = strrchr_result? strrchr_result + 1 : filename_abs;
    // 引数を文字列化するためのostringstream
    std::ostringstream oss;
    // oss << funcName << "(";
    ((oss << args << ", "), ...);  // C++17の折り返し式を使って引数を順番に追加
    std::string args_str = oss.str();
    if (!args_str.empty()) {
        args_str.pop_back();  // 最後の", "を削除
        args_str.pop_back();
    }
    // argsStr += ")";

    // 実際にCUDA関数を呼び出し
    cudaError_t err = func(std::forward<Args>(args)...);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "[debug] %s:%d call:%s args:%s error:%s\n", filename, lineno, funcname, args_str.c_str(), cudaGetErrorString(err));
        // throw std::runtime_error(
        //     std::string("[error] %s:%d CUDA call: ") + funcName + 
        //     " failed with arguments: " + argsStr + 
        //     " with error: " + cudaGetErrorString(err)
        // );
    }
}

// マクロで簡単に呼び出せるようにラップ
#define CHECK_CUDA(func, ...) check_cuda(__FILE__, __LINE__, #func, func, __VA_ARGS__)

// 任意の型の引数を取る関数をラップするテンプレート
template <typename Func, typename Arg>
class Defer {
public:
    // コンストラクタで関数ポインタと引数を受け取る
    Defer(Func func, Arg&& arg)
        : func_(func), arg_(std::forward<Arg>(arg)) {}

    // デストラクタで関数ポインタを呼び出す
    ~Defer() {
        func_(arg_);
    }

private:
    Func func_;  // 関数ポインタ
    Arg arg_;    // 関数の引数
};

typedef struct /* hadamard_impl */ {
    static __device__ __host__ void apply(int64_t thread_num, int64_t const num_qubits, int64_t const target_qubit_num, my_float_t* const state_data) {
        int64_t const local_mask = (((int64_t)1)<<target_qubit_num) - (int64_t)1;

        int64_t const index_state_local = thread_num & local_mask;
        int64_t const index_state_global = (thread_num & ~local_mask) << ((int64_t)1);

        int64_t const index_state_0 = index_state_local | index_state_global;
        int64_t const index_state_1 = index_state_0 | (((int64_t)1)<<target_qubit_num);

        my_float_t const amp_state_0 = state_data[index_state_0];
        my_float_t const amp_state_1 = state_data[index_state_1];

        state_data[index_state_0] = (amp_state_0 + amp_state_1) * inv_sqrt2;
        state_data[index_state_1] = (amp_state_0 - amp_state_1) * inv_sqrt2;
    }
} hadamard_impl;

template<class Functor>
__global__ void gpu_gate(int64_t const num_qubits, int64_t const target_qubit_num, my_float_t* const state_data) {
    int64_t const thread_num = threadIdx.x + blockIdx.x * blockDim.x;
    Functor::apply(thread_num, num_qubits, target_qubit_num, state_data);
}

template<class Functor>
void cpu_gate(int64_t const num_qubits, int64_t const target_qubit_num, my_float_t* const state_data) {
    
    int64_t const num_threads = ((int64_t)1) << ((int64_t)(num_qubits-1));
    // #pragma omp parallel for 
    for(int64_t thread_num=0; thread_num<num_threads; thread_num++) {
        Functor::apply(thread_num, num_qubits, target_qubit_num, state_data);
    }
}

// bool areArraysEqualWithTolerance(double* ptr1, double* ptr2, size_t size, double tolerance) {
//     // カスタム比較関数：許容誤差以内で等しいかを比較
//     auto toleranceCompare = [tolerance](double a, double b) {
//         if(b != 0.0) {
//             return std::fabs(a / b - 1.0) <= tolerance;  // 誤差が許容範囲内であればtrue
//         } else {
//             return a == 0.0;
//         }
//         // return std::fabs(a-b) <= tolerance;  // 誤差が許容範囲内であればtrue
//     };
    
//     // std::equalにカスタム比較関数を渡す
//     return std::equal(ptr1, ptr1 + size, ptr2, toleranceCompare);
// }



int main() {

    setvbuf(stdout, NULL, _IOLBF, 1024 * 512);

    int const num_qubits = 16;
    int const log_block_size = 8;

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate, &stream);

    cudaEvent_t event_1;
    // fprintf(stderr, "[info] cudaEventCreateWithFlags =%p\n", &cudaEventCreateWithFlags );
    CHECK_CUDA(cudaEventCreateWithFlags, &event_1, cudaEventDefault);
    // CHECK_CUDA((cudaError_t(*)(cudaEvent_t*))cudaEventCreate, &event_1);
    // CHECK_CUDA(::cudaEventCreate, &event_1);

    cudaEvent_t event_2;
    CHECK_CUDA(cudaEventCreateWithFlags, &event_2, cudaEventDefault);
    // CHECK_CUDA((cudaError_t(*)(cudaEvent_t*))cudaEventCreate, &event_2);
    // CHECK_CUDA(cudaEventCreate, &event_2);


    int64_t const num_states = ((int64_t)1) << ((int64_t)num_qubits);
    int const block_size = 1 << log_block_size;
    int64_t const num_blocks = ((int64_t)1) << ((int64_t)(num_qubits-1-log_block_size));
    // int64_t const num_threads = ((int64_t)1) << ((int64_t)(num_qubits-1));

    my_float_t* state_data_host;
    // fprintf(stderr, "[info] cudaMallocHost=%p\n", &cudaMallocHost);
    // cudaError_t (*cudaMallocHostType)(void**, size_t);
    // CHECK_CUDA((cudaError_t (*)(void**, size_t))&cudaMallocHost, (void**)&state_data_host, num_states * sizeof(*state_data_host));
    CHECK_CUDA(cudaMallocHost<void>, (void**)&state_data_host, num_states * sizeof(*state_data_host), 0);
    // CHECK_CUDA([&state_data_host, num_states](){ return cudaMallocHost(&state_data_host, num_states * sizeof(*state_data_host), 0); });
    // state_data_host[0] = 1;
    // Defer<cudaError_t (*)(void*), void*> deferCudaFreeHost((cudaError_t (*)(void*))&cudaFreeHost, (void*)state_data_host);
    // Defer deferCudaFreeHost((cudaError_t (*)(void*))&cudaFreeHost, (void*)state_data_host);
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
    // CHECK_CUDA(cudaMallocHost, &state_data_host_2, num_states * sizeof(*state_data_host_2), 0);
    // CHECK_CUDA([&state_data_host_2, num_states](){ return cudaMallocHost(&state_data_host_2, num_states * sizeof(*state_data_host_2), 0); });
    // state_data_host_2[0] = 1;
    // CHECK_CUDA((cudaError_t (*)(void**, size_t))&cudaMallocHost, (void**)&state_data_host_2, num_states * sizeof(*state_data_host_2));
    CHECK_CUDA(cudaMallocHost<void>, (void**)&state_data_host_2, num_states * sizeof(*state_data_host_2), 0);
    Defer defer_free_state_data_host_2(cudaFreeHost, (void*)state_data_host_2);
    // CHECK_CUDA(cudaMallocHost<my_float_t>, &state_data_host_2, num_states * sizeof(*state_data_host_2));
    memcpy(state_data_host_2, state_data_host, num_states * sizeof(*state_data_host_2));

    my_float_t* state_data_device;
    // CHECK_CUDA(cudaMalloc, &state_data_device, num_states * sizeof(*state_data_device));
    // CHECK_CUDA([&state_data_device, num_states](){ return cudaMalloc(&state_data_device, num_states * sizeof(*state_data_device)); });
    // CHECK_CUDA((cudaError_t (*)(void**, size_t))&cudaMalloc, (void**)&state_data_device, num_states * sizeof(*state_data_device));
    CHECK_CUDA(cudaMalloc<void>, (void**)&state_data_device, num_states * sizeof(*state_data_device));
    Defer defer_free_state_data_device(cudaFree, (void*)state_data_device);

    CHECK_CUDA(cudaMemcpyAsync, state_data_device, state_data_host, num_states * sizeof(*state_data_device), cudaMemcpyHostToDevice, stream);

    fprintf(stderr, "[info] gpu_hadamard\n");

    CHECK_CUDA(cudaEventRecord, event_1, stream);

    gpu_gate<hadamard_impl><<<num_blocks, block_size, 0, stream>>>(num_qubits, 3, state_data_device);

    CHECK_CUDA(cudaEventRecord, event_2, stream);

    CHECK_CUDA(cudaMemcpyAsync, state_data_host, state_data_device, num_states * sizeof(*state_data_host), cudaMemcpyDeviceToHost, stream);

    fprintf(stderr, "[info] cpu_hadamard\n");
    cpu_gate<hadamard_impl>(num_qubits, 3, state_data_host_2);

    if (cudaSuccess != cudaStreamQuery(stream)) {
        fprintf(stderr, "[info] wait for GPU job completion...\n");
    }

    // CHECK_CUDA(cudaEventSynchronize(event_1));
    // CHECK_CUDA(cudaEventSynchronize(event_2));
    CHECK_CUDA(cudaStreamSynchronize, stream);

    double const elapsed = [](cudaEvent_t event_1, cudaEvent_t event_2) {
        float _elapsed_fp32;
        CHECK_CUDA(cudaEventElapsedTime, &_elapsed_fp32, event_1, event_2);
        return _elapsed_fp32 * 1e-3;
    }(event_1, event_2);

    fprintf(stderr, "[info] elapsed=%lf\n", elapsed);

    double const tolerance_list[] = {0.1, 0.01, 0.001, 0.0001};
    for(int i=0; i<sizeof(tolerance_list)/sizeof(*tolerance_list); i++) {
        double const tolerance = tolerance_list[i];
        bool const is_equal = std::equal(state_data_host, state_data_host + num_states, state_data_host_2, [tolerance](double a, double b) {
            if (b != 0.0) {
                return std::fabs(a / b - 1.0) <= tolerance;
                // 比率で比較
            } else {
                return a == 0.0;
            }
        });
        fprintf(stderr, "[info] tolerance=%lf is_equal=%d\n", tolerance_list[i], is_equal);
    }

    for(int i=0; i<num_states; i++) {
        fprintf(stdout, "%lld,%.16e,%.16e\n", i, state_data_host[i], state_data_host_2[i]);
    }

    // CHECK_CUDA(cudaFree, state_data_device);
    // CHECK_CUDA(cudaFreeHost, state_data_host);
    // CHECK_CUDA(cudaFreeHost, state_data_host_2);

    return 0;
}