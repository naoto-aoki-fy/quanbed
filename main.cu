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

#include <omp.h>
#include <cuda_runtime.h>
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

class hadamard { public:
    static __device__ __host__ void apply(int const num_split_areas, int const log_num_split_areas, int64_t const thread_num, int64_t const num_qubits, int64_t const target_qubit_num, my_complex_t** const state_data) {

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

        my_complex_t const amp_state_0 = state_data[index_state_0_split_num][index_state_0_split_address];
        my_complex_t const amp_state_1 = state_data[index_state_1_split_num][index_state_1_split_address];

        state_data[index_state_0_split_num][index_state_0_split_address] = (amp_state_0 + amp_state_1) * INV_SQRT2;
        state_data[index_state_1_split_num][index_state_1_split_address] = (amp_state_0 - amp_state_1) * INV_SQRT2;

    }

};

template<class Gate>
__global__ void cuda_gate(int const num_split_areas, int const log_num_split_areas, int64_t const split_num, int64_t const num_qubits, int64_t const target_qubit_num, my_complex_t* state_data[max_num_gpus]) {
    int64_t const num_qubits_local = num_qubits - log_num_split_areas;
    int64_t const num_threads_local = ((int64_t)1) << (num_qubits_local-1);

    int64_t const thread_num = threadIdx.x + blockIdx.x * blockDim.x + num_threads_local * split_num;
    Gate::apply(num_split_areas, log_num_split_areas, thread_num, num_qubits, target_qubit_num, state_data);
}

template<class Gate>
void omp_gate(int const num_split_areas, int const log_num_split_areas, int64_t const split_num, int64_t const num_qubits, int64_t const target_qubit_num, my_complex_t** const state_data) {
    int64_t const num_qubits_local = num_qubits - log_num_split_areas;
    int64_t const num_threads_local = ((int64_t)1) << (num_qubits_local-1);

    for(int64_t thread_num_local = 0; thread_num_local < num_threads_local; thread_num_local++) {
        int64_t thread_num = thread_num_local + num_threads_local * split_num;
        Gate::apply(num_split_areas, log_num_split_areas, thread_num, num_qubits, target_qubit_num, state_data);
    }
}

int main() {

    setvbuf(stdout, NULL, _IOLBF, 1024 * 512);

    int const num_samples = 128;
    int const rng_seed = 12345;
    // std::vector<int> gpu_list{0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<int> gpu_list{0, 1, 2, 3};
    // std::vector<int> gpu_list{0, 1};
    // std::vector<int> gpu_list{0};

    int const num_gpus = gpu_list.size();
    int const log_num_gpus = log2_int(num_gpus);

    int const omp_max_threads = omp_get_max_threads();
    int const log_num_omp_threads = log2_int(omp_max_threads);
    int const num_omp_threads = 1 << log_num_omp_threads;
    omp_set_num_threads(num_omp_threads);

    int const num_qubits = 33;
    int const log_block_size = 8;

    fprintf(stderr, "[info] num_qubits=%d\n", num_qubits);
    fprintf(stderr, "[info] num_gpus=%d (", num_gpus);
    for(int i=0; i<num_gpus; i++) {
        fprintf(stderr, "%d, ", gpu_list[i]);
    }
    fprintf(stderr, ")\n");
    fprintf(stderr, "[info] log_block_size=%d\n", log_block_size);

    fprintf(stderr, "[info] num_omp_threads=%d\n", num_omp_threads);

    std::vector<int> gpu_list_dedup;
    {
        std::unordered_set<int> gpu_set(gpu_list.begin(), gpu_list.end());
        gpu_list_dedup = {gpu_set.begin(), gpu_set.end()};
    }

    std::vector<cudaStream_t> stream(num_gpus);
    std::vector<cudaEvent_t> event_1(num_gpus);
    std::vector<cudaEvent_t> event_2(num_gpus);

    std::vector<decltype(Defer(cudaStreamDestroy, stream[0]))> defer_destroy_streams(num_gpus);
    std::vector<decltype(Defer(cudaEventDestroy, event_1[0]))> defer_destroy_event_1(num_gpus);
    std::vector<decltype(Defer(cudaEventDestroy, event_2[0]))> defer_destroy_event_2(num_gpus);

    for(int i=0; i<num_gpus; i++) {

        int const gpu_i = gpu_list[i]; 
        CHECK_CUDA(cudaSetDevice, gpu_i);

        CHECK_CUDA(cudaStreamCreate, &stream[i]);
        defer_destroy_streams[i] = {cudaStreamDestroy, stream[i]};

        CHECK_CUDA(cudaEventCreateWithFlags, &event_1[i], cudaEventDefault);
        defer_destroy_event_1[i] = {cudaEventDestroy, event_1[i]};

        CHECK_CUDA(cudaEventCreateWithFlags, &event_2[i], cudaEventDefault);
        defer_destroy_event_2[i] = {cudaEventDestroy, event_2[i]};

    }

    int64_t const num_states = ((int64_t)1) << ((int64_t)num_qubits);

    int const num_qubits_local = num_qubits - log_num_gpus;
    int64_t const num_states_local = ((int64_t)1) << ((int64_t)num_qubits_local);
    int const block_size = 1 << log_block_size;
    int64_t const num_blocks = ((int64_t)1) << ((int64_t)(num_qubits_local - 1 - log_block_size));

    int64_t const num_qubits_local_omp = num_qubits - log_num_omp_threads;
    int64_t const num_states_local_omp = ((int64_t)1) << num_qubits_local_omp;

    fprintf(stderr, "[info] cudaMallocHost state_data_host\n");
    my_complex_t* state_data_host;
    CHECK_CUDA(cudaMallocHost<void>, (void**)&state_data_host, num_states * sizeof(*state_data_host), 0);

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

        std::mt19937_64 mt(rng_seed + omp_thread_num);
        std::uniform_real_distribution<double> uni01(0.0,1.0);

        my_float_t sum_pow2_local = 0;
        for(int64_t state_num = num_states_local_omp * omp_thread_num;
            state_num < num_states_local_omp * (omp_thread_num + 1);
            state_num++) {
            my_float_t const amp_real = uni01(mt);
            my_float_t const amp_imag = uni01(mt);
            state_data_host[state_num] = {amp_real, amp_imag};
            sum_pow2_local += amp_real * amp_real + amp_imag * amp_imag;
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

    my_complex_t* const state_data_host_2 = (my_complex_t*)malloc(num_states * sizeof(*state_data_host_2));
    Defer defer_free_state_data_host_2(free, (void*)state_data_host_2);
    memcpy(state_data_host_2, state_data_host, num_states * sizeof(*state_data_host_2));

    // std::vector<my_float_t*> state_data_device_list(num_gpus);
    my_complex_t* state_data_device_list[max_num_gpus];
    std::vector<decltype(Defer(cudaFree, (void*)0))> defer_free_device_mem(num_gpus);

    for(int i=0; i<num_gpus; i++) {

        int const gpu_i = gpu_list[i]; 
        CHECK_CUDA(cudaSetDevice, gpu_i);

        my_complex_t* state_data_device;
        CHECK_CUDA(cudaMalloc<void>, (void**)&state_data_device, num_states_local * sizeof(*state_data_device));
        state_data_device_list[i] = state_data_device;

        CHECK_CUDA(cudaMemcpyAsync, state_data_device, &state_data_host[num_states_local * i], num_states_local * sizeof(*state_data_device), cudaMemcpyHostToDevice, stream[i]);
        defer_free_device_mem[i] = {cudaFree, (void*)state_data_device};

    }

    for(int i=0; i<gpu_list_dedup.size(); i++) {
        int const gpu_i = gpu_list_dedup[i]; 
        for(int j=0; j<gpu_list_dedup.size(); j++) {
            if(i==j) continue;
            int const gpu_j = gpu_list_dedup[j]; 
            if (gpu_i == gpu_j) continue;
            CHECK_CUDA(cudaSetDevice, gpu_i);
            CHECK_CUDA(cudaDeviceEnablePeerAccess, gpu_j, 0);
        }
    }

    for(int i=0; i<gpu_list_dedup.size(); i++) {
        int const gpu_i = gpu_list_dedup[i]; 
        for(int j=0; j<gpu_list_dedup.size(); j++) {
            if(i==j) continue;
            int const gpu_j = gpu_list_dedup[j]; 
            if (gpu_i == gpu_j) continue;
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

    fprintf(stderr, "[info] gpu_hadamard\n");
    fprintf(stdout, "cuda\n");

    for(int sample_num=0; sample_num < num_samples; ++sample_num) {

        for(int i=0; i<num_gpus; i++) {
            CHECK_CUDA(cudaEventRecord, event_1[i], stream[i]);
        }

        for(int target_qubit_num = 0; target_qubit_num < num_qubits; target_qubit_num++) {

            for(int i = 0; i < num_gpus; i++) {

                int const gpu_i = gpu_list[i]; 
                CHECK_CUDA(cudaSetDevice, gpu_i);

                cuda_gate<hadamard><<<num_blocks, block_size, 0, stream[i]>>>(num_gpus, log_num_gpus, i, num_qubits, target_qubit_num, &state_data_device_list[0]);

            }

            if (target_qubit_num >= num_qubits - log_num_gpus) {
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

    fprintf(stderr, "[info] cpu_hadamard\n");

    std::vector<my_complex_t*> state_data_host_2_split(num_omp_threads);
    std::vector<decltype(Defer(free, (void*)0))> defer_free_state_data_host_2_split(num_omp_threads);
    std::vector<std::chrono::_V2::system_clock::time_point>
        time_cpu_begin(num_omp_threads),
        time_cpu_end(num_omp_threads);

    fprintf(stdout, "omp\n");

    #pragma omp parallel
    {

        int const omp_thread_num = omp_get_thread_num();

        my_complex_t* const state_data_host_2_split_i = (my_complex_t*)malloc(num_states_local_omp * sizeof(*state_data_host_2_split_i));
        state_data_host_2_split[omp_thread_num] = state_data_host_2_split_i;
        defer_free_state_data_host_2_split[omp_thread_num] = {free, state_data_host_2_split_i};

        memcpy(state_data_host_2_split[omp_thread_num], state_data_host_2 + num_states_local_omp * omp_thread_num, num_states_local_omp * sizeof(*state_data_host_2));

        #pragma omp barrier

        for(int sample_num=0; sample_num < num_samples; ++sample_num) {

            time_cpu_begin[omp_thread_num] = std::chrono::high_resolution_clock::now();

            for(int target_qubit_num = 0; target_qubit_num < num_qubits; target_qubit_num++) {

                omp_gate<hadamard>(num_omp_threads, log_num_omp_threads, omp_thread_num, num_qubits, target_qubit_num, &state_data_host_2_split[0]);

                if (target_qubit_num >= num_qubits - log_num_omp_threads - 1) {
                    #pragma omp barrier
                }

            }

            time_cpu_end[omp_thread_num] = std::chrono::high_resolution_clock::now();

            #pragma omp master
            {
                double elapsed_cpu = 0;
                for(int i=0; i<num_omp_threads; i++) {
                    double const elapsed_cpu_i = std::chrono::duration<double>(time_cpu_end[i] - time_cpu_begin[i]).count();
                    if(elapsed_cpu_i>elapsed_cpu) {
                        elapsed_cpu = elapsed_cpu_i;
                    }
                }
                fprintf(stderr, "[info] elapsed_cpu=%lf\n", elapsed_cpu);
                fprintf(stdout, "%lf\n", elapsed_cpu);
            }

        }

        #pragma omp barrier

        memcpy(state_data_host_2 + num_states_local_omp * omp_thread_num, state_data_host_2_split[omp_thread_num], num_states_local_omp * sizeof(*state_data_host_2));

    }

    fprintf(stderr, "[info] transfer device data to host memory\n");
    for(int i=0; i<num_gpus; i++) {
        int const gpu_i = gpu_list[i]; 
        CHECK_CUDA(cudaSetDevice, gpu_i);

        CHECK_CUDA(cudaMemcpyAsync, &state_data_host[num_states_local * i], state_data_device_list[i], num_states_local * sizeof(*state_data_device_list[0]), cudaMemcpyDeviceToHost, stream[i]);

    }

    for(int i=0; i<num_gpus; i++) {
        CHECK_CUDA(cudaStreamSynchronize, stream[i]);
    }

    my_float_t max_diff = 0;
    for(int i=0; i<num_states; i++) {
        my_float_t diff = cuda::std::abs(state_data_host[i] - state_data_host_2[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    fprintf(stderr, "[info] max_diff=%.16e\n", max_diff);

    my_float_t max_rel_diff = 0;
    for(int i=0; i<num_states; i++) {
        my_complex_t const state_data_host_i = state_data_host[i];
        my_complex_t const state_data_host_2_i = state_data_host_2[i];
        if (state_data_host_i == 0.0 || state_data_host_2_i == 0.0) {
            continue;
        }
        my_float_t rel_diff = (state_data_host_i!=0.0)? cuda::std::abs(state_data_host_i / state_data_host_2_i - 1.0) : cuda::std::abs(state_data_host_2_i / state_data_host_i - 1.0);
        if (rel_diff > max_rel_diff) {
            max_rel_diff = rel_diff;
        }
    }
    fprintf(stderr, "[info] max_rel_diff=%.16e\n", max_rel_diff);

    return 0;

}