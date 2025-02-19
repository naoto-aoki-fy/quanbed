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
#include <string_view>
#include <algorithm>
#include <string>
#include <unordered_set>
#include <tuple>

#include <openssl/evp.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cuda/std/complex>
#include <nccl.h>

#define SQRT2 (1.41421356237309504880168872420969807856967187537694)
#define INV_SQRT2 (1.0/SQRT2)

unsigned int log2_int(unsigned int arg) {
    return sizeof(unsigned int) * CHAR_BIT - __builtin_clz(arg) - 1;
}
unsigned int log2_int(int arg) {
    return log2_int((unsigned int)arg);
}

#if UINT_MAX != ULONG_MAX
// #if sizeof(unsigned int) != sizeof(unsigned long)
unsigned int log2_int(unsigned long arg) {
    return sizeof(unsigned long) * CHAR_BIT - __builtin_clzl(arg) - 1;
}
unsigned int log2_int(long arg) {
    return log2_int((unsigned long)arg);
}
#endif

#if ULONG_MAX != ULLONG_MAX
// #if sizeof(unsigned long) != sizeof(unsigned long long)
unsigned int log2_int(unsigned long long arg) {
    return sizeof(unsigned long long) * CHAR_BIT - __builtin_clzll(arg) - 1;
}
unsigned int log2_int(long long arg) {
    return log2_int((unsigned long long)arg);
}
#endif

typedef double my_float_t;
typedef cuda::std::complex<my_float_t> my_complex_t;

constexpr const char* get_filename(const char* filename_abs) {
    size_t const pos = std::string_view(filename_abs).rfind("/");
    return (pos != std::string_view::npos) ? &filename_abs[pos+1] : filename_abs;
}

template <typename Func>
void check_cuda(char const* const filename, int const lineno, char const* const funcname, Func func)
{
    auto err = func();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "[debug] %s:%d call:%s error:%s\n", filename, lineno, funcname, cudaGetErrorString(err));
        exit(1);
    }
}

#define CHECK_CUDA(func, ...) check_cuda(get_filename(__FILE__), __LINE__, #func "(" #__VA_ARGS__ ")", [&](){return func(__VA_ARGS__);})

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

template <typename Func>
void check_curand(char const* const filename, int const lineno, char const* const funcname, Func func)
{
    auto err = func();
    if (err != CURAND_STATUS_SUCCESS)
    {
        fprintf(stderr, "[debug] %s:%d call:%s error:%s\n", filename, lineno, funcname, curandGetErrorString(err));
        exit(1);
    }
}

#define CHECK_CURAND(func, ...) check_curand(get_filename(__FILE__), __LINE__, #func "(" #__VA_ARGS__ ")", [&](){return func(__VA_ARGS__);})

template <typename Func>
void check_nccl(char const* const filename, int const lineno, char const* const funcname, Func func)
{
    auto err = func();
    if (err != ncclSuccess)
    {
        fprintf(stderr, "[debug] %s:%d call:%s error:%s\n", filename, lineno, funcname, ncclGetErrorString(err));
        exit(1);
    }
}

#define CHECK_NCCL(func, ...) check_nccl(get_filename(__FILE__), __LINE__, #func "(" #__VA_ARGS__ ")", [&](){return func(__VA_ARGS__);})

template <typename Func>
class Defer {
public:
    Defer(Func func) : func_(func) {}
    ~Defer() { this->func_(); }
private:
    Func func_;
};

#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a ## b
#define UNIQUE_NAME(base) CONCAT(base, __LINE__)

#define DEFER_FUNC(func, ...) Defer UNIQUE_NAME(defer_)([&](){ func(__VA_ARGS__); })

#define DEFER_CHECK_CUDA(func, ...) Defer UNIQUE_NAME(defer_)([&](){ CHECK_CUDA(func, __VA_ARGS__);})

#define DEFER_CODE(code) Defer UNIQUE_NAME(defer_)([&]()code)

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
        state_data_device[index_state_1] = (amp_state_0 - amp_state_1) * INV_SQRT2;

    }
};

template<class Gate>
__global__ void cuda_gate(int64_t const num_qubits, int64_t const target_qubit_num, my_complex_t* const state_data_device) {
    int64_t const thread_num = threadIdx.x + blockIdx.x * blockDim.x;
    Gate::apply(thread_num, num_qubits, target_qubit_num, state_data_device);
}



auto group_by_host(int const rank, int const size) {

    // 各プロセスでホスト名を取得
    char hostname[MPI_MAX_PROCESSOR_NAME];
    int nameLen;
    MPI_Get_processor_name(hostname, &nameLen);
    std::string my_hostname(hostname, nameLen);

    // rank 0で各プロセスのホスト名を受け取るためのバッファ（固定長）
    std::vector<char> gatheredBuffer;
    if (rank == 0) {
        gatheredBuffer.resize(size * MPI_MAX_PROCESSOR_NAME, '\0');
    }

    // 各プロセスのホスト名をrank 0に集約（固定長文字列）
    MPI_Gather(hostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
               (rank == 0 ? gatheredBuffer.data() : nullptr),
               MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
               0, MPI_COMM_WORLD);

    // rank 0側で重複排除とノード番号の付与、さらにノード内のローカルランクを計算する
    std::vector<int> node_numbers; // 各プロセスが所属するノード番号
    std::vector<int> node_local_ranks; // 同一ノード内でのプロセス順（0から開始）
    int node_count = 0;
    if (rank == 0) {
        // 集約された固定長文字列から std::vector<std::string> を作成
        std::vector<std::string> hostnames;
        hostnames.reserve(size);
        for (int i = 0; i < size; i++) {
            const char* ptr = gatheredBuffer.data() + i * MPI_MAX_PROCESSOR_NAME;
            hostnames.push_back(std::string(ptr));
        }

        // ノード番号の付与（重複排除）
        std::unordered_set<std::string> uniqueSet;
        std::vector<std::string> uniqueHosts;
        node_numbers.resize(size, -1);
        for (int i = 0; i < size; i++) {
            const std::string& host = hostnames[i];
            if (uniqueSet.find(host) == uniqueSet.end()) {
                uniqueSet.insert(host);
                uniqueHosts.push_back(host);
                node_numbers[i] = static_cast<int>(uniqueHosts.size()) - 1;
            } else {
                auto it = std::find(uniqueHosts.begin(), uniqueHosts.end(), host);
                node_numbers[i] = static_cast<int>(std::distance(uniqueHosts.begin(), it));
            }
        }
        node_count = static_cast<int>(uniqueHosts.size());

        // 同一ノード内でのプロセス順（ローカルランク）の計算
        node_local_ranks.resize(size, -1);
        std::unordered_map<std::string, int> countMap;
        for (int i = 0; i < size; i++) {
            // 現在のホスト名の出現回数が、そのプロセスのローカルランクになる
            node_local_ranks[i] = countMap[hostnames[i]];
            countMap[hostnames[i]]++;
        }
    }

    // rank 0で決定したノード数を全プロセスにブロードキャスト
    MPI_Bcast(&node_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 各プロセスに自分のノード番号とノード内のローカルランクを通知（scatter）
    int my_node_number = -1;
    int my_node_local_rank = -1;
    MPI_Scatter((rank == 0 ? node_numbers.data() : nullptr), 1, MPI_INT,
                &my_node_number, 1, MPI_INT,
                0, MPI_COMM_WORLD);
    MPI_Scatter((rank == 0 ? node_local_ranks.data() : nullptr), 1, MPI_INT,
                &my_node_local_rank, 1, MPI_INT,
                0, MPI_COMM_WORLD);

    // 結果をstderrに出力
    // fprintf(stderr,
    //         "Rank %d on host %s -> assigned node number: %d, local node rank: %d (total nodes: %d)\n",
    //         rank, my_hostname.c_str(), my_node_number, my_node_local_rank, node_count);

    return std::make_tuple(my_hostname, my_node_number, my_node_local_rank, node_count);

}


int main(int argc, char** argv) {

    // **注意**: normalize_factorが並列方法によって若干計算結果に違いがあるので、ノーマライズしてしまうと、チェックサムが一致しなくなる
    // **Note**: The `normalize_factor` may cause slight differences in calculation results due to parallel processing methods. As a result, normalization can lead to a mismatch in the checksum.
    bool const do_normalization = false;
    bool const calc_checksum = false;
    int const num_rand_areas = 1;
    bool const use_unified_memory = true;

    float elapsed_ms, elapsed_ms_2;

    setvbuf(stdout, NULL, _IOLBF, 1024 * 512);

    MPI_Init(&argc, &argv);

    int num_procs, proc_num;

    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_num);

    if (proc_num==0) {
        fprintf(stderr, "[info] num_procs=%d\n", num_procs);
    }

    /* ==== begin local rank ==== */
    auto [my_hostname, my_node_number, my_node_local_rank, node_count] = group_by_host(proc_num, num_procs);
    fprintf(stderr,
            "[debug] Rank %d on host %s -> assigned node number: %d, local node rank: %d (total nodes: %d)\n",
            proc_num, my_hostname.c_str(), my_node_number, my_node_local_rank, node_count);
    // MPI_Finalize();
    // return 0;
    /* ==== end local rank ==== */

    // int const gpu_id = proc_num;
    int const gpu_id = my_node_local_rank;
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

    int const num_qubits = 29;
    if (proc_num == 0) { fprintf(stderr, "[info] num_qubits=%d\n", num_qubits); }

    std::vector<int> perm_p2l(num_qubits);
    std::vector<int> perm_l2p(num_qubits);

    for(int qubit_num=0; qubit_num<num_qubits; qubit_num++) {
        perm_p2l[qubit_num] = qubit_num;
        perm_l2p[qubit_num] = qubit_num;
    }

    int const num_samples = 64;
    int const rng_seed = 12345;

    int const log_num_procs = log2_int(num_procs);

    int const log_block_size = 8;
    int const target_qubit_num_begin = 0;
    // int const target_qubit_num_end = 0;
    int const target_qubit_num_end = num_qubits;

    if (proc_num == 0) { fprintf(stderr, "[info] log_block_size=%d\n", log_block_size); }

    cudaStream_t stream;
    cudaEvent_t event_1;
    cudaEvent_t event_2;

    CHECK_CUDA(cudaStreamCreate, &stream);
    DEFER_CHECK_CUDA(cudaStreamDestroy, stream);

    CHECK_CUDA(cudaEventCreateWithFlags, &event_1, cudaEventDefault);
    DEFER_CHECK_CUDA(cudaEventDestroy, event_1);

    CHECK_CUDA(cudaEventCreateWithFlags, &event_2, cudaEventDefault);
    DEFER_CHECK_CUDA(cudaEventDestroy, event_2);

    int64_t const num_states = ((int64_t)1) << ((int64_t)num_qubits);

    int const num_qubits_local = num_qubits - log_num_procs;
    
    int64_t const num_states_local = ((int64_t)1) << ((int64_t)num_qubits_local);
    int const block_size = 1 << log_block_size;
    int64_t const num_blocks = ((int64_t)1) << ((int64_t)(num_qubits_local - 1 - log_block_size));

    if (proc_num == 0) { fprintf(stderr, "[info] malloc device memory\n"); }

    my_complex_t* state_data_device;
    if (use_unified_memory) {
        CHECK_CUDA(cudaMallocManaged, &state_data_device, num_states_local * sizeof(*state_data_device));
        CHECK_CUDA(cudaMemAdvise, state_data_device, num_states_local * sizeof(*state_data_device), cudaMemAdviseSetPreferredLocation, gpu_id);
    } else {
        CHECK_CUDA(cudaMalloc, &state_data_device, num_states_local * sizeof(*state_data_device));
    }
    DEFER_CHECK_CUDA(cudaFree, state_data_device);

    int const log_swap_buffer_total_length = (num_qubits_local>30)? num_qubits_local - 3 : num_qubits_local;
    // int const log_swap_buffer_total_length = num_qubits_local;
    uint64_t const swap_buffer_total_length = UINT64_C(1) << log_swap_buffer_total_length;
    my_complex_t* swap_buffer;
    CHECK_CUDA(cudaMalloc, &swap_buffer, swap_buffer_total_length * sizeof(my_complex_t));
    // CHECK_CUDA(cudaMallocManaged, &swap_buffer, swap_buffer_total_length * sizeof(my_complex_t));
    DEFER_CHECK_CUDA(cudaFree, swap_buffer);

    my_float_t* norm_sum_device;
    CHECK_CUDA(cudaMalloc, &norm_sum_device, (num_states_local>>log_block_size) * sizeof(my_float_t));
    // DEFER_CHECK_CUDA(cudaFree, norm_sum_device);

    if (proc_num == 0) { fprintf(stderr, "[info] generating random state\n"); }
    curandGenerator_t rng_device;

    // CHECK_CURAND(curandCreateGenerator, &rng_device, CURAND_RNG_PSEUDO_DEFAULT);
    // CHECK_CURAND(curandSetStream, rng_device, stream);
    // CHECK_CURAND(curandSetPseudoRandomGeneratorSeed, rng_device, rng_seed + proc_num);

    CHECK_CUDA(cudaEventRecord, event_1, stream);

    // if (false) {
    //     CHECK_CURAND(curandCreateGenerator, &rng_device, CURAND_RNG_PSEUDO_DEFAULT);
    //     CHECK_CURAND(curandSetStream, rng_device, stream);
    //     CHECK_CURAND(curandSetPseudoRandomGeneratorSeed, rng_device, rng_seed + proc_num);
    //     CHECK_CURAND(curandGenerateNormalDouble, rng_device, (my_float_t*)(void*)state_data_device, num_states_local * 2 /* complex */, 0.0, 1.0);
    //     CHECK_CURAND(curandDestroyGenerator, rng_device);
    // } else
    {
        // int const num_rand_areas = 4;
        int const log_num_rand_areas = log2_int(num_rand_areas);
        // if (log_num_rand_areas!=1) { throw; }
        uint64_t const num_states_rand_area = num_states_local >> log_num_rand_areas;
        for (int rand_area_num = 0; rand_area_num < num_rand_areas; rand_area_num++) {
            CHECK_CURAND(curandCreateGenerator, &rng_device, CURAND_RNG_PSEUDO_DEFAULT);
            CHECK_CURAND(curandSetStream, rng_device, stream);
            CHECK_CURAND(curandSetPseudoRandomGeneratorSeed, rng_device, rng_seed + proc_num * num_rand_areas + rand_area_num);
            CHECK_CURAND(curandGenerateNormalDouble, rng_device, (my_float_t*)(void*)(state_data_device + num_states_rand_area * ((uint64_t)rand_area_num)), num_states_rand_area * 2 /* complex */, 0.0, 1.0);
            CHECK_CURAND(curandDestroyGenerator, rng_device);
        }
    }
    // CHECK_CURAND(curandSetPseudoRandomGeneratorSeed, rng_device, rng_seed + proc_num * 2);
    // CHECK_CURAND(curandGenerateNormalDouble, rng_device, (my_float_t*)(void*)state_data_device, num_states_local, 0.0, 1.0);

    // curandGenerator_t rng_device_2;

    // CHECK_CURAND(curandCreateGenerator, &rng_device_2, CURAND_RNG_PSEUDO_DEFAULT);
    // CHECK_CURAND(curandSetStream, rng_device_2, stream);
    // CHECK_CURAND(curandSetPseudoRandomGeneratorSeed, rng_device_2, rng_seed + proc_num * 2 + 1);

    // CHECK_CURAND(curandGenerateNormalDouble, rng_device_2, &((my_float_t*)(void*)state_data_device)[num_states_local], num_states_local, 0.0, 1.0);

    if (do_normalization) {

        if (proc_num == 0) { fprintf(stderr, "[info] gpu reduce\n"); } 
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
            fprintf(stderr, "[info] normalize done\n");
        }

    }

    if(proc_num==0) {
        fprintf(stderr, "[info] gpu_hadamard\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for(int sample_num=0; sample_num < num_samples; ++sample_num) {

        CHECK_CUDA(cudaEventRecord, event_1, stream);

        for(int target_qubit_num_logical = target_qubit_num_begin; target_qubit_num_logical < target_qubit_num_end; target_qubit_num_logical++) {

            int target_qubit_num_physical = perm_l2p[target_qubit_num_logical];
            // if(proc_num==0) fprintf(stderr, "[debug] target_qubit_num_logical=%d target_qubit_num_physical=%d\n", target_qubit_num_logical, target_qubit_num_physical);
            // MPI_Barrier(MPI_COMM_WORLD);

            /* target qubits is global */
            if (target_qubit_num_physical >= num_qubits_local) {

                int const* const swap_target_global_list = &target_qubit_num_physical;
                int const swap_target_local = num_qubits - log_num_procs - 1;
                int const* const swap_target_local_list = &swap_target_local;
                int const num_targets = 1;

                // b_min
                int const swap_target_local_min = *std::min_element(swap_target_local_list, swap_target_local_list + num_targets);

                uint64_t const local_buf_length = UINT64_C(1) << swap_target_local_min;
                uint64_t swap_buffer_length = swap_buffer_total_length;
                if (swap_buffer_length > local_buf_length) {
                    swap_buffer_length = local_buf_length;
                }

                // generate a mask for generating global_nonswap_self
                uint64_t global_swap_self_mask = 0;
                for (int target_num = 0; target_num < num_targets; target_num++) {
                    // a_delta = a – n_local
                    int const swap_target_global_delta = swap_target_global_list[target_num] - num_qubits_local;
                    global_swap_self_mask |= (UINT64_C(1) << swap_target_global_delta);
                }

                // global_nonswap_self = make proc_num_self's a_delta_i-th digit zero
                uint64_t const global_nonswap_self = proc_num & ~global_swap_self_mask;

                // 1<<(num_local_qubits - b_min) 
                uint64_t const num_local_areas = UINT64_C(1) << (num_qubits_local - swap_target_local_min);
                for (uint64_t local_num_self = 0; local_num_self < num_local_areas; local_num_self++) {

                    // global_swap_peer = OR_i (local_num_selfのb_delta_i桁目)をa_delta_i桁目にする
                    uint64_t global_swap_peer = 0;
                    for (int target_num = 0; target_num < num_targets; target_num++) {
                        // a_delta_i
                        int const swap_target_global_delta = swap_target_global_list[target_num] - num_qubits_local;
                        // b_delta_i
                        int const swap_target_local_delta = swap_target_local_list[target_num] - swap_target_local_min;
                        global_swap_peer |=
                            // local_num_selfのb_delta_i桁目
                            ((local_num_self >> swap_target_local_delta) & 1)
                            // をa_delta_i桁目にする
                            << swap_target_global_delta;
                        
                    }

                    uint64_t const proc_num_peer = global_swap_peer | global_nonswap_self;

                    // send & recv
                    if (proc_num_peer == proc_num) { continue; }
                    // CHECK_NCCL(ncclSend, &state_data_device[local_num_self * local_buf_length], local_buf_length, ncclDouble, proc_num_peer, nccl_comm, stream);
                    // CHECK_NCCL(ncclRecv, &state_data_device[local_num_self * local_buf_length], local_buf_length, ncclDouble, proc_num_peer, nccl_comm, stream);
                    bool is_peer_greater = proc_num_peer > proc_num;
                    for (uint64_t buffer_pos = 0; buffer_pos < local_buf_length; buffer_pos += swap_buffer_length) {
                        CHECK_NCCL(ncclGroupStart);
                        for (int send_recv = 0; send_recv < 2; send_recv++) {
                            if (send_recv ^ is_peer_greater) {
                                CHECK_NCCL(ncclSend, &state_data_device[local_num_self * local_buf_length + buffer_pos], swap_buffer_length * 2 /* complex */, ncclDouble, proc_num_peer, nccl_comm, stream);
                            } else {
                                CHECK_NCCL(ncclRecv, swap_buffer, swap_buffer_length * 2 /* complex */, ncclDouble, proc_num_peer, nccl_comm, stream);
                            }
                        }
                        CHECK_NCCL(ncclGroupEnd);
                        CHECK_CUDA(cudaMemcpyAsync, &state_data_device[local_num_self * local_buf_length + buffer_pos], swap_buffer, swap_buffer_length * sizeof(my_complex_t), cudaMemcpyDeviceToDevice, stream);
                    }

                }

                // swap_target_global_logical_list[:] = perm_p2l[swap_target_global_list[:]]
                // swap_target_local_logical_list[:] = perm_p2l[swap_target_local_list[:]]
                std::vector<int> swap_target_local_logical_list(num_targets);
                std::vector<int> swap_target_global_logical_list(num_targets);
                for (int target_num = 0; target_num < num_targets; target_num++) {
                    swap_target_local_logical_list[target_num] = perm_p2l[swap_target_local_list[target_num]];
                    swap_target_global_logical_list[target_num] = perm_p2l[swap_target_global_list[target_num]];
                }

                // update p2l & l2p
                // perm_p2l[swap_target_global_list[:]] = swap_target_local_logical_list[:]
                // perm_p2l[swap_target_local_list[:]] = swap_target_global_logical_list[:]
                // perm_l2p[swap_target_global_logical_list[:]] = swap_target_local_list[:]
                // perm_l2p[swap_target_local_logical_list[:]] = swap_target_global_list[:]

                for (int target_num = 0; target_num < num_targets; target_num++) {
                    perm_p2l[swap_target_global_list[target_num]] = swap_target_local_logical_list[target_num];
                    perm_p2l[swap_target_local_list[target_num]] = swap_target_global_logical_list[target_num];
                    perm_l2p[swap_target_global_logical_list[target_num]] = swap_target_local_list[target_num];
                    perm_l2p[swap_target_local_logical_list[target_num]] = swap_target_global_list[target_num];
                }

                target_qubit_num_physical = swap_target_local;

            }

            cuda_gate<hadamard><<<num_blocks, block_size, 0, stream>>>(num_qubits, target_qubit_num_physical, state_data_device);
        }

        CHECK_CUDA(cudaEventRecord, event_2, stream);

        CHECK_CUDA(cudaStreamSynchronize, stream);

        CHECK_CUDA(cudaEventElapsedTime, &elapsed_ms, event_1, event_2);
        MPI_Reduce(&elapsed_ms, &elapsed_ms_2, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
        elapsed_ms = elapsed_ms_2;
        if (proc_num == 0) {
            fprintf(stderr, "[info] elapsed_gpu=%f\n", elapsed_ms * 1e-3);
            fprintf(stdout, "%lf\n", elapsed_ms * 1e-3);
        }

    }

    if (calc_checksum) {
        if (proc_num==0) {
            fprintf(stderr, "[info] gathering state data\n");

            EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
            if (!mdctx) {
                perror("EVP_MD_CTX_new failed");
                exit(1);
            }
        
            if (EVP_DigestInit_ex(mdctx, EVP_sha256(), NULL) != 1) {
                perror("EVP_DigestInit_ex failed");
                EVP_MD_CTX_free(mdctx);
                exit(1);
            }

            my_complex_t* state_data_host = (my_complex_t*)malloc(num_states * sizeof(my_complex_t));
            DEFER_FUNC(free, state_data_host);

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
                if (EVP_DigestUpdate(mdctx, &state_data_host[state_num_physical], sizeof(my_complex_t)) != 1) {
                    perror("EVP_DigestUpdate failed");
                    EVP_MD_CTX_free(mdctx);
                    exit(1);
                }
            }

            unsigned char evp_hash[EVP_MAX_MD_SIZE];
            unsigned int evp_hash_len;
            if (EVP_DigestFinal_ex(mdctx, evp_hash, &evp_hash_len) != 1) {
                perror("EVP_DigestFinal_ex failed");
                EVP_MD_CTX_free(mdctx);
                exit(1);
            }

            fprintf(stderr, "[info] checksum: ");
            for (unsigned int i = 0; i < evp_hash_len; i++) {
                fprintf(stderr, "%02x", evp_hash[i]);
            }
            fprintf(stderr, "\n");

            EVP_MD_CTX_free(mdctx);
        } else {
            MPI_Send(state_data_device, num_states_local * 2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();

    return 0;

}