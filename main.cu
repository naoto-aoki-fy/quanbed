#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <stdint.h>
#include <unistd.h>

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

#include <mpi.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cuda/std/complex>
#include <cub/cub.cuh>
#include <nccl.h>
#include <openssl/evp.h>

#include <atlc/log2_int.hpp>
#include <atlc/mpi.hpp>
#include <atlc/cuda.hpp>
#include <atlc/check_mpi.hpp>
#include <atlc/check_cuda.hpp>
#include <atlc/check_curand.hpp>
#include <atlc/check_nccl.hpp>

namespace qcs {
    typedef double float_t;
    typedef cuda::std::complex<qcs::float_t> complex_t;
}

#define QCS_KERNEL_INPUT_MAX_SIZE 256
namespace qcs {

    __global__ void initstate_sequential_kernel(qcs::complex_t* const data_global, int proc_num)
    {
        uint64_t const num_threads = (uint64_t)gridDim.x * (uint64_t)blockDim.x;
        uint64_t const idx = (uint64_t)blockDim.x * (uint64_t)blockIdx.x + (uint64_t)threadIdx.x;
        data_global[idx] = idx + num_threads * (uint64_t)proc_num;
    }

    struct kernel_common_struct {
        int num_qubits;
        qcs::complex_t* state_data_device;
    };

    __constant__ qcs::kernel_common_struct kernel_common_constant;

    __constant__ unsigned char kernel_input_constant[QCS_KERNEL_INPUT_MAX_SIZE];

    struct kernel_input_qnlist_struct {
        int num_target_qubits;
        int num_positive_control_qubits;
        int num_negative_control_qubits;
        int qubit_num_list[0];

        static __host__ __device__ uint64_t needed_size(
            int const num_positive_control_qubits,
            int const num_negative_control_qubits,
            int const num_target_qubits
        ) {
            return
                sizeof(qcs::kernel_input_qnlist_struct)
                + sizeof(int) * (
                    2 * num_positive_control_qubits
                    + num_negative_control_qubits
                    + 2 * num_target_qubits
                );
        }

        __host__ __device__ uint64_t byte_size() const {
            return needed_size(this->num_positive_control_qubits, this->num_negative_control_qubits, this->num_target_qubits);
        }

        __host__ __device__ int get_num_operand_qubits() const {
            return
              this->num_positive_control_qubits
              + this->num_negative_control_qubits
              + this->num_target_qubits;
        }

        __host__ __device__ int const* get_operand_qubit_num_list_sorted() const {
            return this->qubit_num_list;
        }

        __host__ __device__ int* get_operand_qubit_num_list_sorted() {
            return this->qubit_num_list;
        }

        __host__ __device__ int const* get_positive_control_qubit_num_list() const {
            return this->qubit_num_list + this->get_num_operand_qubits();
        }

        __host__ __device__ int* get_positive_control_qubit_num_list() {
            return this->qubit_num_list + this->get_num_operand_qubits();
        }

        __host__ __device__ int const* get_target_qubit_num_list() const {
            return this->qubit_num_list
                + 2 * this->num_positive_control_qubits
                + this->num_negative_control_qubits
                + this->num_target_qubits;
        }

        __host__ __device__ int* get_target_qubit_num_list() {
            return this->qubit_num_list
                + 2 * this->num_positive_control_qubits
                + this->num_negative_control_qubits
                + this->num_target_qubits;
        }
    };

    struct kernel_input_qnp_struct {
        int num_target_qubits;
        int num_positive_control_qubits;
        int num_negative_control_qubits;
        int sizeof_param;
        char data[0];

        static __host__ __device__ uint64_t needed_size(
            int const num_positive_control_qubits,
            int const num_negative_control_qubits,
            int const num_target_qubits,
            int const sizeof_param
        ) {
            return
                sizeof(qcs::kernel_input_qnlist_struct)
                + sizeof(int) * (
                    2 * num_positive_control_qubits
                    + num_negative_control_qubits
                    + 2 * num_target_qubits
                )
                + sizeof_param;
        }

        __host__ __device__ void const* get_parameter_pointer() const {
            return (void const*)data;
        }

        __host__ __device__ void* get_parameter_pointer() {
            return (void*)data;
        }

        template<typename ParamType>
        __host__ __device__ ParamType const& get_parameter_reference() const {
            return *(ParamType const*)(void const*)data;
        }

        template<typename ParamType>
        __host__ __device__ ParamType& get_parameter_reference() {
            return *(ParamType*)(void*)data;
        }


        __host__ __device__ uint64_t byte_size() const {
            return needed_size(this->num_positive_control_qubits, this->num_negative_control_qubits, this->num_target_qubits, this->sizeof_param);
        }

        __host__ __device__ int get_num_operand_qubits() const {
            return
              this->num_positive_control_qubits 
              + this->num_negative_control_qubits
              + this->num_target_qubits;
        }

        __host__ __device__ int const* get_operand_qubit_num_list_sorted() const {
            return (int const*)(void*)(data + sizeof_param);
        }

        __host__ __device__ int* get_operand_qubit_num_list_sorted() {
            return (int*)(void*)(data + sizeof_param);
        }

        __host__ __device__ int const* get_positive_control_qubit_num_list() const {
            return (int const*)(void*)(data + sizeof_param) + this->get_num_operand_qubits();
        }

        __host__ __device__ int* get_positive_control_qubit_num_list() {
            return (int*)(void*)(data + sizeof_param) + this->get_num_operand_qubits();
        }

        __host__ __device__ int const* get_target_qubit_num_list() const {
            return (int const*)(void*)(data + sizeof_param)
                + 2 * this->num_positive_control_qubits
                + this->num_negative_control_qubits
                + this->num_target_qubits;
        }

        __host__ __device__ int* get_target_qubit_num_list() {
            return (int*)(void*)(data + sizeof_param) +
                2 * this->num_positive_control_qubits
                + this->num_negative_control_qubits
                + this->num_target_qubits;
        }

    };

}

struct cn_h {
    static __device__ void apply() {

        int64_t const thread_num = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;

        // int const target_qubit_num = 1;
        auto args = (qcs::kernel_input_qnlist_struct const*)(void*)qcs::kernel_input_constant;

        uint64_t index_state_0 = 0;

        int const num_operand_qubits = args->get_num_operand_qubits();
        int const* const qubit_num_list_sorted = args->get_operand_qubit_num_list_sorted();

        // generate index_state_0
        // ignoring positive control qubits
        uint64_t lower_mask = 0;
        for(int i = 0; i < num_operand_qubits; i++) {
            uint64_t const mask = (1ULL << (qubit_num_list_sorted[i] - i)) - 1;
            uint64_t const upper_mask = mask & ~lower_mask;
            lower_mask = mask;
            index_state_0 |= (thread_num & upper_mask) << i;
        }
        index_state_0 |= (thread_num & ~lower_mask) << num_operand_qubits;

        // update index_state_0
        // considering positive control qubits
        int const* const positive_control_qubit_num_list = args->get_positive_control_qubit_num_list();
        for(int i = 0; i < args->num_positive_control_qubits; i++) {
            index_state_0 |= 1ULL << positive_control_qubit_num_list[i];
        }

        // generate index_state_1
        // num_target_qubits == 1
        auto const target_qubit_num = args->get_target_qubit_num_list()[0];
        uint64_t const index_state_1 = index_state_0 | (1ULL << target_qubit_num);

        qcs::complex_t const amp_state_0 = qcs::kernel_common_constant.state_data_device[index_state_0];
        qcs::complex_t const amp_state_1 = qcs::kernel_common_constant.state_data_device[index_state_1];

        qcs::kernel_common_constant.state_data_device[index_state_0] = (amp_state_0 + amp_state_1) * M_SQRT1_2;
        qcs::kernel_common_constant.state_data_device[index_state_1] = (amp_state_0 - amp_state_1) * M_SQRT1_2;

    }
};

__global__ void cuda_gate_cn_h() {
    cn_h::apply();
}


struct cn_x {
    static __device__ void apply() {

        int64_t const thread_num = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;

        // int const target_qubit_num = 1;
        auto args = (qcs::kernel_input_qnlist_struct const*)(void*)qcs::kernel_input_constant;

        uint64_t index_state_0 = 0;

        int const num_operand_qubits = args->get_num_operand_qubits();
        int const* const qubit_num_list_sorted = args->get_operand_qubit_num_list_sorted();

        // generate index_state_0
        // ignoring positive control qubits
        uint64_t lower_mask = 0;
        // for(int i = 0; i < num_operand_qubits; i++) {
        //     uint64_t const lower_mask_next = (1ULL << (qubit_num_list_sorted[i] - i)) - 1;
        //     uint64_t const mask = lower_mask_next & ~lower_mask;
        //     lower_mask = lower_mask_next;
        //     index_state_0 |= (thread_num & mask) << i;
        // }
        for(int i = 0; i < num_operand_qubits; i++) {
            uint64_t const mask = (1ULL << (qubit_num_list_sorted[i] - i)) - 1;
            uint64_t const upper_mask = mask & ~lower_mask;
            lower_mask = mask;
            index_state_0 |= (thread_num & upper_mask) << i;
        }
        index_state_0 |= ((thread_num & ~lower_mask) << num_operand_qubits);

        // update index_state_0
        // considering positive control qubits
        int const* const positive_control_qubit_num_list = args->get_positive_control_qubit_num_list();
        for(int i = 0; i < args->num_positive_control_qubits; i++) {
            index_state_0 |= (1ULL << positive_control_qubit_num_list[i]);
        }

        // generate index_state_1
        // num_target_qubits == 1
        auto const target_qubit_num = args->get_target_qubit_num_list()[0];
        uint64_t const index_state_1 = index_state_0 | (1ULL << target_qubit_num);

        qcs::complex_t const amp_state_0 = qcs::kernel_common_constant.state_data_device[index_state_0];
        qcs::complex_t const amp_state_1 = qcs::kernel_common_constant.state_data_device[index_state_1];

        qcs::kernel_common_constant.state_data_device[index_state_0] = amp_state_1;
        qcs::kernel_common_constant.state_data_device[index_state_1] = amp_state_0;

    }
};

__global__ void cuda_gate_cn_x() {
    // printf("kernel: cuda_gate_cn_x\n");
    cn_x::apply();
}


struct hadamard {
    static __device__ void apply() {

        uint64_t const thread_num = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;
        // qcs_kernel_input_q1_struct* args = (qcs_kernel_input_q1_struct*)(void*)qcs::kernel_input_constant;
        auto args = (qcs::kernel_input_qnlist_struct const*)(void*)qcs::kernel_input_constant;
        auto const target_qubit_num = args->get_target_qubit_num_list()[0];

        uint64_t const lower_mask = (1ULL << target_qubit_num) - 1ULL;

        uint64_t const index_state_lower = thread_num & lower_mask;
        uint64_t const index_state_higher = (thread_num & ~lower_mask) << ((int64_t)1);

        uint64_t const index_state_0 = index_state_lower | index_state_higher;
        uint64_t const index_state_1 = index_state_0 | (((int64_t)1)<<target_qubit_num);

        qcs::complex_t const amp_state_0 = qcs::kernel_common_constant.state_data_device[index_state_0];
        qcs::complex_t const amp_state_1 = qcs::kernel_common_constant.state_data_device[index_state_1];

        qcs::kernel_common_constant.state_data_device[index_state_0] = (amp_state_0 + amp_state_1) * M_SQRT1_2;
        qcs::kernel_common_constant.state_data_device[index_state_1] = (amp_state_0 - amp_state_1) * M_SQRT1_2;

    }
};

struct identity {
    static __device__ void apply() { }
};

template<class Gate>
__global__ void cuda_gate() {
    Gate::apply();
}

namespace qcs {

enum class initstate_enum {
    sequential,
    zero,
    use_curand,
    laod_statevector,
};

struct simulator {

int argc;
char** argv;

int num_qubits;

bool flag_calculate_checksum;
bool use_unified_memory;

initstate_enum initstate_choice;

bool flag_save_statevector;

float elapsed_ms, elapsed_ms_2;

int num_procs, proc_num;

int num_rand_areas;

std::string my_hostname;
int my_node_number;
int my_node_local_rank;
int node_count;

int gpu_id;

ncclUniqueId nccl_id;
ncclComm_t nccl_comm;
int nccl_rank;

std::vector<int> perm_p2l;
std::vector<int> perm_l2p;

int num_samples;
int rng_seed;

int log_num_procs;
int log_block_size_max;
int target_qubit_num_begin;
int target_qubit_num_end;

cudaStream_t stream;
cudaEvent_t event_1;
cudaEvent_t event_2;

uint64_t num_states;
int num_qubits_local;
uint64_t num_states_local;
int block_size_max;

qcs::complex_t* state_data_device;
qcs::kernel_common_struct* qcs_kernel_common_constant_addr;
qcs::kernel_common_struct qcs_kernel_common_host;
qcs::kernel_input_qnlist_struct* qcs_kernel_input_constant_addr;
std::vector<char> qcs_kernel_input_host_buffer;
int log_swap_buffer_total_length;
uint64_t swap_buffer_total_length;
qcs::complex_t* swap_buffer;
qcs::float_t* norm_sum_device;

std::vector<int> operand_qubit_num_list;
std::vector<int> target_qubit_num_physical_list;
std::vector<int> swap_target_global_list;
std::vector<int> swap_target_local_list;
std::vector<int> swap_target_local_logical_list;
std::vector<int> swap_target_global_logical_list;
std::vector<int> positive_control_qubit_num_physical_list;
std::vector<int> positive_control_qubit_num_physical_global_list;
std::vector<int> positive_control_qubit_num_physical_local_list;
std::vector<int> negative_control_qubit_num_physical_list;
std::vector<int> negative_control_qubit_num_physical_global_list;
std::vector<int> negative_control_qubit_num_physical_local_list;

std::vector<int> measured_1_qubit_num_logical_list;
std::vector<int> measured_0_qubit_num_logical_list;

std::vector<int> target_qubit_num_logical_list;
std::vector<int> positive_control_qubit_num_logical_list;
std::vector<int> negative_control_qubit_num_logical_list;

bool control_condition;

simulator(int argc, char** argv) {
    this->argc = argc;
    this->argv = argv;
}

void setup(int num_rand_areas_times_num_procs) {

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_num);

    if (proc_num==0) {
        fprintf(stderr, "[info] num_procs=%d\n", num_procs);
    }

    num_rand_areas = num_rand_areas_times_num_procs / num_procs;

    atlc::group_by_hostname(proc_num, num_procs, my_hostname, my_node_number, my_node_local_rank, node_count);
    fprintf(stderr,
            "[debug] Rank %d on host %s -> assigned node number: %d, local node rank: %d (total nodes: %d)\n",
            proc_num, my_hostname.c_str(), my_node_number, my_node_local_rank, node_count);

    // gpu_id = proc_num;
    gpu_id = my_node_local_rank;
    // gpu_id = 0;
    ATLC_CHECK_CUDA(cudaSetDevice, gpu_id);

    if (proc_num == 0) {
        ATLC_CHECK_NCCL(ncclGetUniqueId, &nccl_id);
    }

    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
    nccl_rank = proc_num;
    ATLC_CHECK_NCCL(ncclCommInitRank, &nccl_comm, num_procs, nccl_id, nccl_rank);

    num_qubits = 14;
    if (proc_num == 0) { fprintf(stderr, "[info] num_qubits=%d\n", num_qubits); }

    perm_p2l.resize(num_qubits);
    perm_l2p.resize(num_qubits);

    for(int qubit_num=0; qubit_num<num_qubits; qubit_num++) {
        perm_p2l[qubit_num] = qubit_num;
        perm_l2p[qubit_num] = qubit_num;
    }

    // num_samples = 64;
    num_samples = 1ULL << num_qubits;
    rng_seed = 12345;

    log_num_procs = atlc::log2_int(num_procs);

    log_block_size_max = 9;
    target_qubit_num_begin = 0;
    target_qubit_num_end = num_qubits;
    // target_qubit_num_end = 2;

    if (proc_num == 0) { fprintf(stderr, "[info] log_block_size_max=%d\n", log_block_size_max); }

    ATLC_CHECK_CUDA(cudaStreamCreate, &stream);
    // ATLC_DEFER_CHECK_CUDA(cudaStreamDestroy, stream);

    ATLC_CHECK_CUDA(cudaEventCreateWithFlags, &event_1, cudaEventDefault);
    // ATLC_DEFER_CHECK_CUDA(cudaEventDestroy, event_1);

    ATLC_CHECK_CUDA(cudaEventCreateWithFlags, &event_2, cudaEventDefault);
    // ATLC_DEFER_CHECK_CUDA(cudaEventDestroy, event_2);

    num_states = 1ULL << num_qubits;

    num_qubits_local = num_qubits - log_num_procs;

    num_states_local = 1ULL << num_qubits_local;
    block_size_max = 1 << log_block_size_max;
    // num_blocks = 1ULL << (num_qubits_local - 1 - log_block_size_max);

    if (proc_num == 0) { fprintf(stderr, "[info] malloc device memory\n"); }

    if (use_unified_memory) {
        ATLC_CHECK_CUDA(cudaMallocManaged, &state_data_device, num_states_local * sizeof(*state_data_device));
        ATLC_CHECK_CUDA(cudaMemAdvise, state_data_device, num_states_local * sizeof(*state_data_device), cudaMemAdviseSetPreferredLocation, gpu_id);
    } else {
        ATLC_CHECK_CUDA(cudaMallocAsync, &state_data_device, num_states_local * sizeof(*state_data_device), stream);
    }
    // ATLC_DEFER_CHECK_CUDA(cudaFreeAsync, state_data_device, stream);

    ATLC_CHECK_CUDA(cudaGetSymbolAddress, (void**)&qcs_kernel_common_constant_addr, qcs::kernel_common_constant);

    qcs_kernel_common_host.num_qubits = num_qubits;
    qcs_kernel_common_host.state_data_device = state_data_device;
    ATLC_CHECK_CUDA(cudaMemcpyAsync, qcs_kernel_common_constant_addr, &qcs_kernel_common_host, sizeof(qcs::kernel_common_struct), cudaMemcpyHostToDevice, stream);

    ATLC_CHECK_CUDA(cudaGetSymbolAddress, (void**)&qcs_kernel_input_constant_addr, qcs::kernel_input_constant);

    // qcs::kernel_input_qnlist_struct* qcs_kernel_input_host = (qcs::kernel_input_qnlist_struct*)malloc(QCS_KERNEL_INPUT_SIZE);
    // ATLC_DEFER_FUNC(free, qcs_kernel_input_host);

    log_swap_buffer_total_length = (num_qubits_local>30)? num_qubits_local - 3 : num_qubits_local;
    // log_swap_buffer_total_length = num_qubits_local;
    swap_buffer_total_length = 1ULL << log_swap_buffer_total_length;
    ATLC_CHECK_CUDA(cudaMallocAsync, &swap_buffer, swap_buffer_total_length * sizeof(qcs::complex_t), stream);
    // ATLC_CHECK_CUDA(cudaMallocManaged, &swap_buffer, swap_buffer_total_length * sizeof(qcs::complex_t));
    // ATLC_DEFER_CHECK_CUDA(cudaFreeAsync, swap_buffer, stream);

    ATLC_CHECK_CUDA(cudaMallocAsync, &norm_sum_device, (num_states_local>>log_block_size_max) * sizeof(qcs::float_t), stream);
    // ATLC_DEFER_CHECK_CUDA(cudaFreeAsync, norm_sum_device, stream);

}

void initialize_sequential() {
    // std::vector<qcs::complex_t> state_data_host(num_states_local);
    // for (uint64_t state_num_local = 0; state_num_local < num_states_local; state_num_local++) {
    //     state_data_host[state_num_local] = state_num_local + proc_num * num_states_local;
    // }
    // ATLC_CHECK_CUDA(cudaMemcpyAsync, state_data_device, state_data_host.data(), sizeof(qcs::complex_t) * num_states_local, cudaMemcpyHostToDevice, stream);
    uint64_t num_blocks_init;
    uint64_t block_size_init;
    if (num_qubits_local >= log_block_size_max) {
        num_blocks_init = num_states_local >> log_block_size_max;
        block_size_init = block_size_max;
    } else {
        num_blocks_init = 1;
        block_size_init = num_states_local;
    }
    // fprintf(stderr, "debug: num_blocks_init=%llu\n", num_blocks_init);
    // fprintf(stderr, "debug: block_size_init=%llu\n", block_size_init);

    ATLC_CHECK_CUDA(atlc::cudaLaunchKernel, qcs::initstate_sequential_kernel, num_blocks_init, block_size_init, 0, stream, state_data_device, proc_num);

}

void initialize_zero() {
    if (proc_num == 0) {
        qcs::complex_t const one = 1;
        ATLC_CHECK_CUDA(cudaMemcpyAsync, state_data_device, &one, sizeof(qcs::float_t), cudaMemcpyHostToDevice, stream);
    }
}

void initialize_use_curand() {

    MPI_Barrier(MPI_COMM_WORLD);
    if (proc_num == 0) { fprintf(stderr, "[info] generating random state\n"); }
    curandGenerator_t rng_device;

    // ATLC_CHECK_CURAND(curandCreateGenerator, &rng_device, CURAND_RNG_PSEUDO_DEFAULT);
    // ATLC_CHECK_CURAND(curandSetStream, rng_device, stream);
    // ATLC_CHECK_CURAND(curandSetPseudoRandomGeneratorSeed, rng_device, rng_seed + proc_num);

    ATLC_CHECK_CUDA(cudaEventRecord, event_1, stream);

    // if (false) {
    //     ATLC_CHECK_CURAND(curandCreateGenerator, &rng_device, CURAND_RNG_PSEUDO_DEFAULT);
    //     ATLC_CHECK_CURAND(curandSetStream, rng_device, stream);
    //     ATLC_CHECK_CURAND(curandSetPseudoRandomGeneratorSeed, rng_device, rng_seed + proc_num);
    //     ATLC_CHECK_CURAND(curandGenerateNormalDouble, rng_device, (qcs::float_t*)(void*)state_data_device, num_states_local * 2 /* complex */, 0.0, 1.0);
    //     ATLC_CHECK_CURAND(curandDestroyGenerator, rng_device);
    // } else
    {
        // int const num_rand_areas = 4;
        int const log_num_rand_areas = atlc::log2_int(num_rand_areas);
        // if (log_num_rand_areas!=1) { throw; }
        uint64_t const num_states_rand_area = num_states_local >> log_num_rand_areas;
        for (int rand_area_num = 0; rand_area_num < num_rand_areas; rand_area_num++) {
            ATLC_CHECK_CURAND(curandCreateGenerator, &rng_device, CURAND_RNG_PSEUDO_DEFAULT);
            ATLC_CHECK_CURAND(curandSetStream, rng_device, stream);
            ATLC_CHECK_CURAND(curandSetPseudoRandomGeneratorSeed, rng_device, rng_seed + proc_num * num_rand_areas + rand_area_num);
            ATLC_CHECK_CURAND(curandGenerateNormalDouble, rng_device, (qcs::float_t*)(void*)(state_data_device + num_states_rand_area * ((uint64_t)rand_area_num)), num_states_rand_area * 2 /* complex */, 0.0, 1.0);
            ATLC_CHECK_CURAND(curandDestroyGenerator, rng_device);
        }
    }
    // ATLC_CHECK_CURAND(curandSetPseudoRandomGeneratorSeed, rng_device, rng_seed + proc_num * 2);
    // ATLC_CHECK_CURAND(curandGenerateNormalDouble, rng_device, (qcs::float_t*)(void*)state_data_device, num_states_local, 0.0, 1.0);

    // curandGenerator_t rng_device_2;

    // ATLC_CHECK_CURAND(curandCreateGenerator, &rng_device_2, CURAND_RNG_PSEUDO_DEFAULT);
    // ATLC_CHECK_CURAND(curandSetStream, rng_device_2, stream);
    // ATLC_CHECK_CURAND(curandSetPseudoRandomGeneratorSeed, rng_device_2, rng_seed + proc_num * 2 + 1);

    // ATLC_CHECK_CURAND(curandGenerateNormalDouble, rng_device_2, &((qcs::float_t*)(void*)state_data_device)[num_states_local], num_states_local, 0.0, 1.0);
}

void initialize_laod_statevector() {

    MPI_Barrier(MPI_COMM_WORLD);

    if (proc_num == 0) { fprintf(stderr, "[info] load statevector\n"); }

    qcs::complex_t* state_data_host = (qcs::complex_t*)malloc(num_states_local * sizeof(qcs::complex_t));
    ATLC_DEFER_FUNC(free, state_data_host);

    for(int proc_num_active=0; proc_num_active<num_procs; proc_num_active++) {
        if (proc_num_active == proc_num) {
            FILE* const fp = fopen("statevector_input.bin", "rb");
            if (fp == NULL) {
                throw std::runtime_error("open failed");
            }
            fseek(fp, proc_num * num_states_local * sizeof(qcs::complex_t), SEEK_SET);
            size_t const ret = fread(state_data_host, sizeof(qcs::complex_t), num_states_local, fp);
            if (ret != num_states_local) {
                throw std::runtime_error("fread failed");
            }
            fclose(fp);

            ATLC_CHECK_CUDA(cudaMemcpyAsync, state_data_device, state_data_host, num_states_local * sizeof(qcs::complex_t), cudaMemcpyHostToDevice, stream);

        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
}

void ensure_local_qubits() {
    target_qubit_num_physical_list.resize(target_qubit_num_logical_list.size());
    for (int tqni = 0; tqni < target_qubit_num_logical_list.size(); tqni++) {
        target_qubit_num_physical_list[tqni] = perm_l2p[target_qubit_num_logical_list[tqni]];
        // fprintf(stderr, "[debug] target_qubit_num_physical_list[tqni]=%d tqni=%d\n", target_qubit_num_physical_list[tqni], tqni);
    }

    swap_target_global_list.resize(0);
    swap_target_local_list.resize(0);
    for (int tqni = 0; tqni < target_qubit_num_physical_list.size(); tqni++) {
        auto const tqn_i = target_qubit_num_physical_list[tqni];
        if (tqn_i >= num_qubits_local) {
            swap_target_global_list.push_back(tqn_i);
            int const swap_target_local = num_qubits_local - swap_target_global_list.size();
            swap_target_local_list.push_back(swap_target_local);
            target_qubit_num_physical_list[tqni] = swap_target_local;
            // fprintf(stderr, "[debug] target_qubit_num_physical=%d swap_target_local=%d\n", tqn_i, swap_target_local);
        }
    }
    int const num_swap_qubits = swap_target_global_list.size();

    /* target qubits is global */
    if (swap_target_global_list.size() > 0) {

        // b_min
        int const swap_target_local_min = *std::min_element(swap_target_local_list.data(), swap_target_local_list.data() + num_swap_qubits);

        uint64_t const local_buf_length = UINT64_C(1) << swap_target_local_min;
        uint64_t swap_buffer_length = swap_buffer_total_length;
        if (swap_buffer_length > local_buf_length) {
            swap_buffer_length = local_buf_length;
        }

        // generate a mask for generating global_nonswap_self
        uint64_t global_swap_self_mask = 0;
        for (int target_num = 0; target_num < num_swap_qubits; target_num++) {
            // a_delta = a – n_local
            int const swap_target_global_delta = swap_target_global_list[target_num] - num_qubits_local;
            global_swap_self_mask |= (UINT64_C(1) << swap_target_global_delta);
        }

        // global_nonswap_self = make proc_num_self's a_delta_i-th digit zero
        uint64_t const global_nonswap_self = proc_num & ~global_swap_self_mask;

        ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
        ATLC_CHECK_MPI(MPI_Barrier, MPI_COMM_WORLD);

        // 1<<(num_local_qubits - b_min) 
        uint64_t const num_local_areas = UINT64_C(1) << (num_qubits_local - swap_target_local_min);
        for (uint64_t local_num_self = 0; local_num_self < num_local_areas; local_num_self++) {

            // global_swap_peer = OR_i (local_num_selfのb_delta_i桁目)をa_delta_i桁目にする
            uint64_t global_swap_peer = 0;
            for (int target_num = 0; target_num < num_swap_qubits; target_num++) {
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

            bool is_peer_greater = proc_num_peer > proc_num;
            for (uint64_t buffer_pos = 0; buffer_pos < local_buf_length; buffer_pos += swap_buffer_length) {
                ATLC_CHECK_NCCL(ncclGroupStart);
                for (int send_recv = 0; send_recv < 2; send_recv++) {
                    if (send_recv ^ is_peer_greater) {
                        ATLC_CHECK_NCCL(ncclSend, &state_data_device[local_num_self * local_buf_length + buffer_pos], swap_buffer_length * 2 /* complex */, ncclDouble, proc_num_peer, nccl_comm, stream);
                    } else {
                        ATLC_CHECK_NCCL(ncclRecv, swap_buffer, swap_buffer_length * 2 /* complex */, ncclDouble, proc_num_peer, nccl_comm, stream);
                    }
                }
                ATLC_CHECK_NCCL(ncclGroupEnd);
                ATLC_CHECK_CUDA(cudaMemcpyAsync, &state_data_device[local_num_self * local_buf_length + buffer_pos], swap_buffer, swap_buffer_length * sizeof(qcs::complex_t), cudaMemcpyDeviceToDevice, stream);
            }

        }

        // swap_target_global_logical_list[:] = perm_p2l[swap_target_global_list[:]]
        // swap_target_local_logical_list[:] = perm_p2l[swap_target_local_list[:]]
        swap_target_local_logical_list.resize(num_swap_qubits);
        swap_target_global_logical_list.resize(num_swap_qubits);
        for (int target_num = 0; target_num < num_swap_qubits; target_num++) {
            swap_target_local_logical_list[target_num] = perm_p2l[swap_target_local_list[target_num]];
            swap_target_global_logical_list[target_num] = perm_p2l[swap_target_global_list[target_num]];
        }

        // update p2l & l2p
        // perm_p2l[swap_target_global_list[:]] = swap_target_local_logical_list[:]
        // perm_p2l[swap_target_local_list[:]] = swap_target_global_logical_list[:]
        // perm_l2p[swap_target_global_logical_list[:]] = swap_target_local_list[:]
        // perm_l2p[swap_target_local_logical_list[:]] = swap_target_global_list[:]

        for (int target_num = 0; target_num < num_swap_qubits; target_num++) {
            perm_p2l[swap_target_global_list[target_num]] = swap_target_local_logical_list[target_num];
            perm_p2l[swap_target_local_list[target_num]] = swap_target_global_logical_list[target_num];
            perm_l2p[swap_target_global_logical_list[target_num]] = swap_target_local_list[target_num];
            perm_l2p[swap_target_local_logical_list[target_num]] = swap_target_global_list[target_num];
        }

        // target_qubit_num_physical = swap_target_local;

    }
};

void check_control_qubit_num_physical() {

    /* check whether proc_num is under control condition */
    control_condition = true;

    positive_control_qubit_num_physical_list.resize(positive_control_qubit_num_logical_list.size());
    positive_control_qubit_num_physical_global_list.resize(0);
    positive_control_qubit_num_physical_local_list.resize(0);

    for (int cqni = 0; cqni < positive_control_qubit_num_logical_list.size(); cqni++) {
        auto const positive_control_qubit_num_physical = perm_l2p[positive_control_qubit_num_logical_list[cqni]];
        positive_control_qubit_num_physical_list[cqni] = positive_control_qubit_num_physical;
        if (positive_control_qubit_num_physical >= num_qubits_local) {
            positive_control_qubit_num_physical_global_list.push_back(positive_control_qubit_num_physical);
            if (!(1 & (proc_num >> (positive_control_qubit_num_physical - num_qubits_local))) /* 0 */ ) {
                control_condition = false;
            }
            // fprintf(stderr, "debug: proc_num=%d ctrl(global)=%d\n", proc_num, positive_control_qubit_num_logical_list[cqni]);
        } else {
            positive_control_qubit_num_physical_local_list.push_back(positive_control_qubit_num_physical);
            // fprintf(stderr, "debug: proc_num=%d ctrl(local)=%d\n", proc_num, positive_control_qubit_num_logical_list[cqni]);
        }

    }

    negative_control_qubit_num_physical_list.resize(negative_control_qubit_num_logical_list.size());
    negative_control_qubit_num_physical_global_list.resize(0);
    negative_control_qubit_num_physical_local_list.resize(0);

    for (int cqni = 0; cqni < negative_control_qubit_num_logical_list.size(); cqni++) {
        auto const negative_control_qubit_num_physical = perm_l2p[negative_control_qubit_num_logical_list[cqni]];
        negative_control_qubit_num_physical_list[cqni] = negative_control_qubit_num_physical;
        if (negative_control_qubit_num_physical >= num_qubits_local) {
            negative_control_qubit_num_physical_global_list.push_back(negative_control_qubit_num_physical);
            if (1 & (proc_num >> (negative_control_qubit_num_physical - num_qubits_local)) /* 1 */ ) {
                control_condition = false;
            }
            // fprintf(stderr, "debug: proc_num=%d negctrl(global)=%d\n", proc_num, negative_control_qubit_num_logical_list[cqni]);
        } else {
            negative_control_qubit_num_physical_local_list.push_back(negative_control_qubit_num_physical);
            // fprintf(stderr, "debug: proc_num=%d negctrl(local)=%d\n", proc_num, negative_control_qubit_num_logical_list[cqni]);
        }
    }
};

void operate_gate() {

    if (!control_condition) { return; }

    uint64_t const qkiqn_size = qcs::kernel_input_qnlist_struct::needed_size(
        positive_control_qubit_num_physical_list.size(),
        negative_control_qubit_num_physical_list.size(),
        target_qubit_num_physical_list.size()
    );
    if (qkiqn_size > QCS_KERNEL_INPUT_MAX_SIZE) {
        std::vector<char> runtime_error_message(256);
        sprintf(runtime_error_message.data(), "qkiqn_size(%llu) > QCS_KERNEL_INPUT_MAX_SIZE(%llu)", qkiqn_size, QCS_KERNEL_INPUT_MAX_SIZE);
        throw std::runtime_error(runtime_error_message.data());
    }
    qcs_kernel_input_host_buffer.resize(qkiqn_size);
    qcs::kernel_input_qnlist_struct* const qcs_kernel_input_host = (qcs::kernel_input_qnlist_struct*)qcs_kernel_input_host_buffer.data();

    qcs_kernel_input_host->num_positive_control_qubits = positive_control_qubit_num_physical_local_list.size();
    qcs_kernel_input_host->num_negative_control_qubits = negative_control_qubit_num_physical_local_list.size();
    qcs_kernel_input_host->num_target_qubits = target_qubit_num_physical_list.size();

    auto positive_control_qubit_num_list_kernel_arg = qcs_kernel_input_host->get_positive_control_qubit_num_list();
    for (int pcqi = 0; pcqi < positive_control_qubit_num_physical_local_list.size(); pcqi++) {
        positive_control_qubit_num_list_kernel_arg[pcqi] = positive_control_qubit_num_physical_local_list[pcqi];
    }

    auto target_qubit_num_list_kernel_arg = qcs_kernel_input_host->get_target_qubit_num_list();
    for (int tqi = 0; tqi < target_qubit_num_physical_list.size(); tqi++) {
        target_qubit_num_list_kernel_arg[tqi] = target_qubit_num_physical_list[tqi];
    }

    auto const num_operand_qubits =
        positive_control_qubit_num_physical_local_list.size()
        + negative_control_qubit_num_physical_local_list.size()
        + target_qubit_num_physical_list.size();

    /* get sorted operand qubits */
    operand_qubit_num_list.resize(0);
    operand_qubit_num_list.reserve(num_operand_qubits);
    operand_qubit_num_list.insert(operand_qubit_num_list.end(), positive_control_qubit_num_physical_local_list.begin(), positive_control_qubit_num_physical_local_list.end());
    operand_qubit_num_list.insert(operand_qubit_num_list.end(), negative_control_qubit_num_physical_local_list.begin(), negative_control_qubit_num_physical_local_list.end());
    operand_qubit_num_list.insert(operand_qubit_num_list.end(), target_qubit_num_physical_list.begin(), target_qubit_num_physical_list.end());

    std::sort(operand_qubit_num_list.begin(), operand_qubit_num_list.end()); /* ascending order */

    auto qubit_num_list_sorted_kernel_arg = qcs_kernel_input_host->get_operand_qubit_num_list_sorted();
    for (int qni = 0; qni < operand_qubit_num_list.size(); qni++) {
        qubit_num_list_sorted_kernel_arg[qni] = operand_qubit_num_list[qni];
    }

    ATLC_CHECK_CUDA(cudaMemcpyAsync, qcs_kernel_input_constant_addr, qcs_kernel_input_host, qkiqn_size, cudaMemcpyHostToDevice, stream);

    uint64_t const log_num_threads = num_qubits_local - num_operand_qubits;
    uint64_t log_block_size_gateop;
    uint64_t num_blocks_gateop;

    if (log_block_size_max > log_num_threads) {
        log_block_size_gateop = log_num_threads;
        num_blocks_gateop = 1;
    } else {
        log_block_size_gateop = log_block_size_max;
        num_blocks_gateop = 1ULL << (log_num_threads - log_block_size_max);
    }

    uint64_t const block_size_gateop = 1ULL << log_block_size_gateop;

    // cuda_gate<hadamard><<<num_blocks_gateop, block_size, 0, stream>>>();
    ATLC_CHECK_CUDA(atlc::cudaLaunchKernel, cuda_gate_cn_x, num_blocks_gateop, block_size_gateop, 0, stream);
    // cuda_gate<hadamard><<<num_blocks_gateop, block_size, 0, stream>>>();
}

void save_statevector() {

    MPI_Barrier(MPI_COMM_WORLD);
    if (proc_num == 0) { fprintf(stderr, "[info] dump statevector\n"); }

    qcs::complex_t* state_data_host = (qcs::complex_t*)malloc(num_states_local * sizeof(qcs::complex_t));
    ATLC_DEFER_FUNC(free, state_data_host);

    ATLC_CHECK_CUDA(cudaMemcpyAsync, state_data_host, state_data_device, num_states_local * sizeof(qcs::complex_t), cudaMemcpyDeviceToHost, stream);

    for(int proc_num_active=0; proc_num_active<num_procs; proc_num_active++) {
        if (proc_num_active == proc_num) {
            FILE* const fp = fopen("statevector_output.bin", (proc_num==0)? "wb": "rb+");
            if (fp == NULL) {
                throw std::runtime_error("open failed");
            }

            for (uint64_t state_num_physical_local = 0; state_num_physical_local < num_states_local; state_num_physical_local++) {
                uint64_t const state_num_physical = state_num_physical_local | (((uint64_t)proc_num) << num_qubits_local);
                uint64_t state_num_logical = 0;
                for(int qubit_num_physical = 0; qubit_num_physical < num_qubits; qubit_num_physical++) {
                    int qubit_num_logical = perm_p2l[qubit_num_physical];
                    state_num_logical = state_num_logical | (((state_num_physical >> qubit_num_physical) & 1) << qubit_num_logical);
                }
                int const ret_fseek = fseek(fp, state_num_logical * sizeof(qcs::complex_t), SEEK_SET);
                if (ret_fseek!=0) {
                    fprintf(stderr, "errno=%d\n", errno);
                    std::vector<char> error_buf(128);
                    sprintf(error_buf.data(), "ret_fseek=%d", ret_fseek);
                    throw std::runtime_error(error_buf.data());
                }
                size_t const ret = fwrite(&state_data_host[state_num_physical_local], sizeof(qcs::complex_t), 1, fp);
                if (ret != 1) {
                    fprintf(stderr, "errno=%d\n", errno);
                    std::vector<char> error_buf(128);
                    sprintf(error_buf.data(), "fwrite failed ret=%d", ret);
                    throw std::runtime_error(error_buf.data());
                }
            }
            fflush(fp);
            fclose(fp);
            fsync(fileno(fp));
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

void calculate_checksum() {

    if (proc_num==0) {
        fprintf(stderr, "[info] gathering state data\n");

        EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
        if (!mdctx) {
            perror("EVP_MD_CTX_new failed");
            exit(1);
        }

        if (EVP_DigestInit_ex(mdctx, EVP_md5(), NULL) != 1) {
            perror("EVP_DigestInit_ex failed");
            EVP_MD_CTX_free(mdctx);
            exit(1);
        }

        qcs::complex_t* state_data_host = (qcs::complex_t*)malloc(num_states * sizeof(qcs::complex_t));
        ATLC_DEFER_FUNC(free, state_data_host);

        ATLC_CHECK_CUDA(cudaMemcpyAsync, state_data_host, state_data_device, num_states_local * sizeof(qcs::complex_t), cudaMemcpyDeviceToHost, stream);
        for(int peer_proc_num=1; peer_proc_num<num_procs; peer_proc_num++) {
            MPI_Status mpi_status;
            MPI_Recv(&state_data_host[peer_proc_num * num_states_local], num_states_local * 2, MPI_DOUBLE, peer_proc_num, 0, MPI_COMM_WORLD, &mpi_status);
        }
        ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);

        for(int64_t state_num_logical = 0; state_num_logical < num_states; state_num_logical++) {
            int64_t state_num_physical = 0;
            for(int qubit_num_logical = 0; qubit_num_logical < num_qubits; qubit_num_logical++) {
                int qubit_num_physical = perm_l2p[qubit_num_logical];
                state_num_physical = state_num_physical | (((state_num_logical >> qubit_num_logical) & 1) << qubit_num_physical);
            }

            if (EVP_DigestUpdate(mdctx, &state_data_host[state_num_physical], sizeof(qcs::complex_t)) != 1) {
                perror("EVP_DigestUpdate failed");
                EVP_MD_CTX_free(mdctx);
                exit(1);
            }
        }

        std::vector<unsigned char> evp_hash(EVP_MAX_MD_SIZE); // [EVP_MAX_MD_SIZE];
        unsigned int evp_hash_len;
        if (EVP_DigestFinal_ex(mdctx, evp_hash.data(), &evp_hash_len) != 1) {
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

struct IndirectLoad
{
    __host__ __device__ __forceinline__
    qcs::complex_t operator()(uint64_t thread_num) const
    {
#ifdef __CUDA_ARCH__
        auto args = (qcs::kernel_input_qnlist_struct const*)(void*)qcs::kernel_input_constant;

        uint64_t index_state_0 = 0;

        int const num_operand_qubits = args->get_num_operand_qubits();
        int const* const qubit_num_list_sorted = args->get_operand_qubit_num_list_sorted();

        // generate index_state_0
        // ignoring positive control qubits
        uint64_t lower_mask = 0;
        for(int i = 0; i < num_operand_qubits; i++) {
            uint64_t const mask = (1ULL << (qubit_num_list_sorted[i] - i)) - 1;
            uint64_t const upper_mask = mask & ~lower_mask;
            lower_mask = mask;
            index_state_0 |= (thread_num & upper_mask) << i;
        }
        index_state_0 |= (thread_num & ~lower_mask) << num_operand_qubits;

        // update index_state_0
        // considering positive control qubits
        int const* const positive_control_qubit_num_list = args->get_positive_control_qubit_num_list();
        for(int i = 0; i < args->num_positive_control_qubits; i++) {
            index_state_0 |= 1ULL << positive_control_qubit_num_list[i];
        }

        // generate index_state_1
        // num_target_qubits == 1
        auto const target_qubit_num = args->get_target_qubit_num_list()[0];
        uint64_t const index_state_1 = index_state_0 | (1ULL << target_qubit_num);

        return {cuda::std::norm(qcs::kernel_common_constant.state_data_device[index_state_0]), cuda::std::norm(qcs::kernel_common_constant.state_data_device[index_state_1])};
#else
        return 0;
#endif
    }
};

int main(int argc, char** argv) {

    flag_calculate_checksum = true;
    use_unified_memory = false;
    initstate_choice = initstate_enum::use_curand;
    flag_save_statevector = false;

    setup(8 /* num_rand_areas_times_num_procs */);
    ATLC_DEFER_FUNC(dispose);

    switch (initstate_choice) {
        case initstate_enum::sequential:
            initialize_sequential();
            break;
        case initstate_enum::zero:
            initialize_zero();
            break;
        case initstate_enum::use_curand:
            initialize_use_curand();
            break;
        case initstate_enum::laod_statevector:
            initialize_laod_statevector();
            break;
        default:
            throw initstate_choice;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(proc_num==0) {
        fprintf(stderr, "[info] gpu measurement\n");
    }

    // std::random_device seed_gen;
    // auto const seed = seed_gen();
    std::mt19937 engine(rng_seed);

    for(int sample_num=0; sample_num < num_samples; ++sample_num) {

        measured_0_qubit_num_logical_list.clear();
        measured_1_qubit_num_logical_list.clear();
        uint64_t measured_bit = 0;

        for (int measure_qubit_num_logical = 0; measure_qubit_num_logical < num_qubits; measure_qubit_num_logical++) {

            target_qubit_num_logical_list = {measure_qubit_num_logical};
            positive_control_qubit_num_logical_list = measured_1_qubit_num_logical_list;
            negative_control_qubit_num_logical_list = measured_0_qubit_num_logical_list;

            check_control_qubit_num_physical();

            qcs::complex_t measure_norm_host;

            if (control_condition) {
                using CountingIter = cub::CountingInputIterator<uint64_t>;
                using TransformIter = cub::TransformInputIterator<qcs::complex_t, IndirectLoad, CountingIter>;

                IndirectLoad loader;
                CountingIter counting(0);
                TransformIter in_it(counting, loader);

                qcs::complex_t* measure_norm_device;
                ATLC_CHECK_CUDA(cudaMallocAsync, &measure_norm_device, sizeof(qcs::complex_t), stream);

                size_t temp_sz = 0;
                ATLC_CHECK_CUDA(cub::DeviceReduce::Sum, NULL, temp_sz, in_it, measure_norm_device, num_states_local, stream);

                void* d_temp;
                ATLC_CHECK_CUDA(cudaMallocAsync, &d_temp, temp_sz, stream);

                ATLC_CHECK_CUDA(cub::DeviceReduce::Sum, d_temp, temp_sz, in_it, measure_norm_device, num_states_local, stream);

                ATLC_CHECK_CUDA(cudaFreeAsync, d_temp, stream);


                ATLC_CHECK_CUDA(cudaMemcpyAsync, &measure_norm_host, measure_norm_device, sizeof(qcs::complex_t), cudaMemcpyDeviceToHost, stream);

                ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
            } else {
                measure_norm_host = 0;
            }

#if 1 /* parallel measurement */
            qcs::complex_t measure_norm_global;
            MPI_Allreduce((qcs::float_t*)&measure_norm_host, (qcs::float_t*)&measure_norm_global, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            qcs::float_t const measure_norm_sum = measure_norm_global.real() + measure_norm_global.imag();

            std::uniform_real_distribution<qcs::float_t> dist1(0, measure_norm_sum);
            qcs::float_t const random_value = dist1(engine);
            bool measure_result = measure_norm_global.real() < random_value;

            if (measure_result) { /* 1 */
                measured_1_qubit_num_logical_list.push_back(measure_qubit_num_logical);
                measured_bit |= 1ULL << measure_qubit_num_logical;
            } else { /* 0 */
                measured_0_qubit_num_logical_list.push_back(measure_qubit_num_logical);
            }
#else /* measurement by master */
            qcs::complex_t measure_norm_global;
            MPI_Reduce((qcs::float_t*)&measure_norm_host, (qcs::float_t*)&measure_norm_global, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            // MPI_Allreduce((qcs::float_t*)&measure_norm_host, (qcs::float_t*)&measure_norm_global, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            bool measure_result_0;
            if (proc_num == 0) {
                qcs::float_t const measure_norm_sum = measure_norm_global.real() + measure_norm_global.imag();

                std::uniform_real_distribution<qcs::float_t> dist1(0, measure_norm_sum);
                qcs::float_t const random_value = dist1(engine);
                measure_result_0 = measure_norm_global.real() < random_value;
            }
            bool measure_result;
            MPI_Scatter(&measure_result_0, 1, MPI_C_BOOL, &measure_result, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

            if (measure_result) { /* 1 */
                measured_1_qubit_num_logical_list.push_back(measure_qubit_num_logical);
                measured_bit |= 1ULL << measure_qubit_num_logical;
            } else { /* 0 */
                measured_0_qubit_num_logical_list.push_back(measure_qubit_num_logical);
            }
#endif

            // if (proc_num == 0) {
            //     fprintf(stderr, "[debug] qubit[%d]=%d\n", measure_qubit_num_logical, (int)measure_result);
            // }

        }

        if (proc_num == 0) {
            fprintf(stdout, "%llu\n", measured_bit);
        }
    }


#if 0
    if(proc_num==0) {
        fprintf(stderr, "[info] gpu_hadamard\n");
    }

    for(int sample_num=0; sample_num < num_samples; ++sample_num) {

        ATLC_CHECK_CUDA(cudaEventRecord, event_1, stream);

        for(int target_qubit_num_logical = target_qubit_num_begin; target_qubit_num_logical < target_qubit_num_end; target_qubit_num_logical++)
        {
            target_qubit_num_logical_list = {target_qubit_num_logical};
            positive_control_qubit_num_logical_list = {};
            negative_control_qubit_num_logical_list = {};

            positive_control_qubit_num_logical_list.reserve(positive_control_qubit_num_logical_list.size() + measured_1_qubit_num_logical_list.size());
            positive_control_qubit_num_logical_list.insert(positive_control_qubit_num_logical_list.end(), measured_1_qubit_num_logical_list.begin(), measured_1_qubit_num_logical_list.end());

            negative_control_qubit_num_logical_list.reserve(negative_control_qubit_num_logical_list.size() + measured_0_qubit_num_logical_list.size());
            negative_control_qubit_num_logical_list.insert(negative_control_qubit_num_logical_list.end(), measured_0_qubit_num_logical_list.begin(), measured_0_qubit_num_logical_list.end());

            ensure_local_qubits();
            check_control_qubit_num_physical();
            operate_gate();

        } /* target_qubit_num_logical loop */

        ATLC_CHECK_CUDA(cudaEventRecord, event_2, stream);

        ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);

        ATLC_CHECK_CUDA(cudaEventElapsedTime, &elapsed_ms, event_1, event_2);

        MPI_Reduce(&elapsed_ms, &elapsed_ms_2, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
        elapsed_ms = elapsed_ms_2;
        if (proc_num == 0) {
            fprintf(stderr, "[info] elapsed_gpu=%f\n", elapsed_ms * 1e-3);
            fprintf(stdout, "%lf\n", elapsed_ms * 1e-3);
        }

    }
#endif

    if (flag_save_statevector) { save_statevector(); }
    if (flag_calculate_checksum) { calculate_checksum(); }

    MPI_Finalize();

    return 0;

};

void dispose() {
    ATLC_CHECK_CUDA(cudaFreeAsync, state_data_device, stream);
    ATLC_CHECK_CUDA(cudaFreeAsync, swap_buffer, stream);
    ATLC_CHECK_CUDA(cudaEventDestroy, event_1);
    ATLC_CHECK_CUDA(cudaEventDestroy, event_2);
    ATLC_CHECK_CUDA(cudaStreamDestroy, stream);
};

};
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IOLBF, 1024 * 512);
    qcs::simulator simulator(argc, argv);
    return simulator.main(argc, argv);
}