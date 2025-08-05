#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cinttypes>
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
#include <cuda/std/array>
#include <cub/cub.cuh>
#include <nccl.h>
#include <openssl/evp.h>

#include <atlc/format.hpp>
#include <atlc/log2_int.hpp>
#include <atlc/mpi.hpp>
#include <atlc/cuda.hpp>
#include <atlc/check_mpi.hpp>
#include <atlc/check_cuda.hpp>
#include <atlc/check_curand.hpp>
#include <atlc/check_nccl.hpp>
#include <atlc/reorder.h>

#define QCS_DEBUG
#ifdef QCS_DEBUG
#define DEBUG_UINT_VAR(var) do { fprintf(stderr, "debug: " #var "=%" PRIu64 "\n", (uint64_t)var); } while(0)
#define DEBUG_FLOAT_VAR(var) do { fprintf(stderr, "debug: " #var "=%.18g\n", (double)var); } while(0)
#else
#define DEBUG_UINT_VAR(var) do {} while(0)
#define DEBUG_FLOAT_VAR(var) do {} while(0)
#endif

#define VAR_STR_FLOAT(var) #var "=%g", (double)var
#define VAR_STR_UINT(var) #var "=%" PRIu64, (uint64_t)var
#define VAR_STR_INT(var) #var "=%" PRId64, (int64_t)var

namespace qcs {

typedef double float_t;
typedef cuda::std::complex<qcs::float_t> complex_t;
typedef cuda::std::array<qcs::float_t, 2> float2_t;

constexpr uint64_t kernel_input_max_size = 256;

__global__ void initstate_sequential_kernel(qcs::complex_t* const data_global, int proc_num)
{
    uint64_t const num_threads = (uint64_t)gridDim.x * (uint64_t)blockDim.x;
    uint64_t const idx = (uint64_t)blockDim.x * (uint64_t)blockIdx.x + (uint64_t)threadIdx.x;
    data_global[idx] = 1 + idx + num_threads * proc_num;
}

__global__ void initstate_flat_kernel(qcs::complex_t* const data_global)
{
    uint64_t const idx = (uint64_t)blockDim.x * (uint64_t)blockIdx.x + (uint64_t)threadIdx.x;
    data_global[idx] = 1;
}

struct kernel_common_struct {
    int num_qubits;
    qcs::complex_t* state_data_device;
};

__constant__ qcs::kernel_common_struct kernel_common_constant;

__constant__ unsigned char kernel_input_constant[qcs::kernel_input_max_size];

struct kernel_input_qnlist_struct {
    int num_target_qubits;
    int num_positive_control_qubits;
    int num_negative_control_qubits;
    uint64_t is_measured_bits;
    uint64_t measured_value_bits;
    int qubit_num_list[1];

    static __host__ __device__ uint64_t needed_size(
        int const num_positive_control_qubits,
        int const num_negative_control_qubits,
        int const num_target_qubits
    ) {
        return
            sizeof(qcs::kernel_input_qnlist_struct)
            - sizeof(qubit_num_list)
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
}; /* kernel_input_qnlist_struct */

static __device__ void thread_num_to_state_index(uint64_t thread_num, uint64_t& index_state_0, uint64_t& index_state_1, int& measured_state) {
    auto args = (qcs::kernel_input_qnlist_struct const*)(void*)qcs::kernel_input_constant;

    index_state_0 = 0;

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
    index_state_1 = index_state_0 | (1ULL << target_qubit_num);

    if (args->is_measured_bits & 1) {
        if (args->measured_value_bits & 1) {
            measured_state = 1;
        } else {
            measured_state = 0;
        }
    } else {
        measured_state = -1;
    }

} /* thread_num_to_state_index */

namespace gate {

    struct hadamard {
        __device__ void apply(qcs::complex_t const s0_in, qcs::complex_t const s1_in, qcs::complex_t& s0_out, qcs::complex_t& s1_out) const {
            s0_out = s0_in + s1_in;
            s1_out = s0_in - s1_in;
        }
    };

    struct identity {
        __device__ void apply(qcs::complex_t const s0_in, qcs::complex_t const s1_in, qcs::complex_t& s0_out, qcs::complex_t& s1_out) const {
            s0_out = s0_in;
            s1_out = s1_in;
        }
    };

    struct x {
        __device__ void apply(qcs::complex_t const s0_in, qcs::complex_t const s1_in, qcs::complex_t& s0_out, qcs::complex_t& s1_out) const {
            s0_out = s1_in;
            s1_out = s0_in;
        }
    };

}

__global__ void clear_alternative_state(int target_qubit_num, int measured_value) {
    int64_t const thread_num = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;

    // generate index_state_0
    uint64_t lower_mask = 0;
    uint64_t const mask = (1ULL << (target_qubit_num)) - 1;
    uint64_t const upper_mask = mask & ~lower_mask;
    lower_mask = mask;
    uint64_t index_state = (thread_num & upper_mask) | ((thread_num & ~lower_mask) << 1);

    if (!measured_value) {
        index_state = index_state | (1ULL << target_qubit_num);
    }

    qcs::kernel_common_constant.state_data_device[index_state] = 0.0;

}

template<typename GateType>
__global__ void cuda_gate(GateType const gateobj) {

    int64_t const thread_num = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;

    uint64_t index_state_0, index_state_1;
    int measured_state;
    thread_num_to_state_index(thread_num, index_state_0, index_state_1, measured_state);

    qcs::complex_t const s0_in = (measured_state != 1)? qcs::kernel_common_constant.state_data_device[index_state_0] : 0;
    qcs::complex_t const s1_in = (measured_state != 0)? qcs::kernel_common_constant.state_data_device[index_state_1] : 0;

    gateobj.apply(s0_in, s1_in, qcs::kernel_common_constant.state_data_device[index_state_0], qcs::kernel_common_constant.state_data_device[index_state_1]);
}

namespace cubUtility {

    struct float2Add {
        __device__ qcs::float2_t operator()(const qcs::float2_t& a, const qcs::float2_t& b) const {
            return {a[0] + b[0], a[1] + b[1]};
        }
    };

    struct IndirectLoad
    {
        __device__ qcs::float2_t operator()(uint64_t thread_num) const
        {
            uint64_t index_state_0, index_state_1;
            int measured_state;
            thread_num_to_state_index(thread_num, index_state_0, index_state_1, measured_state);

            // since target_qubit must be unmeasured, branching is not necessary.
            // return qcs::float2_t{
            //     cuda::std::norm(qcs::kernel_common_constant.state_data_device[index_state_0]),
            //     cuda::std::norm(qcs::kernel_common_constant.state_data_device[index_state_1])
            // };

            return qcs::float2_t{
                (measured_state != 1)? cuda::std::norm(qcs::kernel_common_constant.state_data_device[index_state_0]): 0,
                (measured_state != 0)? cuda::std::norm(qcs::kernel_common_constant.state_data_device[index_state_1]): 0
            };

        }
    };

} /* cubUtility */

enum class initstate_enum {
    sequential,
    flat,
    zero,
    entangled,
    use_curand,
    laod_statevector,
};

struct simulator {

/* begin simulator variables */

int num_qubits;

bool use_unified_memory;

initstate_enum initstate_choice;

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
std::mt19937 engine;
int num_rand_areas_times_num_procs;

int log_num_procs;
int log_block_size_max;
int block_size_max;
int target_qubit_num_begin;
int target_qubit_num_end;

cudaStream_t stream;
cudaEvent_t event_1;
cudaEvent_t event_2;

uint64_t num_states;
int num_qubits_local;
uint64_t num_states_local;

uint64_t num_blocks_gateop;
uint64_t block_size_gateop;
uint64_t num_operand_qubits;

qcs::complex_t* state_data_device;
qcs::kernel_common_struct* qcs_kernel_common_constant_addr;
qcs::kernel_common_struct qcs_kernel_common_host;
qcs::kernel_input_qnlist_struct* qcs_kernel_input_constant_addr;
std::vector<char> qcs_kernel_input_host_buffer;
int log_swap_buffer_total_length;
uint64_t swap_buffer_total_length;
qcs::complex_t* swap_buffer;
qcs::float2_t* measure_norm_device;

void* cub_temp_buffer_device;
uint64_t cub_temp_buffer_device_size;

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
std::vector<int> measured_1_qubit_num_logical_list_copy;
std::vector<int> measured_0_qubit_num_logical_list_copy;

std::vector<int> target_qubit_num_logical_list;
std::vector<int> positive_control_qubit_num_logical_list;
std::vector<int> negative_control_qubit_num_logical_list;

bool control_condition;

/* end simulator variables */

simulator() :
rng_seed(123456),
use_unified_memory(false),
num_rand_areas_times_num_procs(8)
{}

void setup() {

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_num);

    if (proc_num==0) {
        fprintf(stderr, "[info] num_procs=%d\n", num_procs);
    }

    if (num_rand_areas_times_num_procs < num_procs) {
        throw std::runtime_error(atlc::format("num_rand_areas_times_num_procs %d < num_procs %d", num_rand_areas_times_num_procs, num_procs));
    }
    num_rand_areas = num_rand_areas_times_num_procs / num_procs;

    atlc::group_by_hostname(proc_num, num_procs, my_hostname, my_node_number, my_node_local_rank, node_count);
    fprintf(stderr,
            "[debug] Rank %d on host %s -> assigned node number: %d, local node rank: %d (total nodes: %d)\n",
            proc_num, my_hostname.c_str(), my_node_number, my_node_local_rank, node_count);

    gpu_id = my_node_local_rank;
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

    log_num_procs = atlc::log2_int(num_procs);

    log_block_size_max = 9;

    if (proc_num == 0) { fprintf(stderr, "[info] log_block_size_max=%d\n", log_block_size_max); }

    ATLC_CHECK_CUDA(cudaStreamCreate, &stream);
    ATLC_CHECK_CUDA(cudaEventCreateWithFlags, &event_1, cudaEventDefault);
    ATLC_CHECK_CUDA(cudaEventCreateWithFlags, &event_2, cudaEventDefault);

    num_states = 1ULL << num_qubits;

    num_qubits_local = num_qubits - log_num_procs;

    num_states_local = 1ULL << num_qubits_local;
    block_size_max = 1 << log_block_size_max;

    if (proc_num == 0) { fprintf(stderr, "[info] malloc device memory\n"); }

    if (use_unified_memory) {
        ATLC_CHECK_CUDA(cudaMallocManaged, &state_data_device, num_states_local * sizeof(*state_data_device));
        ATLC_CHECK_CUDA(cudaMemAdvise, state_data_device, num_states_local * sizeof(*state_data_device), cudaMemAdviseSetPreferredLocation, gpu_id);
    } else {
        ATLC_CHECK_CUDA(cudaMallocAsync, &state_data_device, num_states_local * sizeof(*state_data_device), stream);
    }

    ATLC_CHECK_CUDA(cudaGetSymbolAddress, (void**)&qcs_kernel_common_constant_addr, qcs::kernel_common_constant);

    qcs_kernel_common_host.num_qubits = num_qubits;
    qcs_kernel_common_host.state_data_device = state_data_device;
    ATLC_CHECK_CUDA(cudaMemcpyAsync, qcs_kernel_common_constant_addr, &qcs_kernel_common_host, sizeof(qcs::kernel_common_struct), cudaMemcpyHostToDevice, stream);

    ATLC_CHECK_CUDA(cudaGetSymbolAddress, (void**)&qcs_kernel_input_constant_addr, qcs::kernel_input_constant);

    log_swap_buffer_total_length = (num_qubits_local>30)? num_qubits_local - 3 : num_qubits_local;
    swap_buffer_total_length = 1ULL << log_swap_buffer_total_length;
    ATLC_CHECK_CUDA(cudaMallocAsync, &swap_buffer, swap_buffer_total_length * sizeof(qcs::complex_t), stream);

    ATLC_CHECK_CUDA(cudaMallocAsync, &measure_norm_device, sizeof(qcs::complex_t), stream);

    ATLC_CHECK_CUDA(cudaMallocAsync, &cub_temp_buffer_device, 1, stream);
    cub_temp_buffer_device_size = 1;

    engine = std::mt19937(rng_seed + 1);
    measured_0_qubit_num_logical_list.clear();
    measured_1_qubit_num_logical_list.clear();

    MPI_Barrier(MPI_COMM_WORLD);
} /* setup */

void initialize_sequential() {

    uint64_t num_blocks_init;
    uint64_t block_size_init;
    if (num_qubits_local >= log_block_size_max) {
        num_blocks_init = num_states_local >> log_block_size_max;
        block_size_init = block_size_max;
    } else {
        num_blocks_init = 1;
        block_size_init = num_states_local;
    }

    ATLC_CHECK_CUDA(atlc::cudaLaunchKernel, qcs::initstate_sequential_kernel, num_blocks_init, block_size_init, 0, stream, state_data_device, proc_num);

} /* initialize_sequential */

void initialize_flat() {

    uint64_t num_blocks_init;
    uint64_t block_size_init;
    if (num_qubits_local >= log_block_size_max) {
        num_blocks_init = num_states_local >> log_block_size_max;
        block_size_init = block_size_max;
    } else {
        num_blocks_init = 1;
        block_size_init = num_states_local;
    }

    ATLC_CHECK_CUDA(atlc::cudaLaunchKernel, qcs::initstate_flat_kernel, num_blocks_init, block_size_init, 0, stream, state_data_device);

} /* initialize_flat */

void initialize_zero() {
    if (proc_num == 0) {
        qcs::complex_t const one = 1;
        ATLC_CHECK_CUDA(cudaMemcpyAsync, state_data_device, &one, sizeof(qcs::complex_t), cudaMemcpyHostToDevice, stream);
    }
}

void initialize_entangled() {
    if (proc_num == 0) {
        qcs::complex_t const one = 1;
        ATLC_CHECK_CUDA(cudaMemcpyAsync, state_data_device, &one, sizeof(qcs::complex_t), cudaMemcpyHostToDevice, stream);
    } else if (proc_num == num_procs - 1) {
        qcs::complex_t const one = 1;
        ATLC_CHECK_CUDA(cudaMemcpyAsync, state_data_device + num_states_local - 1, &one, sizeof(qcs::complex_t), cudaMemcpyHostToDevice, stream);
    }
}

void initialize_use_curand() {

    MPI_Barrier(MPI_COMM_WORLD);
    if (proc_num == 0) { fprintf(stderr, "[info] generating random state\n"); }
    curandGenerator_t rng_device;

    ATLC_CHECK_CUDA(cudaEventRecord, event_1, stream);

    {
        int const log_num_rand_areas = atlc::log2_int(num_rand_areas);
        uint64_t const num_states_rand_area = num_states_local >> log_num_rand_areas;
        for (int rand_area_num = 0; rand_area_num < num_rand_areas; rand_area_num++) {
            ATLC_CHECK_CURAND(curandCreateGenerator, &rng_device, CURAND_RNG_PSEUDO_DEFAULT);
            ATLC_CHECK_CURAND(curandSetStream, rng_device, stream);
            ATLC_CHECK_CURAND(curandSetPseudoRandomGeneratorSeed, rng_device, rng_seed + proc_num * num_rand_areas + rand_area_num);
            ATLC_CHECK_CURAND(curandGenerateNormalDouble, rng_device, (qcs::float_t*)(void*)(state_data_device + num_states_rand_area * ((uint64_t)rand_area_num)), num_states_rand_area * 2 /* complex */, 0.0, 1.0);
            ATLC_CHECK_CURAND(curandDestroyGenerator, rng_device);
        }
    }
} /* initialize_use_curand */

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
} /* initialize_laod_statevector */

void fputs_vector_int(std::vector<int> data) {
    fprintf(stderr, "{");
    for (int value: data) {
        fprintf(stderr, "%d, ", value);
    }
    fprintf(stderr, "}");
}

void prepare_control_qubit_num_list() {

    // fflush(stderr); fflush(stdout);
    // MPI_Barrier(MPI_COMM_WORLD);
    // if (proc_num==0) {
    //     fprintf(stderr, "debug: measured_0_qubit_num_logical_list=");
    //     fputs_vector_int(measured_0_qubit_num_logical_list);
    //     fprintf(stderr, "\n");

    //     fprintf(stderr, "debug: measured_1_qubit_num_logical_list=");
    //     fputs_vector_int(measured_1_qubit_num_logical_list);
    //     fprintf(stderr, "\n");

    //     fprintf(stderr, "debug: negative_control_qubit_num_logical_list=");
    //     fputs_vector_int(negative_control_qubit_num_logical_list);
    //     fprintf(stderr, "\n");

    //     fprintf(stderr, "debug: positive_control_qubit_num_logical_list=");
    //     fputs_vector_int(positive_control_qubit_num_logical_list);
    //     fprintf(stderr, "\n");

    //     fprintf(stderr, "debug: target_qubit_num_logical_list=");
    //     fputs_vector_int(target_qubit_num_logical_list);
    //     fprintf(stderr, "\n");
    //     fflush(stderr);
    // }
    // MPI_Barrier(MPI_COMM_WORLD);

    std::vector<int>* const measured_X_qubit_num_logical_list_list[] = {
        &measured_0_qubit_num_logical_list,
        &measured_1_qubit_num_logical_list
    };

    std::vector<int>* const measured_X_qubit_num_logical_list_copy_list[] = {
        &measured_0_qubit_num_logical_list_copy,
        &measured_1_qubit_num_logical_list_copy
    };

    std::vector<int>* const X_control_qubit_num_logical_list_list[] = {
        &negative_control_qubit_num_logical_list,
        &positive_control_qubit_num_logical_list
    };

    control_condition = true;

    #pragma unroll
    for(int measured_value = 0; measured_value < 2; measured_value++) {
        measured_X_qubit_num_logical_list_copy_list[measured_value]->clear();
        for (int mXqnl_idx = 0; mXqnl_idx < measured_X_qubit_num_logical_list_list[measured_value]->size(); mXqnl_idx++) {
            auto const mXqn = measured_X_qubit_num_logical_list_list[measured_value]->operator[](mXqnl_idx);
            bool const is_target = std::find(target_qubit_num_logical_list.begin(), target_qubit_num_logical_list.end(), mXqn) != target_qubit_num_logical_list.end();
            if (!is_target) {
                // the qubit is kept measured
                measured_X_qubit_num_logical_list_copy_list[measured_value]->push_back(mXqn);
                bool const is_control = std::find(X_control_qubit_num_logical_list_list[measured_value]->begin(), X_control_qubit_num_logical_list_list[measured_value]->end(), mXqn) != X_control_qubit_num_logical_list_list[measured_value]->end();
                bool const is_control_other = std::find(X_control_qubit_num_logical_list_list[1-measured_value]->begin(), X_control_qubit_num_logical_list_list[1-measured_value]->end(), mXqn) != X_control_qubit_num_logical_list_list[1-measured_value]->end();
                if ((!is_control)&&(!is_control_other)) {
                    X_control_qubit_num_logical_list_list[measured_value]->push_back(mXqn);
                } else if (is_control_other) {
                    control_condition = false;
                }
            }
        }
    }

    // if (proc_num==0) {
    //     fprintf(stderr, "debug: control_condition=%d\n", (int)control_condition);
    // }

} /* prepare_control_qubit_num_list */

void ensure_local_qubits() {
    target_qubit_num_physical_list.resize(target_qubit_num_logical_list.size());
    for (int tqni = 0; tqni < target_qubit_num_logical_list.size(); tqni++) {
        target_qubit_num_physical_list[tqni] = perm_l2p[target_qubit_num_logical_list[tqni]];
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
}; /* ensure_local_qubits */

void check_control_qubit_num_physical() {

    /* check whether proc_num is under control condition */
    control_condition = true;

    std::vector<int>* x_control_qubit_num_logical_list_list[] = {&negative_control_qubit_num_logical_list, &positive_control_qubit_num_logical_list};
    std::vector<int>* x_control_qubit_num_physical_list_list[] = {&negative_control_qubit_num_physical_list, &positive_control_qubit_num_physical_list};
    std::vector<int>* x_control_qubit_num_physical_global_list_list[] = {&negative_control_qubit_num_physical_global_list, &positive_control_qubit_num_physical_global_list};
    std::vector<int>* x_control_qubit_num_physical_local_list_list[] = {&negative_control_qubit_num_physical_local_list, &positive_control_qubit_num_physical_local_list};

    #pragma unroll
    for (int control_np = 0; control_np < 2; control_np++) {
        x_control_qubit_num_physical_list_list[control_np]->resize(x_control_qubit_num_logical_list_list[control_np]->size());
        x_control_qubit_num_physical_global_list_list[control_np]->resize(0);
        x_control_qubit_num_physical_local_list_list[control_np]->resize(0);

        for (int cqni = 0; cqni < x_control_qubit_num_logical_list_list[control_np]->size(); cqni++) {
            auto const x_control_qubit_num_physical = perm_l2p[x_control_qubit_num_logical_list_list[control_np]->operator[](cqni)];
            x_control_qubit_num_physical_list_list[control_np]->operator[](cqni) = x_control_qubit_num_physical;
            if (x_control_qubit_num_physical >= num_qubits_local) {
                x_control_qubit_num_physical_global_list_list[control_np]->push_back(x_control_qubit_num_physical);
                if ((1 & (proc_num >> (x_control_qubit_num_physical - num_qubits_local))) != control_np) {
                    control_condition = false;
                }
            } else {
                x_control_qubit_num_physical_local_list_list[control_np]->push_back(x_control_qubit_num_physical);
            }
        }
    }

}; /* check_control_qubit_num_physical */

void prepare_operating_gate() {

    if (!control_condition) { return; }

    uint64_t const qkiqn_size = qcs::kernel_input_qnlist_struct::needed_size(
        positive_control_qubit_num_physical_list.size(),
        negative_control_qubit_num_physical_list.size(),
        target_qubit_num_physical_list.size()
    );
    if (qkiqn_size > qcs::kernel_input_max_size) {
        throw std::runtime_error(atlc::format("qkiqn_size(%" PRIu64 ") > qcs::kernel_input_max_size(%" PRIu64 ")", qkiqn_size, qcs::kernel_input_max_size));
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
    qcs_kernel_input_host->is_measured_bits = 0;
    qcs_kernel_input_host->measured_value_bits = 0;

    for (int tqi = 0; tqi < target_qubit_num_physical_list.size(); tqi++) {
        auto const tqn_phys = target_qubit_num_physical_list[tqi];
        target_qubit_num_list_kernel_arg[tqi] = tqn_phys;

        bool is_measured = false;
        int measured_value = 0;

        std::vector<int>* const measured_X_qubit_num_logical_list_list[] = {
            &measured_0_qubit_num_logical_list,
            &measured_1_qubit_num_logical_list
        };

        for (int loop_measured_value = 0; loop_measured_value < 2; loop_measured_value++) {
            for(int mXqnl_idx = 0; mXqnl_idx < measured_X_qubit_num_logical_list_list[loop_measured_value]->size(); mXqnl_idx++) {
                auto const mXqn_phys = perm_l2p[measured_X_qubit_num_logical_list_list[loop_measured_value]->operator[](mXqnl_idx)];
                if (mXqn_phys == tqn_phys) {
                    is_measured = true;
                    measured_value = loop_measured_value;
                    break;
                }
            }
            if (is_measured) { break; }
        }

        if (is_measured) {
            qcs_kernel_input_host->is_measured_bits |= 1ULL << tqi;
            if (measured_value /* == 1 */) {
                qcs_kernel_input_host->measured_value_bits |= 1ULL << tqi;
            }
        }

    }

    num_operand_qubits =
        positive_control_qubit_num_physical_local_list.size()
        + negative_control_qubit_num_physical_local_list.size()
        + target_qubit_num_physical_list.size();

    /* get sorted operand qubits */
    operand_qubit_num_list.clear();
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

    if (log_block_size_max > log_num_threads) {
        log_block_size_gateop = log_num_threads;
        num_blocks_gateop = 1;
    } else {
        log_block_size_gateop = log_block_size_max;
        num_blocks_gateop = 1ULL << (log_num_threads - log_block_size_max);
    }


    block_size_gateop = 1ULL << log_block_size_gateop;

    // if (num_blocks_gateop==0||block_size_gateop==0) {
    //     DEBUG_UINT_VAR(num_blocks_gateop);
    //     DEBUG_UINT_VAR(block_size_gateop);
    //     DEBUG_UINT_VAR(log_num_threads);
    //     DEBUG_UINT_VAR(log_block_size_max);
    // }
    
} /* prepare_operating_gate */

void update_measured_list() {
    // if (proc_num==0) {
    //     fprintf(stderr, "debug: update_measured_list\n");
    // }
    measured_0_qubit_num_logical_list = measured_0_qubit_num_logical_list_copy;
    measured_1_qubit_num_logical_list = measured_1_qubit_num_logical_list_copy;
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
                    auto const errno_saved = errno;
                    throw std::runtime_error(atlc::format("errorno=%d ret_fseek=%d", errno_saved, ret_fseek));
                }
                size_t const ret = fwrite(&state_data_host[state_num_physical_local], sizeof(qcs::complex_t), 1, fp);
                if (ret != 1) {
                    auto const errno_saved = errno;
                    throw std::runtime_error(atlc::format("fwrite failed ret=%zu errno=%d", ret, errno_saved));
                }
            }
            fflush(fp);
            fclose(fp);
            fsync(fileno(fp));
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
} /* save_statevector */

void calculate_checksum() {

    if (proc_num==0) {
        fprintf(stderr, "[info] gathering state data\n");

        EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
        if (!mdctx) {
            throw std::runtime_error("EVP_MD_CTX_new failed");
        }
        ATLC_DEFER_FUNC(EVP_MD_CTX_free, mdctx);

        if (EVP_DigestInit_ex(mdctx, EVP_md5(), NULL) != 1) {
            throw std::runtime_error("EVP_DigestInit_ex failed");
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
                throw std::runtime_error("EVP_DigestUpdate failed");
            }
        }

        std::vector<unsigned char> evp_hash(EVP_MAX_MD_SIZE); // [EVP_MAX_MD_SIZE];
        unsigned int evp_hash_len;
        if (EVP_DigestFinal_ex(mdctx, evp_hash.data(), &evp_hash_len) != 1) {
            throw std::runtime_error("EVP_DigestFinal_ex failed");
        }

        fprintf(stderr, "[info] checksum: ");
        for (unsigned int i = 0; i < evp_hash_len; i++) {
            fprintf(stderr, "%02x", evp_hash[i]);
        }
        fprintf(stderr, "\n");

    } else {
        MPI_Send(state_data_device, num_states_local * 2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

} /* calculate_checksum */


void show_state() {
    qcs::complex_t* state_data_host = (qcs::complex_t*)malloc(num_states_local * sizeof(qcs::complex_t));
    ATLC_DEFER_FUNC(free, state_data_host);
    ATLC_CHECK_CUDA(cudaMemcpyAsync, state_data_host, state_data_device, num_states_local * sizeof(qcs::complex_t), cudaMemcpyDeviceToHost, stream);
    ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);

    for (uint64_t state_num = 0; state_num < num_states_local; state_num++) {

        bool state_num_control_condition = true;

        for(int pcqn: positive_control_qubit_num_physical_local_list) {
            if (!((state_num>>pcqn)&1)) {
                state_num_control_condition = false;
            }
        }
        if (!state_num_control_condition) continue;

        for(int ncqn: negative_control_qubit_num_physical_local_list) {
            if (((state_num>>ncqn)&1)) {
                state_num_control_condition = false;
            }
        }
        if (!state_num_control_condition) continue;

        auto real = state_data_host[state_num].real();
        auto imag = state_data_host[state_num].imag();
        fprintf(stderr, ATLC_REORDER("debug: " VAR_STR_INT(proc_num), " " VAR_STR_UINT(state_num), " " VAR_STR_FLOAT(real), " " VAR_STR_FLOAT(imag), "\n"));
        // int tqn = target_qubit_num_physical_list[0];
        // if ((state_num>>tqn)&1) {
        //     measure_norm_host[1] += norm;
        // } else {
        //     measure_norm_host[0] += norm;
        // }
    }
}

int measure_qubit(int const measure_qubit_num_logical) {

    for (auto const m0qn : measured_0_qubit_num_logical_list) {
        if (measure_qubit_num_logical == m0qn) {
            return 0;
        }
    }

    for (auto const m1qn : measured_1_qubit_num_logical_list) {
        if (measure_qubit_num_logical == m1qn) {
            return 1;
        }
    }

    target_qubit_num_logical_list = {measure_qubit_num_logical};
    positive_control_qubit_num_logical_list = measured_1_qubit_num_logical_list;
    negative_control_qubit_num_logical_list = measured_0_qubit_num_logical_list;

    ensure_local_qubits();
    check_control_qubit_num_physical();
    prepare_operating_gate();

    float2_t measure_norm_host{0, 0};

    if (control_condition) {

        cubUtility::IndirectLoad loader;

        using CountingIter = cub::CountingInputIterator<uint64_t>;
        using TransformIter = cub::TransformInputIterator<float2_t, decltype(loader), CountingIter>;

        CountingIter counting(0);
        TransformIter in_it(counting, loader);

        uint64_t temp_sz_required;

        cubUtility::float2Add float2AddObj;
        float2_t zero{0, 0};
        ATLC_CHECK_CUDA(cub::DeviceReduce::Reduce, NULL, temp_sz_required, in_it, measure_norm_device, num_states_local >> num_operand_qubits, float2AddObj, zero, stream);

        if (cub_temp_buffer_device_size < temp_sz_required) {
            ATLC_CHECK_CUDA(cudaFreeAsync, cub_temp_buffer_device, stream);
            ATLC_CHECK_CUDA(cudaMallocAsync, &cub_temp_buffer_device, temp_sz_required, stream);
            cub_temp_buffer_device_size = temp_sz_required;
        }

        ATLC_CHECK_CUDA(cub::DeviceReduce::Reduce, cub_temp_buffer_device, cub_temp_buffer_device_size, in_it, measure_norm_device, num_states_local >> num_operand_qubits, float2AddObj, zero, stream);

        ATLC_CHECK_CUDA(cudaMemcpyAsync, &measure_norm_host, measure_norm_device, sizeof(float2_t), cudaMemcpyDeviceToHost, stream);

        ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);

        // show_state();
    } else {
        measure_norm_host = {0, 0};
    }

    // DEBUG_FLOAT_VAR(measure_norm_host[0]);
    // DEBUG_FLOAT_VAR(measure_norm_host[1]);
    // fprintf(stderr, "debug: proc_num=%d control_condition=%d measure_qubit_num_logical=%llu measure_norm_host[0]=%e measure_norm_host[1]=%e\n", proc_num, (int)control_condition, measure_qubit_num_logical, measure_norm_host[0], measure_norm_host[1]);

#if 1 /* parallel measurement */
    qcs::float_t measure_norm_global[2] = {0, 0};
    MPI_Allreduce(measure_norm_host.data(), measure_norm_global, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    // if (proc_num==0) {
    //     fprintf(stderr, "debug: measure_qubit_num_logical=%llu measure_norm_global[0]=%e measure_norm_global[1]=%e\n", measure_qubit_num_logical, measure_norm_global[0], measure_norm_global[1]);
    //     // DEBUG_UINT_VAR(measure_qubit_num_logical);
    //     // DEBUG_FLOAT_VAR(measure_norm_global[0]);
    //     // DEBUG_FLOAT_VAR(measure_norm_global[1]);
    // }

    qcs::float_t const measure_norm_sum = measure_norm_global[0] + measure_norm_global[1];

    std::uniform_real_distribution<qcs::float_t> dist1(0, measure_norm_sum);
    qcs::float_t const random_value = dist1(engine);
    int measure_result = measure_norm_global[0] < random_value;

    if (measure_result) { /* 1 */
        measured_1_qubit_num_logical_list.push_back(measure_qubit_num_logical);
        // measured_bit |= 1ULL << measure_qubit_num_logical;
    } else { /* 0 */
        measured_0_qubit_num_logical_list.push_back(measure_qubit_num_logical);
    }
#else /* measurement by master */
    qcs::float_t measure_norm_global[2];
    MPI_Reduce(measure_norm_host.data(), measure_norm_global, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    bool measure_result_0;
    if (proc_num == 0) {
        qcs::float_t const measure_norm_sum = measure_norm_global[0] + measure_norm_global[1];

        std::uniform_real_distribution<qcs::float_t> dist1(0, measure_norm_sum);
        qcs::float_t const random_value = dist1(engine);
        measure_result_0 = measure_norm_global[0] < random_value;
    }
    bool measure_result;
    MPI_Scatter(&measure_result_0, 1, MPI_C_BOOL, &measure_result, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    if (measure_result) { /* 1 */
        measured_1_qubit_num_logical_list.push_back(measure_qubit_num_logical);
        // measured_bit |= 1ULL << measure_qubit_num_logical;
    } else { /* 0 */
        measured_0_qubit_num_logical_list.push_back(measure_qubit_num_logical);
    }
#endif

    {
        uint64_t const log_num_threads = num_qubits_local - 1;
        uint64_t log_block_size_gateop;
    
        if (log_block_size_max > log_num_threads) {
            log_block_size_gateop = log_num_threads;
            num_blocks_gateop = 1;
        } else {
            log_block_size_gateop = log_block_size_max;
            num_blocks_gateop = 1ULL << (log_num_threads - log_block_size_max);
        }

        block_size_gateop = 1ULL << log_block_size_gateop;

        auto const measure_qubit_num_physical = perm_l2p[measure_qubit_num_logical];
        ATLC_CHECK_CUDA(atlc::cudaLaunchKernel, clear_alternative_state, num_blocks_gateop, block_size_gateop, 0, stream, measure_qubit_num_physical, measure_result);
    }

    return measure_result;
}

template<typename GateType>
void operate_gate(GateType gateobj) {

    prepare_control_qubit_num_list();
    if (!control_condition) return;
    // fprintf(stderr, "debug: 1037\n");

    ensure_local_qubits();
    check_control_qubit_num_physical();
    prepare_operating_gate();

    if (control_condition) {
        // fprintf(stderr, "debug: 1042\n");
        // if (proc_num==0) {
        // fprintf(stderr, "debug: proc_num=%d num_blocks_gateop=%lu, block_size_gateop=%lu\n", proc_num, (uint64_t)num_blocks_gateop, (uint64_t)block_size_gateop);
        // }
        // fflush(stderr);
        ATLC_CHECK_CUDA(atlc::cudaLaunchKernel, cuda_gate<GateType>, num_blocks_gateop, block_size_gateop, 0, stream, gateobj);
    }

    // fprintf(stderr, "debug: 1044\n");
    update_measured_list();

}

void GHZ_circuit_sample() {

    // initialize_zero();
    initialize_flat();

    // measured_bit = 0;
    uint64_t measured_bit = 0;

    for (int measure_qubit_num_logical = 0; measure_qubit_num_logical < num_qubits; measure_qubit_num_logical++) {
        int const measured_value = measure_qubit(measure_qubit_num_logical);
        if (measured_value) {
            measured_bit |= 1ULL << measure_qubit_num_logical;
        }
    }

    if (proc_num == 0) {
        fprintf(stdout, "%" PRIu64 "\n", measured_bit);
    }

    // uint64_t measured_bit = 0;
    uint64_t const num_samples = 1ULL << num_qubits;

    for(int sample_num = 0; sample_num < num_samples; ++sample_num) {

        /* begin gate operation */

        target_qubit_num_logical_list = {0};
        positive_control_qubit_num_logical_list = {};
        negative_control_qubit_num_logical_list = {};

        operate_gate(gate::hadamard());

        for(int target_qubit_num_logical = 1; target_qubit_num_logical < num_qubits; target_qubit_num_logical++)
        {
            target_qubit_num_logical_list = {target_qubit_num_logical};
            if ((measured_bit>>target_qubit_num_logical)&1) {
                positive_control_qubit_num_logical_list = {};
                negative_control_qubit_num_logical_list = {0};
            } else {
                positive_control_qubit_num_logical_list = {0};
                negative_control_qubit_num_logical_list = {};
            }

            operate_gate(gate::x());

        } /* target_qubit_num_logical loop */

        /* end gate operation */
        // if (proc_num==0) {
        //     fprintf(stderr, "debug: measured_0_qubit_num_logical_list.size()=%llu, measured_1_qubit_num_logical_list.size()=%llu\n", measured_0_qubit_num_logical_list.size(), measured_1_qubit_num_logical_list.size());
        // }

        /* begin measurement */
        measured_bit = 0;

        for (int measure_qubit_num_logical = 0; measure_qubit_num_logical < num_qubits; measure_qubit_num_logical++) {
            int const measured_value = measure_qubit(measure_qubit_num_logical);
            if (measured_value) {
                measured_bit |= 1ULL << measure_qubit_num_logical;
            }
        }

        if (proc_num == 0) {
            fprintf(stdout, "%" PRIu64 "\n", measured_bit);
        }
        /* end measurement */

    }

}; /* GHZ_circuit_sample */



void measurement_sample() {

    constexpr initstate_enum initstate_choice = initstate_enum::flat;
    switch (initstate_choice) {
        case initstate_enum::sequential:
            initialize_sequential();
            break;
        case initstate_enum::flat:
            initialize_flat();
            break;
        case initstate_enum::zero:
            initialize_zero();
            break;
        case initstate_enum::entangled:
            initialize_entangled();
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


    uint64_t measured_bit = 0;
    uint64_t const num_samples = 1ULL << num_qubits;

    for(int sample_num = 0; sample_num < num_samples; ++sample_num) {

        measured_0_qubit_num_logical_list.clear();
        measured_1_qubit_num_logical_list.clear();

        /* begin measurement */
        measured_bit = 0;

        for (int measure_qubit_num_logical = 0; measure_qubit_num_logical < num_qubits; measure_qubit_num_logical++) {
            int const measured_value = measure_qubit(measure_qubit_num_logical);
            if (measured_value) {
                measured_bit |= 1ULL << measure_qubit_num_logical;
            }
        }

        if (proc_num == 0) {
            fprintf(stdout, "%" PRIu64 "\n", measured_bit);
        }
        /* end measurement */

    }

} /* measurement_sample */

int main() {

    constexpr bool flag_calculate_checksum = false;
    constexpr bool flag_save_statevector = false;

    setup();
    ATLC_DEFER_FUNC(dispose);

    // measurement_sample();
    GHZ_circuit_sample();

    if (flag_save_statevector) { save_statevector(); }
    if (flag_calculate_checksum) { calculate_checksum(); }

    MPI_Finalize();

    return 0;

}; /* main */


void dispose() {
    ATLC_CHECK_CUDA(cudaFreeAsync, cub_temp_buffer_device, stream);
    ATLC_CHECK_CUDA(cudaFreeAsync, state_data_device, stream);
    ATLC_CHECK_CUDA(cudaFreeAsync, swap_buffer, stream);
    ATLC_CHECK_CUDA(cudaEventDestroy, event_1);
    ATLC_CHECK_CUDA(cudaEventDestroy, event_2);
    ATLC_CHECK_CUDA(cudaStreamDestroy, stream);
};

}; /* simulator */

} /* qcs */

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IOLBF, 1024 * 512);
    qcs::simulator simulator;
    return simulator.main();
}