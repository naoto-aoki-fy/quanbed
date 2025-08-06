#pragma once
#include <vector>
#include <stdexcept>
#include <cstdint>

namespace qcs {
    struct simulator_core;
    class simulator {
    private:
        simulator_core* core;
        int num_qubits;
        void ensure_qubits_allocated();
        std::vector<uint8_t> clbits;
    public:
        simulator();
        void setup();
        void dispose();

        int get_num_procs();
        int get_proc_num();

        inline void promise_qubits(int num_qubits) {
            this->num_qubits += num_qubits;
        }

        inline void alloc_clbits(int num_clbits) {
            clbits.resize(clbits.size() + num_clbits);
        }
        inline uint8_t clbits_get(std::size_t idx) {
            return clbits[idx];
        }
        inline uint8_t clbits_set(std::size_t idx, uint8_t value) {
            return (clbits[idx] = value);
        }

        void reset();
        void set_zero_state();
        void set_sequential_state();
        void set_flat_state();
        void set_entangled_state();
        void set_random_state();

        void hadamard(int target_qubit_num, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);
        inline void hadamard_pow(double exponent, int target_qubit_num, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list) { throw std::runtime_error("not implemented"); }

        void gate_x(int target_qubit_num, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);
        inline void gate_x_pow(double exponent, int target_qubit_num, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list)  { throw std::runtime_error("not implemented"); }

        void gate_u4(double theta, double phi, double lambda, double gamma, int target_qubit_num, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list)  { throw std::runtime_error("not implemented"); }
        inline void gate_u4_pow(double theta, double phi, double lambda, double gamma, double exponent, int target_qubit_num, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list)  { throw std::runtime_error("not implemented"); }

        int measure(int qubit_num);
    };
}