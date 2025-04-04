#pragma once

#include <cstdio>

#include <stdexcept>
#include <string_view>
#include <vector>

// #include <mpi.h>
// #include <cuda_runtime.h>
// #include <curand.h>
// #include <nccl.h>

constexpr const char* get_filename(const char* filename_abs) {
    size_t const pos = std::string_view(filename_abs).rfind("/");
    return (pos != std::string_view::npos) ? &filename_abs[pos+1] : filename_abs;
}

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

#define DEFER_CODE(code) Defer UNIQUE_NAME(defer_)([&]()code)

#define DEFER_FUNC(func, ...) DEFER_CODE({func(__VA_ARGS__);})

template <typename Func>
auto check_zero(char const* const filename, int const lineno, char const* const funcname, Func func)
{
    auto err = func();
    if (err != 0)
    {
        std::vector<char> strbuf(1);

        auto printf_lambda = [=](char* strbuf, size_t buf_length){ return snprintf(strbuf, buf_length, "%s:%d:%s error:%d\n", filename, lineno, funcname, err); };

        int str_length = printf_lambda(strbuf.data(), strbuf.size());
        strbuf.resize(str_length + 1);
        str_length = printf_lambda(strbuf.data(), strbuf.size() + 1);

        throw std::runtime_error(strbuf.data());
    }
    return err;
}

#define CHECK_ZERO(func, ...) check_zero(get_filename(__FILE__), __LINE__, #func "(" #__VA_ARGS__ ")", [&](){return func(__VA_ARGS__);})

#define DEFER_CHECK_ZERO(func, ...) DEFER_CODE({CHECK_ZERO(func, __VA_ARGS__);})


template <typename Func>
auto check_nonzero(char const* const filename, int const lineno, char const* const funcname, Func func)
{
    auto err = func();
    if (err == 0)
    {
        std::vector<char> strbuf(1);

        auto printf_lambda = [=](char* strbuf, size_t buf_length){ return snprintf(strbuf, buf_length, "%s:%d:%s error:%d\n", filename, lineno, funcname, err); };

        int str_length = printf_lambda(strbuf.data(), strbuf.size());
        strbuf.resize(str_length + 1);
        str_length = printf_lambda(strbuf.data(), strbuf.size() + 1);

        throw std::runtime_error(strbuf.data());
    }
    return err;
}

#define CHECK_NONZERO(func, ...) check_nonzero(get_filename(__FILE__), __LINE__, #func "(" #__VA_ARGS__ ")", [&](){return func(__VA_ARGS__);})

#define DEFER_CHECK_NONZERO(func, ...) DEFER_CODE({CHECK_NONZERO(func, __VA_ARGS__);})


template <typename CheckerFunc, typename Func>
auto check_value(char const* const filename, int const lineno, CheckerFunc checker_func, char const* const funcname, Func func)
{
    auto value = func();
    auto err = checker_func(value);
    if (err != 0)
    {
        std::vector<char> strbuf(1);

        auto printf_lambda = [=](char* strbuf, size_t buf_length){ return snprintf(strbuf, buf_length, "%s:%d:%s error:%d\n", filename, lineno, funcname, err); };

        int str_length = printf_lambda(strbuf.data(), strbuf.size());
        strbuf.resize(str_length + 1);
        str_length = printf_lambda(strbuf.data(), strbuf.size() + 1);

        throw std::runtime_error(strbuf.data());
    }
    return value;
}


#define CHECK_VALUE(checker_code, func, ...) check_value(get_filename(__FILE__), __LINE__, [&](auto arg)checker_code, #func "(" #__VA_ARGS__ ")", [&](){return func(__VA_ARGS__);})


#if defined(MPI_COMM_WORLD)

template <typename Func>
void check_mpi(char const* const filename, int const lineno, char const* const funcname, Func func)
{
    auto err = func();
    if (err != MPI_SUCCESS)
    {
        std::vector<char> error_string_vector(MPI_MAX_ERROR_STRING);
        int resultlen;
        MPI_Error_string(err, error_string_vector.data(), &resultlen);
        // fprintf(stderr, "[debug] %s:%d call:%s error:%s\n", filename, lineno, funcname, error_string);
        char const* const error_string = error_string_vector.data();

        std::vector<char> strbuf(1);

        auto printf_lambda = [=](char* strbuf, size_t buf_length){ return snprintf(strbuf, buf_length, "%s:%d:%s error:%s\n", filename, lineno, funcname, error_string); };

        int str_length = printf_lambda(strbuf.data(), strbuf.size());
        strbuf.resize(str_length + 1);
        str_length = printf_lambda(strbuf.data(), strbuf.size() + 1);

        throw std::runtime_error(strbuf.data());
    }
}

#define CHECK_MPI(func, ...) check_mpi(get_filename(__FILE__), __LINE__, #func "(" #__VA_ARGS__ ")", [&](){return func(__VA_ARGS__);})

#define DEFER_CHECK_MPI(func, ...) DEFER_CODE({CHECK_MPI(func, __VA_ARGS__);})

#endif /* MPI_COMM_WORLD */

#if defined(__CUDA_RUNTIME_H__)

template <typename Func>
void check_cuda(char const* const filename, int const lineno, char const* const funcname, Func func)
{
    auto err = func();
    if (err != cudaSuccess)
    {
        // fprintf(stderr, "[debug] %s:%d call:%s error:%s\n", filename, lineno, funcname, cudaGetErrorString(err));
        std::vector<char> strbuf(1);
        char const* const error_string = cudaGetErrorString(err);

        auto printf_lambda = [=](char* strbuf, size_t buf_length){ return snprintf(strbuf, buf_length, "%s:%d:%s error:%s\n", filename, lineno, funcname, error_string); };

        int str_length = printf_lambda(strbuf.data(), strbuf.size());
        strbuf.resize(str_length + 1);
        str_length = printf_lambda(strbuf.data(), strbuf.size() + 1);

        throw std::runtime_error(strbuf.data());
    }
}

#define CHECK_CUDA(func, ...) check_cuda(get_filename(__FILE__), __LINE__, #func "(" #__VA_ARGS__ ")", [&](){return func(__VA_ARGS__);})

#define DEFER_CHECK_CUDA(func, ...) DEFER_CODE({CHECK_CUDA(func, __VA_ARGS__);})

#endif /* __CUDA_RUNTIME_H__ */

#if defined(CURAND_H_)

#define CASE_RETURN(code) case code: return #code

inline static char const* curandGetErrorString(curandStatus_t error) {
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
        // fprintf(stderr, "[debug] %s:%d call:%s error:%s\n", filename, lineno, funcname, curandGetErrorString(err));

        std::vector<char> strbuf(1);
        char const* const error_string = curandGetErrorString(err);
        
        auto printf_lambda = [=](char* strbuf, size_t buf_length){ return snprintf(strbuf, buf_length, "%s:%d:%s error:%s\n", filename, lineno, funcname, error_string); };

        int str_length = printf_lambda(strbuf.data(), strbuf.size());
        strbuf.resize(str_length + 1);
        str_length = printf_lambda(strbuf.data(), strbuf.size() + 1);

        throw std::runtime_error(strbuf.data());
    }
}

#define CHECK_CURAND(func, ...) check_curand(get_filename(__FILE__), __LINE__, #func "(" #__VA_ARGS__ ")", [&](){return func(__VA_ARGS__);})

#define DEFER_CHECK_CURAND(func, ...) DEFER_CODE({CHECK_CURAND(func, __VA_ARGS__);})

#endif /* CURAND_H_ */

#if defined(NCCL_H_)

template <typename Func>
void check_nccl(char const* const filename, int const lineno, char const* const funcname, Func func)
{
    auto err = func();
    if (err != ncclSuccess)
    {
        // fprintf(stderr, "[debug] %s:%d call:%s error:%s\n", filename, lineno, funcname, ncclGetErrorString(err));

        char const* const error_string = ncclGetErrorString(err);
        std::vector<char> strbuf(1);

        auto printf_lambda = [=](char* strbuf, size_t buf_length){ return snprintf(strbuf, buf_length, "%s:%d:%s error:%s\n", filename, lineno, funcname, error_string); };

        int str_length = printf_lambda(strbuf.data(), strbuf.size());
        strbuf.resize(str_length + 1);
        str_length = printf_lambda(strbuf.data(), strbuf.size() + 1);

        throw std::runtime_error(strbuf.data());
    }
}

#define CHECK_NCCL(func, ...) check_nccl(get_filename(__FILE__), __LINE__, #func "(" #__VA_ARGS__ ")", [&](){return func(__VA_ARGS__);})

#define DEFER_CHECK_NCCL(func, ...) DEFER_CODE({CHECK_NCCL(func, __VA_ARGS__);})

#endif /* NCCL_H_ */

#if defined(_NVSHMEMX_H_)

template <typename Func>
void check_nvshmemx(char const* const filename, int const lineno, char const* const funcname, Func func)
{
    auto err = func();
    if (err != NVSHMEMX_SUCCESS)
    {
        std::vector<char> strbuf(1);

        auto printf_lambda = [=](char* strbuf, size_t buf_length){ return snprintf(strbuf, buf_length, "%s:%d:%s error:%d\n", filename, lineno, funcname, err); };

        int str_length = printf_lambda(strbuf.data(), strbuf.size());
        strbuf.resize(str_length + 1);
        str_length = printf_lambda(strbuf.data(), strbuf.size() + 1);

        throw std::runtime_error(strbuf.data());
    }
}

#define CHECK_NVSHMEMX(func, ...) check_nvshmem(get_filename(__FILE__), __LINE__, #func "(" #__VA_ARGS__ ")", [&](){ return func(__VA_ARGS__); })


#define DEFER_CHECK_NVSHMEMX(func, ...) DEFER_CODE({CHECK_NVSHMEMX(func, __VA_ARGS__);})

#endif /* _NVSHMEMX_H_ */
