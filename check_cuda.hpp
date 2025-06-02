#pragma once

// #include <cuda_runtime.h>
#include "check_x.hpp"

// #if defined(__CUDA_RUNTIME_H__)

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

// #endif /* __CUDA_RUNTIME_H__ */