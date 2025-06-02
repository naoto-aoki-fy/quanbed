#pragma once

// #include <nccl.h>
#include "check_x.hpp"

// #if defined(NCCL_H_)

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

// #endif /* NCCL_H_ */