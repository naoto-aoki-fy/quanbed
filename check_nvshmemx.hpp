#pragma once

#include "check_x.hpp"

// #if defined(_NVSHMEMX_H_)

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

// #endif /* _NVSHMEMX_H_ */