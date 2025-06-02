#pragma once

// #include <mpi.h>
#include "check_x.hpp"

// #if defined(MPI_COMM_WORLD)

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

// #endif /* MPI_COMM_WORLD */