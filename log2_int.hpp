#pragma once

#include <climits>

unsigned int log2_floor_int(unsigned int arg) {
    return sizeof(unsigned int) * CHAR_BIT - __builtin_clz(arg) - 1;
}
unsigned int log2_floor_int(int arg) {
    return log2_floor_int((unsigned int)arg);
}

inline unsigned int log2_int(unsigned int arg) {
    return log2_floor_int(arg);
}
inline unsigned int log2_int(int arg) {
    return log2_floor_int(arg);
}

unsigned int log2_ceil_int(unsigned int arg) {
    return (sizeof(unsigned int) * CHAR_BIT - __builtin_clz(arg) - 1) + ((arg & (arg - 1)) != 0);
}
unsigned int log2_ceil_int(int arg) {
    return log2_ceil_int((unsigned int)arg);
}

#if UINT_MAX != ULONG_MAX
unsigned int log2_floor_int(unsigned long arg) {
    return sizeof(unsigned long) * CHAR_BIT - __builtin_clzl(arg) - 1;
}
unsigned int log2_floor_int(long arg) {
    return log2_floor_int((unsigned long)arg);
}

inline unsigned int log2_int(unsigned long arg) {
    return log2_floor_int(arg);
}
inline unsigned int log2_int(long arg) {
    return log2_floor_int(arg);
}

unsigned int log2_ceil_int(unsigned long arg) {
    return (sizeof(unsigned long) * CHAR_BIT - __builtin_clzl(arg) - 1) + ((arg & (arg - 1)) != 0);
}
unsigned int log2_ceil_int(long arg) {
    return log2_ceil_int((unsigned long)arg);
}
#endif

#if ULONG_MAX != ULLONG_MAX
unsigned int log2_floor_int(unsigned long long arg) {
    return sizeof(unsigned long long) * CHAR_BIT - __builtin_clzll(arg) - 1;
}
unsigned int log2_floor_int(long long arg) {
    return log2_floor_int((unsigned long long)arg);
}

inline unsigned int log2_int(unsigned long long arg) {
    return log2_floor_int(arg);
}
inline unsigned int log2_int(long long arg) {
    return log2_floor_int(arg);
}

unsigned int log2_ceil_int(unsigned long long arg) {
    return (sizeof(unsigned long long) * CHAR_BIT - __builtin_clzll(arg) - 1) + ((arg & (arg - 1)) != 0);
}
unsigned int log2_ceil_int(long long arg) {
    return log2_ceil_int((unsigned long long)arg);
}
#endif