#pragma once
// Compile-only stand-in for CUDA's cuda_fp8.h in HIP builds. Maya's kernels call the __nv_* FP8 conversion
// builtins only inside `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890`, a branch no AMD target takes (the
// compat layer defines no __CUDA_ARCH__), so a HIP compile needs these type names alone.
#include <cstdint>

typedef unsigned char __nv_fp8_storage_t;
typedef unsigned short __nv_fp8x2_storage_t;
typedef unsigned int __nv_fp8x4_storage_t;

typedef enum { __NV_NOSAT = 0x00, __NV_SATFINITE = 0x08 } __nv_saturation_t;
typedef enum { __NV_E4M3 = 0x00, __NV_E5M2 = 0x01 } __nv_fp8_interpretation_t;

struct __nv_fp8_e4m3 {
    __nv_fp8_storage_t __x;
};
struct __nv_fp8_e5m2 {
    __nv_fp8_storage_t __x;
};
struct __nv_fp8x2_e4m3 {
    __nv_fp8x2_storage_t __x;
};
struct __nv_fp8x2_e5m2 {
    __nv_fp8x2_storage_t __x;
};
struct __nv_fp8x4_e4m3 {
    __nv_fp8x4_storage_t __x;
};
struct __nv_fp8x4_e5m2 {
    __nv_fp8x4_storage_t __x;
};
