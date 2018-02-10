#include <moderngpu/sort_networks.hxx>
#pragma once

enum { WARPSIZE = 32 };

// Simple block reduction.
// Needs to be initialized with a __shared__ buffer of size no less than
// ceil(blockDim.x / WARPSIZE)
template <typename T> struct block_sum_t {
  T *reduced;
  __device__ block_sum_t(T *reduced) : reduced(reduced) {}

  __device__ T sum(T item) {
    int const warp_id = threadIdx.x >> 5;
    int const lane    = threadIdx.x & 0x1f;
    for (int o = WARPSIZE / 2; o > 0; o >>= 1) {
      item += __shfl_xor_sync(0xffffffff, item, o, WARPSIZE);
    }
    if (!lane) {
      reduced[warp_id] = item;
    }
    __syncthreads();
    if (!warp_id) {
      item = reduced[lane];
      for (int o = WARPSIZE / 2; o > 0; o >>= 1) {
        item += __shfl_xor_sync(0xffffffff, item, o, WARPSIZE);
      }
    }
    return item;
  }
};

template <int L> struct n_largest_t {
  mgpu::kv_array_t<float, uint, L> *reduced;

  __device__ __forceinline__
  n_largest_t(mgpu::kv_array_t<float, uint, L> *buffer)
      : reduced(buffer) {}

  __device__ __forceinline__ void
  warp_max(mgpu::kv_array_t<float, uint, L> &item) {
    mgpu::kv_array_t<float, uint, 2 * L> together;
    mgpu::iterate<L>([&](int l) {
      together.keys[l] = item.keys[l];
      together.vals[l] = item.vals[l];
    });
    for (int o = WARPSIZE / 2; o > 0; o >>= 1) {
      mgpu::iterate<L>([&](int l) {
        together.keys[L + l] =
            __shfl_xor_sync(0xffffffff, together.keys[l], o, WARPSIZE);
        together.vals[L + l] =
            __shfl_xor_sync(0xffffffff, together.vals[l], o, WARPSIZE);
      });
      together =
          mgpu::odd_even_sort(together, [](float a, float b) { return a > b; });
    }

    mgpu::iterate<L>([&](int l) {
      item.keys[l] = together.keys[l];
      item.vals[l] = together.vals[l];
    });
  }

  // The first warp will have the L-largest items stored in item.
  __device__ __forceinline__ void max(mgpu::kv_array_t<float, uint, L> &item) {
    int const warp_id = threadIdx.x >> 5;
    int const lane    = threadIdx.x & 0x1f;
    warp_max(item);
    if (!lane) {
      reduced[warp_id] = item;
    }
    __syncthreads();
    if (!warp_id) {
      item = reduced[lane];
      warp_max(item);
    }
  }
};