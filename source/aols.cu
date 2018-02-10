// Copyright (c) 2018, Coren Bialik
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "caching_context.hxx"
#include "gols.hxx"
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <moderngpu/context.hxx>
#include <moderngpu/memory.hxx>
#include <moderngpu/sort_networks.hxx>
#include <moderngpu/transform.hxx>
#include <type_traits>

namespace {

enum { WARPSIZE = 32 };

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
} // namespace

void compute_gamma(float const *A,
                   float const *T,
                   float const *r,
                   unsigned int *rows,
                   int m,
                   int n,
                   float *q,
                   float *gamma,
                   mgpu::context_t &context) {

  auto kernel = [] MGPU_DEVICE(int tid, int cta, float const *A, float const *T,
                               float const *r, unsigned int *rows, int n,
                               float *q, float *gamma) {

    __shared__ float reduced[WARPSIZE];
    if (tid < WARPSIZE)
      reduced[tid] = 0;
    __syncthreads();

    // Select your row from the dictionary
    float const *a = A + rows[cta] * n;
    float const *t = T + rows[cta] * n;

    float ar = 0;
    float at = 0;
    float tt = 0;
    for (int i = tid; i < n; i += blockDim.x) {
      ar += a[i] * r[i];
      at += a[i] * t[i];
      tt += t[i] * t[i];
    }

    ar = block_sum_t<float>(reduced).sum(ar);
    at = block_sum_t<float>(reduced).sum(at);
    tt = block_sum_t<float>(reduced).sum(tt);

    if (!tid) {
      float tmp  = ar / max(at, 1.0e-20f);
      gamma[cta] = fabs(sqrt(tt) * tmp);
      q[cta]     = tmp;
    }
  };
  int const num_ctas = m;
  mgpu::cta_launch<256>(kernel, num_ctas, context, A, T, r, rows, n, q, gamma);
}

void largest_gammas(float const *gamma,
                    int m,
                    unsigned int *largest_rows,
                    unsigned int *done,
                    mgpu::context_t &context) {
  enum { L = 1 };
  typedef mgpu::kv_array_t<float, uint, L> item_t;

  auto kernel = [] MGPU_DEVICE(int tid, int n, float const *gamma,
                               unsigned int *largest_rows, item_t *block_items,
                               unsigned int *done) {
    __shared__ item_t buffer[WARPSIZE];

    n_largest_t<L> n_largest(buffer);

    item_t item;
    mgpu::iterate<L>([&](int l) {
      int const j  = L * tid + l;
      item.vals[l] = j;
      item.keys[l] = j < n ? gamma[j] : 0;
    });

    n_largest.max(item);
    if (!threadIdx.x) {
      block_items[blockIdx.x] = item;
    }

    // Turnstile
    bool last_block = false;
    if (!threadIdx.x) {
      last_block = atomicInc(done, 0xffffffff) == gridDim.x - 1;
      if (last_block)
        *done = 0;
    }

    last_block = __syncthreads_or(last_block);
    if (not last_block)
      return;

    //
    mgpu::kv_array_t<float, uint, L * 2> together;
    mgpu::iterate<L * 2>([&](int l) {
      together.keys[l] = 0;
      together.vals[l] = 0;
    });
    for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
      item = block_items[i];
      mgpu::iterate<L>([&](int l) {
        together.keys[L + l] = item.keys[l];
        together.vals[L + l] = item.vals[l];
      });
      together =
          mgpu::odd_even_sort(together, [](float a, float b) { return a > b; });
    }
    mgpu::iterate<L>([&](int l) {
      item.keys[l] = together.keys[l];
      item.vals[l] = together.vals[l];
    });
    n_largest.max(item);
    if (!threadIdx.x) {
      mgpu::array_t<unsigned int, L> rows;
      mgpu::iterate<L>([&](int l) { rows[l] = item.vals[l]; });
      rows = mgpu::odd_even_sort(
          rows, [](unsigned int a, unsigned int b) { return a < b; });
      mgpu::iterate<L>([&](int l) { largest_rows[l] = rows[l]; });
    }
  };

  enum { NT = 128 };
  int const num_ctas = ((m + L - 1) / L + NT - 1) / NT;
  mgpu::mem_t<item_t> block_items(num_ctas, context);

  mgpu::transform<NT>(kernel, num_ctas * NT, context, m, gamma, largest_rows,
                      block_items.data(), done);
}

void update(float *T,
            float const *A,
            float const *q,
            float *r,
            unsigned int const *rows,
            unsigned int const *indices,
            unsigned int *done,
            int n,
            int m,
            mgpu::context_t &context) {
  auto kernel =
      [] MGPU_DEVICE(int tid, int cta, float *T, float const *A, float const *q,
                     float *r, unsigned int const *rows,
                     unsigned int const *indices, unsigned int *done, int n) {
        int index_max = indices[0];
        if (cta == index_max)
          return;

        __shared__ float reduced[WARPSIZE];
        if (tid < WARPSIZE)
          reduced[tid] = 0;
        __syncthreads();

        float const *tmax = T + rows[index_max] * n;
        float *t          = T + rows[cta] * n;
        float const *a    = A + rows[cta] * n;
        float qmax        = q[index_max];
        float tt          = 0;
        float ta          = 0;

        // 1. compute t[j].dot(t[j])
        // 2. compute t[j].dot(t[j])
        for (int i = tid; i < n; i += blockDim.x) {
          tt += tmax[i] * tmax[i];
          ta += tmax[i] * a[i];
        }
        tt = block_sum_t<float>(reduced).sum(tt);
        ta = block_sum_t<float>(reduced).sum(ta);

        if (!tid) {
          // 3. t[j].dot(a[j]) / t[j].dot(t[j])
          reduced[0] = ta / tt;
        }
        __syncthreads();
        float const scale = reduced[0];

        // Turnstile
        bool last_block = false;
        if (!tid) {
          last_block = atomicInc(done, 0xffffffff) == gridDim.x - 2;
          if (last_block)
            *done = 0;
        }
        last_block = __syncthreads_or(last_block);
        // 4. t[j] -= t[j] - t[jmax] * t[j].dot(a[j]) / t[j].dot(t[j])
        for (int i = tid; i < n; i += blockDim.x) {
          t[i] -= scale * tmax[i];
          if (last_block) {
            // 5. r -= t[jmax] * qmax
            r[i] = r[i] - tmax[i] * qmax;
          }
        }
      };

  int const num_ctas = m;
  mgpu::cta_launch<256>(kernel, num_ctas, context, T, A, q, r, rows, indices,
                        done, n);
}

// Assume rows is a list. updated_rows will be rows with elements at
// the L indices specified in `selection` removed and appended at the end.
// Both `rows` and `updated_rows` are arrays of length `m`.
void update_partition(unsigned int const *rows,
                      unsigned int const *selection,
                      unsigned int *updated_rows,
                      int m,
                      int L,
                      mgpu::context_t &context) {

  auto kernel = [] MGPU_DEVICE(int tid, unsigned int const *rows,
                               unsigned int const *selection,
                               unsigned int *updated_rows, int m, int L) {
    // Do a prefix sum on the fly to figure out where to gather the row
    // from. Requires indices to be sorted.
    int offset = 0;
    for (; offset < L and selection[offset] <= (tid + offset); ++offset)
      ;

    auto src_value    = rows[tid + offset];
    updated_rows[tid] = src_value;

    // First L threads ship the selected rows to the end of the array.
    if (tid < L)
      updated_rows[m - tid - 1] = rows[selection[tid]];
  };

  int const N = m - L;
  mgpu::transform(kernel, N, context, rows, selection, updated_rows, m, L);
}

std::vector<unsigned int> aols_solve(float const *dictionary,
                                     float const *signal,
                                     int m,
                                     int n,
                                     int k,
                                     double epsilon) {
  caching_context_t context;

  mgpu::mem_t<float> A(n * m, context);
  mgpu::mem_t<float> T(n * m, context);
  mgpu::mem_t<float> y(n, context);
  mgpu::mem_t<float> r(n, context);
  mgpu::mem_t<float> q(m, context);
  mgpu::mem_t<float> gamma(m, context);
  mgpu::mem_t<unsigned int> rows(m, context);
  mgpu::mem_t<unsigned int> updated_rows(m, context);
  mgpu::mem_t<unsigned int> indices(1, context);
  mgpu::mem_t<unsigned int> done(1, context);

  cudaMemset(done.data(), 0x0, done.size() * sizeof(unsigned int));

  cudaError_t error = cudaSuccess;
  error             = mgpu::htod(A.data(), dictionary, n * m);
  assert(cudaSuccess == error);
  error = mgpu::dtod(T.data(), A.data(), n * m);
  assert(cudaSuccess == error);
  error = mgpu::htod(y.data(), signal, n);
  assert(cudaSuccess == error);
  error = mgpu::dtod(r.data(), y.data(), n);
  assert(cudaSuccess == error);

  std::vector<unsigned int> I(m);
  // initial set of columns
  std::iota(I.begin(), I.end(), 0);
  error = mgpu::htod(rows.data(), I.data(), m);

  std::vector<int> S;

  int ms = m;
  for (int i = 0; i < std::min(k, std::min(n, m)); ++i, ms--) {
    // This will compute the new gammas, and z,
    compute_gamma(A.data(), T.data(), r.data(), rows.data(), ms, n, q.data(),
                  gamma.data(), context);
    largest_gammas(gamma.data(), ms, indices.data(), done.data(), context);
    // Now we have the indices of the L largest rows in indices.
    // We should remove them from the active set and add them to the S set.
    update(T.data(), A.data(), q.data(), r.data(), rows.data(), indices.data(),
           done.data(), n, ms, context);
    update_partition(rows.data(), indices.data(), updated_rows.data(), m, 1,
                     context);
    updated_rows.swap(rows);

    if (epsilon > 0) {
      auto hr     = mgpu::from_mem(r);
      double rrms = 0;
      for (auto x : hr) {
        rrms += x * x;
      }
      if (std::sqrt(rrms / hr.size()) < epsilon)
        break;
    }
  }
  // Done, now transfer the result which is the top (m - ms) values of the
  // `rows` array.
  auto result_rows = mgpu::from_mem(rows);
  result_rows.erase(result_rows.begin(), result_rows.begin() + ms);
  return result_rows;
}
