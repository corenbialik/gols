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

  __device__ n_largest_t(mgpu::kv_array_t<float, uint, L> *buffer)
      : reduced(buffer) {}

  __device__ void warp_max(mgpu::kv_array_t<float, uint, L> &item) {
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
  __device__ void max(mgpu::kv_array_t<float, uint, L> &item) {
    int const warp_id = threadIdx.x >> 5;
    int const lane = threadIdx.x & 0x1f;
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
    int const lane = threadIdx.x & 0x1f;
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

void abracadabra(float const *U, float const *A, float const *r,
                 unsigned int *rows, int m, int n, int k, float *Q, float *z,
                 float *gamma, mgpu::context_t &context) {

  auto kernel = [] MGPU_DEVICE(int tid, int cta, float const *U, float const *A,
                               float const *r, unsigned int *rows, int n, int k,
                               float *Q, float *z, float *gamma) {

    __shared__ float reduced[WARPSIZE];
    if (tid < WARPSIZE)
      reduced[tid] = 0;
    __syncthreads();

    // Select your row from the dictionary
    float const *a = A + rows[cta] * n;
    float *q = Q + cta * n;
    for (int i = tid; i < n; i += blockDim.x) {
      q[i] = 0;
    }
    for (int l = 0; l < k; ++l) {
      float const *u = U + l * n;
      float u2 = 0;
      float ua = 0;
      for (int i = tid; i < n; i += blockDim.x) {
        u2 += u[i] * u[i];
        ua += u[i] * a[i];
      }

      u2 = block_sum_t<float>(reduced).sum(u2);
      ua = block_sum_t<float>(reduced).sum(ua);

      if (!tid) {
        reduced[0] = ua / max(u2, 1.0e-20f);
      }
      __syncthreads();
      float scale = reduced[0];

      for (int i = tid; i < n; i += blockDim.x) {
        q[i] += scale * u[i];
      }
    }

    // Now we need to compute t and its dot product with a
    float ta = 0;
    float ra = 0;
    for (int i = tid; i < n; i += blockDim.x) {
      float ai = a[i];
      float t = ai - q[i];
      ra += r[i] * ai;
      ta += ai * t;
      q[i] = t;
    }
    // Got to make sure everybody is done reading "scale"
    __syncthreads();
    ta = block_sum_t<float>(reduced).sum(ta);
    ra = block_sum_t<float>(reduced).sum(ra);
    if (!tid) {
      reduced[0] = ta;
      reduced[1] = ra;
      z[cta] = ra;
    }
    __syncthreads();

    float tainv = 1 / reduced[0];
    float zj = reduced[1];
    float q2 = 0;
    for (int i = tid; i < n; i += blockDim.x) {
      float qi = q[i] * tainv;
      q2 += qi * qi;
      q[i] = qi; // store the new q
    }
    q2 *= zj * zj;
    __syncthreads(); // protect the `reduced` that's used for `tainv`, and `zj`
    q2 = block_sum_t<float>(reduced).sum(q2);
    if (!tid) {
      gamma[cta] = q2;
    }

    // Here we could do a turnstile and compute the L-largest gamma`s
    // on-the-fly, but that would be quite a bit of work if the dictionary
    // has thousands of atoms.
  };
  int const num_ctas = m;
  mgpu::cta_launch<256>(kernel, num_ctas, context, U, A, r, rows, n, k, Q, z,
                        gamma);
}

template <int L>
void largest_gammas(float const *gamma, int m, unsigned int *largest_rows,
                    unsigned int *done, mgpu::context_t &context) {
  typedef mgpu::kv_array_t<float, uint, L> item_t;

  auto kernel = [] MGPU_DEVICE(int tid, int n, float const *gamma,
                               unsigned int *largest_rows, item_t *block_items,
                               unsigned int *done) {
    __shared__ item_t buffer[WARPSIZE];

    n_largest_t<L> n_largest(buffer);

    item_t item;
    mgpu::iterate<L>([&](int l) {
      int const j = L * tid + l;
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

void update_ur(float const *Q, unsigned int const *indices, float const *z,
               int offset, int n, int L, float *U, float *r,
               mgpu::context_t &context) {
  auto kernel = [] MGPU_DEVICE(int tid, float const *Q, float const *z,
                               unsigned int const *indices, int offset, int n,
                               int L, float *U, float *r) {
    float ri = r[tid];
    for (int l = 0; l < L; ++l) {
      int const j = indices[l]; // inner indexing OKAY FIXME
      float const qi = Q[j * n + tid];
      ri -= z[j] * qi;
      U[(offset + l) * n + tid] = qi;
    }
    r[tid] = ri;
  };

  mgpu::transform(kernel, n, context, Q, z, indices, offset, n, L, U, r);
}

// Assume rows is a list. updated_rows will be rows with elements at
// the L indices specified in `selection` removed and appended at the end.
// Both `rows` and `updated_rows` are arrays of length `m`.
void udpate_partition(unsigned int const *rows, unsigned int const *selection,
                      unsigned int *updated_rows, int m, int L,
                      mgpu::context_t &context) {

  auto kernel = [] MGPU_DEVICE(int tid, unsigned int const *rows,
                               unsigned int const *selection,
                               unsigned int *updated_rows, int m, int L) {
    // Do a prefix sum on the fly to figure out where to gather the row
    // from. Requires indices to be sorted.
    int offset = 0;
    for (; offset < L and selection[offset] <= (tid + offset); ++offset)
      ;

    auto src_value = rows[tid + offset];
    updated_rows[tid] = src_value;

    // First L threads ship the selected rows to the end of the array.
    if (tid < L)
      updated_rows[m - tid - 1] = rows[selection[tid]];
  };

  int const N = m - L;
  mgpu::transform(kernel, N, context, rows, selection, updated_rows, m, L);
}

std::vector<unsigned int> aols(float const *dictionary, float const *signal,
                               int m, int n, int k, int L, float epsilon) {

  caching_context_t context;
  cublasStatus_t status;
  cublasHandle_t handle;
  // cublasSetStream(handle, context.stream());
  status = cublasCreate(&handle);
  assert(CUBLAS_STATUS_SUCCESS == status);

  // Q can maximally take size k * L x n
  mgpu::mem_t<float> U((k * L) * n, context);
  mgpu::mem_t<float> Q(m * n, context);
  mgpu::mem_t<float> A(n * m, context);
  mgpu::mem_t<float> y(n, context);
  mgpu::mem_t<float> r(n, context);
  mgpu::mem_t<float> z(m, context);
  mgpu::mem_t<float> gamma(m, context);
  mgpu::mem_t<unsigned int> rows(m, context);
  mgpu::mem_t<unsigned int> updated_rows(m, context);
  mgpu::mem_t<unsigned int> largest_rows(L, context);
  mgpu::mem_t<unsigned int> done(1, context);

  cudaMemset(done.data(), 0x0, done.size() * sizeof(unsigned int));

  cudaError_t error = cudaSuccess;
  error = mgpu::htod(A.data(), dictionary, n * m);
  assert(cudaSuccess == error);
  error = mgpu::htod(y.data(), signal, n);
  assert(cudaSuccess == error);

  // initialize r <- y
  error = mgpu::dtod(r.data(), y.data(), n);

  std::vector<unsigned int> I(m);
  // initial set of columns
  std::iota(I.begin(), I.end(), 0);
  error = mgpu::htod(rows.data(), I.data(), m);

  std::vector<int> S;

  int ms = m;
  for (int i = 0; i < std::min(k, std::min(n, m) / L); ++i, ms -= L) {
    // This will compute the new gammas, and z,
    abracadabra(U.data(), A.data(), r.data(), rows.data(), ms, n, L * i,
                Q.data(), z.data(), gamma.data(), context);

    switch (L) {
    case 1:
      largest_gammas<1>(gamma.data(), ms, largest_rows.data(), done.data(),
                        context);
      break;
    case 2:
      largest_gammas<2>(gamma.data(), ms, largest_rows.data(), done.data(),
                        context);
      break;
    case 3:
      largest_gammas<3>(gamma.data(), ms, largest_rows.data(), done.data(),
                        context);
      break;
    case 6:
      largest_gammas<6>(gamma.data(), ms, largest_rows.data(), done.data(),
                        context);
      break;
    default:
      throw std::runtime_error("Only L=1,2,3,6 are supported.");
    }
#if 1
    auto hrows = mgpu::from_mem(rows);
    auto now = mgpu::from_mem(largest_rows);
    auto hgamma = mgpu::from_mem(gamma);
    printf("gmamma: ");
    for (auto v : hgamma)
      printf("%f ", std::sqrt(v));
    printf("\n");
    printf("indices ");
    for (auto v : now)
      printf("%i ", hrows[v]);
    printf("\n");
#endif
    // Now we have the indices of the L largest rows in largest_rows.
    // We should remove them from the active set and add them to the S set.
    udpate_partition(rows.data(), largest_rows.data(), updated_rows.data(), m,
                     L, context);
    updated_rows.swap(rows);
#if 0
    auto new_rows = mgpu::from_mem(rows);
    printf("new rows: ");
    for (auto v : new_rows)
      printf("%i ", v);
    printf("\n");
#endif
    // Now update r and u
    update_ur(Q.data(), largest_rows.data(), z.data(), i * L, n, L, U.data(),
              r.data(), context);
    // TODO: add epsilon-based early exit
  }
  // Done, now transfer the result which is the top (m - ms) values of the
  // `rows` array.
  auto result_rows = mgpu::from_mem(rows);
  result_rows.erase(result_rows.begin(), result_rows.begin() + ms);
  return result_rows;
}
