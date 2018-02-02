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
#include <moderngpu/transform.hxx>

#if defined(__CUDA_ARCH__) and __CUDA_ARCH__ < 300
#error Compute capabilties < 3.0 are not supported
#endif

namespace {

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

// Solve the linear system `A` x = `b` using QR factorization.
// Only supports the case where the system is not over-determined, which is
// okay since we're trying to do _sparse_ approximation here.
// The contents of "A" will get overwritten, the contents of "B" will be
// the solution vector "x".
// Pulled largely from
// http://docs.nvidia.com/cuda/cusolver/index.html#ormqr-example1
void sgels(float *A,
           float *B,
           int lda,
           int m,
           int n,
           int ldb,
           int nrhs,
           cublasHandle_t blas_handle,
           mgpu::context_t &context) {

  cusolverDnHandle_t solver_handle = nullptr;

  cublasStatus_t blas_status;
  auto solver_status = cusolverDnCreate(&solver_handle);
  cusolverDnSetStream(solver_handle, context.stream());

  assert(CUSOLVER_STATUS_SUCCESS == solver_status);

  int lwork = 0;
  solver_status =
      cusolverDnSgeqrf_bufferSize(solver_handle, m, n, A, lda, &lwork);
  assert(CUSOLVER_STATUS_SUCCESS == solver_status);

  mgpu::mem_t<float> work(lwork, context);
  mgpu::mem_t<int> device_info(1, context);
  mgpu::mem_t<float> tau(m, context);

  assert(m == lda);
  // QR
  solver_status = cusolverDnSgeqrf(solver_handle, m, n, A, lda, tau.data(),
                                   work.data(), lwork, device_info.data());
  assert(CUSOLVER_STATUS_SUCCESS == solver_status);
  assert(0 == mgpu::from_mem(device_info).front());

  // compute Q^T*B
  solver_status = cusolverDnSormqr(solver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
                                   m, nrhs, n, A, lda, tau.data(), B, ldb,
                                   work.data(), lwork, device_info.data());
  assert(0 == mgpu::from_mem(device_info).front());

  // compute x = R \ Q^T*B
  float const one = 1;
  blas_status     = cublasStrsm(
      blas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
      CUBLAS_DIAG_NON_UNIT, n, nrhs, &one, A, lda, B, ldb);
  assert(CUBLAS_STATUS_SUCCESS == blas_status);

  context.synchronize();
  solver_status = cusolverDnDestroy(solver_handle);
  assert(CUSOLVER_STATUS_SUCCESS == solver_status);
}

// Find the (value, index) of the `n` largest entries in `data`
// FIXME instead of passing `m` here, can't we resize `data` to size `m`
// outside?
template <typename T>
std::vector<std::pair<T, int>>
n_largest(std::vector<T> const &data, int n, int m) {
  std::vector<std::pair<T, int>> kv(m);
  for (size_t i = 0; i < m; ++i) {
    kv[i] = std::make_pair(data[i], i);
  }
  std::partial_sort(kv.begin(), kv.begin() + n, kv.end(),
                    [](std::pair<T, int> a, std::pair<T, int> b) {
                      return a.first >= b.first;
                    });
  kv.erase(kv.begin() + n, kv.end());
  return kv;
}

// Compute `da = D.dot(A[l])/norm(D.dot(A[l]))`, where `D` is a `n` x `n`
// matrix, and `A` is our dictionary.
void compute_DA(float const *D,
                float const *A,
                int n,
                int l,
                float *da,
                unsigned int *done,
                mgpu::context_t &context) {

  auto kernel = [] MGPU_DEVICE(int tid, int cta, float const *D, float const *A,
                               int n, int l, float *da, unsigned int *done) {

    __shared__ float reduced[WARPSIZE];
    if (tid < WARPSIZE)
      reduced[tid] = 0;
    __syncthreads();

    float const *d = D + cta * n;
    float const *a = A + l * n;
    float dot      = 0;

    for (int i = tid; i < n; i += blockDim.x) {
      dot += a[i] * d[i];
    }

    dot = block_sum_t<float>(reduced).sum(dot);
    if (!tid)
      da[cta] = dot;

    __syncthreads();

    bool last_block = false;
    if (!tid) {
      last_block = atomicInc(done, 0xffffffff) == gridDim.x - 1;
      if (last_block)
        *done = 0; // reset the turnstile
    }

    last_block = __syncthreads_or(last_block);

    if (not last_block)
      return;

    dot   = 0;
    for (int i = tid; i < n; i += blockDim.x) {
      dot += da[i] * da[i];
    }
    dot = block_sum_t<float>(reduced).sum(dot);
    if (!tid)
      reduced[0] = dot;
    __syncthreads();

    dot = rsqrt(reduced[0]);

    // Normalize
    for (int i = tid; i < n; i += blockDim.x) {
      da[i] *= dot;
    }
  };

  int const num_ctas = n;
  mgpu::cta_launch<256>(kernel, num_ctas, context, D, A, n, l, da, done);
}

// The `n` x `n` matrix `D` is updated in-place as:
// `D -= outer(d, d)`.
void update_D(float *D, float const *d, int n, mgpu::context_t &context) {
  auto kernel = [] MGPU_DEVICE(int tid, int cta, float *D, float const *d,
                               int n) {
    float *Drow = D + cta * n;
    float dval  = d[cta];
    for (int i = tid; i < n; i += blockDim.x) {
      Drow[i] -= d[i] * dval;
    }
  };

  int const num_ctas = n;
  mgpu::cta_launch<256>(kernel, num_ctas, context, D, d, n);
}

// This computes the correlation of the projections with the signal we're
// trying to reconstruct, viz.
// `gamma[j] = |dot(signal, Pa[j])|` for all rows `j` in `Pa`, and returns
// the indices of the `L` rows with the largest `gamma[j]`.
std::vector<unsigned int> compute_largest_projection(float const *Pa,
                                                     int m,
                                                     int n,
                                                     float const *signal,
                                                     mgpu::mem_t<float> &gamma,
                                                     int L,
                                                     mgpu::context_t &context) {

  auto kernel = [] MGPU_DEVICE(int tid, int cta, float const *Pa, int n,
                               float const *signal, float *gamma, int L) {
    __shared__ float reduced[WARPSIZE];
    if (tid < WARPSIZE)
      reduced[tid] = 0;
    __syncthreads();

    float const *pa = Pa + cta * n;
    float dot = 0;
    float norm = 0;
    for (int i = tid; i < n; i += blockDim.x) {
      float2 value = make_float2(signal[i], pa[i]);
      dot += value.x * value.y;
      norm += value.y * value.y;
    }

    dot = block_sum_t<float>(reduced).sum(dot);
    __syncthreads();
    norm = block_sum_t<float>(reduced).sum(norm);

    if (!tid) {
      gamma[cta] = fabs(dot * rsqrt(norm));
    }
  };

  int const num_ctas = m;
  mgpu::cta_launch<256>(kernel, num_ctas, context, Pa, n, signal, gamma.data(),
                        L);

  auto hgamma         = mgpu::from_mem(gamma);
  auto largest_gammas = n_largest(hgamma, L, m);
  std::vector<unsigned int> largest_indices(largest_gammas.size());
  std::transform(largest_gammas.begin(), largest_gammas.end(),
                 largest_indices.begin(),
                 [](std::pair<float, int> item) { return item.second; });
  return largest_indices;
}

// Initialize `P` as identity(`n`)
void initialize_identity(float *P, int n, mgpu::context_t &context) {
  auto kernel = [] MGPU_DEVICE(int tid, int cta, float *P, int n) {
    float *prow = P + cta * n;
    for (int i = tid; i < n; i += blockDim.x) {
      prow[i] = (i == cta);
    }
  };
  int const num_ctas = n;
  mgpu::cta_launch<256>(kernel, num_ctas, context, P, n);
}

// Construct `A_subset = [A[c, :] for c in columns]`,
// where `A` is a dictionary with atoms of size `n`, and `columns` is a list
// of `num_columns` indices. `A_subset` be able to hold `num_columns` x `n`
// entries.
void gather_dictionary(float const *A,
                       float *A_subset,
                       int const *columns,
                       int n,
                       int num_columns,
                       mgpu::context_t &context) {
  auto kernel = [] MGPU_DEVICE(int tid, int cta, float const *A,
                               float *A_subset, int const *columns, int n,
                               int num_columns) {
    float *a_subset       = A_subset + cta * n;
    float const *a_source = A + columns[cta] * n;
    for (int i = tid; i < n; i += blockDim.x) {
      a_subset[i] = a_source[i];
    }
  };
  int const num_ctas = num_columns;
  mgpu::cta_launch<256>(kernel, num_ctas, context, A, A_subset, columns, n,
                        num_columns);
}

// Create a new dictionary `newA` which consists of all the rows in `A`
// except the `L` rows specified in `indices`. The original dictionary
// has `m` rows, and `n` columns.
// The list of rows `indices` must be sorted in ascending order, otherwise
// this will give the wrong result.
void compact_dictionary(float const *A,
                        float *newA,
                        unsigned int const *indices,
                        int m,
                        int n,
                        int L,
                        mgpu::context_t &context) {
  auto kernel = [] MGPU_DEVICE(int tid, int cta, float const *A, float *newA,
                               unsigned int const *indices, int m, int n,
                               int L) {
    // Do a prefix sum on the fly to figure out where to gather the row
    // from. Requires indices to be sorted.
    int offset = 0;
    for (; offset < L and indices[offset] <= (cta + offset); ++offset)
      ;

    float const *arow = A + (cta + offset) * n;
    float *arow_new   = newA + cta * n;

    for (int i = tid; i < n; i += blockDim.x) {
      arow_new[i] = arow[i];
    }
  };

  int const num_ctas = m - L;
  mgpu::cta_launch<256>(kernel, num_ctas, context, A, newA, indices, m, n, L);
}
} // namespace

// See interface description in header file.
std::tuple<std::vector<int>, std::vector<float>>
gols_solve(float const *dictionary,
           float const *signal,
           int n,
           int m,
           int sparsity,
           int L,
           bool solve_lstsq) {
  caching_context_t context;

  cublasStatus_t status;
  cublasHandle_t handle;
  // cublasSetStream(handle, context.stream());
  status = cublasCreate(&handle);
  assert(CUBLAS_STATUS_SUCCESS == status);

  mgpu::mem_t<float> P(n * n, context);
  mgpu::mem_t<float> Pa(n * m, context);
  mgpu::mem_t<float> gamma(m, context);
  mgpu::mem_t<float> d(n, context);
  mgpu::mem_t<float> A(n * m, context);
  mgpu::mem_t<float> temp(n * m, context);
  mgpu::mem_t<float> y(n, context);
  mgpu::mem_t<unsigned int> done(1, context);
  mgpu::mem_t<unsigned int> device_largest_indices(L, context);
  cudaMemset(done.data(), 0x0, sizeof(unsigned int));

  initialize_identity(P.data(), n, context);
  cudaError_t error = cudaSuccess;
  error             = mgpu::htod(A.data(), dictionary, n * m);
  assert(cudaSuccess == error);
  error = mgpu::htod(y.data(), signal, n);
  assert(cudaSuccess == error);

  std::vector<int> I(m);
  // initial set of columns
  std::iota(I.begin(), I.end(), 0);
  std::vector<int> S;

  for (int i = 0, ms = m; i < std::min(sparsity, std::min(n, m) / L);
       ++i, ms -= L) {
    // Compute [P.dot(A[:, j]) for j in I]
    float alpha = 1, beta = 0;
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, ms, n, &alpha,
                         P.data(), n, A.data(), n, &beta, Pa.data(), n);

    assert(CUBLAS_STATUS_SUCCESS == status);

    // FIXME why would you call this compute_largest_projection if it returns
    // the largest
    // indices.
    auto L_largest_indices = compute_largest_projection(
        Pa.data(), ms, n, y.data(), gamma, L, context);

    // Next, add these L largest colums (given in local indexing) to the
    // sparse set S and remove them from I.
    std::sort(L_largest_indices.begin(), L_largest_indices.end(),
              [](int a, int b) { return a > b; });

    for (auto l : L_largest_indices) {
      S.push_back(I[l]);
      assert(l < I.size());
      I.erase(I.begin() + l);
    }

    // We could do this in one pass, but let's do it in L passes right now
    for (auto l : L_largest_indices) {
      compute_DA(P.data(), A.data(), n, l, d.data(), done.data(), context);
      update_D(P.data(), d.data(), n, context);
    }

    std::sort(L_largest_indices.begin(), L_largest_indices.end());
    error = mgpu::htod(device_largest_indices.data(), L_largest_indices);
    assert(cudaSuccess == error);
    compact_dictionary(A.data(), temp.data(), device_largest_indices.data(), ms,
                       n, L, context);
    A.swap(temp);
  }

  // Solve the resulting A_sparse x_sparse = y least-squares problem.
  // Note: this overwrites y.
  std::vector<float> sparse_solution;
  if (solve_lstsq) {
    int const sparse_m = std::min(sparsity, std::min(n, m) / L) * L;
    error              = mgpu::htod(temp.data(), dictionary, n * m);
    assert(cudaSuccess == error);
    auto device_s = mgpu::to_mem(S, context);
    gather_dictionary(temp.data(), A.data(), device_s.data(), n,
                      device_s.size(), context);
    sgels(A.data(), y.data(), n, n, sparse_m, n, 1, handle, context);
    sparse_solution = mgpu::from_mem(y);
    sparse_solution.resize(sparse_m);
  }

  context.synchronize();
  status = cublasDestroy(handle);
  assert(CUBLAS_STATUS_SUCCESS == status);

  return std::make_tuple(S, sparse_solution);
}

void gols_solve(float const *dictionary,
                float const *signal,
                int n,
                int m,
                int sparsity,
                int L,
                int *best_rows,
                float *x,
                bool solve_lstsq) {

  auto result = gols_solve(dictionary, signal, n, m, sparsity, L, solve_lstsq);
  std::copy(std::get<0>(result).begin(), std::get<0>(result).end(), best_rows);
  assert(not solve_lstsq or x != nullptr);
  if (solve_lstsq)
    std::copy(std::get<1>(result).begin(), std::get<1>(result).end(), x);
}
