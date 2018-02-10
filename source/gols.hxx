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

// -----------------------------------------------------------------------------
// Generalized Orthogonal Least-Squares (GOLS)
// -----------------------------------------------------------------------------
//
// This is a CUDA implementation of GOLS as described in
//
// A. Hashemi and H. Vikalo,
//   "Sparse linear regression via generalized orthogonal least-squares,"
//   2016 IEEE Global Conference on Signal and Information Processing,
//   Washington, DC, 2016, pp. 1305-1309.
//
// https://arxiv.org/pdf/1602.06916.pdf

#ifdef __cplusplus
#include <tuple>
#include <vector>
// `gols_solve` finds an approximate solution to the folling problem:
// Minimize[norm(signal - A.dot(x)), x], with len(nonzero(x)) <= sparsity.
// It does this using the generalized orthogonal least-squares approach
// cited above.
// The `m` x `n` matrix `dictionary` corresponds to `transpose(A)`,
// and represents the set of `m` "atoms" among which we want to find the
// `k` vectors that best represent the signal.
// The `signal` is a length `n` vector which corresponds to `y` in the
// above problem formulation.  `L` is GOLS' multiplicity parameter (i.e.,
// every step of the procedure we select `L` "best" vectors).
// The function `gols_solve` will return a vector of `sparsity * L` row
// indices into `dictionary` that correspond to the sparse approximation.
// If `solve_lstsq` is true, `gols_solve` will also return the least-squares
// `x_sparse` of `A[:,subset].dot(x_sparse) = y`. If `solve_lstsq` is false
// the vector will be emtpy.
std::tuple<std::vector<int>, std::vector<float>>
gols_solve(float const *dictionary,
           float const *signal,
           int n,
           int m,
           int sparsity,
           int L,
           bool solve_lstsq = false);

std::vector<unsigned int> aols_solve(float const *dictionary,
                                     float const *signal,
                                     int n,
                                     int m,
                                     int k,
                                     double epsilon);
extern "C" {
#endif
#include <stdbool.h>
// For my special friends: same interface as above except:
// `best_rows` must be allocated to have a length of at least `sparsity * L`,
// If `solve_lstsq` is true, `x` must be allocated to have a length of at
// least `sparsity * L`.
void gols_solve(float const *dictionary,
                float const *signal,
                int n,
                int m,
                int sparsity,
                int L,
                int *best_rows,
                float *x,
                bool solve_lstsq);

void aols_solve(float const *dictionary,
                float const *signal,
                int n,
                int m,
                int k,
                double epsilon,
                unsigned int *best_rows);
#ifdef __cplusplus
}
#endif