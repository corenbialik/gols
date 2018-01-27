#include "../source/gols.hxx"
#include <algorithm>
#include <random>
#include <iostream>

int main(int argc, char const *argv[]) {
  int const n = 640;
  int const m = 1280;

  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.0, 1 / std::sqrt(n));

  std::vector<float> A(n * m);
  std::generate(A.begin(), A.end(), [&]() { return distribution(generator); });
  std::vector<float> x(m);

  std::vector<int> indices(m);
  std::iota(indices.begin(), indices.end(), 0);
  std::random_shuffle(indices.begin(), indices.end());

  std::vector<float> signal(n);

  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      signal[j] += A[j * m + i] * x[i];
    }
  }
  int const k = 6;
  int const L = 2;
  auto result = gols_solve(A.data(), signal.data(), n, m, k, L);

  return 0;
}