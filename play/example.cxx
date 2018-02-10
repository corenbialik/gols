#include "../source/gols.hxx"
#include <algorithm>
#include <iostream>
#include <random>
#include <chrono>

int main(int argc, char const *argv[]) {
  int const n = 1280;
  int const m = 12800;

  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.0, 1 / std::sqrt(n));

  std::vector<float> A(n * m);
  std::generate(A.begin(), A.end(), [&]() { return distribution(generator); });
  std::vector<float> x(m);

  std::vector<int> indices(m);
  std::iota(indices.begin(), indices.end(), 0);
  std::random_shuffle(indices.begin(), indices.end());

  for (size_t i = 0; i < m/10; ++i){
    x[i] = distribution(generator);
  }

  std::vector<float> signal(n);

  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      signal[j] += A[j * m + i] * x[i];
    }
  }

  int const k  = 6;
  int const L  = 2;

using namespace std::chrono;

  high_resolution_clock::time_point t1 = high_resolution_clock::now();



  for (int i = 0; i < 10; ++i)
  auto gresult = gols_solve(A.data(), signal.data(), n, m, k, L);

  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  for (int i = 0; i < 10; ++i)
  auto aresult = aols_solve(A.data(), signal.data(), n, m, k  , -1);
  high_resolution_clock::time_point t3 = high_resolution_clock::now();

  duration<double> time_span1 = duration_cast<duration<double>>(t2 - t1);
  duration<double> time_span2 = duration_cast<duration<double>>(t3 - t2);

  printf("GOLS(%i) %g ms, AOLS %g ms\n",
    L, 1000 * time_span1.count(), 1000 * time_span2.count());
  return 0;
}