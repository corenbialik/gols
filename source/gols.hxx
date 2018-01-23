#include <vector>
#include <tuple>
std::tuple<std::vector<int>, std::vector<float>>
gols_solve(float const *dictionary,
           float const *signal,
           int n,
           int m,
           int sparsity,
           int L,
           bool solve_lstsq = false);
