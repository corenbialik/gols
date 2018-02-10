#include "gols.hxx"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

std::tuple<std::vector<int>, std::vector<float>> _gols_solve(
    py::array_t<float, py::array::c_style | py::array::forcecast> dictionary,
    py::array_t<float, py::array::c_style | py::array::forcecast> signal,
    int sparsity,
    int L,
    bool solve_lstsq) {
  int const m = dictionary.shape(0);
  int const n = dictionary.shape(1);

  return gols_solve(dictionary.data(), signal.data(), n, m, sparsity, L,
                    solve_lstsq);
}

std::vector<unsigned int> _aols_solve(
    py::array_t<float, py::array::c_style | py::array::forcecast> dictionary,
    py::array_t<float, py::array::c_style | py::array::forcecast> signal,
    int k,
    double epslion) {
  int const m = dictionary.shape(0);
  int const n = dictionary.shape(1);

  return aols_solve(dictionary.data(), signal.data(), n, m, k, epslion);
}
}

PYBIND11_MODULE(_gols, m) {
  m.doc() = "Generalized orthogonal least-squares.";
  m.def("gols_solve", &_gols_solve);
  m.def("aols_solve", &_aols_solve);
}