# Generalized Orthogonal Least Squares

[Coren Bialik](mailto:coren.bialik@protonmail.com), 2018.

This is a GPU implementation of [Generalized Orthogonal Least-Squares][gols] (GOLS) following Hashemi and Vikalo. It solves the following problem:
$$
\mathop{\mathrm{min}} \lVert A_{ij} x_j - b_i\rVert_2, \:\text{subject to} \:\lVert x\rVert_0 \leq k
$$

Thus it can be used for a set of sparse approximation problems, like supervised feature selection, or [classification][clss], or just generally as an **EDC for your sparse approximation tasks**.

The implementation is based on CUDA/[cuBLAS][blas], uses `moderngpu` for buffer allocation and copies,  the caching allocator from `cub` to create a memory pool. 

**Note**: some of the many things that keep this short of a production-level implementation

- error handling and reporting would have to be improved
- `cta` launches are not optimal for all problem sizes -- especially small problems (like I could run it with "python" small)  could benefit from better parallelization
- `setup.py` maybe?

## Building

```
git clone --recursive URL
mkdir build
cd build
cmake ..
make
```

## Usage

There's a python interface based on [pybind11][pybind11] (btw: <3), a C, and a C++ interface (`gols.hxx`). 

```python
import sys
sys.path.insert('build')
import _gols as gols
```

See the jupyter notebook in `./play` for example usage.

## References

[gols]: https://arxiv.org/abs/1602.06916 Sparse Linear Regression via Generalized Orthogonal Least-Squares
[clss]: https://arxiv.org/abs/1607.04942 Sparse Representation-Based Classification: Orthogonal Least Squares or Orthogonal Matching Pursuit?
[blas]: https://developer.nvidia.com/cublas
[mgpu]: https://github.com/moderngpu/moderngpu
[cub]: https://github.com/NVlabs/cub
[pybind]: https://github.com/pybind/pybind11

