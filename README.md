# Generalized Orthogonal Least Squares

[Coren Bialik](mailto:coren.bialik@protonmail.com), 2018.

This is a GPU implementation of [Generalized Orthogonal Least-Squares]( https://arxiv.org/abs/1602.06916) (GOLS) following Hashemi and Vikalo. It solves the following problem:
$$
\mathop{\mathrm{min}} \lVert A_{ij} x_j - b_i\rVert_2, \:\text{subject to} \:\lVert x\rVert_0 \leq k
$$

Thus it can be used for a set of sparse approximation problems, like supervised feature selection, or [classification](https://arxiv.org/abs/1607.04942), or just generally as an **EDC for your sparse approximation tasks**.

The implementation is based on CUDA/[cuBLAS](https://developer.nvidia.com/cublas), uses [ModernGPU](https://github.com/moderngpu/moderngpu) for buffer allocation and copies,  the caching allocator from [CUB](https://github.com/NVlabs/cub) to create a memory pool.

**Note**: some of the many things that keep this short of a production-level implementation

- error handling and reporting would have to be improved
- `cta` launches are not optimal for all problem sizes -- especially small problems (like I could run it with "python" small)  could benefit from better parallelization

## Building

Basic:
```
git clone --recursive URL
mkdir build
cd build
cmake ..
make
```

Or simply:
```
python setup.py install
```

## Usage

There's a python interface based on [pybind11](https://github.com/pybind/pybind11) (btw: <3), a C, and a C++ interface (`gols.hxx`).

```python
import gols
help(gols.solve)
```

See the jupyter notebook in `./play` for example usage.
