# Generalized Orthogonal Least Squares

This is an implementation of [Generalized Orthogonal Least-Squares][gols] (GOLS) following Hashemi and Vikalo 

https://arxiv.org/pdf/1602.06916.pdf

Solve the following problem:

$$
\mathop{\mathrm{min}} \lVert A_{ij} x_j - b_i\rVert_2, \:\text{subject to} \:\lVert x\rVert_0 \leq k
$$


```Python
def gols_solve(dictionary, y, sparsity, L):
    '''
    Generalized Orthogonal Least-Squares\n",
    "https://arxiv.org/pdf/1602.06916.pdf

    Solve
    >>> min_{||x||_0} |dictionary.dot(x) - y|

    Args:
        dictionary (np.ndarray): n x m matrix, where m is the number
            of "atoms" in the dictionary (i.e. overcomplete "basis vectors"),
            and n is the length of the signal to reconstruct.
        y (np.ndarray): Signal to reconstruct
        sparsity (int): Degree of sparsity, i.e. we'll try to reconstruct
            the signal using 'sparsity' x 'L' atoms.
        L (int): Number of atoms to consider at a time.

    Returns:
        tuple: S, xs: 'S' is the set of atoms that we picked, and
        'xs' is the coefficient associated with each atom, i.e. the
        sparse reconstruction of the signal is
        >>> recon = numpy.array([dictionary[:,j] for j in S]).T.dot(xs)
    '''
    import bisect
    import numpy
    '''
    dictionary is a n x m matrix
    looking for y = A * x with x super-sparse
    '''
    A = dictionary
    k = sparsity
    n, m = A.shape
    Port = numpy.eye(n)
    S = set()
    I = range(m)
    for i in range(0, min(k, m / L)):
        Porta = numpy.array([Port.dot(A[:, j]) for j in I])
        Porta = numpy.divide(Porta, numpy.linalg.norm(Porta, axis=1)[:,None])
        gamma = numpy.array([abs(y.dot(porta)) for porta in Porta])
        L_largest = gamma.argsort()[-L:]
        largest_indices = [I[j] for j in L_largest]
        [S.add(I[j]) for j in L_largest]
        [I.pop(bisect.bisect_left(I, j)) for j in largest_indices]
        D = Port
        for j in largest_indices:
            Da = D.dot(A[:, j])
            d = Da / numpy.linalg.norm(Da)
            D = D - numpy.outer(d, d)
        Port = D
    S = list(S)
    As = numpy.array([A[:, j] for j in S]).T
    xs, _, _, _ = numpy.linalg.lstsq(As, y)
    return S, xs
```

[gols]: https://arxiv.org/pdf/1602.06916.pdf Sparse linear regression via generalized orthogonal least-squares