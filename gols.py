import _gols


def solve(A, y, k, L=1, solve_least_squares=False):
    """Generalized Orthogonal Least-Squares,
    https://arxiv.org/pdf/1602.06916.pdf

    Find a set of k * L rows that approximately solve

        min_{x} || A.T.dot(x) - y ||_2, s.t. len(nonzero(x)) <= k * L

    Parameters
    ----------
    A : (M, N) ndarray
        Array representing a "dictionary" of M "atoms" of length N.
    y : (N) ndarray
        Vector representing the "signal" we want to sparsely reconstruct
    k : int
        Degree of sparsity, i.e., we'll try to reconstruct the signal
        using k * L atoms.
    L : int, optional
        Number of atoms to consdier at a time, default 1
    solve_least_squares : bool, optional
        Whether we should solve the sparse least squares problem
            A_sparse.T.dot(x_sparse) = y
        on the GPU and return x_sparse, default False

    Returns
    -------
    list of int
        Set of k * L row indices that best approximate y, viz.
            A_sparse.T = np.array([A[c] for r in rows]).T
    list of float, optional
        If `solve_least_squares` is True, then return the least-squares
        solution of
            A_sparse.T.dot(x_sparse) = y

    Examples
    --------

    >>> import gols
    >>> A = np.array([[1,0], [0,1], [1,1]])
    >>> y = np.array([0, 2])
    >>> rows, _ = gols.solve(A, y, 1)
    >>> A_sparse = np.array([A[r] for r in rows])
    >>> x_sparse, _, _, _ = np.linalg.lstsq(A_sparse.T, y)
    >>> print np.linalg.norm(A_sparse.T.dot(x_sparse) - y)
    0.0
    """
    return _gols.gols_solve(A, y, k, L, solve_least_squares)


def _python_gols_solve(dictionary, y, k, L):
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
        k (int): Degree of sparsity, i.e. we'll try to reconstruct
            the signal using 'k' x 'L' atoms.
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
    n, m = A.shape
    P = numpy.eye(n)
    S = set()
    I = range(m)
    for i in range(0, min(k, m / L)):
        Pa = numpy.array([P.dot(A[:, j]) for j in I])
        Pa = numpy.divide(Pa, numpy.linalg.norm(Pa, axis=1)[:,None])
        gamma = numpy.array([abs(y.dot(porta)) for porta in Pa])
        L_largest = gamma.argsort()[-L:]
        largest_indices = [I[j] for j in L_largest]
        S.update(largest_indices)
        [I.pop(bisect.bisect_left(I, j)) for j in largest_indices]
        for j in largest_indices:
            Da = P.dot(A[:, j])
            d = Da / numpy.linalg.norm(Da)
            P = P - numpy.outer(d, d)
    S = list(S)
    As = numpy.array([A[:, j] for j in S]).T
    xs, _, _, _ = numpy.linalg.lstsq(As, y)
    return S, xs