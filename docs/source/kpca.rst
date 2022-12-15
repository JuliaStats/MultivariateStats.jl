Kernel Principal Component Analysis
==============================

`Kernel Principal Component Analysis <https://en.wikipedia.org/wiki/Kernel_principal_component_analysis>`_ (kernel PCA) is an extension of principal component analysis (PCA) using techniques of kernel methods. Using a kernel, the originally linear operations of PCA are performed in a reproducing kernel Hilbert space.

This package defines a ``KernelPCA`` type to represent a kernel PCA model, and provides a set of methods to access the properties.

Properties
~~~~~~~~~~~

Let ``M`` be an instance of ``KernelPCA``, ``d`` be the dimension of observations, and ``p`` be the output dimension (*i.e* the dimension of the principal subspace)

.. function:: indim(M)

    Get the input dimension ``d``, *i.e* the dimension of the observation space.

.. function:: outdim(M)

    Get the output dimension ``p``, *i.e* the dimension of the principal subspace.

.. function:: projection(M)

    Get the projection matrix (of size ``(n, p)``). Each column of the projection matrix corresponds to an eigenvector, and ``n`` is a number of observations.

    The principal components are arranged in descending order of the corresponding eigenvalues.

.. function:: principalvars(M)

    The variances of principal components.

Transformation and Construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The package provides methods to do so:

.. function:: transform(M, x)

    Transform observations ``x`` into principal components.

    Here, ``x`` can be either a vector of length ``d`` or a matrix where each column is an observation.

.. function:: reconstruct(M, y)

    Approximately reconstruct observations from the principal components given in ``y``.

    Here, ``y`` can be either a vector of length ``p`` or a matrix where each column gives the principal components for an observation.


Data Analysis
~~~~~~~~~~~~~~~

One can use the ``fit`` method to perform kernel PCA over a given dataset.

.. function:: fit(KernelPCA, X; ...)

    Perform kernel PCA over the data given in a matrix ``X``. Each column of ``X`` is an observation.

    This method returns an instance of ``KernelPCA``.

    **Keyword arguments:**

    Let ``(d, n) = size(X)`` be respectively the input dimension and the number of observations:

    =========== =============================================================== ===============
      name         description                                                   default
    =========== =============================================================== ===============
     kernel     The kernel function:                                             ``(x,y)->x'y``

                This functions accepts two vector arguments ``x`` and ``y``,
                and returns a scalar value.
                
                If ``X`` is a precomputed kernel matrix (Gramian), set 
                ``kernel=nothing``.
    ----------- --------------------------------------------------------------- ---------------
     solver     The choice of solver:                                            ``:eig``

                - ``:eig``: uses ``eigfact``
                - ``:eigs``: uses ``eigs`` (always used for sparse data)
    ----------- --------------------------------------------------------------- ---------------
     maxoutdim  Maximum output dimension.                                        ``min(d, n)``
    ----------- --------------------------------------------------------------- ---------------
     inverse    Whether to perform calculation for inverse transform for         ``false``
                non-precomputed kernels.
    ----------- --------------------------------------------------------------- ---------------
     β          Hyperparameter of the ridge regression that learns the           ``1.0``
                inverse transform (when ``inverse`` is ``true``).
    ----------- --------------------------------------------------------------- ---------------
     tol        Convergence tolerance for ``eigs`` solver                        ``0.0``
    ----------- --------------------------------------------------------------- ---------------
     maxiter    Maximum number of iterations for ``eigs`` solver                 ``300``
    =========== =============================================================== ===============

Kernels
~~~~~~~~~~~~~~~

List of the commonly used kernels:

    ================================================== ========================================
      function                                           description
    ================================================== ========================================
     ``(x,y)->x'y``                                      Linear
    -------------------------------------------------- ----------------------------------------
     ``(x,y)->(x'y+c)^d``                                Polynomial
    -------------------------------------------------- ----------------------------------------
     ``(x,y)->exp(-γ*norm(x-y)^2.0)``                    Radial basis function (RBF)
    ================================================== ========================================

**Example:**

.. code-block:: julia

    using MultivariateStats

    # suppose Xtr and Xte are training and testing data matrix,
    # with each observation in a column

    # train a kernel PCA model
    M = fit(KernelPCA, Xtr; maxoutdim=100, inverse=true)

    # apply kernel PCA model to testing set
    Yte = transform(M, Xte)

    # reconstruct testing observations (approximately)
    Xr = reconstruct(M, Yte)
