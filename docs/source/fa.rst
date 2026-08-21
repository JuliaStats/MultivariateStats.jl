Factor Analysis
===============

`Factor Analysis <https://en.wikipedia.org/wiki/Factor_analysis>`_ (FA) is a linear-Gaussian latent variable model that is closely related to probabilistic PCA. In contrast to the probabilistic PCA model, the covariance of conditional distribution of the observed variable  given the latent variable is diagonal rather than isotropic [BSHP06]_.

This package defines a ``FactorAnalysis`` type to represent a factor analysis model, and provides a set of methods to access the properties.

Properties
~~~~~~~~~~~

Let ``M`` be an instance of ``FactorAnalysis``, ``d`` be the dimension of observations, and ``p`` be the output dimension (*i.e* the dimension of the principal subspace)

.. function:: indim(M)

    Get the input dimension ``d``, *i.e* the dimension of the observation space.

.. function:: outdim(M)

    Get the output dimension ``p``, *i.e* the dimension of the principal subspace.

.. function:: mean(M)

    Get the mean vector (of length ``d``).

.. function:: projection(M)

    Get the projection matrix (of size ``(d, p)``). Each column of the projection matrix corresponds to a principal component.

    The principal components are arranged in descending order of the corresponding variances.

.. function:: loadings(M)

    The factor loadings matrix (of size ``(d, p)``).

.. function:: cov(M)

    The diagonal covariance matrix.


Transformation and Construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given a probabilistic PCA model ``M``, one can use it to transform observations into latent variables, as

.. math::

    \mathbf{z} =  \mathbf{W}^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})

or use it to reconstruct (approximately) the observations from latent variables, as

.. math::

    \tilde{\mathbf{x}} = \mathbf{\Sigma} \mathbf{W} (\mathbf{W}^T \mathbf{W})^{-1} \mathbf{z} + \boldsymbol{\mu}

Here, :math:`\mathbf{W}` is the factor loadings or weight matrix, :math:`\mathbf{\Sigma} = \mathbf{\Psi} + \mathbf{W} \mathbf{W}^T` is the covariance matrix.

The package provides methods to do so:

.. function:: transform(M, x)

    Transform observations ``x`` into latent variables.

    Here, ``x`` can be either a vector of length ``d`` or a matrix where each column is an observation.

.. function:: reconstruct(M, z)

    Approximately reconstruct observations from the latent variable given in ``z``.

    Here, ``y`` can be either a vector of length ``p`` or a matrix where each column gives the latent variables for an observation.


Data Analysis
~~~~~~~~~~~~~~~

One can use the ``fit`` method to perform factor analysis over a given dataset.

.. function:: fit(FactorAnalysis, X; ...)

    Perform factor analysis over the data given in a matrix ``X``. Each column of ``X`` is an observation.

    This method returns an instance of ``FactorAnalysis``.

    **Keyword arguments:**

    Let ``(d, n) = size(X)`` be respectively the input dimension and the number of observations:

    =========== =============================================================== ===============
      name         description                                                   default
    =========== =============================================================== ===============
     method     The choice of methods:                                           ``:cm``

                - ``:em``: use EM version of factor analysis
                - ``:cm``: use CM version of factor analysis
    ----------- --------------------------------------------------------------- ---------------
     maxoutdim  Maximum output dimension                                         ``d-1``
    ----------- --------------------------------------------------------------- ---------------
     mean       The mean vector, which can be either of:                         ``nothing``

                - ``0``: the input data has already been centralized
                - ``nothing``: this function will compute the mean
                - a pre-computed mean vector
    ----------- --------------------------------------------------------------- ---------------
     tol        Convergence tolerance                                            ``1.0e-6``
    ----------- --------------------------------------------------------------- ---------------
     tot        Maximum number of iterations                                     ``1000``
    ----------- --------------------------------------------------------------- ---------------
     η          Variance low bound                                               ``1.0e-6``
    =========== =============================================================== ===============

    **Notes:**

    - This function calls ``facm`` or ``faem`` internally, depending on the choice of method.

**Example:**

.. code-block:: julia

    using MultivariateStats

    # suppose Xtr and Xte are training and testing data matrix,
    # with each observation in a column

    # train a FactorAnalysis model
    M = fit(FactorAnalysis, Xtr; maxoutdim=100)

    # apply FactorAnalysis model to testing set
    Yte = transform(M, Xte)

    # reconstruct testing observations (approximately)
    Xr = reconstruct(M, Yte)


Core Algorithms
~~~~~~~~~~~~~~~~~

Two algorithms are implemented in this package: ``faem`` and ``facm``.

.. function:: faem(S, mean, n; ...)

    Perform factor analysis using an expectation-maximization algorithm for a given sample covariance matrix ``S`` [RUBN82]_.

    :param S: The sample covariance matrix.

    :param mean: The mean vector of original samples, which can be a vector of length ``d``,
           or an empty vector ``Float64[]`` indicating a zero mean.

    :param n: The number of observations.

    :return: The resultant FactorAnalysis model.

    :note: This function accepts two keyword arguments: ``maxoutdim``,``tol``, and ``tot``.

.. function:: facm(S, mean, n; ...)

    Perform factor analysis using an fast conditional maximization algorithm for a given sample covariance matrix ``S`` [ZHAO08]_.

    :param S: The sample covariance matrix.

    :param mean: The mean vector of original samples, which can be a vector of length ``d``,
           or an empty vector ``Float64[]`` indicating a zero mean.

    :param n: The number of observations.

    :return: The resultant FactorAnalysis model.

    :note: This function accepts two keyword arguments: ``maxoutdim``, ``tol``, ``tot``, and ``η``.


References
~~~~~~~~~~

.. [RUBN82] Rubin, Donald B., and Dorothy T. Thayer. EM algorithms for ML factor analysis. Psychometrika 47.1 (1982): 69-76.
.. [ZHAO08] Zhao, J-H., Philip LH Yu, and Qibao Jiang. ML estimation for factor analysis: EM or non-EM?. Statistics and computing 18.2 (2008): 109-123.
