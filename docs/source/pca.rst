Principal Component Analysis
==============================

`Principal Component Analysis <http://en.wikipedia.org/wiki/Principal_component_analysis>`_ (PCA) derives an orthogonal projection to convert a given set of observations to linearly uncorrelated variables, called *principal components*.

This package defines a ``PCA`` type to represent a PCA model, and provides a set of methods to access the properties.

Properties
~~~~~~~~~~~

Let ``M`` be an instance of ``PCA``, ``d`` be the dimension of observations, and ``p`` be the output dimension (*i.e* the dimension of the principal subspace)


Transformation and Construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given a PCA model ``M``, one can use it to transform observations into principal components, as

.. math::

    \mathbf{y} = \mathbf{P}^T (\mathbf{x} - \boldsymbol{\mu})

or use it to reconstruct (approximately) the observations from principal components, as

.. math::

    \tilde{\mathbf{x}} = \mathbf{P} \mathbf{y} + \boldsymbol{\mu}

Here, :math:`\mathbf{P}` is the projection matrix.

The package provides methods to do so:

.. function:: transform(M, x)

    Transform observations ``x`` into principal components.

    Here, ``x`` can be either a vector of length ``d`` or a matrix where each column is an observation.

.. function:: reconstruct(M, y)

    Approximately reconstruct observations from the principal components given in ``y``.

    Here, ``y`` can be either a vector of length ``p`` or a matrix where each column gives the principal components for an observation.


Data Analysis
~~~~~~~~~~~~~~~

One can use the ``fit`` method to perform PCA over a given dataset.

.. function:: fit(PCA, X; ...)

    Perform PCA over the data given in a matrix ``X``. Each column of ``X`` is an observation.

    This method returns an instance of ``PCA``.

    **Keyword arguments:**

    Let ``(d, n) = size(X)`` be respectively the input dimension and the number of observations:

    =========== =============================================================== ===============
      name         description                                                   default
    =========== =============================================================== ===============
     method     The choice of methods:                                           ``:auto``

                - ``:auto``: use ``:cov`` when ``d < n`` or ``:svd`` otherwise
                - ``:cov``: based on covariance matrix
                - ``:svd``: based on SVD of the input data
    ----------- --------------------------------------------------------------- ---------------
     maxoutdim  Maximum output dimension.                                        ``min(d, n)``
    ----------- --------------------------------------------------------------- ---------------
     pratio     The ratio of variances preserved in the principal subspace.      ``0.99``
    ----------- --------------------------------------------------------------- ---------------
     mean       The mean vector, which can be either of:                         ``nothing``

                - ``0``: the input data has already been centralized
                - ``nothing``: this function will compute the mean
                - a pre-computed mean vector
    =========== =============================================================== ===============

    **Notes:**

    - The output dimension ``p`` depends on both ``maxoutdim`` and ``pratio``, as follows. Suppose
      the first ``k`` principal components preserve at least ``pratio`` of the total variance, while the
      first ``k-1`` preserves less than ``pratio``, then the actual output dimension will be ``min(k, maxoutdim)``.

    - This function calls ``pcacov`` or ``pcasvd`` internally, depending on the choice of method.

**Example:**

.. code-block:: julia

    using MultivariateStats

    # suppose Xtr and Xte are training and testing data matrix,
    # with each observation in a column

    # train a PCA model
    M = fit(PCA, Xtr; maxoutdim=100)

    # apply PCA model to testing set
    Yte = transform(M, Xte)

    # reconstruct testing observations (approximately)
    Xr = reconstruct(M, Yte)

**Example with iris dataset and plotting:**

.. code-block:: julia

    using MultivariateStats, RDatasets, Plots
    plotly() # using plotly for 3D-interacive graphing

    # load iris dataset
    iris = dataset("datasets", "iris")

    # split half to training set
    Xtr = convert(Matrix, iris[1:2:end,1:4])'
    Xtr_labels = convert(Vector, iris[1:2:end,5])

    # split other half to testing set
    Xte = convert(Matrix, iris[2:2:end,1:4])'
    Xte_labels = convert(Vector, iris[2:2:end,5])

    # suppose Xtr and Xte are training and testing data matrix,
    # with each observation in a column

    # train a PCA model, allowing up to 3 dimensions
    M = fit(PCA, Xtr; maxoutdim=3)

    # apply PCA model to testing set
    Yte = transform(M, Xte)

    # reconstruct testing observations (approximately)
    Xr = reconstruct(M, Yte)

    # group results by testing set labels for color coding
    setosa = Yte[:,Xte_labels.=="setosa"]
    versicolor = Yte[:,Xte_labels.=="versicolor"]
    virginica = Yte[:,Xte_labels.=="virginica"]

    # visualize first 3 principal components in 3D interacive plot
    p = scatter(setosa[1,:],setosa[2,:],setosa[3,:],marker=:circle,linewidth=0)
    scatter!(versicolor[1,:],versicolor[2,:],versicolor[3,:],marker=:circle,linewidth=0)
    scatter!(virginica[1,:],virginica[2,:],virginica[3,:],marker=:circle,linewidth=0)
    plot!(p,xlabel="PC1",ylabel="PC2",zlabel="PC3")

Core Algorithms
~~~~~~~~~~~~~~~~~

Two algorithms are implemented in this package: ``pcacov`` and ``pcasvd``.

.. function:: pcacov(C, mean; ...)

    Compute PCA based on eigenvalue decomposition of a given covariance matrix ``C``.

    :param C: The covariance matrix.

    :param mean: The mean vector of original samples, which can be a vector of length ``d``,
           or an empty vector ``Float64[]`` indicating a zero mean.

    :return: The resultant PCA model.

    :note: This function accepts two keyword arguments: ``maxoutdim`` and ``pratio``.

.. function:: pcasvd(Z, mean, tw; ...)

    Compute PCA based on singular value decomposition of a centralized sample matrix ``Z``.

    :param Z: provides centralized samples.

    :param mean: The mean vector of the **original** samples, which can be a vector of length ``d``,
                 or an empty vector ``Float64[]`` indicating a zero mean.

    :return: The resultant PCA model.

    :note: This function accepts two keyword arguments: ``maxoutdim`` and ``pratio``.

