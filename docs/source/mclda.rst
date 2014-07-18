.. _mclda:

Multi-class Linear Discriminant Analysis
==========================================

*Multi-class LDA* is a generalization of standard two-class LDA that can handle arbitrary number of classes. 

Overview
~~~~~~~~~~

*Multi-class LDA* is based on the analysis of two scatter matrices: *with-class scatter matrix* and *between-class scatter matrix*.

Given a set of samples :math:`\mathbf{x}_1, \ldots, \mathbf{x}_n`, and their class labels :math:`y_1, \ldots, y_n`:

The **with-class scatter matrix** is defined as:

.. math::

    \mathbf{S}_w = \sum_{i=1}^n (\mathbf{x}_i - \boldsymbol{\mu}_{y_i}) (\mathbf{x}_i - \boldsymbol{\mu}_{y_i})^T

Here, :math:`\boldsymbol{\mu}_k` is the sample mean of the ``k``-th class. 

The **between-class scatter matrix** is defined as:

.. math::

    \mathbf{S}_b = \sum_{k=1}^m n_k (\boldsymbol{\mu}_k - \boldsymbol{\mu}) (\boldsymbol{\mu}_k - \boldsymbol{\mu})^T

Here, ``m`` is the number of classes, :math:`\boldsymbol{\mu}` is the overall sample mean, and :math:`n_k` is the number of samples in the ``k``-th class.

Then, multi-class LDA can be formulated as an optimization problem to find a set of linear combinations (with coefficients :math:`\mathbf{w}`) that maximizes the ratio of the between-class scattering to the within-class scattering, as

.. math::

    \hat{\mathbf{w}} = \mathop{\mathrm{argmax}}_{\mathbf{w}}
    \frac{\mathbf{w}^T \mathbf{S}_b \mathbf{w}}{\mathbf{w}^T \mathbf{S}_w \mathbf{w}}

The solution is given by the following generalized eigenvalue problem:

.. math::

    \mathbf{S}_b \mathbf{w} = \lambda \mathbf{S}_w \mathbf{w}

Generally, at most ``m - 1`` generalized eigenvectors are useful to discriminate between ``m`` classes.


Multi-class LDA
~~~~~~~~~~~~~~~~~

The package defines a ``MulticlassLDA`` type to represent a multi-class LDA model, as:

.. code-block:: julia

    type MulticlassLDA
        proj::Matrix{Float64}
        pmeans::Matrix{Float64}
        stats::MulticlassLDAStats
    end

Here, ``proj`` is the projection matrix, ``pmeans`` is the projected means of all classes, ``stats`` is an instance of ``MulticlassLDAStats`` that captures all statistics computed to train the model (which we will discuss later).

Several methods are provided to access properties of the LDA model. Let ``M`` be an instance of ``MulticlassLDA``:

.. function:: indim(M)

    Get the input dimension (*i.e* the dimension of the observation space).

.. function:: outdim(M)

    Get the output dimension (*i.e* the dimension of the transformed features).

.. function:: projection(M)

    Get the projection matrix (of size ``d x p``). 

.. function:: mean(M)

    Get the overall sample mean vector (of length ``d``).

.. function:: classmeans(M)

    Get the matrix comprised of class-specific means as columns (of size ``(d, m)``).

.. function:: classweights(M)

    Get the weights of individual classes (a vector of length ``m``). If the samples are not weighted, the weight equals the number of samples of each class. 

.. function:: withinclass_scatter(M)

    Get the within-class scatter matrix (of size ``(d, d)``).

.. function:: betweenclass_scatter(M)

    Get the between-class scatter matrix (of size ``(d, d)``).

.. function:: transform(M, x)

    Transform input sample(s) in ``x`` to the output space. Here, ``x`` can be either a sample vector or a matrix comprised of samples in columns.

    In the pratice of classification, one can transform testing samples using this ``transform`` method, and compare them with ``M.pmeans``.


Data Analysis
~~~~~~~~~~~~~~

One can use ``fit`` to perform multi-class LDA over a set of data:

.. function:: fit(MulticlassLDA, nc, X, y; ...)

    Perform multi-class LDA over a given data set.

    :param nc:  the number of classes
    :param X:   the matrix of input samples, of size ``(d, n)``. Each column in ``X`` is an observation.
    :param y:   the vector of class labels, of length ``n``. Each element of ``y`` must be an integer between ``1`` and ``nc``.

    :return: The resultant multi-class LDA model, of type ``MulticlassLDA``.

    **Keyword arguments:**

    =========== =============================================================== ====================
      name         description                                                   default
    =========== =============================================================== ====================
     method     The choice of methods:                                           ``:gevd``

                - ``:gevd``: based on generalized eigenvalue decomposition
                - ``:whiten``: first derive a whitening transform from ``Sw``
                  and then solve the problem based on eigenvalue
                  decomposition of the whiten ``Sb``.
    ----------- --------------------------------------------------------------- --------------------
     outdim     The output dimension, *i.e* dimension of the transformed space   ``min(d, nc-1)``
    ----------- --------------------------------------------------------------- --------------------
     regcoef    The regularization coefficient.                                  ``1.0e-6``
                A positive value ``regcoef * eigmax(Sw)`` is added to the
                diagonal of ``Sw`` to improve numerical stability. 
    =========== =============================================================== ====================    

    **Note:** The resultant projection matrix ``P`` satisfies:

    .. math::

        \mathbf{P}^T (\mathbf{S}_w + \kappa \mathbf{I}) \mathbf{P} = \mathbf{I}

    Here, :math:`\kappa` equals ``regcoef * eigmax(Sw)``. The columns of ``P`` are arranged in descending order of the corresponding generalized eigenvalues.


Task Functions
~~~~~~~~~~~~~~~

The multi-class LDA consists of several steps:

1. Compute statistics, such as class means, scatter matrices, etc. 
2. Solve the projection matrix.
3. Construct the model.

Sometimes, it is useful to only perform one of these tasks. The package exposes several functions for this purpose:

.. function:: multiclass_lda_stats(nc, X, y)

    Compute statistics required to train a multi-class LDA.

    :param nc:  the number of classes
    :param X:   the matrix of input samples.
    :param y:   the vector of class labels.

    This function returns an instance of ``MulticlassLDAStats``, defined as below, that captures all relevant statistics.

    .. code-block:: julia

        type MulticlassLDAStats
            dim::Int                    # sample dimensions
            nclasses::Int               # number of classes
            cweights::Vector{Float64}   # class weights
            tweight::Float64            # total sample weight
            mean::Vector{Float64}       # overall sample mean
            cmeans::Matrix{Float64}     # class-specific means
            Sw::Matrix{Float64}         # within-class scatter matrix
            Sb::Matrix{Float64}         # between-class scatter matrix
        end

    This type has the following constructor. Under certain circumstances, one might collect statistics in other ways and want to directly construct this instance.

.. function:: MulticlassLDAStats(cweights, mean, cmeans, Sw, Sb)

    Construct an instance of type ``MulticlassLDAStats``.

    :param cweights:  the class weights, a vector of length ``m``.
    :param mean: the overall sample mean, a vector of length ``d``.
    :param cmeans: the class-specific sample means, a matrix of size ``(d, m)``.
    :param Sw: the within-class scatter matrix, a matrix of size ``(d, d)``.
    :param Sb: the between-class scatter matrix, a matrix of size ``(d, d)``. 


.. function:: multiclass_lda(S; ...)

    Perform multi-class LDA based on given statistics. Here ``S`` is an instance of ``MulticlassLDAStats``.

    This function accepts the following keyword arguments (as above): ``method``, ``outdim``, and ``regcoef``.

.. function:: mclda_solve(Sb, Sw, method, p, regcoef)

    Solve the projection matrix given both scatter matrices.

    :param Sb: the between-class scatter matrix.
    :param Sw: the within-class scatter matrix.
    :param method: the choice of method, which can be either ``:gevd`` or ``:whiten``.
    :param p: output dimension.
    :param regcoef: regularization coefficient.

.. function:: mclda_solve!(Sb, Sw, method, p, regcoef)

    Solve the projection matrix given both scatter matrices. 

    **Note:** In this function, ``Sb`` and ``Sw`` will be overwritten (saving some space).



