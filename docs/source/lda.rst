Linear Discriminant Analysis
==============================

`Linear Discriminant Analysis <http://en.wikipedia.org/wiki/Linear_discriminant_analysis>`_ are statistical analysis methods to find a linear combination of features for separating observations in two classes.

**Note:** Please refer to :ref:`mclda` for methods that can discriminate between multiple classes. 

Overview of LDA
~~~~~~~~~~~~~~~~

Suppose the samples in the positive and negative classes respectively with means: :math:`\boldsymbol{\mu}_p` and :math:`\boldsymbol{\mu}_n`, and covariances :math:`\mathbf{C}_p` and :math:`\mathbf{C}_n`. Then based on *Fisher's Linear Discriminant Criteria*, the optimal projection direction can be expressed as:

.. math::

    \mathbf{w} = \alpha \cdot (\mathbf{C}_p + \mathbf{C}_n)^{-1} (\boldsymbol{\mu}_p - \boldsymbol{\mu}_n)

Here ``α`` is an arbitrary non-negative coefficient. 


Linear Discriminant
~~~~~~~~~~~~~~~~~~~~

A linear discriminant functional can be written as

.. math::

    f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b

Here, ``w`` is the coefficient vector, and ``b`` is the bias constant.

This package uses the ``LinearDiscriminant`` type, defined as below, to capture a linear discriminant functional:

.. code-block:: julia

    immutable LinearDiscriminant <: Discriminant
        w::Vector{Float64}
        b::Float64
    end

This type comes with several methods. Let ``f`` be an instance of ``LinearDiscriminant``

.. function:: length(f)

    Get the length of the coefficient vector.

.. function:: evaluate(f, x)

    Evaluate the linear discriminant value, *i.e* ``w'x + b``.

    When ``x`` is a vector, it returns a real value; 
    when ``x`` is a matrix with samples in columns, it returns a vector of length ``size(x, 2)``. 

.. function:: predict(f, x)

    Make prediction. It returns ``true`` iff ``evaluate(f, x)`` is positive.


Data Analysis
~~~~~~~~~~~~~~

The package provides several functions to perform Linear Discriminant Analysis.

.. function:: ldacov(Cp, Cn, μp, μn)

    Performs LDA given covariances and mean vectors.

    :param Cp: The covariance matrix of the positive class.
    :param Cn: The covariance matrix of the negative class.
    :param μp: The mean vector of the positive class.
    :param μn: The mean vector of the negative class.

    :return: The resultant linear discriminant functional of type ``LinearDiscriminant``.

    **Note:** The coefficient vector is scaled such that ``w'μp + b = 1`` and ``w'μn + b = -1``.

.. function:: ldacov(C, μp, μn)

    Performs LDA given a covariance matrix and both mean vectors. 

    :param C: The pooled covariane matrix (*i.e* ``(Cp + Cn)/2``)
    :param μp: The mean vector of the positive class.
    :param μn: The mean vector of the negative class.

    :return: The resultant linear discriminant functional of type ``LinearDiscriminant``.

    **Note:** The coefficient vector is scaled such that ``w'μp + b = 1`` and ``w'μn + b = -1``.

.. function:: fit(LinearDiscriminant, Xp, Xn; covestimator = SimpleCovariance())

    Performs LDA given both positive and negative samples. 

    :param Xp: The sample matrix of the positive class.
    :param Xn: The sample matrix of the negative class.

    :return: The resultant linear discriminant functional of type ``LinearDiscriminant``.

    **Keyword arguments:**

    ============== =============================================================== =========================
      name                    description                                                          default
    ============== =============================================================== =========================
     covestimator   Custom covariance estimator for between-class covariance.          ``SimpleCovariance()``
                    The covariance matrix will be calculated as
                    ``cov(covestimator_between, #=data=#; dims=2,
                    mean=zeros(#=...=#)``.
                    Custom covariance estimators, available in other packages,
                    may result in more robust discriminants for data with more
                    features than observations.
    ============== =============================================================== =========================
