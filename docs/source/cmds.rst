Classical Multidimensional Scaling
====================================

In general, `Multidimensional Scaling <http://en.wikipedia.org/wiki/Multidimensional_scaling>`_ (MDS) refers to techniques that transforms samples into lower dimensional space while preserving the inter-sample distances as well as possible.

Overview of Classical MDS
~~~~~~~~~~~~~~~~~~~~~~~~~~

*Classical MDS* is a specific technique in this family that accomplishes the embedding in two steps:

1. Convert the distance matrix to a Gram matrix.

   This conversion is based on the following relations between a distance matrix ``D`` and a Gram matrix ``G``:

   .. math::

        \mathrm{sqr}(\mathbf{D}) = \mathbf{g} \mathbf{1}^T + \mathbf{1} \mathbf{g}^T - 2 \mathbf{G}

   Here, :math:`\mathrm{sqr}(\mathbf{D})` indicates the element-wise square of :math:`\mathbf{D}`, and :math:`\mathbf{g}` is the diagonal elements of :math:`\mathbf{G}`. This relation is
   itself based on the following decomposition of squared Euclidean distance:

   .. math::

        \| \mathbf{x} - \mathbf{y} \|^2 = \| \mathbf{x} \|^2 + \| \mathbf{y} \|^2 - 2 \mathbf{x}^T \mathbf{y}

2. Perform eigenvalue decomposition of the Gram matrix to derive the coordinates.


This package defines a ``MDS`` type to represent a classical MDS model, and provides a set of methods to access the properties.

Properties
~~~~~~~~~~~

Let ``M`` be an instance of ``MDS``, ``d`` be the dimension of observations, and ``p`` be the embedding dimension

.. function:: indim(M)

    Get the input dimension ``d``, *i.e* the dimension of the observation space.

.. function:: outdim(M)

    Get the output dimension ``p``, *i.e* the dimension of the embedding.

.. function:: projection(M)

    Get the eigenvectors matrix (of size ``(n, p)``) of the embedding space.

    The eigenvectors are arranged in descending order of the corresponding eigenvalues.

.. function:: eigvals(M)

    Get the eigenvalues.

.. function:: stress(M)

    Get the model stress.


Transformation and Construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The package provides methods to do so:

.. function:: transform(M)

    It returns a coordinate matrix of size ``(p, n)``, where each column is the
    coordinates for an observation.

.. function:: transform(M, x)

    Out-of-sample transformation of the observation ``x``.

    Here, ``x`` is a vector of length ``d``.


Data Analysis
~~~~~~~~~~~~~~~

One can use the ``fit`` method to perform classical MDS over a given dataset.

.. function:: fit(MDS, X; ...)

    Perform classical MDS over the data given in a matrix ``X``. Each column of ``X`` is an observation.

    This method returns an instance of ``MDS``.

    **Keyword arguments:**

    Let ``(d, n) = size(X)`` be respectively the input dimension and the number of observations:

    =========== =============================================================== ===============
      name         description                                                   default
    =========== =============================================================== ===============
     maxoutdim  Maximum output dimension.                                        ``d-1``
    ----------- --------------------------------------------------------------- ---------------
     distances  When the parameter is ``true``, the input matrix ``X`` is        ``false``
                treated as a distance matrix.
    =========== =============================================================== ===============

.. note::

    The Gramian derived from ``D`` may have nonpositive or degenerate
    eigenvalues.  The subspace of nonpositive eigenvalues is projected out
    of the MDS solution so that the strain function is minimized in a
    least-squares sense.  If the smallest remaining eigenvalue that is used
    for the MDS is degenerate, then the solution is not unique, as any
    linear combination of degenerate eigenvectors will also yield a MDS
    solution with the same strain value. By default, warnings are emitted
    if either situation is detected, which can be suppressed with
    ``dowarn=false``.

    If the MDS uses an eigenspace of dimension ``m`` less than ``p``, then
    the MDS coordinates will be padded with ``p-m`` zeros each.


Miscellaneous Functions
~~~~~~~~~~~~~~~~~~~~~~~

This package provides functions related to classical MDS.

.. function:: gram2dmat(G)

    Convert a Gram matrix ``G`` to a distance matrix.

.. function:: gram2dmat!(D, G)

    Convert a Gram matrix ``G`` to a distance matrix, and write the results to ``D``.

.. function:: dmat2gram(D)

    Convert a distance matrix ``D`` to a Gram matrix.

.. function:: dmat2gram!(G, D)

    Convert a distance matrix ``D`` to a Gram matrix, and write the results to ``G``.

    Reference::

        @inbook{Borg2005,
	Author = {Ingwer Borg and Patrick J. F. Groenen},
	Title = {Modern Multidimensional Scaling: Theory and Applications},
	Edition = {2},
	Year = {2005},
	Chapter = {12},
	Doi = {10.1007/0-387-28981-X},
	Pages = {201--268},
	Series = {Springer Series in Statistics},
	Publisher = {Springer},
        }

