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

   Here, ``sqr(D)`` indicates the element-wise square of ``D``, and ``g`` is the diagonal elements of ``G``. This relation is
   itself based on the following decomposition of squared Euclidean distance:

   .. math::

        \| \mathbf{x} - \mathbf{y} \|^2 = \| \mathbf{x} \|^2 + \| \mathbf{y} \|^2 - 2 \mathbf{x}^T \mathbf{y}

2. Perform eigenvalue decomposition of the Gram matrix to derive the coordinates.


Functions
~~~~~~~~~~

This package provides functions related to classical MDS.

.. function:: gram2dmat(G)

    Convert a Gram matrix ``G`` to a distance matrix.

.. function:: gram2dmat!(D, G)

    Convert a Gram matrix ``G`` to a distance matrix, and write the results to ``D``.

.. function:: dmat2gram(D)

    Convert a a distance matrix ``D`` to a Gram matrix.

.. function:: dmat2gram!(G, D)

    Convert a a distance matrix ``D`` to a Gram matrix, and write the results to ``G``.

.. function:: classical_mds(D, p)

    Perform classical MDS. This function derives a ``p``-dimensional embedding based on a given distance matrix ``D``. 

    It returns a coordinate matrix of size ``(p, n)``, where each column is the coordinates for an observation.

