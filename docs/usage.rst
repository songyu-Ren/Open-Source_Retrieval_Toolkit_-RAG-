Usage
=====

Install and Setup
-----------------

.. code-block:: bash

   make setup
   make dvc-init

Indexing
--------

.. code-block:: bash

   rag index --data data/raw

Querying
--------

.. code-block:: bash

   rag query --q "what is in these docs?" --k 5

Evaluation
----------

.. code-block:: bash

   rag eval --qrels data/qrels.tsv --queries data/queries.tsv --k 10

Docs
----

.. code-block:: bash

   make docs