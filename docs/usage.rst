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

Using LangChain/LangGraph
-------------------------

Run orchestration via CLI:

.. code-block:: bash

   rag chain --q "what is in these docs?" --k 3 --engine langchain --no-stream
   rag chain --q "summarize the corpus" --engine langgraph --stream

Serve API with orchestration config:

.. code-block:: bash

   rag serve --engine langgraph --stream

API usage:

.. code-block:: bash

   curl -s -X POST localhost:8002/chain_query \
     -H 'Content-Type: application/json' \
     -d '{"query":"test","k":3,"engine":"langchain","stream":false}'

   curl -s -X POST localhost:8002/chain_stream \
     -H 'Content-Type: application/json' \
     -d '{"query":"stream me","engine":"langgraph","stream":true}' | head

Evaluation
----------

.. code-block:: bash

   rag eval --qrels data/qrels.tsv --queries data/queries.tsv --k 10

Docs
----

.. code-block:: bash

   make docs
