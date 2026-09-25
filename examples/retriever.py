"""The retriever: the latest articles on a topic, as documents with their link, outlet and score.

pip install langchain-typesearch
TYPESEARCH_API_KEY=ts_live_... python examples/retriever.py "lithium royalties in Chile"
"""

import sys

from langchain_typesearch import TypesearchRetriever

retriever = TypesearchRetriever(k=5, days=30, languages=["es", "en"])

for doc in retriever.invoke(" ".join(sys.argv[1:]) or "lithium royalties in Chile"):
    print(f"{doc.metadata['score']:.2f}  {doc.metadata['title']} — {doc.metadata.get('source')}\n      {doc.metadata['url']}")
