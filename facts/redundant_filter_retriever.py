from typing import Optional, List, Dict, Any
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Chroma
from langchain.schema import BaseRetriever
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document

class RedundantFilterRetriever(BaseRetriever):
    embeddings: Embeddings
    db: Chroma

    def get_relevant_documents(
        self,
        query: str,
        *,
        callbacks: Callbacks = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        # calculate embeddings for the 'query' string
        emb = self.embeddings.embed_query(query)

        # take embeddings and feed them into that
        #max_marginal_relevance_search_by_vector
        return self.db.max_marginal_relevance_search_by_vector(
            embedding = emb,
            lambda_mult = 0.8
        )

    def aget_relevant_documents(
        self,
        query: str,
        *,
        callbacks: Callbacks = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        return []