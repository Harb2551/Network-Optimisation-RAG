import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import pickle
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

class VectorStoreManager:
    """
    Manages hybrid vector retrieval combining FAISS and BM25 for enhanced search.

    This class creates and manages an ensemble retriever that combines dense vector search
    (FAISS) with sparse vector search (BM25). The hybrid approach provides better recall
    by capturing both semantic similarity and keyword matching, making it effective for 
    technical support and troubleshooting scenarios.

    The manager can load existing vector stores from disk or create new ones when needed.
    It handles both incident documents and technical documents separately, allowing for
    targeted retrieval based on document type.

    Retrieval includes filtering by product ID to ensure relevant results for the specific
    product in question.
    """
    def __init__(self, incident_docs, tech_docs, top_k=5,
                 incident_faiss_path="index/incident_faiss",
                 tech_faiss_path="index/tech_faiss",
                 incident_bm25_path="index/incident_bm25",
                 tech_bm25_path="index/tech_bm25",
                 api_key=None):
        """
        Initialize the hybrid vector store manager with incident and technical documents.

        This class creates and manages both FAISS vector stores and BM25 indexes for efficient
        hybrid search across two document collections. It uses OpenAI's 'text-embedding-ada-002'
        model for vector embeddings.

        Args:
            incident_docs (list): List of incident documents to be indexed.
            tech_docs (list): List of technical documents to be indexed.
            top_k (int, optional): Number of documents to retrieve in each search. Defaults to 5.
            incident_faiss_path (str, optional): Path for storing incident FAISS vectors.
                Defaults to "index/incident_faiss".
            tech_faiss_path (str, optional): Path for storing technical FAISS vectors.
                Defaults to "index/tech_faiss".
            incident_bm25_path (str, optional): Path for storing incident BM25 index.
                Defaults to "index/incident_bm25".
            tech_bm25_path (str, optional): Path for storing technical BM25 index.
                Defaults to "index/tech_bm25".
            api_key (str, optional): OpenAI API key for embeddings. If None, will attempt
                to load from environment variables. Defaults to None.

        Attributes:
            incident_docs (list): List of incident documents.
            tech_docs (list): List of technical documents.
            k (int): Number of documents to retrieve.
            incident_faiss_path (str): Path for incident FAISS index.
            tech_faiss_path (str): Path for technical FAISS index.
            incident_bm25_path (str): Path for incident BM25 index.
            tech_bm25_path (str): Path for technical BM25 index.
            api_key (str): OpenAI API key.
            embeddings (OpenAIEmbeddings): Embedding model instance.
            incident_faiss_vectorstore (FAISS): Vector store for incident documents.
            tech_faiss_vectorstore (FAISS): Vector store for technical documents.
            incident_ensemble_retriever (EnsembleRetriever): Hybrid retriever for incident docs.
            tech_ensemble_retriever (EnsembleRetriever): Hybrid retriever for technical docs.

        Raises:
            ValueError: If OPENAI_API_KEY is not found in environment variables when api_key is None.
            Exception: Re-raises any other exceptions that occur during initialization.

        Note:
            The method automatically calls _initialize_vector_stores() to set up the search
            infrastructure when instantiated.
        """
        load_dotenv()
        self.incident_docs = incident_docs
        self.tech_docs = tech_docs
        self.k = top_k
        self.incident_faiss_path = incident_faiss_path
        self.tech_faiss_path = tech_faiss_path
        self.incident_bm25_path = incident_bm25_path
        self.tech_bm25_path = tech_bm25_path
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY must be set")

        self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        self._initialize_vector_stores()


    def _initialize_vector_stores(self):
        """
        Initialize vector stores by loading existing ones or creating new ones.

        This method performs the following operations:
        1. Checks if incident and technical indexes (both FAISS and BM25) exist on disk
        2. Loads existing indexes if they're available
        3. Creates new indexes from the provided documents if they don't exist
        4. Sets up individual FAISS and BM25 retrievers for each document collection
        5. Combines the retrievers into ensemble retrievers for hybrid search capability

        The method handles two separate document collections (incident and technical)
        with parallel index structures for each. This separation allows for targeted
        searches within specific document domains.

        Attributes set:
            incident_faiss_vectorstore (FAISS): Vector store for incident documents
            tech_faiss_vectorstore (FAISS): Vector store for technical documents
            incident_ensemble_retriever (EnsembleRetriever): Hybrid retriever for incident docs
            tech_ensemble_retriever (EnsembleRetriever): Hybrid retriever for technical docs

        Methods called internally:
            _load_faiss_vectorstore(): To load existing FAISS indexes
            _load_bm25(): To load existing BM25 indexes
            _create_ensemble(): To combine FAISS and BM25 retrievers
            _create_incident_store(): To build new incident indexes if needed
            _create_tech_store(): To build new technical indexes if needed

        Raises:
            Exception: Re-raises any exceptions that occur during initialization of vector stores,
                including file I/O errors, serialization errors, or issues with embeddings.
        """
        if os.path.exists(self.incident_faiss_path) and os.path.exists(self.incident_bm25_path):
            self.incident_faiss_vectorstore = self._load_faiss_vectorstore(self.incident_faiss_path)
            incident_bm25 = self._load_bm25(self.incident_bm25_path)
            self.incident_ensemble_retriever = self._create_ensemble(
                self.incident_faiss_vectorstore.as_retriever(search_kwargs={"k": self.k}),
                incident_bm25
            )
        else:
            self._create_incident_store()

        if os.path.exists(self.tech_faiss_path) and os.path.exists(self.tech_bm25_path):
            self.tech_faiss_vectorstore = self._load_faiss_vectorstore(self.tech_faiss_path)
            tech_bm25 = self._load_bm25(self.tech_bm25_path)
            self.tech_ensemble_retriever = self._create_ensemble(
                self.tech_faiss_vectorstore.as_retriever(search_kwargs={"k": self.k}),
                tech_bm25
            )
        else:
            self._create_tech_store()



    def _create_incident_store(self):
        """
        Create and save incident vector store and retrievers.

        This method creates the complete search infrastructure for incident documents,
        including both vector-based and keyword-based search capabilities. It performs
        the following operations:

        1. Creates necessary directories if they don't exist for both FAISS and BM25 indexes
        2. Creates a FAISS vectorstore from incident documents using the configured embeddings
        3. Saves the FAISS vectorstore to the specified path
        4. Creates a FAISS retriever with the configured top-k parameter
        5. Creates a BM25 retriever from incident documents with the configured top-k parameter
        6. Saves the BM25 retriever to disk using pickle serialization
        7. Creates an ensemble retriever that combines both search approaches for hybrid search

        Attributes set:
            incident_faiss_vectorstore (FAISS): Vector store for incident documents
            incident_ensemble_retriever (EnsembleRetriever): Hybrid retriever combining FAISS and BM25

        Methods called internally:
            _create_ensemble(): To combine FAISS and BM25 retrievers into a single hybrid retriever

        Raises:
            Exception: Re-raises any exceptions that occur during store creation, including:
                - Directory creation errors
                - Embedding generation issues
                - FAISS index creation failures
                - Pickle serialization errors
                - Memory allocation problems
        """
        os.makedirs(os.path.dirname(self.incident_faiss_path), exist_ok=True)
        self.incident_faiss_vectorstore = FAISS.from_documents(self.incident_docs, self.embeddings)
        self.incident_faiss_vectorstore.save_local(self.incident_faiss_path)

        bm25 = BM25Retriever.from_documents(self.incident_docs)
        bm25.k = self.k
        with open(self.incident_bm25_path, 'wb') as f:
            pickle.dump(bm25, f)

        self.incident_ensemble_retriever = self._create_ensemble(
            self.incident_faiss_vectorstore.as_retriever(search_kwargs={"k": self.k}), bm25
        )


    def _create_tech_store(self):
        """
        Create and save technical vector store and retrievers.

        This method creates the complete search infrastructure for technical documents,
        including both vector-based and keyword-based search capabilities. It performs
        the following operations:

        1. Creates necessary directories if they don't exist for both FAISS and BM25 indexes
        2. Creates a FAISS vectorstore from technical documents using the configured embeddings
        3. Saves the FAISS vectorstore to the specified path
        4. Creates a FAISS retriever with the configured top-k parameter
        5. Creates a BM25 retriever from technical documents with the configured top-k parameter
        6. Saves the BM25 retriever to disk using pickle serialization
        7. Creates an ensemble retriever that combines both search approaches for hybrid search

        Attributes set:
            tech_faiss_vectorstore (FAISS): Vector store for technical documents
            tech_ensemble_retriever (EnsembleRetriever): Hybrid retriever combining FAISS and BM25

        Methods called internally:
            _create_ensemble(): To combine FAISS and BM25 retrievers into a single hybrid retriever

        Raises:
            Exception: Re-raises any exceptions that occur during store creation, including:
                - Directory creation errors
                - Embedding generation issues
                - FAISS index creation failures
                - Pickle serialization errors
                - Memory allocation problems
        """
        os.makedirs(os.path.dirname(self.tech_faiss_path), exist_ok=True)
        self.tech_faiss_vectorstore = FAISS.from_documents(self.tech_docs, self.embeddings)
        self.tech_faiss_vectorstore.save_local(self.tech_faiss_path)

        bm25 = BM25Retriever.from_documents(self.tech_docs)
        bm25.k = self.k
        with open(self.tech_bm25_path, 'wb') as f:
            pickle.dump(bm25, f)

        self.tech_ensemble_retriever = self._create_ensemble(
            self.tech_faiss_vectorstore.as_retriever(search_kwargs={"k": self.k}), bm25
        )



    def _create_ensemble(self, faiss_retriever, bm25_retriever, weights=None):
        """
        Create an ensemble retriever from FAISS and BM25 retrievers.

        Args:
            faiss_retriever: FAISS-based retriever
            bm25_retriever: BM25-based retriever
            weights (list, optional): Weights for the retrievers. Default is [0.8, 0.2]

        Returns:
            EnsembleRetriever: Combined retriever
        """
        weights = weights or [0.8, 0.2]
        return EnsembleRetriever(retrievers=[faiss_retriever, bm25_retriever], weights=weights)


    def _load_faiss_vectorstore(self, path):
        """
        Load a FAISS vectorstore from disk.

        Args:
            path (str): Path to the stored FAISS index

        Returns:
            FAISS: FAISS vectorstore

        Raises:
            Exception: Re-raises any exceptions that occur during loading
            This includes file not found errors, deserialization errors, or embedding issues
        """
        return FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)


    def _load_bm25(self, path):
        """
        Load a BM25 retriever from disk.

        Args:
            path (str): Path to the stored BM25 retriever

        Returns:
            Retriever: BM25-based retriever

        Raises:
            Exception: Re-raises any exceptions that occur during loading
            This includes file not found errors or pickle deserialization errors
        """
        with open(path, 'rb') as f:
            return pickle.load(f)



    def retrieve_documents(self, query, product_id, store_type="incident", metadata_col=None):
        """
        Retrieve relevant documents using hybrid search with product filtering.

        This method performs a hybrid search combining vector similarity (FAISS) and
        keyword matching (BM25) to find the most relevant documents for a given query.
        Results are filtered by product_id to ensure they match the specific product
        in question.

        The search process follows these steps:
        1. Validates input parameters
        2. Selects the appropriate ensemble retriever and vector store based on store_type
        3. Performs a hybrid search using the ensemble retriever
        4. Filters results to only include documents matching the specified product_id

        Args:
            query (str): User's question or search query to find relevant documents.
            product_id (str): Product identifier to filter results. Only documents
                with matching ProductID in their metadata will be returned.
            store_type (str, optional): Type of document store to query. Must be either
                'incident' or 'tech'. Defaults to "incident".
            metadata_col (str, optional): Specific metadata column to extract from
                documents. If provided, only this column is returned. Currently unused
                in the implementation. Defaults to None.

        Returns:
            list: List of Document objects that match both the query and product ID.
                Returns an empty list if no matching documents are found.
            Exception: Returns the exception object itself if retrieval fails, rather
                than raising it. This allows the caller to handle errors appropriately.

        Raises:
            ValueError: If any of the following validation checks fail:
                - Query is not a non-empty string (message: "Query must be a non-empty string")
                - Product ID is not a string (message: "Product ID must be a string")
                - Store type is not 'incident' or 'tech' (message: "Store type must be 'incident' or 'tech'")

        Note:
            This method catches all exceptions during retrieval and returns the exception object
            rather than raising it. The caller should check if the return value is an Exception
            and handle it appropriately.

            The metadata_col parameter appears to be unused in the current implementation
            but is maintained for future compatibility.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")
        if not isinstance(product_id, str):
            raise ValueError("Product ID must be a string")
        if store_type not in ["incident", "tech"]:
            raise ValueError("Store type must be 'incident' or 'tech'")

        try:
            retriever = self.incident_ensemble_retriever if store_type == "incident" else self.tech_ensemble_retriever
            results = retriever.get_relevant_documents(query)
            filtered = [doc for doc in results if doc.metadata.get("ProductID") == product_id]
            return filtered
        except Exception as e:
            return e
