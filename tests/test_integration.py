import pytest
from unittest.mock import patch, MagicMock

class TestIntegrationWorkflow:
    """Integration tests for the document processing workflow"""
    
    @patch('src.document_loader.Guide_DocumentLoader')
    @patch('src.vector_store.VectorStoreManager')
    @patch('src.rag_chain.RAGChain')
    def test_basic_workflow(self, mock_rag_chain_class, mock_vector_store_class, mock_loader_class):
        """Test basic end-to-end workflow with mocked components"""
        # Configure mocks
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        
        mock_vector_store = MagicMock()
        mock_vector_store_class.return_value = mock_vector_store
        
        mock_rag_chain = MagicMock()
        mock_rag_chain_class.return_value = mock_rag_chain
        
        # Mock document loading
        incident_docs = [MagicMock(page_content="Network issue", metadata={"ProductID": "P1"})]
        tech_docs = [MagicMock(page_content="Network config steps", metadata={"ProductID": "P1"})]
        mock_loader.load_all_documents.return_value = (incident_docs, tech_docs)
        
        # Mock document retrieval
        mock_vector_store.retrieve_documents.return_value = [
            MagicMock(page_content="Solution for network issue", 
                     metadata={"ProductID": "P1", "SolutionDetails": "Reset adapter"})
        ]
        
        # Mock RAG chain response
        mock_rag_chain.run.return_value = "To fix your network issue: Reset adapter and restart router."
        
        # Execute workflow
        # 1. Load documents
        loader = mock_loader_class()
        incident_docs, tech_docs = loader.load_all_documents()
        
        # 2. Initialize vector store
        vector_store = mock_vector_store_class(
            incident_docs=incident_docs,
            tech_docs=tech_docs,
            api_key="test_api_key"
        )
        
        # 3. Retrieve documents for a query
        query = "How to fix network connectivity issues?"
        product_id = "P1"
        
        incident_results = vector_store.retrieve_documents(
            query=query, 
            product_id=product_id,
            store_type="incident"
        )
        
        tech_results = vector_store.retrieve_documents(
            query=query, 
            product_id=product_id,
            store_type="tech"
        )
        
        # 4. Process with RAG chain
        rag_chain = mock_rag_chain_class()
        response = rag_chain.run(
            query=query,
            tech_results=tech_results,
            incident_results=incident_results
        )
        
        # Verify workflow executed correctly
        mock_loader.load_all_documents.assert_called_once()
        mock_vector_store.retrieve_documents.assert_called()
        mock_rag_chain.run.assert_called_once()
        assert "network" in response.lower()
    
    @patch('src.document_loader.Guide_DocumentLoader')
    @patch('src.vector_store.VectorStoreManager')
    @patch('src.rag_chain.RAGChain')
    def test_no_relevant_documents(self, mock_rag_chain_class, mock_vector_store_class, mock_loader_class):
        """Test workflow when no relevant documents are found"""
        # Configure mocks
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        
        mock_vector_store = MagicMock()
        mock_vector_store_class.return_value = mock_vector_store
        
        mock_rag_chain = MagicMock()
        mock_rag_chain_class.return_value = mock_rag_chain
        
        # Mock document loading
        incident_docs = [MagicMock(page_content="Database issue", metadata={"ProductID": "P2"})]
        tech_docs = [MagicMock(page_content="Database config steps", metadata={"ProductID": "P2"})]
        mock_loader.load_all_documents.return_value = (incident_docs, tech_docs)
        
        # Mock document retrieval for no results
        mock_vector_store.retrieve_documents.return_value = "Contact customer support."
        
        # Mock RAG chain response
        mock_rag_chain.run.return_value = "Please enter a valid query with relevant context."
        
        # Execute workflow
        # 1. Load documents
        loader = mock_loader_class()
        incident_docs, tech_docs = loader.load_all_documents()
        
        # 2. Initialize vector store
        vector_store = mock_vector_store_class(
            incident_docs=incident_docs,
            tech_docs=tech_docs,
            api_key="test_api_key"
        )
        
        # 3. Retrieve documents for query with no relevant results
        query = "How to install operating system?"
        product_id = "P99"  # Non-existent product
        
        incident_results = vector_store.retrieve_documents(
            query=query, 
            product_id=product_id,
            store_type="incident"
        )
        
        tech_results = vector_store.retrieve_documents(
            query=query, 
            product_id=product_id,
            store_type="tech"
        )
        
        # 4. Process with RAG chain
        rag_chain = mock_rag_chain_class()
        response = rag_chain.run(
            query=query,
            tech_results=tech_results,
            incident_results=incident_results
        )
        
        # Verify workflow executed correctly
        assert incident_results == "Contact customer support."
        assert tech_results == "Contact customer support."
        assert "valid query" in response.lower()