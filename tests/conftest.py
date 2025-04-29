import pytest
from unittest.mock import MagicMock, patch
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def mock_csv_data():
    incident_src_data = {
        'TicketID': ['T1', 'T2', 'T3'],
        'ProductID': ['P1', 'P2', 'P3'],
        'ProblemDescription': [
            'System crashes when starting the network application',
            'Cannot connect to database after Windows update',
            'Network configuration issues after driver update'
        ]
    }
    
    incident_meta_data = {
        'TicketID': ['T1', 'T2', 'T3'],
        'CustomerID': ['C1', 'C2', 'C3'],
        'ProductID': ['P1', 'P2', 'P3'],
        'ProductInfo': ['Router X500', 'Database Server Y200', 'Network Switch Z100'],
        'SolutionDetails': [
            'Reset network adapter and update driver',
            'Restore TCP/IP stack and restart service',
            'Configure IP settings manually and update firmware'
        ],
        'Status': ['Resolved', 'Resolved', 'In Progress'],
        'Tags': ['network,crash', 'database,connectivity', 'network,configuration'],
        'Timestamp': ['2023-01-15', '2023-02-20', '2023-03-10'],
        'DocID': ['D1', 'D2', 'D3']
    }
    
    tech_src_data = {
        'DocID': ['TD1', 'TD2', 'TD3'],
        'ProductID': ['P1', 'P2', 'P3'],
        'step_description': [
            'Access network settings via control panel',
            'Navigate to database connection settings',
            'Configure IPv4 settings manually'
        ]
    }
    
    tech_meta_data = {
        'DocID': ['TD1', 'TD2', 'TD3'],
        'ProductID': ['P1', 'P2', 'P3'],
        'ProductInformation': ['Router X500 Technical Guide', 'Database Server Y200 Manual', 'Network Switch Z100 Configuration'],
        'SolutionSteps': [
            '1. Open Settings 2. Access Network 3. Select Adapter 4. Update Driver',
            '1. Open Services 2. Restart Database Service 3. Test Connection',
            '1. Access IP Config 2. Set Static IP 3. Apply Changes'
        ],
        'TechnicalTags': ['network,configuration', 'database,service', 'network,ip'],
        'DocumentType': ['Guide', 'Manual', 'Configuration']
    }
    
    return {
        "incident_src": pd.DataFrame(incident_src_data),
        "incident_meta": pd.DataFrame(incident_meta_data),
        "tech_src": pd.DataFrame(tech_src_data),
        "tech_meta": pd.DataFrame(tech_meta_data)
    }

@pytest.fixture
def mock_documents():
    incident_doc1 = MagicMock()
    incident_doc1.page_content = "System crashes when starting the network application"
    incident_doc1.metadata = {
        "TicketID": "T1", "CustomerID": "C1", "ProductID": "P1",
        "ProductInfo": "Router X500", "SolutionDetails": "Reset network adapter and update driver",
        "Status": "Resolved", "Tags": "network,crash", "Timestamp": "2023-01-15", "DocID": "D1"
    }
    
    incident_doc2 = MagicMock()
    incident_doc2.page_content = "Cannot connect to database after Windows update"
    incident_doc2.metadata = {
        "TicketID": "T2", "CustomerID": "C2", "ProductID": "P2",
        "ProductInfo": "Database Server Y200", "SolutionDetails": "Restore TCP/IP stack and restart service",
        "Status": "Resolved", "Tags": "database,connectivity", "Timestamp": "2023-02-20", "DocID": "D2"
    }
    
    tech_doc1 = MagicMock()
    tech_doc1.page_content = "Access network settings via control panel"
    tech_doc1.metadata = {
        "DocID": "TD1", "ProductID": "P1", "ProductInformation": "Router X500 Technical Guide",
        "SolutionSteps": "1. Open Settings 2. Access Network 3. Select Adapter 4. Update Driver",
        "TechnicalTags": "network,configuration", "DocumentType": "Guide"
    }
    
    tech_doc2 = MagicMock()
    tech_doc2.page_content = "Navigate to database connection settings"
    tech_doc2.metadata = {
        "DocID": "TD2", "ProductID": "P2", "ProductInformation": "Database Server Y200 Manual",
        "SolutionSteps": "1. Open Services 2. Restart Database Service 3. Test Connection",
        "TechnicalTags": "database,service", "DocumentType": "Manual"
    }
    
    return {
        "incident": [incident_doc1, incident_doc2],
        "tech": [tech_doc1, tech_doc2]
    }

@pytest.fixture
def mock_document_loader():
    with patch('src.document_loader.Guide_DocumentLoader') as mock_loader_class:
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        
        incident_docs = [
            MagicMock(page_content="System crashes when starting the network application", 
                     metadata={"ProductID": "P1", "SolutionDetails": "Reset network adapter and update driver"}),
            MagicMock(page_content="Cannot connect to database after Windows update",
                     metadata={"ProductID": "P2", "SolutionDetails": "Restore TCP/IP stack and restart service"})
        ]
        
        tech_docs = [
            MagicMock(page_content="Access network settings via control panel",
                     metadata={"ProductID": "P1", "SolutionSteps": "1. Open Settings 2. Access Network 3. Select Adapter 4. Update Driver"}),
            MagicMock(page_content="Navigate to database connection settings",
                     metadata={"ProductID": "P2", "SolutionSteps": "1. Open Services 2. Restart Database Service 3. Test Connection"})
        ]
        
        mock_loader.load_incident_documents.return_value = incident_docs
        mock_loader.load_tech_documents.return_value = tech_docs
        mock_loader.load_all_documents.return_value = (incident_docs, tech_docs)
        
        yield mock_loader_class, mock_loader, incident_docs, tech_docs

@pytest.fixture
def mock_vector_store():
    with patch('src.vector_store.VectorStoreManager') as mock_store_class:
        mock_store = MagicMock()
        mock_store_class.return_value = mock_store
        
        def mock_retrieve(query, product_id, store_type):
            if query == "" or len(query) < 15:
                raise ValueError("Query must be a non-empty string")
            if not isinstance(product_id, str):
                raise ValueError("Product ID must be a string")
            if store_type not in ["incident", "tech"]:
                raise ValueError("Store type must be 'incident' or 'tech'")
                
            if product_id == "P1":
                if store_type == "incident":
                    return [MagicMock(page_content="System crashes when starting the network application", 
                                     metadata={"ProductID": "P1", "SolutionDetails": "Reset network adapter and update driver"})]
                else:
                    return [MagicMock(page_content="Access network settings via control panel",
                                     metadata={"ProductID": "P1", "SolutionSteps": "1. Open Settings 2. Access Network 3. Select Adapter 4. Update Driver"})]
            elif product_id == "P2":
                if store_type == "incident":
                    return [MagicMock(page_content="Cannot connect to database after Windows update",
                                     metadata={"ProductID": "P2", "SolutionDetails": "Restore TCP/IP stack and restart service"})]
                else:
                    return [MagicMock(page_content="Navigate to database connection settings",
                                     metadata={"ProductID": "P2", "SolutionSteps": "1. Open Services 2. Restart Database Service 3. Test Connection"})]
            else:
                return "Contact customer support."
                
        mock_store.retrieve_documents.side_effect = mock_retrieve
        
        yield mock_store_class, mock_store

@pytest.fixture
def mock_rag_chain():
    with patch('src.rag_chain.RAGChain') as mock_chain_class:
        mock_chain = MagicMock()
        mock_chain_class.return_value = mock_chain
        
        def mock_run(query, tech_results, incident_results):
            if not isinstance(query, str):
                raise ValueError("Query must be a string")
                
            response = "Based on the technical documentation and incident records:\n\n"
            
            if tech_results and "network" in query.lower():
                response += "Technical Solution Steps:\n"
                response += "1. Open Settings\n2. Access Network\n3. Select Adapter\n4. Update Driver\n\n"
                
            if incident_results and "network" in query.lower():
                response += "From Similar Incidents:\n"
                response += "Reset network adapter and update driver\n"
                
            if not tech_results and not incident_results:
                response = "Please enter a valid query with relevant context."
                
            return response
            
        mock_chain.run.side_effect = mock_run
        
        yield mock_chain_class, mock_chain

@pytest.fixture
def sample_queries():
    return {
        "network_query": "How do I fix network connectivity issues with my Router X500?",
        "database_query": "What should I do when I can't connect to the database after a Windows update?",
        "unrelated_query": "How do I format my hard drive?",
        "empty_query": "",
        "short_query": "Help me"
    }