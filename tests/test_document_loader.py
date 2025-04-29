import os
import pandas as pd
import pytest
from unittest.mock import patch, mock_open, MagicMock
from src.document_loader import Guide_DocumentLoader

class TestGuideDocumentLoader:
    """Test suite for Guide_DocumentLoader class"""

    def test_init_file_not_found(self):
        """Test initialization with non-existent file path raises FileNotFoundError"""
        with patch('os.path.exists', return_value=False):
            with pytest.raises(FileNotFoundError) as excinfo:
                Guide_DocumentLoader()
            assert "Required file not found" in str(excinfo.value)

    def test_init_success(self):
        """Test successful initialization"""
        with patch('os.path.exists', return_value=True):
            loader = Guide_DocumentLoader()
            assert loader.tech_src_path == "data/src_tech_records.csv"
            assert loader.incident_src_path == "data/src_incident_records.csv"
            assert loader.tech_content_col == "step_description"
            assert "ProductID" in loader.tech_metadata_cols

    @patch('os.path.exists', return_value=True)
    @patch('pandas.read_csv')
    @patch('os.makedirs')
    @patch('pandas.DataFrame.to_csv')
    @patch('langchain_community.document_loaders.csv_loader.CSVLoader.load')
    def test_load_incident_documents_row_mismatch(self, mock_load, mock_to_csv, mock_makedirs, 
                                                 mock_read_csv, mock_exists):
        """Test row count mismatch between source and metadata raises ValueError"""
        # Mock the read_csv to return DataFrames with different row counts
        incident_src = pd.DataFrame({'ProblemDescription': ['desc1', 'desc2']})
        incident_meta = pd.DataFrame({'DocID': [1]}) # Only one row
        
        mock_read_csv.side_effect = [incident_src, incident_meta]
        
        loader = Guide_DocumentLoader()
        
        with pytest.raises(ValueError) as excinfo:
            loader.load_incident_documents()
        assert "Row count mismatch" in str(excinfo.value)

    @patch('os.path.exists', return_value=True)
    @patch('pandas.read_csv')
    @patch('os.makedirs')
    @patch('pandas.DataFrame.to_csv')
    @patch('langchain_community.document_loaders.csv_loader.CSVLoader.load')
    def test_load_incident_documents_success(self, mock_load, mock_to_csv, mock_makedirs, 
                                            mock_read_csv, mock_exists):
        """Test successful loading of incident documents"""
        # Mock the necessary components for successful execution
        incident_src = pd.DataFrame({'ProblemDescription': ['desc1']})
        incident_meta = pd.DataFrame({
            'CustomerID': [1],
            'ProductID': [101],
            'ProductInfo': ['Info'],
            'SolutionDetails': ['Solution'],
            'Status': ['Resolved'],
            'Tags': ['tag1'],
            'Timestamp': ['2023-01-01'],
            'DocID': [1]
        })
        
        mock_read_csv.side_effect = [incident_src, incident_meta]
        mock_load.return_value = ["document1"]
        
        loader = Guide_DocumentLoader()
        result = loader.load_incident_documents()
        
        assert result == ["document1"]
        mock_to_csv.assert_called_once()
        mock_load.assert_called_once()

    @patch('os.path.exists', return_value=True)
    @patch('pandas.read_csv')
    @patch('os.makedirs')
    @patch('pandas.DataFrame.to_csv')
    @patch('langchain_community.document_loaders.csv_loader.CSVLoader.load')
    def test_load_tech_documents_success(self, mock_load, mock_to_csv, mock_makedirs, 
                                        mock_read_csv, mock_exists):
        """Test successful loading of technical documents"""
        # Mock the necessary components for successful execution
        tech_src = pd.DataFrame({'step_description': ['step1']})
        tech_meta = pd.DataFrame({
            'ProductID': [101],
            'ProductInformation': ['Info'],
            'SolutionSteps': ['Steps'],
            'TechnicalTags': ['tag1'],
            'DocumentType': ['Manual']
        })
        
        mock_read_csv.side_effect = [tech_src, tech_meta]
        mock_load.return_value = ["tech_document1"]
        
        loader = Guide_DocumentLoader()
        result = loader.load_tech_documents()
        
        assert result == ["tech_document1"]
        mock_to_csv.assert_called_once()
        mock_load.assert_called_once()

    @patch('os.path.exists', return_value=True)
    @patch('os.remove')
    @patch('pandas.read_csv')
    @patch('os.makedirs')
    @patch('pandas.DataFrame.to_csv')
    @patch('langchain_community.document_loaders.csv_loader.CSVLoader.load')
    def test_cleanup_temp_files(self, mock_load, mock_to_csv, mock_makedirs, 
                              mock_read_csv, mock_remove, mock_exists):
        """Test that temporary files are cleaned up after processing"""
        # Setup for successful execution with cleanup
        tech_src = pd.DataFrame({'step_description': ['step1']})
        tech_meta = pd.DataFrame({
            'ProductID': [101],
            'ProductInformation': ['Info'],
            'SolutionSteps': ['Steps'],
            'TechnicalTags': ['tag1'],
            'DocumentType': ['Manual']
        })
        
        mock_read_csv.side_effect = [tech_src, tech_meta]
        mock_load.return_value = ["tech_document1"]
        
        # Mock that temp file exists for cleanup
        mock_exists.side_effect = [True, True, True, True, True]  # Initial checks + temp file exists check
        
        loader = Guide_DocumentLoader()
        loader.load_tech_documents()
        
        # Verify temp file was removed
        mock_remove.assert_called_once_with("data/filtered_temp_tech_doc.csv")

    @patch('os.path.exists', return_value=True)
    @patch.object(Guide_DocumentLoader, 'load_incident_documents')
    @patch.object(Guide_DocumentLoader, 'load_tech_documents')
    def test_load_all_documents(self, mock_tech, mock_incident, mock_exists):
        """Test that load_all_documents calls both individual loaders and returns their results"""
        mock_incident.return_value = ["incident1", "incident2"]
        mock_tech.return_value = ["tech1", "tech2"]
        
        loader = Guide_DocumentLoader()
        incident_result, tech_result = loader.load_all_documents()
        
        assert incident_result == ["incident1", "incident2"]
        assert tech_result == ["tech1", "tech2"]
        mock_incident.assert_called_once()
        mock_tech.assert_called_once()

    @patch('os.path.exists', return_value=True)
    @patch.object(Guide_DocumentLoader, 'load_incident_documents')
    @patch.object(Guide_DocumentLoader, 'load_tech_documents')
    def test_load_all_documents_propagates_errors(self, mock_tech, mock_incident, mock_exists):
        """Test that errors from individual loaders are propagated"""
        mock_incident.side_effect = ValueError("Test error")
        
        loader = Guide_DocumentLoader()
        
        with pytest.raises(ValueError) as excinfo:
            loader.load_all_documents()
        
        assert "Test error" in str(excinfo.value)
        mock_incident.assert_called_once()
        # Tech documents should not be loaded since incident loading failed
        mock_tech.assert_not_called()