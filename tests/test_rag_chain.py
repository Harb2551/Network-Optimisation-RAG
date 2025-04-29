import pytest
from unittest.mock import patch, MagicMock, Mock
import sys
import os

# Add the parent directory to sys.path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the RAGChain class
from src.rag_chain import RAGChain

class TestRAGChain:
    """Test suite for RAGChain class"""
    
    @pytest.fixture
    def setup_mocks(self):
        """Setup mocks for testing"""
        with patch('src.rag_chain.ChatPromptTemplate') as mock_prompt_template, \
             patch('src.rag_chain.ChatOpenAI') as mock_chat_openai, \
             patch('src.rag_chain.StrOutputParser') as mock_output_parser, \
             patch('src.rag_chain.RunnablePassthrough') as mock_runnable:
            
            # Set up ChatPromptTemplate mock
            mock_prompt = MagicMock()
            mock_prompt_template.from_template.return_value = mock_prompt
            
            # Set up ChatOpenAI mock
            mock_llm = MagicMock()
            mock_chat_openai.return_value = mock_llm
            
            # Set up StrOutputParser mock
            mock_parser = MagicMock()
            mock_output_parser.return_value = mock_parser
            
            # Set up RunnablePassthrough mock
            mock_assign = MagicMock()
            mock_runnable.assign.return_value = mock_assign
            
            # Set up chain mock
            mock_chain = MagicMock()
            # Configure the pipeline behavior
            mock_assign.__or__.return_value = mock_assign
            mock_assign.__or__.return_value.__or__.return_value = mock_assign
            mock_assign.__or__.return_value.__or__.return_value.__or__.return_value = mock_chain
            mock_chain.invoke.return_value = "Mocked response with solution steps"
            
            yield {
                "prompt_template": mock_prompt_template,
                "prompt": mock_prompt,
                "chat_openai": mock_chat_openai,
                "llm": mock_llm,
                "output_parser": mock_output_parser,
                "parser": mock_parser,
                "runnable": mock_runnable,
                "assign": mock_assign,
                "chain": mock_chain
            }
    
    def test_init_success(self, setup_mocks):
        """Test successful initialization of RAGChain"""
        # Create a RAGChain instance
        rag_chain = RAGChain()
        
        # Verify correct initialization
        setup_mocks["prompt_template"].from_template.assert_called_once()
        setup_mocks["chat_openai"].assert_called_once_with(model_name="gpt-4o", temperature=0)
        
        # Verify template name is set correctly
        assert rag_chain.template_name == "tech_incident_template"
        
        # Verify the chain is configured
        assert rag_chain.chain is not None
    
    def test_init_custom_parameters(self, setup_mocks):
        """Test initialization with custom parameters"""
        # Create a RAGChain instance with custom parameters
        rag_chain = RAGChain(template_name="custom_template", model_name="gpt-3.5-turbo")
        
        # Verify template name is set correctly
        assert rag_chain.template_name == "custom_template"
        
        # Verify model name is passed correctly
        setup_mocks["chat_openai"].assert_called_once_with(model_name="gpt-3.5-turbo", temperature=0)
    
    def test_init_error_handling(self, setup_mocks):
        """Test error handling during initialization"""
        # Configure the mock to raise an exception
        setup_mocks["prompt_template"].from_template.side_effect = Exception("Template error")
        
        # Verify that the exception is propagated
        with pytest.raises(Exception) as excinfo:
            RAGChain()
        
        assert "Template error" in str(excinfo.value)
    
    def test_run_success(self, setup_mocks):
        """Test successful execution of run method"""
        # Create a RAGChain instance and replace its chain with our mock
        rag_chain = RAGChain()
        rag_chain.chain = setup_mocks["chain"]
        
        # Test data
        query = "How do I configure network settings?"
        tech_results = "1. Open control panel 2. Navigate to Network settings 3. Update configuration"
        incident_results = "Previous users fixed this by updating network drivers"
        
        # Call run method
        result = rag_chain.run(query, tech_results, incident_results)
        
        # Verify chain.invoke was called with correct parameters
        expected_input = {
            "query": query,
            "tech_results": tech_results,
            "incident_results": incident_results
        }
        setup_mocks["chain"].invoke.assert_called_once_with(expected_input)
        
        # Verify result matches the mocked response
        assert result == "Mocked response with solution steps"
    
    def test_run_invalid_query(self, setup_mocks):
        """Test run method with invalid query type"""
        # Create a RAGChain instance
        rag_chain = RAGChain()
        rag_chain.chain = setup_mocks["chain"]
        
        # Call run method with invalid query type
        with pytest.raises(ValueError) as excinfo:
            rag_chain.run(123, "Tech results", "Incident results")
        
        assert "Query must be a string" in str(excinfo.value)
        
        # Verify chain.invoke was not called
        setup_mocks["chain"].invoke.assert_not_called()
    
    def test_run_empty_contexts(self, setup_mocks):
        """Test run method with empty contexts"""
        # Create a RAGChain instance and replace its chain with our mock
        rag_chain = RAGChain()
        rag_chain.chain = setup_mocks["chain"]
        
        # Test with empty contexts
        result = rag_chain.run("How do I configure network?", "", "")
        
        # Verify chain.invoke was called with correct parameters
        expected_input = {
            "query": "How do I configure network?",
            "tech_results": "",
            "incident_results": ""
        }
        setup_mocks["chain"].invoke.assert_called_once_with(expected_input)
        
        # Verify result matches the mocked response
        assert result == "Mocked response with solution steps"
    