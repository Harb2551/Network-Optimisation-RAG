# Network Optimization Assistant

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system specifically designed for technical support applications. The system helps support agents quickly find relevant solutions to customer inquiries by retrieving and synthesizing information from two types of knowledge sources:

1. **Incident Records**: Historical customer support tickets containing problem descriptions and their solutions
2. **Technical Documentation**: Product-specific technical guides and step-by-step procedures

The system uses a hybrid retrieval approach combining dense (FAISS) and sparse (BM25) vector search to maximize relevant document retrieval, followed by a generation step that creates coherent, actionable solution guides.

## Key Features

- **Dual Document Types**: Specialized handling for both incident records and technical documentation
- **Hybrid Retrieval**: Combines semantic search (FAISS) with keyword matching (BM25) for robust document retrieval
- **Product-Specific Filtering**: Automatically filters results by product ID to ensure relevance
- **Structured Solution Generation**: Formats responses as clear, step-by-step solution guides
- **Automatic Source Attribution**: Clearly labels which information comes from technical documentation vs. incident records
- **Comprehensive Testing**: Includes unit tests and integration tests for all components

## System Components

### 1. Document Loader (`src/document_loader.py`)

The `Guide_DocumentLoader` class handles:
- Loading and processing CSV files containing incident records and technical documentation
- Merging source content with metadata
- Converting raw data into structured document objects suitable for vector embedding
- Maintaining separation between content and metadata while ensuring both are accessible

### 2. Vector Store Manager (`src/vector_store.py`)

The `VectorStoreManager` class provides:
- Creation and management of FAISS vector indexes for semantic search
- Implementation of BM25 indexes for keyword-based search
- Ensemble retrieval combining both approaches with adjustable weighting
- Document filtering by product ID to ensure results match the product in question
- Persistent storage and loading of vector stores for efficiency

### 3. RAG Chain (`src/rag_chain.py`)

The `RAGChain` class implements:
- A specialized prompt template that prioritizes solution steps based on availability and relevance
- A language model chain using gpt-4o for high-quality response generation
- Logic for combining retrieved context with user queries to produce coherent solutions
- Fallback mechanisms when no suitable documents are found

### 4. Web Application (`app.py`)

The Streamlit-based interface provides:
- User-friendly input for technical questions
- Product selection dropdown
- Real-time status updates during processing
- Formatted display of solution guides
- Download option for generated solutions

## Setup and Installation

### Prerequisites

- Python 3.10+
- OpenAI API key

### Installation

1. Clone the repository
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Set up your OpenAI API key in a `.env` file:
   ```
   OPENAI_API_KEY="your-api-key-here"
   ```
4. Ensure the data directory contains the required CSV files:
   - `src_tech_records.csv`
   - `metadata_tech_records.csv`
   - `src_incident_records.csv`
   - `metadata_incident_records.csv`

### Running the Application

Start the web application:
```
streamlit run app.py
```

## Data Format

### Incident Records CSV Structure

#### Source CSV Columns:
- `ProblemDescription`: Detailed description of the customer's issue

#### Metadata CSV Columns:
- `CustomerID`: Unique customer identifier
- `ProductID`: Product identifier for filtering
- `ProductInfo`: Additional product information
- `SolutionDetails`: Description of how the issue was resolved
- `Status`: Status of the incident (e.g., resolved, pending)
- `Tags`: Categorization tags
- `Timestamp`: When the incident was recorded
- `DocID`: Unique document identifier

### Technical Documentation CSV Structure

#### Source CSV Columns:
- `step_description`: Detailed step in a technical procedure

#### Metadata CSV Columns:
- `ProductID`: Product identifier for filtering
- `ProductInformation`: Additional product details
- `SolutionSteps`: Step-by-step instructions
- `TechnicalTags`: Categorization tags
- `DocumentType`: Type of technical document

## Testing

The project includes comprehensive tests for all components:

### Running Tests

```
pytest tests/
```

### Test Files

- `conftest.py`: Common test fixtures and mocks
- `test_document_loader.py`: Tests for the document loader component
- `test_vector_store.py`: Tests for vector store creation and retrieval
- `test_rag_chain.py`: Tests for the RAG chain functionality
- `test_integration.py`: Integration tests for the full workflow

### Sample Test Queries

The `test_queries.txt` file contains sample queries for testing the system with different product IDs and scenarios. These queries cover various technical issues such as:

- Network connectivity problems
- Authentication failures
- Broadcast storms
- VPN certificate issues
- DNS resolution problems
- Power-related equipment failures
- Wireless network outages
- Load balancer issues
- Fiber optic link degradation
- Voice call quality problems

## Customization

### Adjusting Retrieval Parameters

You can customize the retrieval process by adjusting:

1. **Top-k value**: Number of documents to retrieve (default: 5)
2. **Ensemble weights**: Relative importance of FAISS vs. BM25 (default: [0.8, 0.2])

### Changing the Language Model

You can use different OpenAI models by modifying the model name parameter when initializing the RAGChain.

### Modifying the Prompt Template

The system's response style and prioritization logic can be customized by editing the prompt template in `rag_chain.py`.

## Error Handling

The system includes comprehensive error handling:

- File validation to ensure all required data files exist
- Data validation to check for proper structure and matching row counts
- Graceful fallbacks when no relevant documents are found
- Automatic cleanup of temporary files