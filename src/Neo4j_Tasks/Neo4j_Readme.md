# Neo4j RAG Implementation

This folder contains implementations of various Retrieval-Augmented Generation (RAG) approaches using Neo4j as the database backend.

## Overview

Neo4j is a native graph database designed for storing and querying highly connected data. This implementation leverages Neo4j's powerful graph capabilities to enhance RAG systems through relationship-aware retrieval strategies.

## Folder Structure

```
Neo4j_Tasks/
├── data_manager.py                  # Neo4j database operations and FAISS indexing
├── Neo_Naive_Rag.py                 # Baseline vector similarity implementation
├── Neo_Knowledge.py                 # Knowledge Graph enhanced RAG implementation
├── Neo_Graph_rag.py                 # Full graph-enhanced RAG with traversal
├── Neo_LangChain_Naive_Rag.py       # LangChain integration for naive RAG
├── Neo_LangChain_Knowledge_Rag.py   # LangChain integration for knowledge graph RAG
├── Neo_LangGraph_Rag.py             # LangGraph workflow implementation
├── Neo_Main.py                      # Main execution script for all implementations
├── main.py                          # Alternative entry point
└── Neo4j_Readme.md                  # This documentation file
```

### Associated Files

```
project_root/
├── graph_visualization.png          # Visualization of the Neo4j knowledge graph
├── lang_graph_vis_what_is_artificial_i.png  # LangGraph workflow for AI query
├── lang_graph_vis_what_is_image_forger.png  # LangGraph workflow for image forgery query
├── orig_graph_vis_what_is_artificial_i.png  # Original graph for AI query
├── orig_graph_vis_what_is_image_forger.png  # Original graph for image forgery query
└── faiss_db/                        # Vector indices shared across implementations
    ├── index.faiss                  # Main FAISS index
    ├── index.pkl                    # Index metadata
    ├── text_chunks.npy              # Stored text chunks
    └── k_graph_embeddings.index     # Knowledge graph embeddings
```

## Methodology

### Data Preparation
1. **Database Schema Creation**: Establishing constraints and indexes for Articles, Authors, Categories, and Tags
2. **Data Import**: Loading structured data from CSV files into Neo4j's labeled property graph model
3. **Embedding Generation**: Creating vector embeddings from text content using Sentence Transformers
4. **FAISS Index Setup**: Building an efficient vector index for similarity search operations

### RAG Methodologies

#### Naive RAG Implementation
- **Vector Search**: Retrieves documents using FAISS index for efficient similarity computation
- **Pure Similarity Ranking**: Ranks results based solely on embedding similarity scores
- **No Graph Context**: Deliberately avoids using graph relationships to establish a baseline
- **ROUGE Evaluation**: Calculates precision, recall, and F1 scores against ground truth

#### Knowledge Graph RAG Implementation
- **Hybrid Knowledge Retrieval**: Combines vector similarity with knowledge graph information
- **Graph-Enhanced Context**: Uses article relationships to enhance semantic understanding
- **Cypher Query Integration**: Leverages Neo4j's Cypher query language to traverse relationships efficiently
- **Weighted Scoring System**: Blends embedding similarity with graph-derived relevance signals

#### Full Graph RAG Implementation
- **Complete Graph Traversal**: Explores the knowledge graph's structure to enhance retrieval
- **Relationship Type Analysis**: Weights different relationship types (e.g., CITES, EXTENDS) differently
- **Connected Node Boosting**: Increases relevance scores based on meaningful connections
- **Graph Context Aggregation**: Collects and leverages connected content for enriched retrieval

#### LangChain Integration
- **Custom Retriever Implementation**: Creates Neo4j-specific retrievers for LangChain
- **OpenAI Embeddings**: Uses OpenAI's embedding models instead of local models
- **Document-Oriented Processing**: Structures Neo4j content as LangChain documents with metadata
- **Retrieval Pipelines**: Implements structured retrieval chains with Neo4j as the knowledge source

#### LangGraph Implementation
- **State Graph Architecture**: Implements a directed workflow graph for retrieval operations
- **Multi-Stage Processing**: Breaks retrieval into discrete steps:
  1. Initial retrieval based on vector similarity
  2. Graph-based enhancement of initial results
  3. Final scoring and ranking of enhanced results
- **State Management**: Tracks and updates retrieval state throughout the workflow
- **Conditional Routing**: Implements decision logic for graph traversal paths

## Components

### Data Management

- **DataManager (data_manager.py)**: Central component for Neo4j database operations:
  - Handles Neo4j connection and session management
  - Initializes database schema and constraints
  - Imports data from CSV files
  - Generates and manages embeddings
  - Sets up FAISS index for similarity search
  - Provides utility functions for graph traversal

### RAG Implementations

1. **Basic RAG Approaches**:
   - **NeoNaiveRAG**: Baseline vector similarity-based document retrieval
   - **NeoKnowledgeGraph**: Knowledge graph enhanced RAG with contextual information
   - **NeoGraphRAG**: Full graph-enhanced RAG with relationship-based scoring

2. **LangChain Integrations**:
   - **NeoLangChainNaiveRAG**: Implements basic RAG using LangChain's components
   - **NeoLangChainKnowledgeRAG**: Knowledge Graph RAG with OpenAI embeddings via LangChain

3. **LangGraph Implementation**:
   - **NeoLangGraphRAG**: Advanced graph workflow implementation using the LangGraph framework, constructing a StateGraph for multi-step retrieval

## Execution

The system can be run through:
- **Neo_Main.py**: Orchestrates all RAG approaches, runs queries, and compares performance metrics

## Performance

Based on the evaluation results in `results/DataBases_Results.txt`, Neo4j demonstrates:

- **Retrieval Accuracy**: Competitive precision and recall metrics, especially with graph-enhanced approaches
- **Processing Speed**: Generally good performance with longer processing times for graph-heavy operations
- **Quality Enhancement**: Graph context information notably improves the relevance of retrieved documents

## Graph Capabilities

Neo4j's strengths in this implementation include:
- Rich relationship modeling between articles, authors, categories and tags
- Efficient graph traversal for context enrichment
- Constraint management for data integrity
- Fast node and relationship retrieval

## Usage

To use this implementation:
1. Ensure Neo4j is running locally (default: bolt://localhost:7687)
2. Set environment variables for Neo4j credentials in a .env file or directly in code
3. Run using `python Neo_Main.py`
4. The system will process the predefined test queries and output comparison results