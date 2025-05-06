# PostgreSQL RAG Implementation

This folder contains implementations of various Retrieval-Augmented Generation (RAG) approaches using PostgreSQL as the database backend.

## Overview

PostgreSQL is a powerful open-source relational database with strong support for structured data and complex queries. This implementation demonstrates how traditional relational databases can be adapted for effective RAG systems through careful schema design and embedding integration.

## Folder Structure

```
Postgresql_Tasks/
├── data_manager.py                     # PostgreSQL connection and FAISS index management
├── Postgres_Naive_Rag.py               # Baseline vector similarity implementation
├── Postgres_Knowledge.py               # Knowledge Graph enhanced RAG implementation
├── Postgres_Graph_rag.py               # Full graph-enhanced RAG with NetworkX
├── Postgres_LangChain_Naive_Rag.py     # LangChain integration for naive RAG
├── Postgres_LangChain_Knowledge_Rag.py # LangChain integration with metadata
├── Postgres_LangGraph_Rag.py           # LangGraph workflow implementation
├── Post_main.py                        # Main execution script for all implementations
└── Postgresql_Readme.md                # This documentation file
```

### Associated Files

```
project_root/
├── postgres_graph_visualization.png    # Visualization of PostgreSQL knowledge graph
└── faiss_db/                           # Vector indices for retrieval
    ├── index.faiss                     # Main FAISS index
    ├── index.pkl                       # Index metadata
    ├── text_chunks.npy                 # Stored text chunks
    └── k_graph_embeddings.index        # Knowledge graph embeddings
```

## Methodology

### Data Preparation
1. **Schema Design**: Creating a relational schema with articles, authors, categories, and relationship tables
2. **Constraint Management**: Setting up primary and foreign key constraints for data integrity
3. **Data Import**: Loading from CSV files with data cleaning and transformation
4. **Embedding Generation and Storage**: Creating and storing vector embeddings in a dedicated PostgreSQL table
5. **Graph Representation**: Building an in-memory NetworkX graph from relational data for graph operations

### RAG Methodologies

#### Naive RAG Implementation
- **Vector Similarity**: Using FAISS for efficient similarity search between query and document embeddings
- **Baseline Retrieval**: Implementing straightforward retrieval without any graph enhancement
- **Performance Measurement**: Tracking latency and calculating ROUGE metrics against ground truth answers
- **Pure Embedding Focus**: Deliberately avoiding relationship data to establish a baseline

#### Knowledge Graph RAG Implementation
- **Hybrid Scoring System**: Combining three components with weighted importance:
  - Base similarity (50%): Embedding-based cosine similarity
  - Metadata relevance (30%): Matching query terms with tags, categories, and authors
  - Graph relationship strength (20%): Evaluating node connections in the graph
- **Metadata Enrichment**: Boosting scores when metadata matches query terms
- **Cross-Table Relationship Analysis**: Leveraging the relational schema to extract graph context
- **SQL-Powered Context Building**: Using SQL joins to gather relationship information

#### Full Graph RAG Implementation
- **Complete Graph Exploration**: Traversing the NetworkX graph to discover connected content
- **Relationship Type Weighting**: Assigning importance based on different relationship types
- **Connection-Based Boosting**: Applying up to 50% score enhancement based on number of connections
- **Result Enhancement**: Combining base similarity with graph-derived boosting factors

#### LangChain Integration
- **HuggingFaceEmbeddings Integration**: Using LangChain's embedding interfaces with local models
- **Component-Based Approach**: Structuring the retrieval process using LangChain abstractions
- **Document Processing**: Converting SQL results to LangChain document format with metadata
- **Custom Similarity Calculation**: Implementing PostgreSQL-specific similarity functions

#### LangGraph Implementation
- **Structured Workflow**: Building a multi-step retrieval workflow:
  1. Initial embedding-based retrieval
  2. Graph enhancement of candidate documents
  3. Final scoring and result formatting
- **Node Relationship Analysis**: Evaluating connection patterns for result enrichment
- **Two-Stage Retrieval**: First casting a wider net, then narrowing based on graph relationships
- **Similarity and Graph Balancing**: Dynamically adjusting the influence of graph connections on final scores

## Components

### Data Management

- **PostgresDataManager (data_manager.py)**: Core component that manages database operations:
  - Handles connection and cursor management
  - Creates tables and constraints for articles, authors, categories, and relationships
  - Imports data from CSV files
  - Generates and stores embeddings
  - Implements FAISS vector index for similarity search
  - Provides utilities for graph context retrieval

### RAG Implementations

1. **Basic RAG Approaches**:
   - **PostgresNaiveRAG**: Simple vector similarity-based document retrieval
   - **PostgresKnowledgeRAG**: Enhanced RAG with metadata and graph relationship information
   - **PostgresGraphRAG**: Comprehensive graph-enhanced retrieval with NetworkX for graph operations

2. **LangChain Integrations**:
   - **PostgresLangChainNaiveRAG**: Uses LangChain's HuggingFaceEmbeddings for basic RAG functionality
   - **PostgresLangChainKnowledgeRAG**: LangChain-based implementation with knowledge graph enhancement

3. **LangGraph Implementation**:
   - **PostgresLangGraphRAG**: Advanced implementation using LangGraph for structured workflow, with separate steps for embedding retrieval and graph enhancement

## Execution

The system can be run through:
- **Post_main.py**: Orchestrates all RAG approaches, runs test queries, and compares performance

## Performance

Based on the evaluation results in `results/DataBases_Results.txt`, PostgreSQL shows:

- **Latency Performance**: Generally competitive with other database implementations, though with some overhead for relational operations
- **Accuracy**: Good precision and recall metrics that demonstrate effective retrieval
- **Graph Enhancement**: Effective use of relationship data despite PostgreSQL not being a native graph database

## Graph Representation

Unlike native graph databases, PostgreSQL implements graph functionality through:
- Relationship tables with explicit foreign keys
- NetworkX graph construction for in-memory graph operations
- Dedicated query patterns for traversing relationships

## Visualization

The implementation includes graph visualization capabilities:
- `postgres_graph_visualization.png`: Visual representation of the knowledge graph constructed from PostgreSQL data

## Usage

To use this implementation:
1. Ensure PostgreSQL is running with a database called "Rag DB" (default credentials in code)
2. Run using `python Post_main.py`
3. The system processes predefined test queries and outputs comparative metrics