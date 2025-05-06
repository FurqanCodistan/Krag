# FalkorDB RAG Implementation

This folder contains the implementation of various Retrieval-Augmented Generation (RAG) approaches using FalkorDB as the database backend.

## Overview

FalkorDB is a Redis-compatible graph database that provides graph capabilities alongside the performance benefits of Redis. This implementation demonstrates how FalkorDB can be used to build effective RAG systems.

## Folder Structure

```
FalkorDb_Tasks/
├── data_manager.py                   # Database operations and FAISS index management
├── Falkor_Naive_Rag.py               # Baseline vector similarity implementation
├── Falkor_Knowledge.py               # Knowledge Graph enhanced RAG implementation
├── Falkor_Graph_rag.py               # Full graph-enhanced RAG with relationship scoring
├── Falkor_LangChain_Naive_Rag.py     # LangChain integration for naive RAG
├── Falkor_LangChain_Knowledge_Rag.py # LangChain integration for knowledge graph RAG
├── Falkor_LangGraph_Rag.py           # LangGraph workflow implementation
├── Falkor_LangChain_Main.py          # Entry point for LangChain implementations
├── FK_Main.py                        # Entry point for basic implementations
├── main.py                           # Comprehensive execution of all approaches
└── Falkor_Readme.md                  # This documentation file
```

### Associated Files

```
project_root/
├── falkor_graph_visualization.png    # Visualization of the FalkorDB knowledge graph
├── falkor_langgraph_visualization.png # Visualization of the LangGraph workflow
└── faiss_db/                         # Vector indices for retrieval
    ├── falkor_langchain_knowledge/   # Indices for LangChain Knowledge implementation
    │   ├── index.faiss
    │   └── index.pkl
    ├── falkor_langchain_naive/       # Indices for LangChain Naive implementation
    │   ├── index.faiss
    │   └── index.pkl
    └── falkor_langgraph/             # Indices for LangGraph implementation
        ├── index.faiss
        └── index.pkl
```

## Methodology

### Data Preparation
1. **Data Import**: Source data from CSV files is imported into FalkorDB as graph nodes and relationships
2. **Embedding Generation**: Text content is processed using Sentence Transformers (all-MiniLM-L6-v2) to create vector embeddings
3. **Graph Construction**: A knowledge graph is built with articles as primary nodes, connected to metadata nodes (authors, categories, tags) and other articles through typed relationships

### RAG Methodologies

#### Naive RAG Implementation
- **Vector similarity search**: Uses cosine similarity between query embedding and document embeddings
- **Relevance Scoring**: Pure vector similarity without graph context
- **Result Ranking**: Results sorted by similarity score
- **Evaluation**: Uses ROUGE metrics against ground truth answers

#### Knowledge RAG Implementation
- **Hybrid Retrieval**: Combines vector similarity with graph-based knowledge
- **Scoring Components**:
  - Base similarity score (50% weight)
  - Metadata relevance score (30% weight)
  - Graph relationship score (20% weight)
- **Metadata Scoring**: Boosts relevance when query terms match tags (0.1 per match), categories (0.15 per match), or authors (0.05 per match)
- **Graph Scoring**: Analyzes graph connections and weights relationships based on type and strength

#### Full Graph RAG Implementation
- **Advanced Graph Traversal**: Explores the knowledge graph to find semantically connected content
- **Relationship Boost**: Applies weighted boosts based on relationship types:
  - Citations: 0.3 boost
  - Extensions: 0.4 boost
  - Contradictions: 0.2 boost
  - Other relationships: 0.1 boost
- **Path Analysis**: Considers multi-hop connections in the graph
- **Result Enhancement**: Combines vector similarity with graph-based relevance signals

#### LangChain Integration
- **Component-Based Architecture**: Uses LangChain's modular components
- **Embedding Enhancement**: Integrates with advanced embedding models
- **Retrieval Chain**: Implements structured retrieval process with context window management

#### LangGraph Implementation
- **State-Based Workflow**: Uses LangGraph's StateGraph for multi-step retrieval
- **Workflow Steps**:
  1. Start search (find entry points)
  2. Explore nodes (traverse graph connections)
  3. End search (finalize and rank results)
- **Decision Logic**: Implements routing conditions to determine when to continue exploration vs. complete search
- **Dynamic Scoring**: Updates relevance scores during graph traversal

## Components

### Data Management

- **FalkorDataManager (data_manager.py)**: Handles database connection, initialization, and provides utility methods for data retrieval and embedding operations.
  - Creates and manages graph structures in FalkorDB
  - Handles embeddings generation and storage
  - Provides similarity search functionality

### RAG Implementations

1. **Basic RAG Approaches**:
   - **FalkorNaiveRAG**: Simple vector similarity-based retrieval without graph enhancements
   - **FalkorKnowledgeRAG**: Enhances retrieval with knowledge graph context
   - **FalkorGraphRAG**: Full graph-enhanced RAG with relationship-based scoring boosts

2. **LangChain Integrations**:
   - **FalkorLangChainNaiveRAG**: Implements naive RAG using LangChain's components
   - **FalkorLangChainKnowledgeRAG**: Knowledge Graph RAG with LangChain's embeddings and components

3. **LangGraph Implementation**:
   - **FalkorLangGraphRAG**: Advanced implementation using LangGraph for workflow management, enabling complex retrieval strategies with multi-step graph traversal

## Execution

The system can be run through multiple entry points:
- **FK_Main.py**: Runs the basic RAG implementations
- **Falkor_LangChain_Main.py**: Runs LangChain-based RAG implementations
- **main.py**: Comprehensive execution of all RAG approaches for comparison

## Performance

Based on the results in `results/DataBases_Results.txt`, FalkorDB shows strong performance characteristics:

- **Retrieval Quality**: Good precision and recall metrics compared to other database implementations
- **Latency**: Low latency for basic operations, with higher but acceptable latency for LangGraph operations
- **Graph Boost Effect**: Relationship-based scoring provides meaningful improvements to retrieval relevance

## Visualization

The system includes graph visualization capabilities, generating PNG files to represent the knowledge graph structure:
- `falkor_graph_visualization.png`: Basic graph structure
- `falkor_langgraph_visualization.png`: LangGraph workflow visualization

## Usage

To use this implementation:
1. Ensure FalkorDB is running locally (default: localhost:6379)
2. Run one of the entry point scripts (e.g., `python main.py`)
3. The system will process predefined test queries and output comparison metrics