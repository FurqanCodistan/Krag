# RAG: Knowledge Graph RAG Implementations

This is a comprehensive evaluation framework for Retrieval-Augmented Generation (RAG) systems using different database backends and methodologies. The project implements and compares various RAG approaches across three database technologies: FalkorDB, Neo4j, and PostgreSQL.

## Project Overview

This project explores the implementation and performance of RAG systems with a focus on knowledge graph integration. The framework includes:

- Three database backends (FalkorDB, Neo4j, PostgreSQL)
- Multiple RAG methodologies (Naive, Knowledge Graph, Full Graph)
- LangChain and LangGraph integrations
- Comprehensive evaluation metrics

## Folder Structure

The project is organized with the following structure:

```
.
├── README.md                          # Main project documentation
├── requirements.txt                   # Python dependencies
├── docker-compose.yml                 # Database containerization setup
├── data/                              # Source data files
│   ├── articles.csv                   # Main article content
│   ├── authors.csv                    # Author information
│   ├── categories.csv                 # Category definitions
│   ├── tags.csv                       # Tag data
│   ├── article_authors.csv            # Article-author relationships
│   ├── article_categories.csv         # Article-category relationships
│   ├── article_tags.csv               # Article-tag relationships
│   ├── relationships.csv              # Article-article relationships
│   └── queries.csv                    # Test queries with ground truth
├── faiss_db/                          # Vector indices for retrieval
│   ├── index.faiss                    # Main FAISS index
│   ├── index.pkl                      # Index metadata
│   ├── text_chunks.npy                # Stored text chunks
│   └── various subdirectories         # Framework-specific indices
├── results/                           # Performance evaluation results
│   └── DataBases_Results.txt          # Comparison metrics across databases
├── src/                               # Source code directory
│   ├── FalkorDb_Tasks/                # FalkorDB implementations
│   │   ├── data_manager.py            # Database operations
│   │   ├── Falkor_Naive_Rag.py        # Naive RAG implementation
│   │   ├── Falkor_Knowledge.py        # Knowledge Graph RAG
│   │   ├── Falkor_Graph_rag.py        # Full Graph RAG
│   │   ├── Falkor_LangChain_*.py      # LangChain integrations
│   │   ├── Falkor_LangGraph_Rag.py    # LangGraph implementation
│   │   ├── Falkor_Readme.md           # FalkorDB documentation
│   │   └── main.py                    # Execution entry point
│   ├── Neo4j_Tasks/                   # Neo4j implementations
│   │   ├── data_manager.py            # Neo4j database operations
│   │   ├── Neo_Naive_Rag.py           # Naive RAG implementation
│   │   ├── Neo_Knowledge.py           # Knowledge Graph RAG
│   │   ├── Neo_Graph_rag.py           # Full Graph RAG
│   │   ├── Neo_LangChain_*.py         # LangChain integrations
│   │   ├── Neo_LangGraph_Rag.py       # LangGraph implementation
│   │   ├── Neo4j_Readme.md            # Neo4j documentation
│   │   └── Neo_Main.py                # Execution entry point
│   └── Postgresql_Tasks/              # PostgreSQL implementations
│       ├── data_manager.py            # PostgreSQL operations
│       ├── Postgres_Naive_Rag.py      # Naive RAG implementation
│       ├── Postgres_Knowledge.py      # Knowledge Graph RAG
│       ├── Postgres_Graph_rag.py      # Full Graph RAG
│       ├── Postgres_LangChain_*.py    # LangChain integrations
│       ├── Postgres_LangGraph_Rag.py  # LangGraph implementation
│       ├── Postgresql_Readme.md       # PostgreSQL documentation
│       └── Post_main.py               # Execution entry point
└── *.png                              # Generated graph visualizations
```

### Key Files

- **data_manager.py**: Database-specific adapters for data operations
- **Naive_RAG variants**: Baseline implementation using only vector similarity
- **Knowledge variants**: Intermediate implementation with metadata and basic graph features
- **Graph_rag variants**: Advanced implementation with full graph traversal
- **LangChain variants**: Integration with the LangChain framework
- **LangGraph variants**: Implementation using LangGraph's workflow management
- **Main scripts**: Entry points for running evaluations (main.py, Neo_Main.py, Post_main.py)
- **Readme files**: Documentation specific to each database implementation

## Methodology

The project follows a structured approach to implementing and evaluating RAG systems across different database technologies:

### Research Design

1. **Comparative Framework**: Implements identical RAG approaches across three database systems to isolate database effects from methodology effects
2. **Controlled Variables**: Uses the same embedding model, evaluation metrics, and test queries across all implementations
3. **Incremental Complexity**: Builds from simple (Naive RAG) to complex (Full Graph RAG) approaches to measure the impact of graph integration

### Data Processing Pipeline

1. **Data Extraction**: Loads structured article data from CSV files with metadata and relationship information
2. **Data Transformation**:
   - Creates embeddings for all article content using Sentence Transformers (all-MiniLM-L6-v2)
   - Builds graph structures appropriate for each database backend
   - Generates FAISS indices for efficient similarity search
3. **Data Loading**: Populates databases with both content and relationship information
4. **Index Optimization**: Applies appropriate indices to optimize query performance in each database

### RAG System Architecture

Each RAG implementation follows a consistent architecture across database backends:

1. **DataManager Layer**: Database-specific adapters that handle connection, queries, and data transformation
2. **RAG System Layer**: Implements different retrieval strategies:
   - **Naive RAG**: Baseline vector similarity search
   - **Knowledge RAG**: Hybrid approach combining vectors with metadata and simple graph features
   - **Graph RAG**: Comprehensive approach with full graph traversal and relationship analysis
3. **Framework Integration Layer**: Adapts the RAG systems to work with LangChain and LangGraph
4. **Evaluation Layer**: Consistent evaluation using ROUGE metrics against ground truth answers

### Evaluation Methodology

1. **Test Queries**: Uses a standard set of queries across all systems
2. **Ground Truth**: Compares results against predetermined ground truth answers
3. **Metrics Collection**:
   - **Quality Metrics**: ROUGE precision, recall, and F1 scores
   - **Performance Metrics**: Latency, total processing time
   - **System Metrics**: Memory usage, database-specific metrics
4. **Comparative Analysis**: Direct comparison of metrics across database implementations and RAG methodologies

### RAG Implementation Details

#### Naive RAG (Baseline)
- Simple vector search without graph context
- Pure embedding similarity scoring
- Results ranked solely by vector similarity

#### Knowledge Graph RAG (Intermediate)
- Weighted scoring system combining:
  - Vector similarity (50%)
  - Metadata relevance (30%)
  - Basic graph relationships (20%)
- Metadata matching with query terms
- First-degree relationship analysis

#### Full Graph RAG (Advanced)
- Complete graph traversal for context enrichment
- Relationship type analysis with weighted importance
- Connection strength evaluation
- Multi-hop relationship paths
- Dynamic score boosting based on graph context

#### LangChain Implementations
- Custom retrievers for each database backend
- Integration with LangChain's component architecture
- Chain-based retrieval pipelines

#### LangGraph Implementations
- State-based workflow graphs for retrieval
- Multi-stage processing with explicit state transitions
- Conditional routing based on intermediate results
- Enhanced explainability through workflow visualization

## Architecture

The project is organized into three main directories, each implementing the same RAG methodologies with different database backends:

- **FalkorDb_Tasks**: Implementations using FalkorDB (Redis-compatible graph database)
- **Neo4j_Tasks**: Implementations using Neo4j (native graph database)
- **Postgresql_Tasks**: Implementations using PostgreSQL (relational database)

## RAG Methodologies

Each database implementation includes the following RAG approaches:

1. **Naive RAG**:
   - Basic vector similarity search without graph context
   - Direct embedding comparison for document retrieval
   - Baseline for performance comparison

2. **Knowledge Graph RAG**:
   - Enhances retrieval with metadata and graph relationships
   - Adds context from connected nodes
   - Improves relevance through metadata scoring

3. **Full Graph RAG**:
   - Comprehensive graph-aware retrieval
   - Relationship strength influences document scoring
   - Traverses the graph to find semantically related content

4. **LangChain Integrations**:
   - Implementations of Naive and Knowledge Graph RAG using LangChain components
   - Leverages advanced embedding models and retrieval patterns

5. **LangGraph Implementations**:
   - Structured workflow approach to document retrieval
   - Multi-step graph traversal and scoring
   - Stateful retrieval process

## Data Structure

The system works with a dataset of articles and their relationships:

- Articles with content, metadata, and embeddings
- Author information and article-author relationships
- Category classification and hierarchies
- Explicit relationships between articles (cites, extends, contradicts, etc.)

## Performance Comparison

### Retrieval Quality


Based on the evaluation results (see `results/DataBases_Results.txt`):

| Database   | Avg Precision | Avg Recall | Avg F-measure |
|------------|--------------|------------|---------------|
| FalkorDB   | 0.500         | 0.124      | 0.190         |
| Neo4j      | 0.455         | 0.107      | 0.171         |
| PostgreSQL | 0.438         | 0.096      | 0.152         |

### Latency Performance

| Approach                | FalkorDB | Neo4j    | PostgreSQL |
|-------------------------|----------|----------|------------|
| Naive RAG              | 0.065s   | 0.077s   | 0.068s     |
| Knowledge Graph RAG     | 0.060s   | 0.062s   | 0.059s     |
| Full Graph RAG          | 0.051s   | 0.058s   | 0.061s     |
| LangChain Naive RAG     | 0.840s   | 1.033s   | 0.835s     |
| LangChain Knowledge RAG | 0.045s   | 0.049s   | 0.063s     |
| LangGraph Full Graph RAG| 0.087s   | 3.634s   | 0.087s     |

### Key Findings

1. **Graph Integration Benefits**: All databases show improved retrieval quality when using graph-enhanced methodologies compared to naive approaches
2. **Database Specialization**: Native graph databases (FalkorDB, Neo4j) provide marginal improvements in retrieval quality for graph-heavy operations
3. **LangChain Overhead**: LangChain implementations typically show higher latency due to additional abstraction layers
4. **LangGraph Performance**: LangGraph implementations vary significantly in performance across database backends

## Getting Started

### Prerequisites

- Python 3.8+
- Docker (optional, for containerized database instances)
- Database installations:
  - FalkorDB (Redis-compatible)
  - Neo4j
  - PostgreSQL

### Installation

1. Clone the repository
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Set up database connections in respective implementation folders

### Running Evaluations

Each database implementation has its own main script:

- FalkorDB: `python src/FalkorDb_Tasks/main.py`
- Neo4j: `python src/Neo4j_Tasks/Neo_Main.py`
- PostgreSQL: `python src/Postgresql_Tasks/Post_main.py`

For comprehensive evaluation across all systems:
```
python src/FalkorDb_Tasks/main.py
python src/Neo4j_Tasks/Neo_Main.py
python src/Postgresql_Tasks/Post_main.py
```

## Visualization

The project generates knowledge graph visualizations for each database implementation:
- `falkor_graph_visualization.png`
- `falkor_langgraph_visualization.png`
- `postgres_graph_visualization.png`

## Conclusion

This project demonstrates that:

1. Knowledge graph integration significantly enhances RAG system quality
2. Different database technologies can be effectively adapted for graph-enhanced RAG
3. The choice of database backend should be based on specific use case requirements:
   - FalkorDB offers good performance with moderate graph complexity
   - Neo4j provides rich graph operations but with some performance overhead
   - PostgreSQL can be effectively adapted for graph operations despite not being a native graph database

The integration of LangChain and LangGraph with these database backends offers powerful combinations for building sophisticated RAG systems with varying trade-offs in performance and capability.