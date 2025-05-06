from data_manager import FalkorDataManager
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.graphs import NetworkxEntityGraph
from langgraph.graph import StateGraph
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer
import time
import os
from dotenv import load_dotenv
import torch
import networkx as nx
from typing import List, Dict, Any, Optional, TypedDict
from pydantic import BaseModel, Field

load_dotenv()

# Define state for the graph
class GraphState(TypedDict):
    query: str
    visited_nodes: List[str]
    current_node: str
    results: List[Dict]
    depth: int
    max_depth: int
    include_content: bool

class FalkorLangGraphRAG:
    def __init__(self, data_manager=None, embedding_model_name="text-embedding-ada-002"):
        """Initialize with existing data manager or create new one"""
        if isinstance(data_manager, FalkorDataManager):
            self.data_manager = data_manager
            self._shared_data_manager = True
        else:
            # Create a new data manager if none was provided
            from falkordb import FalkorDB
            host = "localhost"
            port = 6379
            db = FalkorDB(host=host, port=port)
            self.data_manager = FalkorDataManager(db)
            self._shared_data_manager = False
        
        # Initialize LangChain embeddings
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model_name,
            # openai_api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        # Initialize ROUGE scorer
        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        
        # Initialize vector store
        self._initialize_vector_store()
        
        # Build knowledge graph
        self.nx_graph = nx.DiGraph()
        self._build_knowledge_graph()
        
        # Initialize LangGraph components
        self.entity_graph = NetworkxEntityGraph(self.nx_graph)
        self.workflow = self._build_workflow_graph()
        
    def _initialize_vector_store(self):
        """Initialize or load vector store with content from FalkorDB"""
        # Check if we have a persisted vector store
        vector_store_path = "faiss_db/falkor_langgraph"
        if os.path.exists(vector_store_path):
            self.vector_store = FAISS.load_local(
                vector_store_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            return
            
        # If not, we need to create one from scratch
        print("Building new vector store for LangGraph RAG...")
        
        # Get all articles from FalkorDB
        articles = self.data_manager.get_all_articles()
        
        if not articles:
            raise ValueError("No articles found in FalkorDB")
            
        # Process articles for vector store
        texts = []
        metadatas = []
        
        for article in articles:
            # Split content into chunks for better retrieval
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            chunks = text_splitter.split_text(article['content'])
            
            for chunk in chunks:
                texts.append(chunk)
                metadatas.append({
                    'article_id': article['article_id'],
                    'title': article['title'],
                    'topic': article.get('topic', ''),
                    'node_id': f"article_{article['article_id']}",
                    'chunk_index': len(texts) - 1
                })
        
        # Create vector store
        self.vector_store = FAISS.from_texts(
            texts, 
            self.embeddings, 
            metadatas=metadatas
        )
        
        # Persist vector store
        os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
        self.vector_store.save_local(vector_store_path)
    
    def _build_knowledge_graph(self):
        """Build a knowledge graph using data from FalkorDB"""
        print("Building knowledge graph for LangGraph RAG...")
        
        # Get article data
        articles = self.data_manager.get_all_articles()
        authors = self.data_manager.get_all_authors()
        categories = self.data_manager.get_all_categories()
        article_authors = self.data_manager.get_article_authors()
        article_categories = self.data_manager.get_article_categories()
        
        # Add article nodes
        for article in articles:
            self.nx_graph.add_node(
                f"article_{article['article_id']}", 
                type='article',
                title=article['title'],
                content=article['content'], 
                topic=article.get('topic', ''),
                embedding=None  # Will be populated on demand
            )
        
        # Add author nodes and edges
        for author in authors:
            self.nx_graph.add_node(
                f"author_{author['author_id']}", 
                type='author', 
                name=author['name']
            )
        
        for aa in article_authors:
            self.nx_graph.add_edge(
                f"article_{aa['article_id']}", 
                f"author_{aa['author_id']}", 
                type='written_by'
            )
        
        # Add category nodes and edges
        for category in categories:
            self.nx_graph.add_node(
                f"category_{category['category_id']}", 
                type='category', 
                name=category['name']
            )
        
        for ac in article_categories:
            self.nx_graph.add_edge(
                f"article_{ac['article_id']}", 
                f"category_{ac['category_id']}", 
                type='belongs_to',
                is_primary=ac.get('is_primary', False)
            )
        
        # Add article relationships
        relationships = self.data_manager.get_article_relationships()
        for rel in relationships:
            self.nx_graph.add_edge(
                f"article_{rel['source_article_id']}", 
                f"article_{rel['target_article_id']}", 
                type=rel['relationship_type'],
                weight=rel.get('strength', 1.0)
            )
    
    def _build_workflow_graph(self):
        """Build the LangGraph workflow for RAG"""
        
        workflow = StateGraph(GraphState)
        
        # Define the starting point for search
        def start_search(state: GraphState) -> GraphState:
            """Find the most relevant nodes for the query to begin search"""
            query = state["query"]
            
            # Search vector store for the best starting points
            docs_with_scores = self.vector_store.similarity_search_with_score(query, k=3)
            starting_nodes = []
            
            for doc, score in docs_with_scores:
                node_id = doc.metadata.get('node_id')
                if node_id:
                    starting_nodes.append({
                        "node_id": node_id,
                        "score": 1.0 / (1.0 + score),
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    })
            
            # If no nodes found, return empty results
            if not starting_nodes:
                return {**state, "results": []}
            
            # Sort by score and get best node
            starting_nodes.sort(key=lambda x: x["score"], reverse=True)
            best_node = starting_nodes[0]["node_id"]
            
            # Initialize graph traversal
            return {
                **state, 
                "current_node": best_node,
                "visited_nodes": [best_node],
                "results": starting_nodes,
                "depth": 0
            }
        
        # Define node exploration function
        def explore_node(state: GraphState) -> GraphState:
            """Explore connected nodes from the current node"""
            current_node = state["current_node"]
            visited = state["visited_nodes"]
            results = state["results"]
            query = state["query"]
            depth = state["depth"]
            max_depth = state["max_depth"]
            
            # Don't explore if we've reached max depth
            if depth >= max_depth:
                return state
            
            # Get neighbors from the graph
            neighbors = []
            for neighbor in self.nx_graph.neighbors(current_node):
                if neighbor not in visited:
                    edge_data = self.nx_graph.get_edge_data(current_node, neighbor)
                    neighbors.append((neighbor, edge_data))
            
            if not neighbors:
                return state
            
            # Process each neighbor
            for neighbor, edge_data in neighbors:
                node_data = self.nx_graph.nodes[neighbor]
                
                # Skip if not an article (for now we only want to return articles)
                if node_data.get('type') != 'article' and not state["include_content"]:
                    continue
                
                # Get node content and relationship boost
                relationship_type = edge_data.get('type', '')
                relationship_weight = edge_data.get('weight', 1.0)
                
                # Calculate connection boost based on relationship type
                if relationship_type == 'cites':
                    boost = 0.3
                elif relationship_type == 'extends':
                    boost = 0.4
                elif relationship_type == 'contradicts':
                    boost = 0.2
                else:
                    boost = 0.1
                
                boost *= relationship_weight
                
                if node_data.get('type') == 'article':
                    # For articles, we need full content and similarity score
                    content = node_data.get('content', '')
                    title = node_data.get('title', '')
                    
                    # Get similarity score with query
                    similarity = 0.5  # Default score
                    if content:
                        # Get cached document from vector store for efficiency
                        docs = self.vector_store.similarity_search_with_score(
                            query, k=1, 
                            filter={"node_id": neighbor}
                        )
                        if docs:
                            similarity = 1.0 / (1.0 + docs[0][1])
                    
                    # Apply relationship boost
                    final_score = similarity * (1 + boost)
                    
                    results.append({
                        "node_id": neighbor,
                        "type": "article",
                        "title": title,
                        "content": content,
                        "score": final_score,
                        "base_score": similarity,
                        "relationship_boost": boost,
                        "relationship_type": relationship_type,
                        "path": visited + [neighbor]
                    })
                else:
                    # For non-article nodes, we just include metadata
                    if state["include_content"]:
                        results.append({
                            "node_id": neighbor,
                            "type": node_data.get('type', 'unknown'),
                            "name": node_data.get('name', ''),
                            "relationship_type": relationship_type,
                            "path": visited + [neighbor]
                        })
            
            # Mark this node as visited
            visited.append(current_node)
            
            # Find the next unvisited node with highest score
            unvisited_results = [r for r in results if r["node_id"] not in visited and r.get("type") == "article"]
            if unvisited_results:
                unvisited_results.sort(key=lambda x: x["score"], reverse=True)
                next_node = unvisited_results[0]["node_id"]
                return {**state, "current_node": next_node, "visited_nodes": visited, "results": results, "depth": depth + 1}
            else:
                return {**state, "visited_nodes": visited, "results": results}
        
        # Define end function to finalize results
        def end_search(state: GraphState) -> GraphState:
            """Finalize search results"""
            # Just return the state as is, any final processing can be added here
            return state
        
        # Define routing condition
        def should_continue_exploring(state: GraphState) -> str:
            """Check if we should continue exploring"""
            if state["depth"] < state["max_depth"] and len(state["visited_nodes"]) < len(self.nx_graph):
                # Find if there are unvisited high-value nodes
                unvisited = [r for r in state["results"] if r["node_id"] not in state["visited_nodes"] and r.get("type") == "article"]
                if unvisited:
                    return "continue"
            return "complete"
        
        # Build the graph workflow
        workflow.add_node("start", start_search)
        workflow.add_node("explore", explore_node)
        workflow.add_node("end", end_search)  # Define the end node properly
        
        workflow.add_edge("start", "explore")
        workflow.add_conditional_edges(
            "explore",
            should_continue_exploring,
            {
                "continue": "explore",
                "complete": "end"
            }
        )
        
        workflow.set_entry_point("start")
        workflow.set_finish_point("end")
        
        return workflow.compile()
    
    def retrieve(self, query: str, k: int = 2, ground_truth: list = None):
        """Retrieve documents using graph-enhanced retrieval with LangGraph"""
        start_time = time.time()
        
        # Run the workflow graph
        inputs = {
            "query": query,
            "visited_nodes": [],
            "current_node": "",
            "results": [],
            "depth": 0,
            "max_depth": 3,  # How deep to explore the graph
            "include_content": False  # Only include article content in results
        }
        
        result = self.workflow.invoke(inputs)
        
        # Post-process results
        graph_results = result["results"]
        
        # Sort by score and get the top K
        graph_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        top_results = graph_results[:k]
        
        # Add latency information
        for result in top_results:
            result["latency"] = time.time() - start_time
            
            # Calculate graph boost info
            result["graph_boost"] = result.get("relationship_boost", 0)
            
            # Get connected nodes
            node_id = result["node_id"]
            connected_nodes = []
            for n in self.nx_graph.neighbors(node_id):
                node_data = self.nx_graph.nodes[n]
                edge_data = self.nx_graph.get_edge_data(node_id, n)
                connected_nodes.append({
                    'type': node_data.get('type', ''),
                    'name': node_data.get('name', node_data.get('title', '')),
                    'relationship': edge_data.get('type', '')
                })
            
            result['connected_nodes'] = connected_nodes
        
        # Calculate metrics if ground truth is provided
        if ground_truth:
            for result in top_results:
                rouge_scores = [self.scorer.score(result["content"], gt) for gt in ground_truth]
                if rouge_scores:
                    best_score = max(rouge_scores, key=lambda x: x['rougeL'].fmeasure)
                    result['precision'] = best_score['rougeL'].precision
                    result['recall'] = best_score['rougeL'].recall
                    result['f1'] = best_score['rougeL'].fmeasure
                else:
                    result['precision'] = result['recall'] = result['f1'] = 0.0
        else:
            for result in top_results:
                result['precision'] = result['recall'] = result['f1'] = 0.0
        
        return top_results
    
    def visualize(self, output_file='falkor_langgraph_visualization.png'):
        """Visualize the knowledge graph"""
        plt.figure(figsize=(20, 20))
        
        try:
            # Define node colors by type
            node_colors = {
                'article': 'skyblue',
                'author': 'lightgreen',
                'category': 'orange'
            }
            
            # Create layout
            pos = nx.spring_layout(self.nx_graph, k=2, iterations=50)
            
            # Draw nodes by type
            for node_type, color in node_colors.items():
                nodes = [n for n, d in self.nx_graph.nodes(data=True) if d.get('type') == node_type]
                if nodes:
                    nx.draw_networkx_nodes(
                        self.nx_graph, pos,
                        nodelist=nodes,
                        node_color=color,
                        node_size=2000,
                        label=node_type
                    )
            
            # Draw edges and labels
            nx.draw_networkx_edges(self.nx_graph, pos, edge_color='gray', arrows=True)
            edge_labels = {}
            for u, v, data in self.nx_graph.edges(data=True):
                edge_labels[(u, v)] = data.get('type', '')
            
            nx.draw_networkx_edge_labels(
                self.nx_graph, pos,
                edge_labels=edge_labels,
                font_size=8
            )
            
            # Draw node labels
            labels = {}
            for node in self.nx_graph.nodes():
                node_data = self.nx_graph.nodes[node]
                if node_data.get('type') == 'article':
                    labels[node] = f"{node}\n{node_data.get('title', '')[:20]}..."
                else:
                    labels[node] = f"{node}\n{node_data.get('name', '')}"
            
            nx.draw_networkx_labels(self.nx_graph, pos, labels, font_size=8)
            
            plt.title("FalkorDB LangGraph Visualization")
            plt.legend()
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_file, format='PNG', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error in visualization: {str(e)}")
    
    def close(self):
        """Only close if we created our own data manager"""
        if not self._shared_data_manager:
            self.data_manager.close()