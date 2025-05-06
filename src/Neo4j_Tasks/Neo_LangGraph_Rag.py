from neo4j import GraphDatabase
import time
from data_manager import DataManager
from rouge_score import rouge_scorer
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import StateGraph, END, START
from typing import Dict, Any, TypedDict, List, Annotated
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Define state types
class GraphRetrieverState(TypedDict):
    query: str
    results: List[Dict[str, Any]]
    k: int
    ground_truth: List[str]
    scores: Dict[str, float]
    latency: float

class NeoLangGraphRAG:
    def __init__(self, driver=None):
        self.data_manager = DataManager(driver)
        self.driver = driver
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Use OpenAI embeddings instead of HuggingFace
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Build the LangGraph workflow
        self.workflow = self._build_graph()
    
    def _build_graph(self):
        # Create a StateGraph for managing RAG workflow
        graph = StateGraph(GraphRetrieverState)
        
        # Define nodes for the workflow
        graph.add_node("retrieve_chunks", self._retrieve_similar_chunks)
        graph.add_node("enhance_with_graph", self._enhance_with_graph)
        graph.add_node("calculate_scores", self._calculate_scores)
        
        # Add START node connection - this was missing
        graph.add_edge(START, "retrieve_chunks")
        
        # Connect the nodes
        graph.add_edge("retrieve_chunks", "enhance_with_graph")
        graph.add_edge("enhance_with_graph", "calculate_scores")
        graph.add_edge("calculate_scores", END)
        
        # Compile the graph
        return graph.compile()
    
    def _retrieve_similar_chunks(self, state: GraphRetrieverState) -> GraphRetrieverState:
        start_time = time.time()
        query = state["query"]
        k = state.get("k", 2)
        
        # Get similar chunks using semantic search
        similar_chunks = self.data_manager.get_similar_chunks(query, k)
        
        # Update state
        state["results"] = similar_chunks
        state["latency"] = time.time() - start_time
        return state
    
    def _enhance_with_graph(self, state: GraphRetrieverState) -> GraphRetrieverState:
        """Enhance retrieval results with graph context from Neo4j"""
        results = state["results"]
        enhanced_results = []
        
        for chunk in results:
            # Get graph context for each chunk
            graph_context = self.data_manager.get_graph_context(chunk["text"])
            
            # Add graph context to the chunk
            enhanced_chunk = {**chunk, "graph_context": graph_context}
            enhanced_results.append(enhanced_chunk)
        
        # Update state with enhanced results
        state["results"] = enhanced_results
        return state
    
    def _calculate_scores(self, state: GraphRetrieverState) -> GraphRetrieverState:
        """Calculate Rouge scores for the results"""
        results = state["results"]
        ground_truth = state.get("ground_truth", [])
        
        for i, result in enumerate(results):
            # Calculate ROUGE scores if ground truth is available
            if ground_truth:
                scores = self.calculate_rouge_scores(result["text"], ground_truth)
                results[i].update(scores)
        
        # Update state with scored results
        state["results"] = results
        return state
    
    def retrieve(self, query: str, k: int = 2, ground_truth: list = None):
        # Prepare initial state
        initial_state: GraphRetrieverState = {
            "query": query,
            "results": [],
            "k": k,
            "ground_truth": ground_truth or [],
            "scores": {},
            "latency": 0
        }
        
        # Execute the workflow
        start_time = time.time()
        final_state = self.workflow.invoke(initial_state)
        total_latency = time.time() - start_time
        
        # Update latency in results
        results = final_state["results"]
        for result in results:
            result["latency"] = total_latency
        
        return results

    def calculate_rouge_scores(self, prediction, ground_truth):
        if not ground_truth:
            return {'precision': 0, 'recall': 0, 'f1': 0}
        
        # Calculate ROUGE scores against all ground truth passages
        scores = []
        for truth in ground_truth:
            score = self.scorer.score(prediction, truth)
            scores.append({
                'precision': score['rouge1'].precision,
                'recall': score['rouge1'].recall,
                'f1': score['rouge1'].fmeasure
            })
        
        # Return the best scores
        best_score = max(scores, key=lambda x: x['f1'])
        return best_score