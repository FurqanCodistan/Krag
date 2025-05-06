from falkordb import FalkorDB
from data_manager import FalkorDataManager
from Falkor_Knowledge import FalkorKnowledgeRAG
from Falkor_Graph_rag import FalkorGraphRAG
from Falkor_Naive_Rag import FalkorNaiveRAG
from Falkor_LangChain_Knowledge_Rag import FalkorLangChainKnowledgeRAG
from Falkor_LangGraph_Rag import FalkorLangGraphRAG
from Falkor_LangChain_Naive_Rag import FalkorLangChainNaiveRAG
from tabulate import tabulate
import os
from dotenv import load_dotenv
import traceback
import time
import numpy as np

load_dotenv()

def get_ground_truth(query: str):
    # Sample ground truth data for testing
    ground_truth_data = {
        "What is artificial intelligence?": [
            "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals and humans.",
            "AI is a branch of computer science that aims to create intelligent machines that can perform tasks that typically require human intelligence."
        ],
        "What is Image Forgery?": [
            "Image forgery refers to the manipulation of an image to create a false representation of reality.",
            "It involves altering an image in a way that misleads viewers, often for malicious purposes."
        ]
    }
    return ground_truth_data.get(query, ["No ground truth available"])

def calculate_averages(results):
    """Calculate average metrics for a set of RAG results"""
    if not results:
        return {
            'avg_score': 0,
            'avg_latency': 0,
            'avg_precision': 0,
            'avg_recall': 0,
            'avg_f1': 0
        }
    
    scores = [result.get('score', 0) for result in results]
    latencies = [result.get('latency', 0) for result in results]
    precisions = [result.get('precision', 0) for result in results]
    recalls = [result.get('recall', 0) for result in results]
    f1s = [result.get('f1', 0) for result in results]
    
    return {
        'avg_score': np.mean(scores) if scores else 0,
        'avg_latency': np.mean(latencies) if latencies else 0,
        'avg_precision': np.mean(precisions) if precisions else 0,
        'avg_recall': np.mean(recalls) if recalls else 0,
        'avg_f1': np.mean(f1s) if f1s else 0
    }

def run_all_rag_systems(query: str, k: int = 2, verbose: bool = False, data_manager=None):
    """Run all RAG systems (original and LangChain) using a shared data manager instance"""
    try:
        print(f"\n{'-'*20} Processing query: '{query}' {'-'*20}")
        
        # Use existing data manager or create new one
        if data_manager is None:
            host = "localhost"
            port = 6379
            db = FalkorDB(host=host, port=port)
            data_manager = FalkorDataManager(db)

        # Get ground truth for evaluation
        ground_truth = get_ground_truth(query)
        if verbose:
            print("Ground truth:", ground_truth)
        
        # Initialize all RAG systems with shared data manager
        print("\nInitializing RAG systems...")
        
        # Original implementations
        naive_rag = FalkorNaiveRAG(data_manager)
        knowledge_rag = FalkorKnowledgeRAG(data_manager)
        graph_rag = FalkorGraphRAG(data_manager)
        
        # LangChain implementations
        lang_naive_rag = FalkorLangChainNaiveRAG(data_manager)
        lang_knowledge_rag = FalkorLangChainKnowledgeRAG(data_manager)
        lang_graph_rag = FalkorLangGraphRAG(data_manager)

        # Run all retrievals with timing
        all_results = {}
        
        # Original Naive RAG
        print("\nRunning Original Naive RAG...")
        start_time = time.time()
        naive_results = naive_rag.retrieve(query, k, ground_truth)
        orig_naive_time = time.time() - start_time
        all_results['naive'] = naive_results
        
        # LangChain Naive RAG
        print("Running LangChain Naive RAG...")
        start_time = time.time()
        lang_naive_results = lang_naive_rag.retrieve(query, k, ground_truth)
        lang_naive_time = time.time() - start_time
        all_results['lang_naive'] = lang_naive_results
        
        # Original Knowledge Graph RAG
        print("Running Original Knowledge Graph RAG...")
        start_time = time.time()
        knowledge_results = knowledge_rag.retrieve(query, k, ground_truth)
        orig_knowledge_time = time.time() - start_time
        all_results['knowledge'] = knowledge_results
        
        # LangChain Knowledge Graph RAG
        print("Running LangChain Knowledge Graph RAG...")
        start_time = time.time()
        lang_knowledge_results = lang_knowledge_rag.retrieve(query, k, ground_truth)
        lang_knowledge_time = time.time() - start_time
        all_results['lang_knowledge'] = lang_knowledge_results
        
        # Original Full Graph RAG
        print("Running Original Full Graph RAG...")
        start_time = time.time()
        graph_results = graph_rag.retrieve(query, k, ground_truth)
        orig_graph_time = time.time() - start_time
        all_results['graph'] = graph_results
        
        # LangGraph Full Graph RAG
        print("Running LangGraph Full Graph RAG...")
        start_time = time.time()
        lang_graph_results = lang_graph_rag.retrieve(query, k, ground_truth)
        lang_graph_time = time.time() - start_time
        all_results['lang_graph'] = lang_graph_results

        # Calculate averages for each system
        avg_metrics = {
            'naive': calculate_averages(naive_results),
            'lang_naive': calculate_averages(lang_naive_results),
            'knowledge': calculate_averages(knowledge_results),
            'lang_knowledge': calculate_averages(lang_knowledge_results),
            'graph': calculate_averages(graph_results),
            'lang_graph': calculate_averages(lang_graph_results)
        }

        # Format results for tabulation in the requested order
        combined_results = []
        
        # The order requested: Sentence Naive, LangChain Naive, Sentence Graph, LangGraph Graph
        # First: Add Original Naive RAG results
        for rank, result in enumerate(naive_results, 1):
            combined_results.append([
                "Original Naive RAG",
                rank,
                result.get("score", 0),
                result.get("latency", 0),
                result.get("precision", 0),
                result.get("recall", 0),
                result.get("f1", 0)
            ])
        
        # Second: Add LangChain Naive RAG results
        for rank, result in enumerate(lang_naive_results, 1):
            combined_results.append([
                "LangChain Naive RAG",
                rank,
                result.get("score", 0),
                result.get("latency", 0),
                result.get("precision", 0),
                result.get("recall", 0),
                result.get("f1", 0)
            ])
            
        # # Third: Add Original Knowledge Graph RAG results
        for rank, result in enumerate(knowledge_results, 1):
        #     if verbose:
        #         print(f"\nOriginal Knowledge Graph result {rank}:")
        #         print(f"Base score: {result.get('base_score', 0):.4f}")
        #         print(f"Metadata score: {result.get('metadata_score', 0):.4f}")
        #         print(f"Final score: {result.get('score', 0):.4f}")
            
            combined_results.append([
                "Original Knowledge RAG",
                rank,
                result.get("score", 0),
                result.get("latency", 0),
                result.get("precision", 0),
                result.get("recall", 0),
                result.get("f1", 0)
            ])
        
        # Fourth: Add LangChain Knowledge Graph RAG results
        for rank, result in enumerate(lang_knowledge_results, 1):
            # if verbose:
            #     print(f"\nLangChain Knowledge Graph result {rank}:")
            #     print(f"Base score: {result.get('base_score', 0):.4f}")
            #     print(f"Metadata score: {result.get('metadata_score', 0):.4f}")
            #     print(f"Final score: {result.get('score', 0):.4f}")
            
            combined_results.append([
                "LangChain Knowledge RAG",
                rank,
                result.get("score", 0),
                result.get("latency", 0),
                result.get("precision", 0),
                result.get("recall", 0),
                result.get("f1", 0)
            ])
            
        # Fifth: Add Original Full Graph RAG results
        for rank, result in enumerate(graph_results, 1):
            # if verbose:
            #     print(f"\nOriginal Full Graph result {rank}:")
            #     print(f"Base score: {result.get('base_score', 0):.4f}")
            #     print(f"Graph boost: {result.get('graph_boost', 0):.4f}")
            #     print(f"Final score: {result.get('score', 0):.4f}")

            combined_results.append([
                "Original Full Graph RAG",
                rank,
                result.get("score", 0),
                result.get("latency", 0),
                result.get("precision", 0),
                result.get("recall", 0),
                result.get("f1", 0)
            ])
        
        # Sixth: Add LangGraph Full Graph RAG results
        for rank, result in enumerate(lang_graph_results, 1):
            # if verbose:
            #     print(f"\nLangGraph Full Graph result {rank}:")
            #     print(f"Base score: {result.get('base_score', 0):.4f}")
            #     print(f"Graph boost: {result.get('graph_boost', 0):.4f}")
            #     print(f"Final score: {result.get('score', 0):.4f}")

            combined_results.append([
                "LangGraph Full Graph RAG",
                rank,
                result.get("score", 0),
                result.get("latency", 0),
                result.get("precision", 0),
                result.get("recall", 0),
                result.get("f1", 0)
            ])

        # Add overall execution time and average metrics for each approach
        overall_times = [
            ["Original Naive RAG", "-", orig_naive_time, 
             avg_metrics['naive']['avg_latency'], 
             avg_metrics['naive']['avg_precision'], 
             avg_metrics['naive']['avg_recall'], 
             avg_metrics['naive']['avg_f1']],
            
            ["LangChain Naive RAG", "-", lang_naive_time, 
             avg_metrics['lang_naive']['avg_latency'], 
             avg_metrics['lang_naive']['avg_precision'], 
             avg_metrics['lang_naive']['avg_recall'], 
             avg_metrics['lang_naive']['avg_f1']],
            
            ["Original Knowledge RAG", "-", orig_knowledge_time, 
             avg_metrics['knowledge']['avg_latency'], 
             avg_metrics['knowledge']['avg_precision'], 
             avg_metrics['knowledge']['avg_recall'], 
             avg_metrics['knowledge']['avg_f1']],
            
            ["LangChain Knowledge RAG", "-", lang_knowledge_time, 
             avg_metrics['lang_knowledge']['avg_latency'], 
             avg_metrics['lang_knowledge']['avg_precision'], 
             avg_metrics['lang_knowledge']['avg_recall'], 
             avg_metrics['lang_knowledge']['avg_f1']],
            
            ["Original Full Graph RAG", "-", orig_graph_time, 
             avg_metrics['graph']['avg_latency'], 
             avg_metrics['graph']['avg_precision'], 
             avg_metrics['graph']['avg_recall'], 
             avg_metrics['graph']['avg_f1']],
            
            ["LangGraph Full Graph RAG", "-", lang_graph_time, 
             avg_metrics['lang_graph']['avg_latency'], 
             avg_metrics['lang_graph']['avg_precision'], 
             avg_metrics['lang_graph']['avg_recall'], 
             avg_metrics['lang_graph']['avg_f1']],
        ]

        # Print comparison table
        headers = ["Method", "Rank", "Score", "Latency", "Precision", "Recall", "F-measure"]
        print("\n" + "="*50)
        print(f"All RAG Systems Comparison for Query: '{query}'")
        print("="*50)
        print(tabulate(combined_results, headers=headers, tablefmt="grid"))
        
        print("\n" + "="*50)
        print("Overall Processing Times and Average Metrics")
        print("="*50)
        time_headers = ["Method", "Rank", "Total Time (s)", "Avg Latency", "Avg Precision", "Avg Recall", "Avg F-measure"]
        print(tabulate(overall_times, headers=time_headers, tablefmt="grid"))

        # Generate graph visualizations
        print("\nGenerating graph visualizations...")
        graph_rag.visualize(output_file=f"orig_graph_vis_{query.replace(' ', '_').lower()[:20]}.png")
        lang_graph_rag.visualize(output_file=f"lang_graph_vis_{query.replace(' ', '_').lower()[:20]}.png")
        
        print("\nProcessing complete!")
        return data_manager

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    test_queries = [
        "What is Image Forgery?",
        "What is artificial intelligence?"
    ]
    
    # Use single data manager instance for all queries
    data_manager = None
    for query in test_queries:
        data_manager = run_all_rag_systems(query, k=2, verbose=True, data_manager=data_manager)
        print("\n" + "="*80 + "\n")