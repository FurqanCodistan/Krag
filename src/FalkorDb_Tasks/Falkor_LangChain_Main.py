from falkordb import FalkorDB
from data_manager import FalkorDataManager
from Falkor_LangChain_Knowledge_Rag import FalkorLangChainKnowledgeRAG
from Falkor_LangGraph_Rag import FalkorLangGraphRAG
from Falkor_LangChain_Naive_Rag import FalkorLangChainNaiveRAG
from tabulate import tabulate
import os
from dotenv import load_dotenv
import traceback

load_dotenv(override=True)

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

def run_all_langchain_rags(query: str, k: int = 2, verbose: bool = False, data_manager=None):
    """Run all LangChain/LangGraph RAG systems using a shared data manager instance"""
    try:
        # Use existing data manager or create new one
        if data_manager is None:
            host = "localhost"
            port = 6379
            db = FalkorDB(host=host, port=port)
            data_manager = FalkorDataManager(db)

        # Get ground truth for evaluation
        ground_truth = get_ground_truth(query)
        
        # Initialize RAG systems with shared data manager
        naive_rag = FalkorLangChainNaiveRAG(data_manager)
        knowledge_rag = FalkorLangChainKnowledgeRAG(data_manager)
        graph_rag = FalkorLangGraphRAG(data_manager)

        print(f"\nProcessing query with LangChain/LangGraph: {query}")
        if verbose:
            print("Ground truth:", ground_truth)

        # Get results from each system
        naive_results = naive_rag.retrieve(query, k, ground_truth)
        knowledge_results = knowledge_rag.retrieve(query, k, ground_truth)
        graph_results = graph_rag.retrieve(query, k, ground_truth)

        # Format results for tabulation
        all_results = []
        
        # Add Naive RAG (LangChain) results
        for rank, result in enumerate(naive_results, 1):
            all_results.append([
                "LangChain Naive RAG",
                rank,
                result.get("score", 0),
                result.get("latency", 0),
                result.get("precision", 0),
                result.get("recall", 0),
                result.get("f1", 0)
            ])

        # Add Knowledge Graph RAG (LangChain) results
        for rank, result in enumerate(knowledge_results, 1):
            if verbose:
                print(f"\nLangChain Knowledge Graph result {rank}:")
                print(f"Base score: {result.get('base_score', 0):.4f}")
                print(f"Metadata score: {result.get('metadata_score', 0):.4f}")
                print(f"Final score: {result.get('score', 0):.4f}")
                print("Metadata:", result.get('metadata', {}))
            
            all_results.append([
                "LangChain Knowledge RAG",
                rank,
                result.get("score", 0),
                result.get("latency", 0),
                result.get("precision", 0),
                result.get("recall", 0),
                result.get("f1", 0)
            ])

        # # Add Full Graph RAG (LangGraph) results
        # for rank, result in enumerate(graph_results, 1):
        #     if verbose:
        #         print(f"\nLangGraph result {rank}:")
        #         print(f"Base score: {result.get('base_score', 0):.4f}")
        #         print(f"Graph boost: {result.get('graph_boost', 0):.4f}")
        #         print(f"Final score: {result.get('score', 0):.4f}")
        #         print("Connected nodes:", result.get('connected_nodes', []))

            all_results.append([
                "LangGraph Full RAG",
                rank,
                result.get("score", 0),
                result.get("latency", 0),
                result.get("precision", 0),
                result.get("recall", 0),
                result.get("f1", 0)
            ])

        # Print comparison table
        headers = ["Method", "Rank", "Score", "Latency", "Precision", "Recall", "F-measure"]
        print("\nLangChain/LangGraph Results Comparison:")
        print(tabulate(all_results, headers=headers, tablefmt="grid"))

        # Generate graph visualization
        graph_rag.visualize()

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
        data_manager = run_all_langchain_rags(query, k=2, verbose=True, data_manager=data_manager)
        print("\n" + "="*80 + "\n")