from neo4j import GraphDatabase
from data_manager import DataManager
from Neo_Knowledge import NeoKnowledgeGraph
from Neo_Graph_rag import NeoGraphRAG
from Neo_Naive_Rag import NeoNaiveRAG
from Neo_LangChain_Naive_Rag import NeoLangChainNaiveRAG
from Neo_LangChain_Knowledge_Rag import NeoLangChainKnowledgeRAG
from Neo_LangGraph_Rag import NeoLangGraphRAG
from tabulate import tabulate
import os
import time
from dotenv import load_dotenv

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
        # Add more ground truth pairs as needed
    }
    return ground_truth_data.get(query, ["No ground truth available"])

def run_all_rags(query: str, k: int = 2):
    uri = os.getenv("NEO4J_URI_local", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "your_password")
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    # For tracking overall metrics
    overall_metrics = {
        "Original Naive RAG": {"total_time": 0, "latencies": [], "precisions": [], "recalls": [], "f_measures": []},
        "LangChain Naive RAG": {"total_time": 0, "latencies": [], "precisions": [], "recalls": [], "f_measures": []},
        "Original Knowledge RAG": {"total_time": 0, "latencies": [], "precisions": [], "recalls": [], "f_measures": []},
        "LangChain Knowledge RAG": {"total_time": 0, "latencies": [], "precisions": [], "recalls": [], "f_measures": []},
        "Original Full Graph RAG": {"total_time": 0, "latencies": [], "precisions": [], "recalls": [], "f_measures": []},
        "LangGraph Full Graph RAG": {"total_time": 0, "latencies": [], "precisions": [], "recalls": [], "f_measures": []}
    }

    try:
        # Get ground truth for evaluation
        ground_truth = get_ground_truth(query)
        
        # Initialize DataManager and ensure database setup
        data_manager = DataManager(driver)
        
        # Fetch all required data
        all_data = data_manager.get_all_data()
        
        # Initialize RAG systems
        knowledge_rag = NeoKnowledgeGraph(driver)
        graph_rag = NeoGraphRAG(driver)
        naive_rag = NeoNaiveRAG(driver)
        
        # Initialize LangChain RAG systems
        lc_naive_rag = NeoLangChainNaiveRAG(driver)
        lc_knowledge_rag = NeoLangChainKnowledgeRAG(driver)
        
        # Initialize LangGraph RAG system
        lg_rag = NeoLangGraphRAG(driver)

        # Print header
        print("=================================NEO4J DATABASE========================================================")
        print(f"\n-------------------- Processing query: '{query}' --------------------")
        print(f"\nGround truth: {ground_truth}")
        print("\n==================================================")
        print(f"All RAG Systems Comparison for Query: '{query}'")
        print("==================================================")

        # Track all results for table formatting
        all_results = []
        
        # Get results from each RAG system with ground truth
        start_time = time.time()
        naive_results = naive_rag.retrieve(query, k, ground_truth)
        overall_metrics["Original Naive RAG"]["total_time"] = time.time() - start_time
        
        start_time = time.time()
        knowledge_results = knowledge_rag.retrieve(query, k, ground_truth)
        overall_metrics["Original Knowledge RAG"]["total_time"] = time.time() - start_time
        
        start_time = time.time()
        graph_results = graph_rag.retrieve(query, k, ground_truth)
        overall_metrics["Original Full Graph RAG"]["total_time"] = time.time() - start_time
        
        # Get results from LangChain RAG systems
        start_time = time.time()
        lc_naive_results = lc_naive_rag.retrieve(query, k, ground_truth)
        overall_metrics["LangChain Naive RAG"]["total_time"] = time.time() - start_time
        
        start_time = time.time()
        lc_knowledge_results = lc_knowledge_rag.retrieve(query, k, ground_truth)
        overall_metrics["LangChain Knowledge RAG"]["total_time"] = time.time() - start_time
        
        # Get results from LangGraph RAG system
        start_time = time.time()
        lg_results = lg_rag.retrieve(query, k, ground_truth)
        overall_metrics["LangGraph Full Graph RAG"]["total_time"] = time.time() - start_time
        
        # Add results to the all_results list in the desired order
        for rank, result in enumerate(naive_results, 1):
            latency = result.get("latency", 0)
            precision = result.get("precision", 0)
            recall = result.get("recall", 0)
            f1 = result.get("f1", 0)
            
            all_results.append([
                "Original Naive RAG",
                rank,
                result.get("score", 0),
                latency,
                precision,
                recall,
                f1
            ])
            
            overall_metrics["Original Naive RAG"]["latencies"].append(latency)
            overall_metrics["Original Naive RAG"]["precisions"].append(precision)
            overall_metrics["Original Naive RAG"]["recalls"].append(recall)
            overall_metrics["Original Naive RAG"]["f_measures"].append(f1)
            
        for rank, result in enumerate(lc_naive_results, 1):
            latency = result.get("latency", 0)
            precision = result.get("precision", 0)
            recall = result.get("recall", 0)
            f1 = result.get("f1", 0)
            
            all_results.append([
                "LangChain Naive RAG",
                rank,
                result.get("score", 0),
                latency,
                precision,
                recall,
                f1
            ])
            
            overall_metrics["LangChain Naive RAG"]["latencies"].append(latency)
            overall_metrics["LangChain Naive RAG"]["precisions"].append(precision)
            overall_metrics["LangChain Naive RAG"]["recalls"].append(recall)
            overall_metrics["LangChain Naive RAG"]["f_measures"].append(f1)
            
        for rank, result in enumerate(knowledge_results, 1):
            latency = result.get("latency", 0)
            precision = result.get("precision", 0)
            recall = result.get("recall", 0)
            f1 = result.get("f1", 0)
            
            all_results.append([
                "Original Knowledge RAG",
                rank,
                result.get("score", 0),
                latency,
                precision,
                recall,
                f1
            ])
            
            overall_metrics["Original Knowledge RAG"]["latencies"].append(latency)
            overall_metrics["Original Knowledge RAG"]["precisions"].append(precision)
            overall_metrics["Original Knowledge RAG"]["recalls"].append(recall)
            overall_metrics["Original Knowledge RAG"]["f_measures"].append(f1)
            
        for rank, result in enumerate(lc_knowledge_results, 1):
            latency = result.get("latency", 0)
            precision = result.get("precision", 0)
            recall = result.get("recall", 0)
            f1 = result.get("f1", 0)
            
            all_results.append([
                "LangChain Knowledge RAG",
                rank,
                result.get("score", 0),
                latency,
                precision,
                recall,
                f1
            ])
            
            overall_metrics["LangChain Knowledge RAG"]["latencies"].append(latency)
            overall_metrics["LangChain Knowledge RAG"]["precisions"].append(precision)
            overall_metrics["LangChain Knowledge RAG"]["recalls"].append(recall)
            overall_metrics["LangChain Knowledge RAG"]["f_measures"].append(f1)

        for rank, result in enumerate(graph_results, 1):
            latency = result.get("latency", 0)
            precision = result.get("precision", 0)
            recall = result.get("recall", 0)
            f1 = result.get("f1", 0)
            
            all_results.append([
                "Original Full Graph RAG",
                rank,
                result.get("score", 0),
                latency,
                precision,
                recall,
                f1
            ])
            
            overall_metrics["Original Full Graph RAG"]["latencies"].append(latency)
            overall_metrics["Original Full Graph RAG"]["precisions"].append(precision)
            overall_metrics["Original Full Graph RAG"]["recalls"].append(recall)
            overall_metrics["Original Full Graph RAG"]["f_measures"].append(f1)
        
        for rank, result in enumerate(lg_results, 1):
            latency = result.get("latency", 0)
            precision = result.get("precision", 0)
            recall = result.get("recall", 0)
            f1 = result.get("f1", 0)
            
            all_results.append([
                "LangGraph Full Graph RAG",
                rank,
                result.get("score", 0),
                latency,
                precision,
                recall,
                f1
            ])
            
            overall_metrics["LangGraph Full Graph RAG"]["latencies"].append(latency)
            overall_metrics["LangGraph Full Graph RAG"]["precisions"].append(precision)
            overall_metrics["LangGraph Full Graph RAG"]["recalls"].append(recall)
            overall_metrics["LangGraph Full Graph RAG"]["f_measures"].append(f1)

        headers = ["Method", "Rank", "Score", "Latency", "Precision", "Recall", "F-measure"]
        print(tabulate(all_results, headers=headers, tablefmt="grid"))
        
        # Calculate average metrics
        overall_table_data = []
        for method, metrics in overall_metrics.items():
            avg_latency = sum(metrics["latencies"]) / max(len(metrics["latencies"]), 1)
            avg_precision = sum(metrics["precisions"]) / max(len(metrics["precisions"]), 1)
            avg_recall = sum(metrics["recalls"]) / max(len(metrics["recalls"]), 1)
            avg_f_measure = sum(metrics["f_measures"]) / max(len(metrics["f_measures"]), 1)
            
            overall_table_data.append([
                method,
                "-",
                metrics["total_time"],
                avg_latency,
                avg_precision,
                avg_recall,
                avg_f_measure
            ])

        # Print overall metrics table
        print("\n==================================================")
        print("Overall Processing Times and Average Metrics")
        print("==================================================")
        overall_headers = ["Method", "Rank", "Total Time (s)", "Avg Latency", "Avg Precision", "Avg Recall", "Avg F-measure"]
        print(tabulate(overall_table_data, headers=overall_headers, tablefmt="grid"))
        print("\n\n================================================================================\n\n")

    finally:
        driver.close()

if __name__ == "__main__":
    # Process first query
    query1 = "What is Image Forgery?"
    run_all_rags(query1)
    
    # Process second query
    query2 = "What is artificial intelligence?"
    run_all_rags(query2)
