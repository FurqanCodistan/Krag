import psycopg2
from data_manager import PostgresDataManager
from Postgres_Knowledge import PostgresKnowledgeRAG
from Postgres_Graph_rag import PostgresGraphRAG
from Postgres_Naive_Rag import PostgresNaiveRAG
from Postgres_LangChain_Naive_Rag import PostgresLangChainNaiveRAG
from Postgres_LangChain_Knowledge_Rag import PostgresLangChainKnowledgeRAG
from Postgres_LangGraph_Rag import PostgresLangGraphRAG
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

def run_all_rags(query: str, k: int = 2, verbose: bool = False):
    # Database connection parameters
    db_params = {
        "dbname": "Rag DB",
        "user": "postgres",
        "password": "12345",
        "host": "localhost",
        "port": "5432"
    }

    # Establish database connection
    conn = psycopg2.connect(**db_params)

    try:
        # Get ground truth for evaluation
        ground_truth = get_ground_truth(query)
        
        print(f"\n-------------------- Processing query: '{query}' --------------------\n")
        if verbose:
            print("Ground truth:", ground_truth)
            print()

        # Initialize RAG systems
        naive_rag = PostgresNaiveRAG(conn)
        knowledge_rag = PostgresKnowledgeRAG(conn)
        graph_rag = PostgresGraphRAG(conn)
        langchain_naive_rag = PostgresLangChainNaiveRAG(conn)
        langchain_knowledge_rag = PostgresLangChainKnowledgeRAG(conn)
        langgraph_rag = PostgresLangGraphRAG(conn)

        # Get results from each system with timing
        start_time = time.time()
        naive_results = naive_rag.retrieve(query, k, ground_truth)
        naive_total_time = time.time() - start_time
        
        start_time = time.time()
        knowledge_results = knowledge_rag.retrieve(query, k, ground_truth)
        knowledge_total_time = time.time() - start_time
        
        start_time = time.time()
        graph_results = graph_rag.retrieve(query, k, ground_truth)
        graph_total_time = time.time() - start_time
        
        start_time = time.time()
        langchain_naive_results = langchain_naive_rag.retrieve(query, k, ground_truth)
        langchain_naive_total_time = time.time() - start_time
        
        start_time = time.time()
        langchain_knowledge_results = langchain_knowledge_rag.retrieve(query, k, ground_truth)
        langchain_knowledge_total_time = time.time() - start_time
        
        start_time = time.time()
        langgraph_results = langgraph_rag.retrieve(query, k, ground_truth)
        langgraph_total_time = time.time() - start_time

        # Format results for tabulation
        all_results = []
        
        # Add Original Naive RAG results
        for rank, result in enumerate(naive_results, 1):
            all_results.append([
                "Original Naive RAG",
                rank,
                result.get("score", 0),
                result.get("latency", 0),
                result.get("precision", 0),
                result.get("recall", 0),
                result.get("f1", 0)
            ])
        
        # Add LangChain Naive RAG results
        for rank, result in enumerate(langchain_naive_results, 1):
            all_results.append([
                "LangChain Naive RAG",
                rank,
                result.get("score", 0),
                result.get("latency", 0),
                result.get("precision", 0),
                result.get("recall", 0),
                result.get("f1", 0)
            ])

        # Add Original Knowledge Graph RAG results
        for rank, result in enumerate(knowledge_results, 1):
            all_results.append([
                "Original Knowledge RAG",
                rank,
                result.get("score", 0),
                result.get("latency", 0),
                result.get("precision", 0),
                result.get("recall", 0),
                result.get("f1", 0)
            ])
            
        # Add LangChain Knowledge Graph RAG results
        for rank, result in enumerate(langchain_knowledge_results, 1):
            all_results.append([
                "LangChain Knowledge RAG",
                rank,
                result.get("score", 0),
                result.get("latency", 0),
                result.get("precision", 0),
                result.get("recall", 0),
                result.get("f1", 0)
            ])

        # Add Original Full Graph RAG results
        for rank, result in enumerate(graph_results, 1):
            all_results.append([
                "Original Full Graph RAG",
                rank,
                result.get("score", 0),
                result.get("latency", 0),
                result.get("precision", 0),
                result.get("recall", 0),
                result.get("f1", 0)
            ])
            
        # Add LangGraph Full Graph RAG results
        for rank, result in enumerate(langgraph_results, 1):
            all_results.append([
                "LangGraph Full Graph RAG",
                rank,
                result.get("score", 0),
                result.get("latency", 0),
                result.get("precision", 0),
                result.get("recall", 0),
                result.get("f1", 0)
            ])

        # Print comparison table
        print("==================================================")
        print(f"All RAG Systems Comparison for Query: '{query}'")
        print("==================================================")
        headers = ["Method", "Rank", "Score", "Latency", "Precision", "Recall", "F-measure"]
        print(tabulate(all_results, headers=headers, tablefmt="grid"))
        print()

        # Calculate average metrics for each system
        system_metrics = {
            "Original Naive RAG": {
                "total_time": naive_total_time,
                "avg_latency": sum(r.get("latency", 0) for r in naive_results) / len(naive_results),
                "avg_precision": sum(r.get("precision", 0) for r in naive_results) / len(naive_results),
                "avg_recall": sum(r.get("recall", 0) for r in naive_results) / len(naive_results),
                "avg_f1": sum(r.get("f1", 0) for r in naive_results) / len(naive_results),
            },
            "LangChain Naive RAG": {
                "total_time": langchain_naive_total_time,
                "avg_latency": sum(r.get("latency", 0) for r in langchain_naive_results) / len(langchain_naive_results),
                "avg_precision": sum(r.get("precision", 0) for r in langchain_naive_results) / len(langchain_naive_results),
                "avg_recall": sum(r.get("recall", 0) for r in langchain_naive_results) / len(langchain_naive_results),
                "avg_f1": sum(r.get("f1", 0) for r in langchain_naive_results) / len(langchain_naive_results),
            },
            "Original Knowledge RAG": {
                "total_time": knowledge_total_time,
                "avg_latency": sum(r.get("latency", 0) for r in knowledge_results) / len(knowledge_results),
                "avg_precision": sum(r.get("precision", 0) for r in knowledge_results) / len(knowledge_results),
                "avg_recall": sum(r.get("recall", 0) for r in knowledge_results) / len(knowledge_results),
                "avg_f1": sum(r.get("f1", 0) for r in knowledge_results) / len(knowledge_results),
            },
            "LangChain Knowledge RAG": {
                "total_time": langchain_knowledge_total_time,
                "avg_latency": sum(r.get("latency", 0) for r in langchain_knowledge_results) / len(langchain_knowledge_results),
                "avg_precision": sum(r.get("precision", 0) for r in langchain_knowledge_results) / len(langchain_knowledge_results),
                "avg_recall": sum(r.get("recall", 0) for r in langchain_knowledge_results) / len(langchain_knowledge_results),
                "avg_f1": sum(r.get("f1", 0) for r in langchain_knowledge_results) / len(langchain_knowledge_results),
            },
            "Original Full Graph RAG": {
                "total_time": graph_total_time,
                "avg_latency": sum(r.get("latency", 0) for r in graph_results) / len(graph_results),
                "avg_precision": sum(r.get("precision", 0) for r in graph_results) / len(graph_results),
                "avg_recall": sum(r.get("recall", 0) for r in graph_results) / len(graph_results),
                "avg_f1": sum(r.get("f1", 0) for r in graph_results) / len(graph_results),
            },
            "LangGraph Full Graph RAG": {
                "total_time": langgraph_total_time,
                "avg_latency": sum(r.get("latency", 0) for r in langgraph_results) / len(langgraph_results),
                "avg_precision": sum(r.get("precision", 0) for r in langgraph_results) / len(langgraph_results),
                "avg_recall": sum(r.get("recall", 0) for r in langgraph_results) / len(langgraph_results),
                "avg_f1": sum(r.get("f1", 0) for r in langgraph_results) / len(langgraph_results),
            },
        }

        # Print overall metrics
        metrics_table = []
        for system, metrics in system_metrics.items():
            metrics_table.append([
                system,
                "-",
                metrics["total_time"],
                metrics["avg_latency"],
                metrics["avg_precision"],
                metrics["avg_recall"],
                metrics["avg_f1"]
            ])
            
        print("==================================================")
        print("Overall Processing Times and Average Metrics")
        print("==================================================")
        metrics_headers = ["Method", "Rank", "Total Time (s)", "Avg Latency", "Avg Precision", "Avg Recall", "Avg F-measure"]
        print(tabulate(metrics_table, headers=metrics_headers, tablefmt="grid"))
        print("\n")

    except Exception as e:
        print(f"Error: {str(e)}")
        raise e
    finally:
        # Close RAG systems
        try:
            naive_rag.close()
            knowledge_rag.close()
            graph_rag.close()
            langchain_naive_rag.close()
            langchain_knowledge_rag.close()
            langgraph_rag.close()
        except:
            pass
        
        # Close database connection
        if conn:
            conn.close()

if __name__ == "__main__":
    print("=================================POSTGRESQL DATABASE========================================================")
    
    test_queries = [
        "What is Image Forgery?",
        "What is artificial intelligence?"
    ]
    
    for query in test_queries:
        run_all_rags(query, k=2, verbose=True)
        print("\n" + "="*80 + "\n")
