from neo4j import GraphDatabase
import time
from data_manager import DataManager
from rouge_score import rouge_scorer
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

class NeoLangChainKnowledgeRAG:
    def __init__(self, driver=None):
        self.data_manager = DataManager(driver)
        self.driver = driver
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Initialize OpenAI embeddings instead of HuggingFace
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    def retrieve(self, query: str, k: int = 2, ground_truth: list = None):
        start_time = time.time()
        
        # Use get_similar_chunks instead of get_similar_chunks_with_kg
        # since the latter doesn't exist in DataManager
        similar_chunks = self.data_manager.get_similar_chunks(query, k)
            
        results = []
        for chunk in similar_chunks:
            # Calculate ROUGE scores if ground truth is available
            scores = self.calculate_rouge_scores(chunk['text'], ground_truth) if ground_truth else {
                'precision': 0,
                'recall': 0,
                'f1': 0
            }
            
            results.append({
                'text': chunk['text'],
                'score': chunk['score'],
                'latency': time.time() - start_time,
                **scores
            })
            
        return results
            
    def get_related_entities(self, query: str, k: int = 2):
        """
        Execute a Cypher query directly using the driver
        """
        cypher_query = """
        CALL db.index.vector.queryNodes('article_embeddings', $k, $query)
        YIELD node, score
        MATCH (node)-[r]->(related)
        RETURN node.title, related.title, TYPE(r), score
        LIMIT $limit
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query, {"query": query, "k": k, "limit": k*2})
                return list(result)
        except Exception as e:
            print(f"Error querying Neo4j: {e}")
            return []

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