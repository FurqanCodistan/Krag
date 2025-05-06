from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List, Dict, Any
import time
from rouge_score import rouge_scorer
import numpy as np

class PostgresLangChainNaiveRAG:
    """LangChain implementation of Naive RAG for PostgreSQL"""
    
    def __init__(self, conn=None):
        self.conn = conn
        # Use LangChain's HuggingFaceEmbeddings instead of SentenceTransformer
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Create cursor for database operations
        self.cursor = conn.cursor() if conn else None
        
    def retrieve(self, query: str, k: int = 2, ground_truth: list = None):
        """Retrieve documents using LangChain's embedding and cosine similarity"""
        start_time = time.time()
        
        # Get text chunks from database
        self.cursor.execute("SELECT chunk_id, content FROM content_embeddings")
        chunks = self.cursor.fetchall()
        
        # Create documents from chunks
        documents = [Document(page_content=content, metadata={"chunk_id": chunk_id}) 
                     for chunk_id, content in chunks]
        
        # Embed the query using LangChain
        query_embedding = self.embeddings.embed_query(query)
        
        # Embed all documents using LangChain
        doc_embeddings = self.embeddings.embed_documents([doc.page_content for doc in documents])
        
        # Calculate cosine similarity
        similarities = []
        for i, doc_embedding in enumerate(doc_embeddings):
            # Compute cosine similarity
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((similarity, documents[i]))
        
        # Sort by similarity
        similarities.sort(reverse=True)
        
        # Take top k results
        results = []
        for similarity, doc in similarities[:k]:
            results.append({
                "text": doc.page_content,
                "score": similarity,
                "metadata": doc.metadata
            })
        
        # Add latency to results
        latency = time.time() - start_time
        for result in results:
            result['latency'] = latency
        
        # Calculate metrics using provided ground truth
        if ground_truth:
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            for result in results:
                rouge_scores = [scorer.score(result["text"], gt) for gt in ground_truth]
                best_score = max(rouge_scores, key=lambda x: x['rougeL'].fmeasure)
                result['precision'] = best_score['rougeL'].precision
                result['recall'] = best_score['rougeL'].recall
                result['f1'] = best_score['rougeL'].fmeasure
        else:
            for result in results:
                result['precision'] = result['recall'] = result['f1'] = 0.0
                
        return results
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2)
        
    def close(self):
        if self.cursor:
            self.cursor.close()