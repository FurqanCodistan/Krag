from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List, Dict, Any
import time
from rouge_score import rouge_scorer
import networkx as nx
import numpy as np

class PostgresLangChainKnowledgeRAG:
    """LangChain implementation of Knowledge Graph RAG for PostgreSQL"""
    
    def __init__(self, conn=None):
        self.conn = conn
        self.cursor = conn.cursor() if conn else None
        # Use LangChain's HuggingFaceEmbeddings instead of SentenceTransformer
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.graph = nx.DiGraph()
        self._build_knowledge_graph()
        
    def _build_knowledge_graph(self):
        """Build knowledge graph with articles, metadata, and relationships"""
        # Get articles with metadata
        self.cursor.execute("""
            SELECT a.article_id, a.title, a.content, a.topic, a.tags,
                   string_agg(DISTINCT c.name, ', ') as categories,
                   string_agg(DISTINCT au.name, ', ') as authors
            FROM articles a
            LEFT JOIN article_categories ac ON a.article_id = ac.article_id
            LEFT JOIN categories c ON ac.category_id = c.category_id
            LEFT JOIN article_authors aa ON a.article_id = aa.article_id
            LEFT JOIN authors au ON aa.author_id = au.author_id
            GROUP BY a.article_id, a.title, a.content, a.topic, a.tags
        """)
        
        articles = {}
        for row in self.cursor.fetchall():
            article_id, title, content, topic, tags, categories, authors = row
            
            # Add article node with rich metadata
            articles[article_id] = {
                'title': title,
                'content': content,
                'topic': topic,
                'tags': tags.split(',') if tags else [],
                'categories': categories.split(', ') if categories else [],
                'authors': authors.split(', ') if authors else []
            }
            
            # Generate embeddings in batches using LangChain
            contents = [article['content'] for article in articles.values()]
            embeddings = self.embeddings.embed_documents(contents)
            
            # Add embeddings to article data
            for i, (article_id, article) in enumerate(articles.items()):
                article['embedding'] = embeddings[i]
                self.graph.add_node(
                    article_id,
                    **article
                )
        
        # Add relationships between articles
        self.cursor.execute("""
            SELECT source_article_id, target_article_id, 
                   relationship_type, strength
            FROM relationships
        """)
        for row in self.cursor.fetchall():
            source_id, target_id, rel_type, strength = row
            if source_id in articles and target_id in articles:
                self.graph.add_edge(
                    source_id,
                    target_id,
                    type=rel_type,
                    weight=strength
                )
                
    def retrieve(self, query: str, k: int = 2, ground_truth: list = None):
        """Hybrid retrieval combining LangChain embeddings, metadata, and graph relationships"""
        start_time = time.time()
        query_embedding = self.embeddings.embed_query(query)
        
        results = []
        for node_id, data in self.graph.nodes(data=True):
            # 1. Calculate base similarity score using cosine similarity
            content_embedding = data['embedding']
            base_similarity = self._cosine_similarity(query_embedding, content_embedding)
            
            # 2. Calculate metadata relevance score
            metadata_score = 0
            query_terms = set(query.lower().split())
            
            # Check metadata fields for relevance
            for tag in data['tags']:
                if any(term in tag.lower() for term in query_terms):
                    metadata_score += 0.1
            
            for category in data['categories']:
                if any(term in category.lower() for term in query_terms):
                    metadata_score += 0.15
                    
            for author in data['authors']:
                if any(term in author.lower() for term in query_terms):
                    metadata_score += 0.05
            
            # 3. Calculate graph relationship score
            neighbors = list(self.graph.neighbors(node_id))
            relationship_score = sum(
                self.graph[node_id][neighbor]['weight']
                for neighbor in neighbors
            ) / max(len(neighbors), 1)
            
            # Combine all scores with appropriate weights
            final_score = (
                0.5 * base_similarity +  # Base similarity using LangChain embeddings
                0.3 * metadata_score +   # Metadata relevance
                0.2 * relationship_score # Graph relationship strength
            )
            
            results.append({
                'text': data['content'],
                'score': final_score,
                'base_score': base_similarity,
                'metadata_score': metadata_score,
                'graph_score': relationship_score,
                'metadata': {
                    'title': data['title'],
                    'tags': data['tags'],
                    'categories': data['categories'],
                    'authors': data['authors']
                },
                'latency': time.time() - start_time
            })
        
        # Sort by final score and take top k
        results = sorted(results, key=lambda x: x['score'], reverse=True)[:k]
        
        # Calculate metrics if ground truth is provided
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