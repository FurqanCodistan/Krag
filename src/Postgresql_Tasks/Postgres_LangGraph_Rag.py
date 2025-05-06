from langchain_community.embeddings import HuggingFaceEmbeddings
import langgraph.graph as lg
from typing import List, Dict, Any
import time
from rouge_score import rouge_scorer
import networkx as nx
import numpy as np

class PostgresLangGraphRAG:
    """LangGraph implementation of Full Graph RAG for PostgreSQL"""
    
    def __init__(self, conn=None):
        self.conn = conn
        self.cursor = conn.cursor() if conn else None
        # Use LangChain's HuggingFaceEmbeddings instead of SentenceTransformer
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.graph = nx.DiGraph()
        self._build_full_graph()
        
    def _build_full_graph(self):
        """Build full knowledge graph including articles, authors, and categories"""
        # Add article nodes
        self.cursor.execute("""
            SELECT article_id, title, content, topic, publication_date, 
                   author_id, author_name, category_id, tags
            FROM articles
        """)
        articles = {}
        for row in self.cursor.fetchall():
            article_id, title, content, topic, pub_date, author_id, author_name, category_id, tags = row
            articles[article_id] = {
                'title': title,
                'content': content,
                'topic': topic,
                'publication_date': pub_date,
                'author_id': author_id,
                'author_name': author_name,
                'category_id': category_id,
                'tags': tags if tags else ""
            }
        
        # Generate embeddings for all articles in batch using LangChain
        contents = [article['content'] for article in articles.values()]
        embeddings = self.embeddings.embed_documents(contents)
        
        # Add nodes with embeddings
        for idx, (article_id, article) in enumerate(articles.items()):
            self.graph.add_node(
                f"article_{article_id}",
                type='article',
                title=article['title'],
                content=article['content'],
                topic=article['topic'],
                publication_date=article['publication_date'],
                author_id=article['author_id'],
                author_name=article['author_name'],
                category_id=article['category_id'],
                tags=article['tags'],
                embedding=embeddings[idx]
            )
        
        # Add author nodes and edges with contribution info
        self.cursor.execute("""
            SELECT a.author_id, a.name, a.expertise, a.rating,
                   aa.article_id, aa.contribution_type, aa.contribution_percentage
            FROM authors a
            JOIN article_authors aa ON a.author_id = aa.author_id
        """)
        for row in self.cursor.fetchall():
            author_id, name, expertise, rating, article_id, contrib_type, contrib_pct = row
            author_node = f"author_{author_id}"
            article_node = f"article_{article_id}"
            
            if not self.graph.has_node(author_node):
                self.graph.add_node(
                    author_node,
                    type='author',
                    name=name,
                    expertise=expertise,
                    rating=rating
                )
            
            if self.graph.has_node(article_node):
                self.graph.add_edge(
                    article_node,
                    author_node,
                    type='written_by',
                    contribution_type=contrib_type,
                    contribution_percentage=contrib_pct
                )
        
        # Add category nodes and edges with confidence scores
        self.cursor.execute("""
            SELECT c.category_id, c.name, c.description, c.level, c.is_active,
                   ac.article_id, ac.primary_category, ac.confidence_score
            FROM categories c
            JOIN article_categories ac ON c.category_id = ac.category_id
        """)
        for row in self.cursor.fetchall():
            cat_id, name, desc, level, is_active, article_id, is_primary, conf_score = row
            category_node = f"category_{cat_id}"
            article_node = f"article_{article_id}"
            
            if not self.graph.has_node(category_node):
                self.graph.add_node(
                    category_node,
                    type='category',
                    name=name,
                    description=desc,
                    level=level,
                    is_active=is_active
                )
            
            if self.graph.has_node(article_node):
                self.graph.add_edge(
                    article_node,
                    category_node,
                    type='belongs_to',
                    is_primary=is_primary,
                    confidence_score=conf_score
                )
        
        # Add article relationships with temporal information
        self.cursor.execute("""
            SELECT source_article_id, target_article_id, 
                   relationship_type, strength, created_date
            FROM relationships
        """)
        for row in self.cursor.fetchall():
            source_id, target_id, rel_type, strength, created_date = row
            source_node = f"article_{source_id}"
            target_node = f"article_{target_id}"
            
            if self.graph.has_node(source_node) and self.graph.has_node(target_node):
                self.graph.add_edge(
                    source_node,
                    target_node,
                    type=rel_type,
                    weight=strength,
                    created_date=created_date
                )
    
    def _retrieve_with_embeddings(self, query: str, k: int) -> List[Dict]:
        """First step: retrieve documents using embedding similarity"""
        query_embedding = self.embeddings.embed_query(query)
        
        # Get similarity scores for all articles
        article_similarities = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get('type') == 'article' and 'embedding' in data:
                content_embedding = data['embedding']
                similarity = self._cosine_similarity(query_embedding, content_embedding)
                article_similarities.append((node_id, data, similarity))
        
        # Sort by similarity and take top k*2 (we'll narrow down further with graph analysis)
        article_similarities.sort(key=lambda x: x[2], reverse=True)
        
        # Return top candidates
        return [{
            'node_id': node_id,
            'content': data['content'],
            'title': data['title'],
            'base_score': similarity
        } for node_id, data, similarity in article_similarities[:k*2]]
    
    def _enhance_with_graph(self, candidates: List[Dict], k: int) -> List[Dict]:
        """Second step: enhance candidates with graph relationships"""
        for candidate in candidates:
            node_id = candidate['node_id']
            
            # Get connected nodes
            connected_nodes = []
            for _, neighbor, edge_data in self.graph.edges(node_id, data=True):
                neighbor_data = self.graph.nodes[neighbor]
                connected_nodes.append({
                    'type': neighbor_data['type'],
                    'name': neighbor_data.get('name', ''),
                    'relationship': edge_data['type']
                })
            
            # Calculate graph enhancement
            # More connections = higher boost, up to 50%
            connection_boost = min(len(connected_nodes) * 0.1, 0.5)
            
            # Store graph-enhanced score
            candidate['graph_boost'] = connection_boost
            candidate['connected_nodes'] = connected_nodes
            candidate['final_score'] = candidate['base_score'] * (1 + connection_boost)
        
        # Sort by final score and limit to k
        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        return candidates[:k]
    
    def retrieve(self, query: str, k: int = 2, ground_truth: list = None):
        """Retrieve documents using LangGraph workflow"""
        start_time = time.time()
        
        # Create dictionary to hold the workflow results
        workflow = {}
        
        # First step: retrieve candidates
        candidates = self._retrieve_with_embeddings(query, k)
        
        # Second step: enhance with graph
        enhanced_results = self._enhance_with_graph(candidates, k)
        
        # Format results
        final_results = []
        for idx, item in enumerate(enhanced_results):
            result_obj = {
                'text': item['content'],
                'score': item['final_score'],
                'base_score': item['base_score'],
                'graph_boost': item['graph_boost'],
                'connected_nodes': item['connected_nodes'],
                'latency': time.time() - start_time
            }
            
            # Calculate metrics if ground truth is provided
            if ground_truth:
                scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
                rouge_scores = [scorer.score(item["content"], gt) for gt in ground_truth]
                best_score = max(rouge_scores, key=lambda x: x['rougeL'].fmeasure)
                result_obj['precision'] = best_score['rougeL'].precision
                result_obj['recall'] = best_score['rougeL'].recall
                result_obj['f1'] = best_score['rougeL'].fmeasure
            else:
                result_obj['precision'] = result_obj['recall'] = result_obj['f1'] = 0.0
                
            final_results.append(result_obj)
            
        return final_results
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2)
    
    def close(self):
        if self.cursor:
            self.cursor.close()