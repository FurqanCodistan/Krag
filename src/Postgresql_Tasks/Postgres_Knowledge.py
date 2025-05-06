from data_manager import PostgresDataManager
from rouge_score import rouge_scorer
import time
import networkx as nx
from sentence_transformers import SentenceTransformer, util
import torch

class PostgresKnowledgeRAG:
    def __init__(self, conn=None, model_name="all-MiniLM-L6-v2"):
        self.data_manager = PostgresDataManager(conn)
        self.graph = nx.DiGraph()
        self.model = SentenceTransformer(model_name)
        self._build_graph()

    def _build_graph(self):
        """Build comprehensive knowledge graph including articles, metadata, and relationships"""
        with self.data_manager.conn.cursor() as cur:
            # Get articles with metadata
            cur.execute("""
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
            
            for row in cur.fetchall():
                article_id, title, content, topic, tags, categories, authors = row
                # Add article node with rich metadata
                self.graph.add_node(
                    article_id,
                    title=title,
                    content=content,
                    topic=topic,
                    tags=tags.split(',') if tags else [],
                    categories=categories.split(', ') if categories else [],
                    authors=authors.split(', ') if authors else [],
                    embedding=self.model.encode(content)
                )
            
            # Add relationships between articles
            cur.execute("""
                SELECT source_article_id, target_article_id, 
                       relationship_type, strength
                FROM relationships
            """)
            for row in cur.fetchall():
                source_id, target_id, rel_type, strength = row
                self.graph.add_edge(
                    source_id,
                    target_id,
                    type=rel_type,
                    weight=strength
                )

    def retrieve(self, query: str, k: int = 2, ground_truth: list = None):
        """Hybrid retrieval combining naive RAG, metadata, and graph relationships"""
        start_time = time.time()
        query_embedding = torch.tensor(self.model.encode(query))
        
        results = []
        for node_id, data in self.graph.nodes(data=True):
            # 1. Calculate base similarity score (Naive RAG component)
            content_embedding = torch.tensor(data['embedding'])
            base_similarity = util.pytorch_cos_sim(
                query_embedding.unsqueeze(0),
                content_embedding.unsqueeze(0)
            ).item()
            
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
                0.5 * base_similarity +  # Base similarity from naive RAG
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

    def close(self):
        self.data_manager.close()