from data_manager import PostgresDataManager
from sentence_transformers import SentenceTransformer, util
import networkx as nx
import matplotlib.pyplot as plt
import time
from rouge_score import rouge_scorer
import torch

class PostgresGraphRAG:
    def __init__(self, conn=None, model_name="all-MiniLM-L6-v2"):
        self.data_manager = PostgresDataManager(conn)
        self.model = SentenceTransformer(model_name)
        self.graph = nx.DiGraph()
        self._build_full_graph()

    def _build_full_graph(self):
        """Build full knowledge graph including articles, authors, and categories"""
        with self.data_manager.conn.cursor() as cur:
            # Add article nodes
            cur.execute("""
                SELECT article_id, title, content, topic, publication_date, 
                       author_id, author_name, category_id, tags
                FROM articles
            """)
            for row in cur.fetchall():
                article_id, title, content, topic, pub_date, author_id, author_name, category_id, tags = row
                embedding = self.model.encode(content, convert_to_tensor=True)
                self.graph.add_node(
                    f"article_{article_id}",
                    type='article',
                    title=title,
                    content=content,
                    topic=topic,
                    publication_date=pub_date,
                    author_id=author_id,
                    author_name=author_name,
                    category_id=category_id,
                    tags=tags,
                    embedding=embedding
                )
            
            # Add author nodes and edges with contribution info
            cur.execute("""
                SELECT a.author_id, a.name, a.expertise, a.rating,
                       aa.article_id, aa.contribution_type, aa.contribution_percentage
                FROM authors a
                JOIN article_authors aa ON a.author_id = aa.author_id
            """)
            for row in cur.fetchall():
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
                
                self.graph.add_edge(
                    article_node,
                    author_node,
                    type='written_by',
                    contribution_type=contrib_type,
                    contribution_percentage=contrib_pct
                )
            
            # Add category nodes and edges with confidence scores
            cur.execute("""
                SELECT c.category_id, c.name, c.description, c.level, c.is_active,
                       ac.article_id, ac.primary_category, ac.confidence_score
                FROM categories c
                JOIN article_categories ac ON c.category_id = ac.category_id
            """)
            for row in cur.fetchall():
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
                
                self.graph.add_edge(
                    article_node,
                    category_node,
                    type='belongs_to',
                    is_primary=is_primary,
                    confidence_score=conf_score
                )
            
            # Add article relationships with temporal information
            cur.execute("""
                SELECT source_article_id, target_article_id, 
                       relationship_type, strength, created_date
                FROM relationships
            """)
            for row in cur.fetchall():
                source_id, target_id, rel_type, strength, created_date = row
                self.graph.add_edge(
                    f"article_{source_id}",
                    f"article_{target_id}",
                    type=rel_type,
                    weight=strength,
                    created_date=created_date
                )

    def retrieve(self, query: str, k: int = 2, ground_truth: list = None):
        """Retrieve documents using full graph-enhanced search"""
        start_time = time.time()
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        results = []

        # Search through article nodes
        for node_id, data in self.graph.nodes(data=True):
            if data.get('type') == 'article':
                # Calculate similarity
                content_embedding = data.get('embedding')
                if content_embedding is not None:
                    similarity = util.pytorch_cos_sim(
                        query_embedding.unsqueeze(0),
                        content_embedding.unsqueeze(0)
                    ).item()

                    # Get connected nodes
                    connected_nodes = []
                    for _, neighbor, edge_data in self.graph.edges(node_id, data=True):
                        neighbor_data = self.graph.nodes[neighbor]
                        connected_nodes.append({
                            'type': neighbor_data['type'],
                            'name': neighbor_data.get('name', ''),
                            'relationship': edge_data['type']
                        })

                    # Calculate enhanced score
                    base_score = similarity
                    # More connections = higher boost, up to 50%
                    connection_boost = min(len(connected_nodes) * 0.1, 0.5)
                    final_score = base_score * (1 + connection_boost)

                    results.append({
                        'text': data['content'],
                        'title': data['title'],
                        'score': final_score,
                        'base_score': base_score,
                        'graph_boost': connection_boost,
                        'connected_nodes': connected_nodes,
                        'latency': time.time() - start_time
                    })

        # Sort by enhanced score and take top k
        results = sorted(results, key=lambda x: x['score'], reverse=True)[:k]
        
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

    def visualize(self, output_file='postgres_graph_visualization.png'):
        """Visualize the full knowledge graph"""
        plt.figure(figsize=(20, 20))
        pos = nx.spring_layout(self.graph, k=2, iterations=50)

        # Draw nodes for each type with different colors
        node_colors = {
            'article': 'skyblue',
            'author': 'lightgreen',
            'category': 'orange'
        }

        for node_type, color in node_colors.items():
            nodes = [n for n, d in self.graph.nodes(data=True) 
                    if d.get('type') == node_type]
            if nodes:
                nx.draw_networkx_nodes(
                    self.graph, pos,
                    nodelist=nodes,
                    node_color=color,
                    node_size=2000,
                    label=node_type
                )

        # Draw edges with labels
        nx.draw_networkx_edges(self.graph, pos, edge_color='gray', arrows=True)
        edge_labels = nx.get_edge_attributes(self.graph, 'type')
        nx.draw_networkx_edge_labels(
            self.graph, pos,
            edge_labels=edge_labels,
            font_size=8
        )

        # Draw node labels
        labels = {}
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            if node_data['type'] == 'article':
                labels[node] = f"{node}\n{node_data.get('title', '')[:20]}..."
            else:
                labels[node] = f"{node}\n{node_data.get('name', '')}"

        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)

        plt.title("PostgreSQL Knowledge Graph Visualization")
        plt.legend()
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, format='PNG', dpi=300, bbox_inches='tight')
        plt.close()

    def close(self):
        self.data_manager.close()