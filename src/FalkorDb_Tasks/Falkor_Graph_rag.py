from data_manager import FalkorDataManager
from sentence_transformers import SentenceTransformer, util
import time
from rouge_score import rouge_scorer
import torch
import networkx as nx
import matplotlib.pyplot as plt

class FalkorGraphRAG:
    def __init__(self, data_manager=None):
        """Initialize with existing data manager or create new one"""
        if isinstance(data_manager, FalkorDataManager):
            self.data_manager = data_manager
            self._shared_data_manager = True
        else:
            self.data_manager = FalkorDataManager(data_manager)  # data_manager here would be the connection
            self._shared_data_manager = False
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def retrieve(self, query: str, k: int = 2, ground_truth: list = None):
        """Retrieve documents using full graph-enhanced search"""
        start_time = time.time()
        
        # Get base results using cosine similarity
        base_results = self.data_manager.get_similar_content(query, k)
        
        # Enhance results with graph analysis
        enhanced_results = []
        for result in base_results:
            # Get graph relationships
            graph_context = self.data_manager.get_graph_context(result['text'])
            
            # Calculate graph-based boost
            relationship_boost = sum(
                0.1 for _ in graph_context  # Each relationship adds a 10% boost
            )
            
            # Combine scores
            final_score = result['score'] * (1 + min(relationship_boost, 0.5))
            
            enhanced_result = {
                'text': result['text'],
                'title': result.get('title', ''),
                'topic': result.get('topic', ''),
                'score': final_score,
                'base_score': result['score'],
                'graph_boost': relationship_boost,
                'graph_context': graph_context,
                'latency': time.time() - start_time
            }
            
            # Calculate metrics if ground truth is provided
            if ground_truth:
                scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
                rouge_scores = [scorer.score(enhanced_result["text"], gt) for gt in ground_truth]
                best_score = max(rouge_scores, key=lambda x: x['rougeL'].fmeasure)
                enhanced_result['precision'] = best_score['rougeL'].precision
                enhanced_result['recall'] = best_score['rougeL'].recall
                enhanced_result['f1'] = best_score['rougeL'].fmeasure
            else:
                enhanced_result['precision'] = enhanced_result['recall'] = enhanced_result['f1'] = 0.0
            
            enhanced_results.append(enhanced_result)
        
        # Sort by final score
        enhanced_results.sort(key=lambda x: x['score'], reverse=True)
        return enhanced_results[:k]

    def get_node_id(self, node, node_type):
        """Get appropriate ID from a node based on its type"""
        try:
            if node_type == 'Article':
                return getattr(node, 'article_id', None)
            elif node_type == 'Author':
                return getattr(node, 'author_id', None)
            elif node_type == 'Category':
                return getattr(node, 'category_id', None)
            return None
        except Exception:
            return None

    def get_node_label(self, node, node_type):
        """Get appropriate label from a node based on its type"""
        try:
            if node_type == 'Article':
                return str(getattr(node, 'title', ''))[:20]
            elif node_type in ['Author', 'Category']:
                return str(getattr(node, 'name', ''))[:20]
            return ''
        except Exception:
            return ''

    def visualize(self, output_file='falkor_graph_visualization.png'):
        """Visualize the knowledge graph"""
        plt.figure(figsize=(20, 20))
        G = nx.DiGraph()
        
        try:
            # Get all nodes and relationships with a single query
            results = self.data_manager.graph.query("""
                MATCH (n)
                OPTIONAL MATCH (n)-[r]->(m)
                RETURN DISTINCT
                    n.article_id as source_id,
                    n.author_id as source_author_id,
                    n.category_id as source_category_id,
                    n.title as source_title,
                    n.name as source_name,
                    labels(n)[0] as source_type,
                    type(r) as rel_type,
                    m.article_id as target_id,
                    m.author_id as target_author_id,
                    m.category_id as target_category_id,
                    m.title as target_title,
                    m.name as target_name,
                    labels(m)[0] as target_type
            """)
            
            if not hasattr(results, 'result_set'):
                print("No results found for visualization")
                return
                
            node_colors = {
                'Article': 'skyblue',
                'Author': 'lightgreen',
                'Category': 'orange'
            }

            nodes_added = set()
            edges = []

            for record in results.result_set:
                try:
                    # Get source node info
                    source_type = record[5]
                    source_id = record[0] or record[1] or record[2]  # Use appropriate ID based on node type
                    source_label = record[3] or record[4]  # Use title or name as label
                    
                    # Add source node if not already added
                    if source_id and source_type:
                        full_source_id = f"{source_type}_{source_id}"
                        if full_source_id not in nodes_added:
                            G.add_node(
                                full_source_id,
                                type=source_type,
                                label=f"{source_type}\n{source_label if source_label else 'Unknown'}"
                            )
                            nodes_added.add(full_source_id)

                        # Process target node and relationship if they exist
                        rel_type = record[6]
                        target_type = record[12]
                        target_id = record[7] or record[8] or record[9]  # Use appropriate ID based on node type
                        target_label = record[10] or record[11]  # Use title or name as label

                        if target_id and target_type and rel_type:
                            full_target_id = f"{target_type}_{target_id}"
                            if full_target_id not in nodes_added:
                                G.add_node(
                                    full_target_id,
                                    type=target_type,
                                    label=f"{target_type}\n{target_label if target_label else 'Unknown'}"
                                )
                                nodes_added.add(full_target_id)

                            edges.append((
                                full_source_id,
                                full_target_id,
                                {'type': rel_type}
                            ))

                except Exception as e:
                    print(f"Error processing node: {str(e)}")
                    continue

            G.add_edges_from(edges)

            # Draw the graph if we have nodes
            if len(G.nodes()) > 0:
                pos = nx.spring_layout(G, k=2, iterations=50)

                # Draw nodes by type
                for node_type, color in node_colors.items():
                    nodes = [n for n, d in G.nodes(data=True) if d.get('type') == node_type]
                    if nodes:
                        nx.draw_networkx_nodes(
                            G, pos,
                            nodelist=nodes,
                            node_color=color,
                            node_size=2000,
                            label=node_type
                        )

                # Draw edges and labels
                nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)
                edge_labels = nx.get_edge_attributes(G, 'type')
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

                # Draw node labels
                labels = nx.get_node_attributes(G, 'label')
                nx.draw_networkx_labels(G, pos, labels, font_size=8)

                plt.title("FalkorDB Knowledge Graph Visualization")
                plt.legend()
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(output_file, format='PNG', dpi=300, bbox_inches='tight')
                plt.close()
            else:
                print("No nodes found to visualize")
                
        except Exception as e:
            print(f"Error in visualization: {str(e)}")

    def close(self):
        """Only close if we created our own data manager"""
        if not hasattr(self, '_shared_data_manager') or not self._shared_data_manager:
            self.data_manager.close()