from falkordb import FalkorDB
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
import os

class FalkorDataManager:
    def __init__(self, conn=None):
        # Initialize database connection
        if conn:
            self.db = conn
        else:
            # Local FalkorDB connection settings
            host = "localhost"  # or '127.0.0.1'
            port = 6379        # Default Redis/FalkorDB port
            self.db = FalkorDB(host=host, port=port)
        
        self.graph = self.db.select_graph('knowledge_graph')
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.initialize_database()
        self.setup_embeddings()

    def close(self):
        """Close connections and cleanup"""
        self.db = None
        self.graph = None

    def initialize_database(self):
        """Initialize database and import data from CSV files"""
        try:
            # Clear existing data
            self.graph.query("MATCH (n) DETACH DELETE n")
            
            # Create indexes - FalkorDB will handle duplicate index errors
            try:
                self.graph.query("CREATE INDEX ON :Article(article_id)")
            except: pass
            try:
                self.graph.query("CREATE INDEX ON :Author(author_id)")
            except: pass
            try:
                self.graph.query("CREATE INDEX ON :Category(category_id)")
            except: pass
            
            # Import data in correct order
            tables_order = [
                ('articles', 'articles.csv'),
                ('authors', 'authors.csv'),
                ('categories', 'categories.csv'),
                ('article_authors', 'article_authors.csv'),
                ('article_categories', 'article_categories.csv'),
                ('relationships', 'relationships.csv')
            ]
            
            for table, csv_file in tables_order:
                try:
                    print(f"Importing {csv_file}...")
                    df = pd.read_csv(f"data/{csv_file}")
                    df = df.where(pd.notnull(df), None)
                    
                    if table == 'articles':
                        for _, row in df.iterrows():
                            self.graph.query("""
                                CREATE (a:Article {
                                    article_id: $id,
                                    title: $title,
                                    content: $content,
                                    topic: $topic,
                                    publication_date: $pub_date,
                                    tags: $tags
                                })
                            """, {
                                'id': int(row['article_id']),
                                'title': row['title'],
                                'content': row['content'],
                                'topic': row['topic'],
                                'pub_date': row['publication_date'],
                                'tags': row['tags'].split(',') if row['tags'] else []
                            })
                    
                    elif table == 'authors':
                        for _, row in df.iterrows():
                            self.graph.query("""
                                CREATE (a:Author {
                                    author_id: $id,
                                    name: $name,
                                    email: $email,
                                    bio: $bio,
                                    expertise: $expertise,
                                    join_date: $join_date,
                                    rating: $rating
                                })
                            """, {
                                'id': int(row['author_id']),
                                'name': row['name'],
                                'email': row['email'],
                                'bio': row['bio'],
                                'expertise': row['expertise'],
                                'join_date': row['join_date'],
                                'rating': float(row['rating']) if row['rating'] else 0.0
                            })
                    
                    elif table == 'categories':
                        for _, row in df.iterrows():
                            self.graph.query("""
                                CREATE (c:Category {
                                    category_id: $id,
                                    name: $name,
                                    description: $description,
                                    level: $level,
                                    is_active: $is_active
                                })
                            """, {
                                'id': int(row['category_id']),
                                'name': row['name'],
                                'description': row['description'],
                                'level': int(row['level']) if row['level'] else 0,
                                'is_active': bool(row['is_active'])
                            })
                    
                    elif table == 'article_authors':
                        for _, row in df.iterrows():
                            self.graph.query("""
                                MATCH (a:Article {article_id: $article_id})
                                MATCH (au:Author {author_id: $author_id})
                                CREATE (a)-[r:WRITTEN_BY {
                                    contribution_type: $contrib_type,
                                    contribution_percentage: $contrib_pct
                                }]->(au)
                            """, {
                                'article_id': int(row['article_id']),
                                'author_id': int(row['author_id']),
                                'contrib_type': row['contribution_type'],
                                'contrib_pct': int(row['contribution_percentage'])
                            })
                    
                    elif table == 'article_categories':
                        for _, row in df.iterrows():
                            self.graph.query("""
                                MATCH (a:Article {article_id: $article_id})
                                MATCH (c:Category {category_id: $category_id})
                                CREATE (a)-[r:BELONGS_TO {
                                    primary_category: $is_primary,
                                    confidence_score: $conf_score
                                }]->(c)
                            """, {
                                'article_id': int(row['article_id']),
                                'category_id': int(row['category_id']),
                                'is_primary': bool(row['primary_category']),
                                'conf_score': float(row['confidence_score'])
                            })
                    
                    elif table == 'relationships':
                        for _, row in df.iterrows():
                            self.graph.query("""
                                MATCH (a1:Article {article_id: $source_id})
                                MATCH (a2:Article {article_id: $target_id})
                                CREATE (a1)-[r:RELATED_TO {
                                    type: $rel_type,
                                    strength: $strength,
                                    created_date: $created_date
                                }]->(a2)
                            """, {
                                'source_id': int(row['source_article_id']),
                                'target_id': int(row['target_article_id']),
                                'rel_type': row['relationship_type'],
                                'strength': float(row['strength']),
                                'created_date': row['created_date']
                            })
                    
                    print(f"Successfully imported {csv_file}")
                except Exception as e:
                    print(f"Error importing {csv_file}: {str(e)}")
                    raise e
        except Exception as e:
            print(f"Error in initialize_database: {str(e)}")
            raise e

    def setup_embeddings(self):
        """Create and store embeddings for article content"""
        try:
            # Check if embeddings exist
            result = self.graph.query("MATCH (e:Embedding) RETURN count(e) as count")
            count = 0
            if hasattr(result, 'result_set') and len(result.result_set) > 0:
                count = result.result_set[0][0]
            
            if count == 0:
                print("Creating content embeddings...")
                
                # Get all articles
                articles = self.graph.query("MATCH (a:Article) RETURN a.article_id, a.content")
                
                if hasattr(articles, 'result_set'):
                    # Create embeddings
                    for article_data in articles.result_set:
                        article_id = article_data[0]
                        content = article_data[1]
                        if content:
                            # Create embedding
                            embedding = self.model.encode(content)
                            
                            # Store the embedding
                            self.graph.query("""
                                MATCH (a:Article {article_id: $article_id})
                                CREATE (e:Embedding {
                                    content: $content,
                                    embedding: $embedding
                                })-[:BELONGS_TO]->(a)
                            """, {
                                'article_id': article_id,
                                'content': content,
                                'embedding': embedding.tolist()
                            })
        except Exception as e:
            print(f"Error in setup_embeddings: {str(e)}")

    def get_similar_content(self, query, k=2):
        """Get similar content using cosine similarity"""
        try:
            # Encode the query
            query_embedding = torch.tensor(self.model.encode(query))
            
            # Get all embeddings from the database
            results = self.graph.query("""
                MATCH (e:Embedding)-[:BELONGS_TO]->(a:Article)
                RETURN e.content as content, e.embedding as embedding, 
                       a.title as title, a.topic as topic
            """)
            
            if hasattr(results, 'result_set'):
                similarities = []
                for result in results.result_set:
                    content = result[0]
                    embedding = torch.tensor(result[1])
                    title = result[2]
                    topic = result[3]
                    
                    # Calculate cosine similarity
                    similarity = torch.nn.functional.cosine_similarity(
                        query_embedding.unsqueeze(0),
                        embedding.unsqueeze(0)
                    ).item()
                    
                    similarities.append({
                        'text': content,
                        'title': title,
                        'topic': topic,
                        'score': similarity
                    })
                
                # Sort by similarity score and return top k
                similarities.sort(key=lambda x: x['score'], reverse=True)
                return similarities[:k]
            
            return []
        except Exception as e:
            print(f"Error in get_similar_content: {str(e)}")
            return []

    def get_graph_context(self, text):
        """Get graph context for a given text"""
        try:
            # First find the article with the given content
            article_result = self.graph.query("""
                MATCH (a:Article)
                WHERE a.content = $text
                RETURN a.article_id as id
            """, {'text': text})
            
            if not hasattr(article_result, 'result_set') or not article_result.result_set:
                return []
                
            article_id = article_result.result_set[0][0]
            
            # Then get all relationships and related articles
            result = self.graph.query("""
                MATCH (a1:Article {article_id: $id})-[r]->(a2)
                RETURN type(r) as relationship_type,
                       a2.content as content,
                       a2.title as title,
                       labels(a2)[0] as node_type
            """, {'id': article_id})
            
            context = []
            if hasattr(result, 'result_set'):
                for record in result.result_set:
                    context.append({
                        'type': record[0],
                        'content': record[1],
                        'title': record[2],
                        'node_type': record[3]
                    })
            return context
        except Exception as e:
            print(f"Error in get_graph_context: {str(e)}")
            return []

    def get_ground_truth(self, query_text):
        """Get ground truth for a given query"""
        queries_df = pd.read_csv(r"data\\queries.csv")
        return queries_df.loc[queries_df['query_text'] == query_text, 'ground_truth'].tolist()
        
    def get_all_articles(self):
        """Get all articles from the database"""
        try:
            results = self.graph.query("""
                MATCH (a:Article)
                RETURN a.article_id as article_id, a.title as title, 
                       a.content as content, a.topic as topic, 
                       a.publication_date as publication_date, a.tags as tags
            """)
            
            if not hasattr(results, 'result_set'):
                return []
                
            articles = []
            for record in results.result_set:
                articles.append({
                    'article_id': record[0],
                    'title': record[1],
                    'content': record[2],
                    'topic': record[3],
                    'publication_date': record[4],
                    'tags': record[5]
                })
            return articles
        except Exception as e:
            print(f"Error in get_all_articles: {str(e)}")
            return []
            
    def get_all_authors(self):
        """Get all authors from the database"""
        try:
            results = self.graph.query("""
                MATCH (a:Author)
                RETURN a.author_id as author_id, a.name as name, 
                       a.email as email, a.bio as bio, 
                       a.expertise as expertise, a.rating as rating
            """)
            
            if not hasattr(results, 'result_set'):
                return []
                
            authors = []
            for record in results.result_set:
                authors.append({
                    'author_id': record[0],
                    'name': record[1],
                    'email': record[2],
                    'bio': record[3],
                    'expertise': record[4],
                    'rating': record[5]
                })
            return authors
        except Exception as e:
            print(f"Error in get_all_authors: {str(e)}")
            return []
            
    def get_all_categories(self):
        """Get all categories from the database"""
        try:
            results = self.graph.query("""
                MATCH (c:Category)
                RETURN c.category_id as category_id, c.name as name, 
                       c.description as description, c.level as level, 
                       c.is_active as is_active
            """)
            
            if not hasattr(results, 'result_set'):
                return []
                
            categories = []
            for record in results.result_set:
                categories.append({
                    'category_id': record[0],
                    'name': record[1],
                    'description': record[2],
                    'level': record[3],
                    'is_active': record[4]
                })
            return categories
        except Exception as e:
            print(f"Error in get_all_categories: {str(e)}")
            return []
            
    def get_article_authors(self):
        """Get all article-author relationships"""
        try:
            results = self.graph.query("""
                MATCH (a:Article)-[r:WRITTEN_BY]->(au:Author)
                RETURN a.article_id as article_id, au.author_id as author_id,
                       r.contribution_type as contribution_type,
                       r.contribution_percentage as contribution_percentage
            """)
            
            if not hasattr(results, 'result_set'):
                return []
                
            article_authors = []
            for record in results.result_set:
                article_authors.append({
                    'article_id': record[0],
                    'author_id': record[1],
                    'contribution_type': record[2],
                    'contribution_percentage': record[3]
                })
            return article_authors
        except Exception as e:
            print(f"Error in get_article_authors: {str(e)}")
            return []
            
    def get_article_categories(self):
        """Get all article-category relationships"""
        try:
            results = self.graph.query("""
                MATCH (a:Article)-[r:BELONGS_TO]->(c:Category)
                RETURN a.article_id as article_id, c.category_id as category_id,
                       r.primary_category as is_primary,
                       r.confidence_score as confidence_score
            """)
            
            if not hasattr(results, 'result_set'):
                return []
                
            article_categories = []
            for record in results.result_set:
                article_categories.append({
                    'article_id': record[0],
                    'category_id': record[1],
                    'is_primary': record[2],
                    'confidence_score': record[3]
                })
            return article_categories
        except Exception as e:
            print(f"Error in get_article_categories: {str(e)}")
            return []
            
    def get_article_relationships(self):
        """Get all article-article relationships"""
        try:
            results = self.graph.query("""
                MATCH (a1:Article)-[r:RELATED_TO]->(a2:Article)
                RETURN a1.article_id as source_article_id, 
                       a2.article_id as target_article_id,
                       r.type as relationship_type,
                       r.strength as strength,
                       r.created_date as created_date
            """)
            
            if not hasattr(results, 'result_set'):
                return []
                
            relationships = []
            for record in results.result_set:
                relationships.append({
                    'source_article_id': record[0],
                    'target_article_id': record[1],
                    'relationship_type': record[2],
                    'strength': record[3],
                    'created_date': record[4]
                })
            return relationships
        except Exception as e:
            print(f"Error in get_article_relationships: {str(e)}")
            return []