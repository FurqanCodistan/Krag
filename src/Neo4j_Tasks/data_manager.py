from neo4j import GraphDatabase
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd

class DataManager:
    def __init__(self, driver=None):
        # Initialize Neo4j connection with database name
        self.driver = driver or GraphDatabase.driver("bolt://localhost:7687", 
                                                   auth=("neo4j", "password"))
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.initialize_database()
        self.setup_embeddings()
        self.setup_faiss_index()

    def close(self):
        self.driver.close()

    def check_database_exists(self, db_name):
        with self.driver.session() as session:
            result = session.run(f"""
                MATCH (n:{db_name})
                RETURN count(n) as count
            """)
            return result.single()['count'] > 0

    def create_database(self, db_name, data_path):
        try:
            df = pd.read_csv(f"data/{data_path}")
            
            # Map specific ID columns to 'id'
            id_column_mappings = {
                'Author': 'author_id',
                'Article': 'article_id',
                'Category': 'category_id',
                'Tag': 'tag_id'
            }
            
            # Rename ID column if needed
            if db_name in id_column_mappings and id_column_mappings[db_name] in df.columns:
                df = df.rename(columns={id_column_mappings[db_name]: 'id'})
            
            # Remove rows with null IDs
            df = df.dropna(subset=['id'])
            
            with self.driver.session() as session:
                # Create nodes with sanitized names and proper relationships
                if db_name == 'Article':
                    session.run("""
                        UNWIND $rows AS row
                        MERGE (n:Article {id: row.id})
                        SET n += row
                    """, rows=df.to_dict('records'))
                    
                    # Create relationships with categories
                    try:
                        categories_df = pd.read_csv("data/article_categories.csv")
                        categories_df = categories_df.dropna()
                        if not categories_df.empty:
                            session.run("""
                                UNWIND $rows AS row
                                MATCH (a:Article {id: row.article_id})
                                MATCH (c:Category {id: row.category_id})
                                MERGE (a)-[:BELONGS_TO]->(c)
                            """, rows=categories_df.to_dict('records'))
                    except Exception as e:
                        print(f"Warning: Could not create category relationships: {str(e)}")
                    
                    # Create relationships with tags
                    try:
                        tags_df = pd.read_csv("data/article_tags.csv")
                        tags_df = tags_df.dropna()
                        if not tags_df.empty:
                            session.run("""
                                UNWIND $rows AS row
                                MATCH (a:Article {id: row.article_id})
                                MATCH (t:Tag {id: row.tag_id})
                                MERGE (a)-[:HAS_TAG]->(t)
                            """, rows=tags_df.to_dict('records'))
                    except Exception as e:
                        print(f"Warning: Could not create tag relationships: {str(e)}")
                    
                    # Create relationships with authors
                    try:
                        authors_df = pd.read_csv("data/article_authors.csv")
                        authors_df = authors_df.dropna()
                        if not authors_df.empty:
                            session.run("""
                                UNWIND $rows AS row
                                MATCH (a:Article {id: row.article_id})
                                MATCH (auth:Author {id: row.author_id})
                                MERGE (a)-[:WRITTEN_BY]->(auth)
                            """, rows=authors_df.to_dict('records'))
                    except Exception as e:
                        print(f"Warning: Could not create author relationships: {str(e)}")
                else:
                    # For other node types, just create the nodes
                    session.run(f"""
                        UNWIND $rows AS row
                        MERGE (n:{db_name} {{id: row.id}})
                        SET n += row
                    """, rows=df.to_dict('records'))
                
            return True
        except Exception as e:
            print(f"Error creating {db_name} database: {str(e)}")
            return False

    def initialize_database(self):
        # Create constraints first
        with self.driver.session() as session:
            try:
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Article) REQUIRE a.id IS UNIQUE")
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.id IS UNIQUE")
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Category) REQUIRE c.id IS UNIQUE")
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Tag) REQUIRE t.id IS UNIQUE")
                session.run("CREATE INDEX IF NOT EXISTS FOR (a:Article) ON (a.content)")
            except Exception as e:
                print(f"Warning: Could not create all constraints: {str(e)}")

        # List of required databases and their corresponding CSV files
        databases = {
            'Author': 'authors.csv',  # Load authors first
            'Category': 'categories.csv',
            'Tag': 'tags.csv',
            'Article': 'articles.csv',  # Load articles last so relationships can be created
        }

        # Check and create each database if not exists
        for db_name, csv_file in databases.items():
            if not self.check_database_exists(db_name):
                print(f"Creating {db_name} database...")
                self.create_database(db_name, csv_file)

    def setup_embeddings(self):
        with self.driver.session() as session:
            # Check if ContentEmbeddings exists
            exists = session.run("""
                MATCH (e:ContentEmbedding)
                RETURN count(e) as count
            """).single()['count'] > 0

            if not exists:
                print("Creating content embeddings...")
                # Get all articles
                articles = session.run("""
                    MATCH (a:Article)
                    RETURN a.content as content
                """).data()

                # Create chunks and embeddings
                contents = [article['content'] for article in articles]
                chunks = self.create_chunks(contents)
                embeddings = self.model.encode(chunks)

                # Store embeddings in Neo4j
                session.run("""
                    UNWIND $embeddings as embedding
                    CREATE (e:ContentEmbedding)
                    SET e.vector = embedding.vector,
                        e.text = embedding.text
                """, embeddings=[{
                    'vector': emb.tolist(),
                    'text': chunk
                } for emb, chunk in zip(embeddings, chunks)])

    def setup_faiss_index(self):
        """Setup FAISS index for fast similarity search"""
        with self.driver.session() as session:
            # Get all embeddings
            result = session.run("""
                MATCH (e:ContentEmbedding)
                RETURN e.vector as vector, e.text as text
            """)
            
            embeddings = []
            self.texts = []
            for record in result:
                embeddings.append(record['vector'])
                self.texts.append(record['text'])
            
            if embeddings:
                embeddings = np.array(embeddings, dtype=np.float32)
                dimension = len(embeddings[0])
                self.index = faiss.IndexFlatIP(dimension)
                self.index.add(embeddings)

    def create_chunks(self, texts, chunk_size=512):
        chunks = []
        for text in texts:
            # Simple chunking by splitting on sentences and combining until chunk_size
            if isinstance(text, str):
                sentences = text.split('. ')
                current_chunk = []
                current_length = 0
                
                for sentence in sentences:
                    if current_length + len(sentence) > chunk_size:
                        chunks.append(' '.join(current_chunk))
                        current_chunk = [sentence]
                        current_length = len(sentence)
                    else:
                        current_chunk.append(sentence)
                        current_length += len(sentence)
                
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
        
        return chunks

    def get_all_data(self):
        with self.driver.session() as session:
            data = {
                'articles': session.run("MATCH (a:Article) RETURN a").data(),
                'categories': session.run("MATCH (c:Category) RETURN c").data(),
                'relationships': session.run("MATCH (r:Relationship) RETURN r").data(),
                'metadata': session.run("MATCH (m:Metadata) RETURN m").data(),
                'tags': session.run("MATCH (t:Tag) RETURN t").data(),
                'embeddings': session.run("MATCH (e:ContentEmbedding) RETURN e").data()
            }
            return data

    def get_similar_chunks(self, query, k=2):
        """Get similar chunks using FAISS index"""
        query_vector = self.model.encode([query])[0].astype(np.float32)
        
        if not hasattr(self, 'index'):
            return []
        
        # Search using FAISS
        scores, indices = self.index.search(query_vector.reshape(1, -1), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.texts):  # Make sure index is valid
                results.append({
                    'text': self.texts[idx],
                    'score': float(score)  # Convert numpy float to Python float
                })
        
        return results

    def get_graph_context(self, text):
        with self.driver.session() as session:
            result = session.run("""
                MATCH (a:Article {content: $text})-[r]->(related)
                RETURN type(r) as relation_type, related.content as related_content
                LIMIT 5
            """, text=text)
            
            context = []
            for record in result:
                context.append({
                    'type': record['relation_type'],
                    'content': record['related_content']
                })
            return context

    def get_ground_truth(self, query_text):
        with self.driver.session() as session:
            result = session.run("""
                MATCH (q:Query {text: $query_text})-[:HAS_ANSWER]->(a:Answer)
                RETURN a.text as answer_text
            """, query_text=query_text)
            
            return [record['answer_text'] for record in result]