from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import faiss
import os

class PostgresDataManager:
    def __init__(self, conn=None):
        # Initialize database connection
        self.conn = conn or psycopg2.connect(
            dbname="Rag DB",
            user="postgres",
            password="12345",
            host="localhost",
            port="5432"
        )
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.initialize_database()
        self.setup_embeddings()
        self.setup_faiss_index()

    def close(self):
        if self.conn:
            self.conn.close()

    def initialize_database(self):
        """Initialize database tables and import data from CSV files"""
        with self.conn.cursor() as cur:
            # Drop existing tables in reverse order to handle dependencies
            cur.execute("""
                DROP TABLE IF EXISTS content_embeddings CASCADE;
                DROP TABLE IF EXISTS relationships CASCADE;
                DROP TABLE IF EXISTS article_categories CASCADE;
                DROP TABLE IF EXISTS article_authors CASCADE;
                DROP TABLE IF EXISTS articles CASCADE;
                DROP TABLE IF EXISTS authors CASCADE;
                DROP TABLE IF EXISTS categories CASCADE;
            """)
            
            self.conn.commit()
            
            # Create base tables first
            cur.execute("""
                CREATE TABLE articles (
                    article_id INTEGER PRIMARY KEY,
                    title TEXT,
                    content TEXT,
                    topic TEXT,
                    publication_date TEXT,
                    author_id INTEGER,
                    author_name TEXT,
                    category_id INTEGER,
                    tags TEXT,
                    ground_truth TEXT
                );

                CREATE TABLE authors (
                    author_id INTEGER PRIMARY KEY,
                    name TEXT,
                    email TEXT,
                    bio TEXT,
                    expertise TEXT,
                    join_date TEXT,
                    rating FLOAT
                );

                CREATE TABLE categories (
                    category_id INTEGER PRIMARY KEY,
                    name TEXT,
                    description TEXT,
                    parent_category_id FLOAT,
                    level INTEGER,
                    is_active BOOLEAN
                );
            """)
            
            # Create relation tables
            cur.execute("""
                CREATE TABLE article_authors (
                    article_id INTEGER,
                    author_id INTEGER,
                    contribution_type TEXT,
                    contribution_percentage INTEGER
                );

                CREATE TABLE article_categories (
                    article_id INTEGER,
                    category_id INTEGER,
                    primary_category BOOLEAN,
                    confidence_score FLOAT
                );

                CREATE TABLE relationships (
                    id SERIAL,
                    source_article_id INTEGER,
                    target_article_id INTEGER,
                    relationship_type TEXT,
                    strength FLOAT,
                    created_date TEXT
                );

                CREATE TABLE content_embeddings (
                    chunk_id SERIAL PRIMARY KEY,
                    article_id INTEGER,
                    content TEXT,
                    embedding FLOAT[],
                    chunk_index INTEGER
                );
            """)
            
            self.conn.commit()
            
            # Import data in correct order
            tables_order = [
                ('articles', 'articles.csv'),
                ('authors', 'authors.csv'),
                ('categories', 'categories.csv'),
                ('article_authors', 'article_authors.csv'),
                ('article_categories', 'article_categories.csv'),
                ('relationships', 'relationships.csv')
            ]
            
            # Import data
            for table, csv_file in tables_order:
                try:
                    # print(f"Importing {csv_file}...")
                    df = pd.read_csv(f"data/{csv_file}")
                    
                    # Clean up data based on table
                    if table in ['article_authors', 'article_categories', 'relationships']:
                        # Get valid article IDs
                        cur.execute("SELECT article_id FROM articles")
                        valid_article_ids = [row[0] for row in cur.fetchall()]
                        
                        if table in ['article_authors', 'article_categories']:
                            # Filter rows where article_id exists
                            df = df[df['article_id'].isin(valid_article_ids)]
                        elif table == 'relationships':
                            # Filter rows where both source and target articles exist
                            df = df[
                                df['source_article_id'].isin(valid_article_ids) & 
                                df['target_article_id'].isin(valid_article_ids)
                            ]
                            # Remove duplicate source-target pairs
                            df = df.drop_duplicates(['source_article_id', 'target_article_id'])
                    
                    # Replace NaN values with None
                    df = df.where(pd.notnull(df), None)
                    cols = df.columns.tolist()
                    vals = [tuple(row) for row in df.values]
                    
                    if vals:  # Only insert if we have valid data
                        execute_values(
                            cur,
                            f"INSERT INTO {table} ({','.join(cols)}) VALUES %s",
                            vals
                        )
                    
                    self.conn.commit()
                    # print(f"Successfully imported {csv_file}")
                except Exception as e:
                    print(f"Error importing {csv_file}: {str(e)}")
                    self.conn.rollback()
            
            # Add constraints after data import
            cur.execute("""
                -- Add primary keys
                ALTER TABLE article_authors 
                ADD PRIMARY KEY (article_id, author_id);
                
                ALTER TABLE article_categories 
                ADD PRIMARY KEY (article_id, category_id);
                
                ALTER TABLE relationships 
                ADD PRIMARY KEY (id);
                
                -- Add foreign key constraints
                ALTER TABLE article_authors
                ADD CONSTRAINT article_authors_article_id_fkey
                FOREIGN KEY (article_id) REFERENCES articles(article_id) ON DELETE CASCADE;
                
                ALTER TABLE article_authors
                ADD CONSTRAINT article_authors_author_id_fkey
                FOREIGN KEY (author_id) REFERENCES authors(author_id) ON DELETE CASCADE;
                
                ALTER TABLE article_categories
                ADD CONSTRAINT article_categories_article_id_fkey
                FOREIGN KEY (article_id) REFERENCES articles(article_id) ON DELETE CASCADE;
                
                ALTER TABLE article_categories
                ADD CONSTRAINT article_categories_category_id_fkey
                FOREIGN KEY (category_id) REFERENCES categories(category_id) ON DELETE CASCADE;
                
                ALTER TABLE relationships
                ADD CONSTRAINT relationships_source_article_id_fkey
                FOREIGN KEY (source_article_id) REFERENCES articles(article_id) ON DELETE CASCADE;
                
                ALTER TABLE relationships
                ADD CONSTRAINT relationships_target_article_id_fkey
                FOREIGN KEY (target_article_id) REFERENCES articles(article_id) ON DELETE CASCADE;
                
                ALTER TABLE content_embeddings
                ADD CONSTRAINT content_embeddings_article_id_fkey
                FOREIGN KEY (article_id) REFERENCES articles(article_id) ON DELETE CASCADE;
                
                -- Add index on relationships to speed up graph queries
                CREATE INDEX idx_relationships_source ON relationships(source_article_id);
                CREATE INDEX idx_relationships_target ON relationships(target_article_id);
            """)
            
            self.conn.commit()

    def setup_embeddings(self):
        """Create and store embeddings for article content"""
        with self.conn.cursor() as cur:
            # Check if embeddings exist
            cur.execute("SELECT COUNT(*) FROM content_embeddings")
            if cur.fetchone()[0] == 0:
                # print("Creating content embeddings...")
                
                # Get all articles
                cur.execute("SELECT article_id, content FROM articles")
                articles = cur.fetchall()
                
                # Create chunks and embeddings
                for article_id, content in articles:
                    chunks = self.create_chunks([content])
                    embeddings = self.model.encode(chunks)
                    
                    # Store each chunk and its embedding
                    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                        cur.execute("""
                            INSERT INTO content_embeddings 
                            (article_id, content, embedding, chunk_index)
                            VALUES (%s, %s, %s, %s)
                        """, (article_id, chunk, embedding.tolist(), idx))
                
                self.conn.commit()

    def setup_faiss_index(self):
        """Setup FAISS index for fast similarity search"""
        with self.conn.cursor() as cur:
            # Get all embeddings
            cur.execute("SELECT embedding, content FROM content_embeddings")
            results = cur.fetchall()
            
            if results:
                embeddings = []
                self.texts = []
                
                for embedding, text in results:
                    embeddings.append(embedding)
                    self.texts.append(text)
                
                embeddings = np.array(embeddings, dtype=np.float32)
                dimension = len(embeddings[0])
                self.index = faiss.IndexFlatIP(dimension)
                self.index.add(embeddings)

    def create_chunks(self, texts, chunk_size=512):
        """Split texts into chunks"""
        chunks = []
        for text in texts:
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
        """Get all data from PostgreSQL database"""
        with self.conn.cursor() as cur:
            data = {}
            
            # Get articles
            cur.execute("SELECT * FROM articles")
            data['articles'] = cur.fetchall()
            
            # Get categories
            cur.execute("SELECT * FROM categories")
            data['categories'] = cur.fetchall()
            
            # Get relationships
            cur.execute("SELECT * FROM relationships")
            data['relationships'] = cur.fetchall()
            
            # Get embeddings
            cur.execute("SELECT * FROM content_embeddings")
            data['embeddings'] = cur.fetchall()
            
            return data

    def get_similar_chunks(self, query, k=2):
        """Get similar chunks using FAISS index"""
        query_vector = self.model.encode([query])[0].astype(np.float32)
        
        if not hasattr(self, 'index'):
            return []
        
        scores, indices = self.index.search(query_vector.reshape(1, -1), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.texts):
                results.append({
                    'text': self.texts[idx],
                    'score': float(score)
                })
        
        return results

    def get_graph_context(self, text):
        """Get graph context for a given text"""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT r.relationship_type, a2.content
                FROM articles a1
                JOIN relationships r ON a1.article_id = r.source_article_id
                JOIN articles a2 ON r.target_article_id = a2.article_id
                WHERE a1.content = %s
                LIMIT 5
            """, (text,))
            
            context = []
            for record in cur.fetchall():
                context.append({
                    'type': record[0],
                    'content': record[1]
                })
            return context

    def get_ground_truth(self, query_text):
        """Get ground truth for a given query"""
        # Load queries from CSV since it's not in the database
        queries_df = pd.read_csv(r"data\\queries.csv")
        return queries_df.loc[queries_df['query_text'] == query_text, 'ground_truth'].tolist()