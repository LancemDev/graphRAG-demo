import os
import shutil
import kuzu
import lancedb
import openai
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import textwrap
import uuid

class GraphRAG:
    def __init__(self, 
                 openai_api_key: str, 
                 model_name: str = "gpt-4o", 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 db_path: str = "./graph_rag_db", 
                 lance_db_path: str = "./lance_db",
                 rebuild: bool = False):
        """
        Initialize GraphRAG with KùzuDB for graph operations and LanceDB for vector embeddings.
        
        Args:
            openai_api_key: API key for OpenAI
            model_name: OpenAI model name to use for generation
            embedding_model: Sentence transformer model for embeddings
            db_path: Directory path for KùzuDB
            lance_db_path: Directory path for LanceDB
            rebuild: Whether to rebuild the databases from scratch
        """
        # Set up OpenAI client
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.model_name = model_name
        
        # Set up embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Clean up and initialize databases if needed
        if rebuild:
            if os.path.exists(db_path):
                print(f"Removing existing KùzuDB at {db_path}...")
                shutil.rmtree(db_path)
            if os.path.exists(lance_db_path):
                print(f"Removing existing LanceDB at {lance_db_path}...")
                shutil.rmtree(lance_db_path)
        
        # Set up KùzuDB for graph operations
        print("Initializing KùzuDB for graph operations...")
        self.graph_db = kuzu.Database(db_path)
        self.graph_conn = kuzu.Connection(self.graph_db)
        
        # Set up LanceDB for vector embeddings
        print("Initializing LanceDB for vector storage...")
        self.lance_db = lancedb.connect(lance_db_path)
        
        # Initialize schema if it doesn't exist
        self._initialize_schema()
    
    def _initialize_schema(self):
        """Initialize the database schema if it doesn't exist."""
        try:
            # Check if Document table exists in KùzuDB
            check_result = self.graph_conn.execute("SHOW TABLES")
            tables = []
            while check_result.has_next():
                tables.append(check_result.get_next()[0])
            
            if "Document" not in tables:
                print("Creating graph schema...")
                # Create Document node table
                self.graph_conn.execute("""
                    CREATE NODE TABLE Document(
                        id STRING, 
                        title STRING, 
                        source STRING,
                        chunk_id INT64,
                        PRIMARY KEY (id)
                    )
                """)
                
                # Create Entity node table
                self.graph_conn.execute("""
                    CREATE NODE TABLE Entity(
                        id STRING,
                        name STRING, 
                        type STRING,
                        PRIMARY KEY (id)
                    )
                """)
                
                # Create relationships
                self.graph_conn.execute("""
                    CREATE REL TABLE Mentions(
                        FROM Document TO Entity,
                        confidence FLOAT32
                    )
                """)
                
                self.graph_conn.execute("""
                    CREATE REL TABLE RelatedTo(
                        FROM Document TO Document,
                        similarity FLOAT32
                    )
                """)
                
                self.graph_conn.execute("""
                    CREATE REL TABLE EntityRelation(
                        FROM Entity TO Entity,
                        relation STRING,
                        confidence FLOAT32
                    )
                """)
            
            # Check if vectors table exists in LanceDB
            if "document_vectors" not in self.lance_db.table_names():
                print("Creating vector table in LanceDB...")
                # Create schema for document vectors
                self.lance_db.create_table(
                    "document_vectors",
                    [
                        {"id": "", 
                         "text": "", 
                         "title": "", 
                         "source": "",
                         "chunk_id": 0,
                         "vector": np.zeros(384, dtype=np.float32)}
                    ]
                )
        except Exception as e:
            print(f"Error initializing schema: {e}")
            raise
    
    def add_document(self, text: str, title: str = "", source: str = "", chunk_size: int = 500):
        """
        Add a document to both databases - KùzuDB and LanceDB.
        
        Args:
            text: The document text
            title: Title of the document 
            source: Source of the document
            chunk_size: Size of chunks to split the document into
        """
        # Split text into chunks
        chunks = self._split_text(text, chunk_size)
        
        doc_nodes = []
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            # Generate a unique ID
            doc_id = str(uuid.uuid4())
            
            # Store in graph database
            self.graph_conn.execute(f"""
                CREATE (d:Document {{
                    id: '{doc_id}',
                    title: '{title.replace("'", "''")}',
                    source: '{source.replace("'", "''")}',
                    chunk_id: {i}
                }})
            """)
            
            # Create embedding
            embedding = self.embedding_model.encode(chunk)
            
            # Store in LanceDB
            table = self.lance_db.open_table("document_vectors")
            table.add([{
                "id": doc_id,
                "text": chunk,
                "title": title,
                "source": source,
                "chunk_id": i,
                "vector": embedding
            }])
            
            doc_nodes.append({"id": doc_id, "chunk_id": i})
        
        # Create relationships between chunks of the same document
        for i in range(len(doc_nodes)):
            for j in range(i+1, len(doc_nodes)):
                self.graph_conn.execute(f"""
                    MATCH (a:Document), (b:Document)
                    WHERE a.id = '{doc_nodes[i]["id"]}' AND b.id = '{doc_nodes[j]["id"]}'
                    CREATE (a)-[:RelatedTo {{similarity: 0.9}}]->(b)
                    CREATE (b)-[:RelatedTo {{similarity: 0.9}}]->(a)
                """)
        
        # Extract entities and relationships with OpenAI
        if len(chunks) > 0:
            # Use the first chunk to extract entities
            self._extract_entities(chunks[0], doc_nodes[0]["id"])
        
        return doc_nodes
    
    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks of roughly equal size."""
        # Simple splitting by chunk size
        return textwrap.wrap(text, width=chunk_size, break_long_words=False, break_on_hyphens=False)
    
    def _extract_entities(self, text: str, doc_id: str):
        """Extract entities and relationships using OpenAI."""
        try:
            # Define the prompt for entity extraction
            system_prompt = """
            Extract named entities and their relationships from the text.
            Return the result as a JSON object with the following structure:
            {
                "entities": [
                    {"name": "Entity Name", "type": "person/organization/location/concept"}
                ],
                "relationships": [
                    {"source": "Entity1", "target": "Entity2", "relation": "relationship type"}
                ]
            }
            Extract only the most important and clearly mentioned entities and relationships.
            """
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model_name,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ]
            )
            
            # Parse the response
            content = response.choices[0].message.content
            import json
            result = json.loads(content)
            
            # Add entities to the graph
            for entity in result.get("entities", []):
                entity_id = str(uuid.uuid4())
                entity_name = entity["name"].replace("'", "''")
                entity_type = entity["type"]
                
                # Check if entity already exists
                check_entity = self.graph_conn.execute(f"""
                    MATCH (e:Entity)
                    WHERE e.name = '{entity_name}' AND e.type = '{entity_type}'
                    RETURN e.id
                """)
                
                if check_entity.has_next():
                    # Entity exists, use its ID
                    entity_id = check_entity.get_next()[0]
                else:
                    # Create new entity
                    self.graph_conn.execute(f"""
                        CREATE (e:Entity {{
                            id: '{entity_id}',
                            name: '{entity_name}',
                            type: '{entity_type}'
                        }})
                    """)
                
                # Create relationship between document and entity
                self.graph_conn.execute(f"""
                    MATCH (d:Document), (e:Entity)
                    WHERE d.id = '{doc_id}' AND e.id = '{entity_id}'
                    CREATE (d)-[:Mentions {{confidence: 0.8}}]->(e)
                """)
            
            # Add relationships between entities
            for rel in result.get("relationships", []):
                source_name = rel["source"].replace("'", "''")
                target_name = rel["target"].replace("'", "''")
                relation_type = rel["relation"].replace("'", "''")
                
                # Find the entities
                self.graph_conn.execute(f"""
                    MATCH (s:Entity), (t:Entity)
                    WHERE s.name = '{source_name}' AND t.name = '{target_name}'
                    CREATE (s)-[:EntityRelation {{
                        relation: '{relation_type}',
                        confidence: 0.7
                    }}]->(t)
                """)
        
        except Exception as e:
            print(f"Error extracting entities: {e}")
    
    def query(self, query_text: str, top_k: int = 5) -> str:
        """
        Query the GraphRAG system and generate a response.
        
        Args:
            query_text: The query text
            top_k: Number of top results to retrieve
        
        Returns:
            Generated response based on retrieved context
        """
        # Get vector embedding for the query
        query_embedding = self.embedding_model.encode(query_text)
        
        # Search for similar documents in LanceDB
        table = self.lance_db.open_table("document_vectors")
        vector_results = table.search(query_embedding).limit(top_k).to_list()
        
        # Extract document IDs
        doc_ids = [result["id"] for result in vector_results]
        
        # Get related documents through graph connections
        graph_results = []
        for doc_id in doc_ids:
            # Find documents directly connected to the retrieved documents
            result = self.graph_conn.execute(f"""
                MATCH (d:Document {{id: '{doc_id}'}})-[:RelatedTo]->(related:Document)
                RETURN related.id, related.title
                LIMIT 2
            """)
            
            while result.has_next():
                related_id = result.get_next()[0]
                if related_id not in doc_ids:
                    doc_ids.append(related_id)
            
            # Find documents that mention the same entities
            result = self.graph_conn.execute(f"""
                MATCH (d:Document {{id: '{doc_id}'}})-[:Mentions]->(e:Entity)<-[:Mentions]-(related:Document)
                WHERE d.id <> related.id
                RETURN related.id, related.title, e.name
                LIMIT 3
            """)
            
            while result.has_next():
                related_id = result.get_next()[0]
                if related_id not in doc_ids:
                    doc_ids.append(related_id)
        
        # Retrieve the full text from LanceDB for all document IDs
        context_docs = []
        for doc_id in doc_ids:
            result = table.search().where(f"id = '{doc_id}'").to_list()
            if result:
                context_docs.append(result[0])
        
        # Sort by relevance (assuming the original vector results are most relevant)
        context_docs.sort(key=lambda x: doc_ids.index(x["id"]) if x["id"] in doc_ids else len(doc_ids))
        
        # Build the context prompt
        context_text = "\n\n".join([
            f"Document: {doc['title']}\nSource: {doc['source']}\nContent: {doc['text']}"
            for doc in context_docs[:top_k]  # Limit to top_k most relevant docs
        ])
        
        # Get entity information
        entity_info = ""
        for doc_id in doc_ids[:3]:  # Limit to first 3 doc IDs to avoid too much info
            result = self.graph_conn.execute(f"""
                MATCH (d:Document {{id: '{doc_id}'}})-[:Mentions]->(e:Entity)
                RETURN e.name, e.type
                LIMIT 5
            """)
            
            entities = []
            while result.has_next():
                row = result.get_next()
                entities.append(f"{row[0]} ({row[1]})")
            
            if entities:
                entity_info += f"Entities in context: {', '.join(entities)}\n"
        
        # Generate response with OpenAI
        system_prompt = f"""
        You are a helpful assistant that answers questions based on the provided context.
        Use the context to generate accurate and helpful answers.
        
        Additional context entities: {entity_info}
        """
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context information is below:\n\n{context_text}\n\nGiven the context information and not prior knowledge, answer the query: {query_text}"}
            ]
        )
        
        return response.choices[0].message.content

    def run_graph_query(self, cypher_query: str):
        """
        Run a raw Cypher query on the graph database.
        
        Args:
            cypher_query: Cypher query to run
            
        Returns:
            Query results as a list of rows
        """
        result = self.graph_conn.execute(cypher_query)
        rows = []
        
        while result.has_next():
            rows.append(result.get_next())
            
        return rows
    
    def visualize_graph(self, limit: int = 20):
        """
        Return data to visualize the graph (could be used with networkx or similar tools).
        
        Args:
            limit: Maximum number of nodes to return
            
        Returns:
            Dictionary with nodes and edges for visualization
        """
        nodes = []
        edges = []
        
        # Get documents
        doc_result = self.graph_conn.execute(f"""
            MATCH (d:Document)
            RETURN d.id, d.title, 'Document' as type
            LIMIT {limit}
        """)
        
        while doc_result.has_next():
            row = doc_result.get_next()
            nodes.append({
                "id": row[0],
                "label": row[1] if row[1] else f"Doc {row[0][:8]}...",
                "type": row[2]
            })
        
        # Get entities
        entity_result = self.graph_conn.execute(f"""
            MATCH (e:Entity)
            RETURN e.id, e.name, e.type
            LIMIT {limit}
        """)
        
        while entity_result.has_next():
            row = entity_result.get_next()
            nodes.append({
                "id": row[0],
                "label": row[1],
                "type": f"Entity:{row[2]}"
            })
        
        # Get document-entity edges
        mention_result = self.graph_conn.execute(f"""
            MATCH (d:Document)-[m:Mentions]->(e:Entity)
            RETURN d.id, e.id, m.confidence
            LIMIT {limit * 2}
        """)
        
        while mention_result.has_next():
            row = mention_result.get_next()
            edges.append({
                "source": row[0],
                "target": row[1],
                "type": "Mentions",
                "weight": row[2]
            })
        
        # Get document-document edges
        related_result = self.graph_conn.execute(f"""
            MATCH (d1:Document)-[r:RelatedTo]->(d2:Document)
            RETURN d1.id, d2.id, r.similarity
            LIMIT {limit * 2}
        """)
        
        while related_result.has_next():
            row = related_result.get_next()
            edges.append({
                "source": row[0],
                "target": row[1],
                "type": "RelatedTo",
                "weight": row[2]
            })
        
        # Get entity-entity edges
        entity_rel_result = self.graph_conn.execute(f"""
            MATCH (e1:Entity)-[r:EntityRelation]->(e2:Entity)
            RETURN e1.id, e2.id, r.relation, r.confidence
            LIMIT {limit * 2}
        """)
        
        while entity_rel_result.has_next():
            row = entity_rel_result.get_next()
            edges.append({
                "source": row[0],
                "target": row[1],
                "type": row[2],
                "weight": row[3]
            })
        
        return {"nodes": nodes, "edges": edges}


# Example usage
if __name__ == "__main__":
    # Initialize GraphRAG
    openai_api_key = "your_openai_api_key"  # Replace with your OpenAI API key
    rag = GraphRAG(openai_api_key=openai_api_key, rebuild=True)
    
    # Add documents
    doc1 = """
    Machine learning (ML) is a field of inquiry devoted to understanding and building methods that 'learn', 
    that is, methods that leverage data to improve performance on some set of tasks. It is seen as a 
    part of artificial intelligence. Machine learning algorithms build a model based on sample data, 
    known as training data, in order to make predictions or decisions without being explicitly programmed to do so.
    Machine learning algorithms are used in a wide variety of applications, such as in medicine, email filtering, 
    speech recognition, and computer vision, where it is difficult or unfeasible to develop conventional algorithms 
    to perform the needed tasks. A subset of machine learning is closely related to computational statistics, 
    which focuses on making predictions using computers, but not all machine learning is statistical learning.
    """
    rag.add_document(doc1, title="Introduction to Machine Learning", source="Wikipedia")
    
    doc2 = """
    Deep learning is part of a broader family of machine learning methods based on artificial neural networks 
    with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep-learning 
    architectures such as deep neural networks, deep belief networks, deep reinforcement learning, recurrent 
    neural networks and convolutional neural networks have been applied to fields including computer vision, 
    speech recognition, natural language processing, machine translation, bioinformatics, drug design, medical 
    image analysis, climate science, material inspection and board game programs, where they have produced results 
    comparable to and in some cases surpassing human expert performance.
    """
    rag.add_document(doc2, title="Deep Learning Overview", source="Research Paper")
    
    doc3 = """
    Graph databases store data in a graph structure where entities are connected through relationships.
    This architecture allows for efficient traversal of connected data points, which is particularly
    useful for applications like social networks, recommendation engines, and knowledge graphs.
    KùzuDB is a high-performance graph database designed for complex querying of interconnected data.
    Its query language is inspired by Cypher, making it familiar to those who have worked with Neo4j
    or other Cypher-compatible databases.
    """
    rag.add_document(doc3, title="Graph Databases and KùzuDB", source="Documentation")
    
    # Query the system
    query = "How are machine learning and graph databases related?"
    response = rag.query(query)
    print(f"Query: {query}")
    print(f"Response: {response}")
    
    # Show a sample graph query
    print("\nRunning a graph query to find entities related to machine learning:")
    results = rag.run_graph_query("""
        MATCH (d:Document)-[:Mentions]->(e:Entity)
        WHERE d.title CONTAINS 'Machine Learning'
        RETURN e.name, e.type
    """)
    
    for row in results:
        print(f"Entity: {row[0]} (Type: {row[1]})")
    
    # Get graph visualization data
    graph_data = rag.visualize_graph()
    print(f"\nGraph contains {len(graph_data['nodes'])} nodes and {len(graph_data['edges'])} edges")