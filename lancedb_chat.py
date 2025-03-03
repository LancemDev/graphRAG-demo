import openai
import lancedb
import kuzudb_test
import os
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Define LanceDB database URI
lance_db_uri = "data/sample-lancedb"
db = lancedb.connect(lance_db_uri)

# Open LanceDB table
tbl = db.open_table("pdf_chunks")

# Define KuzuGraphDB database
graph_db_path = "data/kuzu-graph"
graph = kuzudb_test.Database(graph_db_path)

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to retrieve relevant text chunks from LanceDB
def retrieve_relevant_chunks(query, top_k=5):
    return tbl.to_pandas().head(top_k).to_dict(orient="records")

# Function to retrieve related concepts from KuzuGraphDB
def retrieve_graph_context(query):
    conn = graph.connect()
    result = conn.execute(f"""
    MATCH (e1)-[r]->(e2) 
    WHERE e1.name CONTAINS "{query}" 
    RETURN e1.name, r, e2.name
    """)
    return result.fetchall()

# Function to generate a response using OpenAI's API
def generate_response(query, text_chunks, graph_context):
    text_context = "\n".join(chunk["chunk"] for chunk in text_chunks)
    graph_context_str = "\n".join([f"{r[0]} -[{r[1]}]-> {r[2]}" for r in graph_context])

    context = f"Text Context:\n{text_context}\n\nGraph Context:\n{graph_context_str}"
    
    messages = [
        {"role": "system", "content": "You are an AI assistant using text and graph knowledge."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=150
    )
    
    return response.choices[0].message.content.strip()

# Main chatbot function
def chatbot(query):
    text_chunks = retrieve_relevant_chunks(query)
    graph_context = retrieve_graph_context(query)
    response = generate_response(query, text_chunks, graph_context)
    return response

# Example usage
if __name__ == "__main__":
    query = "What is the main character's name?"
    response = chatbot(query)
    print(response)
