import lancedb
import kuzudb_test
import pandas as pd
import pyarrow as pa
from PyPDF2 import PdfReader
import os

# Define database URIs
lance_db_uri = "data/sample-lancedb"
db = lancedb.connect(lance_db_uri)

# Define KuzuGraphDB database
graph_db_path = "data/kuzu-graph"
graph = kuzudb_test.Database(graph_db_path)

# Function to chunk PDF into text
def chunk_pdf(file_path, chunk_size=1000):
    pdf = PdfReader(open(file_path, "rb"))
    text_chunks = []
    text = ""
    for page_num in range(len(pdf.pages)):
        text += pdf.pages[page_num].extract_text()
        while len(text) > chunk_size:
            text_chunks.append(text[:chunk_size])
            text = text[chunk_size:]
    if text:
        text_chunks.append(text)
    return text_chunks

# Function to insert relationships into KuzuGraphDB
def insert_graph_data(entity1, relation, entity2):
    conn = graph.connect()
    query = f"""
    CREATE (:Entity {{name: "{entity1}"}})-[:{relation}]->(:Entity {{name: "{entity2}"}})
    """
    conn.execute(query)
    conn.commit()

# Extract named entities and create relationships (basic example)
def extract_relationships(text):
    # Example: Extract relationships using simple rules (can be replaced with NLP models)
    if "king" in text.lower() and "queen" in text.lower():
        insert_graph_data("King", "married_to", "Queen")
    if "war" in text.lower() and "kingdom" in text.lower():
        insert_graph_data("Kingdom", "involved_in", "War")

# Path to the PDF file
pdf_file_path = "assets/pdf/story.pdf"

# Chunk the PDF
chunks = chunk_pdf(pdf_file_path)

# Prepare data for LanceDB
data = [{"chunk": chunk, "chunk_id": idx} for idx, chunk in enumerate(chunks)]

# Create or open the table
if "pdf_chunks" in db.table_names():
    tbl = db.open_table("pdf_chunks")
else:
    tbl = db.create_table("pdf_chunks", data=data)

# Upsert new data
tbl.add(data, mode="overwrite")  # Use "append" to keep existing data

# Store relationships in KuzuGraphDB
for chunk in chunks:
    extract_relationships(chunk)

# Fetch and display data to verify upsert
print("Updated Table Data:")
print(tbl.to_pandas().to_dict(orient="records"))
