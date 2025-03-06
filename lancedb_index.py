import lancedb
import kuzu
import spacy  # For Named Entity Recognition (NER)
import pandas as pd
from PyPDF2 import PdfReader
import os

# Load spaCy NLP model (Use 'en_core_web_sm' for smaller model)
nlp = spacy.load("en_core_web_md")

# Define database paths
lance_db_uri = "data/sample-lancedb"
kuzu_db_uri = "data/kuzu_graph"

# Connect to databases
lance_db = lancedb.connect(lance_db_uri)
kuzu_db = kuzu.Database(kuzu_db_uri)
kuzu_conn = kuzu.Connection(kuzu_db)

# Function to extract named entities from text
def extract_entities(text):
    doc = nlp(text)
    return list(set(ent.text for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "GPE"}))  # Extract people, organizations, locations

# Function to chunk PDF into text
def chunk_pdf(file_path, chunk_size=1000):
    pdf = PdfReader(open(file_path, "rb"))
    text_chunks = []
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
        while len(text) > chunk_size:
            text_chunks.append(text[:chunk_size])
            text = text[chunk_size:]
    if text:
        text_chunks.append(text)
    return text_chunks

# Path to the PDF file
pdf_file_path = "assets/pdf/story.pdf"

# Chunk the PDF
chunks = chunk_pdf(pdf_file_path)

# Prepare data for LanceDB
data = []
for idx, chunk in enumerate(chunks):
    entities = extract_entities(chunk)  # Extract entities
    data.append({"chunk": chunk, "chunk_id": idx, "entities": entities})

# Create or open the LanceDB table
if "pdf_chunks" in lance_db.table_names():
    tbl = lance_db.open_table("pdf_chunks")
else:
    tbl = lance_db.create_table("pdf_chunks", data=data)

# Upsert new data
tbl.add(data, mode="overwrite")  # Use "append" to keep existing data

# Kùzu Graph Schema Definition
kuzu_conn.execute("CREATE NODE TABLE IF NOT EXISTS Chunk(id INT PRIMARY KEY, text STRING)")
kuzu_conn.execute("CREATE NODE TABLE IF NOT EXISTS Entity(name STRING PRIMARY KEY)")
kuzu_conn.execute("CREATE REL TABLE IF NOT EXISTS MENTIONS(FROM Chunk TO Entity)")
kuzu_conn.execute("CREATE REL TABLE IF NOT EXISTS RELATED_TO(FROM Entity TO Entity)")

# Insert data into Kùzu Graph
for entry in data:
    kuzu_conn.execute(f"INSERT INTO Chunk VALUES ({entry['chunk_id']}, '{entry['chunk'].replace("'", "''")}')")
    
    for entity in entry["entities"]:
        kuzu_conn.execute(f"MERGE INTO Entity VALUES ('{entity.replace("'", "''")}')")
        kuzu_conn.execute(f"INSERT INTO MENTIONS VALUES ({entry['chunk_id']}, '{entity.replace("'", "''")}')")

# Fetch and display data to verify upsert
print("Updated Table Data in LanceDB:")
print(tbl.to_pandas().to_dict(orient="records"))

print("Graph Data Inserted into Kùzu!")
