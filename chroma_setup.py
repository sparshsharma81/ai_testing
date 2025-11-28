import chromadb
from chromadb.config import Settings

def get_chroma_client():
    client = chromadb.Client(
        Settings(chroma_db_impl="duckdb+parquet", persist_directory="chroma_db")
    )
    return client

def get_collection():
    client = get_chroma_client()
    collection = client.get_or_create_collection(name="astro_knowledge")
    return collection
