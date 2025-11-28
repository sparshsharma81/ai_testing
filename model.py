from google import genai
from chromadb import Client
import chromadb

client = genai.Client(api_key="YOUR_API_KEY")

chroma = chromadb.Client()
collection = chroma.get_or_create_collection(name="astro_knowledge")
