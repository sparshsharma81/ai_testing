import google.generativeai as genai
from chroma_setup import get_collection
from dotenv import load_dotenv
import os

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-pro")

def add_astrology_knowledge(text, doc_id):
    collection = get_collection()
    collection.add(
        documents=[text],
        ids=[doc_id]
    )
    return {"status": "success", "added_id": doc_id}

def search_knowledge(query):
    collection = get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=5
    )
    return results

def generate_astrology_answer(user_query):
    # Step 1: Retrieve knowledge
    retrieved = search_knowledge(user_query)

    context = ""
    for doc in retrieved["documents"][0]:
        context += doc + "\n\n"

    # Step 2: Build prompt for Gemini
    prompt = f"""
You are an expert Vedic astrology bot. Use the context below to answer the user's question accurately.
If the user gives a D1 chart, interpret it logically using Vedic rules.

Context:
{context}

User Question:
{user_query}

Astrology Answer:
"""

    # Step 3: Gemini call
    response = model.generate_content(prompt)
    return response.text
