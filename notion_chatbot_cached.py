import os
import numpy as np
import faiss
import streamlit as st
import pickle
from notion_client import Client
from openai import OpenAI

# -------------------------
# CONFIGURATION
# -------------------------
# Load secrets securely from Streamlit
notion_token = st.secrets["NOTION_TOKEN"]
openai_api_key = st.secrets["OPENAI_API_KEY"]

client = OpenAI(api_key=openai_api_key)
notion = Client(auth=notion_token)

database_ids = [
    "8f31a1a8d6f243719bfcbe99712d140c",
    "c44d53e50fc54a7387e7d40f21850e5f",
    "f8d52321d6cb479d99d62e89a4f72f54"
]

EMBEDDINGS_FILE = "notion_embeddings.pkl"

# -------------------------
# 1. PULL NOTION PAGES
# -------------------------
@st.cache_data(show_spinner=True)
def get_all_notion_pages(database_ids):
    pages = []
    for db_id in database_ids:
        response = notion.databases.query(database_id=db_id)
        for page in response['results']:
            try:
                title = page['properties']['Name']['title'][0]['text']['content']
            except (KeyError, IndexError):
                title = "Untitled"
            url = page['url']
            pages.append({
                "title": title,
                "url": url,
                "database_id": db_id
            })
    return pages

notion_pages = get_all_notion_pages(database_ids)

# -------------------------
# 2. LOAD OR CREATE EMBEDDINGS
# -------------------------
def build_or_load_embeddings(pages):
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "rb") as f:
            cached_pages = pickle.load(f)
        if len(cached_pages) == len(pages):
            return cached_pages  # All pages already embedded

    # Otherwise, create embeddings
    for page in pages:
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=page["title"]
        )
        page["embedding"] = response.data[0].embedding

    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(pages, f)
    return pages

notion_pages = build_or_load_embeddings(notion_pages)

# -------------------------
# 3. BUILD FAISS INDEX
# -------------------------
embeddings = np.array([page["embedding"] for page in notion_pages]).astype("float32")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# -------------------------
# 4. SEARCH & CHAT FUNCTIONS
# -------------------------
def search_similar_pages(query, top_k=3):
    query_embedding = np.array([client.embeddings.create(
        model="text-embedding-3-large",
        input=query
    ).data[0].embedding]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    return [notion_pages[i] for i in indices[0]]

def ask_chat_with_links(question, top_k=3):
    top_pages = search_similar_pages(question, top_k)
    context_text = ""
    for page in top_pages:
        context_text += f"- [{page['title']}]({page['url']})\n"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Answer questions using the provided Notion pages."},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"}
        ]
    )
    return response.choices[0].message.content

# -------------------------
# 5. STREAMLIT UI
# -------------------------
st.set_page_config(page_title="Notion Chatbot", layout="wide")
st.title("📖 Notion Chatbot (Cached & GPT-3.5)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_question = st.text_input("Ask a question about your Notion pages:")

if user_question:
    with st.spinner("Fetching answer..."):
        answer = ask_chat_with_links(user_question)
        st.session_state.chat_history.append((user_question, answer))

# Display chat history
for q, a in st.session_state.chat_history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {a}")
    st.markdown("---")
