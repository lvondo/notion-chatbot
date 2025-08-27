import os
import numpy as np
import faiss
import streamlit as st
import pickle
from notion_client import Client
from openai import OpenAI

# -------------------------
# CONFIGURATION FROM SECRETS
# -------------------------
NOTION_TOKEN = st.secrets["notion"]["NOTION_TOKEN"]
OPENAI_API_KEY = st.secrets["openai"]["OPENAI_API_KEY"]

# Databases
database_ids = [
    st.secrets["databases"]["database_1_id"],
    st.secrets["databases"]["database_2_id"],
    st.secrets["databases"]["database_3_id"]
]

database_urls = [
    st.secrets["databases"]["database_1_url"],
    st.secrets["databases"]["database_2_url"],
    st.secrets["databases"]["database_3_url"]
]

client = OpenAI(api_key=OPENAI_API_KEY)
notion = Client(auth=NOTION_TOKEN)

EMBEDDINGS_FILE = "notion_embeddings.pkl"

# -------------------------
# 1. PULL NOTION PAGES
# -------------------------
@st.cache_data(show_spinner=True)
def get_all_notion_pages(database_ids, database_urls):
    pages = []
    for db_id, db_url in zip(database_ids, database_urls):
        try:
            response = notion.databases.query(database_id=db_id)
        except Exception as e:
            st.error(f"Failed to query database {db_id}: {e}")
            continue
        for page in response['results']:
            try:
                title = page['properties']['Name']['title'][0]['text']['content']
            except (KeyError, IndexError):
                title = "Untitled"
            url = page.get('url', db_url)
            pages.append({
                "title": title,
                "url": url,
                "database_id": db_id
            })
    return pages

notion_pages = get_all_notion_pages(database_ids, database_urls)

# -------------------------
# 2. LOAD OR CREATE EMBEDDINGS
# -------------------------
def build_or_load_embeddings(pages):
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "rb") as f:
            cached_pages = pickle.load(f)
        if len(cached_pages) == len(pages):
            return cached_pages

    for page in pages:
        try:
            response = client.embeddings.create(
                model="text-embedding-3-large",
                input=page["title"]
            )
            page["embedding"] = response.data[0].embedding
        except Exception as e:
            st.error(f"Failed to generate embedding for '{page['title']}': {e}")
            page["embedding"] = np.zeros(1536).tolist()  # fallback

    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(pages, f)
    return pages

notion_pages = build_or_load_embeddings(notion_pages)

# -------------------------
# 3. BUILD FAISS INDEX
# -------------------------
def build_faiss_index(pages):
    embeddings = np.array([page["embedding"] for page in pages]).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

index = build_faiss_index(notion_pages)

# -------------------------
# 4. SEARCH & CHAT FUNCTIONS
# -------------------------
def search_similar_pages(query, top_k=3):
    try:
        query_embedding = np.array([client.embeddings.create(
            model="text-embedding-3-large",
            input=query
        ).data[0].embedding]).astype("float32")
    except Exception as e:
        st.error(f"Failed to generate embedding for query: {e}")
        return []
    distances, indices = index.search(query_embedding, top_k)
    return [notion_pages[i] for i in indices[0]]

def ask_chat_with_links(question, top_k=3):
    top_pages = search_similar_pages(question, top_k)
    context_text = ""
    for page in top_pages:
        context_text += f"- [{page['title']}]({page['url']})\n"
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Answer questions using the provided Notion pages."},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"OpenAI request failed: {e}")
        return "Sorry, I couldn't get an answer from OpenAI."

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

for q, a in st.session_state.chat_history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {a}")
    st.markdown("---")
