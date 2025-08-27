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
NOTION_TOKEN = st.secrets["notion"]["NOTION_TOKEN"]
OPENAI_API_KEY = st.secrets["openai"]["OPENAI_API_KEY"]

database_ids = [
    st.secrets["databases"]["database_1_id"],
    st.secrets["databases"]["database_2_id"],
    st.secrets["databases"]["database_3_id"],
    st.secrets["databases"].get("database_4_id")  # optional 4th DB
]
database_urls = [
    st.secrets["databases"]["database_1_url"],
    st.secrets["databases"]["database_2_url"],
    st.secrets["databases"]["database_3_url"],
    st.secrets["databases"].get("database_4_url")  # optional 4th DB
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
    for i, db_id in enumerate(database_ids):
        if db_id is None:
            continue
        try:
            response = notion.databases.query(database_id=db_id)
            for page in response['results']:
                try:
                    title = page['properties']['Name']['title'][0]['text']['content']
                except (KeyError, IndexError):
                    title = "Untitled"
                url = page.get('url', database_urls[i])
                pages.append({"title": title, "url": url, "database_id": db_id})
        except Exception as e:
            st.warning(f"Failed to query database {db_id}: {e}")
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
    # Create embeddings for pages missing them
    for page in pages:
        if "embedding" not in page:
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
    query_embedding = np.array([client.embeddings.create(
        model="text-embedding-3-large",
        input=query
    ).data[0].embedding]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    return [notion_pages[i] for i in indices[0]]

def ask_chat_with_links(question, top_k=3):
    top_pages = search_similar_pages(question, top_k)
    context_text = "\n".join([f"- [{p['title']}]({p['url']})" for p in top_pages])
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

# Add a slider to adjust number of pages to search
top_k = st.slider("Number of related pages to include", min_value=1, max_value=10, value=3)

if user_question:
    with st.spinner("Fetching answer..."):
        try:
            answer = ask_chat_with_links(user_question, top_k=top_k)
        except Exception as e:
            answer = f"Sorry, I couldn't get an answer. Error: {e}"
        st.session_state.chat_history.append((user_question, answer))

# Display chat history
for q, a in st.session_state.chat_history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {a}")
    st.markdown("---")
