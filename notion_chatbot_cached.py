import os
import numpy as np
import faiss
import streamlit as st
import pickle
from notion_client import Client
from openai import OpenAI

# -------------------------
# 0. LOAD SECRETS
# -------------------------
NOTION_TOKEN = st.secrets["notion"]["NOTION_TOKEN"]
OPENAI_API_KEY = st.secrets["openai"]["OPENAI_API_KEY"]

database_ids = [
    st.secrets["databases"]["database_1_id"],
    st.secrets["databases"]["database_2_id"],
    st.secrets["databases"]["database_3_id"],
    st.secrets["databases"]["database_4_id"]
]

# -------------------------
# 1. CLIENTS
# -------------------------
client = OpenAI(api_key=OPENAI_API_KEY)
notion = Client(auth=NOTION_TOKEN)

EMBEDDINGS_FILE = "notion_embeddings.pkl"

# -------------------------
# 2. PULL NOTION PAGES
# -------------------------
@st.cache_data(show_spinner=True)
def get_all_notion_pages(database_ids):
    pages = []
    for db_id in database_ids:
        try:
            response = notion.databases.query(database_id=db_id)
        except Exception as e:
            st.warning(f"Failed to query database {db_id}: {e}")
            continue
        for page in response['results']:
            title_list = page.get("properties", {}).get("Name", {}).get("title", [])
            title_text = title_list[0].get("text", {}).get("content", "Untitled") if title_list else "Untitled"
            url = page.get("url", "")
            pages.append({
                "title": title_text,
                "url": url,
                "database_id": db_id
            })
    return pages

notion_pages = get_all_notion_pages(database_ids)

# -------------------------
# 3. BUILD OR LOAD EMBEDDINGS
# -------------------------
def build_or_load_embeddings(pages):
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "rb") as f:
            cached_pages = pickle.load(f)
        if len(cached_pages) == len(pages):
            return cached_pages
    # Create embeddings for pages
    for page in pages:
        try:
            response = client.embeddings.create(
                model="text-embedding-3-large",
                input=page["title"]
            )
            page["embedding"] = response.data[0].embedding
        except Exception as e:
            st.warning(f"Failed to generate embedding for page {page['title']}: {e}")
            page["embedding"] = np.zeros(1536).tolist()  # fallback embedding
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(pages, f)
    return pages

notion_pages = build_or_load_embeddings(notion_pages)

# -------------------------
# 4. BUILD FAISS INDEX
# -------------------------
def build_faiss_index(pages):
    embeddings = np.array([page["embedding"] for page in pages]).astype("float32")
    if embeddings.size == 0:
        st.error("No embeddings available to build FAISS index.")
        return None
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

index = build_faiss_index(notion_pages)

# -------------------------
# 5. SEARCH & CHAT FUNCTIONS
# -------------------------
def search_similar_pages(query, top_k=3):
    try:
        query_embedding = np.array([client.embeddings.create(
            model="text-embedding-3-large",
            input=query
        ).data[0].embedding]).astype("float32")
    except Exception as e:
        st.warning(f"Failed to generate embedding for query: {e}")
        return []
    distances, indices = index.search(query_embedding, top_k)
    return [notion_pages[i] for i in indices[0] if i < len(notion_pages)]

def ask_chat_with_links(question, top_k=3):
    top_pages = search_similar_pages(question, top_k)
    if not top_pages:
        return "Sorry, I couldn't retrieve relevant Notion pages."
    
    # Build references section
    references = "\n".join([f"- [{p['title']}]({p['url']})" for p in top_pages])
    
    # Ask model
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Answer questions using the provided Notion pages. Include the URL for reference."},
                {"role": "user", "content": f"Question: {question}\n\nReferences:\n{references}"}
            ]
        )
        answer = response.choices[0].message.content
    except Exception as e:
        st.warning(f"OpenAI request failed: {e}")
        return f"Sorry, I couldn't get an answer from OpenAI.\n\nReferences:\n{references}"
    
    return f"{answer}\n\nReferences:\n{references}"

# -------------------------
# 6. STREAMLIT UI
# -------------------------
st.set_page_config(page_title="Notion Chatbot", layout="wide")
st.title("📖 Notion Chatbot (Cached & GPT-3.5)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_question = st.text_input("Ask a question about your Notion pages:")

if user_question:
    with st.spinner("Fetching answer..."):
        answer = ask_chat_with_links(user_question, top_k=5)
        st.session_state.chat_history.append((user_question, answer))

# Display chat history
for q, a in st.session_state.chat_history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {a}")
    st.markdown("---")
