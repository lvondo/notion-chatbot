import os
import streamlit as st
from notion_client import Client
from openai import OpenAI
import numpy as np
import faiss

# -------------------------------
# Load secrets from Streamlit
# -------------------------------
NOTION_TOKEN = st.secrets["NOTION_TOKEN"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
DATABASE_IDS = st.secrets["DATABASE_IDS"].split(",")  # comma-separated list

# Initialize clients
notion = Client(auth=NOTION_TOKEN)
client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------------
# Notion helpers
# -------------------------------
def get_page_title(page):
    return (
        page["properties"]
        .get("Name", {})
        .get("title", [{}])[0]
        .get("text", {})
        .get("content", "Untitled")
    )

def get_page_url(page_id):
    clean_id = page_id.replace("-", "")
    return f"https://www.notion.so/{clean_id}"

def get_page_content(page_id):
    """Fetch plain text content of a Notion page"""
    try:
        blocks = notion.blocks.children.list(page_id).get("results", [])
        text_chunks = []
        for block in blocks:
            if block["type"] in ["paragraph", "heading_1", "heading_2", "heading_3"]:
                text_chunks.append(
                    block[block["type"]]
                    .get("rich_text", [{}])[0]
                    .get("text", {})
                    .get("content", "")
                )
        return "\n".join([t for t in text_chunks if t])
    except Exception as e:
        st.error(f"❌ Failed to fetch content for {page_id}: {e}")
        return ""

@st.cache_data(show_spinner="Fetching Notion pages...")
def get_all_notion_pages(database_ids):
    pages = []
    for db_id in database_ids:
        try:
            response = notion.databases.query(database_id=db_id)
            results = response.get("results", [])
            for page in results:
                pages.append(
                    {
                        "id": page["id"],
                        "title": get_page_title(page),
                        "url": get_page_url(page["id"]),
                    }
                )
        except Exception as e:
            st.error(f"❌ Failed to query database {db_id}: {e}")
    return pages

# -------------------------------
# Embeddings + FAISS index
# -------------------------------
@st.cache_resource(show_spinner="Building FAISS index...")
def build_faiss_index(pages):
    texts = [page["title"] for page in pages]
    embeddings = []
    for text in texts:
        try:
            emb = client.embeddings.create(
                model="text-embedding-3-small", input=text
            ).data[0].embedding
            embeddings.append(emb)
        except Exception as e:
            st.error(f"❌ Failed to embed '{text}': {e}")
            embeddings.append([0.0] * 1536)

    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, pages

def search_similar_pages(query, index, pages, top_k=3):
    query_embedding = np.array(
        client.embeddings.create(model="text-embedding-3-small", input=query).data[0].embedding
    ).astype("float32").reshape(1, -1)

    D, I = index.search(query_embedding, top_k)
    return [pages[i] for i in I[0]]

# -------------------------------
# Ask OpenAI with context + footnotes
# -------------------------------
def ask_with_context(question, similar_pages):
    context_texts = []
    footnotes = []
    for i, p in enumerate(similar_pages, start=1):
        content = get_page_content(p["id"])
        if content:
            context_texts.append(f"[{i}] {p['title']} ({p['url']})\n{content}")
            footnotes.append(f"[{i}] [{p['title']}]({p['url']})")

    context = "\n\n".join(context_texts)

    if not context:
        return "⚠️ No relevant content found in Notion.", []

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Use ONLY the provided Notion content as your source. "
                        "When you use information from a page, cite it inline with its footnote number like [1], [2], etc."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Here is the context:\n\n{context}\n\nQuestion: {question}",
                },
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content, footnotes
    except Exception as e:
        return f"❌ OpenAI request failed: {e}", []

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📖 Notion Chatbot with Sources + Footnotes")

st.markdown("Ask me a question, and I’ll answer using your Notion databases — with inline citations and source links.")

# Fetch data
notion_pages = get_all_notion_pages(DATABASE_IDS)
index, pages = build_faiss_index(notion_pages)

# Input
user_question = st.text_input("❓ Ask a question")

if user_question:
    similar_pages = search_similar_pages(user_question, index, pages)
    st.write("🔎 Best matching Notion pages:")
    for p in similar_pages:
        st.markdown(f"- [{p['title']}]({p['url']})")

    answer, footnotes = ask_with_context(user_question, similar_pages)

    st.subheader("💡 Answer")
    st.write(answer)

    if footnotes:
        st.markdown("#### 📚 Sources")
        for f in footnotes:
            st.markdown(f"- {f}")
