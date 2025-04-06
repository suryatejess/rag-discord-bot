import os
import pickle
import fitz
import faiss
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.docstore.in_memory import InMemoryDocstore

# ----------- CONFIG ------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

LOAD_EMBEDDINGS = True  # Flip to True to load existing FAISS & docstore
PDF_PATH = 'Bhagavad-gita.pdf'
FAISS_FILE = 'faiss_index.index'
DOCSTORE_FILE = 'in_memory_docstore.pkl'
INDEX_TO_DOCSTORE_ID_FILE = 'index_to_docstore_id.pkl'

# ----------- GLOBALS ------------
chat_history = []
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = None

# ----------- MODULE 1: PDF ------------
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def chunk_text(text, chunk_size=512):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# ----------- MODULE 2: Vector Store ------------
def store_embeddings(chunks):
    chunk_embeddings = embedding_model.embed_documents(chunks)
    chunk_embeddings_np = np.array(chunk_embeddings).astype("float32")

    index = faiss.IndexFlatL2(chunk_embeddings_np.shape[1])
    index.add(chunk_embeddings_np)

    documents = [Document(page_content=chunk) for chunk in chunks]
    docstore = InMemoryDocstore({i: documents[i] for i in range(len(documents))})

    index_to_docstore_id = {i: i for i in range(len(documents))}

    faiss.write_index(index, FAISS_FILE)
    with open(DOCSTORE_FILE, 'wb') as f:
        pickle.dump(docstore, f)
    with open(INDEX_TO_DOCSTORE_ID_FILE, 'wb') as f:
        pickle.dump(index_to_docstore_id, f)

    return FAISS(embedding_function=embedding_model, index=index,
                 docstore=docstore, index_to_docstore_id=index_to_docstore_id)

def load_embeddings():
    index = faiss.read_index(FAISS_FILE)
    with open(DOCSTORE_FILE, 'rb') as f:
        docstore = pickle.load(f)
    with open(INDEX_TO_DOCSTORE_ID_FILE, 'rb') as f:
        index_to_docstore_id = pickle.load(f)

    return FAISS(embedding_function=embedding_model, index=index,
                 docstore=docstore, index_to_docstore_id=index_to_docstore_id)

# ----------- MODULE 3: Gemini Model & RAG ------------
generation_config = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config
)

def retrieve_relevant_chunks(query, k=5):
    docs = vector_store.similarity_search(query, k=k)
    return " ".join([doc.page_content for doc in docs])

def generate_with_google(context, query):
    response = model.generate_content(f"Based on the context provided below, answer the following query. Context: {context}\nQuery: {query}\nAnswer:")
    return response.text

# ----------- MODULE 4: Chat History ------------
def format_chat_history(history, max_turns=5):
    formatted = ""
    for user_input, bot_reply in history[-max_turns:]:
        formatted += f"User: {user_input}\nBot: {bot_reply}\n"
    return formatted

def run_rag_pipeline_with_history(query, chat_history, k=5):
    retrieved_text = retrieve_relevant_chunks(query, k=k)
    history_context = format_chat_history(chat_history)
    full_context = f"{history_context}\nContext from documents:\n{retrieved_text}"
    response = generate_with_google(full_context, query)
    chat_history.append((query, response))
    return response

# ----------- MODULE 5: Main Logic ------------
def initialize_vector_store():
    global vector_store
    if LOAD_EMBEDDINGS:
        print("🔄 Loading embeddings from disk...")
        vector_store = load_embeddings()
    else:
        print("📘 Extracting and storing embeddings...")
        text = extract_text_from_pdf(PDF_PATH)
        chunks = chunk_text(text)
        vector_store = store_embeddings(chunks)

def chat():
    print("🤖 RAG Chatbot Initialized! Type 'exit' to quit.")
    while True:
        user_query = input("You: ")
        if user_query.lower() in ["exit", "quit"]:
            break
        answer = run_rag_pipeline_with_history(user_query, chat_history)
        print("Bot:", answer)

# ----------- RUN ------------
if __name__ == "__main__":
    initialize_vector_store()
    # query = "Summarize what is true power"
    # print("Initial Query:")
    # print(run_rag_pipeline_with_history(query, chat_history))

    chat()
