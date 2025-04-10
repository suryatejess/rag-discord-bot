import pickle
import fitz
import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from params import *

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def chunk_text(text, chunk_size=512):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

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

def initialize_vector_store(LOAD_EMBEDDINGS):
    global vector_store
    if LOAD_EMBEDDINGS:
        print("🔄 Loading embeddings from disk...")
        vector_store = load_embeddings()
    else:
        print("📘 Extracting and storing embeddings...")
        text = extract_text_from_pdf(PDF_PATH)
        chunks = chunk_text(text)
        vector_store = store_embeddings(chunks)

def chat(user_query, chat_history):
    print("🤖 RAG Chatbot Initialized! Type 'exit' to quit.")
    answer = run_rag_pipeline_with_history(user_query, chat_history)
    print(answer)
    return answer
    
