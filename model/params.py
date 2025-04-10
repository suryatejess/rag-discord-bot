import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

FAISS_FILE = 'faiss_index.index'
DOCSTORE_FILE = 'in_memory_docstore.pkl'
INDEX_TO_DOCSTORE_ID_FILE = 'index_to_docstore_id.pkl'
PDF_PATH = 'Bhagavad-gita.pdf'

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

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