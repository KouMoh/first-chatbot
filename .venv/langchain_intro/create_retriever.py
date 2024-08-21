import dotenv
import os
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import google.generativeai as genai

# Load environment variables from .env file
dotenv.load_dotenv()

# Set the path for the CSV file and Chroma data
REVIEWS_CSV_PATH = "data/reviews.csv"
REVIEWS_CHROMA_PATH = "chroma_data"

# Configure the Gemini API key
api_key = os.environ.get("GOOGLE_API_KEY")
if api_key is None:
    raise ValueError("API key not found. Please set the GOOGLE_API_KEY in your .env file.")
genai.configure(api_key=api_key)

# Load reviews from the CSV file
loader = CSVLoader(file_path=REVIEWS_CSV_PATH, source_column="review")
reviews = loader.load()

# Create an embedding instance using SentenceTransformerEmbeddings
gemini_embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create the Chroma vector store with the embedding function
reviews_vector_db = Chroma.from_documents(
    reviews,
    gemini_embeddings,  # Pass the embedding class instance
    persist_directory=REVIEWS_CHROMA_PATH
)