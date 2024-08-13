import dotenv
from langchain_ollama import OllamaLLM

dotenv.load_dotenv()

llm = OllamaLLM(model="llama3.1")
