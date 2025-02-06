from grafhrag import RAGSystem

from dotenv import load_dotenv
import os

# Carrega as variáveis do arquivo .env
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DOCUMENTS_PATH = "grafhrag/assets/arquive"

# Initialize RAG system
rag = RAGSystem(OPENAI_API_KEY, DOCUMENTS_PATH)

# Load and process documents
documents = rag.load_documents()
splits = rag.process_documents(documents)

# Create vector store
vectorstore = rag.create_vectorstore(splits)

# Setup QA chain
qa_chain = rag.setup_qa_chain(vectorstore)

# Example query
question = "What is the main topic of the documents?"
result = rag.query(qa_chain, question)
print(f"Answer: {result['answer']}")
print("\nSource Documents:")
for doc in result['source_documents']:
    print(f"- {doc.page_content[:200]}...")

