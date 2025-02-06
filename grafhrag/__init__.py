import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.prompts import PromptTemplate

class RAGSystem:
    def __init__(self, openai_api_key: str, documents_path: str):
        """
        Initialize the RAG system
        
        Args:
            openai_api_key (str): OpenAI API key
            documents_path (str): Path to the directory containing documents
        """
        self.openai_api_key = openai_api_key
        self.documents_path = documents_path
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=openai_api_key
        )
        
    def load_documents(self):
        """Load documents from the specified directory"""
        loader = DirectoryLoader(
            self.documents_path,
            glob="**/*.txt",
            recursive=True
        )
        documents = loader.load()
        return documents
    
    def process_documents(self, documents, chunk_size=1000):
        """
        Process documents by splitting them into chunks
        
        Args:
            documents: List of documents
            chunk_size (int): Size of text chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(documents)
        return splits
    
    def create_vectorstore(self, splits):
        """Create a vector store from document chunks"""
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        return vectorstore
    
    def setup_qa_chain(self, vectorstore):
        """Set up the question-answering chain"""
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        {context}
        
        Question: {question}
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        return chain
    
    def query(self, chain, question: str):
        """
        Query the RAG system
        
        Args:
            chain: The QA chain
            question (str): Question to ask
        """
        result = chain({"query": question})
        return {
            "answer": result["result"],
            "source_documents": result["source_documents"]
        }