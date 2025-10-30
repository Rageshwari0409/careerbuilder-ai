# File: services/rag_service.py

import os
import pinecone
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, PINECONE_API_KEY
from services.chat_service import client
from utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((TimeoutError, ConnectionError, ValueError))
)
def initialize_embeddings():
    return AzureOpenAIEmbeddings(
        deployment="text-embedding-3-small",
        model="text-embedding-3-small",
        openai_api_type="azure",
        openai_api_key=AZURE_OPENAI_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_version="2023-05-15",
        chunk_size=1000
    )

def handle_rag_flow(user_message):
    logger.info("🔍 Starting RAG flow...")

    try:
        uploaded_files = os.listdir("uploads")
        if not uploaded_files:
            logger.warning("⚠️ No files found in 'uploads' directory.")
            return {"result": "Error: No file uploaded for RAG"}
        file_path = os.path.join("uploads", uploaded_files[-1])
        logger.info(f"📄 Using file for RAG: {file_path}")
        pdf_loader = PyPDFLoader(file_path)
        data = pdf_loader.load()
        logger.info(f"✅ Loaded {len(data)} pages from the PDF.")
        logger.debug(f"📄 Preview of first page: {data[0].page_content[:500]}")
    except Exception as e:
        logger.error("❌ Error loading PDF", exc_info=True)
        return {"result": f"Error loading PDF: {str(e)}"}

    try:
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=1000, chunk_overlap=20)
        chunks = text_splitter.split_documents(data)
        logger.info(f"🧩 Split text into {len(chunks)} chunks.")
    except Exception as e:
        logger.error("❌ Error splitting text", exc_info=True)
        return {"result": f"Error splitting text: {str(e)}"}

    try:
        embeddings = initialize_embeddings()
        logger.info("🧠 Embeddings initialized successfully.")
    except Exception as e:
        logger.error("❌ Error initializing embeddings", exc_info=True)
        return {"result": f"Error initializing embeddings: {str(e)}"}

    try:
        os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
        pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
        logger.info("🌲 Pinecone client initialized.")
    except Exception as e:
        logger.error("❌ Error initializing Pinecone client", exc_info=True)
        return {"result": f"Error initializing Pinecone client: {str(e)}"}

    index_name = "careerbuilder-index"
    try:
        if index_name not in pc.list_indexes().names():
            logger.info(f"📦 Creating Pinecone index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1")
            )
        else:
            logger.info(f"✅ Pinecone index '{index_name}' already exists.")
    except Exception as e:
        logger.error("❌ Error managing Pinecone index", exc_info=True)
        return {"result": f"Error managing Pinecone index: {str(e)}"}

    try:
        logger.info("📥 Creating vector store from document chunks...")
        vectorstore = PineconeVectorStore.from_documents(chunks, index_name=index_name, embedding=embeddings)
        logger.info("✅ Vector store created successfully.")
    except Exception as e:
        logger.error("❌ Error creating vector store", exc_info=True)
        return {"result": f"Error creating vector store: {str(e)}"}

    try:
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        logger.info("🔍 Retriever initialized.")
    except Exception as e:
        logger.error("❌ Error creating retriever", exc_info=True)
        return {"result": f"Error creating retriever: {str(e)}"}

    try:
        logger.info("🔗 Setting up RetrievalQA chain...")
        chain = RetrievalQA.from_chain_type(llm=client, chain_type="stuff", retriever=retriever)
        result = chain.invoke({"query": user_message})
        logger.info("✅ RAG flow completed successfully.")
        return result
    except Exception as e:
        logger.error("❌ Error running RetrievalQA chain", exc_info=True)
        return {"result": f"Error running RetrievalQA chain: {str(e)}"}