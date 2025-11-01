# File: services/rag_service.py

import os
import pinecone
import logging
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, PINECONE_API_KEY
from services.chat_service import client, client_hf_endpoint
from utils.logger import setup_logging
from clients.llm_clients import initialize_azure_openai_embeddings, initialize_pinecone_client, initialize_huggingface_embeddings
from langchain_core.messages import HumanMessage, SystemMessage 

setup_logging()
CONTEXT_SUFFIX = "\n\nContext:\n{context}"
logger = logging.getLogger(__name__)


def process_pdf():
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
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=1000, chunk_overlap=20)
        chunks = text_splitter.split_documents(data)
        logger.info(f"🧩 Split text into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        logger.error("❌ Error loading PDF", exc_info=True)
        return {"result": f"Error loading PDF: {str(e)}"}

def setup_pinecone_index(pc,index_name,dimension):
    try:
        if index_name not in pc.list_indexes().names():
            logger.info(f"📦 Creating Pinecone index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1")
            )
        else:
            logger.info(f"✅ Pinecone index '{index_name}' already exists.")
    except Exception as e:
        logger.error("❌ Error managing Pinecone index", exc_info=True)
        return {"result": f"Error managing Pinecone index: {str(e)}"}

def create_vector_store(data, embeddings, index_name):
    try:
        logger.info("📥 Creating vector store from document chunks...")
        vectorstore = PineconeVectorStore.from_documents(data, index_name=index_name, embedding=embeddings)
        logger.info("✅ Vector store created successfully.")
        return vectorstore
    except Exception as e:
        logger.error("❌ Error creating vector store", exc_info=True)
        return {"result": f"Error creating vector store: {str(e)}"}

def handle_rag_flow(user_message,system_message,model):
    logger.info("🔍 Starting RAG flow...")
    logger.info("Collecting Data from PDF...")
    data = process_pdf()
    pc = initialize_pinecone_client()
    index_name = "careerbuilder-index3"
    if model == "azure-openai":
        embeddings = initialize_azure_openai_embeddings()
        model = client
        setup_pinecone_index(pc,index_name,dimension=1536)
    elif model == "huggingface":
        embeddings = initialize_huggingface_embeddings()
        model = client_hf_endpoint
        setup_pinecone_index(pc,index_name,dimension=768)
    vectorstore = create_vector_store(data, embeddings, index_name)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    logger.info("🔍 Retriever initialized.")
    try:
        retriever_doc = retriever.invoke(user_message)
        logger.info(f"✅ Retrieved relevant documents. {retriever_doc}")
        context = "\n\n".join([doc.page_content for doc in retriever_doc])
        logger.info(f"📚 Context for RAG: {context}")
        full_prompt_template = system_message + CONTEXT_SUFFIX
        logger.info(f"📝 Full prompt template: {full_prompt_template}")
        final_system_prompt = full_prompt_template.format(context=context)
        logger.info(f"🛠️ Final system prompt: {final_system_prompt}")
        messages = [
        SystemMessage(content=final_system_prompt),  # The instruction and context.
        HumanMessage(content=user_message),             # The user's question.
    ]   
        logger.info(f"💬 Invoking LLM with constructed messages...{messages}")
        return model.invoke(messages).content
    except Exception as e:
        logger.error("❌ Error during retrieval", exc_info=True)
        return {"result": f"Error during retrieval: {str(e)}"}
