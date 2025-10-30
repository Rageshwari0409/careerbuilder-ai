# CareerBuilder AI Chat & RAG Service

This project implements a real-time AI-powered chat service using Azure OpenAI and Hugging Face models, with Retrieval-Augmented Generation (RAG) capabilities powered by Pinecone and LangChain.

## 🚀 Features

- Chat with Azure OpenAI or Hugging Face models
- RAG pipeline using uploaded PDFs and Pinecone vector store
- Retry logic and robust error handling
- Persistent memory for Azure OpenAI chat
- Modular structure with logging and configuration

## 📁 Project Structure

- `app.py` – Flask application entry point
- `services/chat_service.py` – Chat logic for Azure OpenAI and Hugging Face
- `services/rag_service.py` – RAG pipeline using LangChain and Pinecone
- `clients/llm_clients.py` – LLM client initialization
- `utils/logger.py` – Logging setup
- `config/config.py` – Configuration and API keys
- `uploads/` – Directory for uploaded PDF files