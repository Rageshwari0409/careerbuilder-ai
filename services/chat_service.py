# File: services/chat_service.py

from flask import jsonify
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import FileChatMessageHistory
from huggingface_hub import InferenceClient
from config.config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, DEPLOYMENT_NAME, HUGGINGFACEHUB_API_TOKEN
from utils.logger import setup_logging
from clients.llm_clients import get_azure_openai_client, get_huggingface_client
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging

setup_logging()
logger = logging.getLogger(__name__)

logger.info("Initializing LLM clients...")
client = get_azure_openai_client()
client_hf = get_huggingface_client()
logger.info("LLM clients initialized successfully.")

def validate_input(user_message):
    is_valid = bool(user_message)
    if not is_valid:
        logger.warning("Validation failed: Empty user message.")
    return is_valid

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((TimeoutError, ConnectionError, ValueError))
)
def huggingface_chat_with_retry(user_message, system_message):
    return client_hf.chat_completion(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    )

def handle_huggingface_chat(user_message, system_message):
    logger.debug("Handling Hugging Face chat with retry...")
    try:
        response = huggingface_chat_with_retry(user_message, system_message)
        logger.info("Hugging Face chat handled successfully.")
        return response
    except Exception as e:
        logger.error("Hugging Face Inference API error", exc_info=True)
        raise

def handle_azure_openai_chat(user_message, system_message):
    logger.debug("Handling Azure OpenAI chat...")
    try:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_message),
            HumanMessagePromptTemplate.from_template("{user_input}")
        ])
        history = FileChatMessageHistory("chat_history.json")
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, chat_memory=history)
        chain = LLMChain(llm=client, prompt=prompt, memory=memory)
        response = chain.invoke({"user_input": user_message})
        logger.info("Azure OpenAI chat handled successfully.")
        return response
    except Exception as e:
        logger.error("Azure OpenAI chat error", exc_info=True)
        raise

def handle_chat(data):
    logger.debug(f"Received chat request: {data}")
    user_message = data.get("message", "")
    system_message = data.get("system_message", "")
    model = data.get("model", "")
    rag_enabled = data.get("rag_enabled", False)

    if not validate_input(user_message):
        logger.warning("Chat request rejected due to empty message.")
        return jsonify({"reply": "Error: Empty message"}), 400

    try:
        if rag_enabled:
            logger.info("RAG flow enabled. Delegating to RAG service.")
            from services.rag_service import handle_rag_flow
            reply = handle_rag_flow(user_message)
            logger.debug(f"RAG response: {reply}")
            return jsonify({"reply": reply['result']})

        elif model == "huggingface":
            logger.info("Using Hugging Face model for chat.")
            reply = handle_huggingface_chat(user_message, system_message)
            return jsonify({"reply": reply.choices[0].message["content"]})

        elif model == "azure-openai":
            logger.info("Using Azure OpenAI model for chat.")
            reply = handle_azure_openai_chat(user_message, system_message)
            return jsonify({"reply": reply.get('text')})

        else:
            logger.warning(f"Unknown model specified: {model}")
            return jsonify({"reply": "Error: Unknown model specified"}), 400

    except Exception as e:
        logger.error(f"Unhandled error in chat flow: {e}", exc_info=True)
        return jsonify({"reply": "Internal server error"}), 500