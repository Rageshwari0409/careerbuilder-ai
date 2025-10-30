# clients/llm_clients.py
from langchain.chat_models import AzureChatOpenAI
from huggingface_hub import InferenceClient
from config.config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, DEPLOYMENT_NAME, HUGGINGFACEHUB_API_TOKEN

def get_azure_openai_client():
    return AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_key=AZURE_OPENAI_KEY,
        azure_deployment=DEPLOYMENT_NAME,
        api_version="2024-05-01-preview",
        temperature=0
    )

def get_huggingface_client():
    return InferenceClient(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        token=HUGGINGFACEHUB_API_TOKEN
    )