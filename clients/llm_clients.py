# clients/llm_clients.py
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEndpointEmbeddings, ChatHuggingFace, HuggingFaceEndpoint  # LangChain integrations for Hugging Face models
from config.config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, DEPLOYMENT_NAME, HUGGINGFACEHUB_API_TOKEN, PINECONE_API_KEY
import pinecone


retry_decorator = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((TimeoutError, ConnectionError, ValueError))
)


@retry_decorator(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((TimeoutError, ConnectionError, ValueError))
)
def get_azure_openai_client():
    return AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_key=AZURE_OPENAI_KEY,
        azure_deployment=DEPLOYMENT_NAME,
        api_version="2024-05-01-preview",
        temperature=0
    )

@retry_decorator(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((TimeoutError, ConnectionError, ValueError))
)
def get_huggingface_client():
    return InferenceClient(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        token=HUGGINGFACEHUB_API_TOKEN
    )

@retry_decorator(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((TimeoutError, ConnectionError, ValueError))
)
def initialize_azure_openai_embeddings():
    return AzureOpenAIEmbeddings(
        deployment="text-embedding-3-small",
        model="text-embedding-3-small",
        openai_api_type="azure",
        openai_api_key=AZURE_OPENAI_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_version="2023-05-15",
        chunk_size=1000
    )

@retry_decorator(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((TimeoutError, ConnectionError, ValueError))
)
def initialize_pinecone_client():
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    return pc

@retry_decorator(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((TimeoutError, ConnectionError, ValueError))
)
def initialize_huggingface_embeddings():
    # Placeholder for Hugging Face embeddings initialization
    embeddings = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-mpnet-base-v2", huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN)
    return embeddings

@retry_decorator(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((TimeoutError, ConnectionError, ValueError))
)
def initialize_huggingface_endpoint():
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id, task="text-generation", temperature=0.5,
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN, max_new_tokens=512
    )
    model = ChatHuggingFace(llm=llm)
    return model