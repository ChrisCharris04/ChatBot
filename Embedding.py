import os
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import AzureChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")


def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def chunk_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vector_db(chunks):
    embeddings = OpenAIEmbeddings(
        openai_api_key=AZURE_OPENAI_API_KEY,
        openai_api_base=AZURE_OPENAI_ENDPOINT,
        model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT
    )

    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db

def setup_chatbot(vector_db):
    llm = AzureChatOpenAI(
        deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
        openai_api_key=AZURE_OPENAI_API_KEY,
        openai_api_base=AZURE_OPENAI_ENDPOINT,
        openai_api_version="2023-06-01-preview"
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chatbot = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vector_db.as_retriever(), memory=memory
    )

    return chatbot

def chat_with_bot(chatbot):
    print("Chatbot is ready! Type 'exit' to stop.")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break
        response = chatbot.run(query)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    pdf_path = "//Users//Chris//Downloads//diffie.pdf"
    documents = load_pdf(pdf_path)
    chunks = chunk_text(documents)
    vector_db = create_vector_db(chunks)
    chatbot = setup_chatbot(vector_db)
    chat_with_bot(chatbot)



