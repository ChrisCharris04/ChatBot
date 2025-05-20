from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

#  Loading PDF
def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents

#  Splitting text into chunks
def chunk_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks


#  ChromaDB vector store
def create_vector_db(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(chunks, embedding_model)
    return vector_db

#  Hugging Face LLM
def setup_chatbot(vector_db):
    llm = HuggingFaceHub(
        repo_id = "meta-llama/Llama-3.1-8B",
        #repo_id = "tiiuae/falcon-7b-instruct",
        model_kwargs={"temperature": 0.7, "max_length": 512},
        huggingfacehub_api_token="Your Api Key"
    )

    prompt_template = PromptTemplate(
        input_variables=["user_input"],
        template="Given the input: '{user_input}', generate three relevant follow-up questions."
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chatbot = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vector_db.as_retriever(), memory =memory, prompt =prompt_template
    )
    return chatbot

#  ChatBot and Querying
def chat_with_bot(chatbot):
    print("Chatbot is ready! Type 'exit' to stop.")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            print("Thank You!Bye")
            break
        #response = chatbot.run(query)
        #print(f"Chatbot: {response}")

        response = chatbot.run(user_input)
        print(f"Chatbot:\n{response}")

# Main function
if __name__ == "__main__":
    pdf_path = "/content/Sample2.pdf"
    documents = load_pdf(pdf_path)
    chunks = chunk_text(documents)
    vector_db = create_vector_db(chunks)
    chatbot = setup_chatbot(vector_db)
    chat_with_bot(chatbot)

