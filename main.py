from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline

# Load and split the document
loader = TextLoader("example.txt")  # Replace with your file
documents = loader.load()

# Chunking for better retrieval
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# Use a Hugging Face model for embeddings
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"  # Optimal balance of speed and accuracy
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

# Store in Chroma DB
vector_store = Chroma.from_documents(chunks, embeddings)

# Load Hugging Face model for LLM
llm_pipeline = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1")
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# Create a RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vector_store.as_retriever())

# Query the chatbot
query = "What does the document say about X?"
response = qa_chain.run(query)
print(response)

