# Data injection script to create a vector database using Chroma and OpenAI embeddings.

# Step 1 Load pdf file
# Step 2 Create documents from the pdf file
# Step 3 Create chunks from the documents
# Step 4 Create embeddings for the chunks
# Step 5 Create a vector database using Chroma and store the embeddings

from dotenv import load_dotenv
import os
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

data = PyPDFLoader("document loaders/introfem.pdf")
docs = data.load()

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
    )
chunks = recursive_splitter.split_documents(docs)

embedding_model = OpenAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents = chunks, 
    embedding = embedding_model,
    persist_directory = "chroma_db_introfem"
    )
