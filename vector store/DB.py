from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document

docs = [
    Document(
        page_content="Python is a high-level programming language widely used for AI, Machine Learning, Data Science, automation, and web development.",
        metadata={"source": "Python Handbook"}
    ),
    Document(
        page_content="Pandas is a Python library that provides DataFrame and Series objects for efficient data manipulation, cleaning, and analysis.",
        metadata={"source": "Data Analysis Guide"}
    ),
    Document(
        page_content="Retrieval-Augmented Generation (RAG) enhances Large Language Models by retrieving relevant documents from an external knowledge base before generating responses.",
        metadata={"source": "RAG Fundamentals"}
    ),
    Document(
        page_content="Vector databases store text embeddings and perform similarity search using metrics like cosine similarity to retrieve relevant information.",
        metadata={"source": "Vector Database Guide"}
    ),
    Document(
        page_content="LangChain is a framework for building LLM applications with components for document loading, text splitting, embeddings, vector stores, retrieval, and prompt management.",
        metadata={"source": "LangChain Documentation"}
    )
]

embedding_model = OpenAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents = docs, 
    embedding = embedding_model,
    persist_directory = "chroma_db"
    )

# this method is of the vector store and not a retriever runnable
result = vectorstore.similarity_search("What is LangChain?", k=2)

for r in result:
    print(r)

