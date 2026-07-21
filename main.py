from dotenv import load_dotenv
import os
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

llm = ChatMistralAI(model = "mistral-small-2603")

embedding_model = OpenAIEmbeddings()

# to get the vectors from vector database
vectorstore = Chroma(
    persist_directory="chroma_db_introfem",
    embedding_function=embedding_model,
)

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k" : 4,
        "fetch_k" : 10,
        "lambda_mutl" : 0.5,
    }
)

# Prompt Template
template = ChatPromptTemplate.from_messages(
    [
    ("system", 
     """
     You are a helpful AI assistant. 
     Use only the provided context to answer the question.
     If the answer is not poresent in the context, 
     say: "I could not find the answer in the document."
     """
     ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", 
     """
     Context: 
     {context}

     Question:
     {question}
     """
     )
    ]
)

chat_history = []

print("================== RAG Syatem Created ======================")

print("Type 'exit' to end the chat.")

while True:
    query = input("You : ")
    if query.lower() == 'exit':
        print("Exiting the chatbot. Goodbye!")
        break
    docs = retriever.invoke(query)
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    prompt = template.invoke(
        {
            "chat_history" : chat_history,
            "context" : context,
            "question" : query,
        }
    )
    response = llm.invoke(prompt)

    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response.content))

    # Keep only the last 10 messages (5 exchanges)
    chat_history = chat_history[-10:]

    print(f"AI : {response.content}")



