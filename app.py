from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from langchain_mistralai import ChatMistralAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

##########################################################
# Page Configuration
##########################################################

st.set_page_config(
    page_title="FEM Assistant",
    page_icon="📘",
    layout="wide"
)

##########################################################
# Custom CSS
##########################################################

st.markdown(
    """
    <style>

    .main {
        background-color: #F5F7FA;
    }

    .title{
        font-size:42px;
        font-weight:bold;
        color:#0E4C92;
    }

    .subtitle{
        color:gray;
        font-size:18px;
    }

    .stChatMessage{
        border-radius:15px;
        padding:8px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

##########################################################
# Header
##########################################################

st.markdown(
    '<p class="title">📘 FEM AI Assistant</p>',
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="subtitle">Ask anything related to Finite Element Method (FEM)</p>',
    unsafe_allow_html=True,
)

##########################################################
# Load LLM
##########################################################

@st.cache_resource
def load_llm():

    llm = ChatMistralAI(
        model="mistral-small-2603"
    )

    embedding_model = OpenAIEmbeddings()

    vectorstore = Chroma(
        persist_directory="chroma_db_introfem",
        embedding_function=embedding_model,
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":4,
            "fetch_k":10,
            "lambda_mult":0.5,
        },
    )

    return llm, retriever


llm, retriever = load_llm()

##########################################################
# Prompt
##########################################################

template = ChatPromptTemplate.from_messages(
[
(
"system",
"""
You are an expert in Finite Element Method (FEM).

Answer only from the supplied context.

If the answer is unavailable, say:

'I could not find the answer in the document.'

Keep answers concise and educational.
"""
),

MessagesPlaceholder(variable_name="chat_history"),

(
"human",
"""
Context:

{context}

Question:

{question}
"""
)
])

##########################################################
# Session State
##########################################################

if "chat_history" not in st.session_state:
    st.session_state.chat_history=[]

##########################################################
# Sidebar
##########################################################

with st.sidebar:

    st.title("⚙ Settings")

    st.markdown("---")

    st.write("### Retrieval")

    k = st.slider(
        "Number of Retrieved Chunks",
        2,
        8,
        4
    )

    st.markdown("---")

    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history=[]
        st.rerun()

    st.markdown("---")

    st.info(
        """
        **Knowledge Base**

        • Intro to FEM

        • Chroma Vector DB

        • Mistral AI
        """
    )

##########################################################
# Display Previous Chat
##########################################################

for message in st.session_state.chat_history:

    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)

    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

##########################################################
# Chat Input
##########################################################

query = st.chat_input("Ask a question about FEM...")

if query:

    with st.chat_message("user"):
        st.markdown(query)

    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = template.invoke(
    {
        "chat_history":st.session_state.chat_history,
        "context":context,
        "question":query,
    })

    with st.spinner("Searching document..."):

        response = llm.invoke(prompt)

    with st.chat_message("assistant"):
        st.markdown(response.content)

        with st.expander("Retrieved Context"):

            for i, doc in enumerate(docs):

                st.markdown(f"### Chunk {i+1}")

                st.write(doc.page_content)

                st.markdown("---")

    st.session_state.chat_history.append(
        HumanMessage(content=query)
    )

    st.session_state.chat_history.append(
        AIMessage(content=response.content)
    )

    st.session_state.chat_history = st.session_state.chat_history[-10:]