from langchain_community.retrievers import ArxivRetriever

# Craete the retriever
retriever = ArxivRetriever(
    load_max_docs = 2,
    load_all_available_meta = True,
)

# query arxiv
docs = retriever.invoke("large language models")

# print results
print (docs)