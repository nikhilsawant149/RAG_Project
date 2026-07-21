from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("document loaders/introfem.pdf")

docs = data.load()

print(len(docs))

print(docs[1].page_content)

