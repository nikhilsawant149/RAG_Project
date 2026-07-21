from langchain_community.document_loaders import WebBaseLoader

url = "https://www.apple.com/iphone-17/"

data = WebBaseLoader(url)

docs = data.load()

print(len(docs))

print(docs[0].page_content)
