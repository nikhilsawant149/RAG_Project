from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import TokenTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

data = TextLoader("document loaders/notes.txt")
docs = data.load()

# char_splitter = CharacterTextSplitter(
#     separator="",
#     chunk_size=100, 
#     chunk_overlap=10,
#     )
# chunks = char_splitter.split_documents(docs)

# token_splitter = TokenTextSplitter(
#     chunk_size=100,
#     chunk_overlap=10,
#     )
# chunks = token_splitter.split_documents(docs)

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10,
    separators=["\n\n", "\n", " ", ""]
    )
chunks = recursive_splitter.split_documents(docs)

print(len(chunks))
print(chunks[0].page_content)