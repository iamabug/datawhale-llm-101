import os

file_paths = []
folder_path = "../database/knowledge_db"

for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)
        file_paths.append(file_path)
print(file_paths)

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader

loaders = []
for file_path in file_paths:
    if file_path.endswith(".pdf"):
        loader = PyMuPDFLoader(file_path)
    elif file_path.endswith(".md"):
        loader = UnstructuredMarkdownLoader(file_path)
    loaders.append(loader)

texts = []
for loader in loaders:
    texts.extend(loader.load())

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(texts)

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(model_name="../m3e-base")
persist_dir = "../database/vector_db/chroma"
vertor_db = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory=persist_dir
)
vertor_db.persist()

print(vertor_db._collection.count())