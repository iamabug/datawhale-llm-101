from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="../m3e-base")

vector_db = Chroma(
    embedding_function=embeddings,
    persist_directory="../database/vector_db/chroma"
    )

sim_docs = vector_db.similarity_search("什么是大语言模型", k=3)
print(len(sim_docs))
for i, sim_doc in enumerate(sim_docs):
    print(f"Doc {i}: {sim_doc.metadata}")
    print(f"content: {sim_doc.page_content}")
    print()
