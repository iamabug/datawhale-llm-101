from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("../database/pumpkin/pumpkin_book.pdf")

pdf_pages = loader.load()

print(type(pdf_pages[0]))
print(pdf_pages[0].metadata)
print(pdf_pages[1].page_content)


from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader

loader = UnstructuredMarkdownLoader("../database/knowledge_db/prompt_engineering/1. 简介 Introduction.md")
md_pages = loader.load()

md_page = md_pages[0]
print(md_page.metadata)
print(md_page.page_content[0:][:500])

from langchain.text_splitter import RecursiveCharacterTextSplitter

CHUNK_SIZE = 500
OVERLAP_SIZE = 50
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, 
    chunk_overlap=OVERLAP_SIZE
    )

res = text_splitter.split_text(pdf_pages[1].page_content[:1000])
docs = text_splitter.split_documents(pdf_pages)
print(res)
print(len(docs))