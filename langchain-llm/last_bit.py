import os
from dotenv import load_dotenv, find_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever

_ = load_dotenv(find_dotenv())

os.environ["http_proxy"] = "http://localhost:7897"
os.environ["https_proxy"] = "http://localhost:7897"

persist_dir = "../database/vector_db/chroma"

embeddings = HuggingFaceEmbeddings(model_name="../m3e-base")

vector_db = Chroma(
    embedding_function=embeddings,
    persist_directory=persist_dir
    )

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
#answer = llm.invoke("聊天机器人是什么")
#print(answer.content)

template = """
使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
{context}
问题: {input}
"""
qa_chain_prompt = PromptTemplate(template=template,
                              input_variables=["context", "input"])

basic_chain = create_stuff_documents_chain(llm=llm, prompt=qa_chain_prompt)
#qa_chain = create_retrieval_chain(vector_db.as_retriever(),
#                                  basic_chain,)
#res = qa_chain.invoke({"input": "聊天机器人是什么？"})
#print("RGA:")
#print(res)
# print(res["answer"])

history_prompt = ChatPromptTemplate.from_messages(
    [
        ("system","根据对话历史优化以下问题：\n\n{chat_history}\n\n问题：{input}"),
        ("user", "{input}"),
        ]
)
history_aware_retriever = create_history_aware_retriever(
    llm=llm,
    retriever=vector_db.as_retriever(),
    prompt=history_prompt
)

history_chain = create_retrieval_chain(
    history_aware_retriever,
    basic_chain
)
res = history_chain.invoke({"input": "我可以学习到有关提示工程的知识吗？"})
print(res["answer"])
res = history_chain.invoke({"input": "为什么这么课需要教这方面的知识？"})
print(res["answer"])