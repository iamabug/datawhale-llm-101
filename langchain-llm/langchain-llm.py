import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

_ = load_dotenv(find_dotenv())

os.environ["http_proxy"] = "http://localhost:7897"
os.environ["https_proxy"] = "http://localhost:7897"

openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")
prompt = """
请你将由三个反引号分割的文本翻译成英文！\
text: ```{text}```
"""

text = "我带着比身体重的行李，\
游入尼罗河底，\
经过几道闪电 看到一堆光圈，\
不确定是不是这里。\
"
template = "你是一个翻译助手，帮助我将{input_language}翻译成{output_language}。"

human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        ("human", human_template),
    ]
)

messages = chat_prompt.format_messages(
    input_language="中文", output_language="英文", text=text
)

#output = llm.invoke(messages)
#print(output)

from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()
#print(output_parser.invoke(output))

chain = chat_prompt | llm | output_parser
output = chain.invoke(
    {
        "input_language": "中文",
        "output_language": "英文",
        "text": text
    }
)
print(output)