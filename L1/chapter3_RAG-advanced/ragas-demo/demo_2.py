import os
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms import Tongyi
from langchain_chroma import Chroma
# 新的导入方式（langchain >= 0.1.0）
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore

docs = PyPDFLoader("./浦发上海浦东发展银行西安分行个金客户经理考核办法.pdf").load()

# 初始化大语言模型
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
llm = Tongyi(
    model_name="qwen-max",
    dashscope_api_key=DASHSCOPE_API_KEY
)

# 创建嵌入模型
embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=DASHSCOPE_API_KEY
)

# 创建主文档分割器
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=512)

# 创建子文档分割器
child_splitter = RecursiveCharacterTextSplitter(chunk_size=256)

# 创建向量数据库对象
vectorstore = Chroma(
    collection_name="split_parents", embedding_function=embeddings
)
# 创建内存存储对象
store = InMemoryStore()
# 创建父文档检索器
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 2}
)

# 添加文档集
retriever.add_documents(docs)

# 切割出来主文档的数量
len(list(store.yield_keys()))

from langchain_classic.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_core.output_parsers import StrOutputParser

# 创建prompt模板（RAG Prompt）
template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use two sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""

# 由模板生成prompt
prompt = ChatPromptTemplate.from_template(template)

# 创建 chain（LCEL langchain 表达式语言）
chain = RunnableMap({
    "context": lambda x: retriever.invoke(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | llm | StrOutputParser()

query = "客户经理被投诉了，投诉一次扣多少分？"
response = chain.invoke({"question": query})
print(response)