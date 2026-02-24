import os

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms import Tongyi
from langchain_chroma import Chroma
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# 新的导入方式（langchain >= 0.1.0）
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore
from langchain_classic.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_core.output_parsers import StrOutputParser
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)


# 自定义嵌入类
class BatchingDashScopeEmbeddings(DashScopeEmbeddings):
    """自定义嵌入类，确保每次请求不超过 DashScope API 限制（10 个文本）"""

    def embed_documents(self, texts):
        all_embeddings = []
        batch_size = 10  # DashScope API 限制

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                embeddings = super().embed_documents(batch)
                all_embeddings.extend(embeddings)
            except Exception as e:
                raise RuntimeError(f"Error embedding batch {i}-{i + len(batch)}: {str(e)}") from e

        return all_embeddings

# 解析 pdf 文档
docs = PyPDFLoader("./浦发上海浦东发展银行西安分行个金客户经理考核办法.pdf").load()
print(f"Loaded {len(docs)} document pages")

# 初始化大语言模型
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
llm = Tongyi(
    model_name="qwen-max",
    dashscope_api_key=DASHSCOPE_API_KEY
)

# message: <400> InternalError.Algo.InvalidParameter: Value error, batch size is invalid, it should not be larger than 10.: input.contents
# 创建嵌入模型
embeddings = BatchingDashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key=DASHSCOPE_API_KEY
)

# 创建主文档分割器
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=512)

# 创建子文档分割器
child_splitter = RecursiveCharacterTextSplitter(chunk_size=256)

# 创建向量数据库对象
vectorstore = Chroma(
    collection_name="split_parents",
    embedding_function=embeddings
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
print(f"Stored {len(list(store.yield_keys()))} parent documents")

# 创建prompt模板（RAG Prompt）
template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use two sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""

# 修复：确保 context 是字符串而非 Document 对象列表
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 由模板生成prompt
prompt = ChatPromptTemplate.from_template(template)

# 创建 chain（LCEL langchain 表达式语言）
chain = RunnableMap({
    "context": lambda x: format_docs(retriever.invoke(x["question"])),
    "question": lambda x: x["question"]
}) | prompt | llm | StrOutputParser()

query = "客户经理被投诉了，投诉一次扣多少分？"
response = chain.invoke({"question": query})
print("\nQuestion:", query)
print("Answer:", response)



# 保证问题需要多样性，场景化覆盖
questions = [
    "客户经理被投诉了，投诉一次扣多少分？",
    "客户经理每年评聘申报时间是怎样的？",
    "客户经理在工作中有不廉洁自律情况的，发现一次扣多少分？",
    "客户经理不服从支行工作安排，每次扣多少分？",
    "客户经理需要什么学历和工作经验才能入职？",
    "个金客户经理职位设置有哪些？"
]

ground_truths = [
    "每投诉一次扣2分",
    "每年一月份为客户经理评聘的申报时间",
    "在工作中有不廉洁自律情况的每发现一次扣50分",
    "不服从支行工作安排，每次扣2分",
    "须具备大专以上学历，至少二年以上银行工作经验",
    "个金客户经理职位设置为：客户经理助理、客户经理、高级客户经理、资深客户经理"
]

answers = []
contexts = []

# Inference
for query in questions:
    answers.append(chain.invoke({"question": query}))
    contexts.append([docs.page_content for docs in retriever.invoke(query)])

# To dict
data = {
    "user_input": questions,
    "response": answers,
    "retrieved_contexts": contexts,
    "reference": ground_truths
}

# Convert dict to dataset
dataset = Dataset.from_dict(data)
print(dataset)


# 评测结果
result = evaluate(
    dataset=dataset,
    llm=llm,
    metrics=[
        context_precision,  # 上下文精度
        context_recall,  # 上下文召回率
        faithfulness,  # 忠实度
        answer_relevancy,  # 答案相关性
    ],
    embeddings=embeddings
)

df = result.to_pandas()
print(df)