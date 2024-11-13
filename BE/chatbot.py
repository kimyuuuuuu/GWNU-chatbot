from typing import List

from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.documents.base import Document
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from prompts.inti_prompt import init_template


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small")


def get_db(embeddings: OpenAIEmbeddings) -> FAISS:
    return FAISS.load_local("./db/faiss_index", embeddings, allow_dangerous_deserialization=True)


def get_retriever(db: FAISS, num_docs: int) -> Runnable:
    return db.as_retriever(search_kwargs={"k": num_docs})


def retrieve_docs_sync(retriever: Runnable, user_question: str) -> List[Document]:
    return retriever.invoke(user_question)


def retrieve_docs_async(retriever: Runnable, user_question: str) -> List[Document]:
    return retriever.ainvoke(user_question)


def get_result_sync(chain: Runnable, question: str, context: List[Document]):
    return chain.invoke({"question": question, "context": context})


def get_result_async(chain: Runnable, question: str, context: List[Document]):
    return chain.ainvoke({"question": question, "context": context})


def get_rag_chain(template: str):
    custom_rag_prompt = PromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-4o-mini")

    return custom_rag_prompt | model | StrOutputParser()


## 사용자 질문에 대한 RAG 처리
def process_question(is_sync: bool, user_question: str):
    embeddings = get_embeddings()

    ## 벡터 DB 호출
    new_db = get_db(embeddings)

    ## 관련 문서 3개를 호출하는 Retriever 생성
    retriever = get_retriever(new_db, 3)

    ## 사용자 질문을 기반으로 관련문서 3개 검색
    if is_sync:
        retrieve_docs = retrieve_docs_sync(retriever, user_question)
    else:
        retrieve_docs = retrieve_docs_async(retriever, user_question)

    ## RAG 체인 선언
    chain = get_rag_chain(init_template)

    ## 질문과 문맥을 넣어서 체인 결과 호출
    if is_sync:
        response = get_result_sync(chain, user_question, retrieve_docs)
    else:
        response = get_result_async(chain, user_question, retrieve_docs)

    return response, retrieve_docs
